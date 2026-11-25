# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone HTML report renderer for analytical roofline solve/v2 payloads."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

from .chips import CHIPS, ChipSpec, chip_catalog, count_hosts, format_constant_display
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_TOP_K,
    FRONTIER_2D_TIE_TOLERANCE,
    MAX_EMBEDDED_CANDIDATES,
    MAX_SVG_BG_POINTS,
    MIN_FRONTIER_POINTS_FOR_POLYLINE,
    PERCENT_SCALE,
    PREFIX_RANGE_PAIR_LEN,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RULED_OUT,
    SVG_GRID_CELL_PX,
    UNIT_GIGA,
    UNIT_MEGA,
    UNIT_PETA,
    UNIT_TERA,
)
from .schema import (
    format_host_count,
    format_submodule_parallelism,
    pool_slice_summary,
)

_SINGLE_POOL_LAYOUT_COLUMNS = 12
_DISAGG_LAYOUT_COLUMNS = 13
_ROUND_DECIMALS_2D = 4
_AXIS_GRID_INTERVALS = 4
_NICE_MANTISSA_TWO = 2.0
_NICE_MANTISSA_FOUR = 4.0
_NICE_MANTISSA_FIVE = 5.0
_NICE_MANTISSA_EIGHT = 8.0
_NICE_MANTISSA_TEN = 10.0
# Reserves 8% upper plot headroom (~58 px of 780 px plot width) so r=10 px rings
# and callout labels stay inside the SVG border.
_AXIS_HEADROOM_SCALE = 1.08
# Reserves 22% right-hand bar headroom (~115 px of 640 px bar track) so 'X.XXX ms'
# text labels after the longest resource bar never clip at the right SVG edge.
_BAR_LABEL_HEADROOM_SCALE = 1.22
# Snap tolerance for checking whether a 4-interval axis tick is an exact integer.
_TICK_INT_TOLERANCE = 1e-6
_RIGHT_EDGE_ANCHOR_FRAC = 0.72


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as a mapping when compatible with JSON objects, else empty."""
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> list[object]:
    """Return value as a list for non-string sequences, else empty list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce numeric JSON scalar to float safely."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    """Coerce numeric JSON scalar to int safely."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _interactivity_val(point: Mapping[str, object]) -> float:
    """Read estimated interactivity (tok/s/user) from a point payload."""
    return _as_float(point.get("est_interactivity_tok_s"))


def _ttft_val(point: Mapping[str, object]) -> float:
    """Read estimated TTFT (ms) from a point payload."""
    return _as_float(point.get("est_ttft_ms"))


_MS_PER_SECOND = 1000.0


def _tpot_val(point: Mapping[str, object]) -> float:
    """Read estimated TPOT (ms) from a point payload."""
    return _as_float(point.get("est_tpot_ms"))


def _point_tpot_bound_ms(
    point: Mapping[str, object],
    *,
    accept_len: float = 1.0,
) -> float:
    """Read physical TPOT floor (ms) from bound_tpot_ms or decode_floor."""
    explicit = _as_float(point.get("bound_tpot_ms"))
    if explicit > 0.0:
        return explicit
    floor_map = _as_mapping(point.get("decode_floor"))
    return _as_float(floor_map.get("hard_floor_ms")) / max(1.0, accept_len)


def _render_scatter_points(
    all_pts: Sequence[Mapping[str, object]],
    bounds: tuple[float, float],
    plot_box: tuple[int, int, int, int],
    *,
    accept_len: float = 1.0,
) -> list[str]:
    """Render candidate scatter points with physical TPOT floor whiskers."""
    max_x, max_y = bounds
    pad_left, pad_top, plot_w, plot_h = plot_box
    parts: list[str] = []
    for point in _downsample_svg_points(all_pts, bounds, (plot_w, plot_h)):
        inter = _interactivity_val(point)
        tput = _as_float(point.get("token_throughput_per_chip"))
        floor_tpot = _point_tpot_bound_ms(point, accept_len=accept_len)
        cx_pt = pad_left + (inter / max_x) * plot_w
        cy_pt = pad_top + plot_h - (tput / max_y) * plot_h
        if floor_tpot > 0.0:
            floor_inter = min(max_x, _MS_PER_SECOND / floor_tpot)
            if floor_inter > inter:
                fx_pt = pad_left + (floor_inter / max_x) * plot_w
                parts.append(
                    f'<line x1="{cx_pt:.1f}" y1="{cy_pt:.1f}" '
                    f'x2="{fx_pt:.1f}" y2="{cy_pt:.1f}" '
                    'stroke="#94a3b8" stroke-width="0.8" '
                    'stroke-dasharray="2,2" opacity="0.55"/>'
                )
        tip = (
            f"c={_as_int(point.get('concurrency'), 1)} | "
            f"est {tput:,.1f} tok/s/chip | est {inter:,.1f} tok/s | "
            f"TPOT floor {floor_tpot:.2f}ms / est {_tpot_val(point):.2f}ms"
        )
        parts.append(
            f'<circle cx="{cx_pt:.1f}" cy="{cy_pt:.1f}" r="3.5" '
            'fill="#cbd5e1" stroke="#94a3b8" stroke-width="0.6" opacity="0.72">'
            f"<title>{_escape(tip)}</title></circle>"
        )
    return parts


def _escape(val: object) -> str:
    """Escape untrusted text for safe inline HTML embedding."""
    if val is None:
        return ""
    return html.escape(str(val))


def _nice_axis_max(raw_max: float) -> float:
    """Round upper axis bound up to a clean {2, 4, 5, 8, 10} * 10^k ceiling."""
    if not math.isfinite(raw_max) or raw_max <= 0.0:
        return 1.0
    target = max(1.0, raw_max * _AXIS_HEADROOM_SCALE)
    exponent = math.floor(math.log10(target))
    magnitude = 10.0**exponent
    mantissa = target / magnitude
    if mantissa <= _NICE_MANTISSA_TWO:
        nice_mantissa = _NICE_MANTISSA_TWO
    elif mantissa <= _NICE_MANTISSA_FOUR:
        nice_mantissa = _NICE_MANTISSA_FOUR
    elif mantissa <= _NICE_MANTISSA_FIVE:
        nice_mantissa = _NICE_MANTISSA_FIVE
    elif mantissa <= _NICE_MANTISSA_EIGHT:
        nice_mantissa = _NICE_MANTISSA_EIGHT
    else:
        nice_mantissa = _NICE_MANTISSA_TEN
    return nice_mantissa * magnitude


def _format_axis_tick(tick_val: float) -> str:
    """Format a 4-interval axis tick without rounding uneven fractional steps."""
    nearest_int = round(tick_val)
    if abs(tick_val - nearest_int) <= _TICK_INT_TOLERANCE:
        return f"{nearest_int:,}"
    return f"{tick_val:,.2f}".rstrip("0").rstrip(".")


def _downsample_svg_points(
    points: Sequence[Mapping[str, object]],
    bounds: tuple[float, float],
    plot_size: tuple[int, int],
) -> list[Mapping[str, object]]:
    """Deduplicate dense background scatter points on a 2D pixel grid."""
    max_x, max_y = bounds
    plot_w, plot_h = plot_size
    if len(points) <= MAX_SVG_BG_POINTS:
        return list(points)
    seen: set[tuple[int, int]] = set()
    deduped: list[Mapping[str, object]] = []
    for point in points:
        plot_x = (_interactivity_val(point) / max_x) * plot_w
        plot_y = (_as_float(point.get("token_throughput_per_chip")) / max_y) * plot_h
        cell = (int(plot_x // SVG_GRID_CELL_PX), int(plot_y // SVG_GRID_CELL_PX))
        if cell not in seen:
            seen.add(cell)
            deduped.append(point)
    if len(deduped) <= MAX_SVG_BG_POINTS:
        return deduped
    stride = math.ceil(len(deduped) / MAX_SVG_BG_POINTS)
    return deduped[::stride][:MAX_SVG_BG_POINTS]


def _filter_slo_preferred(
    points: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Prefer estimate_meets points when present, else non-ruled_out points."""
    not_ruled_out = [
        point for point in points if point.get("slo_verdict") != SLO_VERDICT_RULED_OUT
    ]
    estimate_meets = [
        point
        for point in not_ruled_out
        if point.get("slo_verdict") == SLO_VERDICT_ESTIMATE_MEETS
    ]
    if estimate_meets:
        return estimate_meets
    return not_ruled_out


def _frontier_2d_envelope(
    points: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Return non-dominated 2D upper envelope in (interactivity, throughput)."""
    ordered = sorted(
        points,
        key=lambda point_item: (
            -_interactivity_val(point_item),
            -_as_float(point_item.get("token_throughput_per_chip")),
        ),
    )
    envelope: list[Mapping[str, object]] = []
    best_tput = -1.0
    for point in ordered:
        tput = _as_float(point.get("token_throughput_per_chip"))
        if tput > best_tput + FRONTIER_2D_TIE_TOLERANCE:
            envelope.append(point)
            best_tput = tput
    envelope.sort(key=lambda point_item: _interactivity_val(point_item))
    return envelope


def _point_2d_coords(point: Mapping[str, object]) -> tuple[float, float]:
    """Return rounded (interactivity, throughput) pair for 2D frontier matching."""
    return (
        round(_interactivity_val(point), _ROUND_DECIMALS_2D),
        round(_as_float(point.get("token_throughput_per_chip")), _ROUND_DECIMALS_2D),
    )


def _select_ring_entries(
    top_k: Sequence[Mapping[str, object]],
    reference: Mapping[str, object] | None = None,
) -> list[tuple[str, Mapping[str, object]]]:
    """Select up to DEFAULT_TOP_K ring callouts plus any outside reference point."""
    capped: list[tuple[str, Mapping[str, object]]] = [
        (f"#{idx}", entry)
        for idx, entry in enumerate(top_k[:DEFAULT_TOP_K], 1)
        if _as_mapping(entry.get("point"))
    ]
    seen_coords = {
        _point_2d_coords(_as_mapping(entry.get("point"))) for _, entry in capped
    }
    if reference is not None and bool(reference.get("found")):
        ref_point = _as_mapping(reference.get("point"))
        if ref_point and _point_2d_coords(ref_point) not in seen_coords:
            capped.append(("ref", {"frontier_role": "reference", "point": ref_point}))
    return capped


def _render_frontier_rings(
    ring_entries: Sequence[tuple[str, Mapping[str, object]]],
    envelope_coords: set[tuple[float, float]],
    bounds: tuple[float, float],
    plot_box: tuple[int, int, int, int],
) -> list[str]:
    """Render shortlist callout rings and rank/concurrency labels on frontier SVG."""
    max_x, max_y = bounds
    pad_left, pad_top, plot_w, plot_h = plot_box
    anchor_cutoff = pad_left + _RIGHT_EDGE_ANCHOR_FRAC * plot_w
    ring_parts: list[str] = []
    for prefix, entry in ring_entries:
        point = _as_mapping(entry.get("point"))
        if not point:
            continue
        inter = _interactivity_val(point)
        tput = _as_float(point.get("token_throughput_per_chip"))
        conc_val = _as_int(point.get("concurrency"), 1)
        role = str(entry.get("frontier_role", "candidate"))
        on_2d = _point_2d_coords(point) in envelope_coords or role in (
            "max_throughput",
            "balanced_knee",
            "max_interactivity",
        )
        ring_style = (
            'stroke="#16a34a" stroke-width="2.6"'
            if on_2d
            else 'stroke="#d97706" stroke-width="2.2" stroke-dasharray="4,2.5"'
        )
        label_color = "#15803d" if on_2d else "#b45309"
        if role == "reference":
            tag_text = f"ref (c={conc_val})"
        elif on_2d:
            tag_text = f"{prefix} (c={conc_val})"
        else:
            tag_text = f"{prefix} (alt, c={conc_val})"
        cx = pad_left + (inter / max_x) * plot_w
        cy = pad_top + plot_h - (tput / max_y) * plot_h
        label_y_offset = -8.0 if on_2d else 18.0
        if cx > anchor_cutoff:
            label_x = cx - 12.0
            anchor = "end"
        else:
            label_x = cx + 12.0
            anchor = "start"
        ring_parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" '
            f'fill="none" {ring_style}>'
            f"<title>{_escape(prefix)} ({_escape(role)}): {tput:,.1f} tok/s/chip | "
            f"{inter:,.1f} tok/s</title></circle>"
            f'<text x="{label_x:.1f}" y="{cy + label_y_offset:.1f}" '
            f'text-anchor="{anchor}" font-size="12.5" '
            f'font-weight="700" fill="{label_color}">{_escape(tag_text)}</text>'
        )
    return ring_parts


def _render_frontier_svg(
    points: Sequence[Mapping[str, object]],
    frontier: Sequence[Mapping[str, object]],
    top_k: Sequence[Mapping[str, object]] = (),
    *,
    accept_len: float = 1.0,
    reference: Mapping[str, object] | None = None,
) -> str:
    """Render inline SVG estimated 2D Pareto frontier scatter plot."""
    width, height = 920, 400
    pad_left, pad_right, pad_top, pad_bottom = 92, 48, 30, 56
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    anchor_cutoff = pad_left + _RIGHT_EDGE_ANCHOR_FRAC * plot_w

    all_pts = points if points else frontier
    if not all_pts:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b" font-size="14">'
            "No candidate points available</text></svg>"
        )

    ring_entries = _select_ring_entries(top_k, reference=reference)
    bound_pts: list[Mapping[str, object]] = [*all_pts, *frontier]
    for entry in top_k:
        entry_pt = _as_mapping(entry.get("point"))
        if entry_pt:
            bound_pts.append(entry_pt)
    for _, entry in ring_entries:
        ring_pt = _as_mapping(entry.get("point"))
        if ring_pt:
            bound_pts.append(ring_pt)
    raw_max_x = max(
        (_interactivity_val(point_item) for point_item in bound_pts),
        default=1.0,
    )
    raw_max_y = max(
        (
            _as_float(point_item.get("token_throughput_per_chip"))
            for point_item in bound_pts
        ),
        default=1.0,
    )
    max_x = _nice_axis_max(raw_max_x)
    max_y = _nice_axis_max(raw_max_y)

    def scale_x(x_val: float) -> float:
        """Scale X data coordinate (interactivity) to SVG coordinate."""
        return pad_left + (x_val / max_x) * plot_w

    def scale_y(y_val: float) -> float:
        """Scale Y data coordinate (throughput per chip) to SVG coordinate."""
        return pad_top + plot_h - (y_val / max_y) * plot_h

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" '
        f'x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" '
        'stroke="#64748b" stroke-width="1.5"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{pad_top + plot_h}" stroke="#64748b" stroke-width="1.5"/>',
        f'<text x="{pad_left + plot_w // 2}" y="{height - 12}" '
        'text-anchor="middle" font-size="13.5" font-weight="600" fill="#334155">'
        "Est. Interactivity (tok/s)</text>",
        f'<text x="20" y="{pad_top + plot_h // 2}" text-anchor="middle" '
        'font-size="13.5" font-weight="600" fill="#334155" '
        f'transform="rotate(-90, 20, {pad_top + plot_h // 2})">'
        "Est. Token Throughput per Chip (tok/s/chip)</text>",
    ]

    for i in range(_AXIS_GRID_INTERVALS + 1):
        tick_x_val = (max_x / _AXIS_GRID_INTERVALS) * i
        tick_y_val = (max_y / _AXIS_GRID_INTERVALS) * i
        tick_x_px, tick_y_px = scale_x(tick_x_val), scale_y(tick_y_val)
        svg_parts.extend(
            [
                f'<line x1="{tick_x_px:.1f}" y1="{pad_top}" '
                f'x2="{tick_x_px:.1f}" y2="{pad_top + plot_h}" '
                'stroke="#e2e8f0" stroke-dasharray="4,4"/>',
                f'<text x="{tick_x_px:.1f}" y="{pad_top + plot_h + 20}" '
                f'text-anchor="middle" font-size="12" fill="#475569">'
                f"{_format_axis_tick(tick_x_val)}</text>",
                f'<line x1="{pad_left}" y1="{tick_y_px:.1f}" '
                f'x2="{pad_left + plot_w}" y2="{tick_y_px:.1f}" '
                'stroke="#e2e8f0" stroke-dasharray="4,4"/>',
                f'<text x="{pad_left - 10}" y="{tick_y_px + 4:.1f}" '
                f'text-anchor="end" font-size="12" fill="#475569">'
                f"{_format_axis_tick(tick_y_val)}</text>",
            ]
        )

    svg_parts.extend(
        _render_scatter_points(
            all_pts,
            (max_x, max_y),
            (pad_left, pad_top, plot_w, plot_h),
            accept_len=accept_len,
        )
    )

    envelope_2d = _frontier_2d_envelope(_filter_slo_preferred(frontier))
    envelope_coords = {_point_2d_coords(point_item) for point_item in envelope_2d}
    ring_coords = {
        _point_2d_coords(_as_mapping(entry.get("point")))
        for _, entry in ring_entries
        if _as_mapping(entry.get("point"))
    }
    if len(envelope_2d) >= MIN_FRONTIER_POINTS_FOR_POLYLINE:
        poly_pts = " ".join(
            f"{scale_x(_interactivity_val(point_item)):.1f},"
            f"{scale_y(_as_float(point_item.get('token_throughput_per_chip'))):.1f}"
            for point_item in envelope_2d
        )
        svg_parts.append(
            f'<polyline points="{poly_pts}" fill="none" '
            'stroke="#2563eb" stroke-width="2.8"/>'
        )

    for point in envelope_2d:
        inter = _interactivity_val(point)
        tput = _as_float(point.get("token_throughput_per_chip"))
        conc_val = _as_int(point.get("concurrency"), 1)
        cx_env = scale_x(inter)
        cy_env = scale_y(tput)
        frontier_tip = (
            f"Est. 2D Frontier: concurrency={conc_val} | "
            f"{tput:,.1f} tok/s/chip | {inter:,.1f} tok/s | "
            f"TTFT: {_ttft_val(point):,.1f}ms | "
            f"TPOT: {_tpot_val(point):.2f}ms"
        )
        svg_parts.append(
            f'<circle cx="{cx_env:.1f}" cy="{cy_env:.1f}" r="6" '
            'fill="#1d4ed8" stroke="#ffffff" stroke-width="1.6">'
            f"<title>{_escape(frontier_tip)}</title></circle>"
        )
        if _point_2d_coords(point) not in ring_coords:
            if cx_env > anchor_cutoff:
                env_x = cx_env - 9.0
                env_anchor = "end"
            else:
                env_x = cx_env + 9.0
                env_anchor = "start"
            svg_parts.append(
                f'<text x="{env_x:.1f}" y="{cy_env - 8:.1f}" '
                f'text-anchor="{env_anchor}" font-size="11.5" '
                'font-weight="500" fill="#1e40af">'
                f"c={conc_val}</text>"
            )

    svg_parts.extend(
        _render_frontier_rings(
            ring_entries,
            envelope_coords,
            (max_x, max_y),
            (pad_left, pad_top, plot_w, plot_h),
        )
    )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_limits_svg(floor: Mapping[str, object] | None) -> str:
    """Render horizontal bar chart for decode-step resource times."""
    width, height = 920, 305
    pad_left, pad_right, pad_top = 165, 115, 24

    if not floor:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b" font-size="14">'
            "No decode floor data available</text></svg>"
        )

    hbm_ms = _as_float(floor.get("hbm_bound_ms", floor.get("hbm_ms")))
    matmul_ms = _as_float(floor.get("matmul_bound_ms", floor.get("matmul_ms")))
    vector_ms = _as_float(floor.get("vector_bound_ms"))
    ici_ms = _as_float(floor.get("coll_bound_ms", floor.get("ici_ms")))
    chain_ms = _as_float(floor.get("chain_ms"))
    hop_ms = _as_float(floor.get("hop_latency_ms"))
    hard_floor_ms = _as_float(floor.get("hard_floor_ms"))

    items = [
        ("HBM Bound", hbm_ms, "#3b82f6"),
        ("MatMul Bound", matmul_ms, "#10b981"),
        ("Vector Bound", vector_ms, "#06b6d4"),
        ("Collective Bound", ici_ms, "#f59e0b"),
        ("Op Chain", chain_ms, "#8b5cf6"),
        ("Hop Latency", hop_ms, "#ec4899"),
        ("Hard Floor", hard_floor_ms, "#ef4444"),
    ]
    finite_vals = [val for _, val, _ in items if math.isfinite(val) and val > 0.0]
    max_val = (max(finite_vals, default=1.0) or 1.0) * _BAR_LABEL_HEADROOM_SCALE
    plot_w = width - pad_left - pad_right
    bar_h, bar_gap = 24, 13

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]
    for i, (label, val, color) in enumerate(items):
        y = pad_top + i * (bar_h + bar_gap)
        if not math.isfinite(val) or val <= 0.0:
            bar_w = 0.0
            width_attr = "0"
        else:
            bar_w = 0.0 if val < 5e-4 else max(2.5, (val / max_val) * plot_w)
            width_attr = f"{bar_w:.1f}"
        weight = "700" if label == "Hard Floor" else "500"
        svg_parts.extend(
            [
                f'<text x="{pad_left - 14}" y="{y + 16}" text-anchor="end" '
                f'font-size="13" font-weight="{weight}" fill="#1e293b">'
                f"{_escape(label)}</text>",
                f'<rect x="{pad_left}" y="{y}" width="{width_attr}" '
                f'height="{bar_h}" fill="{color}" rx="4">'
                f"<title>{_escape(label)}: {val:.3f} ms</title></rect>",
                f'<text x="{pad_left + bar_w + 10:.1f}" y="{y + 16}" font-size="12.5" '
                f'font-weight="{weight}" fill="#334155">{val:.3f} ms</text>',
            ]
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_memory_bar(
    pool: Mapping[str, object],
    y: int,
    plot_cfg: tuple[int, int, int],
    default_limit_gb: float,
) -> list[str]:
    """Render one stacked horizontal HBM bar row for a pool."""
    pad_left, plot_w, bar_h = plot_cfg
    role = str(pool.get("role") or "pool")
    topology = str(pool.get("topology") or "")
    limit_gb = _as_float(pool.get("hbm_limit_gb"), default_limit_gb)
    if limit_gb <= 0.0:
        limit_gb = max(0.0, default_limit_gb)
    weights_gb = max(0.0, _as_float(pool.get("hbm_weights_gb")))
    kv_cache_gb = max(0.0, _as_float(pool.get("hbm_kv_gb")))
    state_gb = max(0.0, _as_float(pool.get("hbm_state_gb")))
    used_gb = max(
        0.0,
        _as_float(pool.get("hbm_used_gb"), weights_gb + kv_cache_gb + state_gb),
    )
    free_gb = max(0.0, limit_gb - used_gb)

    bar_title = _escape(f"{role.capitalize()} ({topology})")
    parts = [
        f'<text x="{pad_left - 14}" y="{y + 18}" text-anchor="end" '
        f'font-size="13" font-weight="500" fill="#1e293b">{bar_title}</text>'
    ]
    segments = [
        (weights_gb, "#3b82f6", f"Weights: {weights_gb:.2f} GB"),
        (kv_cache_gb, "#10b981", f"KV Cache: {kv_cache_gb:.2f} GB"),
        (state_gb, "#f59e0b", f"Recurrent State: {state_gb:.2f} GB"),
        (free_gb, "#e2e8f0", f"Headroom: {free_gb:.2f} GB"),
    ]
    cur_x = float(pad_left)
    max_x_px = float(pad_left + plot_w)
    if limit_gb > 0.0:
        for gb_val, fill, title in segments:
            remaining_w = max(0.0, max_x_px - cur_x)
            if remaining_w <= 0.0 or gb_val <= 0.0:
                continue
            seg_w = (
                remaining_w
                if fill == "#e2e8f0"
                else min(remaining_w, (gb_val / limit_gb) * plot_w)
            )
            if seg_w <= 0.0:
                continue
            parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{seg_w:.1f}" '
                f'height="{bar_h}" fill="{fill}">'
                f"<title>{_escape(title)}</title></rect>"
            )
            cur_x += seg_w

    util_pct = (used_gb / limit_gb) * PERCENT_SCALE if limit_gb > 0.0 else 0.0
    parts.extend(
        [
            f'<rect x="{pad_left}" y="{y}" width="{plot_w}" height="{bar_h}" '
            'fill="none" stroke="#94a3b8" rx="4"/>',
            f'<text x="{pad_left + plot_w + 12}" y="{y + 18}" font-size="12.5" '
            f'font-weight="600" fill="#334155">{used_gb:.1f} / {limit_gb:.1f} GB '
            f"({util_pct:.0f}%)</text>",
        ]
    )
    return parts


def _render_memory_svg(
    pools: Sequence[Mapping[str, object]],
    default_limit_gb: float,
) -> str:
    """Render inline stacked horizontal bar chart for HBM footprint vs capacity."""
    width, row_h = 920, 62
    pad_left, pad_right, pad_top, bar_h = 165, 185, 22, 26
    height = pad_top + max(1, len(pools)) * row_h + 36
    plot_w = width - pad_left - pad_right

    if not pools:
        return (
            f'<svg viewBox="0 0 {width} 120" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="60" text-anchor="middle" '
            'fill="#64748b" font-size="14">'
            "No pool memory data available</text></svg>"
        )

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]
    for i, pool in enumerate(pools):
        y = pad_top + i * row_h
        svg_parts.extend(
            _render_memory_bar(pool, y, (pad_left, plot_w, bar_h), default_limit_gb)
        )

    legend_y = height - 14
    svg_parts.append(
        f'<g transform="translate({pad_left}, {legend_y})">'
        '<rect x="0" y="-9" width="11" height="11" fill="#3b82f6" rx="2"/>'
        '<text x="16" y="0" font-size="12" fill="#334155">Weights</text>'
        '<rect x="92" y="-9" width="11" height="11" fill="#10b981" rx="2"/>'
        '<text x="108" y="0" font-size="12" fill="#334155">KV Cache</text>'
        '<rect x="192" y="-9" width="11" height="11" fill="#f59e0b" rx="2"/>'
        '<text x="208" y="0" font-size="12" fill="#334155">State</text>'
        '<rect x="272" y="-9" width="11" height="11" fill="#e2e8f0" '
        'stroke="#94a3b8" rx="2"/>'
        '<text x="288" y="0" font-size="12" fill="#334155">Free Headroom</text>'
        "</g>"
    )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


CSS_CONTENT = """
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --border: #e2e8f0;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --primary: #2563eb;
  --primary-light: #eff6ff;
  --success: #10b981;
  --warning-bg: #fffbeb;
  --warning-border: #fef3c7;
  --warning-text: #92400e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  background-color: var(--bg-secondary);
  color: var(--text-main);
  line-height: 1.5;
  padding: 20px 24px;
}
.container { max-width: 1280px; margin: 0 auto; }
header {
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
h1 { font-size: 25px; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #1e293b; }
.meta { font-size: 13.5px; color: var(--text-muted); line-height: 1.6; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11.5px;
  font-weight: 600;
  background-color: var(--primary-light);
  color: var(--primary);
  margin-right: 6px;
}
.badge-fact { background-color: #dcfce7; color: #166534; }
.badge-derived { background-color: #e0f2fe; color: #0369a1; }
.badge-assumed { background-color: #fef3c7; color: #92400e; }
.badge-config { background-color: #f3e8ff; color: #6b21a8; }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 20px;
  margin-bottom: 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  overflow-x: auto;
}
.grid-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.stat-box {
  background: var(--bg-secondary);
  padding: 11px 13px;
  border-radius: 6px;
  border: 1px solid var(--border);
}
.stat-label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  font-weight: 600;
}
.stat-value {
  font-size: 19px;
  font-weight: 700;
  color: var(--text-main);
  margin-top: 2px;
}
.alert-box {
  padding: 11px 15px;
  border-radius: 6px;
  margin-bottom: 16px;
  font-size: 13px;
}
.alert-warning {
  background: var(--warning-bg);
  border: 1px solid var(--warning-border);
  color: var(--warning-text);
}
.funnel-container {
  display: flex;
  gap: 8px;
  align-items: stretch;
  margin-top: 10px;
}
.funnel-step {
  flex: 1;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.funnel-step-muted {
  background: #f8fafc;
  border-style: dashed;
  opacity: 0.85;
}
.funnel-arrow {
  color: var(--text-muted);
  font-size: 16px;
  display: flex;
  align-items: center;
}
.chart-container {
  display: flex;
  justify-content: center;
  margin: 8px 0;
}
.chart-svg {
  width: 100%;
  max-width: 940px;
  height: auto;
  font-family: inherit;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 8px;
}
th, td {
  padding: 9px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
th {
  background: var(--bg-secondary);
  font-weight: 600;
  color: #475569;
  font-size: 12px;
}
tr:hover td { background: #f1f5f9; }
.pool-main {
  font-weight: 600;
  color: #0f172a;
  font-size: 13px;
  line-height: 1.35;
}
.pool-sub {
  font-size: 11.5px;
  color: var(--text-muted);
  margin-top: 2px;
}
#top-layouts-table td:first-child { font-weight: 600; }
details.glossary {
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 8px;
  font-size: 12px;
  color: var(--text-muted);
}
details.glossary > summary {
  cursor: pointer;
  font-weight: 600;
  color: #475569;
  font-size: 12.5px;
}
details.glossary ul {
  margin: 6px 0 6px 18px;
  line-height: 1.5;
}
.chart-legend {
  color: var(--text-muted);
  font-size: 12.5px;
  margin-top: 6px;
  text-align: center;
}
.funnel-sub {
  font-size: 11.5px;
  color: var(--text-muted);
  margin-top: 3px;
  line-height: 1.35;
}
footer {
  margin-top: 24px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  font-size: 11.5px;
  color: var(--text-muted);
  text-align: center;
}
"""


def _downsample_embedded_candidates(
    raw_cands: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Downsample raw candidates list when exceeding maximum embedded budget."""
    if len(raw_cands) <= MAX_EMBEDDED_CANDIDATES:
        return list(raw_cands)
    stride = math.ceil(len(raw_cands) / MAX_EMBEDDED_CANDIDATES)
    return list(raw_cands[::stride][:MAX_EMBEDDED_CANDIDATES])


def _pool_inline_label(
    pool: Mapping[str, object],
    *,
    has_moe: bool,
    chip: ChipSpec | None = None,
) -> str:
    """Format one-line pool summary with submodule axes and slice counts."""
    axes_str = format_submodule_parallelism(pool, has_moe=has_moe)
    slice_str = pool_slice_summary(pool, chip=chip)
    return f"{axes_str} · {slice_str}"


def _pool_cell_html(
    pool: Mapping[str, object],
    *,
    has_moe: bool,
    chip: ChipSpec | None = None,
) -> str:
    """Render two-line HTML table cell separating submodule axes from slice meta."""
    axes_str = format_submodule_parallelism(pool, has_moe=has_moe)
    slice_str = pool_slice_summary(pool, chip=chip)
    return (
        f'<td><div class="pool-main"><code>{_escape(axes_str)}</code></div>'
        f'<div class="pool-sub">{_escape(slice_str)}</div></td>'
    )


def _extract_recommended_view(
    recommended: Mapping[str, object] | None,
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
    accept_len: float = 1.0,
) -> tuple[dict[str, object], list[Mapping[str, object]], str]:
    """Extract display metrics, instance pools, and summary for recommended."""
    rec_verdict = recommended.get("slo_verdict") if recommended is not None else None
    rec_feasible = recommended is not None and (
        rec_verdict != SLO_VERDICT_RULED_OUT
        if rec_verdict is not None
        else bool(recommended.get("slo_met", True))
    )
    if recommended is not None and rec_feasible:
        pools_list: list[Mapping[str, object]] = []
        prefill_pool = _as_mapping(recommended.get("prefill_pool"))
        decode_pool = _as_mapping(recommended.get("decode_pool"))
        if prefill_pool:
            pools_list.append(prefill_pool)
        if decode_pool:
            pools_list.append(decode_pool)
        pools_summary = " + ".join(
            f"{str(point.get('role', '')).capitalize()}: "
            f"{_pool_inline_label(point, has_moe=has_moe, chip=chip)}"
            for point in pools_list
        )
        floor = _as_mapping(recommended.get("decode_floor"))
        ttft_val = _ttft_val(recommended)
        tpot_val = _tpot_val(recommended)
        bound_ttft = recommended.get("bound_ttft_ms")
        tpot_bound = _point_tpot_bound_ms(recommended, accept_len=accept_len)
        rec_fields: dict[str, object] = {
            "mode": str(recommended.get("mode", "")).upper(),
            "chips": _as_int(recommended.get("total_chips"), 0),
            "concurrency": _as_int(recommended.get("concurrency"), 1),
            "tput": f"{_as_float(recommended.get('token_throughput_per_chip')):,.1f}",
            "out_tput": (
                f"{_as_float(recommended.get('output_throughput_per_chip')):,.1f}"
            ),
            "interactivity": f"{_interactivity_val(recommended):,.1f}",
            "ttft": (
                f"{_as_float(bound_ttft):.1f} ~ {ttft_val:.1f}"
                if bound_ttft is not None
                else f"{ttft_val:.1f}"
            ),
            "tpot": (
                f"{tpot_bound:.2f} ~ {tpot_val:.2f}"
                if tpot_bound > 0.0
                else f"{tpot_val:.2f}"
            ),
            "binding": str(recommended.get("binding", "hbm")),
            "decode_floor": floor or None,
        }
        return rec_fields, pools_list, pools_summary

    return (
        {
            "mode": "INFEASIBLE",
            "chips": 0,
            "concurrency": 0,
            "tput": "0.0",
            "out_tput": "0.0",
            "interactivity": "0.0",
            "ttft": "-",
            "tpot": "-",
            "binding": "None",
            "decode_floor": None,
        },
        [],
        "No feasible layout met constraints",
    )


def _render_warnings_html(warnings: Sequence[str]) -> str:
    """Format warning messages into an alert box HTML element."""
    if not warnings:
        return ""
    warn_items = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
    return (
        '<div class="alert-box alert-warning">'
        f"<strong>Warnings:</strong><ul>{warn_items}</ul></div>"
    )


def _table_entries(
    top_k: Sequence[Mapping[str, object]],
    reference: Mapping[str, object] | None = None,
) -> list[Mapping[str, object]]:
    """Return the top-k entries plus the reference layout when it is outside top-k."""
    entries: list[Mapping[str, object]] = list(top_k)
    if (
        reference is not None
        and bool(reference.get("found"))
        and not bool(reference.get("in_top_k"))
        and _as_mapping(reference.get("point"))
    ):
        entries.append(
            {
                "frontier_role": "reference",
                "point": _as_mapping(reference.get("point")),
            }
        )
    return entries


def _has_disagg_entries(entries: Sequence[Mapping[str, object]]) -> bool:
    """Return True when any displayed entry carries a separate prefill pool."""
    return any(
        bool(_as_mapping(_as_mapping(entry.get("point")).get("prefill_pool")))
        for entry in entries
    )


def _pools_display(
    candidate_point: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Join prefill and decode pool labels for summary headings."""
    prefill_pool = _as_mapping(candidate_point.get("prefill_pool"))
    decode_pool = _as_mapping(candidate_point.get("decode_pool"))
    if prefill_pool:
        return (
            f"P [{_pool_inline_label(prefill_pool, has_moe=has_moe, chip=chip)}] + "
            f"D [{_pool_inline_label(decode_pool, has_moe=has_moe, chip=chip)}]"
        )
    return _pool_inline_label(decode_pool, has_moe=has_moe, chip=chip)


def _rank_label(rank: int, frontier_role: str) -> str:
    """Return the displayed rank, or a dash for the reference row."""
    return f"#{rank}" if frontier_role != "reference" else "-"


def _render_table_rows(
    top_k: Sequence[Mapping[str, object]],
    reference: Mapping[str, object] | None = None,
    *,
    is_disagg: bool | None = None,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
    accept_len: float = 1.0,
) -> str:
    """Format top candidate layout options into one HTML table row per rank."""
    entries = _table_entries(top_k, reference)
    disagg = _has_disagg_entries(entries) if is_disagg is None else is_disagg
    col_count = _DISAGG_LAYOUT_COLUMNS if disagg else _SINGLE_POOL_LAYOUT_COLUMNS
    if not entries:
        return (
            f'<tr><td colspan="{col_count}" style="text-align:center;">'
            "No feasible layouts found</td></tr>"
        )
    rows: list[str] = []
    for rank, entry in enumerate(entries, 1):
        candidate_point = _as_mapping(entry.get("point"))
        frontier_role = str(entry.get("frontier_role", "candidate"))
        capacity_raw = candidate_point.get("kv_cache_capacity_tokens")
        capacity_str = (
            f"{_as_int(capacity_raw):,}" if capacity_raw is not None else "inf"
        )
        raw_hit_rate = candidate_point.get("effective_prefix_hit_rate")
        hit_rate_str = (
            f"{_as_float(raw_hit_rate):.2f}" if raw_hit_rate is not None else "-"
        )
        prefill_pool = _as_mapping(candidate_point.get("prefill_pool"))
        decode_pool = _as_mapping(candidate_point.get("decode_pool"))
        if disagg:
            prefill_cell = (
                _pool_cell_html(prefill_pool, has_moe=has_moe, chip=chip)
                if prefill_pool
                else "<td>-</td>"
            )
            pool_cells = (
                f"{prefill_cell}"
                f"{_pool_cell_html(decode_pool, has_moe=has_moe, chip=chip)}"
            )
        else:
            pool_cells = _pool_cell_html(decode_pool, has_moe=has_moe, chip=chip)
        tpot_bound_ms = _point_tpot_bound_ms(candidate_point, accept_len=accept_len)
        rows.append(
            f"<tr>"
            f"<td>{_rank_label(rank, frontier_role)}</td>"
            f"<td><code>{_escape(frontier_role)}</code></td>"
            f"<td><strong>{_escape(str(candidate_point.get('mode', '')))}</strong></td>"
            f"<td>{_as_int(candidate_point.get('concurrency'), 1)}</td>"
            f"<td>{_as_int(candidate_point.get('total_chips'), 0)}</td>"
            f"{pool_cells}"
            f"<td>{_interactivity_val(candidate_point):,.1f}</td>"
            f"<td>{_as_float(candidate_point.get('token_throughput_per_chip')):,.1f}</td>"
            f"<td>{tpot_bound_ms:.2f}</td>"
            f"<td>{_ttft_val(candidate_point):,.1f}</td>"
            f"<td>{_tpot_val(candidate_point):.2f}</td>"
            f"<td>{_escape(capacity_str)} / {hit_rate_str}</td>"
            f"</tr>"
        )
    return "".join(rows)


def _derive_default_hbm_limit_gb(payload: Mapping[str, object]) -> float:
    """Derive per-core physical HBM limit (GB) from chip_spec minus reserve."""
    chip_spec = _as_mapping(payload.get("chip_spec"))
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    fallback_chip = CHIPS["tpu7x"]
    hbm_bytes = _as_float(chip_spec.get("hbm_bytes"), float(fallback_chip.hbm_bytes))
    return max(0.0, hbm_bytes / UNIT_GIGA - reserve_gb)


def _build_dynamic_chip_spec(
    chip_spec: Mapping[str, object],
    fallback: ChipSpec,
    chip_name: str,
) -> ChipSpec:
    """Build ChipSpec dynamically from serialized payload chip_spec mapping."""
    kwargs: dict[str, object] = {}
    for spec_field in fields(ChipSpec):
        field_name = spec_field.name
        if field_name == "name":
            kwargs[field_name] = chip_name
            continue
        default_val = getattr(fallback, field_name)
        raw_val = chip_spec.get(field_name)
        if isinstance(default_val, bool):
            kwargs[field_name] = bool(raw_val) if raw_val is not None else default_val
        elif isinstance(default_val, int):
            kwargs[field_name] = (
                int(_as_float(raw_val, float(default_val)))
                if raw_val is not None
                else default_val
            )
        elif isinstance(default_val, float):
            kwargs[field_name] = _as_float(raw_val, default_val)
        else:
            kwargs[field_name] = (
                str(raw_val) if raw_val is not None and str(raw_val) else default_val
            )
    return ChipSpec(**kwargs)  # type: ignore[arg-type]


def _payload_chip_spec(payload: Mapping[str, object]) -> ChipSpec | None:
    """Reconstruct ChipSpec from payload['chip_spec'] when present."""
    chip_spec = _as_mapping(payload.get("chip_spec"))
    if not chip_spec:
        return None
    fallback = CHIPS["tpu7x"]
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    return _build_dynamic_chip_spec(chip_spec, fallback, chip_name)


def _format_catalog_entry_row(key: str, entry: Mapping[str, object]) -> str:
    """Render one catalog entry into an HTML table row."""
    kind = _escape(entry.get("kind", ""))
    scope = _escape(entry.get("scope", ""))
    unit = str(entry.get("unit", ""))
    val = entry.get("value", "")
    val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
    desc_str = _escape(entry.get("description", ""))
    src = _escape(entry.get("source", ""))
    display = entry.get("display")
    computed = (
        format_constant_display(_as_float(val), unit)
        if (val != "" and unit != "")
        else ""
    )
    if not display or (
        computed and str(computed).split()[0] != str(display).split()[0]
    ):
        display = computed or display
    display_str = _escape(display or f"{val_str} {unit}")
    val_sub = f"{val} {unit}".strip()
    return (
        "          <tr>"
        f"<td><code>{_escape(key)}</code></td>"
        f"<td><strong>{display_str}</strong><br>"
        f'<small style="color: #64748b;">{_escape(val_sub)}</small></td>'
        f"<td><code>{scope}</code></td>"
        f'<td><span class="badge badge-{kind}">{kind}</span></td>'
        f"<td>{desc_str}</td>"
        f'<td style="font-size: 0.85em; color: #475569;">{src}</td>'
        "</tr>"
    )


def _render_chip_specs_item(payload: Mapping[str, object]) -> str:
    """Render dynamic Per-Core hardware specification line from chip_spec."""
    fallback = CHIPS["tpu7x"]
    chip_spec = _as_mapping(payload.get("chip_spec"))
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    hbm_gb = (
        _as_float(chip_spec.get("hbm_bytes"), float(fallback.hbm_bytes)) / UNIT_GIGA
    )
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    usable_gb = max(0.0, hbm_gb - reserve_gb)
    reserve_note = (
        f" minus <code>{reserve_gb:.2f} GB</code> HBM reserve"
        if reserve_gb > 0.0
        else ""
    )
    hbm_tbs = _as_float(chip_spec.get("hbm_bw"), fallback.hbm_bw) / UNIT_TERA
    bf16_pf = _as_float(chip_spec.get("bf16_flops"), fallback.bf16_flops) / UNIT_PETA
    fp8_pf = _as_float(chip_spec.get("fp8_flops"), fallback.fp8_flops) / UNIT_PETA
    extra_compute_parts: list[str] = []
    for key, label, default_val in (
        ("fp4_flops", "FP4", fallback.fp4_flops),
        ("int8_flops", "INT8", fallback.int8_flops),
        ("int4_flops", "INT4", fallback.int4_flops),
    ):
        rate_pf = _as_float(chip_spec.get(key), default_val) / UNIT_PETA
        if rate_pf > 0.0:
            extra_compute_parts.append(f"<code>{rate_pf:.3f} PFLOP/s</code> {label}, ")
    vector_tf = (
        _as_float(chip_spec.get("vector_flops"), fallback.vector_flops) / UNIT_TERA
    )
    if vector_tf > 0.0:
        extra_compute_parts.append(f"<code>{vector_tf:.2f} TFLOP/s</code> Vector, ")
    extra_compute_str = "".join(extra_compute_parts)
    ici_link_gbps = (
        _as_float(chip_spec.get("ici_link_bw"), fallback.ici_link_bw) / UNIT_GIGA
    )
    ici_bw_gbps = _as_float(chip_spec.get("ici_bw"), fallback.ici_bw) / UNIT_GIGA
    d2d_gbps = _as_float(chip_spec.get("d2d_bw"), fallback.d2d_bw) / UNIT_GIGA
    hop_us = (
        _as_float(chip_spec.get("ici_hop_latency_s"), fallback.ici_hop_latency_s)
        * UNIT_MEGA
    )
    dcn_gbps = _as_float(chip_spec.get("dcn_bw"), fallback.dcn_bw) / UNIT_GIGA
    pcie_gbps = _as_float(chip_spec.get("pcie_bw"), fallback.pcie_bw) / UNIT_GIGA
    matrix_tile_rows = _as_int(chip_spec.get("mxu_m_tile"), 0)
    tile_str = (
        f"<code>{matrix_tile_rows}</code>-row systolic matrix tile, "
        if matrix_tile_rows > 0
        else ""
    )
    return (
        f"      <li><strong>{_escape(chip_name)} Per-Core Specs:</strong> "
        f"<code>{hbm_gb:.1f} GB</code> HBM "
        f"(<code>{usable_gb:.2f} GB</code> usable{reserve_note}), "
        f"<code>{hbm_tbs:.2f} TB/s</code> HBM, "
        f"<code>{bf16_pf:.3f} PFLOP/s</code> BF16, "
        f"<code>{fp8_pf:.3f} PFLOP/s</code> FP8, "
        f"{extra_compute_str}"
        f"{tile_str}"
        f"<code>{ici_link_gbps:.0f} GB/s</code>/link "
        f"(<code>{ici_bw_gbps:.0f} GB/s</code>/core, "
        f"<code>{d2d_gbps:.0f} GB/s</code>/chip D2D) ICI + "
        f"<code>{hop_us:.1f} us</code> hop, "
        f"<code>{dcn_gbps:.2f} GB/s</code>/chip DCN, and "
        f"<code>{pcie_gbps:.1f} GB/s</code>/chip PCIe.</li>"
    )


def _format_assumption_entry(name: str, entry: object) -> str:
    """Format one supplied or solver assumption entry as inline HTML."""
    if isinstance(entry, Mapping):
        label = str(entry.get("flag") or entry.get("name") or name)
        kind = str(entry.get("kind") or "")
        unit = str(entry.get("unit") or "")
        val = entry.get("value", entry.get("default", ""))
        val_txt = f"{val} {unit}".strip()
        kind_suffix = f" ({_escape(kind)})" if kind else ""
        return f"<code>{_escape(label)}={_escape(val_txt)}</code>{kind_suffix}"
    return f"<code>{_escape(name)}={_escape(entry)}</code>"


def _format_supplied_values_html(raw_supplied: object) -> str:
    """Format user-supplied values from provenance metadata as inline HTML."""
    if isinstance(raw_supplied, Mapping):
        parts = [
            _format_assumption_entry(str(k), v)
            for k, v in raw_supplied.items()
            if v is not None
        ]
        return ", ".join(parts)
    if isinstance(raw_supplied, Sequence) and not isinstance(
        raw_supplied, (str, bytes)
    ):
        seq_parts: list[str] = []
        for idx, item in enumerate(raw_supplied):
            if isinstance(item, Mapping):
                key = str(item.get("name") or item.get("flag") or f"arg_{idx}")
                seq_parts.append(_format_assumption_entry(key, item))
            elif item is not None:
                seq_parts.append(f"<code>{_escape(item)}</code>")
        return ", ".join(seq_parts)
    return ""


def _render_provenance_item(payload: Mapping[str, object]) -> str:
    """Render guarded Model Provenance HTML list item when present in payload."""
    prov = _as_mapping(payload.get("provenance"))
    if not prov:
        return ""
    calibrated = bool(prov.get("calibrated", False))
    basis = str(prov.get("basis") or "")
    ranking_basis = str(prov.get("ranking_basis") or "")
    supplied_html = _format_supplied_values_html(prov.get("supplied_values"))
    segments = [f"calibrated=<code>{calibrated}</code>"]
    if basis:
        segments.append(_escape(basis))
    if ranking_basis:
        segments.append(f"ranking: <code>{_escape(ranking_basis)}</code>")
    if supplied_html:
        segments.append(f"supplied: {supplied_html}")
    return f"      <li><strong>Provenance:</strong> {' &middot; '.join(segments)}.</li>"


def _render_solver_assumptions_items(payload: Mapping[str, object]) -> list[str]:
    """Render guarded Solver Assumptions list item(s) when present in payload."""
    raw_assumptions = payload.get("solver_assumptions")
    formatted = _format_supplied_values_html(raw_assumptions)
    if not formatted:
        return []
    return [f"      <li><strong>Solver Assumptions:</strong> {formatted}.</li>"]


def _render_assumptions_card(payload: Mapping[str, object]) -> list[str]:
    """Render Assumptions, Hardware Specs & Scope card with constants catalog."""
    fallback = CHIPS["tpu7x"]
    chip_spec = _as_mapping(payload.get("chip_spec"))
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    cat_raw = chip_spec.get("catalog")
    if isinstance(cat_raw, Mapping):
        cat = cat_raw
    else:
        cat = chip_catalog(_build_dynamic_chip_spec(chip_spec, fallback, chip_name))
    cat_rows = [
        _format_catalog_entry_row(str(k), _as_mapping(entry_raw))
        for k, entry_raw in cat.items()
        if isinstance(entry_raw, Mapping)
    ]
    chip_specs_item = _render_chip_specs_item(payload)
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    reserve_items = (
        [
            "      <li><strong>HBM Workspace Reserve:</strong> "
            f"<code>{reserve_gb:.2f} GB/core</code> subtracted for runtime/XLA "
            "workspace.</li>",
            "      <li><strong>Omitted in Analytical Scope:</strong> "
            "Empirical kernel utilization derates and trace-level LRU cache "
            "eviction or arrival burstiness.</li>",
        ]
        if reserve_gb > 0.0
        else [
            "      <li><strong>Omitted in Analytical Scope:</strong> "
            "Empirical kernel utilization derates, XLA temporary workspace, "
            "and trace-level LRU cache eviction or arrival burstiness.</li>",
        ]
    )
    provenance_item = _render_provenance_item(payload)
    provenance_items = [provenance_item] if provenance_item else []
    assumption_items = _render_solver_assumptions_items(payload)
    return [
        '  <section class="card assumptions-card">',
        "    <h2>Assumptions, Hardware Specs &amp; Scope</h2>",
        "    <ul>",
        "      <li><strong>Datasheet Bounds:</strong> <code>Hard Floor</code> is "
        "<code>max(HBM, Matmul, Vector, ICI)</code> "
        "(full communication-compute overlap); "
        "<code>Serial Estimate</code> sums per-module "
        "<code>max(HBM, Matmul, Vector)</code> plus exposed <code>ICI Chain</code> "
        "(zero overlap) for layout ranking.</li>",
        chip_specs_item,
        *reserve_items,
        *provenance_items,
        *assumption_items,
        "    </ul>",
        "    <details open><summary>Constants Catalog "
        "(fact / derived / assumed / config)</summary>",
        '    <div style="overflow-x: auto; margin-top: 0.75rem;">',
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th style="text-align: left;">Constant</th>',
        '            <th style="text-align: left;">Value</th>',
        '            <th style="text-align: left;">Scope</th>',
        '            <th style="text-align: left;">Kind</th>',
        '            <th style="text-align: left;">Description</th>',
        '            <th style="text-align: left;">Source / Derivation</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        *[f"    {row}" for row in cat_rows],
        "        </tbody>",
        "      </table>",
        "    </div>",
        "    </details>",
        "  </section>",
    ]


def _render_crossover_stat_html(payload: Mapping[str, object]) -> str:
    """Render prefix-hit regime banner box when sweep winner range exists."""
    win_range = _as_sequence(payload.get("winner_prefix_hit_range"))
    if len(win_range) != PREFIX_RANGE_PAIR_LEN:
        return ""
    cross_rate = payload.get("prefix_hit_crossover_rate")
    range_txt = f"[{_as_float(win_range[0]):.2f}, {_as_float(win_range[1]):.2f}]"
    cross_txt = (
        f"crossover={_as_float(cross_rate):.2f}" if cross_rate is not None else "stable"
    )
    return (
        '      <div class="stat-box">'
        '        <div class="stat-label">Prefix-Hit Regime</div>'
        '        <div class="stat-value" style="font-size:15px;">'
        f"{_escape(range_txt)} "
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        f"({_escape(cross_txt)})</span></div>"
        "      </div>"
    )


def _format_bound_family_label(
    entry: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Format a candidate family key cleanly for the analytical handoff table."""
    raw_fam = entry.get("family")
    if isinstance(raw_fam, str) and raw_fam:
        return raw_fam
    if isinstance(raw_fam, Sequence) and not isinstance(raw_fam, (str, bytes)):
        parts = [str(part) for part in raw_fam if part is not None and str(part)]
        return " | ".join(parts) if parts else "-"
    if isinstance(raw_fam, Mapping):
        return ", ".join(f"{k}={v}" for k, v in raw_fam.items()) or "-"
    members = _as_sequence(entry.get("members"))
    if members and isinstance(members[0], Mapping) and "decode_pool" in members[0]:
        return _pools_display(members[0], has_moe=has_moe, chip=chip)
    return str(members[0]) if members else "-"


def _format_bound_family_members(
    entry: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Format member count and distinct pool labels inside one candidate family."""
    member_count = _as_int(entry.get("member_count"), 0)
    members = _as_sequence(entry.get("members"))
    labels: list[str] = []
    for member in members:
        if isinstance(member, Mapping) and "decode_pool" in member:
            labels.append(_pools_display(member, has_moe=has_moe, chip=chip))
        elif isinstance(member, Mapping) and isinstance(member.get("pool"), Mapping):
            labels.append(
                _pool_inline_label(
                    _as_mapping(member.get("pool")), has_moe=has_moe, chip=chip
                )
            )
        elif isinstance(member, Mapping) and "tp" in member:
            labels.append(_pool_inline_label(member, has_moe=has_moe, chip=chip))
        elif member is not None:
            labels.append(str(member))
    count_prefix = f"{member_count} total: " if member_count > 0 else ""
    joined = "; ".join(labels) if labels else "-"
    return f"{count_prefix}{joined}"


def _render_bound_family_row(
    conc_key: str,
    rank_idx: int,
    entry: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Render one bound-ordered candidate family row for analytical handoff."""
    rank = _as_int(entry.get("rank"), rank_idx)
    family_label = _format_bound_family_label(entry, has_moe=has_moe, chip=chip)
    min_bound_tpot_raw = entry.get("min_bound_tpot_ms")
    min_bound_ttft_raw = entry.get("min_bound_ttft_ms")
    min_bound_tpot_str = (
        f"{_as_float(min_bound_tpot_raw):.2f}"
        if min_bound_tpot_raw is not None
        else "-"
    )
    min_bound_ttft_str = (
        f"{_as_float(min_bound_ttft_raw):.1f}"
        if min_bound_ttft_raw is not None
        else "-"
    )
    env_tpot_raw = entry.get("min_envelope_tpot_ms")
    env_ttft_raw = entry.get("min_envelope_ttft_ms")
    env_tpot_str = f"{_as_float(env_tpot_raw):.2f}" if env_tpot_raw is not None else "-"
    env_ttft_str = f"{_as_float(env_ttft_raw):.1f}" if env_ttft_raw is not None else "-"
    binding = str(entry.get("binding") or "-")
    any_not_ruled_out = bool(entry.get("any_not_ruled_out", True))
    verdict_str = "yes" if any_not_ruled_out else "ruled_out"
    members_str = _format_bound_family_members(entry, has_moe=has_moe, chip=chip)
    return (
        "<tr>"
        f"<td><code>c={_escape(conc_key)}</code></td>"
        f"<td>#{rank}</td>"
        f"<td><code>{_escape(family_label)}</code></td>"
        f"<td>{min_bound_tpot_str}</td>"
        f"<td>{min_bound_ttft_str}</td>"
        f"<td>{env_tpot_str} / {env_ttft_str}</td>"
        f"<td><code>{_escape(binding)}</code></td>"
        f"<td><code>{verdict_str}</code></td>"
        f"<td><small>{_escape(members_str)}</small></td>"
        "</tr>"
    )


def _render_bound_families_section(
    payload: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> list[str]:
    """Render guarded Bound-Ordered Candidate Families section."""
    raw_by_conc = _as_mapping(payload.get("bound_families_by_concurrency"))
    if not raw_by_conc:
        return []
    rows: list[str] = []
    for conc_key in sorted(raw_by_conc, key=lambda key: _as_int(key, 0)):
        entries = _as_sequence(raw_by_conc[conc_key])
        for rank_idx, raw_entry in enumerate(entries, 1):
            entry = _as_mapping(raw_entry)
            if entry:
                rows.append(
                    _render_bound_family_row(
                        str(conc_key),
                        rank_idx,
                        entry,
                        has_moe=has_moe,
                        chip=chip,
                    )
                )
    if not rows:
        return []
    return [
        '  <section id="bound-families-table" class="card">',
        "    <h2>Bound-Ordered Candidate Families (Benchmark Handoff)</h2>",
        '    <p class="section-note" '
        'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
        "Top candidate families per concurrency ordered strictly by physical "
        "<code>min(bound_tpot_ms)</code> (tie-broken by "
        "<code>min(bound_ttft_ms)</code>) for TPU benchmarking handoff.</p>",
        "    <table>",
        "      <thead><tr>"
        "<th>Concurrency</th><th>Rank</th><th>Family</th>"
        "<th>Min Bound TPOT (ms)</th><th>Min Bound TTFT (ms)</th>"
        "<th>Min Env TPOT / TTFT (ms)</th><th>Binding</th>"
        "<th>Not Ruled Out</th><th>Pool Variants</th>"
        "</tr></thead>",
        f"      <tbody>{''.join(rows)}</tbody>",
        "    </table>",
        "  </section>",
    ]


def _format_opt_ms(value: object) -> str:
    """Format an optional millisecond value or a dash when absent."""
    if value is None or isinstance(value, bool):
        return "-"
    return f"{_as_float(value):,.2f}"


def _render_single_pool_table(
    title: str,
    records: Sequence[Mapping[str, object]],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> list[str]:
    """Render one ordered prefill or decode pool table."""
    if not records:
        return []
    rows: list[str] = []
    for record in records:
        pool = _as_mapping(record.get("pool"))
        label = _pool_inline_label(pool, has_moe=has_moe, chip=chip)
        concurrency = _as_int(record.get("concurrency"), 1)
        chips = _as_int(pool.get("chips_per_instance"), 0) * max(
            1, _as_int(pool.get("instances"), 1)
        )
        role = str(record.get("role", ""))
        floor_ms = (
            record.get("ttft_floor_ms")
            if role == "prefill"
            else record.get("tpot_floor_ms")
        )
        est_ms = record.get("ttft_ms") if role == "prefill" else record.get("tpot_ms")
        inter = record.get("interactivity_tok_s")
        inter_str = f"{_as_float(inter):,.1f}" if inter is not None else "-"
        tput = _as_float(record.get("token_throughput_per_chip"))
        pareto_tag = "yes" if bool(record.get("is_pareto")) else "no"
        binding = _escape(record.get("binding", ""))
        rows.append(
            f"<tr><td><code>c={concurrency}</code></td><td>{_escape(label)}</td>"
            f"<td>{chips}</td><td>{_format_opt_ms(floor_ms)}</td>"
            f"<td>{_format_opt_ms(est_ms)}</td><td>{inter_str}</td>"
            f"<td>{tput:,.1f}</td><td><code>{pareto_tag}</code></td>"
            f"<td><code>{binding}</code></td></tr>"
        )
    latency_hdr = "TTFT" if title.startswith("Prefill") else "TPOT"
    return [
        f"    <h3>{_escape(title)}</h3>",
        "    <table>",
        f"      <thead><tr><th>Concurrency</th><th>Pool</th><th>Chips</th>"
        f"<th>{latency_hdr} Floor (ms)</th><th>Est. {latency_hdr} (ms)</th>"
        "<th>Est. Int (tok/s)</th><th>Est. Tput (tok/s/chip)</th>"
        "<th>Pareto</th><th>Binding</th></tr></thead>",
        f"      <tbody>{''.join(rows)}</tbody>",
        "    </table>",
    ]


def _render_pool_tables_section(
    payload: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> list[str]:
    """Render ordered per-pool prefill and decode tables when present."""
    prefill = [_as_mapping(x) for x in _as_sequence(payload.get("prefill_pools"))]
    decode = [_as_mapping(x) for x in _as_sequence(payload.get("decode_pools"))]
    if not prefill and not decode:
        return []
    sort_by = str(payload.get("sort_by") or "throughput")
    parts: list[str] = [
        '  <section id="pool-tables" class="card">',
        "    <h2>Ordered Per-Pool Tables (Prefill &amp; Decode)</h2>",
        '    <p class="section-note" '
        'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
        f"All physically legal pool layouts ordered per concurrency by "
        f"<code>{_escape(sort_by)}</code> with physical floors, estimates, "
        "and Pareto marks.</p>",
    ]
    parts.extend(
        _render_single_pool_table(
            "Prefill Pool Layouts", prefill, has_moe=has_moe, chip=chip
        )
    )
    parts.extend(
        _render_single_pool_table(
            "Decode Pool Layouts", decode, has_moe=has_moe, chip=chip
        )
    )
    parts.append("  </section>")
    return parts


def _render_ordered_concurrency_section(
    payload: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
    accept_len: float = 1.0,
) -> list[str]:
    """Render ordered candidates per concurrency with floor, estimate, and Pareto."""
    raw_by_conc = _as_mapping(payload.get("concurrency_top_k"))
    if not raw_by_conc:
        return []
    sort_by = str(payload.get("sort_by") or "throughput")
    rows: list[str] = []
    for conc_key in sorted(raw_by_conc, key=lambda key: _as_int(key, 0)):
        entries = _as_sequence(raw_by_conc[conc_key])
        if not entries:
            continue
        items: list[str] = []
        for rank, raw_entry in enumerate(entries, 1):
            entry = _as_mapping(raw_entry)
            point = _as_mapping(entry.get("point"))
            if not point:
                continue
            label = _pools_display(point, has_moe=has_moe, chip=chip)
            bound_ms = _point_tpot_bound_ms(point, accept_len=accept_len)
            tput = _as_float(point.get("token_throughput_per_chip"))
            inter = _interactivity_val(point)
            pareto_suffix = " · Pareto" if bool(point.get("is_pareto")) else ""
            summary = (
                f"#{rank} {label} "
                f"· floor {bound_ms:.2f} ms · est {tput:,.1f} tok/s/chip "
                f"· est {inter:,.1f} tok/s{pareto_suffix}"
            )
            items.append(f"<div>{_escape(summary)}</div>")
        rows.append(
            f"<tr><td><code>c={_escape(str(conc_key))}</code></td>"
            f"<td>{''.join(items)}</td></tr>"
        )
    if not rows:
        return []
    return [
        '  <section id="concurrency-ordered-table" class="card">',
        "    <h2>Ordered Candidates by Concurrency</h2>",
        '    <p class="section-note" '
        'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
        f"Physically legal layouts ordered at each concurrency by "
        f"<code>{_escape(sort_by)}</code> with physical TPOT floors and "
        "estimates.</p>",
        "    <table>",
        "      <thead><tr><th>Concurrency</th>"
        "<th>Ordered Candidates (Floor &amp; Estimate)</th></tr></thead>",
        f"      <tbody>{''.join(rows)}</tbody>",
        "    </table>",
        "  </section>",
    ]


def _format_chip_count(chips: int) -> str:
    """Format chip count with singular or plural noun."""
    return "1 chip" if chips == 1 else f"{chips} chips"


def _detect_has_moe(
    payload: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> bool:
    """Detect whether payload represents an MoE model requiring Attn | MoE split."""
    raw_flag = payload.get("has_moe")
    if isinstance(raw_flag, bool):
        return raw_flag
    if any(
        _as_int(_as_mapping(cand.get("decode_pool")).get("ep"), 1) > 1
        or _as_int(_as_mapping(cand.get("prefill_pool")).get("ep"), 1) > 1
        for cand in candidates
    ):
        return True
    for pool_key in ("prefill_pools", "decode_pools"):
        for raw_rec in _as_sequence(payload.get(pool_key)):
            pool = _as_mapping(_as_mapping(raw_rec).get("pool"))
            if _as_int(pool.get("ep"), 1) > 1:
                return True
    return False


def _format_dtype_bytes(byte_value: object) -> str:
    """Format byte width per element with trailing 'B' unit."""
    if isinstance(byte_value, (int, float)) and not isinstance(byte_value, bool):
        return f"{byte_value:.6g}B"
    return f"{byte_value}B"


def _render_glossary_unit_item(
    payload: Mapping[str, object],
    report_chip: ChipSpec | None,
) -> str:
    """Render the Core / Chip / Host / Slice glossary bullet for the active chip."""
    tpu7x_spec = CHIPS["tpu7x"]
    if report_chip is None or (
        report_chip.name == tpu7x_spec.name
        and report_chip.cores_per_chip == tpu7x_spec.cores_per_chip
        and report_chip.chips_per_host == tpu7x_spec.chips_per_host
    ):
        return (
            "<li><strong>Core / Chip / Host / Slice:</strong> On TPU v7x, "
            "1 chip = 2 TensorCores; 1 host = 4 chips (8 TensorCores); "
            "<code>2x2x1</code> = 4 chips / 1 host, "
            "<code>2x2x2</code> = 8 chips / 2 hosts. The prefix "
            "<code>1x</code> or <code>5x</code> counts independent "
            "serving instances.</li>"
        )
    cores_per_chip = report_chip.cores_per_chip
    chips_per_host = report_chip.chips_per_host
    cores_per_host = cores_per_chip * chips_per_host
    chip_core_unit = "TensorCore" if cores_per_chip == 1 else "TensorCores"
    host_core_unit = "TensorCore" if cores_per_host == 1 else "TensorCores"
    host_fit = _as_mapping(payload.get("single_host_fit"))
    raw_topo = host_fit.get("topology") if host_fit is not None else None
    host_topo = (
        str(raw_topo) if isinstance(raw_topo, str) and raw_topo else f"{chips_per_host}"
    )
    chip_name_esc = html.escape(report_chip.name)
    host_topo_esc = html.escape(host_topo)
    chips_label = _format_chip_count(chips_per_host)
    return (
        f"<li><strong>Core / Chip / Host / Slice:</strong> On {chip_name_esc}, "
        f"1 chip = {cores_per_chip} {chip_core_unit}; "
        f"1 host = {chips_label} ({cores_per_host} {host_core_unit}); "
        f"<code>{host_topo_esc}</code> = {chips_label} / 1 host. "
        "The prefix <code>1x</code> or <code>5x</code> counts independent "
        "serving instances.</li>"
    )


def _render_header_meta_suffix(
    payload: Mapping[str, object],
    *,
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST,
) -> str:
    """Format budget, pin, and dtype metadata suffix for the report header."""
    chip_budget = _as_mapping(payload.get("chip_budget")) or {}
    budget_chips = _as_int(chip_budget.get("max_chips", payload.get("budget", 0)))
    budget_source = str(
        chip_budget.get("budget_source", payload.get("budget_source", "flag"))
    )
    budget_hosts = count_hosts(budget_chips, chips_per_host=chips_per_host)
    budget_text = (
        f"Search Budget: {_format_chip_count(budget_chips)} / "
        f"{format_host_count(budget_hosts)} ({budget_source})"
        if budget_chips > 0
        else ""
    )
    pins = _as_mapping(chip_budget.get("pins", payload.get("pins"))) or {}
    pin_parts: list[str] = []
    if pins.get("prefill_topology"):
        pin_parts.append(f"P-slice: {pins['prefill_topology']}")
    if pins.get("decode_topology"):
        pin_parts.append(f"D-slice: {pins['decode_topology']}")
    pd_ratio_raw = pins.get("pd_ratio")
    if isinstance(pd_ratio_raw, str) and pd_ratio_raw:
        pin_parts.append(f"P:D={pd_ratio_raw}")
    elif (
        isinstance(pd_ratio_raw, Sequence)
        and not isinstance(pd_ratio_raw, (str, bytes))
        and len(pd_ratio_raw) >= PREFIX_RANGE_PAIR_LEN
    ):
        pin_parts.append(f"P:D={pd_ratio_raw[0]}:{pd_ratio_raw[1]}")
    elif isinstance(pins.get("pd"), str) and pins.get("pd"):
        pin_parts.append(f"P:D={pins['pd']}")
    pin_text = f" [Pins: {', '.join(pin_parts)}]" if pin_parts else ""

    dtypes_map = _as_mapping(payload.get("dtypes")) or {}
    dtypes_text = ""
    if dtypes_map:
        dense_dtype = dtypes_map.get("w_dense_dtype", "dense")
        dense_source = dtypes_map.get("w_dense_source", "default")
        dense_bytes_text = _format_dtype_bytes(dtypes_map.get("w_dense", 0.0))
        expert_dtype = dtypes_map.get("w_expert_dtype", "expert")
        expert_source = dtypes_map.get("w_expert_source", "default")
        expert_bytes_text = _format_dtype_bytes(dtypes_map.get("w_expert", 0.0))
        kv_dtype = dtypes_map.get("kv_dtype", "kv")
        kv_source = dtypes_map.get("kv_source", "default")
        kv_bytes_text = _format_dtype_bytes(dtypes_map.get("kv", 0.0))
        activation_dtype = dtypes_map.get("act", "?")
        dtypes_text = (
            f"Dtypes: dense={dense_dtype} ({dense_bytes_text}, {dense_source}), "
            f"expert={expert_dtype} ({expert_bytes_text}, {expert_source}), "
            f"kv={kv_dtype} ({kv_bytes_text}, {kv_source}), act={activation_dtype}"
        )

    suffix = (
        f" &bull; <strong>{_escape(budget_text)}{_escape(pin_text)}</strong>"
        if budget_text
        else ""
    )
    if dtypes_text:
        suffix += f" &bull; <span>{_escape(dtypes_text)}</span>"
    return suffix


def _render_single_host_fit_card(
    payload: Mapping[str, object],
    *,
    has_moe: bool,
) -> str:
    """Render single-host slice fit verdict card when present in payload."""
    single_host_fit = _as_mapping(payload.get("single_host_fit"))
    if not single_host_fit:
        return ""
    chip_spec = _as_mapping(payload.get("chip_spec")) or {}
    topology = str(single_host_fit.get("topology") or "2x2x1")
    chips_per_host = _as_int(
        single_host_fit.get("chips_per_host"),
        _as_int(chip_spec.get("chips_per_host"), CHIPS["tpu7x"].chips_per_host),
    )
    cores_per_chip = _as_int(
        chip_spec.get("cores_per_chip"), CHIPS["tpu7x"].cores_per_chip
    )
    cores_per_host = _as_int(
        single_host_fit.get("cores_per_host"), chips_per_host * cores_per_chip
    )
    is_feasible = bool(single_host_fit.get("feasible", False))
    fit_status = str(single_host_fit.get("status", "REJECTED"))
    fit_reason = str(single_host_fit.get("reason", ""))
    hbm_used_gb = _as_float(single_host_fit.get("hbm_used_gb"))
    hbm_limit_gb = _as_float(single_host_fit.get("hbm_limit_gb"))
    submod_best = single_host_fit.get("best_submodule_parallelism") if has_moe else None
    raw_best = (
        submod_best
        if submod_best
        else single_host_fit.get("best_parallelism", f"TP{cores_per_host}")
    )
    best_parallelism = "N/A" if raw_best is None else str(raw_best)
    bg_color = "#dcfce7" if is_feasible else "#fee2e2"
    text_color = "#166534" if is_feasible else "#991b1b"
    return (
        '  <section id="single-host-fit" class="card">\n'
        f"    <h2>Single-Host {_escape(topology)} Fit Verdict "
        f"(1 host, {_format_chip_count(chips_per_host)}, "
        f"{cores_per_host} TensorCores)</h2>\n"
        '    <div style="font-size:14px; margin-top:8px;">\n'
        f'      <span class="badge" style="background:{bg_color}; '
        f'color:{text_color}; font-weight:bold;">{_escape(fit_status)}</span> '
        "&bull; \n"
        f"      <strong>{_escape(fit_reason)}</strong> &bull; \n"
        f"      HBM Footprint: <code>{hbm_used_gb:.2f} GB</code> / "
        f"<code>{hbm_limit_gb:.2f} GB</code> limit &bull; \n"
        f"      Top-Ranked Single-Host Candidate: "
        f"<code>{_escape(best_parallelism)}</code>\n"
        "    </div>\n"
        "  </section>"
    )


def _render_funnel_section(
    payload: Mapping[str, object],
    frontier: Sequence[Mapping[str, object]],
    *,
    top_k_len: int,
) -> list[str]:
    """Render 5-stage Search Funnel HTML section for solve/v2 payloads."""
    search = _as_mapping(payload.get("search"))
    scenario = _as_mapping(payload.get("scenario"))
    enumerated_count = _as_int(search.get("enumerated"))
    core_grid_count = max(
        enumerated_count, _as_int(search.get("core_grid"), enumerated_count)
    )
    hbm_fit_count = _as_int(search.get("hbm_fit"))
    slo_met_count = _as_int(search.get("slo_met"))
    has_slo = (
        scenario.get("ttft_slo_ms") is not None
        or scenario.get("tpot_slo_ms") is not None
    )
    slo_step_class = "funnel-step" if has_slo else "funnel-step funnel-step-muted"
    if "not_ruled_out" in search:
        slo_stage_count = _as_int(search.get("not_ruled_out"))
        ruled_out_count = max(0, hbm_fit_count - slo_stage_count)
        undecided_count = max(0, slo_stage_count - slo_met_count)
        slo_label = "Not Ruled Out" if has_slo else "SLO Bound Check"
        slo_breakdown_html = (
            '<div class="funnel-sub">'
            f"Ruled Out (bound &gt; SLO): {ruled_out_count}<br>"
            f"Est. Meets SLO: {slo_met_count} &middot; Undecided: {undecided_count}"
            "</div>"
            if has_slo
            else '<div class="funnel-sub">No SLO set (skipped)</div>'
        )
    else:
        slo_stage_count = slo_met_count if has_slo else hbm_fit_count
        slo_label = "SLO Met" if has_slo else "SLO Check"
        slo_breakdown_html = (
            f'<div class="funnel-sub">{max(0, hbm_fit_count - slo_met_count):,} '
            "violate latency SLO</div>"
            if has_slo
            else '<div class="funnel-sub">No SLO set (skipped)</div>'
        )
    hbm_dropped = max(0, enumerated_count - hbm_fit_count)
    hbm_sub_html = (
        f'<div class="funnel-sub">{hbm_dropped:,} fail physical HBM fit '
        "(weights + logical KV + recurrent state)</div>"
        if hbm_dropped > 0
        else '<div class="funnel-sub">Fits physical HBM '
        "(weights + logical KV + recurrent state)</div>"
    )
    frontier_2d_count = len(_frontier_2d_envelope(_filter_slo_preferred(frontier)))
    raw_regime = payload.get("activation_regime")
    funnel_heading = (
        f"Search Funnel (MoE Activation: {_escape(raw_regime)})"
        if isinstance(raw_regime, str) and raw_regime
        else "Search Funnel"
    )
    envelope_feasible_html = (
        '        <div class="stat-label" style="margin-top:2px;">'
        f"{_as_int(search['envelope_feasible']):,} Envelope Feasible</div>"
        if has_slo and "envelope_feasible" in search
        else ""
    )
    return [
        '  <section id="search-funnel" class="card">',
        f"    <h2>{funnel_heading}</h2>",
        '    <div class="funnel-container">',
        '      <div class="funnel-step">',
        '        <div class="stat-label">Core-Valid Grid</div>',
        f'        <div class="stat-value">{core_grid_count:,}</div>',
        '        <div class="funnel-sub">Core-partitioned meshes &times; '
        "concurrencies</div>",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">Physically Legal</div>',
        f'        <div class="stat-value">{enumerated_count:,}</div>',
        '        <div class="funnel-sub">Head, expert, layer &amp; '
        "core-product divisibility</div>",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">HBM Fit</div>',
        f'        <div class="stat-value">{hbm_fit_count:,}</div>',
        f"        {hbm_sub_html}",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        f'      <div class="{slo_step_class}">',
        f'        <div class="stat-label">{slo_label}</div>',
        f'        <div class="stat-value">{slo_stage_count:,}</div>',
        f"        {slo_breakdown_html}",
        envelope_feasible_html,
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step" style="border-color:var(--primary); '
        'background:var(--primary-light);">',
        '        <div class="stat-label" style="color:var(--primary);">'
        "Est. 2D Frontier</div>",
        '        <div class="stat-value" style="color:var(--primary);">'
        f"{frontier_2d_count:,}</div>",
        f'        <div class="funnel-sub" style="color:var(--primary);">'
        f"{top_k_len} shortlisted layouts</div>",
        "      </div>",
        "    </div>",
        "  </section>",
    ]


def render_html(payload: Mapping[str, object]) -> str:
    """Render a fully self-contained, offline HTML report from a solve/v2 payload."""
    recommended = _as_mapping(payload.get("recommended")) or None
    frontier = [_as_mapping(item) for item in _as_sequence(payload.get("frontier"))]
    candidates = [_as_mapping(item) for item in _as_sequence(payload.get("candidates"))]
    scenario = _as_mapping(payload.get("scenario"))
    accept_len = max(1.0, _as_float(scenario.get("accept_len"), 1.0))
    warnings = [str(warning) for warning in _as_sequence(payload.get("warnings"))]
    top_k = [_as_mapping(item) for item in _as_sequence(payload.get("top_k"))]
    model = str(payload.get("model") or "Model")
    chip = str(payload.get("chip") or "TPU")
    report_chip = _payload_chip_spec(payload)
    chips_per_host = (
        report_chip.chips_per_host
        if report_chip is not None
        else CHIPS["tpu7x"].chips_per_host
    )
    has_moe = _detect_has_moe(payload, candidates)

    embedded_payload = dict(payload)
    embedded_payload["candidates"] = _downsample_embedded_candidates(candidates)
    embedded_json = json.dumps(
        embedded_payload, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")

    search_map = _as_mapping(payload.get("search"))
    is_hbm_oom = (
        recommended is None
        and bool(search_map)
        and _as_int(search_map.get("hbm_fit"), -1) == 0
    )
    infeasible_msg = (
        "No candidate layout fit HBM memory constraints; "
        "recommended layout is infeasible."
        if is_hbm_oom
        else "No candidate layout met latency SLOs; recommended layout is infeasible."
    )
    top_verdict = recommended.get("slo_verdict") if recommended is not None else None
    rec_infeasible = recommended is None or (
        top_verdict == SLO_VERDICT_RULED_OUT
        if top_verdict is not None
        else not bool(recommended.get("slo_met", True))
    )
    if rec_infeasible and infeasible_msg not in warnings:
        warnings.append(infeasible_msg)

    rec_fields, pools_list, pools_summary = _extract_recommended_view(
        recommended, has_moe=has_moe, chip=report_chip, accept_len=accept_len
    )
    decode_floor_raw = rec_fields["decode_floor"]
    decode_floor = _as_mapping(decode_floor_raw) if decode_floor_raw else None
    default_limit_gb = _derive_default_hbm_limit_gb(payload)

    ref_report = _as_mapping(payload.get("reference"))
    frontier_svg = _render_frontier_svg(
        candidates,
        frontier,
        top_k=top_k,
        accept_len=accept_len,
        reference=ref_report,
    )
    limits_svg = _render_limits_svg(decode_floor)
    memory_svg = _render_memory_svg(pools_list, default_limit_gb)

    warnings_html = _render_warnings_html(warnings)
    table_is_disagg = _has_disagg_entries(_table_entries(top_k, ref_report))
    pool_header_html = (
        "          <th>Prefill Pool</th>\n          <th>Decode Pool</th>"
        if table_is_disagg
        else "          <th>Pool</th>"
    )
    top_table_body = _render_table_rows(
        top_k,
        reference=ref_report,
        is_disagg=table_is_disagg,
        has_moe=has_moe,
        chip=report_chip,
        accept_len=accept_len,
    )
    crossover_stat_html = _render_crossover_stat_html(payload)

    scen_name = _escape(scenario.get("name", "custom"))
    scen_isl = _as_int(scenario.get("isl"))
    scen_osl = _as_int(scenario.get("osl"))
    scen_hit = _as_float(scenario.get("prefix_hit_rate"))
    meta_suffix = _render_header_meta_suffix(payload, chips_per_host=chips_per_host)
    sh_fit_html = _render_single_host_fit_card(payload, has_moe=has_moe)
    rec_chips_int = _as_int(rec_fields["chips"])
    if rec_chips_int > 0:
        rec_hosts_str = format_host_count(
            count_hosts(rec_chips_int, chips_per_host=chips_per_host)
        )
        rec_summary_meta = (
            f"Concurrency {rec_fields['concurrency']} across "
            f"{_format_chip_count(rec_chips_int)} ({rec_hosts_str})"
        )
    else:
        rec_summary_meta = "N/A"
    moe_pool_note = (
        "MoE pools show Attention axes (<code>Attn: ...</code>) and "
        "Expert axes (<code>MoE: EP...</code> whole-expert sharding vs. "
        "<code>MoE: TP...</code> width-sliced experts) across the same "
        "TensorCores. "
        if has_moe
        else ""
    )

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>',
        f"<title>Roofline Report - {_escape(model)} on {_escape(chip)}</title>",
        f"<style>{CSS_CONTENT}</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        "  <header>",
        "    <h1>TPU Roofline Solve Report</h1>",
        '    <div class="meta">',
        f'      <span class="badge">{_escape(chip)}</span>',
        f"      <strong>Model:</strong> {_escape(model)} &bull; ",
        f"      <strong>Scenario:</strong> {scen_name} "
        f"(ISL={scen_isl:,}, OSL={scen_osl:,}, hit_rate={scen_hit:.2f}){meta_suffix}",
        "    </div>",
        "  </header>",
        f"  {warnings_html}",
        '  <section id="executive-summary" class="card">',
        "    <h2>Top-Ranked Analytical Candidate (Uncalibrated Shortlist)</h2>",
        '    <div style="font-size:14px;">',
        '      <span class="badge" style="background:#dbeafe; color:#1e40af; '
        f'font-size:12.5px; padding:3px 10px;">{_escape(rec_fields["mode"])}</span>',
        f"      <strong>{_escape(pools_summary)}</strong> &bull; {rec_summary_meta}",
        "    </div>",
        '    <div class="grid-stats">',
        '      <div class="stat-box">',
        '        <div class="stat-label">Est. Throughput / Chip</div>',
        f'        <div class="stat-value">{rec_fields["tput"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        f"tok/s/chip (out: {rec_fields['out_tput']})</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">Est. Interactivity</div>',
        f'        <div class="stat-value">{rec_fields["interactivity"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "tok/s</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">TTFT (bound ~ est)</div>',
        f'        <div class="stat-value">{rec_fields["ttft"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">TPOT (bound ~ est)</div>',
        f'        <div class="stat-value">{rec_fields["tpot"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">Binding Unit</div>',
        '        <div class="stat-value" style="font-size:13.5px; '
        'overflow-wrap:anywhere;">'
        f"{_escape(rec_fields['binding'])}</div>",
        "      </div>",
        crossover_stat_html,
        "    </div>",
        "  </section>",
        sh_fit_html,
        *_render_funnel_section(payload, frontier, top_k_len=len(top_k)),
        '  <section id="frontier-chart" class="card">',
        (
            "    <h2>Estimated Pareto Frontier "
            "(Throughput per Chip vs. Interactivity)</h2>"
        ),
        (
            '    <p class="section-note" '
            'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
            "Both axes are model estimates (<code>est_*</code>), not measured "
            "TPU numbers; latency bounds (<code>bound_*</code>) are hard "
            "physical lower limits.</p>"
        ),
        f'    <div class="chart-container">{frontier_svg}</div>',
        (
            '    <div class="chart-legend">'
            "Blue curve: estimated 2D Pareto frontier &middot; "
            "Green ring: 2D-frontier shortlist &middot; "
            "Dashed amber ring: alternative shortlist / reference layout</div>"
        ),
        "  </section>",
        '  <section id="limits-chart" class="card">',
        "    <h2>Decode Step Resource Times</h2>",
        (
            '    <p class="section-note" '
            'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
            "Uncalibrated analytical per-resource decode step lower bounds "
            "(<code>bound_*</code>) before kernel overlap or scheduling "
            "overhead.</p>"
        ),
        f'    <div class="chart-container">{limits_svg}</div>',
        "  </section>",
        '  <section id="memory-chart" class="card">',
        "    <h2>HBM Memory Footprint vs. Capacity</h2>",
        f'    <div class="chart-container">{memory_svg}</div>',
        "  </section>",
        '  <section id="top-layouts-table" class="card">',
        "    <h2>Shortlisted Layouts (Analytical Candidates to Measure)</h2>",
        (
            '    <p class="section-note" '
            'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
            f"{moe_pool_note}"
            "Columns marked <code>[bound]</code> are physical "
            "lower limits; <code>[est]</code> are uncalibrated analytical model "
            "estimates.</p>"
        ),
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Rank</th>",
        "          <th>Role</th>",
        "          <th>Mode</th>",
        "          <th>Concurrency</th>",
        "          <th>Chips</th>",
        pool_header_html,
        "          <th>Int (tok/s) [est]</th>",
        "          <th>Tput (tok/s/chip) [est]</th>",
        "          <th>TPOT Bound (ms) [bound]</th>",
        "          <th>TTFT (ms) [est]</th>",
        "          <th>TPOT (ms) [est]</th>",
        "          <th>KV Cap / Prefix Hit Rate</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>",
        f"        {top_table_body}",
        "      </tbody>",
        "    </table>",
        *(
            (
                f'    <p class="section-note" '
                f'style="color:var(--text-muted);font-size:13px;margin-top:8px;">'
                f"# reference layout {_escape(str(ref_report.get('layout', '')))}: "
                f"not enumerated "
                f"({_escape(str(ref_report.get('rejected_by', '')))})</p>",
            )
            if (ref_report and not bool(ref_report.get("found")))
            else ()
        ),
        (
            '    <details class="glossary"><summary>'
            "Glossary of Physical Units, Submodule Axes &amp; Roles</summary>"
            "<ul>"
            + _render_glossary_unit_item(payload, report_chip)
            + "<li><strong>Attention vs. MoE Submodule Axes:</strong> "
            "<code>Attn: TP / DP / PCP / DCP</code> partitions attention "
            "heads, request replicas, prefill sequence chunks, and KV cache; "
            "<code>MoE: EP</code> shards whole routed experts across those "
            "same cores, while "
            "<code>MoE: TP</code> slices every expert by width "
            "across each TP group. "
            "<code>PP</code> splits whole transformer layers across stages.</li>"
            "<li><strong>Roles &amp; Callouts:</strong> "
            "<code>c=&lt;N&gt;</code> denotes concurrency "
            "(active in-flight requests); <code>max_throughput</code> "
            "(highest est. throughput/chip), <code>balanced_knee</code> "
            "(interior 2D Pareto knee), <code>max_interactivity</code> "
            "(fastest est. interactivity), "
            "<code>off_frontier_alt</code> "
            "(dominated 2D alternative included in shortlist), "
            "<code>reference</code> "
            "(user-supplied <code>--reference-layout</code>).</li>"
            "</ul></details>"
        ),
        "  </section>",
        *_render_pool_tables_section(payload, has_moe=has_moe, chip=report_chip),
        *_render_ordered_concurrency_section(
            payload, has_moe=has_moe, chip=report_chip, accept_len=accept_len
        ),
        *_render_bound_families_section(payload, has_moe=has_moe, chip=report_chip),
        *_render_assumptions_card(payload),
        "  <footer>",
        "    Analytical roofline report generated by "
        "<code>vllm_torchtpu.roofline</code> &bull; "
        "Bounds (bound_*) are physical lower limits; "
        "estimates (est_*) are uncalibrated analytical model estimates",
        "  </footer>",
        "</div>",
        '<script type="application/json" id="roofline-data">',
        f"{embedded_json}",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)


def write_html(payload: Mapping[str, object], path: Path | str) -> Path:
    """Render and write HTML report to disk at path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_html(payload)
    dest.write_text(rendered, encoding="utf-8")
    return dest
