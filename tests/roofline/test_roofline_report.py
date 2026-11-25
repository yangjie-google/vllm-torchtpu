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
"""CPU unit tests for physics-only roofline HTML report generation."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.roofline.conftest import LOCKED_FIXTURE_FILES, load_locked_bundle
from vllm_torchtpu.roofline import (
    CHIPS,
    SLO_VERDICT_RULED_OUT,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    cli,
    render_html,
    solve_serving_plan,
    write_html,
)
from vllm_torchtpu.roofline.constants import (
    DEFAULT_TOP_K,
    MAX_EMBEDDED_CANDIDATES,
    MXFP4_BYTES,
    UNKNOWN_THINK_TIME_WARNING,
)

if TYPE_CHECKING:
    from vllm_torchtpu.roofline import SolvePayload


@pytest.fixture(scope="module")
def sample_payload() -> SolvePayload:
    """Return a valid physics solve payload derived from a locked basis fixture."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(1, 2),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="auto", max_chips=4),
    )
    return build_solve_payload(
        plan,
        chip=chip,
        dtypes=bundle.dtypes,
        model=bundle.identity.hf_repo,
        top_k=3,
    )


def test_report_html_is_deterministic(sample_payload: SolvePayload) -> None:
    """render_html produces byte-identical output across consecutive calls."""
    first_html = render_html(sample_payload)
    second_html = render_html(sample_payload)
    min_html_len = 500
    assert first_html == second_html
    assert len(first_html) > min_html_len


def test_report_has_no_external_fetch(sample_payload: SolvePayload) -> None:
    """HTML report is fully offline: zero external stylesheets, scripts, or fetches."""
    html = render_html(sample_payload)
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html, re.IGNORECASE)
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", html, re.IGNORECASE)
    assert not re.search(r"""url\(\s*["']?https?://""", html, re.IGNORECASE)
    assert "@import" not in html
    found_urls = re.findall(r"http://[a-zA-Z0-9./_-]+", html)
    assert set(found_urls).issubset({"http://www.w3.org/2000/svg"})


def test_report_escapes_untrusted_strings(sample_payload: SolvePayload) -> None:
    """Model names and warning text with HTML or script tags are properly escaped."""
    payload: dict[str, object] = dict(sample_payload)
    payload["model"] = '<script>alert("xss")</script>'
    payload["warnings"] = ["<b>Unsafe warning</b>", "Normal warning"]
    html = render_html(payload)
    assert '<script>alert("xss")</script>' not in html
    assert (
        "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html
        or "&lt;script&gt;" in html
    )
    assert "<b>Unsafe warning</b>" not in html
    assert "&lt;b&gt;Unsafe warning&lt;/b&gt;" in html
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    json_blob = match.group(1)
    assert "</" not in json_blob or r"<\/" in json_blob


def test_report_sections_present_and_zero_launch_boxes(
    sample_payload: SolvePayload,
) -> None:
    """Required physics sections and pool tables exist with zero launch markup."""
    html = render_html(sample_payload)
    assert 'id="executive-summary"' in html
    assert 'id="search-funnel"' in html
    assert 'id="frontier-chart"' in html
    assert 'id="limits-chart"' in html
    assert 'id="memory-chart"' in html
    assert 'id="top-layouts-table"' in html
    assert 'id="pool-tables"' in html
    assert 'id="roofline-data"' in html
    assert "<svg" in html
    assert "copyLaunch" not in html
    assert "vllm serve" not in html
    assert 'details class="launch"' not in html


def test_report_without_feasible_layout(sample_payload: SolvePayload) -> None:
    """When no layout is feasible, renders warning banner and empty state gracefully."""
    payload: dict[str, object] = dict(sample_payload)
    payload["recommended"] = None
    payload["top_k"] = []
    payload["warnings"] = ["SLO_UNMET: no feasible topology met latency constraints"]
    html = render_html(payload)
    assert "SLO_UNMET" in html
    assert 'xmlns="http://www.w3.org/2000/svg"' in html

    # Distinguish HBM-OOM (hbm_fit == 0) from SLO infeasibility.
    oom_payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    oom_payload["recommended"] = None
    oom_payload["top_k"] = []
    oom_payload["warnings"] = []
    oom_payload["search"] = {
        "enumerated": 10,
        "hbm_fit": 0,
        "slo_met": 0,
        "frontier": 0,
    }
    oom_html = render_html(oom_payload)
    assert "HBM memory constraints" in oom_html
    assert "recommended layout is infeasible" in oom_html


def test_report_caps_embedded_candidates(sample_payload: SolvePayload) -> None:
    """Payloads with large candidate counts downsample embedded JSON strictly."""
    payload: dict[str, object] = dict(sample_payload)
    cands = sample_payload["candidates"]
    base_cand = cands[0] if cands else {}
    payload["candidates"] = [dict(base_cand) for _ in range(3000)]
    html = render_html(payload)
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    raw_json = match.group(1).replace(r"<\/", "</")

    def _reject_constant(val: str) -> object:
        msg = f"Unexpected non-finite JSON constant: {val}"
        raise AssertionError(msg)

    embedded = json.loads(raw_json, parse_constant=_reject_constant)
    assert len(embedded["candidates"]) <= MAX_EMBEDDED_CANDIDATES


def test_report_slo_unmet_recommended_renders_warning_banner(
    sample_payload: SolvePayload,
) -> None:
    """Verify warning banner rendering when recommended layout does not meet SLO."""
    html_ok = render_html(sample_payload)
    assert "max_throughput" in html_ok
    assert "<th>Role</th>" in html_ok
    assert "<th>KV Cap / Prefix Hit Rate</th>" in html_ok

    payload: dict[str, object] = dict(sample_payload)
    assert sample_payload["recommended"] is not None
    rec_unmet = dict(sample_payload["recommended"])
    rec_unmet["slo_met"] = False
    payload["recommended"] = rec_unmet
    payload["top_k"] = []
    payload["warnings"] = []
    html_unmet = render_html(payload)
    assert "INFEASIBLE" in html_unmet
    assert '<div class="alert-box alert-warning">' in html_unmet
    assert "recommended layout is infeasible" in html_unmet


def test_render_html_clean_svg_full_hbm_and_dynamic_chip_spec(
    sample_payload: SolvePayload,
) -> None:
    """Verify clean SVG formatting, full physical HBM spec, and dynamic chip_spec."""
    payload = copy.deepcopy(dict(sample_payload))
    rec = payload.get("recommended")
    if isinstance(rec, dict) and isinstance(rec.get("decode_pool"), dict):
        rec["decode_pool"].pop("hbm_limit_gb", None)
    html = render_html(payload)
    assert "</text>'" not in html
    assert "103.00 GB" in html
    assert "94.76" not in html
    assert "Assumptions, Hardware Specs" in html

    custom_chip = dc_replace(
        CHIPS["tpu7x"],
        name="custom_tpu_v6",
        hbm_bytes=32_000_000_000,
        chips_per_host=1,
        fp4_flops=4.6e15,
        int8_flops=2.3e15,
    )
    custom_payload = copy.deepcopy(dict(sample_payload))
    custom_payload["chip"] = "custom_tpu_v6"
    custom_payload["chip_spec"] = {
        "name": custom_chip.name,
        "hbm_bytes": custom_chip.hbm_bytes,
        "hbm_bw": custom_chip.hbm_bw,
        "bf16_flops": custom_chip.bf16_flops,
        "fp8_flops": custom_chip.fp8_flops,
        "fp4_flops": custom_chip.fp4_flops,
        "int8_flops": custom_chip.int8_flops,
        "vector_flops": custom_chip.vector_flops,
        "chips_per_host": 1,
        "vmem_bytes": custom_chip.vmem_bytes,
        "ici_link_bw": custom_chip.ici_link_bw,
        "ici_bw": custom_chip.ici_bw,
        "ici_hop_latency_s": custom_chip.ici_hop_latency_s,
    }
    custom_html = render_html(custom_payload)
    assert "custom_tpu_v6 Per-Core Specs:" in custom_html
    assert "32.0 GB" in custom_html
    assert "FP4" in custom_html
    assert "INT8" in custom_html
    assert "Vector" in custom_html
    assert "1 chip / 1 host" in custom_html


def test_report_uses_explicit_concurrency_hit_rate_and_chip_headers(
    sample_payload: SolvePayload,
) -> None:
    """Verify HTML report spells out Concurrency, Prefix Hit Rate, and chips headers."""
    html = render_html(sample_payload)
    assert "Concurrency" in html
    assert "Prefix Hit Rate" in html
    assert "chips" in html
    assert not re.search(r"<th[^>]*>\s*c\s*</th>", html, re.IGNORECASE)
    assert not re.search(r"<th[^>]*>\s*h\s*</th>", html, re.IGNORECASE)
    assert not re.search(r"\d+c / \d+h", html)


def test_frontier_chart_plots_monotonic_2d_envelope_with_floor_tooltips(
    sample_payload: SolvePayload,
) -> None:
    """Frontier plots throughput on y, interactivity on x, and per-candidate floors."""
    payload = copy.deepcopy(sample_payload)
    p_low = dict(payload["frontier"][0])
    p_low["concurrency"] = 1
    p_low["est_interactivity_tok_s"] = 1000.0
    p_low["token_throughput_per_chip"] = 2500.0
    p_high = dict(payload["frontier"][0])
    p_high["concurrency"] = 4
    p_high["est_interactivity_tok_s"] = 1000.0
    p_high["token_throughput_per_chip"] = 10000.0
    p_tput = dict(payload["frontier"][0])
    p_tput["concurrency"] = 64
    p_tput["est_interactivity_tok_s"] = 200.0
    p_tput["token_throughput_per_chip"] = 40000.0
    payload["frontier"] = (p_low, p_high, p_tput)
    payload["candidates"] = (p_low, p_high, p_tput)

    html = render_html(payload)
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "Est. Interactivity (tok/s)</text>" in chart
    assert "Est. Token Throughput per Chip (tok/s/chip)</text>" in chart
    assert "TPOT floor" in chart
    match = re.search(r'<polyline points="([^"]+)"', chart)
    assert match is not None
    coords = [
        tuple(float(v) for v in pair.split(",")) for pair in match.group(1).split()
    ]
    expected_envelope_len = 2
    assert len(coords) == expected_envelope_len
    assert coords[0][0] < coords[1][0]
    assert coords[0][1] < coords[1][1]


def test_report_renders_five_stage_physics_funnel_and_submodule_pools(
    sample_payload: SolvePayload,
) -> None:
    """HTML report renders 5-stage physics funnel, Attn | MoE pools, no runtime text."""
    moe_payload = copy.deepcopy(sample_payload)
    moe_payload["has_moe"] = True
    assert moe_payload["search"] is not None
    moe_search = dict(moe_payload["search"])
    moe_search["core_grid"] = 2940
    moe_search["enumerated"] = 98
    moe_search["hbm_fit"] = 98
    moe_search["not_ruled_out"] = 98
    moe_payload["search"] = moe_search
    moe_payload["budget"] = 4
    moe_payload["chip_budget"] = {"max_chips": 4, "budget_source": "flag"}
    rank1_point = dict(moe_payload["top_k"][0]["point"])
    rank1_point["slo_verdict"] = "estimate_meets"
    rank1_point["concurrency"] = 64
    undecided_point = dict(rank1_point)
    undecided_point["slo_verdict"] = "undecided"
    rank1_tput = rank1_point["token_throughput_per_chip"]
    assert isinstance(rank1_tput, (int, float))
    undecided_point["token_throughput_per_chip"] = float(rank1_tput) * 1.15
    moe_payload["top_k"][0]["point"] = rank1_point
    moe_payload["top_k"][0]["frontier_role"] = "max_throughput"
    moe_payload["frontier"] = (rank1_point, undecided_point)
    moe_payload["candidates"] = (rank1_point, undecided_point)

    html = render_html(moe_payload)
    funnel = html[html.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    assert "Core-Valid Grid" in funnel
    assert "2,940" in funnel
    assert "Physically Legal" in funnel
    assert "Architecture &amp; Runtime Legal" not in funnel
    assert "--max-num-seqs" not in funnel
    assert "Attn:" in html
    assert "MoE:" in html
    assert "1 hosts" not in html
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "#1 (c=64)" in chart


def test_report_honest_analytical_headings_and_excludes_ruled_out_frontier(
    sample_payload: SolvePayload,
) -> None:
    """Verify honest analytical headings and ruled-out exclusion on 2D frontier."""
    html_base = render_html(sample_payload)
    assert "Top-Ranked Analytical Candidate (Uncalibrated Shortlist)" in html_base
    assert "Uncalibrated analytical per-resource decode step lower bounds" in html_base
    assert "Executive Recommendation" not in html_base
    assert "Recommended Launch Command" not in html_base
    assert "Best Parallelism:" not in html_base
    assert 'id="bound-families-table"' not in html_base

    payload = copy.deepcopy(sample_payload)
    ruled_out_pts = []
    for point in payload["frontier"]:
        copy_pt = dict(point)
        copy_pt["slo_verdict"] = "ruled_out"
        copy_pt["slo_met"] = False
        ruled_out_pts.append(copy_pt)
    payload["frontier"] = tuple(ruled_out_pts)
    html_ruled = render_html(payload)
    funnel = html_ruled[html_ruled.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    assert (
        '<div class="stat-label" style="color:var(--primary);">'
        'Est. 2D Frontier</div><div class="stat-value" '
        'style="color:var(--primary);">0</div>'
    ) in funnel.replace("\n        ", "")
    chart = html_ruled[html_ruled.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "<polyline" not in chart


def test_report_renders_optional_reserve_provenance_assumptions_and_families(
    sample_payload: SolvePayload,
) -> None:
    """Verify guarded rendering when HBM reserve, provenance, and families exist."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    reserve_gb = 12.0
    expected_usable = 103.0 - reserve_gb
    payload["hbm_reserve_gb"] = reserve_gb
    payload["single_host_fit"] = {
        "feasible": True,
        "status": "FEASIBLE",
        "reason": "Fits within reserved HBM",
        "hbm_used_gb": 50.0,
        "hbm_limit_gb": expected_usable,
        "best_parallelism": "TP8",
    }
    payload["provenance"] = {
        "calibrated": False,
        "basis": "Analytical bounds and uncalibrated serial estimates",
        "ranking_basis": "bound_ordered_handoff",
        "supplied_values": {
            "hbm_reserve_gb": {
                "flag": "--hbm-reserve-gb",
                "value": reserve_gb,
                "kind": "assumed",
            },
        },
    }
    payload["solver_assumptions"] = [
        {"name": "moe_activation", "value": "dense_fallback", "kind": "derived"},
    ]
    sample_point = copy.deepcopy(sample_payload["frontier"][0])
    payload["bound_families_by_concurrency"] = {
        "16": [
            {
                "rank": 1,
                "family": ("mixed", "2x2x1", "TP8"),
                "min_bound_tpot_ms": 1.25,
                "min_bound_ttft_ms": 42.0,
                "min_envelope_tpot_ms": 1.45,
                "min_envelope_ttft_ms": 48.5,
                "binding": "hbm",
                "any_not_ruled_out": True,
                "member_count": 2,
                "members": [sample_point],
            }
        ]
    }
    html = render_html(payload)
    assert f"{expected_usable:.2f} GB" in html
    assert f"minus <code>{reserve_gb:.2f} GB</code> HBM reserve" in html
    assert "HBM Workspace Reserve:" in html
    assert "Top-Ranked Single-Host Candidate:" in html
    assert "--hbm-reserve-gb=12.0" in html
    assert "bound_ordered_handoff" in html
    assert "Solver Assumptions:" in html
    assert "moe_activation=dense_fallback" in html
    assert "Bound-Ordered Candidate Families (Benchmark Handoff)" in html
    assert "mixed | 2x2x1 | TP8" in html
    assert "1.25" in html


def test_report_single_host_fit_renders_na_when_best_parallelism_is_none(
    sample_payload: SolvePayload,
) -> None:
    """Render N/A rather than None when single-host fit has no valid layout."""
    payload = copy.deepcopy(sample_payload)
    payload["single_host_fit"] = {
        "feasible": False,
        "status": "REJECTED",
        "reason": "Exceeds single-host HBM",
        "hbm_used_gb": 120.0,
        "hbm_limit_gb": 103.0,
        "best_parallelism": None,
    }
    html = render_html(payload)
    assert "Top-Ranked Single-Host Candidate: <code>N/A</code>" in html
    assert "<code>None</code>" not in html


def test_report_renders_infeasible_banner_with_existing_warnings(
    sample_payload: SolvePayload,
) -> None:
    """Infeasible banner renders alongside pre-existing informational warnings."""
    payload_ruled: dict[str, object] = copy.deepcopy(dict(sample_payload))
    raw_rec = payload_ruled["recommended"]
    assert isinstance(raw_rec, dict)
    record = dict(raw_rec)
    record["slo_verdict"] = SLO_VERDICT_RULED_OUT
    record["slo_met"] = False
    payload_ruled["recommended"] = record
    payload_ruled["warnings"] = [UNKNOWN_THINK_TIME_WARNING]

    html_ruled = render_html(payload_ruled)
    assert UNKNOWN_THINK_TIME_WARNING in html_ruled
    assert "recommended layout is infeasible" in html_ruled

    payload_none: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload_none["recommended"] = None
    payload_none["top_k"] = []
    payload_none["warnings"] = [UNKNOWN_THINK_TIME_WARNING]

    html_none = render_html(payload_none)
    assert UNKNOWN_THINK_TIME_WARNING in html_none
    assert "recommended layout is infeasible" in html_none


def test_report_renders_na_instead_of_zero_chips_one_host(
    sample_payload: SolvePayload,
) -> None:
    """Empty recommended layout renders N/A instead of 0 chips (1 host)."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["recommended"] = None
    payload["top_k"] = []
    html = render_html(payload)
    assert "0 chips (1 host)" not in html
    assert "Concurrency 0 across" not in html
    assert "&bull; N/A" in html


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("accept_len", (1.0, 3.0))
def test_report_bounds_limits_bars_and_accept_len_across_locked_fixtures(
    fixture_name: str,
    accept_len: float,
) -> None:
    """Verify positive TPOT bounds <= est TPOT and valid limits bars on all models."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="report-bounds-check",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=accept_len,
        think_time_s=0.0,
        concurrencies=(1, 8),
        kv_working_set_turns=1.0,
        spec_tokens=3 if accept_len > 1.0 else None,
    )
    plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    payload = build_solve_payload(
        plan,
        chip=chip,
        dtypes=bundle.dtypes,
        model=bundle.identity.hf_repo,
        top_k=None,
    )
    html = render_html(payload)
    table_sec = html[html.index('id="top-layouts-table"') :]
    table_sec = table_sec[: table_sec.index("</section>")]
    row_matches = re.findall(r"<tbody>(.*?)</tbody>", table_sec, re.DOTALL)
    assert row_matches
    tr_blocks = re.findall(r"<tr>(.*?)</tr>", row_matches[0], re.DOTALL)
    assert tr_blocks
    for tr_html in tr_blocks:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr_html, re.DOTALL)
        tpot_bound = float(cells[8].replace(",", ""))
        tpot_est = float(cells[10].replace(",", ""))
        assert 0.0 < tpot_bound <= tpot_est + 1e-6

    limits_sec = html[html.index('id="limits-chart"') :]
    limits_sec = limits_sec[: limits_sec.index("</section>")]
    bar_vals = {
        label: (float(w_str), float(ms_str))
        for w_str, label, ms_str in re.findall(
            r'width="([0-9.]+)"[^>]*><title>([^:]+): ([0-9.]+) ms</title>',
            limits_sec,
        )
    }
    assert "Hard Floor" in bar_vals
    hard_ms = bar_vals["Hard Floor"][1]
    for bound_name in ("HBM Bound", "MatMul Bound", "Vector Bound", "Collective Bound"):
        width_px, val_ms = bar_vals[bound_name]
        assert val_ms <= hard_ms + 1e-3
        if val_ms == 0.0:
            assert width_px == 0.0

    # Under top_k=None, callout rings stay capped at DEFAULT_TOP_K.
    frontier_sec = html[html.index('id="frontier-chart"') :]
    frontier_sec = frontier_sec[: frontier_sec.index("</section>")]
    ring_count = len(re.findall(r'<circle [^>]*r="10" fill="none"', frontier_sec))
    assert ring_count <= DEFAULT_TOP_K


def test_report_pins_disagg_edge_guards_and_cli_html_output(
    tmp_path: Path,
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify P:D pin display, dtype formatting, edge guards, write_html, and CLI."""
    cfg_path = locked_config_path("qwen3_5_397b_nvfp4.json")
    out_html = tmp_path / "out_report.html"
    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--concurrency",
            "4",
            "--mode",
            "disagg",
            "--prefill-topology",
            "2x2x1",
            "--decode-topology",
            "2x2x1",
            "--pd",
            "1:1",
            "--html",
            str(out_html),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert out_html.is_file()
    html_text = out_html.read_text(encoding="utf-8")
    assert "P:D=1:1" in html_text
    assert "<th>Prefill Pool</th>" in html_text
    assert "<th>Decode Pool</th>" in html_text
    assert "act=a4" in html_text

    # Test MXFP4 6-sig-fig dtype display, None hit rate, dense MoE note omission,
    # and HBM limit <= 0 / overflow bar clamping via write_html.
    mutated = copy.deepcopy(payload)
    mutated["has_moe"] = False
    mutated["dtypes"]["w_expert"] = MXFP4_BYTES
    mutated["dtypes"]["w_expert_dtype"] = "mxfp4"
    mutated["top_k"][0]["point"]["effective_prefix_hit_rate"] = None
    mutated["recommended"]["decode_pool"]["hbm_limit_gb"] = 0.0
    mutated["recommended"]["decode_pool"]["hbm_used_gb"] = 150.0
    mutated["hbm_reserve_gb"] = 1000.0
    dest_path = write_html(mutated, tmp_path / "edge.html")
    edge_html = dest_path.read_text(encoding="utf-8")
    assert f"{MXFP4_BYTES:.6g}B" in edge_html
    assert " / -</td>" in edge_html
    assert "MoE pools show Attention axes" not in edge_html
