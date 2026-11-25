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
"""Unit tests for megakernel advisor regime sweeps, plain-text formatting, and CLI."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest

from tests.megakernel.test_advisor_bounds import iter_case_grid
from tests.megakernel.test_advisor_specs import (
    HF_CONFIG_DIR,
    load_qwen35_snapshot_config,
)
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    cli,
    constants,
    models,
    specs,
    text_format,
)
from vllm_torchtpu.megakernel.advisor.chips import resolve_chip_spec
from vllm_torchtpu.megakernel.advisor.regimes import (
    _build_reference_workloads,
    _compute_disagg_handoff,
    _with_preset_prefill,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError

KERNEL_NAME_DENYLIST = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu|tensorcore|vpu|\bllo\b", re.IGNORECASE
)
TEST_ROW_LIMIT = 80
EXPECTED_DEFAULT_REGIMES = 12


def _regime(
    profile_or_slug: advisor.ModelProfile | str,
    prefix: str,
    options: advisor.RegimeSweepOptions | None = None,
) -> advisor.RegimeEntry:
    """Return the regime row whose label starts with `prefix`."""
    sweep = advisor.advise_regimes(profile_or_slug, "tpu7x", options=options)
    return next(entry for entry in sweep.regimes if entry.label.startswith(prefix))


def test_bare_minimum_auto_parallelism_and_regime_table() -> None:
    """Verify auto-parallelism 12-regime sweep, state_dtype core gate, and table."""
    sweep_fp32 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(deployment_preset=False),
    )
    assert sweep_fp32.parallelism.auto_selected is True
    assert (sweep_fp32.parallelism.tp, sweep_fp32.parallelism.ep) == (8, 8)
    assert "2x2x1" in sweep_fp32.parallelism.topology
    assert len(sweep_fp32.regimes) == EXPECTED_DEFAULT_REGIMES
    fp32_by_label = {entry.label: entry for entry in sweep_fp32.regimes}
    fp32_small = fp32_by_label["Decode Small (B=4, T=1, 4K)"]
    assert fp32_small.recommended_scope == "per_op"
    assert fp32_small.fastest_scope == "whole_step"
    assert fp32_small.evidence == "baseline"
    fp32_whole_scope = next(
        sc for sc in fp32_small.scopes if sc["scope"] == "whole_step"
    )
    assert fp32_whole_scope["support"] == "outside_core_limits"
    whole_blockers = cast("list[str] | tuple[str, ...]", fp32_whole_scope["blockers"])
    assert "kv_or_state_dtype_not_in_core" in whole_blockers

    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    sweep_bf16 = advisor.advise_regimes(
        qwen_bf16,
        "tpu7x",
        options=advisor.RegimeSweepOptions(deployment_preset=False),
    )
    bf16_by_label = {entry.label: entry for entry in sweep_bf16.regimes}
    assert (
        bf16_by_label["Decode Small (B=4, T=1, 4K)"].recommended_scope == "whole_step"
    )
    assert (
        bf16_by_label["Decode Small (B=4, T=1, 4K)"].recommended_moe_mode
        == "masked_on_chip"
    )
    assert bf16_by_label["Decode High (B=64, T=1, 32K)"].fastest_scope == "whole_step"
    assert (
        bf16_by_label["Decode High (B=64, T=1, 32K)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert bf16_by_label["Chunked Prefill (B=1, T=256)"].fastest_scope == "whole_step"
    assert (
        bf16_by_label["Chunked Prefill (B=1, T=256)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert bf16_by_label["Long Prefill (B=1, T=4096)"].recommended_scope == "per_op"
    assert (
        bf16_by_label["Long Prefill (B=1, T=4096)"].recommended_moe_mode
        == "grouped_in_hbm"
    )

    sweep_dcp = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dcp=4, pcp=2),
    )
    assert (sweep_dcp.parallelism.dcp, sweep_dcp.parallelism.pcp) == (4, 2)
    dcp_by_label = {entry.label: entry for entry in sweep_dcp.regimes}
    assert (
        dcp_by_label["Long Prefill (B=1, T=4096)"].active_tokens
        == fp32_by_label["Long Prefill (B=1, T=4096)"].active_tokens // 2
    )

    sweep_k3 = advisor.advise_regimes("kimi-k3", "tpu7x")
    assert (sweep_k3.parallelism.tp, sweep_k3.parallelism.ep) == (32, 32)
    assert "2x2x4" in sweep_k3.parallelism.topology
    assert sweep_k3.regimes[0].fastest_scope == "whole_step"
    assert sweep_k3.regimes[0].recommended_scope == "per_op"
    assert sweep_k3.regimes[-1].recommended_scope == "per_op"

    table_text = advisor.format_regime_table(sweep_bf16)
    assert "Decode Small (B=4, T=1, 4K)" in table_text
    assert "Scope (where the kernel boundary falls):" in table_text
    assert "Tailoring Guide:" in table_text
    assert "Derated Achievable" in table_text


@pytest.mark.parametrize(
    "bad_extra",
    [
        ["--prefix-hit-rate", "1.0"],
        ["--ep", "0"],
        ["--kv-split", "0"],
        ["--op-overhead-us", "-1.0"],
        ["--dma-buffers", "4"],
        ["--vmem-budget-mib", "1.0"],
        ["--hbm-utilization-cap", "0.0"],
        ["--hbm-utilization-cap", "1.2"],
        ["--layer-groups", ","],
        ["--layer-groups", "8,,8"],
    ],
)
def test_cli_parameterized_validation_errors(
    tmp_path: Path, bad_extra: list[str]
) -> None:
    """Verify parameterized CLI validation boundaries raise `MegakernelConfigError`."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(load_qwen35_snapshot_config()), encoding="utf-8")
    with pytest.raises(MegakernelConfigError):
        advisor.main(["--config", str(config_file), *bad_extra])


@pytest.mark.parametrize(
    "bad_argv",
    [
        ["--model", ""],
        ["--model", "   "],
        ["--hf-repo", ""],
        ["--list-models", "--json"],
        ["--list-models", "--unresolved"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--search-layer-groups"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--explain"],
        ["--model", "qwen3.5-397b", "--regime-sweep", "--batch", "4"],
        ["--model", "qwen3.5-397b", "--phase", "prefill", "--decoding-seqs", "8"],
        [
            "--model",
            "qwen3.5-397b",
            "--phase",
            "prefill",
            "--output-tok-s",
            "500.0",
            "--mean-tpot-ms",
            "10.0",
        ],
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--batch",
            "4",
            "--decoding-seqs",
            "4",
        ],
        ["--model", "qwen3.5-397b", "--tp", "8", "--output-tok-s", "500.0"],
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--output-tok-s",
            "-1.0",
            "--mean-tpot-ms",
            "10.0",
        ],
        ["--model", "qwen3.5-397b", "--tp", "8", "--decoding-seqs", "0"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--host-head-us", "per_op=-5"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--host-head-us", "bad_key=10"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--per-op-step-us", "0"],
        ["--model", "qwen3.5-397b", "--tp", "8", "--step-coverage", "1.2"],
    ],
)
def test_cli_rejects_empty_targets_and_incompatible_modes(
    bad_argv: list[str],
) -> None:
    """Verify empty targets and incompatible CLI mode/flag combos fail closed."""
    with pytest.raises(MegakernelConfigError):
        advisor.main(bad_argv)


def test_cli_prefill_defaults_osl_zero_and_clean_arg_type_errors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prefill defaults `--osl` to 0 and invalid numeric types name `int`."""
    captured_workloads: list[advisor.WorkloadSpec] = []
    orig_advise = cli.advise

    def _capture_advise(
        profile: advisor.ModelProfile | str | Mapping[str, object],
        workload: advisor.WorkloadSpec,
        chip: advisor.ChipSpec | str = constants.DEFAULT_CHIP,
        plan_error: str = "",
        measurements: specs.MeasurementsData
        | str
        | Path
        | Mapping[str, object]
        | None = None,
    ) -> advisor.AdvisorReport:
        """Record dispatched WorkloadSpec and forward to real `advise`."""
        captured_workloads.append(workload)
        return orig_advise(profile, workload, chip, plan_error, measurements)

    monkeypatch.setattr(cli, "advise", _capture_advise)
    assert (
        advisor.main(
            ["--model", "qwen3.5-397b", "--phase", "prefill", "--tp", "8", "--ep", "8"]
        )
        == 0
    )
    assert (
        advisor.main(
            ["--model", "qwen3.5-397b", "--phase", "decode", "--tp", "8", "--ep", "8"]
        )
        == 0
    )
    assert captured_workloads[0].osl == 0
    expected_decode_osl = 1024
    assert captured_workloads[1].osl == expected_decode_osl

    parser = advisor.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "qwen3.5-397b", "--batch", "abc"])
    stderr_text = capsys.readouterr().err
    assert "invalid int value: 'abc'" in stderr_text
    assert "_MinIntArg(" not in stderr_text


def test_cli_json_and_plain_text_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI plain-text, JSON output, `--state-dtype`, and missing paths."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(load_qwen35_snapshot_config()), encoding="utf-8")
    base_args = [
        "--config",
        str(config_file),
        "--chip",
        "tpu7x",
        "--tp",
        "8",
        "--ep",
        "8",
        "--json",
    ]
    assert advisor.main(base_args) == 0
    fp32_payload = json.loads(capsys.readouterr().out)
    assert fp32_payload["recommended_scope"] == "per_op"
    assert fp32_payload["fastest_scope"] == "whole_step"
    assert fp32_payload["build_megakernel"] is False
    assert fp32_payload["evidence"] == "baseline"
    assert fp32_payload["hypothetical_best"]["support"] == "outside_core_limits"
    assert (
        "kv_or_state_dtype_not_in_core" in fp32_payload["hypothetical_best"]["blockers"]
    )

    assert advisor.main([*base_args, "--state-dtype", "bf16"]) == 0
    bf16_payload = json.loads(capsys.readouterr().out)
    assert bf16_payload["recommended_scope"] == "whole_step"
    assert bf16_payload["recommended_moe_mode"] == "masked_on_chip"
    assert bf16_payload["build_megakernel"] is True
    assert bf16_payload["hbm_pool_overflow"] is False
    assert bf16_payload["vmem_budget_mib"] == advisor.fit_vmem_budget_bytes(
        chip_spec=resolve_chip_spec("tpu7x")
    ) / (1024 * 1024)

    rc_sweep = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--vmem-budget-mib",
            "60.0",
            "--dma-buffers",
            "2",
        ]
    )
    assert rc_sweep == 0
    sweep_stdout = capsys.readouterr().out
    assert "Decode Parallelism: TP8 x EP8 x DCP4" in sweep_stdout
    assert "limit=62.0 MiB, fit=60.0 MiB" in sweep_stdout

    sweep_json = tmp_path / "sweep.json"
    rc_sweep_json = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--json",
            str(sweep_json),
        ]
    )
    assert rc_sweep_json == 0
    sweep_payload = json.loads(sweep_json.read_text(encoding="utf-8"))
    assert sweep_payload["model"] == "qwen3.5-397b"
    assert len(sweep_payload["regimes"]) >= EXPECTED_DEFAULT_REGIMES

    assert (
        advisor.main(
            [
                "--model",
                "moonshotai/Kimi-K3",
                "--tp",
                "32",
                "--ep",
                "32",
                "--vmem-fit-headroom",
                "--no-vmem-alias",
            ]
        )
        == 0
    )
    assert "Model: kimi-k3" in capsys.readouterr().out

    for bad_argv in (
        ["--config", str(tmp_path / "missing.json")],
        ["--model", str(tmp_path / "missing_dir")],
        ["--model", "org/unknown-preset"],
    ):
        with pytest.raises(MegakernelConfigError):
            advisor.main(bad_argv)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_text_reports_name_mechanisms_not_kernels(model_slug: str) -> None:
    """Verify plain-text regime and summary formatters name mechanisms only."""
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    rendered = "\n".join(
        [
            advisor.format_regime_table(sweep, unresolved=True),
            advisor.format_advisor_summary(report, explain=True, unresolved=True),
            advisor.format_model_listing(),
        ]
    )
    assert KERNEL_NAME_DENYLIST.search(rendered) is None


def test_hbm_utilization_cap_default_and_cli_overflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--hbm-utilization-cap` defaults to 0.92 and separates OOM from cap."""
    parsed = advisor.build_parser().parse_args(["--model", "qwen3.5-397b"])
    assert parsed.hbm_utilization_cap == constants.HBM_UTILIZATION_CAP
    expected_cap = 0.92
    assert parsed.hbm_utilization_cap == expected_cap

    base_cmd = ["--model", "glm-5.2", "--tp", "16", "--ep", "16", "--batch", "16"]
    assert advisor.main([*base_cmd, "--json"]) == 0
    default_payload = json.loads(capsys.readouterr().out)
    assert default_payload["hbm_fits"] is True
    assert default_payload["hbm_pool_overflow"] is False

    assert advisor.main([*base_cmd, "--hbm-utilization-cap", "0.45", "--json"]) == 0
    capped_payload = json.loads(capsys.readouterr().out)
    assert capped_payload["hbm_fits"] is True
    assert capped_payload["hbm_pool_overflow"] is True

    assert advisor.main([*base_cmd, "--hbm-utilization-cap", "0.45"]) == 0
    summary_text = capsys.readouterr().out
    assert "hbm_pool_overflow=True" in summary_text
    assert "hbm_fits=True" in summary_text


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_every_output_number_has_provenance_tag(model_slug: str) -> None:
    """Verify every metric in single-point and sweep dicts carries a valid tag."""
    valid_tags = {"DERIVED", "ASSUMED", "MEASURED"}
    required_keys = (
        "fits_hbm",
        "tokens_per_core",
        "onchip_ceiling",
        "compute_bound_at",
        "step_floor_us",
        "throughput_tok_s_per_chip",
        "boundary_cost_us",
        "fusion_gain_max",
        "step_coverage",
        "expected_gain_max",
    )
    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    metrics = report.to_dict()["metrics"]
    assert isinstance(metrics, dict)
    for key in required_keys:
        assert key in metrics
    for entry in metrics.values():
        assert entry["tag"] in valid_tags
        assert bool(entry["source"])
        assert bool(entry["unit"])
        band = entry["band"]
        assert band[0] <= band[1]

    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    raw_regimes = sweep.to_dict()["regimes"]
    assert isinstance(raw_regimes, list)
    for reg in raw_regimes:
        assert isinstance(reg, dict)
        reg_metrics = reg["metrics"]
        assert isinstance(reg_metrics, dict)
        for key in required_keys:
            assert key in reg_metrics
        for entry in reg_metrics.values():
            assert entry["tag"] in valid_tags
            assert entry["band"][0] <= entry["band"][1]


def test_single_layer_is_fused_layer_not_megakernel() -> None:
    """Verify one call per layer is reported as a fused layer, not a megakernel."""
    row = _regime("kimi-k3", "Decode Saturated")
    assert row.fastest_scope == "single_layer"
    assert row.fastest_verdict == "fused_layer_likely"
    assert _regime("kimi-k3", "Preset Prefill Window (W=2048)").fastest_verdict == (
        "megakernel_likely"
    )


def test_throughput_counts_all_window_tokens_per_physical_chip() -> None:
    """Verify tok/s/chip conserves DP * TP * PP * PCP / devices_per_chip."""
    row = _regime("kimi-k3", "Preset Prefill Window (W=2048)")
    expected = 2048 / 512 / (row.step_bound_us * 1e-6)
    got = float(row.metrics["throughput_tok_s_per_chip"]["value"])
    assert row.active_tokens == 2048 // 32
    assert got == pytest.approx(expected, rel=1e-3)

    dec = _regime("kimi-k3", "Decode Small")
    expected_decode = 4 / 16 / (dec.step_bound_us * 1e-6)
    got_decode = float(dec.metrics["throughput_tok_s_per_chip"]["value"])
    assert got_decode == pytest.approx(expected_decode, rel=1e-3)

    orth_report = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8, ep=32, pcp=4),
        "tpu7x",
    )
    assert orth_report.active_tokens == 2048 // 4
    orth_rec = next(
        sc for sc in orth_report.scopes if sc.scope == orth_report.recommended_scope
    )
    expected_orth = 2048 / 16 / (orth_rec.step_bound_us * 1e-6)
    got_orth = float(orth_report.metrics["throughput_tok_s_per_chip"]["value"])
    assert got_orth == pytest.approx(expected_orth, rel=1e-3)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_rationale_and_assumptions_across_sweeps(model_slug: str) -> None:
    """Verify runner-up margin formatting, coverage placement, and launch gaps."""
    row = _regime(model_slug, "Decode Small")
    assert row.step_upper_us > row.step_bound_us
    assert "vs runner-up " in row.rationale
    assert re.search(r"\+\d+\.\d us, \d+\.\d%\)", row.rationale)

    for reg in advisor.advise_regimes(model_slug, "tpu7x").regimes:
        assert "coverage_assumed" not in reg.flags
        assert "step_coverage" not in reg.verdict
        coverage_items = [
            item
            for item in reg.unresolved_assumptions
            if item["name"] == "step_coverage"
        ]
        assert all(item["affects"] == "gain" for item in coverage_items)

    dec_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    report = advisor.advise(
        model_slug,
        advisor.WorkloadSpec(
            phase="decode",
            batch=4 * dec_layout.dp,
            tp=dec_layout.tp,
            ep=dec_layout.ep,
            dp=dec_layout.dp,
            pp=dec_layout.pp,
            dcp=dec_layout.dcp,
            pcp=dec_layout.pcp,
        ),
        "tpu7x",
    )
    assert (
        specs.scalar_float(report.metrics["boundary_cost_us"]["value"], "boundary_cost")
        >= 0.0
    )
    assert report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert (
        specs.scalar_float(report.metrics["launch_gap_us"]["value"], "launch_gap") > 0.0
    )


def test_import_without_jax_and_torch() -> None:
    """Verify advisor is torch-free and restricts JAX to chip and bridge modules."""
    mk_init = Path(mk.__file__)
    adv_dir = mk_init.parent / "advisor"
    jax_allowed = frozenset({"chips.py", "spec_bridge.py"})
    for py_path in (mk_init, *sorted(adv_dir.glob("*.py"))):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        allow_jax = py_path.name in jax_allowed
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Import):
                for alias in ast_node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root != "torch", f"Unexpected torch import in {py_path}"
                    if not allow_jax:
                        assert root != "jax", (
                            f"Unexpected jax import in non-bridge {py_path}"
                        )
            elif isinstance(ast_node, ast.ImportFrom) and ast_node.module:
                root = ast_node.module.split(".", 1)[0]
                assert root != "torch", f"Unexpected torch import in {py_path}"
                if not allow_jax:
                    assert root != "jax", (
                        f"Unexpected jax import in non-bridge {py_path}"
                    )


def test_tpu7x_topologies_start_at_2x2x1_and_tpu6e_alias() -> None:
    """Verify `TPU7X_TOPOLOGIES` starts at `2x2x1` (8 cores) and `tpu6e` resolves."""
    min_torus_cores = 8
    assert min(advisor.TPU7X_TOPOLOGIES) == min_torus_cores
    assert advisor.TPU7X_TOPOLOGIES[8].startswith("2x2x1")
    for sub_host in (1, 2, 4):
        assert sub_host not in advisor.TPU7X_TOPOLOGIES
    c6e = advisor.resolve_chip_spec("tpu6e")
    assert c6e.name == "v6e"
    assert min(c6e.topologies) == 1
    par = advisor.recommend_parallelism(
        advisor.resolve_model_profile("qwen3.5-397b"), "tpu7x"
    )
    assert par.tp * par.dp * par.pp >= min_torus_cores


def test_pd_disagg_alignment_qwen35_kimi_k3_and_dcn_handoff(
    tmp_path: Path,
) -> None:
    """Verify Qwen3.5-397B, Kimi-K3, and Qwen physical OOM saturation handoffs."""
    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    q_by_lbl = {entry.label: entry for entry in q_sweep.regimes}
    q_sat = next(
        entry for entry in q_sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    assert q_by_lbl["Decode Med (B=16, T=1, 4K)"].recommended_scope == "per_op"
    assert q_by_lbl["Decode Med (B=16, T=1, 4K)"].fastest_scope == "whole_step"
    assert q_by_lbl["Decode High (B=64, T=1, 32K)"].fastest_scope == "whole_step"
    assert q_by_lbl["Decode High (B=64, T=1, 32K)"].hbm_fits is True
    assert q_sat.layout == "TP2 x DP4 x EP8"
    assert q_sat.hbm_fits is False
    assert q_sat.fastest_scope == "per_op"
    assert q_sat.num_vmem_splits == 1
    assert "[PHYSICAL OOM]" in advisor.format_regime_table(q_sweep)
    assert q_by_lbl["Preset Prefill Window (W=2176)"].layout == "PP8"
    assert q_by_lbl["Preset Prefill Window (W=2176)"].recommended_scope == "per_op"
    assert q_sweep.disagg_handoff is not None
    assert q_sweep.disagg_handoff.dcn_provenance == "ASSUMED"
    default_dcn_bw_gbps = 100.0
    assert q_sweep.disagg_handoff.dcn_bw_gbps == default_dcn_bw_gbps
    assert q_sweep.disagg_handoff.dcn_bw_gbps_bracket == (50.0, 200.0)

    q_bf16_sweep = advisor.advise_regimes(
        advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16"), "tpu7x"
    )
    q_bf16_by_lbl = {entry.label: entry for entry in q_bf16_sweep.regimes}
    assert q_bf16_by_lbl["Decode Med (B=16, T=1, 4K)"].recommended_scope == "whole_step"

    k_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    k_by_lbl = {entry.label: entry for entry in k_sweep.regimes}
    assert k_by_lbl["Decode Med (B=16, T=1, 4K)"].fastest_scope == "whole_step"
    assert k_by_lbl["Preset Prefill Window (W=2048)"].layout == "TP32 x EP32 x PCP32"
    assert k_by_lbl["Preset Prefill Window (W=2048)"].fastest_scope == "layer_group"
    assert k_by_lbl["Preset Prefill Window (W=8192)"].fastest_scope == "layer_group"

    measurements_file = tmp_path / "dcn_meas.json"
    measurements_file.write_text(
        json.dumps(
            {
                "schema": "advisor.measurements.v1",
                "source": {"kind": "xplane", "file": "dcn.xplane.pb"},
                "dcn_bw_gbps": 160.0,
            }
        ),
        encoding="utf-8",
    )
    q_meas = advisor.advise_regimes(
        "qwen3.5-397b", "tpu7x", measurements=measurements_file
    )
    assert q_meas.disagg_handoff is not None
    assert q_meas.disagg_handoff.dcn_provenance == "MEASURED"
    measured_dcn_bw_gbps = 160.0
    assert q_meas.disagg_handoff.dcn_bw_gbps == measured_dcn_bw_gbps


def test_glm52_regime_sweep_and_disagg_indexer_wire_bytes() -> None:
    """Verify GLM-5.2 regime sweep stays on `per_op` and prices indexer DCN wire."""
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    sweep = advisor.advise_regimes(glm_profile, "tpu7x")
    assert all(entry.recommended_scope == "per_op" for entry in sweep.regimes)
    assert sweep.disagg_handoff is not None
    # ISL=4096 wire payload: 78*4096*(512+64)*1 + 21*4096*(128+4) = 195.38 MB.
    expected_kv_wire_mb = 195.38
    assert sweep.disagg_handoff.kv_wire_mb == pytest.approx(
        expected_kv_wire_mb, abs=0.01
    )
    doubled = _compute_disagg_handoff(
        glm_profile,
        resolve_chip_spec("tpu7x"),
        advisor.DEPLOYMENT_PRESETS["glm-5.2"],
        None,
        isl=2 * sweep.disagg_handoff.isl,
    )
    assert doubled is not None
    assert doubled.kv_wire_mb == pytest.approx(2.0 * expected_kv_wire_mb, abs=0.02)
    table_text = advisor.format_regime_table(sweep)
    assert (
        f"Disaggregated Prefill/Decode DCN Handoff (ISL={sweep.disagg_handoff.isl})"
        in table_text
    )
    assert KERNEL_NAME_DENYLIST.search(table_text) is None


def test_analytical_crossover_batch_flips_recommendation() -> None:
    """Derive exact VMEM batch crossover from physics and verify flip at `b + 1`."""
    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=1,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
    )
    base_report = advisor.advise(qwen_bf16, base_workload, "tpu7x")
    base_whole_step = next(
        scope_est for scope_est in base_report.scopes if scope_est.scope == "whole_step"
    )
    b_cross = base_whole_step.max_vmem_tokens
    assert b_cross > 1

    at_cross = advisor.advise(
        qwen_bf16,
        dataclasses.replace(base_workload, batch=b_cross, batch_explicit=True),
        "tpu7x",
    )
    above_cross = advisor.advise(
        qwen_bf16,
        dataclasses.replace(base_workload, batch=b_cross + 1, batch_explicit=True),
        "tpu7x",
    )
    at_whole = next(s for s in at_cross.scopes if s.scope == "whole_step")
    above_whole = next(s for s in above_cross.scopes if s.scope == "whole_step")
    assert at_whole.num_vmem_splits == 1
    assert at_cross.fastest_scope == "whole_step"
    expected_split_passes = 2
    assert above_whole.num_vmem_splits == expected_split_passes
    assert above_cross.fastest_scope == "per_op"
    assert above_cross.recommended_scope == "per_op"

    at_capped = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            base_workload,
            batch=b_cross,
            max_rows_per_call=b_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    above_capped = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            base_workload,
            batch=b_cross + 1,
            max_rows_per_call=b_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert at_capped.fastest_scope == "whole_step"
    assert at_capped.recommended_scope == "per_op"
    at_capped_whole = next(s for s in at_capped.scopes if s.scope == "whole_step")
    assert "rows_exceed_core_max_batch_rows" in at_capped_whole.blockers
    assert above_capped.recommended_scope == "per_op"

    small_vmem_mib = 8.0
    small_base = dataclasses.replace(base_workload, vmem_budget_mib=small_vmem_mib)
    small_report = advisor.advise(qwen_bf16, small_base, "tpu7x")
    small_cross = next(
        s for s in small_report.scopes if s.scope == "whole_step"
    ).max_vmem_tokens
    assert 1 < small_cross <= constants.CORE_LIMITS.max_batch_rows
    at_small = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            small_base,
            batch=small_cross,
            max_rows_per_call=small_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    above_small = advisor.advise(
        qwen_bf16,
        dataclasses.replace(
            small_base,
            batch=small_cross + 1,
            max_rows_per_call=small_cross,
            batch_explicit=True,
        ),
        "tpu7x",
    )
    assert at_small.recommended_scope == "whole_step"
    assert above_small.recommended_scope == "per_op"


def test_advisor_cli_with_bundled_measurements(tmp_path: Path) -> None:
    """Verify CLI `estimate` with `--measurements bundled` and missing-chip error."""
    out_json = tmp_path / "bundled_out.json"
    base_argv = [
        "estimate",
        "--model",
        "qwen3.5-397b",
        "--phase",
        "decode",
        "--batch",
        "16",
        "--tp",
        "8",
        "--ep",
        "8",
        "--measurements",
        "bundled",
    ]
    rc = advisor.main([*base_argv, "--chip", "tpu7x", "--json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["confidence"] == "prior_only"
    assert payload["metrics"]["collective_latency_us"]["tag"] == "ASSUMED"
    assert payload["metrics"]["launch_gap_us"]["tag"] == "ASSUMED"

    with pytest.raises(MegakernelConfigError, match="bundled architectural priors"):
        advisor.main([*base_argv, "--chip", "tpu6e"])


def test_cli_invalid_sharding_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify invalid sharding exits 2 on soft mismatches and raises on hard ones."""
    rc = advisor.main(["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "2"])
    assert rc == constants.INVALID_CONFIG_EXIT_CODE
    out = capsys.readouterr().out
    assert "invalid_config(" in out

    rc_sweep = advisor.main(
        ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "2", "--regime-sweep"]
    )
    assert rc_sweep == constants.INVALID_CONFIG_EXIT_CODE
    sweep_out = capsys.readouterr().out
    assert "invalid_config(" in sweep_out

    rc_search = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--tp",
            "2",
            "--search-layer-groups",
            "--json",
        ]
    )
    assert rc_search == constants.INVALID_CONFIG_EXIT_CODE
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["best"] is None

    for extra in ([], ["--regime-sweep"], ["--search-layer-groups"]):
        with pytest.raises(MegakernelConfigError, match="tp_not_power_of_two"):
            advisor.main(
                ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "7", *extra]
            )


def test_cli_prints_evidence_blockers_and_inputs(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify summary prints `Recommended: per_op`, `Evidence:`, `Blocked by:`."""
    glm_snapshot = json.loads(
        (HF_CONFIG_DIR / "glm-5.2.json").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "glm_5_2_config.json"
    config_path.write_text(json.dumps(glm_snapshot["config"]), encoding="utf-8")
    rc = advisor.main(["--model", str(config_path), "--tp", "16", "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    for marker in (
        "Recommended: per_op",
        "Evidence:",
        "Blocked by:",
        "Inputs:",
    ):
        assert marker in out


def test_cli_max_rows_per_call_prices_microbatches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--max-rows-per-call` caps `whole_step` only in CLI and sweep."""
    base = ["--model", "qwen3.5-397b", "--chip", "tpu7x", "--state-dtype", "bf16"]
    mesh = ["--tp", "8", "--dcp", "4", "--ep", "8"]
    verify = ["--phase", "verify", "--batch", "28", "--spec-tokens", "3", "--json"]
    limit = ["--max-rows-per-call", str(TEST_ROW_LIMIT)]
    assert advisor.main([*base, *mesh, *verify, *limit]) == 0
    split = json.loads(capsys.readouterr().out)
    whole = next(s for s in split["scopes"] if s["scope"] == "whole_step")
    assert whole["max_vmem_tokens"] == TEST_ROW_LIMIT
    assert whole["num_vmem_splits"] > 1
    assert "microbatch_rerun" in split["flags"]

    assert advisor.main([*base, *mesh, *verify]) == 0
    assert "microbatch_rerun" not in json.loads(capsys.readouterr().out)["flags"]

    assert advisor.main([*base, *mesh, "--regime-sweep", "--json", *limit]) == 0
    assert json.loads(capsys.readouterr().out)["max_vmem_tokens"] == TEST_ROW_LIMIT

    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="bf16")
    uncapped_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=32,
        spec_tokens=3,
        context=32768,
        tp=8,
        dcp=4,
        ep=8,
        layer_groups=(8, 8, 8, 8, 8, 8, 8, 4),
    )
    capped_wl = dataclasses.replace(uncapped_wl, max_rows_per_call=TEST_ROW_LIMIT)
    uncapped = {
        s.scope: s for s in advisor.advise(qwen_bf16, uncapped_wl, "tpu7x").scopes
    }
    capped_rep = advisor.advise(qwen_bf16, capped_wl, "tpu7x")
    capped = {s.scope: s for s in capped_rep.scopes}
    assert uncapped["whole_step"].max_vmem_tokens > TEST_ROW_LIMIT
    assert capped["whole_step"].max_vmem_tokens == TEST_ROW_LIMIT
    assert capped["whole_step"].num_vmem_splits > uncapped["whole_step"].num_vmem_splits
    for scope_name in ("per_op", "single_layer", "layer_group"):
        assert (
            capped[scope_name].max_vmem_tokens == uncapped[scope_name].max_vmem_tokens
        )
        assert (
            capped[scope_name].num_vmem_splits == uncapped[scope_name].num_vmem_splits
        )
        assert capped[scope_name].step_bound_us == uncapped[scope_name].step_bound_us


def test_regime_entry_uses_recommended_scope_not_hypothetical() -> None:
    """Verify `RegimeEntry` tracks `recommended_scope` with `hypothetical_best`."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    dec_small = next(r for r in k3_sweep.regimes if r.label.startswith("Decode Small"))
    assert dec_small.recommended_scope == "per_op"
    assert dec_small.fastest_scope == "whole_step"
    per_op_sc = next(s for s in dec_small.scopes if s["scope"] == "per_op")
    whole_sc = next(s for s in dec_small.scopes if s["scope"] == "whole_step")
    assert dec_small.step_bound_us == per_op_sc["step_bound_us"]
    assert dec_small.step_upper_us == per_op_sc["step_upper_us"]
    assert dec_small.max_vmem_tokens == per_op_sc["max_vmem_tokens"]
    assert dec_small.hypothetical_best is not None
    assert dec_small.hypothetical_best["step_bound_us"] == whole_sc["step_bound_us"]

    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    q_ref = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=q_sweep.parallelism.tp,
            ep=q_sweep.parallelism.ep,
            dp=q_sweep.parallelism.dp,
            pp=q_sweep.parallelism.pp,
            dcp=q_sweep.parallelism.dcp,
            pcp=q_sweep.parallelism.pcp,
        ),
        "tpu7x",
    )
    q_ref_whole = next(s for s in q_ref.scopes if s.scope == "whole_step")
    assert q_sweep.max_vmem_tokens == q_ref_whole.max_vmem_tokens


def test_recommend_parallelism_raises_and_handles_v6e_dcp_and_options() -> None:
    """Verify `recommend_parallelism` rejects OOM/invalid phase and honors EP/DCP."""
    model_profile = advisor.resolve_model_profile("kimi-k3")
    tiny_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_capacity_bytes=1024 * 1024,
    )
    with pytest.raises(MegakernelConfigError, match="fits in HBM"):
        advisor.recommend_parallelism(model_profile, tiny_chip)

    with pytest.raises(MegakernelConfigError, match="Unsupported phase"):
        advisor.ParallelismSearchOptions(phase=cast("advisor.Phase", "bogus"))

    v6e_spec = advisor.resolve_chip_spec("v6e")
    qwen35_profile = advisor.resolve_model_profile("qwen3.5-397b")
    par_v6e = advisor.recommend_parallelism(qwen35_profile, v6e_spec)
    assert par_v6e.tp * par_v6e.dp * par_v6e.pp in v6e_spec.topologies
    v6e_sweep = advisor.advise_regimes(qwen35_profile, "v6e")
    assert not v6e_sweep.regimes[0].verdict.startswith("invalid_config(")

    dp2_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=4, dp=2),
    )
    assert len(dp2_sweep.regimes) == EXPECTED_DEFAULT_REGIMES
    for reg in dp2_sweep.regimes:
        assert not reg.verdict.startswith("invalid_config(")
        assert int(str(reg.workload["batch"])) >= int(str(reg.workload["dp"]))

    expected_tp = 8
    expected_ep = 8
    rec_default = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dcp=0),
    )
    assert (rec_default.tp, rec_default.ep, rec_default.dcp) == (
        expected_tp,
        expected_ep,
        1,
    )
    expected_dcp = 4
    rec_explicit = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dcp=expected_dcp, phase="verify"),
    )
    assert (rec_explicit.tp, rec_explicit.ep, rec_explicit.dcp) == (
        expected_tp,
        expected_ep,
        expected_dcp,
    )
    rec_ep_dp = advisor.recommend_parallelism(
        qwen35_profile,
        "tpu7x",
        options=advisor.ParallelismSearchOptions(dp=4, ep=2),
    )
    expected_dp4_tp, expected_dp4_ep = 4, 2
    assert (rec_ep_dp.tp, rec_ep_dp.ep, rec_ep_dp.dp) == (
        expected_dp4_tp,
        expected_dp4_ep,
        4,
    )


def test_cli_and_text_render_layer_group_search(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify `--search-layer-groups` plain-text and JSON outputs."""
    rc = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
        ]
    )
    assert rc == 0
    stdout_text = capsys.readouterr().out
    assert "Layer-Group Partition Search: kimi-k3" in stdout_text
    assert "one fused kernel per group" in stdout_text

    json_out = tmp_path / "lg_search.json"
    rc_json = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
            "--json",
            str(json_out),
        ]
    )
    assert rc_json == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["best"]["num_groups"] == 1


def test_cli_decoding_sequences_littles_law_and_host_head_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify Little's Law decoding sequences, host-head flags, and `--unresolved`."""
    # Closed-form Little's Law: 605.0 * 13.42 / 1000.0 = 8.1191 -> 8 sequences.
    expected_exact_seqs = 8.1191
    expected_rounded_seqs = 8
    exact_seqs, rounded_seqs = advisor.derive_decoding_sequences(605.0, 13.42)
    assert exact_seqs == pytest.approx(expected_exact_seqs, abs=1e-6)
    assert rounded_seqs == expected_rounded_seqs

    rc_littles = advisor.main(
        [
            "estimate",
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--phase",
            "decode",
            "--output-tok-s",
            "605.0",
            "--mean-tpot-ms",
            "13.42",
            "--tp",
            "8",
            "--ep",
            "8",
            "--dcp",
            "4",
            "--json",
        ]
    )
    assert rc_littles == 0
    littles_payload = json.loads(capsys.readouterr().out)
    assert littles_payload["active_tokens"] == expected_rounded_seqs

    cal_args = [
        "estimate",
        "--model",
        "qwen3.5-397b",
        "--state-dtype",
        "bf16",
        "--chip",
        "tpu7x",
        "--phase",
        "decode",
        "--decoding-seqs",
        "8",
        "--tp",
        "8",
        "--ep",
        "8",
        "--dcp",
        "4",
        "--host-head-us",
        "per_op=1800,fused=350",
        "--per-op-step-us",
        "9500",
        "--step-coverage",
        "0.85",
    ]
    assert advisor.main([*cal_args, "--json"]) == 0
    cal_payload = json.loads(capsys.readouterr().out)
    assert cal_payload["gain_basis"] == constants.GAIN_BASIS_CALIBRATED_E2E
    scopes_by_name = {s["scope"]: s for s in cal_payload["scopes"]}
    expected_per_op_head = 1800.0
    expected_fused_head = 350.0
    assert scopes_by_name["per_op"]["host_head_us"] == expected_per_op_head
    assert scopes_by_name["whole_step"]["host_head_us"] == expected_fused_head

    assert advisor.main(cal_args) == 0
    assert "Whole-server E2E gain (calibrated):" in capsys.readouterr().out

    rc_unresolved = advisor.main(
        [
            "estimate",
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--phase",
            "decode",
            "--tp",
            "8",
            "--ep",
            "8",
            "--dcp",
            "4",
            "--unresolved",
        ]
    )
    assert rc_unresolved == 0
    unres_out = capsys.readouterr().out
    assert "Unresolved Assumptions:" in unres_out
    assert "assumed=" in unres_out
    assert "| affects=" in unres_out
    assert "| settle:" in unres_out
    assert "decoding_sequences:" in unres_out


def test_cli_list_models_physics_only_and_reference_topologies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--list-models` lists all shipped models and pure 6-axis layouts."""
    exit_code = advisor.main(["--list-models"])
    assert exit_code == 0
    listing_text = capsys.readouterr().out
    assert "tpu7x" in listing_text
    assert "v6e" in listing_text
    assert "chip_without_core_budget" in listing_text
    assert "priced from hardware rates only" in listing_text
    assert "tier=" not in listing_text
    assert "declared megakernel" not in listing_text
    assert "not modeled:" not in listing_text

    assert len(advisor.MODEL_PRESETS) > 0
    for slug in advisor.MODEL_PRESETS:
        assert slug in listing_text
        assert advisor.models.MODEL_PRESET_DTYPE_BASIS[slug] == "config"
        deployment = advisor.DEPLOYMENT_PRESETS[slug]
        assert deployment.decode.label() in listing_text
        assert deployment.decode.submodule_label() in listing_text


def test_format_report_fastest_scope_and_low_gain_phrasing() -> None:
    """Verify report formatting surfaces fastest scope vs per_op and plain terms."""
    kimi_table = advisor.format_regime_table(advisor.advise_regimes("kimi-k3", "tpu7x"))
    assert "Intensity (FLOP/B)" in kimi_table
    assert "(fastest whole_step:" in kimi_table
    assert "% vs per_op)" in kimi_table
    assert "Fastest-scope fusion gain (estimate)=" in kimi_table

    glm_table = advisor.format_regime_table(advisor.advise_regimes("glm-5.2", "tpu7x"))
    assert "[LOW PHYSICS GAIN <1%]" in glm_table


def test_saturation_batch_shared_between_summary_and_regime_rows() -> None:
    """Verify decode and verify saturation rows share unified `saturation_batch`."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    decode_sat = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Saturated")
    )
    verify_sat = next(
        entry
        for entry in sweep.regimes
        if entry.label.startswith("Spec Verify Saturated")
    )
    sat_batch = int(str(decode_sat.workload["batch"]))
    assert f"B={sat_batch}" in decode_sat.label
    verify_rows = int(str(verify_sat.workload["batch"])) * (
        1 + int(str(verify_sat.workload["spec_tokens"]))
    )
    assert verify_rows >= sat_batch

    sat_dp2_layout = models.ParallelLayout(tp=8, ep=8, dp=2, spec_tokens=3)
    expected_unified_batch = 512
    for decode_dp, raw_batch in ((3, 258), (5, 260)):
        decode_layout = models.ParallelLayout(tp=8, ep=8, dp=decode_dp, spec_tokens=3)
        preset = models.DeploymentPreset(
            decode=decode_layout,
            prefill=sat_dp2_layout,
            saturation=sat_dp2_layout,
        )
        base_wl = advisor.WorkloadSpec(
            phase="decode", batch=raw_batch, tp=8, ep=8, dp=decode_dp
        )
        moved = _with_preset_prefill(
            (
                (f"Decode Saturated (B={raw_batch}, T=1, 32K)", base_wl),
                ("Spec Verify High (B=32, K=3)", base_wl),
            ),
            preset,
        )
        moved_dec_label, moved_dec_wl = next(
            (lbl, wl) for lbl, wl in moved if lbl.startswith("Decode Saturated")
        )
        moved_ver_label, moved_ver_wl = next(
            (lbl, wl) for lbl, wl in moved if lbl.startswith("Spec Verify Saturated")
        )
        assert moved_dec_wl.batch == expected_unified_batch
        assert moved_ver_wl.batch == expected_unified_batch
        assert f"B={expected_unified_batch}" in moved_dec_label
        assert f"B={expected_unified_batch}" in moved_ver_label


def test_cli_accepts_model_dir_and_hbm_utilization_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `--config` resolves local directories and honors HBM cap in reports."""
    model_dir = tmp_path / "qwen_local_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(load_qwen35_snapshot_config()), encoding="utf-8"
    )
    cap_fraction = 0.85
    rc = advisor.main(
        [
            "--config",
            str(model_dir),
            "--chip",
            "tpu7x",
            "--tp",
            "8",
            "--ep",
            "8",
            "--hbm-utilization-cap",
            str(cap_fraction),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hbm_utilization_cap"] == pytest.approx(cap_fraction)


def test_advise_regimes_explicit_ep_without_tp_picks_hbm_fitting_tp() -> None:
    """Verify explicit EP without TP evaluates HBM fit using requested EP."""
    sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dp=4, ep=2),
    )
    expected_tp, expected_ep, expected_dcp = 4, 2, 2
    assert sweep.parallelism.tp == expected_tp
    assert sweep.parallelism.ep == expected_ep
    assert sweep.parallelism.dcp == expected_dcp
    direct = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=expected_tp,
            dp=4,
            ep=expected_ep,
            dcp=expected_dcp,
            pcp=sweep.parallelism.pcp,
        ),
        "tpu7x",
    )
    assert sweep.parallelism.hbm_used_gb == pytest.approx(direct.hbm_used_gb)
    assert all(
        entry.hbm_fits
        for entry in sweep.regimes
        if not entry.label.startswith("Decode Saturated")
    )

    sweep_dcp1 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dp=4, ep=2, dcp=1),
    )
    expected_dcp1_tp = 8
    assert sweep_dcp1.parallelism.tp == expected_dcp1_tp
    assert sweep_dcp1.parallelism.ep == expected_ep
    assert sweep_dcp1.parallelism.dcp == 1
    direct_dcp1 = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=constants.RECOMMEND_DEFAULT_BATCH,
            tp=expected_dcp1_tp,
            dp=4,
            ep=expected_ep,
            dcp=1,
            pcp=sweep_dcp1.parallelism.pcp,
        ),
        "tpu7x",
    )
    assert sweep_dcp1.parallelism.hbm_used_gb == pytest.approx(direct_dcp1.hbm_used_gb)
    assert all(entry.hbm_fits for entry in sweep_dcp1.regimes)


@pytest.mark.parametrize("model_slug", sorted(models.MODEL_PRESETS))
def test_cli_sweep_json_matches_advise_regimes_exact(
    model_slug: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI `--regime-sweep --json` matches `advise_regimes` JSON verbatim."""
    rc = advisor.main(["--model", model_slug, "--regime-sweep", "--json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    api_payload = json.loads(json.dumps(sweep.to_dict()))
    assert cli_payload == api_payload
    assert all(
        item["name"] != "decoding_sequences"
        for item in sweep.regimes[0].unresolved_assumptions
    )


def test_regime_table_vmem_header_names_single_layer_when_whole_step_infeasible() -> (
    None
):
    """Verify sweep VMEM header labels `single_layer` when `whole_step` overflows."""
    sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(vmem_budget_mib=6.5)
        ),
    )
    table = advisor.format_regime_table(sweep)
    assert "(single_layer, 2-slot DMA)" in table
    assert "Ideal-kernel VMEM ceiling (whole_step," not in table


def test_text_format_constants_decode_ref_and_none_ring_safety() -> None:
    """Verify gain constant, None ring safety, empty bands, and Decode Med ref."""
    assert text_format.GAIN_BASIS_CALIBRATED_E2E is constants.GAIN_BASIS_CALIBRATED_E2E
    base_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    empty_band_metrics = dict(base_report.metrics)
    empty_band_metrics["launch_gap_us"] = {
        **base_report.metrics["launch_gap_us"],
        "band": [],
    }
    patched_report = dataclasses.replace(base_report, metrics=empty_band_metrics)
    rendered = text_format.format_advisor_summary(patched_report, explain=True)
    assert "Provenance Breakdown:" in rendered

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    first_entry = sweep.regimes[0]
    none_ring_scopes = tuple(
        {**sc, "selected_dense_slots": None, "selected_expert_slots": None}
        for sc in first_entry.scopes
    )
    patched_sweep = dataclasses.replace(
        sweep,
        regimes=(dataclasses.replace(first_entry, scopes=none_ring_scopes),),
    )
    assert "ring=-" in text_format.format_regime_table(patched_sweep)

    decode_med = next(
        entry for entry in sweep.regimes if entry.label.startswith("Decode Med")
    )
    med_step_item = next(
        item
        for item in decode_med.unresolved_assumptions
        if item["name"] == "per_op_step_us"
    )
    table = text_format.format_regime_table(sweep, unresolved=True)
    unres_section = table.split("Unresolved Assumptions (Decode Reference):", 1)[1]
    assert f"per_op_step_us: assumed={med_step_item['value']}" in unres_section


def test_tailoring_guide_derives_row_tile_and_handles_dense_models() -> None:
    """Verify tailoring guide derives row tile, dense rules, and core-limit note."""
    v6e_sweep = advisor.advise_regimes("kimi-k3", "v6e")
    assert "T < 8" in v6e_sweep.tailoring_guide[0]
    assert "T >= 8" in v6e_sweep.tailoring_guide[0]
    assert "T < 32" not in v6e_sweep.tailoring_guide[0]
    assert "outside_core_limits" in v6e_sweep.tailoring_guide[0]

    glm_sweep = advisor.advise_regimes("glm-5.2", "tpu7x")
    assert "outside_core_limits" in glm_sweep.tailoring_guide[0]

    dense_profile = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-small-gqa",
            num_layers=8,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            kv_dtype="fp8",
        )
    )
    dense_sweep = advisor.advise_regimes(
        dense_profile,
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=8),
    )
    joined_guide = "\n".join(dense_sweep.tailoring_guide)
    assert "dense_mlp" in joined_guide
    assert "masked_on_chip" not in joined_guide
    assert (
        f"({dense_sweep.max_vmem_tokens} < T <= {dense_sweep.ridge_tokens}"
        not in joined_guide
    )


def test_cli_rejects_empty_and_directory_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject empty, whitespace, and directory values on CLI path and name flags."""
    monkeypatch.chdir(tmp_path)
    qwen_snapshot = json.loads(
        (HF_CONFIG_DIR / "qwen3.5-397b.json").read_text(encoding="utf-8")
    )
    (tmp_path / "config.json").write_text(
        json.dumps(qwen_snapshot["config"]), encoding="utf-8"
    )
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()
    bad_argvs = (
        ["--config", ""],
        ["--config", "   "],
        ["--model", ""],
        ["--model", "   "],
        ["--hf-repo", ""],
        ["--hf-repo", "   "],
        ["--model", "qwen3.5-397b", "--json", ""],
        ["--model", "qwen3.5-397b", "--json", "   "],
        ["--model", "qwen3.5-397b", "--json", str(out_dir)],
        ["--model", "qwen3.5-397b", "--measurements", ""],
        ["--model", "qwen3.5-397b", "--measurements", "   "],
        ["--model", "qwen3.5-397b", "--measurements", str(out_dir)],
        ["--model", "qwen3.5-397b", "--chip", ""],
        ["--model", "qwen3.5-397b", "--chip", "   "],
        ["--model", "qwen3.5-397b", "--act-dtype", ""],
        ["--model", "qwen3.5-397b", "--state-dtype", ""],
    )
    for argv in bad_argvs:
        before = set(tmp_path.iterdir())
        try:
            rc = advisor.main(argv)
        except MegakernelConfigError:
            rc = constants.INVALID_CONFIG_EXIT_CODE
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        assert rc != 0, f"{argv} returned 0 unexpectedly"
        assert set(tmp_path.iterdir()) == before, f"{argv} modified cwd files"


def test_cli_json_matches_api_over_case_grid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI JSON output matches `advise()` verbatim across base case grid."""
    for label, slug, profile, phase, lay_name, wl in iter_case_grid():
        if (
            lay_name not in ("preset", "tp_ep_only")
            or label != f"{slug}/{phase}/{lay_name}"
        ):
            continue
        argv = [
            "--model",
            slug,
            "--phase",
            phase,
            "--batch",
            str(wl.batch),
            "--isl",
            str(wl.isl),
            "--osl",
            str(wl.osl),
            "--context",
            str(wl.context),
            "--spec-tokens",
            str(wl.spec_tokens),
            "--chunk-size",
            str(wl.chunk_size),
            "--tp",
            str(wl.tp),
            "--ep",
            str(wl.ep),
            "--dp",
            str(wl.dp),
            "--pp",
            str(wl.pp),
            "--dcp",
            str(wl.dcp),
            "--pcp",
            str(wl.pcp),
            "--json",
        ]
        rc = advisor.main(argv)
        cli_payload = json.loads(capsys.readouterr().out)
        api_rep = advisor.advise(
            profile, dataclasses.replace(wl, batch_explicit=True), "tpu7x"
        )
        api_payload = json.loads(json.dumps(api_rep.to_dict()))
        assert cli_payload == api_payload, f"{label}: CLI JSON != advise()"
        expected_rc = (
            constants.INVALID_CONFIG_EXIT_CODE
            if api_rep.verdict.startswith("invalid_config(")
            else 0
        )
        assert rc == expected_rc, f"{label}: rc={rc} != {expected_rc}"


def test_auto_layout_fits_hbm_when_a_searched_layout_fits() -> None:
    """Verify auto-search finds HBM-fitting DCP layout and honors explicit dcp=1."""
    for slug in sorted(models.MODEL_PRESETS):
        sweep = advisor.advise_regimes(
            slug,
            "tpu7x",
            options=advisor.RegimeSweepOptions(deployment_preset=False),
        )
        decode_high = [
            entry for entry in sweep.regimes if entry.label.startswith("Decode High")
        ]
        assert decode_high
        assert all(entry.hbm_fits is True for entry in decode_high)

    sweep_explicit = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dcp=1),
    )
    assert sweep_explicit.parallelism.dcp == 1


def test_sweep_rows_validate_for_every_preset_role() -> None:
    """Verify sweep prefill OSL=0, zero-spec saturation fallback, and role groups."""
    for slug in sorted(models.MODEL_PRESETS):
        sweep = advisor.advise_regimes(slug, "tpu7x")
        for entry in sweep.regimes:
            if entry.phase == "prefill":
                assert entry.workload["osl"] == constants.DEFAULT_PREFILL_OSL

    profile = advisor.resolve_model_profile("kimi-k3")
    chip_tpu7x = resolve_chip_spec("tpu7x")
    base_preset = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    assert base_preset.saturation is not None
    half_layers = profile.num_layers // 2
    decode_groups = (half_layers, profile.num_layers - half_layers)
    custom_preset = models.DeploymentPreset(
        decode=dataclasses.replace(
            base_preset.decode, spec_tokens=0, layer_groups=decode_groups
        ),
        prefill=dataclasses.replace(base_preset.prefill, layer_groups=()),
        saturation=dataclasses.replace(base_preset.saturation, layer_groups=()),
        prefill_windows=base_preset.prefill_windows,
    )
    base_wl = advisor.WorkloadSpec(
        tp=custom_preset.decode.tp,
        ep=custom_preset.decode.ep,
        dp=custom_preset.decode.dp,
        pp=custom_preset.decode.pp,
        dcp=custom_preset.decode.dcp,
        pcp=custom_preset.decode.pcp,
        layer_groups=decode_groups,
        batch_explicit=True,
    )
    moved = _with_preset_prefill(
        _build_reference_workloads(base_wl, constants.DEFAULT_SPEC_TOKENS),
        custom_preset,
    )
    for label, role_wl in moved:
        advisor.validate_advise_inputs(profile, role_wl, chip_tpu7x)
        if role_wl.phase == "prefill" or "Saturated" in label:
            assert role_wl.layer_groups == ()
