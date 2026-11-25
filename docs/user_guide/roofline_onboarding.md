# Roofline Model and Hardware Onboarding

The `vllm_torchtpu.roofline` package computes physical lower bounds and
uncalibrated serial-phase candidate orderings directly from a Hugging Face
`config.json` (`ModelGeometry` and `Dtypes`) and a TPU hardware profile
(`ChipSpec` and `SliceTopology`).

> **Hardware-Floor Scope (`bound_*` Pruning vs. `est_*` Ordering):**
> All subcommands (`solve`, `ceiling`, `plan`, `explain`, and `inspect`) run on
> CPU with zero TPU allocation. Lower bounds (`bound_ttft_ms`, `bound_tpot_ms`,
> `floor_throughput_per_chip`) depend only on hardware specs, HF model
> dimensions, and traffic inputs. Uncalibrated serial-phase estimates
> (`est_ttft_ms`, `est_tpot_ms`, `est_interactivity_tok_s`) order every
> physically legal layout per concurrency (`--sort-by throughput|interactivity`,
> optional `--top-k N`). Supplying optional flags (`--hbm-reserve-gb`,
> `--collective-latency-us`, `--moe-activation`, `--kv-offload-gb`,
> `--dcn-gbit-per-s`, `--weight-dtype`, `--expert-weight-dtype`, `--kv-dtype`)
> records explicit entries in `provenance.supplied_values` without altering
> uncalibrated defaults when omitted.

## 1. How to Onboard a New Model and Fail-Closed Contract

### Strict Fail-Closed Policy and Canonical 1:1 HF Identity

`vllm_torchtpu.roofline` rejects unsupported models instead of extrapolating
from partial config fields. All public entrypoints (`supports`,
`from_hf_config`, `audit_hf_config`, `resolve_model_identity`,
`load_hf_config_with_identity`, and CLI subcommands `inspect`, `solve`,
`ceiling`, `plan`, and `explain`) fail closed with
`UnsupportedArchitectureError` (CLI exit code `2`) whenever a config:

- uses an unlisted or omitted `model_type`;
- declares mismatched `architectures` or a non-causal draft model
  (`is_causal: false`);
- activates an unmodeled structural control (for example, `o_lora_rank > 0`, or
  unmodeled attention or recurrence layer types).

No bypass flag exists. Internally, `audit_hf_config` runs a private dry-run
parse solely to report `unmapped_structural_keys` on blocked configs while
keeping `onboarding_tier != "tier0_ready"` and `geometry = None`.

### Currently Supported Canonical Models

The canonical registry (`SUPPORTED_MODELS` in `constants.py`) maps each
supported model ID 1:1 to public Hugging Face repositories, `(model_type,
architectures)` pairs, and allowed layer kinds:

| Canonical ID | Public HF Repositories | Architecture & Quantization |
| :--- | :--- | :--- |
| `qwen3_5_moe` | `Qwen/Qwen3.5-397B-A17B-FP8`, `nvidia/Qwen3.5-397B-A17B-NVFP4` | Hybrid GDN + GQA + MoE + MTP (FP8, ModelOpt NVFP4) |
| `glm_moe_dsa` | `zai-org/GLM-5.2-FP8`, `nvidia/GLM-5.2-NVFP4` | MLA + DSA + IndexShare + MoE (FP8, ModelOpt NVFP4) |

### Step-by-Step Onboarding Workflow (`inspect --json`)

Run `inspect` against a public Hugging Face repository or local `config.json`:

```bash
python -m vllm_torchtpu.roofline inspect \
  --model <org/repo_or_local_dir> \
  --json
```

`inspect` emits `vllm_torchtpu.roofline.inspect/v1` JSON (`schema_version`,
`chip`, `onboarding_tier`, `canonical_id`, `hf_repo`, `model_type`,
`model_type_status`, `layer_counts`, `param_counts`, `dtypes`,
`unmapped_structural_keys`, `ignored_non_text_keys`, and `blocking_reasons`),
exiting `0` only for `tier0_ready` and `2` otherwise:

| Tier | `onboarding_tier` | Required Action |
| :--- | :--- | :--- |
| Tier 0 | `tier0_ready` | Canonical identity matched and zero unmapped structural keys; run `solve` / `ceiling` / `plan` / `explain` on CPU directly. |
| Tier 1 | `tier1_config_needed` | Primitives already exist; map config or quantization keys in `geometry.py` / `hf_loader.py` and register `ModelSpec` in `SUPPORTED_MODELS`. |
| Tier 2 | `tier2_primitive_needed` | Unmodeled attention, recurrence, FFN, or table primitive; implement through the 4-file primitive seam below. |

### Tier 2 Four-File Primitive Seam

When a new architecture introduces an unmodeled primitive, keep all math inside
four strictly downward modules:

1. `constants.py`: register canonical `ModelSpec` in `SUPPORTED_MODELS` and
   structural constants.
2. `geometry.py`: parse `config.json` fields, track consumed keys, enforce
   `_validate_family_layer_kinds`, and emit `ModelGeometry`.
3. `memory.py`: account for weights, KV cache (separating linear KV from fixed
   sliding-window KV), recurrent state, and `param_counts` conservation.
4. `modules.py`: implement per-layer HBM, matmul, vector, and collective costs
   (with both strict unpadded lower bounds and estimate terms).

### Mandatory Gate to Promote a Model into `SUPPORTED_MODELS`

Every promoted model must pass four CPU test suites:

1. Check in the public HF `config.json` under `tests/roofline/fixtures/` and
   pin its repository, revision, SHA-256, and hand-derived geometry in
   `tests/roofline/fixtures/manifest.json`.
2. Pass `test_model_lockin.py` and `test_roofline_fixtures.py`:
   `resolve_model_identity`, `audit_hf_config` returning `tier0_ready` with
   zero `unmapped_structural_keys`, `param_counts` conservation, and
   spoofed-repo negative controls.
3. Pass independent arithmetic oracles in `test_analytic_oracles.py` and
   physical invariants (`bound <= est`, `DP=1 == DP=2`, zero-think and full-hit
   continuity, batch monotonicity, `pool_capacity_bound` pruning, and zero
   false drops) in `test_physics_invariants.py`, `test_roofline_invariants.py`,
   `test_roofline_occupancy.py`, `test_roofline_step.py`, and
   `test_roofline_solver.py`.
4. Regenerate and verify golden scenarios and rendered tables with
   `tests/roofline/regen_golden.py --check` and
   `tests/roofline/render_docs_tables.py --check`.

## 2. Physics-Only Legality, Transfer, and Offload Floors

All candidate layouts are admitted strictly by physical constraints: core
product equals topology cores, attention heads / KV heads / experts / GDN heads
shard evenly, pipeline stages do not exceed layer count, and logical weights,
recurrent state, and KV cache fit within available per-core HBM (full HBM minus
optional `--hbm-reserve-gb`, default `0`).

- **Uneven PP layer partitioning**: balances per-stage HBM footprints and
  evaluates exact contiguous stage slices (`scale_pp_stage_floor`) when
  embedding and LM-head weights skew stage memory.
- **Disaggregated DCN KV transfer** (`--dcn-gbit-per-s`): models uncached KV
  and recurrent state migration latency and per-pool DCN capacity bounds across
  prefill and decode slices.
- **Host DRAM KV offload** (`--kv-offload-gb <GiB>`): expands effective prefix
  cache capacity per instance and charges host PCIe read latency for prefix
  tokens spilled beyond on-chip HBM.

## 3. Zero-Code Hardware Onboarding (`--chip-spec` and `from_jax`)

Hardware profiles describe per-core compute, memory, interconnect, and mesh
capabilities (`cores_per_chip`, `chips_per_host`, `ici_links_per_chip`,
`fp4_flops`, `int8_flops`, `int4_flops`, `vector_flops`,
`physical_alignment_elems`, `subchannel_block_k_floor`, `min_torus_dim`,
`has_sparse_core`, and `dma_granule_bytes`).

### Live JAX Hydration (`from_jax`)

When running in a JAX-enabled environment, `from_jax(name, chip_version,
overrides)` queries `jax.experimental.pallas.tpu.get_tpu_info_for_chip` for the
requested `pltpu.ChipVersion` to hydrate core count, tiling, and memory traits
before applying optional `ChipOverrides`.

### Declarative JSON Profile (`vllm_torchtpu.roofline.chip_spec/v1`)

To evaluate projected or custom chips without code edits, pass
`--chip-spec <path.json>`. Required fields are `schema_version`, `name`, and
`provenance` (`"public_spec"` or `"projected"`). Unknown keys are rejected.

- **With `"base"`** (for example `"base": "tpu7x"`): omitted fields inherit
  from the base `ChipSpec`. Changing `ici_links_per_chip` or `ici_link_bw`
  recomputes both `ici_bw` and `d2d_bw` unless overridden; changing
  `cores_per_chip` recomputes `ici_bw` while leaving `d2d_bw` unchanged.
  Changing `dcn_host_gbit_per_s` or `chips_per_host` recomputes `dcn_bw`.
- **Without `"base"`** (standalone profile): `hbm_bytes`, `hbm_bw`,
  `bf16_flops`, `vmem_bytes`, `cores_per_chip`, and `chips_per_host` are
  required. Omitted `fp8_flops`, `int8_flops`, `int4_flops`, and `fp4_flops`
  default to `0.0`; omitted ICI and DCN fields use the package defaults; and
  omitted low-level capability fields fall back to `_DEFAULT_TPU_CAPS`
  (initialized once from `TPU_7X` at import time).

```json
{
  "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
  "provenance": "projected",
  "name": "example_1core_2d",
  "cores_per_chip": 1,
  "chips_per_host": 4,
  "hbm_bytes": 128000000000,
  "hbm_bw": 4000000000000,
  "bf16_flops": 2000000000000000,
  "fp8_flops": 4000000000000000,
  "vmem_bytes": 134217728,
  "ici_link_bw": 100000000000,
  "ici_links_per_chip": 4,
  "ici_hop_latency_s": 5e-7,
  "dcn_host_gbit_per_s": 800,
  "topologies": [
    {
      "name": "2x2",
      "chip_axes": [2, 2, 1],
      "wrapped_axes": [false, false, false]
    }
  ]
}
```

Run `solve` with the profile on any supported canonical model:

```bash
python -m vllm_torchtpu.roofline solve \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --chip-spec example_1core_2d.json \
  --example-scenario coding-agent \
  --max-chips 4 \
  --json
```

## 4. Open-Loop vs. Closed-Loop Scenarios (`think_time_s`)

Omitting `--think-time-s` in custom `solve` invocations keeps
`scenario.think_time_s = None` (open-loop mode) instead of assuming zero pause
(`0.0 s`). By contrast, `--example-scenario` presets set `think_time_s` to
`0.0` (closed loop) unless `--think-time-s` is given:

- **Open-loop (`--think-time-s` omitted, `think_time_s = None`)**: physical
  pruning checks per-request `bound_ttft_ms`, batch-1 `bound_tpot_ms`, and
  physical HBM capacity bounds without enforcing closed-loop saturation capacity
  floors.
- **Closed-loop (`--think-time-s <seconds>`)**: enables closed-loop Little's
  law minimum throughput demand and fixed-point residency bounds for saturation
  benchmarks (`0.0`) or fixed client think pauses.
