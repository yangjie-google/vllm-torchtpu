# Roofline Basis Reference Tables

Generated from `tests/roofline/fixtures/manifest.json` and `tests/roofline/golden/scenarios.json`.

## Supported Basis Models

| Fixture | Repo | Layers | Hidden | Heads (Q/KV) | Experts (Total/Top-K) | Vocab |
| :--- | :--- | ---: | ---: | :--- | :--- | ---: |
| `glm_5_2_fp8.json` | `zai-org/GLM-5.2-FP8` | 78 | 6144 | 64/64 | 256/8 | 154880 |
| `glm_5_2_nvfp4.json` | `nvidia/GLM-5.2-NVFP4` | 78 | 6144 | 64/64 | 256/8 | 154880 |
| `qwen3_5_397b_fp8.json` | `Qwen/Qwen3.5-397B-A17B-FP8` | 60 | 4096 | 32/2 | 512/10 | 248320 |
| `qwen3_5_397b_nvfp4.json` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | 60 | 4096 | 32/2 | 512/10 | 248320 |

## Representative Scenario Summary (`chat-8k1k`, mixed)

| Model | Slice | Legal Candidates | Top Layout | TPOT Floor (ms) | Est. Tput (tok/s/chip) | Ceiling (tok/s/chip) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `glm_5_2_fp8` | `2x2x1` | 102 | `TP8 x EP4 x DCP8` | 1.76 | 2,695.2 | 24,565.8 |
| `glm_5_2_fp8` | `2x2x2` | 429 | `TP2 x DP8 x EP16 x DCP2` | 3.53 | 4,016.9 | 19,445.4 |
| `glm_5_2_fp8` | `2x2x4` | 1134 | `TP2 x DP8 x EP16 x DCP2` | 3.53 | 4,016.9 | 19,445.4 |
| `glm_5_2_nvfp4` | `2x2x1` | 120 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `glm_5_2_nvfp4` | `2x2x2` | 465 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `glm_5_2_nvfp4` | `2x2x4` | 1188 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `qwen3_5_397b_fp8` | `2x2x1` | 102 | `TP2 x DP4 x EP8` | 1.88 | 9,966.3 | 74,396.7 |
| `qwen3_5_397b_fp8` | `2x2x2` | 369 | `TP2 x DP4 x EP8` | 1.88 | 9,966.3 | 74,396.7 |
| `qwen3_5_397b_fp8` | `2x2x4` | 924 | `TP2 x DP4 x EP8` | 1.88 | 9,966.3 | 74,396.7 |
| `qwen3_5_397b_nvfp4` | `2x2x1` | 120 | `TP2 x DP4 x EP8` | 2.71 | 10,901.3 | 51,625.6 |
| `qwen3_5_397b_nvfp4` | `2x2x2` | 405 | `TP2 x DP4 x EP8` | 2.71 | 10,901.3 | 51,625.6 |
| `qwen3_5_397b_nvfp4` | `2x2x4` | 978 | `TP2 x DP4 x EP8` | 2.71 | 10,901.3 | 51,625.6 |
