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
"""CPU contract, layout, 3-step plan, and import-boundary tests for Qwen4Exp."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.qwen4_exp.attention_plan import (
    AttentionDesc,
    AttentionPlan,
    CoreAttn,
    OutProj,
    QKVProj,
)
from vllm_torchtpu.models.qwen4_exp.layout import (
    Axis,
    QKVLayout,
    TensorLayout,
    expected_shape,
    kv_hnd,
    kv_nhd,
    packed_q,
    unfused_o,
    unfused_qkv,
    verify_qkv,
    verify_tensor,
)

_PKG_DIR = Path(__file__).resolve().parents[3] / "src/vllm_torchtpu/models/qwen4_exp"


def _desc(
    *,
    sparse: bool = False,
    kv_layout: KVCacheLayout = KVCacheLayout.LBNHC,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: torch.dtype = torch.float8_e4m3fn,
) -> AttentionDesc:
    return AttentionDesc(
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        kv_layout=kv_layout,
        sparse=sparse,
    )


# ── TensorLayout contract checks ───────────────────────────────────────────


def test_layout_equality_catches_merge_order_and_packing_mismatches() -> None:
    assert unfused_o() == TensorLayout(
        (Axis.T, (Axis.H_KV, Axis.G), Axis.D), torch.bfloat16
    )
    # Non-commutative merge: [k0, v0, k1, v1] != [k0, k1, v0, v1].
    kv_concat = TensorLayout(
        (Axis.T, (Axis.KV, Axis.H_KV), Axis.D),
        torch.float8_e4m3fn,
        pack_sublane=True,
        lane_align=128,
        quantized=True,
    )
    assert kv_nhd() != kv_concat
    assert kv_hnd() != TensorLayout(
        ((Axis.H_KV, Axis.KV), Axis.D, Axis.T),
        torch.float8_e4m3fn,
        pack_sublane=True,
        lane_align=16,
        quantized=True,
    )
    assert kv_nhd(fold_lanes=True) != kv_nhd(fold_lanes=False)
    assert kv_nhd(quantized=True) != kv_nhd(quantized=False)


def test_layout_expected_shapes_and_verify() -> None:
    assert expected_shape(unfused_o(), T=64, H_kv=2, G=8, D=128) == (64, 16, 128)
    assert expected_shape(packed_q(), T=64, H_kv=2, G=8, D=128) == (2, 64, 4, 2, 128)
    assert expected_shape(kv_nhd(), T=64, H_kv=2, G=8, D=128) == (64, 1, 4, 128)
    assert expected_shape(kv_nhd(), T=64, H_kv=2, G=8, D=256) == (64, 2, 4, 128)
    assert expected_shape(kv_nhd(fold_lanes=False), T=64, H_kv=2, G=8, D=256) == (
        64,
        1,
        4,
        256,
    )
    assert expected_shape(kv_hnd(), T=64, H_kv=2, G=8, D=128, page_size=128) == (
        4,
        32,
        4,
        128,
    )
    assert expected_shape(kv_hnd(), T=64, H_kv=2, G=8, D=256, page_size=16) == (
        4,
        64,
        4,
        128,
    )

    with pytest.raises(ValueError, match="Expected"):
        verify_tensor(
            packed_q(),
            torch.empty((64, 16, 128), dtype=torch.bfloat16),
            T=64,
            H_kv=2,
            G=8,
            D=128,
        )


def test_layout_verify_works_under_fake_tensor_mode() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty((2, 64, 4, 2, 128), dtype=torch.bfloat16)
        nhd = torch.empty((64, 1, 4, 128), dtype=torch.float8_e4m3fn)
        verify_qkv(
            QKVLayout(q=packed_q(), kv=kv_nhd()), q, nhd, T=64, H_kv=2, G=8, D=128
        )

        hnd = torch.empty((4, 32, 4, 128), dtype=torch.float8_e4m3fn)
        verify_qkv(
            QKVLayout(q=packed_q(), kv=kv_hnd()),
            q,
            hnd,
            T=64,
            H_kv=2,
            G=8,
            D=128,
            page_size=128,
        )


# ── 3-step plan resolution ─────────────────────────────────────────────────


def test_attention_plan_resolve_defaults() -> None:
    dense = AttentionPlan.resolve(_desc(sparse=False))
    assert dense == AttentionPlan(
        qkv_proj=QKVProj.UNFUSED,
        core_attn=CoreAttn.RPA_V3,
        out_proj=OutProj.UNFUSED,
        qkv_layout=unfused_qkv(torch.bfloat16),
        o_layout=unfused_o(torch.bfloat16),
    )

    sparse = AttentionPlan.resolve(_desc(sparse=True))
    assert sparse == AttentionPlan(
        qkv_proj=QKVProj.UNFUSED,
        core_attn=CoreAttn.QSA,
        out_proj=OutProj.UNFUSED,
        qkv_layout=unfused_qkv(torch.bfloat16),
        o_layout=unfused_o(torch.bfloat16),
    )


def test_attention_plan_rejects_unsupported_stage_overrides() -> None:
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), qkv_proj=QKVProj.FUSED)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), out_proj=OutProj.FUSED)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), core_attn=CoreAttn.BATCHED_RPA)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), core_attn=CoreAttn.QSA)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=True, kv_layout=KVCacheLayout.LBHNC))


# ── Mechanical import-boundary enforcement ────────────────────────────────


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_qwen4_exp_import_hierarchy_is_strictly_top_down_and_independent() -> None:
    # 1. Pure leaf modules (layout.py, ops/*) never import vllm or layer files.
    for leaf in [_PKG_DIR / "layout.py", *(_PKG_DIR / "ops").glob("*.py")]:
        imports = _imported_modules(leaf)
        assert not any(m == "vllm" or m.startswith("vllm.") for m in imports), (
            f"{leaf.name} must stay free of vllm imports: {sorted(imports)}"
        )
        assert not any(
            m.startswith("vllm_torchtpu.models.qwen4_exp.")
            and not m.startswith("vllm_torchtpu.models.qwen4_exp.ops.")
            for m in imports
        ), f"{leaf.name} must not import model/layer modules: {sorted(imports)}"

    # 2. Strict top-down and sibling independence within qwen4_exp.
    pkg = "vllm_torchtpu.models.qwen4_exp."

    def local(name: str) -> set[str]:
        return {
            m.removeprefix(pkg)
            for m in _imported_modules(_PKG_DIR / name)
            if m.startswith(pkg)
        }

    assert local("ple.py") <= {"ops.ple_state"}
    assert local("qsa.py") <= {"layout", "ops.qsa"}
    assert local("attention_plan.py") <= {"layout", "qsa"}
    assert local("attention.py") <= {"attention_plan", "qsa", "ops.dense_attn"}
    assert local("model.py") <= {"attention", "ple"}
