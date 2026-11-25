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
"""Tests for the Qwen4Exp PLE state kept in the unified KV pool."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

# The pure-JAX core behind PLEStateCache; the op wrapper itself is
# covered end to end by test_qwen4_exp_tiny_e2e.py.
from vllm_torchtpu.models.qwen4_exp.ops.ple_state import (
    _pages_per_block,
    _read_states,
    _write_states,
    read_ple_state,
    write_ple_state,
)
from vllm_torchtpu.models.qwen4_exp.ple import ragged_tail

# Two 16-token pages per block of a [pages, 16, 1, 4, 128] fp8 pool: 16 KiB
# per block, enough for a 9 x 512 bf16 window and two token ids.
_POOL_SHAPE = (18, 16, 1, 4, 128)
_PAGES_PER_BLOCK = 3
_WINDOW = (9, 512)


def _random_pool(dtype: jnp.dtype) -> jnp.ndarray:
    """A pool of _POOL_SHAPE in ``dtype`` with random bytes."""
    itemsize = jnp.dtype(dtype).itemsize
    shape = (*_POOL_SHAPE[:-1], _POOL_SHAPE[-1] * itemsize)
    raw = np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8)
    return jnp.asarray(raw).view(dtype)


_POOL_DTYPES = pytest.mark.parametrize(
    "pool_dtype", [jnp.float8_e4m3fn, jnp.bfloat16], ids=["fp8_pool", "bf16_pool"]
)


def _write_three_states(pool):
    """Writes 3 states to blocks [3, 5, 0]; row 2 owns no block (null block 0)."""
    rng = np.random.default_rng(1)
    window = jnp.asarray(rng.standard_normal((3, *_WINDOW)), jnp.bfloat16)
    context = jnp.asarray([[7, 8], [9, 10], [11, 12]], jnp.int32)
    blocks = jnp.asarray([3, 5, 0], jnp.int32)
    new_pool = _write_states(
        pool, blocks, window, context, pages_per_block=_PAGES_PER_BLOCK
    )
    return new_pool, blocks, window, context


def _read(pool, blocks):
    return _read_states(
        pool,
        blocks,
        pages_per_block=_PAGES_PER_BLOCK,
        window_shape=_WINDOW,
        window_dtype=jnp.bfloat16,
        context_len=2,
    )


@_POOL_DTYPES
def test_round_trip_is_bit_exact(pool_dtype) -> None:
    new_pool, blocks, window, context = _write_three_states(_random_pool(pool_dtype))
    got_window, got_context = _read(new_pool, blocks[:2])
    np.testing.assert_array_equal(
        np.asarray(got_window, np.float32), np.asarray(window[:2], np.float32)
    )
    np.testing.assert_array_equal(np.asarray(got_context), np.asarray(context[:2]))


@_POOL_DTYPES
def test_single_page_block_round_trip_preserves_special_bits(pool_dtype) -> None:
    """pages_per_block=1 preserves -0.0, subnormals, +/-Inf, and negative int32."""
    pool = jnp.zeros((6, 48, 1, 4, 128), dtype=pool_dtype)
    blocks = jnp.asarray([2, 4], jnp.int32)
    # Cover normal values, -0.0 (0x8000), subnormal (0x0001), +/-Inf, and -1.
    rng = np.random.default_rng(2)
    base = jnp.asarray(rng.standard_normal((2, *_WINDOW)), jnp.bfloat16)
    bits = np.asarray(base.view(jnp.uint16)).copy()
    bits[0, 0, :5] = np.array([0x0000, 0x8000, 0x0001, 0x7F80, 0xFF80], dtype=np.uint16)
    window = jnp.asarray(bits).view(jnp.bfloat16)
    context = jnp.asarray([[-1, 0x7FFFFFFF], [42, -12345]], jnp.int32)
    new_pool = _write_states(pool, blocks, window, context, pages_per_block=1)
    got_window, got_context = _read_states(
        new_pool,
        blocks,
        pages_per_block=1,
        window_shape=_WINDOW,
        window_dtype=jnp.bfloat16,
        context_len=2,
    )
    np.testing.assert_array_equal(np.asarray(got_window.view(jnp.uint16)), bits)
    np.testing.assert_array_equal(np.asarray(got_context), np.asarray(context))


@_POOL_DTYPES
def test_write_touches_only_the_given_blocks(pool_dtype) -> None:
    pool = _random_pool(pool_dtype)
    new_pool, *_ = _write_three_states(pool)
    assert new_pool.shape == pool.shape and new_pool.dtype == pool.dtype
    num_blocks = _POOL_SHAPE[0] // _PAGES_PER_BLOCK
    before = np.asarray(pool.view(jnp.uint8)).reshape(num_blocks, -1)
    after = np.asarray(new_pool.view(jnp.uint8)).reshape(num_blocks, -1)
    for block in (0, 1, 2, 4):
        np.testing.assert_array_equal(after[block], before[block])


def test_mismatched_state_shapes_are_rejected() -> None:
    pool = _random_pool(jnp.bfloat16)
    blocks = jnp.asarray([3, 5], jnp.int32)
    window = jnp.zeros((2, *_WINDOW), jnp.bfloat16)
    with pytest.raises(ValueError, match="ngram_context"):
        _write_states(
            pool,
            blocks,
            window,
            jnp.zeros((2, 2), jnp.float32),
            pages_per_block=_PAGES_PER_BLOCK,
        )
    with pytest.raises(ValueError, match="conv_window"):
        _write_states(
            pool,
            blocks,
            window[:1],
            jnp.zeros((2, 2), jnp.int32),
            pages_per_block=_PAGES_PER_BLOCK,
        )


def test_state_larger_than_a_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="state block has"):
        _read_states(
            _random_pool(jnp.float8_e4m3fn),
            jnp.asarray([1], jnp.int32),
            pages_per_block=1,
            window_shape=_WINDOW,
            window_dtype=jnp.bfloat16,
            context_len=2,
        )


def test_pages_per_block_requires_whole_pages_of_a_5d_pool() -> None:
    assert _pages_per_block(_POOL_SHAPE, 48, is_lbhnc=False) == 3
    assert _pages_per_block((18, 4, 32, 4, 16), 48, is_lbhnc=True) == 3
    with pytest.raises(ValueError, match="multiple"):
        _pages_per_block(_POOL_SHAPE, 24)
    with pytest.raises(NotImplementedError, match="5-D"):
        _pages_per_block((6, 2, 16, 1, 4, 128), 32)


def _tail(rows, history, chunks, ends):
    """Reference: per request, the last len(history[r]) rows of history[r]
    followed by the first ends[r] rows of its chunk."""
    out = []
    for r, (start, _) in enumerate(chunks):
        seq = torch.cat([history[r], rows[start : start + ends[r]]])
        out.append(seq[-history.shape[1] :])
    return torch.stack(out)


@pytest.mark.parametrize("ends_kind", ["query_lens", "checkpoint_offsets"])
def test_ragged_tail_is_last_history_rows_then_chunk_prefix(ends_kind) -> None:
    torch.manual_seed(0)
    # Chunks of 5, 0, 1 and 7 tokens, then padding up to 16 tokens.
    chunks = [(0, 5), (5, 5), (5, 6), (6, 13)]
    query_start_loc = torch.tensor([0, 5, 5, 6, 13])
    rows = torch.randn(16, 3)
    history = torch.randn(4, 4, 3)
    query_lens = torch.tensor([5, 0, 1, 7])
    ends = query_lens if ends_kind == "query_lens" else torch.tensor([2, 0, 0, 6])
    torch.testing.assert_close(
        ragged_tail(rows, history, query_start_loc, ends),
        _tail(rows, history, chunks, ends.tolist()),
        rtol=0,
        atol=0,
    )


def test_ple_state_ops_trace_with_symbolic_fake_tensors() -> None:
    """Static jax_op registrations trace cleanly under symbolic FakeTensorMode."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        num_reqs = shape_env.create_unbacked_symint()
        pool = torch.empty(_POOL_SHAPE, dtype=torch.bfloat16)
        block_ids = torch.empty((num_reqs,), dtype=torch.int32)
        window, context = read_ple_state(
            pool,
            block_ids,
            block_size=48,
            is_lbhnc=False,
            window_shape=_WINDOW,
            window_dtype=torch.bfloat16,
            context_len=2,
        )
        assert window.shape == (num_reqs, *_WINDOW)
        assert window.dtype == torch.bfloat16
        assert context.shape == (num_reqs, 2)
        assert context.dtype == torch.int32
        write_ple_state(
            pool,
            block_ids,
            window,
            context,
            block_size=48,
            is_lbhnc=False,
        )


@_POOL_DTYPES
def test_write_skips_non_positive_blocks_and_protects_block0(pool_dtype) -> None:
    """block_id <= 0 (-1 inactive or 0 null) must never overwrite block 0."""
    pool = jnp.ones((8, 48, 1, 4, 128), dtype=pool_dtype)
    block0_before = np.asarray(pool[:1].view(jnp.uint8)).copy()
    window = jnp.full((3, *_WINDOW), 2.0, dtype=jnp.bfloat16)
    context = jnp.asarray([[5, 6], [7, 8], [9, 10]], jnp.int32)
    blocks = jnp.asarray([2, -1, 0], jnp.int32)
    new_pool = _write_states(pool, blocks, window, context, pages_per_block=1)
    np.testing.assert_array_equal(
        np.asarray(new_pool[:1].view(jnp.uint8)), block0_before
    )
    got_w, got_c = _read_states(
        new_pool,
        jnp.asarray([2], jnp.int32),
        pages_per_block=1,
        window_shape=_WINDOW,
        window_dtype=jnp.bfloat16,
        context_len=2,
    )
    np.testing.assert_array_equal(
        np.asarray(got_w, np.float32), np.asarray(window[:1], np.float32)
    )
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(context[:1]))


def test_ple_ngram_ids_masks_negative_draft_tokens_to_eos() -> None:
    from vllm_torchtpu.models.qwen4_exp.ple import ngram_ids

    eos = 2
    mult = torch.tensor([17, 31, 53], dtype=torch.long)
    vsizes = torch.tensor([97, 101, 103, 107], dtype=torch.long)
    offsets = torch.tensor([0, 97, 198, 301], dtype=torch.long)
    req = torch.tensor([0, 0, 0], dtype=torch.int32)
    cpos = torch.tensor([0, 1, 2], dtype=torch.int32)
    ids_neg = ngram_ids(
        torch.tensor([10, -1, 12], dtype=torch.int32),
        torch.tensor([[-1, 7]], dtype=torch.int32),
        req,
        cpos,
        mult,
        vsizes,
        offsets,
        eos_token_id=eos,
        heads_per_ngram=2,
    )
    ids_eos = ngram_ids(
        torch.tensor([10, eos, 12], dtype=torch.int32),
        torch.tensor([[eos, 7]], dtype=torch.int32),
        req,
        cpos,
        mult,
        vsizes,
        offsets,
        eos_token_id=eos,
        heads_per_ngram=2,
    )
    torch.testing.assert_close(ids_neg, ids_eos, rtol=0, atol=0)


@pytest.mark.parametrize(
    (
        "tp",
        "num_kv_heads",
        "head_dim",
        "cache_dtype",
        "window_dtype",
        "block_size",
        "should_fail",
    ),
    [
        (8, 2, 256, "fp8", torch.bfloat16, 208, True),
        (8, 2, 256, "fp8", torch.bfloat16, 384, False),
        (8, 2, 256, "auto", torch.bfloat16, 208, False),
        (2, 2, 256, "fp8", torch.bfloat16, 832, False),
        (2, 2, 256, "auto", torch.bfloat16, 832, False),
        (8, 2, 256, "auto", torch.float32, 208, True),
        (2, 4, 64, "auto", torch.bfloat16, 256, True),
        (2, 4, 64, "auto", torch.bfloat16, 384, False),
    ],
)
def test_ple_state_cache_validates_resolved_block_capacity(
    tp: int,
    num_kv_heads: int,
    head_dim: int,
    cache_dtype: str,
    window_dtype: torch.dtype,
    block_size: int,
    should_fail: bool,
) -> None:
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_torchtpu.models.qwen4_exp.ple import PLEStateCache

    vc = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda pc: max(1, num_kv_heads // pc.tensor_parallel_size),
            get_head_size=lambda: head_dim,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype=cache_dtype,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    fake_gdn = SimpleNamespace(
        get_kv_cache_spec=lambda cfg: MambaSpec(
            shapes=((3, 128), (4, 16, 16)),
            dtypes=(torch.bfloat16, torch.bfloat16),
            block_size=block_size,
        )
    )
    cache = PLEStateCache(
        prefix="model.layers.1.ple",
        vllm_config=vc,  # type: ignore[arg-type]
        state_spec_layer=fake_gdn,  # type: ignore[arg-type]
        window_shape=(9, 2560 * 4),
        window_dtype=window_dtype,
        context_len=2,
        eos_token_id=2,
    )
    if should_fail:
        with pytest.raises(ValueError, match="--kv-cache-dtype auto"):
            cache.get_kv_cache_spec(vc)  # type: ignore[arg-type]
    else:
        assert isinstance(cache.get_kv_cache_spec(vc), MambaSpec)  # type: ignore[arg-type]


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("num_spec", [0, 1, 3])
def test_mamba_state_shape_includes_num_spec(tp: int, num_spec: int) -> None:
    from types import SimpleNamespace

    from transformers import Qwen4ExpTextConfig
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    from vllm_torchtpu.models.qwen4_exp.model import Qwen4ExpForCausalLM

    tc = Qwen4ExpTextConfig(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    spec = SimpleNamespace(num_speculative_tokens=num_spec) if num_spec > 0 else None
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=tc),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=spec,
    )
    got = Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vc)  # type: ignore[arg-type]
    expected = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp, 16, 48, 128, 128, 4, num_spec=num_spec
    )
    assert got == expected
    assert got[0][0] == (tc.linear_conv_kernel_dim - 1) + num_spec


@pytest.mark.parametrize("k_spec", [0, 1, 3])
def test_ple_state_cache_speculative_rollback_protocol(
    monkeypatch, k_spec: int
) -> None:
    """Drives PLEStateCache + ops/ple_state.py through prefill, multi-step verify
    with accept counts 0..K, mixed decode/verify/prefill/padding batches, prefill
    checkpoint restore, and block-0 preservation."""
    from types import SimpleNamespace

    from transformers import Qwen4ExpTextConfig
    from vllm.model_executor import parameter as vllm_parameter
    from vllm.model_executor.layers import linear as vllm_linear
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_torchtpu.models.qwen4_exp import ple as qwen4_ple
    from vllm_torchtpu.models.qwen4_exp.ple import (
        PLEInputs,
        PLEStateCache,
        Qwen4ExpPLELayer,
    )

    monkeypatch.setattr(qwen4_ple, "tensor_model_parallel_all_reduce", lambda x: x)
    for mod in (vllm_linear, vllm_parameter, vpe):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)

    torch.manual_seed(42)
    tc = Qwen4ExpTextConfig(
        vocab_size=256,
        hidden_size=16,
        hc_count=2,
        ple_layer_ids=[2],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        ple_conv_kernel_size=4,
        eos_token_id=2,
        seed=1234,
    )
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=32,
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    ple_layer = (
        Qwen4ExpPLELayer(tc, 0, None, prefix="model.layers.1.ple")
        .to(torch.bfloat16)
        .eval()
    )
    window_shape = (ple_layer.conv_state_len, tc.hidden_size * tc.hc_count)
    pool_holder = {"pool": jnp.zeros((32, 32, 1, 2, 128), dtype=jnp.bfloat16)}
    block0_init = np.asarray(pool_holder["pool"][:1].view(jnp.uint8)).copy()

    def fake_read(
        pool,
        block_ids,
        *,
        block_size,
        is_lbhnc,
        window_shape,
        window_dtype,
        context_len,
    ):
        w, c = _read_states(
            pool_holder["pool"],
            jnp.asarray(block_ids.numpy(), jnp.int32),
            pages_per_block=1,
            window_shape=window_shape,
            window_dtype=jnp.bfloat16,
            context_len=context_len,
        )
        return (
            torch.from_numpy(np.asarray(w, np.float32).copy()).to(window_dtype),
            torch.from_numpy(np.asarray(c, np.int32).copy()),
        )

    def fake_write(
        pool, block_ids, conv_window, ngram_context, *, block_size, is_lbhnc
    ):
        pool_holder["pool"] = _write_states(
            pool_holder["pool"],
            jnp.asarray(block_ids.numpy(), jnp.int32),
            jnp.asarray(conv_window.float().numpy(), jnp.bfloat16),
            jnp.asarray(ngram_context.numpy(), jnp.int32),
            pages_per_block=1,
        )

    monkeypatch.setattr(qwen4_ple, "read_ple_state", fake_read)
    monkeypatch.setattr(qwen4_ple, "write_ple_state", fake_write)
    current_md: dict[str, object] = {}
    monkeypatch.setattr(
        qwen4_ple,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=current_md),
    )

    cache = PLEStateCache(
        prefix="model.layers.1.ple",
        vllm_config=vc,  # type: ignore[arg-type]
        state_spec_layer=SimpleNamespace(),  # type: ignore[arg-type]
        window_shape=window_shape,
        window_dtype=torch.bfloat16,
        context_len=tc.ngram_size - 1,
        eos_token_id=tc.eos_token_id,
    )
    cache.kv_cache = (torch.ones(1),)

    rng = torch.Generator().manual_seed(9)
    width = tc.hidden_size * tc.hc_count
    full_ids = [
        torch.randint(3, tc.vocab_size, (26,), generator=rng, dtype=torch.int32)
        for _ in range(2)
    ]
    full_h = [
        torch.randn(26, width, generator=rng, dtype=torch.bfloat16) for _ in range(2)
    ]
    ref_out = []
    for r in range(2):
        with torch.no_grad():
            ref_out.append(
                ple_layer(
                    full_h[r],
                    PLEInputs(
                        full_ids[r],
                        torch.tensor([0, 26], dtype=torch.int32),
                        torch.full((1, tc.ngram_size - 1), 2, dtype=torch.int32),
                    ),
                    torch.zeros(1, *window_shape, dtype=torch.bfloat16),
                ).output
            )

    read_offsets = torch.zeros(32, dtype=torch.int32)
    base_slots = [1, 8]
    done = [0, 0]

    def run_step(req_chunks, *, num_windowed: int, prefill_ckpt=None):
        ids_cat = torch.cat([c[1] for c in req_chunks])
        h_cat = torch.cat([c[2] for c in req_chunks])
        qsl_list, seq_lens_list, state_idx_list, ckpt_rows = [0], [], [], []
        for r, tok, _ in req_chunks:
            qsl_list.append(qsl_list[-1] + len(tok))
            seq_lens_list.append(done[r] + len(tok))
            state_idx_list.append(base_slots[r])
            ckpt_rows.append([base_slots[r] + s for s in range(k_spec + 1)])
        current_md["model.layers.1.ple"] = SimpleNamespace(
            query_start_loc=torch.tensor(qsl_list, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens_list, dtype=torch.int32),
            mamba_state_indices=torch.tensor(state_idx_list, dtype=torch.int32),
            mamba_slot_read_offsets=read_offsets if k_spec > 0 else None,
            mamba_ckpt_indices=torch.tensor(ckpt_rows, dtype=torch.int32)
            if k_spec > 0
            else None,
            mamba_request_distribution=torch.tensor(
                [num_windowed, num_windowed, len(req_chunks)], dtype=torch.int32
            )
            if k_spec > 0
            else None,
            mamba_prefill_ckpt_offsets=torch.tensor(prefill_ckpt[0], dtype=torch.int32)
            if prefill_ckpt is not None
            else None,
            mamba_prefill_ckpt_indices=torch.tensor(prefill_ckpt[1], dtype=torch.int32)
            if prefill_ckpt is not None
            else None,
        )
        with torch.no_grad():
            ple_in, prev_w = cache.read(h_cat, ids_cat)
            out = ple_layer(h_cat, ple_in, prev_w)
            cache.write(ple_in, prev_w, out)
        return [
            out.output[qsl_list[i] : qsl_list[i + 1]] for i in range(len(req_chunks))
        ]

    outs = run_step(
        [(0, full_ids[0][:10], full_h[0][:10]), (1, full_ids[1][:8], full_h[1][:8])],
        num_windowed=0,
        prefill_ckpt=([6, 0], [20, -1]),
    )
    torch.testing.assert_close(outs[0], ref_out[0][:10], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[1], ref_out[1][:8], rtol=1e-5, atol=1e-5)
    done = [10, 8]

    schedule = [0, min(1, k_spec), min(2, k_spec), k_spec, 0, min(1, k_spec)]
    for step_idx, acc0 in enumerate(schedule):
        acc1 = schedule[(step_idx + 2) % len(schedule)]
        chunks = []
        accepted = [acc0, acc1]
        for r in range(2):
            pos = done[r]
            win = min(k_spec + 1, 26 - pos)
            if win <= 0:
                continue
            toks = full_ids[r][pos : pos + win].clone()
            hs = full_h[r][pos : pos + win].clone()
            n_good = min(accepted[r] + 1, win)
            if n_good < win:
                toks[n_good:] = 250
                hs[n_good:] = -7.0
            chunks.append((r, toks, hs))
        if not chunks:
            break
        outs = run_step(chunks, num_windowed=len(chunks))
        for (r, toks, _), out_r in zip(chunks, outs):
            pos = done[r]
            n_good = min(accepted[r] + 1, len(toks))
            torch.testing.assert_close(
                out_r[:n_good], ref_out[r][pos : pos + n_good], rtol=1e-5, atol=1e-5
            )
            done[r] += n_good
            read_offsets[base_slots[r]] = n_good - 1

    # Verify prefill checkpoint at block 20 (offset 6 of req 0).
    ckpt_w, ckpt_c = _read_states(
        pool_holder["pool"],
        jnp.asarray([20], jnp.int32),
        pages_per_block=1,
        window_shape=window_shape,
        window_dtype=jnp.bfloat16,
        context_len=tc.ngram_size - 1,
    )
    with torch.no_grad():
        resumed = ple_layer(
            full_h[0][6:10],
            PLEInputs(
                full_ids[0][6:10],
                torch.tensor([0, 4], dtype=torch.int32),
                torch.from_numpy(np.asarray(ckpt_c).copy()),
            ),
            torch.from_numpy(np.asarray(ckpt_w.astype(jnp.float32)).copy()).to(
                torch.bfloat16
            ),
        ).output
    torch.testing.assert_close(resumed, ref_out[0][6:10], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(pool_holder["pool"][:1].view(jnp.uint8)), block0_init
    )
