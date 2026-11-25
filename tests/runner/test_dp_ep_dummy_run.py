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
"""Tests for DP+EP _dummy_run metadata/logits buckets and _prepare_inputs zeroing."""

import contextlib
import types
from types import SimpleNamespace

import pytest
import torch
from runner_input_test_utils import (
    _fake_tpu_events,
    _make_runner,
    _scheduler_output,
)
from vllm import forward_context as fc_mod
from vllm.forward_context import get_forward_context

from vllm_torchtpu.layers.core.sequence_layout import (
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_should_zero_trailing_positions_under_all_sequence_layout(monkeypatch):
    _fake_tpu_events(monkeypatch)
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    runner.positions_cpu.fill_(777)

    so = _scheduler_output([3])
    prepare_inputs = TPUModelRunner._prepare_inputs  # noqa: SLF001 -- runner step entry
    prepare_inputs(runner, so, start_index=0, num_decode_reqs=0)

    assert runner.position_ids.tolist() == [0, 1, 2, 0, 0, 0, 0, 0]

    # Under PARTIAL layout, positions_cpu[total:global_padded] belongs to packed PCP
    # rank slices and must not be overwritten by the ALL-layout zeroing path.
    partial_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    partial_plan = SequenceLayoutPlan(
        descriptor=SequenceLayoutDescriptor(kind=SequenceLayoutKind.PARTIAL),
        token_slice=slice(4, 8),
        global_num_tokens=3,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=4,
    )
    partial_runner.sequence_layout_planner = SimpleNamespace(
        reserve_host_token_capacity=lambda *_args, **_kwargs: None,
        prepare_real=lambda **_kwargs: partial_plan,
    )
    partial_runner.positions_cpu[4:8] = torch.tensor([10, 11, 0, 0], dtype=torch.int32)
    prepare_inputs(partial_runner, so, start_index=0, num_decode_reqs=0)
    assert partial_runner.position_ids.tolist() == [10, 11, 0, 0]

    # Under pcp_manager (layout_plan is None, zero_pad_tail is False),
    # _prepare_inputs must guard tail zeroing via zero_pad_tail rather than
    # dereferencing layout_plan.kind.
    pcp_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[3],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=4,
        block_size=16,
    )
    pcp_plan = SimpleNamespace(
        local_num_reqs=1,
        global_num_tokens=3,
        global_padded_num_tokens=8,
        local_num_tokens=2,
        local_padded_num_tokens=4,
        token_slice=slice(4, 8),
        logits_indices_cpu=None,
        request_distribution_counts=lambda decode, total: (decode, total),
    )
    pcp_runner.pcp_manager = SimpleNamespace(
        num_reqs_in_chunk=lambda q_lens, _seq_lens, capacity: min(
            capacity, len(q_lens)
        ),
        reserve_host_token_capacity=lambda *_args, **_kwargs: None,
        prepare_real=lambda **_kwargs: None,
        sharding_plan=pcp_plan,
    )
    pcp_runner.sequence_layout_planner = None
    pcp_runner.positions_cpu[4:8] = torch.tensor([20, 21, 0, 0], dtype=torch.int32)
    prepare_inputs(pcp_runner, so, start_index=0, num_decode_reqs=0)
    assert pcp_runner.position_ids.tolist() == [20, 21, 0, 0]


def test_should_populate_dp_metadata_and_clamp_idle_logits_indices_in_dummy_run(
    monkeypatch,
):
    _fake_tpu_events(monkeypatch)
    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        tpu_runner_mod, "synchronize_tensors", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        fc_mod.current_platform,
        "set_additional_forward_context",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        fc_mod,
        "coordinate_batch_across_dp",
        lambda **_kwargs: pytest.fail("unexpected coordinate_batch_across_dp call"),
    )

    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32, 64],
        pcp_size=1,
        dp_size=4,
        dp_rank=2,
        enable_ep=True,
        max_num_reqs=48,
        block_size=16,
    )
    req_paddings_fn = tpu_runner_mod._get_req_paddings  # noqa: SLF001 -- bucket builder
    runner.num_reqs_paddings = req_paddings_fn(
        tpu_runner_mod.MIN_NUM_SEQS, runner.max_num_reqs
    )
    runner._profile_inputs = {}  # noqa: SLF001 -- runner profile state
    runner.vocab_size = 128
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: 8)
    runner._hidden_states_dtype = torch.bfloat16  # noqa: SLF001 -- precompile dtype
    runner._precompile_timed = lambda *_args: contextlib.nullcontext()  # noqa: SLF001
    profile_fn = TPUModelRunner._profile_input  # noqa: SLF001 -- bind runner profiler
    runner._profile_input = types.MethodType(profile_fn, runner)  # noqa: SLF001 -- bind
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    dummy_run = TPUModelRunner._dummy_run  # noqa: SLF001 -- runner dummy step under test
    precompile_logits = TPUModelRunner._precompile_compute_selected_logits  # noqa: SLF001

    seen_dp_metadata = []
    seen_logits_shapes = []
    seen_logits_pairs = []

    def fake_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del positions, inputs_embeds, intermediate_tensors
        seen_dp_metadata.append(get_forward_context().dp_metadata)
        return torch.zeros((input_ids.shape[0], 8), dtype=torch.bfloat16), None

    def fake_compute_selected_logits(hidden_states, logits_indices):
        seen_logits_shapes.append(tuple(logits_indices.shape))
        seen_logits_pairs.append((hidden_states.shape[0], logits_indices.shape[0]))
        return torch.zeros((logits_indices.shape[0], 16), dtype=torch.bfloat16)

    runner.forward_model = fake_forward_model
    runner.compute_selected_logits = fake_compute_selected_logits

    precompile_logits(runner)
    precompiled_grid = set(seen_logits_pairs)
    seen_logits_shapes.clear()
    seen_logits_pairs.clear()

    # 1. Startup warmup / profiling (dp_lockstep=False) sets DPMetadata on DP+EP.
    elapsed = dummy_run(
        runner,
        num_tokens=16,
        num_reqs=48,
        num_blocks=4,
        dp_lockstep=False,
        profile_prefix=32,
    )
    assert isinstance(elapsed, float)
    assert seen_dp_metadata[-1] is not None
    assert seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [16, 16, 16, 16]
    assert seen_logits_shapes == []

    # 2. Idle lockstep (dp_lockstep=True) clamps and pads logits to the precompile grid:
    # - num_tokens=16 < num_reqs=48 -> min=16 -> padded 16
    # - num_tokens=64 > num_reqs=24 (unpadded) -> min=24 -> padded 32 (not 24 or 48)
    # - num_tokens=64, num_reqs=64 > max_num_reqs=48 -> capped at max_num_reqs 48
    for num_tokens, num_reqs, expected_logits_reqs in (
        (16, 48, 16),
        (64, 24, 32),
        (64, 64, 48),
    ):
        res = dummy_run(
            runner,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_blocks=4,
            dp_lockstep=True,
        )
        assert res is None
        assert seen_dp_metadata[-1] is not None
        assert (
            seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [num_tokens] * 4
        )
        assert seen_logits_shapes[-1] == (expected_logits_reqs,)
        assert seen_logits_pairs[-1] in precompiled_grid

    # 2b. When DP vocab logits sharding is active, idle lockstep preserves the
    # cross-engine row count from _dp_vocab_logits_rows().
    runner._dp_vocab_logits = True  # noqa: SLF001 -- exercise DP vocab parallel branch
    runner._dp_vocab_logits_rows = lambda: 32  # noqa: SLF001 -- stub DP row bucket
    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=48,
        num_blocks=4,
        dp_lockstep=True,
    )
    assert seen_logits_shapes[-1] == (32,)

    # 3. Non-lockstep configuration (dp_size=1, enable_ep=False) keeps dp_metadata=None.
    non_dp_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32],
        pcp_size=1,
        dp_size=1,
        dp_rank=0,
        enable_ep=False,
        max_num_reqs=16,
        block_size=16,
    )
    non_dp_runner.maybe_select_dummy_loras = runner.maybe_select_dummy_loras
    non_dp_runner.forward_model = fake_forward_model
    non_dp_runner.compute_selected_logits = fake_compute_selected_logits

    dummy_run(
        non_dp_runner,
        num_tokens=16,
        num_reqs=16,
        num_blocks=4,
        dp_lockstep=False,
    )
    assert seen_dp_metadata[-1] is None


@pytest.mark.tpu_test
def test_dp_ep_dummy_run_tpu_positions_and_logits(device, monkeypatch):
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[128, 256],
        scheduled_tokens=[3, 2],
        token_paddings=[8, 16, 32],
        pcp_size=1,
        dp_size=2,
        dp_rank=1,
        enable_ep=True,
        max_num_reqs=32,
        block_size=64,
    )
    runner.device = device
    runner.positions_cpu.fill_(999)

    so = _scheduler_output([3, 2])
    prepare_inputs = TPUModelRunner._prepare_inputs  # noqa: SLF001 -- runner step entry
    prepare_inputs(runner, so, start_index=0, num_decode_reqs=0)
    assert runner.position_ids.device.type == "tpu"
    assert runner.position_ids.cpu().tolist() == [0, 1, 2, 0, 1, 0, 0, 0]

    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    dummy_run = TPUModelRunner._dummy_run  # noqa: SLF001 -- runner dummy step under test
    seen_dp_dev_types = []
    seen_logits_shapes = []

    def tpu_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del positions, inputs_embeds, intermediate_tensors
        dp_md = get_forward_context().dp_metadata
        assert dp_md is not None
        seen_dp_dev_types.append(dp_md.num_tokens_across_dp_cpu.device.type)
        assert dp_md.num_tokens_across_dp_cpu.cpu().tolist() == [16, 16]
        return torch.zeros(
            (input_ids.shape[0], 8), dtype=torch.bfloat16, device=device
        ), None

    def tpu_compute_selected_logits(hidden_states, logits_indices):
        del hidden_states
        seen_logits_shapes.append(
            (tuple(logits_indices.shape), logits_indices.device.type)
        )
        return torch.zeros(
            (logits_indices.shape[0], 16), dtype=torch.bfloat16, device=device
        )

    runner.forward_model = tpu_forward_model
    runner.compute_selected_logits = tpu_compute_selected_logits

    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=32,
        num_blocks=4,
        dp_lockstep=True,
    )
    assert seen_dp_dev_types == ["tpu"]
    assert seen_logits_shapes == [((16,), "tpu")]
