# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for https://github.com/vllm-project/vllm/issues/31624.

Loading a fused ModelOpt/Llama4 MoE checkpoint transposes and chunks the
gate_up_proj weight to match the parameter layout. Without an explicit
`.contiguous()` call, that leaves the tensor as a non-contiguous view, and
copying a non-contiguous CPU tensor to GPU is dramatically slower (seconds
per weight instead of milliseconds), which is what made these checkpoints
take 5+ minutes to load.

These tests exercise the real `Llama4Model.load_moe_expert_weights` weight
transformation with a lightweight, mocked module tree (no GPU or real model
weights required) and assert that every tensor handed to `weight_loader` is
contiguous.
"""

import types

import pytest
import torch

from vllm.model_executor.models import llama4

pytestmark = pytest.mark.cpu_test


def _make_model(monkeypatch: pytest.MonkeyPatch) -> llama4.Llama4Model:
    # Bypass Llama4Model.__init__ (which builds a full transformer stack) and
    # only set what load_moe_expert_weights actually touches.
    model = llama4.Llama4Model.__new__(llama4.Llama4Model)
    model.layers = [
        types.SimpleNamespace(
            feed_forward=types.SimpleNamespace(
                experts=types.SimpleNamespace(expert_map=None)
            )
        )
    ]
    monkeypatch.setattr(llama4, "is_pp_missing_parameter", lambda name, model: False)
    return model


def test_gate_up_proj_weight_is_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    """gate_up_proj is transposed then chunked into w1/w3 halves; the shard
    selection used to leave a non-contiguous slice of the chunked tuple."""
    model = _make_model(monkeypatch)

    num_experts, hidden_in, hidden_out = 2, 8, 16
    loaded_weight = torch.randn(num_experts, hidden_in, 2 * hidden_out)
    name = "model.layers.0.feed_forward.experts.gate_up_proj"

    expert_params_mapping = [
        ("experts.w13_", f"experts.{expert_id}.gate_up_proj.", expert_id, shard_id)
        for expert_id in range(num_experts)
        for shard_id in ("w1", "w3")
    ]

    captured: list[torch.Tensor] = []

    def capture_weight_loader(param, loaded_weight, name, shard_id, expert_id):
        captured.append(loaded_weight)

    params_dict = {
        "model.layers.0.feed_forward.experts.w13_weight": types.SimpleNamespace(
            weight_loader=capture_weight_loader
        ),
    }

    loaded = model.load_moe_expert_weights(
        name=name,
        loaded_weight=loaded_weight,
        params_dict=params_dict,
        loaded_params=set(),
        expert_params_mapping=expert_params_mapping,
        fused=True,
    )

    assert loaded is True
    assert len(captured) == num_experts * 2
    for weight in captured:
        assert weight.is_contiguous()


def test_down_proj_weight_is_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    """down_proj isn't chunked, but the transpose alone already produces a
    non-contiguous view; that path must be materialized too."""
    model = _make_model(monkeypatch)

    num_experts, hidden_in, hidden_out = 2, 16, 8
    loaded_weight = torch.randn(num_experts, hidden_in, hidden_out)
    name = "model.layers.0.feed_forward.experts.down_proj"

    expert_params_mapping = [
        ("experts.w2_", f"experts.{expert_id}.down_proj.", expert_id, "w2")
        for expert_id in range(num_experts)
    ]

    captured: list[torch.Tensor] = []

    def capture_weight_loader(param, loaded_weight, name, shard_id, expert_id):
        captured.append(loaded_weight)

    params_dict = {
        "model.layers.0.feed_forward.experts.w2_weight": types.SimpleNamespace(
            weight_loader=capture_weight_loader
        ),
    }

    loaded = model.load_moe_expert_weights(
        name=name,
        loaded_weight=loaded_weight,
        params_dict=params_dict,
        loaded_params=set(),
        expert_params_mapping=expert_params_mapping,
        fused=True,
    )

    assert loaded is True
    assert len(captured) == num_experts
    for weight in captured:
        assert weight.is_contiguous()
