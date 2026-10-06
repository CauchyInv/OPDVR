# Copyright 2026 OPDVR contributors
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

import copy

import pytest
import torch

from verl.trainer.ppo.disturbed_updates import (
    adamw_displacement,
    choose_adam_candidate,
    coarse_forward_kl,
    repair_token_loss,
)


@pytest.mark.parametrize("history", [0, 1, 7])
@pytest.mark.parametrize("decay", [0.0, 0.01])
@pytest.mark.parametrize("foreach", [False, True])
def test_adamw_exact_candidate_without_state_mutation(history, decay, foreach):
    torch.manual_seed(12)
    p = torch.nn.Parameter(torch.randn(100))
    opt = torch.optim.AdamW([p], lr=1e-6, weight_decay=decay, foreach=foreach)
    for _ in range(history):
        p.grad = torch.randn_like(p)
        opt.step()
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    state = copy.deepcopy(opt.state.get(p, {}))
    predicted = adamw_displacement(p, p.grad, opt.state.get(p, {}), opt.param_groups[0])
    assert torch.equal(p, before)
    for key in state:
        assert torch.equal(state[key], opt.state[p][key])
    opt.step()
    torch.testing.assert_close(p - before, predicted, atol=1e-8, rtol=0.01)


def test_repair_direction_and_inactive_boundary():
    u = torch.tensor([-2.2, -1.8, -1.8, -2.2, -2.0], requires_grad=True)
    a = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
    loss, _ = repair_token_loss(u, torch.full_like(u, -2), a, torch.ones(5).bool(), 0, 0, 0.1, 1)
    loss.sum().backward()
    assert u.grad[0] < 0  # Descent increases positive-A token probability.
    assert u.grad[1] > 0
    assert torch.equal(u.grad[2:], torch.zeros(3))


def test_repair_reliability_and_padding():
    u = torch.tensor([-3.0, -3.0, -3.0], requires_grad=True)
    loss, _ = repair_token_loss(
        u, torch.full_like(u, -2), torch.tensor([1.0, 0.01, 1.0]), torch.tensor([True, True, False]), 0.1, 0, 0.1, 1
    )
    loss.sum().backward()
    assert u.grad[0] < 0
    assert torch.equal(u.grad[1:], torch.zeros(2))


def test_adam_budget_fallback_and_selected_improvement():
    assert choose_adam_candidate([(0, 1, 2), (0.1, 0.95, 1), (0.2, 0.8, 0)], 0.9) == 1
    assert choose_adam_candidate([(0, -1, 2), (0.1, 1, 1)], 0.9) == 0
    assert choose_adam_candidate([(0, 1, -2), (0.1, 1, -1)], 0.9) == 0


def test_coarse_kl_matches_explicit_rest_bucket():
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    q = torch.tensor([[0.3, 0.4, 0.2, 0.1]])
    result = coarse_forward_kl(p[:, :2].log(), q[:, :2].log())
    expected = (
        0.3 * torch.log(torch.tensor(3.0))
        + 0.4 * torch.log(torch.tensor(2.0))
        + 0.3 * torch.log(torch.tensor(0.3 / 0.7))
    )
    torch.testing.assert_close(result[0], expected)
    result.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_coarse_kl_identical_and_full_vocab():
    logp = torch.log_softmax(torch.randn(3, 5), -1)
    torch.testing.assert_close(coarse_forward_kl(logp, logp), torch.zeros(3), atol=1e-6, rtol=0)


@pytest.mark.parametrize("method", ["real_repair", "adam_budget", "distribution_repair", "sign_hybrid"])
@pytest.mark.parametrize("history", [0, 3])
def test_complete_update_with_toy_actor(monkeypatch, method, history):
    """Exercise full control flow without GPU; distributed sharding uses smoke tests."""
    from types import SimpleNamespace

    import torch.distributed.fsdp as fsdp

    from verl import DataProto
    from verl.trainer.ppo.disturbed_opd import DisturbedOPDConfig
    from verl.trainer.ppo.disturbed_updates import update_disturbed_v5

    class ToyShard(torch.nn.Module):
        sharding_strategy = fsdp.ShardingStrategy.FULL_SHARD

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(5, 7))

    monkeypatch.setattr(fsdp, "FullyShardedDataParallel", ToyShard)
    monkeypatch.setattr("verl.utils.device.get_device_id", lambda: "cpu")
    torch.manual_seed(27)
    model = ToyShard()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(history):
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()

    class ToyActor:
        actor_module = model
        actor_optimizer = optimizer
        config = SimpleNamespace(
            ppo_epochs=1, use_kl_loss=False, entropy_coeff=0, use_dynamic_bsz=False, ppo_micro_batch_size_per_gpu=1
        )

        @staticmethod
        def _local_view(x):
            return x

        @staticmethod
        def _disturbed_advantages(inputs, alpha):
            return inputs["advantages"].detach()

        @staticmethod
        def _clip_grad_norm():
            return torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        @staticmethod
        def _forward_micro_batch(inputs, temperature, top_k=0, student_top_k_ids=None):
            logits = model.weight[inputs["input_ids"][:, -3:]] / temperature
            logp = logits.log_softmax(-1)
            sampled = logp.gather(-1, inputs["responses"].unsqueeze(-1)).squeeze(-1)
            if top_k:
                return None, sampled, student_top_k_ids, logp.gather(-1, student_top_k_ids)
            return None, sampled, None, None

    tensors = {
        "input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "responses": torch.tensor([[1, 2, 3], [2, 3, 4]]),
        "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "attention_mask": torch.ones(2, 3),
        "position_ids": torch.arange(3).repeat(2, 1),
        "old_log_probs": torch.zeros(2, 3),
        "advantages": torch.tensor([[0.2, -0.5, 0.7], [0.4, 0.1, 0.0]]),
        "teacher_top_k_ids": torch.tensor([[[0, 1]] * 3] * 2),
        "teacher_top_k_log_probs": torch.tensor([[[0.2, 0.3]] * 3] * 2).log(),
    }
    data = DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0, "global_steps": 1})
    cfg = DisturbedOPDConfig(method=method, weight_mode="budget", kappa=0.5, teacher_topk=2)
    metrics = update_disturbed_v5(ToyActor(), data, cfg)
    assert int(optimizer.state[model.weight]["step"]) == history + 1
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert "audit/surrogate_change" in metrics
    if method == "adam_budget":
        assert metrics["adam_budget/simulator_relative_error"] < 1e-3
    if method == "real_repair":
        assert metrics["repair/norm_fraction"] <= cfg.repair_norm_fraction + 1e-7
    if method == "sign_hybrid":
        assert 0 <= metrics["hybrid/up_selected_fraction"] <= 1
        assert 0 <= metrics["hybrid/down_selected_fraction"] <= 1
        assert metrics["disturbed/selected_share_negative"] >= 0
