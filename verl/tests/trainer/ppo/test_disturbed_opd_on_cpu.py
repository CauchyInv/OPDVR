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
"""CPU tests for the probe-and-reweight OPD primitives."""

import math

import pytest
import torch

from verl.trainer.ppo.disturbed_opd import (
    DIAG_KEYS,
    DisturbedOPDConfig,
    abs_histogram,
    advantage_from_logratio,
    attenuate_opposite_weights,
    budget_lambda,
    continuous_rank_scores,
    continuous_raw_severity,
    continuous_scores_from_severity,
    entropy_mask,
    lambda_from_sums,
    position_quartile,
    probe_delta,
    probe_mask,
    random_mask,
    selection_diagnostics,
    stratified_random_masks,
    stratified_shuffle_scores,
    threshold_from_histogram,
    token_weights,
    token_weights_budget,
    top_fraction_mask,
    topk_tail_forward_kl,
    uniform_weights,
    weighted_policy_sum,
)


def test_alpha_one_is_identical_to_baseline_reward():
    log_ratio = torch.tensor([-3.5, -0.25, 0.0, 0.25, 7.5])
    advantage = advantage_from_logratio(log_ratio, alpha=1.0)
    assert torch.equal(advantage, log_ratio)


def test_alpha_below_one_has_documented_floor_and_is_monotonic():
    alpha = 0.2
    log_ratio = torch.tensor([-40.0, -5.0, 0.0, 5.0, 40.0])
    advantage = advantage_from_logratio(log_ratio, alpha)
    floor = math.log1p(-alpha)
    # A very negative ratio saturates exactly onto the floor in fp32.
    assert torch.all(advantage >= floor)
    assert advantage[0].item() == pytest.approx(floor, abs=1e-6)
    assert torch.all(advantage[1:] > advantage[:-1])
    # A large positive ratio approaches log(alpha) + ratio.
    assert advantage[-1].item() == pytest.approx(math.log(alpha) + 40.0, abs=1e-4)


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        advantage_from_logratio(torch.zeros(3), alpha=0.0)
    with pytest.raises(ValueError):
        advantage_from_logratio(torch.zeros(3), alpha=1.5)


def test_topk_tail_forward_kl_is_zero_for_matching_coarsened_distributions():
    teacher = torch.log(torch.tensor([[[0.5, 0.25]]]))
    student = teacher.clone()
    value = topk_tail_forward_kl(teacher, student)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-7)


def test_topk_tail_forward_kl_matches_explicit_three_bin_kl():
    teacher_prob = torch.tensor([0.5, 0.25])
    student_prob = torch.tensor([0.4, 0.1])
    actual = topk_tail_forward_kl(teacher_prob.log(), student_prob.log())
    teacher_bins = torch.tensor([0.5, 0.25, 0.25])
    student_bins = torch.tensor([0.4, 0.1, 0.5])
    expected = (teacher_bins * (teacher_bins.log() - student_bins.log())).sum()
    assert actual.item() == pytest.approx(expected.item(), abs=1e-7)


def test_distribution_kl_audit_requires_an_output_directory():
    with pytest.raises(ValueError):
        DisturbedOPDConfig(audit_distribution_kl_topk=32)
    DisturbedOPDConfig(audit_dir="/tmp/audit", audit_distribution_kl_topk=32)


def test_probe_delta_follows_adam_descent_sign():
    """delta must oppose the gradient, exactly as optimizer.step() does.

    A flipped sign here would invert the selected token set without any error,
    so this is asserted directly rather than only end-to-end.
    """
    grad = torch.tensor([2e-5, -4e-5, 0.0, 1e-3])
    delta = probe_delta(grad, None, next_step=1, eta_probe=1e-4, beta2=0.999, eps=1e-8)
    assert torch.allclose(torch.sign(delta), -torch.sign(grad))
    nonzero = grad != 0
    assert torch.allclose(delta[nonzero].abs(), torch.full((int(nonzero.sum()),), 1e-4), rtol=2e-3)


def test_probe_delta_matches_a_real_adam_step_direction():
    """The probe and a genuine AdamW step must move the parameter the same way."""
    torch.manual_seed(0)
    start = torch.randn(128, dtype=torch.float64) * 0.02
    grad = torch.randn(128, dtype=torch.float64) * 2e-5

    reference = torch.nn.Parameter(start.clone())
    optimizer = torch.optim.AdamW([reference], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    reference.grad = grad.clone()
    optimizer.step()
    adam_move = (reference.data - start).sign()

    delta = probe_delta(grad, None, next_step=1, eta_probe=1e-4, beta2=0.999, eps=1e-8)
    assert torch.equal(delta.sign().to(adam_move.dtype), adam_move)


def test_probe_delta_uses_bias_corrected_second_moment():
    grad = torch.tensor([3e-5])
    v = torch.tensor([1e-9])
    beta2, eps, eta, step = 0.999, 1e-8, 1e-4, 7
    delta = probe_delta(grad, v, next_step=step, eta_probe=eta, beta2=beta2, eps=eps)
    v_new = beta2 * v + (1 - beta2) * grad**2
    v_hat = v_new / (1 - beta2**step)
    assert delta.item() == pytest.approx((-eta * grad / (v_hat.sqrt() + eps)).item(), rel=1e-6)


def test_probe_delta_rejects_bad_clock():
    with pytest.raises(ValueError):
        probe_delta(torch.zeros(2), None, next_step=0, eta_probe=1e-4, beta2=0.999, eps=1e-8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_clone_copy_restore_is_bitwise_exact(dtype):
    """The probe restores parameters by copying a same-dtype backup, not by subtracting."""
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(512, dtype=torch.float32).to(dtype) * 0.02)
    backup = param.data.clone()
    delta = (torch.randn(512, dtype=torch.float32) * 1e-4).to(dtype)
    param.data.add_(delta)
    param.data.copy_(backup)
    assert torch.equal(param.data, backup)
    assert param.data.dtype == dtype


def test_probe_does_not_mutate_optimizer_state():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(64) * 0.02)
    optimizer = torch.optim.AdamW([param], lr=1e-6, betas=(0.9, 0.999), eps=1e-8)
    param.grad = torch.randn(64) * 2e-5
    optimizer.step()
    state = optimizer.state[param]
    before = {key: value.clone() for key, value in state.items() if torch.is_tensor(value)}

    param.grad = torch.randn(64) * 2e-5
    backup = param.data.clone()
    delta = probe_delta(param.grad, state["exp_avg_sq"], next_step=2, eta_probe=1e-4, beta2=0.999, eps=1e-8)
    param.data.add_(delta.to(param.dtype))
    param.data.copy_(backup)

    for key, value in before.items():
        assert torch.equal(state[key], value), f"probe mutated optimizer state {key}"


def test_first_order_gain_is_positive_on_fp64_model():
    """A real uniform-weight probe must improve the objective it descends."""
    torch.manual_seed(0)
    tokens, features = 24, 6
    embedding = torch.randn(tokens, features, dtype=torch.float64)
    weight = torch.nn.Parameter(torch.randn(features, dtype=torch.float64) * 0.1)
    advantages = torch.randn(tokens, dtype=torch.float64)
    mask = torch.ones(tokens, dtype=torch.bool)

    def log_probs_of(w):
        logits = embedding @ w
        return torch.log_softmax(torch.stack([logits, torch.zeros_like(logits)], dim=-1), dim=-1)[:, 0]

    base = log_probs_of(weight)
    loss = -(advantages * base).sum() / tokens
    loss.backward()
    grad = weight.grad.clone()
    delta = probe_delta(grad, None, next_step=1, eta_probe=1e-6, beta2=0.999, eps=1e-12).to(torch.float64)

    with torch.no_grad():
        shifted = log_probs_of(weight + delta)
    change = shifted - base.detach()
    gain = float((advantages * change).sum())
    # Descending -A*logpi means the parameter move must raise A*logpi overall.
    assert gain > 0, gain
    # Interference is possible per token but the aggregate is first-order exact.
    assert float((advantages * change).abs().sum()) >= abs(gain)


def test_probe_mask_finds_only_opposed_tokens():
    advantages = torch.tensor([1.0, 1.0, -2.0, -2.0, 0.5])
    delta = torch.tensor([0.1, -0.1, 0.1, -0.1, -0.9])
    mask = torch.tensor([True, True, True, True, False])
    selected = probe_mask(advantages, delta, mask, threshold=0.0)
    assert selected.tolist() == [False, True, True, False, False]


def test_probe_mask_respects_threshold_and_direction():
    advantages = torch.tensor([0.05, -3.0, 2.0])
    delta = torch.tensor([-1.0, 1.0, -1.0])
    mask = torch.ones(3, dtype=torch.bool)
    assert probe_mask(advantages, delta, mask, threshold=1.0).tolist() == [False, True, True]
    assert probe_mask(advantages, delta, mask, threshold=1.0, direction="up").tolist() == [False, False, True]
    assert probe_mask(advantages, delta, mask, threshold=1.0, direction="down").tolist() == [False, True, False]


def test_histogram_quantile_excludes_the_smallest_fraction():
    values = torch.arange(1, 101, dtype=torch.float32) / 10.0  # 0.1 .. 10.0
    mask = torch.ones_like(values, dtype=torch.bool)
    counts = abs_histogram(values, mask, bins=2048, hi=32.0)
    assert int(counts.sum()) == 100
    threshold = threshold_from_histogram(counts, quantile=0.2, hi=32.0)
    kept = int((values >= threshold).sum())
    assert kept == pytest.approx(80, abs=1)


def test_histogram_quantile_zero_keeps_everything():
    values = torch.randn(256)
    mask = torch.ones_like(values, dtype=torch.bool)
    counts = abs_histogram(values, mask, bins=512, hi=32.0)
    assert threshold_from_histogram(counts, quantile=0.0, hi=32.0) == 0.0


def test_histogram_handles_overflow_and_empty_mask():
    values = torch.tensor([100.0, -200.0])
    mask = torch.ones(2, dtype=torch.bool)
    counts = abs_histogram(values, mask, bins=16, hi=32.0)
    assert int(counts[-1]) == 2
    empty = abs_histogram(values, torch.zeros(2, dtype=torch.bool), bins=16, hi=32.0)
    assert int(empty.sum()) == 0
    assert threshold_from_histogram(empty, quantile=0.2, hi=32.0) == 0.0


def test_histogram_resolves_quantiles_near_zero():
    """Linear bins collapsed a low quantile of near-zero |A| to a zero threshold.

    Real advantages concentrate within ~1e-3 of zero, so the log-spaced grid must
    still separate the excluded fraction there.
    """
    torch.manual_seed(0)
    # Lognormal spans several decades and is dense near zero, like real |A|.
    values = torch.exp(torch.randn(20000) * 2.0 - 6.0)
    mask = torch.ones_like(values, dtype=torch.bool)
    counts = abs_histogram(values, mask, bins=2048, hi=32.0)
    assert int(counts.sum()) == values.numel()
    threshold = threshold_from_histogram(counts, quantile=0.2, hi=32.0)
    assert threshold > 0.0, threshold
    kept_fraction = float((values >= threshold).float().mean())
    assert kept_fraction == pytest.approx(0.8, abs=0.01)


def test_histogram_underflow_bin_collects_true_zeros():
    values = torch.zeros(10)
    mask = torch.ones(10, dtype=torch.bool)
    counts = abs_histogram(values, mask, bins=64, hi=32.0)
    assert int(counts[0]) == 10
    # The underflow bin must be excluded, not kept: a zero threshold would keep
    # every token and silently disable the filter.
    threshold = threshold_from_histogram(counts, quantile=0.2, hi=32.0)
    assert threshold > 0.0
    assert int((values >= threshold).sum()) == 0
    # quantile=0 still means "keep everything".
    assert threshold_from_histogram(counts, quantile=0.0, hi=32.0) == 0.0


def test_lambda_spends_a_fraction_of_available_gain():
    lam, lam_max, fell_back = lambda_from_sums(gain=2.0e-3, damage=-4.0e-4, kappa=0.5)
    assert not fell_back
    assert lam_max == pytest.approx(5.0)
    assert lam == pytest.approx(2.5)


def test_lambda_falls_back_when_gain_is_not_positive():
    for gain in (0.0, -1e-6):
        lam, _, fell_back = lambda_from_sums(gain=gain, damage=-1e-4, kappa=0.5)
        assert lam == 0.0 and fell_back


def test_lambda_falls_back_on_nonfinite_inputs():
    lam, _, fell_back = lambda_from_sums(gain=float("nan"), damage=-1e-4, kappa=0.5)
    assert lam == 0.0 and fell_back


def test_lambda_is_zero_when_nothing_was_pushed_backwards():
    lam, lam_max, fell_back = lambda_from_sums(gain=1e-3, damage=0.0, kappa=0.5)
    assert lam == 0.0 and not fell_back and lam_max == float("inf")


def test_weights_have_unit_mean_and_respect_the_cap():
    mask = torch.ones(10, dtype=torch.bool)
    selected = torch.tensor([True] * 2 + [False] * 8)
    weights = token_weights(selected, mask, lam=2.5, w_max=5.0, valid_count=10, selected_count=2)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(weights[0] / weights[-1]) == pytest.approx(3.5, rel=1e-6)

    capped = token_weights(selected, mask, lam=99.0, w_max=5.0, valid_count=10, selected_count=2)
    assert float(capped.max() / capped.min()) == pytest.approx(5.0, rel=1e-6)
    assert float(capped.mean()) == pytest.approx(1.0, abs=1e-6)


def test_weights_ignore_padding_and_collapse_without_selection():
    mask = torch.tensor([True, True, False])
    none_selected = torch.zeros(3, dtype=torch.bool)
    weights = token_weights(none_selected, mask, lam=3.0, w_max=5.0, valid_count=2, selected_count=0)
    assert weights.tolist() == [1.0, 1.0, 0.0]


def test_kappa_zero_reproduces_the_uniform_weight_loss():
    torch.manual_seed(0)
    log_probs = torch.randn(4, 7, requires_grad=True)
    advantages = torch.randn(4, 7)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[:, -2:] = False
    selected = torch.rand(4, 7) < 0.3
    valid_count = float(mask.sum())

    lam, _, fell_back = lambda_from_sums(gain=1e-3, damage=-2e-4, kappa=0.0)
    assert lam == 0.0 and not fell_back
    weights = token_weights(selected, mask, lam, w_max=5.0, valid_count=valid_count,
                            selected_count=float((selected & mask).sum()))
    assert torch.allclose(weights, uniform_weights(mask))

    weighted = weighted_policy_sum(log_probs, advantages, weights, mask)
    baseline = weighted_policy_sum(log_probs, advantages, uniform_weights(mask), mask)
    assert torch.equal(weighted, baseline)


def test_weighted_policy_sum_is_differentiable_only_through_log_probs():
    log_probs = torch.randn(3, 5, requires_grad=True)
    advantages = torch.randn(3, 5, requires_grad=True)
    weights = torch.rand(3, 5, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    weighted_policy_sum(log_probs, advantages, weights, mask).backward()
    assert log_probs.grad is not None
    assert advantages.grad is None and weights.grad is None


def test_padding_never_contributes_to_the_loss():
    log_probs = torch.ones(2, 4, requires_grad=True)
    advantages = torch.full((2, 4), 5.0)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    weights = uniform_weights(mask)
    total = weighted_policy_sum(log_probs, advantages, weights, mask)
    assert float(total) == pytest.approx(15.0)


def test_random_selector_matches_the_requested_rate():
    mask = torch.ones(20000, dtype=torch.bool)
    generator = torch.Generator().manual_seed(3)
    selected = random_mask(mask, rate=0.08, generator=generator)
    assert float(selected.float().mean()) == pytest.approx(0.08, abs=0.01)
    assert random_mask(mask, rate=0.0).sum() == 0


def test_stratified_random_exactly_matches_sign_magnitude_and_position_counts():
    advantages = [torch.tensor([
        [0.01, 0.2, -0.01, -0.2, 0.02, 0.3, -0.02, -0.3],
        [0.02, 0.3, -0.02, -0.3, 0.01, 0.2, -0.01, -0.2],
    ])]
    masks = [torch.ones_like(advantages[0], dtype=torch.bool)]
    target = [torch.tensor([
        [True, False, True, False, False, True, False, True],
        [False, True, False, True, True, False, True, False],
    ])]
    selected = stratified_random_masks(
        advantages, masks, target, bins=4, lo=1e-3, hi=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert int(selected[0].sum()) == int(target[0].sum())

    def counts(choice):
        adv = advantages[0]
        quartile = position_quartile(masks[0])
        sign = torch.sign(adv).to(torch.long)
        magnitude = torch.clamp(
            (torch.log(adv.abs().clamp(min=1e-3) / 1e-3) / math.log(1e3) * 4).floor().long(),
            min=0,
            max=3,
        )
        return sorted(
            (int(sign[row, col]), int(magnitude[row, col]), int(quartile[row, col]))
            for row, col in torch.nonzero(choice, as_tuple=False).tolist()
        )

    assert counts(selected[0]) == counts(target[0])


def test_stratified_random_is_reproducible_and_ignores_padding():
    advantages = [torch.linspace(-2, 2, 40).reshape(4, 10)]
    mask = torch.ones_like(advantages[0], dtype=torch.bool)
    mask[:, -2:] = False
    target = torch.zeros_like(mask)
    target[:, 1] = True
    target[:, 5] = True
    first = stratified_random_masks(
        advantages, [mask], [target], 8, 1e-3, 4.0, torch.Generator().manual_seed(3)
    )[0]
    second = stratified_random_masks(
        advantages, [mask], [target], 8, 1e-3, 4.0, torch.Generator().manual_seed(3)
    )[0]
    assert torch.equal(first, second)
    assert not bool((first & ~mask).any())
    assert int(first.sum()) == int(target.sum())


def test_entropy_selector_picks_high_entropy_tokens():
    entropy = torch.tensor([0.1, 2.0, 0.5, 3.0])
    mask = torch.tensor([True, True, True, False])
    assert entropy_mask(entropy, mask, threshold=1.0).tolist() == [False, True, False, False]


def test_config_rejects_unsupported_settings():
    DisturbedOPDConfig()  # defaults must be valid
    for bad in (
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"eta_probe": 0.0},
        {"kappa": 1.0},
        {"kappa": -0.1},
        {"w_max": 0.5},
        {"a_min_quantile": 1.0},
        {"select": "magic"},
        {"direction": "sideways"},
        {"probe_weight_decay": True},
        {"quantile_bins": 0},
        {"quantile_min": 0.0},
        {"quantile_min": 100.0},
        {"stratified_bins": 1},
    ):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)


def test_momentum_probe_matches_the_real_adam_direction():
    """With momentum on, the probe must follow the direction AdamW will take.

    The raw-gradient probe does not: at beta1=0.9 the real step is dominated by
    earlier batches, so the two can even disagree in sign per coordinate.
    """
    torch.manual_seed(0)
    start = torch.randn(256, dtype=torch.float64) * 0.02
    param = torch.nn.Parameter(start.clone())
    optimizer = torch.optim.AdamW([param], lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    # Build up momentum over several batches so history actually dominates.
    for _ in range(6):
        param.grad = torch.randn(256, dtype=torch.float64) * 2e-5
        optimizer.step()
    # optimizer.step() mutates exp_avg/exp_avg_sq in place, so snapshot the
    # pre-step state the probe would actually read.
    state = optimizer.state[param]
    pre_exp_avg = state["exp_avg"].clone()
    pre_exp_avg_sq = state["exp_avg_sq"].clone()
    completed = int(state["step"])
    fresh = torch.randn(256, dtype=torch.float64) * 2e-5

    before = param.data.clone()
    param.grad = fresh.clone()
    optimizer.step()
    real_move = param.data - before

    momentum_probe = probe_delta(
        fresh, pre_exp_avg_sq, next_step=completed + 1, eta_probe=1e-4, beta2=0.999, eps=1e-8,
        exp_avg=pre_exp_avg, use_momentum=True, beta1=0.9,
    )
    grad_probe = probe_delta(
        fresh, pre_exp_avg_sq, next_step=completed + 1, eta_probe=1e-4, beta2=0.999, eps=1e-8
    )

    def cosine(a, b):
        return float((a @ b) / (a.norm() * b.norm()))

    # The momentum probe is collinear with the real step; the raw-gradient probe
    # is materially off it (measured ~0.55 here), which is the mismatch this
    # option exists to remove.
    assert cosine(momentum_probe.to(torch.float64), real_move) > 0.9999
    assert cosine(grad_probe.to(torch.float64), real_move) < 0.7


def test_momentum_and_gradient_probes_coincide_on_the_first_step():
    """No history yet means both formulations must agree exactly."""
    grad = torch.randn(64) * 1e-5
    a = probe_delta(grad, None, next_step=1, eta_probe=1e-4, beta2=0.999, eps=1e-8)
    b = probe_delta(grad, None, next_step=1, eta_probe=1e-4, beta2=0.999, eps=1e-8,
                    exp_avg=None, use_momentum=True, beta1=0.9)
    assert torch.allclose(a, b)


def test_momentum_probe_rejects_bad_beta1():
    with pytest.raises(ValueError):
        probe_delta(torch.zeros(4), None, 2, 1e-4, 0.999, 1e-8, use_momentum=True, beta1=0.0)


def test_config_roundtrip():
    config = DisturbedOPDConfig(enabled=True, alpha=0.2, kappa=0.5)
    assert DisturbedOPDConfig.from_config(config.as_dict()) == config


# ---------------------------------------------------------------------------
# Budget weight mode: decouples rescuing V from attenuating everything else.
# ---------------------------------------------------------------------------


def test_mean_one_mode_suppression_matches_closed_form():
    """Documents the failure the budget mode fixes.

    Under mean-1 normalization the non-selected weight is exactly
    1 / (1 + V_frac * lam), so at the measured V_frac ~= 0.27 and lam ~= 3.4 the
    majority of tokens lost about half their signal.
    """
    n, v_frac, lam = 10000, 0.27, 3.41
    mask = torch.ones(n, dtype=torch.bool)
    selected = torch.zeros(n, dtype=torch.bool)
    selected[: int(n * v_frac)] = True
    weights = token_weights(selected, mask, lam, w_max=99.0,
                            valid_count=float(n), selected_count=float(selected.sum()))
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(weights[selected].mean() / weights[~selected].mean()) == pytest.approx(1 + lam, rel=1e-4)
    assert float(weights[~selected][0]) == pytest.approx(1 / (1 + v_frac * lam), rel=1e-3)


def test_budget_mode_never_attenuates_unselected_tokens():
    n, v_frac, lam = 1000, 0.05, 4.0
    mask = torch.ones(n, dtype=torch.bool)
    selected = torch.zeros(n, dtype=torch.bool)
    selected[: int(n * v_frac)] = True
    weights = token_weights_budget(selected, mask, lam, w_max=20.0)
    assert float(weights[~selected].min()) == 1.0
    assert float(weights[~selected].max()) == 1.0
    assert float(weights[selected][0]) == pytest.approx(1 + lam)
    growth = float(weights.sum()) / n - 1.0
    assert growth == pytest.approx(v_frac * lam, rel=1e-6)


def test_budget_mode_respects_w_max_and_padding():
    mask = torch.tensor([True, True, False])
    selected = torch.tensor([True, False, True])
    weights = token_weights_budget(selected, mask, lam=50.0, w_max=5.0)
    assert weights.tolist() == [5.0, 1.0, 0.0]


def test_budget_lambda_bounds_total_mass_growth():
    # The cap is what stops "shrink V" from silently restoring the old behavior:
    # lambda_max grows as V shrinks, so an uncapped lambda keeps mass growth high.
    assert budget_lambda(3.41, 0.27, 0.2) == pytest.approx(0.2 / 0.27)
    assert budget_lambda(20.0, 0.05, 0.2) == pytest.approx(4.0)
    # An already-small lambda is left alone.
    assert budget_lambda(0.5, 0.05, 0.2) == pytest.approx(0.5)
    # No selection means no boost.
    assert budget_lambda(5.0, 0.0, 0.2) == 0.0
    with pytest.raises(ValueError):
        budget_lambda(1.0, 0.1, 0.0)


def test_budget_mode_growth_is_bounded_for_any_v_fraction():
    budget = 0.2
    for v_frac in (0.01, 0.05, 0.27, 0.6):
        n = 10000
        mask = torch.ones(n, dtype=torch.bool)
        selected = torch.zeros(n, dtype=torch.bool)
        selected[: int(n * v_frac)] = True
        lam = budget_lambda(1e6, v_frac, budget)
        weights = token_weights_budget(selected, mask, lam, w_max=1e9)
        growth = float(weights.sum()) / n - 1.0
        assert growth <= budget + 1e-6, (v_frac, growth)


def test_position_quartile_splits_each_response_by_its_own_length():
    mask = torch.tensor([[True] * 8 + [False] * 4, [True] * 4 + [False] * 8])
    quartile = position_quartile(mask)
    assert quartile[0, :8].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert quartile[1, :4].tolist() == [0, 1, 2, 3]


def test_selection_diagnostics_counts_position_and_sign():
    mask = torch.tensor([[True] * 4])
    advantages = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    selected = torch.tensor([[True, True, False, False]])
    diag = dict(zip(DIAG_KEYS, selection_diagnostics(selected, mask, advantages).tolist()))
    assert diag["valid"] == 4 and diag["selected"] == 2
    assert [diag[f"val_q{q}"] for q in range(4)] == [1, 1, 1, 1]
    assert [diag[f"sel_q{q}"] for q in range(4)] == [1, 1, 0, 0]
    assert diag["sel_apos"] == 1 and diag["sel_aneg"] == 1
    assert diag["val_apos"] == 2 and diag["val_aneg"] == 2


def test_selection_diagnostics_ignores_padding():
    mask = torch.tensor([[True, True, False, False]])
    advantages = torch.tensor([[1.0, -1.0, 5.0, 5.0]])
    selected = torch.tensor([[True, False, True, True]])
    diag = dict(zip(DIAG_KEYS, selection_diagnostics(selected, mask, advantages).tolist()))
    assert diag["valid"] == 2 and diag["selected"] == 1


def test_config_rejects_bad_weight_mode_and_budget():
    DisturbedOPDConfig(weight_mode="budget", mass_growth_budget=0.2)
    DisturbedOPDConfig(weight_mode="continuous_budget", continuous_floor=0.25, continuous_quantile=0.9)
    for bad in ({"weight_mode": "magic"}, {"mass_growth_budget": 0.0}, {"mass_growth_budget": -1.0}):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)
    DisturbedOPDConfig(direction="up", weight_mode="budget", opposite_weight_scale=0.5)
    for bad in ({"opposite_weight_scale": -0.1}, {"opposite_weight_scale": 1.1}):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)
    for bad in (
        {"direction": "both", "weight_mode": "budget", "opposite_weight_scale": 0.5},
        {"direction": "up", "weight_mode": "mean_one", "opposite_weight_scale": 0.5},
    ):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)
    for bad in ({"continuous_floor": -0.1}, {"continuous_floor": 1.1}, {"continuous_quantile": 0.0}):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)
    for bad in (
        {"continuous_score_mode": "magic"},
        {"continuous_rank_power": 0.0},
        {"continuous_top_fraction": 0.0},
        {"continuous_top_fraction": 1.1},
    ):
        with pytest.raises(ValueError):
            DisturbedOPDConfig(**bad)


def test_continuous_damage_scores_and_weights_are_bounded():
    from verl.trainer.ppo.disturbed_opd import continuous_damage_scores, token_weights_continuous_budget

    advantages = torch.tensor([[1.0, 2.0, -1.0, 1.0]])
    changes = torch.tensor([[-0.1, -0.5, 0.2, 0.4]])
    selected = torch.tensor([[True, True, True, False]])
    mask = torch.ones_like(selected)
    scores = continuous_damage_scores(advantages, changes, selected, scale=0.5, floor=0.25)
    assert torch.allclose(scores, torch.tensor([[0.4, 1.0, 0.55, 0.0]]))
    weights = token_weights_continuous_budget(scores, mask, lam=2.0, w_max=2.5)
    assert torch.allclose(weights, torch.tensor([[1.8, 2.5, 2.1, 1.0]]))


def test_opposite_attenuation_scales_only_target_without_renormalizing():
    weights = torch.tensor([[1.8, 1.0, 2.2, 1.0]])
    opposite = torch.tensor([[False, True, False, True]])
    mask = torch.tensor([[True, True, True, False]])
    assert torch.equal(
        attenuate_opposite_weights(weights, opposite, mask, 0.5),
        torch.tensor([[1.8, 0.5, 2.2, 0.0]]),
    )
    assert torch.equal(
        attenuate_opposite_weights(weights, opposite, mask, 0.0),
        torch.tensor([[1.8, 0.0, 2.2, 0.0]]),
    )


def test_continuous_raw_severity_modes_and_mapping():
    advantages = torch.tensor([[2.0, -3.0, 1.0]])
    changes = torch.tensor([[-0.5, 0.25, 0.4]])
    selected = torch.tensor([[True, True, False]])
    delta = continuous_raw_severity(advantages, changes, "delta")
    surrogate = continuous_raw_severity(advantages, changes, "surrogate")
    assert torch.allclose(delta, torch.tensor([[0.5, 0.25, 0.0]]))
    assert torch.allclose(surrogate, torch.tensor([[1.0, 0.75, 0.0]]))
    scores = continuous_scores_from_severity(surrogate, selected, scale=1.0, floor=0.25)
    assert torch.allclose(scores, torch.tensor([[1.0, 0.8125, 0.0]]))


def test_continuous_rank_scores_are_monotone_and_masked():
    severity = torch.tensor([[0.01, 0.1, 1.0, 10.0, 20.0]])
    selected = torch.tensor([[True, True, True, True, False]])
    histogram = abs_histogram(severity, selected, bins=32, hi=32.0, lo=1e-4)
    scores = continuous_rank_scores(
        severity, selected, histogram, hi=32.0, lo=1e-4, floor=0.25, power=1.0
    )
    assert torch.all(scores[0, 1:4] > scores[0, :3])
    assert 0.25 <= float(scores[0, 0]) <= 1.0
    assert scores[0, 4] == 0


def test_top_fraction_mask_respects_existing_selection():
    severity = torch.tensor([0.1, 0.4, 0.7, 1.0])
    selected = torch.tensor([True, False, True, True])
    assert top_fraction_mask(severity, selected, 0.7).tolist() == [False, False, True, True]


def test_stratified_shuffle_preserves_mask_and_score_multiset():
    advantages = [torch.ones(1, 8)]
    masks = [torch.ones(1, 8, dtype=torch.bool)]
    selected = [torch.ones(1, 8, dtype=torch.bool)]
    scores = [torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)]
    generator = torch.Generator().manual_seed(7)
    shuffled = stratified_shuffle_scores(
        advantages, masks, selected, scores, bins=4, lo=1e-4, hi=32.0, generator=generator
    )[0]
    # Position quartiles contain pairs; every pair keeps exactly its two values.
    for start in range(0, 8, 2):
        assert sorted(shuffled[0, start : start + 2].tolist()) == sorted(
            scores[0][0, start : start + 2].tolist()
        )
    assert torch.equal(shuffled != 0, selected[0])
