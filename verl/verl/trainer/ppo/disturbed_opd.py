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
"""CPU-testable primitives for probe-and-reweight (disturbed) OPD.

One update measures which tokens a uniform-weight gradient step would push
*against* their own advantage, then compensates them in the real step. The
probe applies a momentum-free, decay-free Adam-shaped delta to the parameters,
reads the resulting log-prob shift, and restores the parameters bit-exactly.

Nothing here touches optimizer state, distributed collectives, or FSDP; the
caller owns those so that every function below is testable on CPU.
"""

import math
from dataclasses import asdict, dataclass

import torch

SELECT_MODES = ("probe", "random", "stratified_random", "entropy")
WEIGHT_MODES = ("mean_one", "budget", "continuous_budget")
DIRECTIONS = ("both", "up", "down")
CONTINUOUS_SCORE_MODES = ("delta", "surrogate", "rank", "topq")


@dataclass
class DisturbedOPDConfig:
    """Hyperparameters for one probe-and-reweight update.

    beta2/eps are intentionally absent: they are read from the live actor
    optimizer so the probe delta matches the shape of the real Adam step.
    """

    enabled: bool = False
    # Legacy path is unchanged; v5 methods live in disturbed_updates.py.
    method: str = "reweight"
    repair_norm_fraction: float = 0.2
    repair_tolerance: float = 1e-4
    repair_huber: float = 0.1
    repair_adv_cap: float = 1.0
    adam_progress_fraction: float = 0.9
    teacher_topk: int = 32
    distribution_coef: float = 0.2
    audit_every: int = 10
    # Optional mechanism-audit sink.  Empty keeps the historical training path
    # bit-for-bit free of audit I/O and post-update forwards.
    audit_dir: str = ""
    audit_post_update: bool = False
    audit_topk: int = 2
    # Optional teacher-top-K + tail coarsened forward KL for mechanism audits.
    # Zero keeps the historical reward/actor path unchanged.  A positive K
    # asks the fixed teacher for its top-K support while retaining sampled-token
    # OPD as the actual training signal.
    audit_distribution_kl_topk: int = 0
    audit_eos_token_id: int = 151643
    # alpha=1 makes the advantage identical to the existing token_reward_direct
    # reward, so the first experiment differs from the baseline only by the
    # probe and the reweighting.
    alpha: float = 1.0
    eta_probe: float = 1e-4
    # kappa=0 disables the correction: weights collapse to 1 and the update
    # must reproduce the uniform-weight baseline exactly.
    kappa: float = 0.0
    w_max: float = 5.0
    # Fraction of valid tokens excluded from selection by smallest |A|.
    a_min_quantile: float = 0.2
    select: str = "probe"
    direction: str = "both"
    # mean_one: w normalized to global mean 1 (the original formulation; boosting
    #   V necessarily suppresses everything else by 1/(1 + V_frac * lam), which at
    #   V_frac~0.27 halved the majority's signal).
    # budget:   non-V weights stay exactly 1.0 and lam is capped so the total
    #   gradient mass grows by at most mass_growth_budget.
    weight_mode: str = "mean_one"
    mass_growth_budget: float = 0.2
    # Optionally attenuate the *opposite* disturbed set while preserving the
    # primary budget intervention exactly.  For direction=up this scales
    # V_down={A<0, Delta>0}; for direction=down it scales
    # V_up={A>0, Delta<0}.  One is the legacy no-op, zero masks the opposite
    # set out of the sampled-token objective.
    opposite_weight_scale: float = 1.0
    # continuous_budget keeps the same probe mask and total extra-mass budget,
    # but distributes it by the opposed log-prob displacement magnitude.
    # A positive floor prevents tiny-but-reliably-opposed tokens from losing all
    # rescue; the global quantile makes the scale robust to a few outliers.
    continuous_floor: float = 0.25
    continuous_quantile: float = 0.9
    # delta: robustly normalized opposed log-prob displacement (the first
    #   continuous experiment).
    # surrogate: frozen token-surrogate damage [-A * Delta log p]+.
    # rank: percentile rank of opposed displacement, robust to its scale.
    # topq: retain only the largest continuous_top_fraction displacements and
    #   weight that subset uniformly.
    continuous_score_mode: str = "delta"
    continuous_shuffle: bool = False
    continuous_rank_power: float = 1.0
    continuous_top_fraction: float = 0.5
    # False: probe this batch's raw gradient (the original formulation).
    # True:  probe the bias-corrected first moment, i.e. the direction the real
    #        Adam step will actually take.
    probe_momentum: bool = False
    probe_weight_decay: bool = False
    # Diagnostic sweep: probe several eta values and both directions on identical
    # parameters, report V overlap, then update with the configured probe.
    probe_diagnostic: bool = False
    # Global |A| quantile is derived from an all-reduced histogram so every
    # rank computes an identical threshold.
    quantile_bins: int = 2048
    quantile_max: float = 32.0
    # Lower end of the log-spaced |A| grid. Anything below is treated as the
    # smallest bucket and excluded when a_min_quantile > 0.
    quantile_min: float = 1e-12
    random_seed: int = 20260924
    # Number of log-|A| buckets used by the strict random control. Together
    # with sign(A) and within-response position quartile, these strata preserve
    # the probe selector's observable token composition exactly on every rank.
    stratified_bins: int = 16

    def __post_init__(self):
        if self.method not in ("reweight", "real_repair", "adam_budget", "distribution_repair", "sign_hybrid"):
            raise ValueError("Unknown disturbed OPD method")
        if not 0 < self.repair_norm_fraction <= 1 or self.repair_tolerance < 0:
            raise ValueError("Invalid repair displacement budget/tolerance")
        if self.repair_huber <= 0 or self.repair_adv_cap <= 0:
            raise ValueError("Repair Huber and advantage caps must be positive")
        if not 0 < self.adam_progress_fraction <= 1:
            raise ValueError("adam_progress_fraction must be in (0, 1]")
        if self.teacher_topk < 1 or self.distribution_coef < 0 or self.audit_every < 1:
            raise ValueError("Invalid distribution/audit settings")
        if self.audit_topk < 0 or self.audit_topk > 32:
            raise ValueError("audit_topk must be in [0, 32]")
        if self.audit_distribution_kl_topk < 0 or self.audit_distribution_kl_topk > 256:
            raise ValueError("audit_distribution_kl_topk must be in [0, 256]")
        if self.audit_distribution_kl_topk and not self.audit_dir:
            raise ValueError("audit_distribution_kl_topk requires audit_dir")
        if self.audit_eos_token_id < 0:
            raise ValueError("audit_eos_token_id must be nonnegative")
        if self.method != "reweight" and (self.select != "probe" or self.weight_mode != "budget"):
            raise ValueError("New methods require probe selection and budget mode")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(self.eta_probe) or self.eta_probe <= 0:
            raise ValueError("eta_probe must be positive and finite")
        if not math.isfinite(self.kappa) or not 0 <= self.kappa < 1:
            raise ValueError("kappa must be in [0, 1); kappa=1 would cancel all batch progress")
        if not math.isfinite(self.w_max) or self.w_max < 1:
            raise ValueError("w_max must be at least 1")
        if not 0 <= self.a_min_quantile < 1:
            raise ValueError("a_min_quantile must be in [0, 1)")
        if self.select not in SELECT_MODES:
            raise ValueError(f"select must be one of {SELECT_MODES}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
        if not math.isfinite(self.mass_growth_budget) or self.mass_growth_budget <= 0:
            raise ValueError("mass_growth_budget must be positive and finite")
        if not math.isfinite(self.opposite_weight_scale) or not 0 <= self.opposite_weight_scale <= 1:
            raise ValueError("opposite_weight_scale must be in [0, 1]")
        if self.opposite_weight_scale != 1.0:
            if self.method != "reweight" or self.select != "probe":
                raise ValueError("opposite attenuation requires reweight with probe selection")
            if self.direction not in ("up", "down") or self.weight_mode != "budget":
                raise ValueError("opposite attenuation requires up/down direction and budget mode")
        if not 0 <= self.continuous_floor <= 1:
            raise ValueError("continuous_floor must be in [0, 1]")
        if not 0 < self.continuous_quantile < 1:
            raise ValueError("continuous_quantile must be in (0, 1)")
        if self.continuous_score_mode not in CONTINUOUS_SCORE_MODES:
            raise ValueError(f"continuous_score_mode must be one of {CONTINUOUS_SCORE_MODES}")
        if not math.isfinite(self.continuous_rank_power) or self.continuous_rank_power <= 0:
            raise ValueError("continuous_rank_power must be positive and finite")
        if not 0 < self.continuous_top_fraction <= 1:
            raise ValueError("continuous_top_fraction must be in (0, 1]")
        if self.probe_weight_decay:
            raise ValueError("The probe is decay-free by construction; weight decay is data independent")
        if self.quantile_bins < 2 or self.quantile_max <= 0:
            raise ValueError("Histogram needs at least two bins and a positive range")
        if not 0 < self.quantile_min < self.quantile_max:
            raise ValueError("quantile_min must be positive and below quantile_max")
        if self.stratified_bins < 2:
            raise ValueError("stratified_bins must be at least 2")

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config or {}))

    def as_dict(self):
        return asdict(self)


def advantage_from_logratio(log_ratio: torch.Tensor, alpha: float) -> torch.Tensor:
    """A = log(1 - alpha + alpha * exp(log_ratio)), evaluated stably.

    log_ratio is the teacher-minus-student log prob on the sampled token, i.e.
    exactly the existing sampled-token OPD reward (rm_scores at top_k=0).
    alpha=1 returns log_ratio itself, so the advantage is bit-identical to the
    baseline reward. Lower bound is log(1 - alpha); there is no upper bound.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if alpha == 1:
        return log_ratio.clone()
    floor = math.log1p(-alpha)
    shifted = log_ratio.to(torch.float32) + math.log(alpha)
    return torch.logaddexp(torch.full_like(shifted, floor), shifted)


def topk_tail_forward_kl(
    teacher_topk_log_probs: torch.Tensor,
    student_on_teacher_topk_log_probs: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Teacher-to-student KL after coarsening the vocabulary to top-K + tail.

    The teacher's K token identities define K singleton bins and all remaining
    vocabulary items form one tail bin.  The result is a proper nonnegative KL
    (up to floating-point tolerance) and, by data processing, a lower bound on
    the unavailable full-vocabulary ``KL(teacher || student)``.  Unlike the
    sampled-token advantage, it compares probability distributions at each
    response position without changing the OPD objective.
    """
    if teacher_topk_log_probs.shape != student_on_teacher_topk_log_probs.shape:
        raise ValueError("teacher and student top-K tensors must have identical shapes")
    if teacher_topk_log_probs.ndim < 1 or teacher_topk_log_probs.shape[-1] < 1:
        raise ValueError("top-K tensors need a nonempty final dimension")
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    teacher_logp = teacher_topk_log_probs.detach().float()
    student_logp = student_on_teacher_topk_log_probs.detach().float()
    teacher_prob = teacher_logp.exp()
    student_prob = student_logp.exp()
    teacher_mass = teacher_prob.sum(dim=-1).clamp(min=0.0, max=1.0 - eps)
    student_mass = student_prob.sum(dim=-1).clamp(min=0.0, max=1.0 - eps)
    singleton = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
    teacher_tail = (1.0 - teacher_mass).clamp_min(eps)
    student_tail = (1.0 - student_mass).clamp_min(eps)
    tail = teacher_tail * (teacher_tail.log() - student_tail.log())
    # Tiny negatives can arise solely from bf16 log-prob rounding.
    return (singleton + tail).clamp_min(0.0)


def probe_delta(
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor | None,
    next_step: int,
    eta_probe: float,
    beta2: float,
    eps: float,
    exp_avg: torch.Tensor | None = None,
    use_momentum: bool = False,
    beta1: float = 0.9,
) -> torch.Tensor:
    """Momentum-free, decay-free Adam-shaped step for this batch's gradient.

    Returned with Adam's sign convention, i.e. the DESCENT direction
    ``-eta * g / (sqrt(v_hat) + eps)``, so the caller applies it with
    ``p.data.add_(delta)`` and reproduces the direction ``optimizer.step()``
    would take. Getting this sign wrong silently inverts the selected token
    set, so it is asserted by both a unit test and a first-order fp64 check.

    next_step is the Adam step index this update *would* be (completed + 1), so
    the bias correction matches what the real step will use. With no optimizer
    state yet the magnitude degenerates to eta_probe, matching Adam's own first
    step. Always returned in fp32; the caller casts when applying.

    use_momentum selects which direction is probed:
      False - this batch's raw gradient. Measures what *this batch alone* would
              do, but with beta1=0.9 the optimizer's actual step is ~90%
              determined by earlier batches, so the probe direction is not the
              direction that will be taken.
      True  - the bias-corrected first moment the real step will use, i.e. the
              genuine Adam direction scaled by eta_probe/lr. Keeps the probe
              measurable while removing that direction mismatch.
    """
    if next_step < 1:
        raise ValueError("next_step is 1-based")
    if not 0 < beta2 < 1:
        raise ValueError("beta2 must be in (0, 1)")
    if use_momentum and not 0 < beta1 < 1:
        raise ValueError("beta1 must be in (0, 1)")
    grad32 = grad.detach().to(torch.float32)
    previous = torch.zeros_like(grad32) if exp_avg_sq is None else exp_avg_sq.detach().to(torch.float32)
    v_new = beta2 * previous + (1 - beta2) * grad32 * grad32
    v_hat = v_new / (1 - beta2**next_step)
    if use_momentum:
        history = torch.zeros_like(grad32) if exp_avg is None else exp_avg.detach().to(torch.float32)
        m_new = beta1 * history + (1 - beta1) * grad32
        numerator = m_new / (1 - beta1**next_step)
    else:
        numerator = grad32
    return -eta_probe * numerator / (v_hat.sqrt() + eps)


def _bin_edge(index: int, bins: int, lo: float, hi: float) -> float:
    """Lower edge of a log-spaced bin.

    Index 0 is the underflow bin holding |A| < lo. Returning 0.0 for it would
    keep every token and silently disable the a_min filter, which is what
    happened once the student converged toward the teacher and most |A| fell
    below the old lo=1e-6. Returning lo instead excludes that bin, matching the
    filter's intent; it can exclude more than the requested quantile when the
    underflow bin alone exceeds it, which is the desired direction.
    """
    if index <= 0:
        return lo
    span = math.log(hi / lo)
    return lo * math.exp(span * (min(index, bins) - 1) / max(bins - 1, 1))


def abs_histogram(
    values: torch.Tensor, mask: torch.Tensor, bins: int, hi: float, lo: float = 1e-12
) -> torch.Tensor:
    """Counts of |values| over masked entries in log-spaced bins.

    |A| spans many orders of magnitude and is concentrated near zero, so linear
    bins put most of the distribution in the first bin and make any low quantile
    collapse to a zero threshold. Bin 0 collects |values| < lo, bins 1..bins-1
    are log-spaced over [lo, hi), and the last bin collects |values| >= hi. The
    caller all-reduces this vector, so the derived threshold is rank-invariant.
    """
    if bins < 2 or hi <= 0 or not 0 < lo < hi:
        raise ValueError("Histogram needs at least two bins and 0 < lo < hi")
    valid = mask.bool()
    magnitude = values.detach().abs().to(torch.float32)[valid]
    counts = torch.zeros(bins + 1, dtype=torch.int64, device=values.device)
    if magnitude.numel() == 0:
        return counts
    span = math.log(hi / lo)
    ratio = torch.log(magnitude.clamp(min=lo * 1e-12) / lo) / span
    index = torch.where(
        magnitude < lo,
        torch.zeros_like(ratio, dtype=torch.long),
        torch.clamp((ratio * (bins - 1)).long() + 1, min=1, max=bins),
    )
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.int64))
    return counts


def threshold_from_histogram(
    counts: torch.Tensor, quantile: float, hi: float, lo: float = 1e-12
) -> float:
    """Lower edge of the log-spaced bin where the cumulative count reaches `quantile`.

    Tokens with |A| strictly below this edge are excluded from selection.
    """
    if not 0 <= quantile < 1:
        raise ValueError("quantile must be in [0, 1)")
    total = int(counts.sum())
    if quantile == 0 or total == 0:
        return 0.0
    bins = counts.numel() - 1
    target = quantile * total
    cumulative = torch.cumsum(counts, dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(target, device=counts.device), right=False))
    return _bin_edge(index, bins, lo, hi)


def probe_mask(
    advantages: torch.Tensor,
    delta_log_probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    direction: str = "both",
) -> torch.Tensor:
    """V = tokens whose realized log-prob shift opposes their own advantage.

    A_t * Delta_t < 0 means the rest of the batch overpowered this token's own
    contribution, which is the interference this method compensates.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    valid = mask.bool()
    opposed = (advantages * delta_log_probs) < 0
    large_enough = advantages.abs() >= threshold
    if direction == "up":
        wanted = advantages > 0
    elif direction == "down":
        wanted = advantages < 0
    else:
        wanted = torch.ones_like(valid)
    return valid & opposed & large_enough & wanted


def random_mask(mask: torch.Tensor, rate: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Ablation selector: same expected size as the probe set, chosen at random."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    valid = mask.bool()
    if rate == 0:
        return torch.zeros_like(valid)
    draw = torch.rand(valid.shape, generator=generator, device=valid.device)
    return valid & (draw < rate)


def entropy_mask(entropy: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Ablation selector: highest-entropy tokens instead of probe-detected ones."""
    valid = mask.bool()
    return valid & (entropy.detach() >= threshold)


def lambda_from_sums(gain: float, damage: float, kappa: float) -> tuple[float, float, bool]:
    """lambda = kappa * G / |D|, with a documented no-op fallback.

    G is the batch-wide sum of A*Delta and must be positive for the linearised
    picture to hold; a non-positive G means the probe left the linear regime or
    is noise-dominated, so the update falls back to uniform weights.
    Returns (lambda, lambda_max, fell_back).
    """
    if not math.isfinite(gain) or not math.isfinite(damage):
        return 0.0, 0.0, True
    if gain <= 0:
        return 0.0, 0.0, True
    magnitude = abs(damage)
    if magnitude == 0:
        # Nothing was pushed backwards; there is nothing to compensate.
        return 0.0, float("inf"), False
    lambda_max = gain / magnitude
    return kappa * lambda_max, lambda_max, False



def budget_lambda(lam: float, v_fraction: float, budget: float) -> float:
    """Cap lam so the total gradient mass grows by at most `budget`.

    Under the budget weight mode the non-selected tokens keep weight 1, so the
    mass added is exactly v_fraction * lam. Capping it decouples "rescue the
    interfered tokens" from "suppress everyone else", and keeps the effective
    step size bounded instead of implicitly doubling it.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if v_fraction <= 0:
        return 0.0
    return min(lam, budget / v_fraction)


def token_weights_budget(
    selected: torch.Tensor, mask: torch.Tensor, lam: float, w_max: float
) -> torch.Tensor:
    """w = min(1 + lam, w_max) on the selected set, exactly 1.0 elsewhere.

    Deliberately not renormalized: the caller bounds the extra mass through
    budget_lambda instead, so the majority of tokens is never attenuated.
    """
    if lam < 0 or w_max < 1:
        raise ValueError("lam must be nonnegative and w_max at least 1")
    valid = mask.bool()
    boost = min(1.0 + lam, w_max)
    return torch.where(selected & valid, boost, 1.0).to(torch.float32) * valid


def attenuate_opposite_weights(
    weights: torch.Tensor,
    opposite_selected: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Scale only the opposite disturbed set, without renormalizing weights."""
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("scale must be in [0, 1]")
    valid = mask.bool()
    return torch.where(opposite_selected.bool() & valid, weights * scale, weights) * valid


def continuous_damage_scores(
    advantages: torch.Tensor,
    delta_log_probs: torch.Tensor,
    selected: torch.Tensor,
    scale: float,
    floor: float,
) -> torch.Tensor:
    """Detached severity in [floor, 1] on selected opposed tokens.

    ``-sign(A) * Delta log p`` measures how far probability moved against the
    teacher direction without multiplying by |A| a second time (the OPD loss
    already contains A). ``scale`` is a global robust quantile of this value.
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("continuous score scale must be positive and finite")
    if not 0 <= floor <= 1:
        raise ValueError("continuous score floor must be in [0, 1]")
    severity = (-advantages.detach().float().sign() * delta_log_probs.detach().float()).clamp_min(0)
    normalized = (severity / scale).clamp(0, 1)
    score = floor + (1 - floor) * normalized
    return torch.where(selected.bool(), score, torch.zeros_like(score))


def continuous_raw_severity(
    advantages: torch.Tensor,
    delta_log_probs: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Detached nonnegative severity before normalization.

    ``delta`` measures displacement against the teacher direction without
    multiplying by |A| a second time. ``surrogate`` is the exact frozen-
    advantage token loss increase caused by the probe. Rank and top-q use the
    delta definition and change only the downstream mapping.
    """
    if mode not in CONTINUOUS_SCORE_MODES:
        raise ValueError(f"Unknown continuous score mode: {mode}")
    advantages = advantages.detach().float()
    delta_log_probs = delta_log_probs.detach().float()
    if mode == "surrogate":
        return (-advantages * delta_log_probs).clamp_min(0)
    return (-advantages.sign() * delta_log_probs).clamp_min(0)


def continuous_scores_from_severity(
    severity: torch.Tensor,
    selected: torch.Tensor,
    scale: float,
    floor: float,
) -> torch.Tensor:
    """Map raw severity to [floor, 1] on selected tokens."""
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("continuous score scale must be positive and finite")
    if not 0 <= floor <= 1:
        raise ValueError("continuous score floor must be in [0, 1]")
    normalized = (severity.detach().float() / scale).clamp(0, 1)
    score = floor + (1 - floor) * normalized
    return torch.where(selected.bool(), score, torch.zeros_like(score))


def continuous_rank_scores(
    severity: torch.Tensor,
    selected: torch.Tensor,
    global_histogram: torch.Tensor,
    hi: float,
    lo: float,
    floor: float,
    power: float,
) -> torch.Tensor:
    """Approximate global percentile-rank scores using the shared log histogram.

    Every token in one histogram bin receives that bin's midpoint percentile.
    This deliberately trusts the ordering of severity bins, not their absolute
    spacing, and requires no token-sized distributed gather.
    """
    if global_histogram.ndim != 1 or global_histogram.numel() < 3:
        raise ValueError("global_histogram must be a one-dimensional histogram")
    if not 0 < lo < hi or not 0 <= floor <= 1 or power <= 0 or not math.isfinite(power):
        raise ValueError("Invalid rank-score parameters")
    total = int(global_histogram.sum())
    if total <= 0:
        return torch.zeros_like(severity, dtype=torch.float32)
    bins = global_histogram.numel() - 1
    values = severity.detach().abs().float()
    span = math.log(hi / lo)
    ratio = torch.log(values.clamp(min=lo * 1e-12) / lo) / span
    indices = torch.where(
        values < lo,
        torch.zeros_like(ratio, dtype=torch.long),
        torch.clamp((ratio * (bins - 1)).long() + 1, min=1, max=bins),
    )
    cumulative = torch.cumsum(global_histogram, dim=0).to(torch.float64)
    midpoint = cumulative - 0.5 * global_histogram.to(torch.float64)
    percentile = (midpoint[indices] / total).to(torch.float32).clamp(0, 1)
    score = floor + (1 - floor) * percentile.pow(power)
    return torch.where(selected.bool(), score, torch.zeros_like(score))


def top_fraction_mask(
    severity: torch.Tensor,
    selected: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Keep selected tokens whose severity reaches a global quantile edge."""
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("top-fraction threshold must be finite and nonnegative")
    return selected.bool() & (severity.detach().float() >= threshold)


def token_weights_continuous_budget(
    scores: torch.Tensor, mask: torch.Tensor, lam: float, w_max: float
) -> torch.Tensor:
    """Continuous selected-token weights; non-selected valid tokens stay 1."""
    if lam < 0 or w_max < 1:
        raise ValueError("lam must be nonnegative and w_max at least 1")
    valid = mask.bool()
    extra = (lam * scores.detach().float()).clamp(min=0, max=w_max - 1)
    return torch.where(valid, 1.0 + extra, torch.zeros_like(extra))


def position_quartile(mask: torch.Tensor) -> torch.Tensor:
    """Quartile 0..3 of each token's position within its own response."""
    lengths = mask.sum(-1, keepdim=True).clamp(min=1)
    index = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    return (index * 4 // lengths).clamp(max=3)


def stratified_random_masks(
    advantages: list[torch.Tensor],
    masks: list[torch.Tensor],
    probe_selected: list[torch.Tensor],
    bins: int,
    lo: float,
    hi: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Random masks exactly matching the probe mask in observable strata.

    Each stratum is ``sign(A) x log-|A| bucket x response-position quartile``.
    Selection counts are matched exactly across all micro-batches on this data
    parallel rank. Summing exact local matches gives an exact global match,
    without gathering token-sized tensors between ranks.

    The probe-selected mask is used only to determine target counts. Random
    scores choose token identities from all valid tokens in the same stratum.
    """
    if not (len(advantages) == len(masks) == len(probe_selected)):
        raise ValueError("advantages, masks and probe_selected must have equal lengths")
    if bins < 2 or not 0 < lo < hi:
        raise ValueError("stratified random requires bins >= 2 and 0 < lo < hi")
    if not advantages:
        return []

    shapes = [value.shape for value in advantages]
    sizes = [value.numel() for value in advantages]
    flat_adv = torch.cat([value.detach().to(torch.float32).reshape(-1) for value in advantages])
    flat_valid = torch.cat([value.bool().reshape(-1) for value in masks])
    flat_target = torch.cat(
        [(value.bool() & mask.bool()).reshape(-1) for value, mask in zip(probe_selected, masks, strict=True)]
    )
    flat_quartile = torch.cat(
        [position_quartile(mask.bool()).reshape(-1) for mask in masks]
    )

    # Three sign classes keep true zeros out of positive/negative strata. The
    # log buckets include underflow and overflow in their end buckets.
    sign = torch.where(flat_adv > 0, 2, torch.where(flat_adv < 0, 0, 1)).to(torch.long)
    span = math.log(hi / lo)
    scaled = torch.log(flat_adv.abs().clamp(min=lo) / lo) / span
    magnitude = torch.clamp((scaled * bins).floor().to(torch.long), min=0, max=bins - 1)
    strata = (sign * bins + magnitude) * 4 + flat_quartile.to(torch.long)
    selected = torch.zeros_like(flat_valid)
    scores = torch.rand(flat_valid.shape, generator=generator, device=flat_valid.device)

    for stratum in torch.unique(strata[flat_target]).tolist():
        target_count = int((flat_target & (strata == stratum)).sum())
        candidates = torch.nonzero(flat_valid & (strata == stratum), as_tuple=False).flatten()
        if target_count > candidates.numel():
            raise RuntimeError(
                f"Stratum {stratum} asks for {target_count} tokens from {candidates.numel()} candidates"
            )
        if target_count:
            chosen = candidates[torch.topk(scores[candidates], target_count, sorted=False).indices]
            selected[chosen] = True

    if int(selected.sum()) != int(flat_target.sum()):
        raise RuntimeError("Stratified random selector did not preserve the probe-selected count")
    outputs, offset = [], 0
    for shape, size in zip(shapes, sizes, strict=True):
        outputs.append(selected[offset : offset + size].reshape(shape))
        offset += size
    return outputs


def stratified_shuffle_scores(
    advantages: list[torch.Tensor],
    masks: list[torch.Tensor],
    selected: list[torch.Tensor],
    scores: list[torch.Tensor],
    bins: int,
    lo: float,
    hi: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Shuffle score-token assignments inside |A|-and-position strata.

    The selected mask and the multiset of scores in every stratum are preserved
    exactly. Thus total extra mass and all per-token caps are unchanged; only
    whether high measured severity is attached to the token that produced it is
    destroyed.
    """
    if not (len(advantages) == len(masks) == len(selected) == len(scores)):
        raise ValueError("All score-shuffle input lists must have equal lengths")
    if bins < 2 or not 0 < lo < hi:
        raise ValueError("stratified score shuffle requires bins >= 2 and 0 < lo < hi")
    if not advantages:
        return []

    shapes = [value.shape for value in scores]
    sizes = [value.numel() for value in scores]
    flat_adv = torch.cat([value.detach().float().reshape(-1) for value in advantages])
    flat_valid = torch.cat([value.bool().reshape(-1) for value in masks])
    flat_selected = torch.cat(
        [(value.bool() & mask.bool()).reshape(-1) for value, mask in zip(selected, masks, strict=True)]
    )
    flat_scores = torch.cat([value.detach().float().reshape(-1) for value in scores])
    flat_quartile = torch.cat([position_quartile(mask.bool()).reshape(-1) for mask in masks])

    sign = torch.where(flat_adv > 0, 2, torch.where(flat_adv < 0, 0, 1)).long()
    span = math.log(hi / lo)
    scaled = torch.log(flat_adv.abs().clamp(min=lo) / lo) / span
    magnitude = torch.clamp((scaled * bins).floor().long(), min=0, max=bins - 1)
    strata = (sign * bins + magnitude) * 4 + flat_quartile.long()
    shuffled = flat_scores.clone()
    for stratum in torch.unique(strata[flat_selected]).tolist():
        indices = torch.nonzero(flat_selected & flat_valid & (strata == stratum), as_tuple=False).flatten()
        if indices.numel() > 1:
            permutation = torch.randperm(indices.numel(), generator=generator, device=indices.device)
            shuffled[indices] = flat_scores[indices[permutation]]
    shuffled = torch.where(flat_selected, shuffled, torch.zeros_like(shuffled))

    outputs, offset = [], 0
    for shape, size in zip(shapes, sizes, strict=True):
        outputs.append(shuffled[offset : offset + size].reshape(shape))
        offset += size
    return outputs


DIAG_KEYS = (
    "valid", "selected",
    "sel_q0", "sel_q1", "sel_q2", "sel_q3",
    "val_q0", "val_q1", "val_q2", "val_q3",
    "sel_apos", "sel_aneg", "val_apos", "val_aneg",
)


def selection_diagnostics(
    selected: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """Counts describing what the probe actually selected, for one micro-batch.

    Returns a DIAG_KEYS-aligned float64 vector so the caller can all-reduce it in
    one call. Without this the V set is a single opaque fraction, which is why an
    earlier analysis could only speculate about its composition.
    """
    valid = mask.bool()
    sel = selected & valid
    quartile = position_quartile(valid)
    positive = advantages > 0
    parts = [valid.sum(), sel.sum()]
    parts += [(sel & (quartile == q)).sum() for q in range(4)]
    parts += [(valid & (quartile == q)).sum() for q in range(4)]
    parts += [
        (sel & positive).sum(), (sel & ~positive).sum(),
        (valid & positive).sum(), (valid & ~positive).sum(),
    ]
    return torch.stack(parts).to(torch.float64)


def token_weights(
    selected: torch.Tensor,
    mask: torch.Tensor,
    lam: float,
    w_max: float,
    valid_count: float,
    selected_count: float,
) -> torch.Tensor:
    """w = min(1 + lam * 1[V], w_max), rescaled so the global mean over valid tokens is 1.

    The mean is computed from global counts in closed form, so no second
    collective is needed and every rank applies the same scale.
    """
    if lam < 0 or w_max < 1:
        raise ValueError("lam must be nonnegative and w_max at least 1")
    valid = mask.bool()
    raw = torch.where(selected & valid, min(1.0 + lam, w_max), 1.0).to(torch.float32)
    if valid_count <= 0:
        return raw * valid
    boosted = min(1.0 + lam, w_max)
    total = (valid_count - selected_count) + selected_count * boosted
    scale = valid_count / max(total, 1e-12)
    return raw * scale * valid


def weighted_policy_sum(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable sum of w * A * log pi over valid tokens.

    The caller divides by the global valid-token count and compensates FSDP's
    rank averaging, mirroring teacher_takeover.normalized_loss.
    """
    valid = mask.bool()
    contribution = weights.detach() * advantages.detach() * log_probs
    return torch.where(valid, contribution, torch.zeros_like(contribution)).sum()


def uniform_weights(mask: torch.Tensor) -> torch.Tensor:
    """The kappa=0 / fallback weighting, kept explicit for the equivalence test."""
    return mask.bool().to(torch.float32)


def validate_disturbed_opd_training(config):
    """Refuse configurations the probe cannot interpret, before any GPU is taken.

    Constraints are actor-side only: the probe needs this batch's own gradient,
    one optimizer step per batch, full per-token log probs, and readable AdamW
    state. Rollout tensor parallelism is irrelevant here.
    """
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    optimizer = actor.optim
    requirements = {
        "FSDP actor": actor.strategy in ("fsdp", "fsdp2"),
        "Ulysses SP=1": actor.ulysses_sequence_parallel_size == 1,
        "torch AdamW": optimizer.get("optimizer", "AdamW") == "AdamW"
        and optimizer.get("optimizer_impl", "torch.optim") == "torch.optim",
        "no optimizer offload": not actor.fsdp_config.optimizer_offload,
        "no param offload": not actor.fsdp_config.param_offload,
        "direct advantages": config.algorithm.adv_estimator == "token_reward_direct",
        "sampled token OPD": rollout.get("log_prob_top_k", 0) == 0,
        "one update per question batch": actor.ppo_epochs == 1
        and actor.ppo_mini_batch_size == config.data.train_batch_size,
        "unfused logits": not config.actor_rollout_ref.model.get("use_fused_kernels", False),
        # Activation offload tracks forward/backward groups with internal counters
        # that the probe's extra forward passes desynchronize (KeyError in
        # activation_offload.synchronize_on_group_commit_forward).
        "no activation offload": not config.actor_rollout_ref.model.get("enable_activation_offload", False),
        "no reference/entropy loss": not actor.use_kl_loss and actor.entropy_coeff == 0,
        "no reward KL": not config.algorithm.use_kl_in_reward,
        "no preference PPO": not config.algorithm.get("use_pf_ppo", False),
        "vanilla clip": actor.policy_loss.loss_mode == "vanilla" and actor.loss_agg_mode == "token-mean",
        "no legacy gating": not rollout.get("correctness_gated", False) and not rollout.get("grpo_scaled", False),
        "fixed teacher": config.reward_model.enable and config.reward_model.model.input_tokenizer is None,
        "teacher takeover disabled": not config.algorithm.get("teacher_takeover", {}).get("enabled", False),
        "grad clipping configured": actor.grad_clip is not None,
    }
    correction = config.algorithm.get("rollout_correction") or {}
    for key in ("rollout_is", "rollout_rs", "bypass_old_logprob_for_rollout", "use_pure_rollout_correction"):
        requirements[f"no rollout correction: {key}"] = not correction.get(key)
    failures = [name for name, valid in requirements.items() if not valid]
    if failures:
        raise ValueError("Disturbed OPD v1 requires: " + ", ".join(failures))
    # Surfaces the config rather than deferring a crash to the first update.
    return DisturbedOPDConfig.from_config(config.algorithm.disturbed_opd)
