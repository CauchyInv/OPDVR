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
"""Optimizer-aware disturbed OPD, isolated from the historical reweight path.

All objectives use the same global valid-token denominator. Probe state is
read-only. Real-repair changes parameters twice but Adam state exactly once.
Only FULL_SHARD/SP=1 is supported: parameter inner products sum local shards.
"""

import math
import time

import torch

from verl.trainer.ppo.disturbed_opd import (
    abs_histogram,
    budget_lambda,
    lambda_from_sums,
    probe_delta,
    probe_mask,
    threshold_from_histogram,
)


def adamw_displacement(param, grad, state, group):
    """Read-only PyTorch AdamW candidate; grad is already globally clipped.

    Match the decay multiplication and the final addcdiv rounding, not just
    the algebraic displacement. Unsupported optimizer variants fail loudly.
    """
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("Only ordinary minimizing AdamW is supported")
    beta1, beta2 = group["betas"]
    step = int(state.get("step", 0)) + 1
    g = grad.detach().float()
    m = state.get("exp_avg", torch.zeros_like(g)).detach().float().clone()
    v = state.get("exp_avg_sq", torch.zeros_like(g)).detach().float().clone()
    m.lerp_(g, 1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    denom = v.sqrt().div_(math.sqrt(1 - beta2**step)).add_(group["eps"])
    after = param.detach().clone()
    after.mul_(1 - float(group["lr"]) * group.get("weight_decay", 0.0))
    after.addcdiv_(m, denom, value=-float(group["lr"]) / (1 - beta1**step))
    return after - param.detach()


def repair_token_loss(logp, before, advantage, mask, threshold, tolerance, huber, adv_cap):
    """Detached teacher direction; zero force once the old boundary is restored."""
    a = advantage.detach().float()
    residual = (-a.sign() * (logp.float() - before.detach().float()) - tolerance).clamp_min(0)
    eligible = mask.bool() & (a.abs() >= threshold)
    reliability = a.abs().clamp_max(adv_cap) * eligible
    # Huber is quadratic around zero and linear for severe damage.
    penalty = torch.where(residual <= huber, 0.5 * residual.square(), huber * (residual - 0.5 * huber))
    return reliability * penalty, residual


def coarse_forward_kl(student_logp, teacher_logp):
    """Teacher top-K plus ONE rest-of-vocabulary bucket, not renormalized top-K.

    Stable float32 probability arithmetic; network forward precision is not
    changed. Clamping only protects tiny roundoff at total mass=1.
    """
    s, t = student_logp.float(), teacher_logp.detach().float()
    ps, qt = s.exp(), t.exp()
    if bool((ps.sum(-1) > 1.001).any()) or bool((qt.sum(-1) > 1.001).any()):
        raise FloatingPointError("Top-K probability mass exceeds one")
    prest = (1 - ps.sum(-1)).clamp_min(1e-7)
    qrest = (1 - qt.sum(-1)).clamp_min(0)
    return (qt * (t - s)).sum(-1) + qrest * (qrest.clamp_min(1e-7).log() - prest.log())


def choose_adam_candidate(rows, rho):
    """rows=(lambda, -g.delta, c.delta); only strict selected improvement wins."""
    base = rows[0]
    if not all(math.isfinite(x) for x in base) or base[1] <= 0:
        return 0
    best = 0
    for i, row in enumerate(rows[1:], 1):
        if all(math.isfinite(x) for x in row) and row[1] >= rho * base[1] and row[2] < rows[best][2]:
            best = i
    return best


def update_disturbed_v5(actor, data, cfg):
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    from verl.utils.device import get_device_id
    from verl.utils.seqlen_balancing import prepare_dynamic_batch

    if actor.config.ppo_epochs != 1 or actor.config.use_kl_loss or actor.config.entropy_coeff != 0:
        raise ValueError("New disturbed methods require one PPO epoch, no KL/entropy auxiliary")
    if not isinstance(actor.actor_module, FullyShardedDataParallel):
        raise ValueError("v5 currently requires FSDP1 FULL_SHARD (explicit shard-inner-product semantics)")
    if actor.actor_module.sharding_strategy != ShardingStrategy.FULL_SHARD:
        raise ValueError("v5 requires FULL_SHARD, not replicated/hybrid parameters")
    actor.actor_module.train()
    device = get_device_id()
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    temperature = data.meta_info["temperature"]
    step = int(data.meta_info.get("global_steps", 0))
    audit = step <= 2 or step % cfg.audit_every == 0
    metrics = {}
    started = time.monotonic()

    def reduce(values):
        values = torch.as_tensor(values, dtype=torch.float64, device=device)
        if distributed:
            torch.distributed.all_reduce(values)
        return values

    def local(x):
        return actor._local_view(x).detach()

    def dot(xs, ys):
        total = torch.zeros((), dtype=torch.float64, device=device)
        for p in xs:
            total += (local(xs[p]).double() * local(ys[p]).double()).sum()
        return float(reduce(total))

    def finite_grad():
        norm = actor._clip_grad_norm()
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite v5 gradient; abort rather than silently skip")
        return float(norm)

    def grad_copy():
        return {
            p: p.grad.detach().clone()
            for group in actor.actor_optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        }

    keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
    keys += [k for k in ("rm_scores", "teacher_top_k_ids", "teacher_top_k_log_probs") if k in data.batch]
    if cfg.method in ("distribution_repair", "sign_hybrid") and "teacher_top_k_ids" not in keys:
        raise ValueError(f"{cfg.method} needs teacher top-K data, with sampled OPD reward unchanged")
    data = data.select(batch_keys=keys, non_tensor_batch_keys=[])
    count = float(reduce(data.batch["response_mask"].sum().to(device)))
    if count <= 0:
        raise ValueError("No valid response tokens")
    if actor.config.use_dynamic_bsz:
        batches, _ = prepare_dynamic_batch(data, max_token_len=actor.config.ppo_max_token_len_per_gpu)
    else:
        batches = data.split(actor.config.ppo_micro_batch_size_per_gpu)
    batches = [b.to(device) for b in batches]
    scale = world / count

    def forward(batch, topk=False):
        kwargs = {}
        if topk:
            kwargs = {"top_k": cfg.teacher_topk, "student_top_k_ids": batch.batch["teacher_top_k_ids"]}
        return actor._forward_micro_batch(dict(batch.batch), temperature=temperature, **kwargs)

    actor.actor_optimizer.zero_grad()
    cached = []
    hist = torch.zeros(cfg.quantile_bins + 1, dtype=torch.int64, device=device)
    base_loss = 0.0
    for b in batches:
        _, u, *_ = forward(b)
        a = actor._disturbed_advantages(dict(b.batch), cfg.alpha).float()
        mask = b.batch["response_mask"].bool()
        loss = -(a * u * mask).sum() * scale
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite baseline loss")
        loss.backward()
        base_loss += float(loss.detach())
        cached.append({"u0": u.detach().float(), "a": a, "mask": mask})
        hist += abs_histogram(a, mask, cfg.quantile_bins, cfg.quantile_max, cfg.quantile_min)
    if distributed:
        torch.distributed.all_reduce(hist)
    threshold = threshold_from_histogram(hist, cfg.a_min_quantile, cfg.quantile_max, cfg.quantile_min)
    g = grad_copy() if cfg.method == "adam_budget" else None  # MUST be pre-clip.
    metrics["disturbed/probe_grad_norm"] = finite_grad()
    params = [p for group in actor.actor_optimizer.param_groups for p in group["params"] if p.grad is not None]
    groups = {p: group for group in actor.actor_optimizer.param_groups for p in group["params"]}
    before = {p: p.detach().clone() for p in params}

    if cfg.method == "real_repair":
        actor.actor_optimizer.step()  # Exactly one moment/scheduler advance.
        delta0 = {p: p.detach() - before[p] for p in params}
        base_norm = math.sqrt(max(dot(delta0, delta0), 0))
        del before, delta0
        actor.actor_optimizer.zero_grad()
        repair_loss = 0.0
        for b, e in zip(batches, cached, strict=True):
            _, u, *_ = forward(b)
            penalties, residual = repair_token_loss(
                u,
                e["u0"],
                e["a"],
                e["mask"],
                threshold,
                cfg.repair_tolerance,
                cfg.repair_huber,
                cfg.repair_adv_cap,
            )
            e["baseline_change"] = u.detach().float() - e["u0"]
            e["selected"] = (residual.detach() > 0) & e["mask"] & (e["a"].abs() >= threshold)
            loss = penalties.sum() * scale
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite repair loss")
            loss.backward()
            repair_loss += float(loss.detach())
        metrics["repair/grad_norm"] = finite_grad()
        directions = {}
        for p in params:
            if p.grad is None:
                continue
            state, group = actor.actor_optimizer.state[p], groups[p]
            vhat = state["exp_avg_sq"] / (1 - group["betas"][1] ** int(state["step"]))
            directions[p] = -p.grad.detach() / (vhat.sqrt() + group["eps"])
        direction_norm = math.sqrt(max(dot(directions, directions), 0))
        # One bounded stateless correction. No Adam first moment, no decay.
        amount = cfg.repair_norm_fraction * base_norm / max(direction_norm, 1e-30)
        if not math.isfinite(amount):
            raise FloatingPointError("Nonfinite repair displacement")
        with torch.no_grad():
            for p, d in directions.items():
                p.add_(d, alpha=amount)
        metrics.update(
            {
                "repair/loss": repair_loss,
                "repair/baseline_delta_norm": base_norm,
                "repair/delta_norm": amount * direction_norm,
                "repair/norm_fraction": amount * direction_norm / max(base_norm, 1e-30),
                "disturbed/optimizer_steps": 1.0,
                "repair/parameter_updates": 2.0,
            }
        )
        del directions
    else:
        with torch.no_grad():
            for p in params:
                state, group = actor.actor_optimizer.state.get(p, {}), groups[p]
                d = probe_delta(
                    p.grad,
                    state.get("exp_avg_sq"),
                    int(state.get("step", 0)) + 1,
                    cfg.eta_probe,
                    group["betas"][1],
                    group["eps"],
                    exp_avg=state.get("exp_avg"),
                    use_momentum=cfg.probe_momentum,
                    beta1=group["betas"][0],
                )
                p.add_(d)
        sums = torch.zeros(3, dtype=torch.float64, device=device)
        hybrid_down = torch.zeros(2, dtype=torch.float64, device=device)
        try:
            with torch.no_grad():
                for b, e in zip(batches, cached, strict=True):
                    _, shifted, *_ = forward(b)
                    change = shifted.float() - e["u0"]
                    if cfg.method == "sign_hybrid":
                        e["selected_up"] = probe_mask(e["a"], change, e["mask"], threshold, "up")
                        e["selected_down"] = probe_mask(e["a"], change, e["mask"], threshold, "down")
                        e["selected"] = e["selected_up"] | e["selected_down"]
                        budget_selected = e["selected_up"]
                    else:
                        e["selected"] = probe_mask(e["a"], change, e["mask"], threshold, cfg.direction)
                        budget_selected = e["selected"]
                    product = e["a"] * change * e["mask"]
                    sums += torch.stack(
                        (
                            product.double().sum(),
                            (product * budget_selected).double().sum(),
                            budget_selected.double().sum(),
                        )
                    )
                    if cfg.method == "sign_hybrid":
                        hybrid_down += torch.stack(
                            (
                                (product * e["selected_down"]).double().sum(),
                                e["selected_down"].double().sum(),
                            )
                        )
        finally:
            with torch.no_grad():
                for p in params:
                    p.copy_(before[p])
        gain, damage, nsel = reduce(sums).tolist()
        hybrid_down = reduce(hybrid_down)
        raw, _, _ = lambda_from_sums(gain, damage, cfg.kappa)
        lam = min(budget_lambda(raw, nsel / count, cfg.mass_growth_budget), cfg.w_max - 1)
        metrics.update({"disturbed/gain": gain, "disturbed/damage": damage, "disturbed/lambda_uncapped": raw})
        actor.actor_optimizer.zero_grad()
        aux_loss = 0.0
        for b, e in zip(batches, cached, strict=True):
            if cfg.method == "adam_budget":
                _, u, *_ = forward(b)
                loss = -(e["selected"] * e["a"] * u).sum() * scale
            elif cfg.method == "distribution_repair":
                _, u, _, sk = forward(b, topk=True)
                # Padded gathered log-probs are zero, not a distribution.
                # Exclude them BEFORE rest-bucket construction/mass checks.
                kl = coarse_forward_kl(sk[e["mask"]], b.batch["teacher_top_k_log_probs"][e["mask"]])
                loss = (
                    -(e["a"] * u * e["mask"]).sum() + cfg.distribution_coef * (e["selected"][e["mask"]] * kl).sum()
                ) * scale
            else:
                # Sign-split supervision: sampled-token rescue is appropriate
                # for teacher-preferred tokens, while teacher top-K forward KL
                # tells negative-advantage positions where probability mass
                # should move instead of merely pushing the sampled token down.
                _, u, _, sk = forward(b, topk=True)
                kl = coarse_forward_kl(sk[e["mask"]], b.batch["teacher_top_k_log_probs"][e["mask"]])
                loss = (
                    -(e["a"] * u * e["mask"]).sum()
                    - lam * (e["selected_up"] * e["a"] * u).sum()
                    + cfg.distribution_coef * (e["selected_down"][e["mask"]] * kl).sum()
                ) * scale
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite v5 auxiliary loss")
            loss.backward()
            aux_loss += float(loss.detach())
        if cfg.method == "adam_budget":
            c = grad_copy()
            if set(g) != set(c):
                raise ValueError("Baseline/auxiliary gradient parameter sets differ")
            rows = []
            candidates = sorted(set([0.0, lam / 4, lam / 2, lam]))
            delta0 = None
            for candidate in candidates:
                with torch.no_grad():
                    for p in params:
                        p.grad.copy_(g[p] + candidate * c[p])
                finite_grad()  # Reclip each candidate, never clip c separately.
                deltas = {
                    p: adamw_displacement(p, p.grad, actor.actor_optimizer.state.get(p, {}), groups[p]) for p in params
                }
                rows.append((candidate, -dot(g, deltas), dot(c, deltas)))
                if candidate == 0:
                    delta0 = deltas
            best = choose_adam_candidate(rows, cfg.adam_progress_fraction)
            lam = candidates[best]
            with torch.no_grad():
                for p in params:
                    p.grad.copy_(g[p] + lam * c[p])
            metrics["actor/grad_norm"] = finite_grad()
            predicted = {
                p: adamw_displacement(p, p.grad, actor.actor_optimizer.state.get(p, {}), groups[p]) for p in params
            }
            metrics.update(
                {
                    "adam_budget/baseline_gain": rows[0][1],
                    "adam_budget/chosen_gain": rows[best][1],
                    "adam_budget/baseline_selected_change": rows[0][2],
                    "adam_budget/chosen_selected_change": rows[best][2],
                    "adam_budget/fallback": float(best == 0),
                    "disturbed/lambda": lam,
                    "disturbed/weight_mass_growth": lam * nsel / count,
                }
            )
            if audit:
                # Counterfactual baseline at the SAME optimizer state. This is
                # measurement only: restore exactly before the chosen real step.
                try:
                    with torch.no_grad():
                        for p in params:
                            p.add_(delta0[p])
                        for b, e in zip(batches, cached, strict=True):
                            _, u, *_ = forward(b)
                            e["baseline_change"] = u.float() - e["u0"]
                finally:
                    with torch.no_grad():
                        for p in params:
                            p.copy_(before[p])
            del g, c, delta0, deltas
        else:
            metrics["actor/grad_norm"] = finite_grad()
            predicted = None
            metrics["distribution/coef"] = cfg.distribution_coef
            metrics["distribution/topk"] = cfg.teacher_topk
            if cfg.method == "sign_hybrid":
                metrics.update(
                    {
                        "disturbed/lambda": lam,
                        "disturbed/weight_mass_growth": lam * nsel / count,
                        "hybrid/up_selected_fraction": nsel / count,
                        "hybrid/down_selected_fraction": float(hybrid_down[1]) / count,
                        "hybrid/down_damage": float(hybrid_down[0]),
                    }
                )
        actor.actor_optimizer.step()
        if predicted is not None:
            actual = {p: p.detach() - before[p] for p in params}
            error = {p: actual[p] - predicted[p] for p in params}
            relative = math.sqrt(max(dot(error, error), 0) / max(dot(actual, actual), 1e-30))
            metrics["adam_budget/simulator_relative_error"] = relative
            if relative > 0.01:
                raise RuntimeError(f"Adam simulator mismatch {relative}; refusing misleading experiment")
            del actual, error, predicted
        del before
        metrics["disturbed/optimizer_steps"] = 1.0
        metrics["disturbed/aux_loss"] = aux_loss

    actor.actor_optimizer.zero_grad()
    selcounts = reduce(
        torch.stack(
            [sum(e["selected"].sum() for e in cached), sum((e["selected"] & (e["a"] < 0)).sum() for e in cached)]
        )
    )
    metrics.update(
        {
            "disturbed/selected_fraction": float(selcounts[0]) / count,
            "disturbed/selected_share_negative": float(selcounts[1] / selcounts[0].clamp_min(1)),
            "disturbed/a_min_threshold": threshold,
            "actor/pg_loss": base_loss,
        }
    )
    metrics.setdefault("actor/grad_norm", metrics["disturbed/probe_grad_norm"])
    if audit:
        sums = torch.zeros(9, dtype=torch.float64, device=device)
        with torch.no_grad():
            for b, e in zip(batches, cached, strict=True):
                _, u, *_ = forward(b)
                change = u.float() - e["u0"]
                loss_change = -e["a"] * change
                selected, mask = e["selected"], e["mask"]
                sums += torch.stack(
                    (
                        (loss_change * mask).double().sum(),
                        (loss_change.clamp_min(0) * selected).double().sum(),
                        (loss_change.clamp_min(0) * mask * ~selected).double().sum(),
                        ((loss_change > 0) & selected).double().sum(),
                        ((loss_change + 0.5 * change.square()) * mask).double().sum(),
                        (-e["a"] * e.get("baseline_change", torch.zeros_like(change)))
                        .clamp_min(0)
                        .mul(selected)
                        .double()
                        .sum(),
                        selected.double().sum(),
                        (-e["a"] * e.get("baseline_change", torch.zeros_like(change)))
                        .clamp_min(0)
                        .mul(mask & ~selected)
                        .double()
                        .sum(),
                        (-e["a"] * e.get("baseline_change", torch.zeros_like(change))).mul(mask).double().sum(),
                    )
                )
        sums = reduce(sums)
        labels = [
            "surrogate_change",
            "selected_damage",
            "other_damage",
            "selected_still_harmed",
            "k2_proxy_change",
            "baseline_selected_damage",
            "selected_count",
            "baseline_other_damage",
            "baseline_surrogate_change",
        ]
        metrics.update({f"audit/{key}": float(value) / count for key, value in zip(labels, sums, strict=True)})
        metrics["audit/selected_still_harmed_rate"] = float(sums[3] / sums[6].clamp_min(1))
        if cfg.method in ("real_repair", "adam_budget"):
            metrics["audit/selected_damage_rescue_fraction"] = float((sums[5] - sums[1]) / sums[5].clamp_min(1e-30))
            metrics["audit/other_damage_added"] = float(sums[2] - sums[7]) / count
        else:
            for key in ("baseline_selected_damage", "baseline_other_damage", "baseline_surrogate_change"):
                metrics.pop(f"audit/{key}")
    metrics["timing_s/disturbed_v5_update"] = time.monotonic() - started
    return metrics
