# Disturbed OPD: implementation and repository hand-off

This document describes the Disturbed-OPD extension implemented on top of the
official OPDVR repository. Its primary purpose is repository hand-off: it
identifies the files that differ from upstream, explains where the algorithm is
implemented, and records the canonical training and evaluation entry points.

This document does not include model weights, datasets, checkpoints, W&B files,
or experiment outputs. Paths in the current scripts reflect the original
machine and may be adapted by the user of a new machine.

## 1. Scope and default setting

The main setting used by the current experiments is:

| Item | Value |
| --- | --- |
| Student | Qwen3-1.7B-Base |
| Teacher | Qwen3-4B-Base-GRPO |
| Training data | DAPO-Math-17K train |
| Validation data | six-benchmark, 1,588-question validation set |
| Chat template | Qwen3, `enable_thinking=false` |
| Training rollout | `n=1`, temperature 1.0, top-p 1.0 |
| Actor learning rate | `1e-6` |
| Actor loss aggregation | token mean |
| Epochs | 3, normally 165 global steps |
| Student master parameters | FP32 |
| Training forward | BF16 autocast by default |
| vLLM rollout dtype | BF16 |
| Offline evaluation | 1,588 questions, 16 rollouts per question, temperature 0.6 |

Disturbed OPD does not use the verifiable reward as its training advantage. The
token advantage is the sampled-token teacher/student log-probability gap:

\[
A_t = \log p_T(y_t\mid x,y_{<t}) - \log p_S(y_t\mid x,y_{<t}).
\]

With `ALPHA=1`, this is exactly the original sampled-token OPD signal.

## 2. Algorithm summary

For one student rollout batch, baseline OPD optimizes

\[
L_0(\theta)=-\frac{1}{N}\sum_t A_t\log p_\theta(y_t\mid x,y_{<t}).
\]

Disturbed OPD first computes the ordinary OPD gradient, constructs a temporary
probe displacement, evaluates the sampled-token log probabilities again, and
then restores the original parameters. Let

\[
\Delta_t=\log p_{\theta+\delta}(y_t\mid x,y_{<t})-
          \log p_\theta(y_t\mid x,y_{<t}).
\]

A token is disturbed when its own frozen-advantage surrogate becomes worse:

\[
A_t\Delta_t<0.
\]

The direction switch further separates:

- `up`: `A_t > 0` and `Delta_t < 0`; the teacher wants the sampled token to
  increase, but the joint batch update moves it down.
- `down`: `A_t < 0` and `Delta_t > 0`; the teacher wants the sampled token to
  decrease, but the joint batch update moves it up.
- `both`: the union of the two sets.

The selected tokens are then reweighted for the one real optimizer update. In
the recommended budget mode, unselected tokens retain weight 1 and selected
tokens receive extra weight subject to a global added-mass budget:

\[
\frac{1}{N}\sum_t (w_t-1)\leq B.
\]

The current default experimental budget is `B=0.2`. Continuous variants use a
detached severity score inside the same selected set. Opposite attenuation can
also reduce the weights of disturbed tokens with the opposite advantage sign.

The implementation additionally supports historical and diagnostic methods:

- `reweight`: the principal probe-and-reweight method;
- `real_repair`: take the baseline step and perform a local repair;
- `adam_budget`: select or budget using optimizer-aware candidate movement;
- `distribution_repair`: repair selected positions using teacher distribution
  information;
- `sign_hybrid`: combine sign-dependent repair strategies.

## 3. Canonical training entry point

The canonical training launcher is:

```text
opd_1.7b_4b_disturbed.sh
```

Do not use the historical `run_disturbed_queue_v*.sh` scripts as the general
training entry point. They contain fixed experiment paths and exist for exact
historical reproduction.

Example baseline-equivalent run:

```bash
DISTURBED_ENABLED=False \
MODEL_DTYPE=fp32 \
bash opd_1.7b_4b_disturbed.sh
```

Example binary Up-Budget run:

```bash
DISTURBED_ENABLED=True \
SELECT=probe \
DIRECTION=up \
WEIGHT_MODE=budget \
KAPPA=0.5 \
MASS_GROWTH_BUDGET=0.2 \
W_MAX=20 \
A_MIN_QUANTILE=0.2 \
ETA_PROBE=1e-4 \
PROBE_MOMENTUM=False \
bash opd_1.7b_4b_disturbed.sh
```

Important launcher variables:

| Variable | Meaning |
| --- | --- |
| `DISTURBED_ENABLED` | Enable the disturbed update; false uses ordinary OPD |
| `ALPHA` | Advantage transform strength; 1 reproduces the raw log-ratio |
| `ETA_PROBE` | Probe learning rate, independent of the real actor LR |
| `KAPPA` | Fraction of the theoretical correction bound used before caps |
| `W_MAX` | Maximum token weight |
| `A_MIN_QUANTILE` | Exclude the lowest global quantile of `abs(A)` from selection |
| `SELECT` | `probe`, `random`, `stratified_random`, or `entropy` |
| `DIRECTION` | `up`, `down`, or `both` |
| `PROBE_MOMENTUM` | Use current-batch raw direction when false; Adam momentum direction when true |
| `WEIGHT_MODE` | `mean_one`, `budget`, or `continuous_budget` |
| `MASS_GROWTH_BUDGET` | Maximum mean extra token-weight mass in budget modes |
| `OPPOSITE_WEIGHT_SCALE` | Scale the opposite disturbed sign set; 1 is no-op, 0 masks it |
| `CONTINUOUS_SCORE_MODE` | `delta`, `surrogate`, `rank`, or `topq` severity |
| `CONTINUOUS_SHUFFLE` | Shuffle severity within controlled strata for an ablation |
| `CONTINUOUS_TOP_FRACTION` | Selected fraction retained by the `topq` variant |
| `DATA_SHUFFLE` | Training dataloader shuffle switch |
| `AUDIT_DIR` | Non-empty path enables per-token Parquet audit output |
| `AUDIT_POST_UPDATE` | Save measurements after the real optimizer update |
| `AUDIT_DISTRIBUTION_KL_TOPK` | Teacher top-K width used for distribution-KL audit |
| `ROLLOUT_DATA_DIR` | Save training rollouts when non-empty |
| `VALIDATION_DATA_DIR` | Save validation rollouts when non-empty |

## 4. Files required on top of official OPDVR

### 4.1 Runtime-critical Disturbed-OPD files

These files must be committed together. The launcher alone is insufficient.

```text
opd_1.7b_4b_disturbed.sh

verl/verl/trainer/config/actor/actor.yaml
verl/verl/trainer/config/algorithm.py
verl/verl/trainer/config/ppo_trainer.yaml
verl/verl/trainer/ppo/ray_trainer.py
verl/verl/trainer/ppo/disturbed_opd.py
verl/verl/trainer/ppo/disturbed_updates.py

verl/verl/workers/actor/dp_actor.py
verl/verl/workers/config/actor.py
verl/verl/workers/config/rollout.py
verl/verl/workers/fsdp_workers.py
verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
```

Responsibilities:

| File | Responsibility |
| --- | --- |
| `opd_1.7b_4b_disturbed.sh` | Training setting and all Hydra overrides |
| `config/ppo_trainer.yaml` | Default schema for every Disturbed-OPD parameter |
| `config/algorithm.py` | Makes `disturbed_opd` part of the structured algorithm config |
| `config/actor/actor.yaml`, `workers/config/actor.py` | Forward precision controls used by the precision audit |
| `workers/config/rollout.py` | Declares the rollout seed used by controlled replications |
| `trainer/ppo/disturbed_opd.py` | CPU-testable selectors, probe deltas, budgets, weights, severity and diagnostics |
| `trainer/ppo/disturbed_updates.py` | Optimizer-aware and repair variants |
| `trainer/ppo/ray_trainer.py` | Validates the config and transfers required metadata/tensors to the actor |
| `workers/actor/dp_actor.py` | Executes the probe, restores parameters, selects tokens, computes the weighted loss and performs the real step |
| `workers/fsdp_workers.py` | Produces teacher top-K information for distribution repair and KL audits |
| `vllm_rollout_spmd.py` | Fixes validation output padding when `val_kwargs.max_tokens` is null |

### 4.2 Tests that should be committed

```text
verl/tests/trainer/ppo/test_disturbed_opd_on_cpu.py
verl/tests/trainer/ppo/test_disturbed_updates_on_cpu.py
```

Run them before scheduling GPU training:

```bash
pytest -q \
  verl/tests/trainer/ppo/test_disturbed_opd_on_cpu.py \
  verl/tests/trainer/ppo/test_disturbed_updates_on_cpu.py
```

### 4.3 Operational compatibility patches

The current working tree also contains these useful non-algorithm patches:

```text
verl/verl/utils/model.py
verl/verl/utils/tracking.py
```

- `model.py` contains attention-backend and Transformers compatibility fixes.
- `tracking.py` makes the W&B initialization timeout configurable through
  `WANDB_INIT_TIMEOUT`.

They are recommended for reproducing the current runtime behavior, although
they do not define the Disturbed-OPD objective.

### 4.4 Current experiment controllers

The following files record the latest experiment configurations:

```text
run_disturbed_continuous_next4.py
run_disturbed_opposite_attenuation_next3.py
run_disturbed_shuffle4_audit.py
run_disturbed_precision_audit.py
```

`run_disturbed_continuous_next4.py` is also imported as a utility module by the
newer controllers. The controllers contain experiment-specific predecessor
markers and output locations; they are valuable for exact reproduction but are
not the core algorithm.

## 5. Canonical avg@16/pass@16 evaluation

The canonical evaluator is currently:

```text
/root/jiangxitai/eval_pass16_opd.py
```

It must be copied into this repository before publishing the project. The
integrity checker already lives in the repository:

```text
verify_pass16_eval.py
```

The established evaluation protocol is:

| Item | Value |
| --- | --- |
| Questions | 1,588 |
| Samples per question | 16 |
| Temperature | 0.6 |
| Top-p | 1.0 |
| Sampling | enabled |
| Thinking | disabled |
| Maximum response length | 8,192 |
| Scorer | `verl.utils.reward_score.default_compute_score` |
| Sources | AIME24, AIME25, AMC, MATH500, Minerva, OlympiadBench |

First export the FSDP actor checkpoint to Hugging Face format:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/global_step_165/actor \
  --target_dir /path/to/hf_export
```

Then run the evaluator:

```bash
OPD_EVAL_ROOT=/path/to/eval_result \
OPD_EVAL_MODEL_NAME=my_model \
OPD_EVAL_MODEL_PATH=/path/to/hf_export \
OPD_EVAL_N=16 \
OPD_EVAL_TEMPERATURE=0.6 \
OPD_EVAL_TOP_P=1.0 \
OPD_EVAL_DO_SAMPLE=true \
python eval_pass16_opd.py prepare

OPD_EVAL_ROOT=/path/to/eval_result \
OPD_EVAL_MODEL_NAME=my_model \
OPD_EVAL_MODEL_PATH=/path/to/hf_export \
OPD_EVAL_N=16 \
OPD_EVAL_TEMPERATURE=0.6 \
OPD_EVAL_TOP_P=1.0 \
OPD_EVAL_DO_SAMPLE=true \
python eval_pass16_opd.py run --skip-wait

python verify_pass16_eval.py /path/to/eval_result my_model
```

The evaluator reports both:

- `avg@16`: mean correctness across all 16 sampled answers, which estimates
  sample-level pass@1 under this sampling policy;
- `pass@16`: fraction of questions for which at least one of 16 answers is
  correct.

Macro averages give equal weight to the six benchmarks. Micro averages weight
each question equally, so the larger datasets contribute more.

## 6. What is not required for Disturbed OPD

The teacher-takeover branch is separate from Disturbed OPD. Files such as
`main_takeover.py`, `teacher_takeover.py`, `takeover_controller.py`, and the
takeover launch scripts are not required to train the reweighting method.

The original `opdvr.sh`, `opd_baseline.sh`, `grpd.sh`, and
`ablation_inverse_gated.sh` remain useful upstream references but are not the
current Disturbed-OPD entry points. Historical `run_disturbed_queue_v*.sh` and
`run_pass16_*.sh` files encode completed experiment queues and fixed paths; they
are not required for a clean new run.

## 7. Files that must not be committed

Do not add runtime artifacts or large data to Git:

```text
checkpoints/
datasets/*.parquet
datasets/*.rollout_state/
outputs/
log/
wandb/
__pycache__/
*.pyc
*.whl
*.pid
*.lock
*_DONE
*_FAILED
```

Never commit a W&B API key or any other credential. Model weights and datasets
should be downloaded or generated separately on each machine.

## 8. Handoff checklist

1. Start from the intended official OPDVR revision.
2. Pull the commit containing every file in Section 4.1, not only the launcher.
3. Install the same Python/PyTorch/VERL/vLLM/FlashInfer environment.
4. Download the student and teacher weights.
5. Prepare the training and validation parquet files with the expected schema.
6. Run both CPU test files.
7. Inspect the resolved Hydra configuration with:

   ```bash
   bash opd_1.7b_4b_disturbed.sh --cfg job --resolve
   ```

8. Run a two-step smoke test before a full 8-GPU experiment.
9. Confirm W&B logging, rollout saving, validation and checkpoint writing.
10. Run the full experiment, export the final actor, and evaluate it with the
    exact protocol in Section 5.

