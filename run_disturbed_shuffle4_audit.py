#!/usr/bin/env python3
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
"""Strict shuffle=True replication queue with rollout and token audits.

The four runs preserve their original epoch counts and disturbed-OPD settings.
The baseline uses the already-validated kappa=0 shadow-probe path: its real
weights are exactly one, while the virtual probe exposes comparable up/down
token identities for the mechanism audit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

import run_disturbed_continuous_next4 as core

REPO = Path("/root/OPDVR")
PY = "/root/miniconda3/envs/verl/bin/python"
PREDECESSOR = Path(
    "/data/jiangxitai/OPDVR/disturbed_opposite_attenuation_next3_20261004/ALL_DONE"
)
ROOT = Path("/data/jiangxitai/OPDVR/disturbed_shuffle4_audit_20261006")
RECORD = Path("/root/jiangxitai/codex_log")
MONITOR_LOG = RECORD / "disturbed_shuffle4_audit_monitor.log"


@dataclass(frozen=True)
class Experiment:
    name: str
    epochs: int
    steps: int
    direction: str
    select: str
    kappa: float
    w_max: float


EXPERIMENTS = (
    Experiment("baseline_fp32_shuffle", 3, 165, "both", "probe", 0.0, 20.0),
    Experiment("up_budget_shuffle", 3, 165, "up", "probe", 0.5, 20.0),
    Experiment("down_budget_shuffle", 1, 55, "down", "probe", 0.5, 20.0),
    # Exact replication of the prior random_down_lambda_matched control.
    Experiment("random_down_budget_shuffle", 1, 55, "down", "stratified_random", 0.5, 1.96),
)

AUDIT_METRICS = (
    "audit/eos_count",
    "audit/eos_up_fraction",
    "audit/eos_down_fraction",
    "audit/last_up_fraction",
    "audit/last_down_fraction",
    "audit/up_entropy_mean",
    "audit/down_entropy_mean",
    "audit/up_high_entropy_share",
    "audit/down_high_entropy_share",
    "audit/up_entropy_point_biserial",
    "audit/down_entropy_point_biserial",
    "audit/up_kl_mean",
    "audit/down_kl_mean",
    "audit/up_high_kl_share",
    "audit/down_high_kl_share",
    "audit/up_kl_point_biserial",
    "audit/down_kl_point_biserial",
)


def event(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}"
    print(line, flush=True)
    RECORD.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a") as stream:
        stream.write(line + "\n")


def wait_for_predecessor() -> None:
    last = 0.0
    while not PREDECESSOR.is_file():
        if time.monotonic() - last >= 1200:
            event("WAIT predecessor opposite-attenuation queue")
            last = time.monotonic()
        time.sleep(30)
    event("PREDECESSOR_COMPLETE opposite-attenuation ALL_DONE found")


def data_paths(exp: Experiment, smoke: bool = False) -> dict[str, Path]:
    prefix = "smoke_" if smoke else ""
    root = ROOT / "mechanism_data" / f"{prefix}{exp.name}"
    return {
        "root": root,
        "audit": root / "token_audit",
        "train_rollouts": root / "train_rollouts",
        "validation_rollouts": root / "validation_rollouts",
    }


def base_env(exp: Experiment, smoke: bool = False) -> dict[str, str]:
    paths = data_paths(exp, smoke)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    Path("/data/tmp/sh4").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    curand = Path("/root/miniconda3/envs/verl/lib/python3.12/site-packages/nvidia/curand/include")
    if not (curand / "curand.h").is_file():
        raise RuntimeError(f"Missing FlashInfer build header: {curand / 'curand.h'}")
    env["CPATH"] = str(curand) + os.pathsep + env.get("CPATH", "")
    env["LIBRARY_PATH"] = "/root/miniconda3/envs/verl/lib" + os.pathsep + env.get("LIBRARY_PATH", "")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(
        {
            "DISTURBED_ENABLED": "True",
            "ALPHA": "1.0",
            "ETA_PROBE": "1e-4",
            "KAPPA": str(exp.kappa),
            "A_MIN_QUANTILE": "0.2",
            "PROBE_MOMENTUM": "False",
            "PROBE_DIAGNOSTIC": "False",
            "SELECT": exp.select,
            "DIRECTION": exp.direction,
            "WEIGHT_MODE": "budget",
            "MASS_GROWTH_BUDGET": "0.2",
            "OPPOSITE_WEIGHT_SCALE": "1.0",
            "W_MAX": str(exp.w_max),
            "STRATIFIED_BINS": "16",
            "MODEL_DTYPE": "fp32",
            "ROLLOUT_DTYPE": "bfloat16",
            "DATA_SHUFFLE": "True",
            "AUDIT_DIR": str(paths["audit"]),
            "AUDIT_POST_UPDATE": "True",
            "AUDIT_TOPK": "0",
            "AUDIT_DISTRIBUTION_KL_TOPK": "32",
            "AUDIT_EOS_TOKEN_ID": "151643",
            "ROLLOUT_DATA_DIR": str(paths["train_rollouts"]),
            "VALIDATION_DATA_DIR": str(paths["validation_rollouts"]),
            "WANDB_MODE": "offline" if smoke else "online",
            "VAL_BEFORE_TRAIN": "False" if smoke else "True",
            "MINI_BATCH_SIZE": "256",
            "MAX_RESP_LENGTH": "8192",
            "MAX_PROMPT_LENGTH": "1024",
            "NGPUS_PER_NODE": "8",
            "ROLLOUT_SEED": "0",
            "ACTOR_LR": "1e-6",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
            "PYTHONPATH": str(REPO / "verl"),
            "PYTHONUNBUFFERED": "1",
            "EXPERIMENT_NAME": f"opd_1.7b_4b_{exp.name}{'_smoke' if smoke else ''}",
            "RAY_TMPDIR": "/data/ray_sh4",
            "TMPDIR": "/data/tmp/sh4",
        }
    )
    return env


def snapshot(exp: Experiment) -> None:
    target = ROOT / exp.name / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    files = (
        "verl/verl/trainer/ppo/disturbed_opd.py",
        "verl/verl/workers/actor/dp_actor.py",
        "verl/verl/workers/fsdp_workers.py",
        "verl/verl/trainer/ppo/ray_trainer.py",
        "verl/verl/trainer/config/ppo_trainer.yaml",
        "verl/tests/trainer/ppo/test_disturbed_opd_on_cpu.py",
        "opd_1.7b_4b_disturbed.sh",
        "run_disturbed_shuffle4_audit.py",
    )
    hashes = {}
    for relative in files:
        payload = (REPO / relative).read_bytes()
        (target / relative.replace("/", "__")).write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    (target / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def verify_rows(exp: Experiment, rows: dict[int, dict[str, float]], steps: int) -> None:
    if set(rows) != set(range(1, steps + 1)):
        raise RuntimeError(f"{exp.name}: metric steps differ from 1..{steps}")
    for step, row in rows.items():
        missing = [key for key in AUDIT_METRICS if key not in row]
        if missing:
            raise RuntimeError(f"{exp.name} step {step} lacks audit metrics: {missing}")
        if row.get("audit/distribution_kl_topk") != 32.0:
            raise RuntimeError(f"{exp.name} step {step} did not use top-32+tail KL")
        if exp.kappa == 0 and abs(row.get("disturbed/weight_mass_growth", 1.0)) > 1e-6:
            raise RuntimeError(f"baseline shadow probe changed token-weight mass at step {step}")
        if exp.direction == "up" and row.get("disturbed/selected_share_negative") != 0.0:
            raise RuntimeError(f"{exp.name} step {step}: selected set is not A>0")
        if exp.direction == "down" and row.get("disturbed/selected_share_negative") != 1.0:
            raise RuntimeError(f"{exp.name} step {step}: selected set is not A<0")


def verify_artifacts(exp: Experiment, steps: int, smoke: bool = False) -> None:
    paths = data_paths(exp, smoke)
    expected_columns = {
        "token_id",
        "mask_up",
        "mask_down",
        "entropy_before",
        "teacher_entropy",
        "distribution_kl_before",
        "distribution_kl_after",
        "is_eos",
        "is_last_token",
    }
    for step in range(1, steps + 1):
        shards = sorted((paths["audit"] / f"step_{step:05d}").glob("rank_*.parquet"))
        if len(shards) != 8:
            raise RuntimeError(f"{exp.name} step {step}: expected 8 audit shards, got {len(shards)}")
        if step == 1:
            names = set(pq.read_schema(shards[0]).names)
            if missing := expected_columns - names:
                raise RuntimeError(f"{exp.name} audit schema lacks {sorted(missing)}")
        rollout = paths["train_rollouts"] / f"{step}.jsonl"
        if not rollout.is_file() or rollout.stat().st_size == 0:
            raise RuntimeError(f"{exp.name} missing training rollout {rollout}")
    if not smoke:
        validation = sorted(paths["validation_rollouts"].glob("*.jsonl"))
        if not validation or any(path.stat().st_size == 0 for path in validation):
            raise RuntimeError(f"{exp.name} has no complete validation rollout dumps")


def save_step_metrics(exp: Experiment, rows: dict[int, dict[str, float]]) -> None:
    target = data_paths(exp)["root"] / "step_relation_metrics.jsonl"
    with target.open("w") as stream:
        for step in sorted(rows):
            selected = {
                "global_step": step,
                **{key: value for key, value in rows[step].items() if key.startswith("audit/")},
                **{
                    key: rows[step][key]
                    for key in (
                        "disturbed/selected_fraction",
                        "disturbed/lambda",
                        "disturbed/weight_mass_growth",
                        "response_length/mean",
                    )
                    if key in rows[step]
                },
            }
            stream.write(json.dumps(selected, ensure_ascii=False) + "\n")


def run_smoke(exp: Experiment) -> None:
    done = ROOT / f"{exp.name}.SMOKE_DONE"
    log = ROOT / f"smoke_{exp.name}.log"
    if not done.is_file():
        core.wait_gpus()
        env = base_env(exp, smoke=True)
        env.update(
            {
                "FINAL_CKPT_DIR": str(ROOT / f"smoke_{exp.name}"),
                "TEST_FREQ": "-1",
                "SAVE_FREQ": "-1",
                "TOTAL_EPOCHS": "1",
            }
        )
        core.monitor(
            ["bash", str(REPO / "opd_1.7b_4b_disturbed.sh"), "trainer.total_training_steps=2"],
            env,
            log,
            f"smoke:{exp.name}",
        )
        rows = core.parse_rows(log)
        verify_rows(exp, rows, 2)
        verify_artifacts(exp, 2, smoke=True)
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    else:
        verify_rows(exp, core.parse_rows(log), 2)
        verify_artifacts(exp, 2, smoke=True)


def train(exp: Experiment) -> Path:
    directory = ROOT / exp.name
    log = ROOT / f"train_{exp.name}.log"
    done = ROOT / f"{exp.name}.step{exp.steps}.TRAIN_DONE"
    if not done.is_file():
        core.wait_gpus()
        snapshot(exp)
        env = base_env(exp)
        env.update(
            {
                "TOTAL_EPOCHS": str(exp.epochs),
                "SAVE_FREQ": "55" if exp.steps == 55 else "110",
                "TEST_FREQ": "11" if exp.steps == 55 else "22",
                "FINAL_CKPT_DIR": str(directory),
            }
        )
        core.monitor(["bash", str(REPO / "opd_1.7b_4b_disturbed.sh")], env, log, f"train:{exp.name}")
        core.verify_checkpoint(directory, exp.steps, log)
        rows = core.parse_rows(log)
        verify_rows(exp, rows, exp.steps)
        verify_artifacts(exp, exp.steps)
        save_step_metrics(exp, rows)
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    else:
        core.verify_checkpoint(directory, exp.steps, log)
        rows = core.parse_rows(log)
        verify_rows(exp, rows, exp.steps)
        verify_artifacts(exp, exp.steps)
        save_step_metrics(exp, rows)
    return directory / f"global_step_{exp.steps}" / "actor"


def audit_means(exp: Experiment) -> dict[str, float]:
    rows = core.parse_rows(ROOT / f"train_{exp.name}.log")
    return {
        key: sum(row[key] for row in rows.values()) / len(rows)
        for key in AUDIT_METRICS
    }


def write_summary(results: dict[str, dict]) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "results": results}
    (ROOT / "FINAL_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# `data.shuffle=True` four-run replication with mechanism audit",
        "",
        "All runs save every training rollout, every scheduled validation rollout, and per-token/per-step audit data.",
        "KL is teacher top-32 + tail coarsened forward KL, a lower bound on full-vocabulary KL.",
        "",
        "| Experiment | Epochs | Macro avg@16 | Micro avg@16 | Macro pass@16 | Mean length | EOS up | EOS down | Last up | Last down | Up H | Down H | Up KL | Down KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_name = {exp.name: exp for exp in EXPERIMENTS}
    for name, item in results.items():
        metrics = item["eval"]["metrics"]
        audit = item["audit_step_mean"]
        lines.append(
            f"| {name} | {by_name[name].epochs} | {100 * metrics['macro_6sets']['avg@16']:.3f}% | "
            f"{100 * metrics['overall_micro']['avg@16']:.3f}% | "
            f"{100 * metrics['macro_6sets']['pass@16']:.3f}% | "
            f"{metrics['overall_micro']['mean_response_tokens']:.1f} | "
            f"{100 * audit['audit/eos_up_fraction']:.2f}% | "
            f"{100 * audit['audit/eos_down_fraction']:.2f}% | "
            f"{100 * audit['audit/last_up_fraction']:.2f}% | "
            f"{100 * audit['audit/last_down_fraction']:.2f}% | "
            f"{audit['audit/up_entropy_mean']:.4f} | {audit['audit/down_entropy_mean']:.4f} | "
            f"{audit['audit/up_kl_mean']:.4f} | {audit['audit/down_kl_mean']:.4f} |"
        )
    report = "\n".join(lines) + "\n"
    (ROOT / "FINAL_RESULTS.md").write_text(report)
    (RECORD / "disturbed_shuffle4_audit_results.md").write_text(report)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    core.ROOT = ROOT
    core.RECORD = RECORD
    core.MONITOR_LOG = MONITOR_LOG
    core.base_env = base_env
    lock = (ROOT / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another shuffle-audit controller is already running") from exc
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    event("QUEUE_REGISTERED order=" + ",".join(exp.name for exp in EXPERIMENTS))
    wait_for_predecessor()
    results = {}
    for exp in EXPERIMENTS:
        run_smoke(exp)
        actor = train(exp)
        model = core.export(exp, actor)
        evaluated = core.evaluate(exp, model)
        results[exp.name] = {"eval": evaluated, "audit_step_mean": audit_means(exp)}
        event(f"COMPLETE {exp.name}: smoke/train/rollouts/audit/export/avg16 verified")
    write_summary(results)
    (ROOT / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    event("QUEUE_COMPLETE all four shuffle=True audited replications verified")


if __name__ == "__main__":
    main()
