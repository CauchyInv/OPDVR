#!/usr/bin/env python3
"""Queue three opposite-sign attenuation ablations after continuous-next4.

Order:
  1. up budget, down disturbed tokens weighted 0.5;
  2. up budget, down disturbed tokens weighted 0.0;
  3. down budget, up disturbed tokens weighted 0.0.

Each experiment runs a two-step smoke test, a fresh 165-step/3-epoch training
run, exports the final FSDP checkpoint, and evaluates 1588 prompts with n=16.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import run_disturbed_continuous_next4 as core

REPO = Path("/root/OPDVR")
PY = "/root/miniconda3/envs/verl/bin/python"
PREDECESSOR = Path("/data/jiangxitai/OPDVR/disturbed_continuous_next4_20261004/ALL_DONE")
PREDECESSOR_LOCK = PREDECESSOR.parent / "controller.lock"
ROOT = Path("/data/jiangxitai/OPDVR/disturbed_opposite_attenuation_next3_20261004")
RECORD = Path("/root/jiangxitai/codex_log")
MONITOR_LOG = RECORD / "disturbed_opposite_attenuation_next3_monitor.log"


@dataclass(frozen=True)
class Experiment:
    name: str
    direction: str
    opposite_scale: float


EXPERIMENTS = (
    Experiment("up_budget_down_attenuation_mu0.5", "up", 0.5),
    Experiment("up_budget_down_attenuation_mu0.0", "up", 0.0),
    Experiment("down_budget_up_attenuation_mu0.0", "down", 0.0),
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
            pid = PREDECESSOR_LOCK.read_text().strip() if PREDECESSOR_LOCK.is_file() else "unknown"
            event(f"WAIT predecessor continuous-next4 pid={pid}")
            last = time.monotonic()
        time.sleep(30)
    event("PREDECESSOR_COMPLETE continuous-next4 ALL_DONE found")


def base_env(exp: Experiment) -> dict[str, str]:
    Path("/data/tmp/oa").mkdir(parents=True, exist_ok=True)
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
            "KAPPA": "0.5",
            "A_MIN_QUANTILE": "0.2",
            "PROBE_MOMENTUM": "False",
            "PROBE_DIAGNOSTIC": "False",
            "SELECT": "probe",
            "DIRECTION": exp.direction,
            "WEIGHT_MODE": "budget",
            "MASS_GROWTH_BUDGET": "0.2",
            "OPPOSITE_WEIGHT_SCALE": str(exp.opposite_scale),
            "W_MAX": "20.0",
            "STRATIFIED_BINS": "16",
            "MODEL_DTYPE": "fp32",
            "ROLLOUT_DTYPE": "bfloat16",
            "WANDB_MODE": "online",
            "VAL_BEFORE_TRAIN": "True",
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
            "EXPERIMENT_NAME": f"opd_1.7b_4b_{exp.name}",
            "RAY_TMPDIR": "/data/ray_opposite_attenuation",
            # vLLM v1 creates an AF_UNIX/ZMQ endpoint below TMPDIR. Linux caps
            # sockaddr_un.sun_path at 107 bytes, so the experiment root is too
            # long for evaluation workers. Keep the temp root deliberately
            # short; artifacts remain under ROOT.
            "TMPDIR": "/data/tmp/oa",
        }
    )
    return env


def snapshot(exp: Experiment) -> None:
    target = ROOT / exp.name / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    files = (
        "verl/verl/trainer/ppo/disturbed_opd.py",
        "verl/verl/workers/actor/dp_actor.py",
        "verl/verl/trainer/config/ppo_trainer.yaml",
        "verl/tests/trainer/ppo/test_disturbed_opd_on_cpu.py",
        "opd_1.7b_4b_disturbed.sh",
        "run_disturbed_opposite_attenuation_next3.py",
    )
    hashes = {}
    for relative in files:
        payload = (REPO / relative).read_bytes()
        (target / relative.replace("/", "__")).write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    (target / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def verify_metrics(exp: Experiment, rows: dict[int, dict[str, float]], expected_steps: int) -> None:
    if len(rows) != expected_steps or set(rows) != set(range(1, expected_steps + 1)):
        raise RuntimeError(f"{exp.name} has {len(rows)} rows, expected steps 1..{expected_steps}")
    for step, row in rows.items():
        if row.get("disturbed/opposite_weight_scale") != exp.opposite_scale:
            raise RuntimeError(f"{exp.name} step {step} has wrong opposite scale")
        if "disturbed/opposite_selected_fraction" not in row:
            raise RuntimeError(f"{exp.name} step {step} lacks opposite-set metrics")
        expected_removed = (1.0 - exp.opposite_scale) * row["disturbed/opposite_selected_fraction"]
        if abs(row.get("disturbed/opposite_removed_mass_fraction", -1.0) - expected_removed) > 1e-6:
            raise RuntimeError(f"{exp.name} step {step} has inconsistent removed mass")
        if exp.direction == "up" and row.get("disturbed/selected_share_negative") != 0.0:
            raise RuntimeError(f"{exp.name} step {step} primary set is not purely A>0")
        if exp.direction == "down" and row.get("disturbed/selected_share_negative") != 1.0:
            raise RuntimeError(f"{exp.name} step {step} primary set is not purely A<0")


def run_smoke(exp: Experiment) -> None:
    done = ROOT / f"{exp.name}.SMOKE_DONE"
    log = ROOT / f"smoke_{exp.name}.log"
    if done.is_file():
        verify_metrics(exp, core.parse_rows(log), 2)
        return
    env = base_env(exp)
    env.update(
        {
            "FINAL_CKPT_DIR": str(ROOT / f"smoke_{exp.name}"),
            "WANDB_MODE": "offline",
            "VAL_BEFORE_TRAIN": "False",
            "TEST_FREQ": "-1",
            "SAVE_FREQ": "-1",
            "TOTAL_EPOCHS": "1",
        }
    )
    core.wait_gpus()
    core.monitor(
        ["bash", str(REPO / "opd_1.7b_4b_disturbed.sh"), "trainer.total_training_steps=2"],
        env,
        log,
        f"smoke:{exp.name}",
    )
    verify_metrics(exp, core.parse_rows(log), 2)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


def train(exp: Experiment) -> Path:
    done = ROOT / f"{exp.name}.step165.TRAIN_DONE"
    directory = ROOT / exp.name
    log = ROOT / f"train_{exp.name}.log"
    if not done.is_file():
        core.wait_gpus()
        snapshot(exp)
        env = base_env(exp)
        env.update(
            {
                "TOTAL_EPOCHS": "3",
                "SAVE_FREQ": "110",
                "TEST_FREQ": "22",
                "FINAL_CKPT_DIR": str(directory),
            }
        )
        core.monitor(["bash", str(REPO / "opd_1.7b_4b_disturbed.sh")], env, log, f"train:{exp.name}")
        core.verify_checkpoint(directory, 165, log)
        verify_metrics(exp, core.parse_rows(log), 165)
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    else:
        core.verify_checkpoint(directory, 165, log)
        verify_metrics(exp, core.parse_rows(log), 165)
    return directory / "global_step_165" / "actor"


def write_summary(results: dict[str, dict]) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "results": results}
    (ROOT / "FINAL_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Opposite disturbed-set attenuation experiments",
        "",
        "| Experiment | Macro avg@16 | Micro avg@16 | Macro pass@16 | Micro pass@16 | Mean length |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {100 * metrics['macro_6sets']['avg@16']:.3f}% | "
            f"{100 * metrics['overall_micro']['avg@16']:.3f}% | "
            f"{100 * metrics['macro_6sets']['pass@16']:.3f}% | "
            f"{100 * metrics['overall_micro']['pass@16']:.3f}% | "
            f"{metrics['overall_micro']['mean_response_tokens']:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    (ROOT / "FINAL_RESULTS.md").write_text(report)
    (RECORD / "disturbed_opposite_attenuation_next3_results.md").write_text(report)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    core.ROOT = ROOT
    core.RECORD = RECORD
    core.MONITOR_LOG = MONITOR_LOG
    core.base_env = base_env
    lock = (ROOT / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another opposite-attenuation controller is already running") from exc
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    event("QUEUE_REGISTERED predecessor=continuous-next4 order=" + ",".join(e.name for e in EXPERIMENTS))
    wait_for_predecessor()
    results = {}
    for exp in EXPERIMENTS:
        run_smoke(exp)
        actor = train(exp)
        model = core.export(exp, actor)
        results[exp.name] = core.evaluate(exp, model)
        event(f"COMPLETE {exp.name}: smoke/training/export/avg16 verified")
    write_summary(results)
    (ROOT / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    event("QUEUE_COMPLETE all three opposite-attenuation experiments and avg16 verified")


if __name__ == "__main__":
    main()
