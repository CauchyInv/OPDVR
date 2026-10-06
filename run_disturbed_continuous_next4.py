#!/usr/bin/env python3
"""Strict sequential controller for four continuous disturbed-OPD experiments.

Order: shuffled delta severity, surrogate damage, rank severity, top-50% damage.
Every formal 165-step run is preceded by a two-step smoke and followed by an
FSDP-to-HF export and the established 1588-question x 16-rollout evaluation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/root/OPDVR")
PY = "/root/miniconda3/envs/verl/bin/python"
ROOT = Path("/data/jiangxitai/OPDVR/disturbed_continuous_next4_20261004")
RECORD = Path("/root/jiangxitai/codex_log")
EVAL_SCRIPT = "/root/jiangxitai/eval_pass16_opd.py"
MONITOR_LOG = RECORD / "disturbed_continuous_next4_monitor.log"


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    shuffle: bool = False


EXPERIMENTS = (
    Experiment("up_continuous_shuffled_severity", "delta", True),
    Experiment("up_surrogate_damage_budget", "surrogate"),
    Experiment("up_rank_continuous_budget", "rank"),
    Experiment("up_top50_damage_budget", "topq"),
)


def event(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}"
    print(line, flush=True)
    RECORD.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a") as stream:
        stream.write(line + "\n")


def gpu_pids() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def wait_gpus() -> None:
    last = 0.0
    while gpu_pids():
        if time.monotonic() - last >= 1200:
            event(f"WAIT GPUs occupied pids={gpu_pids().replace(chr(10), ',')}")
            last = time.monotonic()
        time.sleep(30)


def base_env(exp: Experiment) -> dict[str, str]:
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
            "DIRECTION": "up",
            "WEIGHT_MODE": "continuous_budget",
            "MASS_GROWTH_BUDGET": "0.2",
            "W_MAX": "3.0",
            "STRATIFIED_BINS": "16",
            "CONTINUOUS_FLOOR": "0.25",
            "CONTINUOUS_QUANTILE": "0.9",
            "CONTINUOUS_SCORE_MODE": exp.mode,
            "CONTINUOUS_SHUFFLE": "True" if exp.shuffle else "False",
            "CONTINUOUS_RANK_POWER": "1.0",
            "CONTINUOUS_TOP_FRACTION": "0.5",
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
            "RAY_TMPDIR": "/data/ray_cont4",
            "TMPDIR": str(ROOT / "tmp"),
        }
    )
    return env


def monitor(command: list[str], env: dict[str, str], log: Path, label: str) -> None:
    event(f"START {label}; log={log}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        last = 0.0
        while process.poll() is None:
            if time.monotonic() - last >= 1200:
                tail = log.read_text(errors="replace")[-300_000:]
                steps = re.findall(r"training/global_step:(\d+)", tail)
                errors = re.findall(
                    r"(?:Traceback|OutOfMemoryError|CUDA out of memory|FloatingPointError|RuntimeError).{0,300}",
                    tail,
                )
                event(
                    f"MONITOR {label} pid={process.pid} step={steps[-1] if steps else 'startup/eval'} "
                    f"bytes={log.stat().st_size} gpu_pids={gpu_pids().replace(chr(10), ',')} errors={errors[-2:]}"
                )
                last = time.monotonic()
            time.sleep(20)
    event(f"EXIT {label} code={process.returncode}")
    if process.returncode:
        raise RuntimeError(f"{label} failed with exit {process.returncode}; inspect {log}")


def parse_rows(log: Path) -> dict[int, dict[str, float]]:
    rows = {}
    if not log.is_file():
        return rows
    pattern = re.compile(r"([\w/@.-]+):(?:np\.float64\()?([-+0-9.eE]+)")
    for line in log.read_text(errors="replace").splitlines():
        if "training/global_step:" not in line:
            continue
        row = {key: float(value) for key, value in pattern.findall(line)}
        if "training/global_step" in row:
            rows[int(row["training/global_step"])] = row
    return rows


def snapshot(exp: Experiment) -> None:
    target = ROOT / exp.name / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    files = (
        "verl/verl/trainer/ppo/disturbed_opd.py",
        "verl/verl/workers/actor/dp_actor.py",
        "verl/verl/trainer/config/ppo_trainer.yaml",
        "verl/tests/trainer/ppo/test_disturbed_opd_on_cpu.py",
        "opd_1.7b_4b_disturbed.sh",
        "run_disturbed_continuous_next4.py",
    )
    hashes = {}
    for relative in files:
        payload = (REPO / relative).read_bytes()
        (target / relative.replace("/", "__")).write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    (target / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def verify_checkpoint(directory: Path, step: int, log: Path) -> None:
    marker = directory / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text().strip() != str(step):
        raise RuntimeError(f"Checkpoint marker is not {step}: {marker}")
    actor = directory / f"global_step_{step}" / "actor"
    for kind in ("model", "optim", "extra_state"):
        for rank in range(8):
            path = actor / f"{kind}_world_size_8_rank_{rank}.pt"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing checkpoint shard: {path}")
    rows = parse_rows(log)
    if step not in rows or "Training Progress: 100%" not in log.read_text(errors="replace"):
        raise RuntimeError(f"Training log does not prove completion: {log}")


def verify_metrics(exp: Experiment, rows: dict[int, dict[str, float]]) -> None:
    expected_id = {"delta": 0.0, "surrogate": 1.0, "rank": 2.0, "topq": 3.0}[exp.mode]
    if len(rows) != 165:
        raise RuntimeError(f"{exp.name} has {len(rows)} metric rows, expected 165")
    required = (
        "continuous/score_mode_id",
        "continuous/score_mean_selected",
        "continuous/target_mass_growth",
        "disturbed/weight_mass_growth",
    )
    for step, row in rows.items():
        if any(key not in row for key in required):
            raise RuntimeError(f"{exp.name} step {step} is missing continuous metrics")
        if row["continuous/score_mode_id"] != expected_id:
            raise RuntimeError(f"{exp.name} step {step} used the wrong score mode")
    if exp.shuffle:
        if not all(row.get("continuous/shuffle_enabled") == 1.0 for row in rows.values()):
            raise RuntimeError("Shuffled severity was not enabled at every step")
        if not any(row.get("continuous/shuffle_mean_abs_change", 0.0) > 0 for row in rows.values()):
            raise RuntimeError("Shuffled severity never changed a score assignment")
    if exp.mode == "topq":
        for key in ("continuous/top_threshold", "continuous/reference_target_mass_growth"):
            if not all(key in row for row in rows.values()):
                raise RuntimeError(f"Top-q training is missing {key}")


def run_smoke(exp: Experiment) -> None:
    done = ROOT / f"{exp.name}.SMOKE_DONE"
    log = ROOT / f"smoke_{exp.name}.log"
    if done.exists():
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
    wait_gpus()
    monitor(
        ["bash", str(REPO / "opd_1.7b_4b_disturbed.sh"), "trainer.total_training_steps=2"],
        env,
        log,
        f"smoke:{exp.name}",
    )
    rows = parse_rows(log)
    if set(rows) != {1, 2}:
        raise RuntimeError(f"Smoke {exp.name} did not finish exactly two steps")
    expected_id = {"delta": 0.0, "surrogate": 1.0, "rank": 2.0, "topq": 3.0}[exp.mode]
    if not all(row.get("continuous/score_mode_id") == expected_id for row in rows.values()):
        raise RuntimeError(f"Smoke {exp.name} did not exercise its score mode")
    if exp.shuffle and not all(row.get("continuous/shuffle_mean_abs_change", 0.0) > 0 for row in rows.values()):
        raise RuntimeError("Shuffle smoke did not alter severity assignments")
    if exp.mode == "topq" and not all("continuous/top_threshold" in row for row in rows.values()):
        raise RuntimeError("Top-q smoke did not report its threshold")
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


def train(exp: Experiment) -> Path:
    done = ROOT / f"{exp.name}.step165.TRAIN_DONE"
    directory = ROOT / exp.name
    log = ROOT / f"train_{exp.name}.log"
    if not done.exists():
        wait_gpus()
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
        monitor(["bash", str(REPO / "opd_1.7b_4b_disturbed.sh")], env, log, f"train:{exp.name}")
        verify_checkpoint(directory, 165, log)
        verify_metrics(exp, parse_rows(log))
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    else:
        verify_checkpoint(directory, 165, log)
        verify_metrics(exp, parse_rows(log))
    return directory / "global_step_165" / "actor"


def export(exp: Experiment, actor: Path) -> Path:
    target = ROOT / "hf_exports" / exp.name
    done = ROOT / f"{exp.name}.EXPORT_DONE"
    if not done.exists():
        wait_gpus()
        log = ROOT / f"export_{exp.name}.log"
        monitor(
            [
                PY,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor),
                "--target_dir",
                str(target),
            ],
            base_env(exp),
            log,
            f"export:{exp.name}",
        )
        if not (target / "config.json").is_file() or not list(target.glob("*.safetensors")):
            raise RuntimeError(f"Incomplete HF export: {target}")
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    return target


def evaluate(exp: Experiment, model_path: Path) -> dict:
    eval_root = ROOT / "eval" / "avg16" / exp.name
    model = f"continuous_next4_{exp.name}_avg16"
    done = ROOT / f"{exp.name}.avg16.EVAL_DONE"
    env = base_env(exp)
    env.update(
        {
            "OPD_EVAL_ROOT": str(eval_root),
            "OPD_EVAL_MODEL_NAME": model,
            "OPD_EVAL_MODEL_PATH": str(model_path),
            "OPD_EVAL_N": "16",
            "OPD_EVAL_TEMPERATURE": "0.6",
            "OPD_EVAL_TOP_P": "1.0",
            "OPD_EVAL_DO_SAMPLE": "true",
        }
    )
    if not done.exists():
        wait_gpus()
        log = ROOT / f"eval_avg16_{exp.name}.log"
        monitor([PY, EVAL_SCRIPT, "prepare"], env, log, f"prepare:avg16:{exp.name}")
        monitor([PY, EVAL_SCRIPT, "run", "--skip-wait"], env, log, f"eval:avg16:{exp.name}")
    status_path = eval_root / "status.json"
    results_path = eval_root / "results.json"
    if not status_path.is_file() or not results_path.is_file():
        raise RuntimeError(f"Missing avg16 output for {exp.name}")
    status = json.loads(status_path.read_text())
    results = json.loads(results_path.read_text())
    if status.get("phase") != "complete" or len(results) != 1:
        raise RuntimeError(f"Incomplete avg16 result for {exp.name}: {status}")
    result = results[0]
    if result["model"] != model or result["metrics"]["overall_micro"]["responses"] != 1588 * 16:
        raise RuntimeError(f"Wrong avg16 population for {exp.name}")
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    return result


def write_summary(results: dict[str, dict]) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "results": results}
    (ROOT / "FINAL_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Continuous disturbed-OPD next-four experiments",
        "",
        "| Experiment | Macro avg@16 | Micro avg@16 | Macro pass@16 | Mean length |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {100 * metrics['macro_6sets']['avg@16']:.3f}% | "
            f"{100 * metrics['overall_micro']['avg@16']:.3f}% | "
            f"{100 * metrics['macro_6sets']['pass@16']:.3f}% | "
            f"{metrics['overall_micro']['mean_response_tokens']:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    (ROOT / "FINAL_RESULTS.md").write_text(report)
    (RECORD / "disturbed_continuous_next4_results.md").write_text(report)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    lock = (ROOT / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another continuous-next4 controller is already running") from exc
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    event("QUEUE_START order=" + ",".join(exp.name for exp in EXPERIMENTS))
    results = {}
    for exp in EXPERIMENTS:
        run_smoke(exp)
        actor = train(exp)
        model = export(exp, actor)
        results[exp.name] = evaluate(exp, model)
        event(f"COMPLETE {exp.name}: smoke/training/export/avg16 verified")
    write_summary(results)
    (ROOT / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    event("QUEUE_COMPLETE all four continuous experiments and avg16 verified")


if __name__ == "__main__":
    main()
