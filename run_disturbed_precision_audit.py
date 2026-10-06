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
"""Paired BF16/FP32 disturbed-selector precision audit.

For each initial/mid/final checkpoint, run one identical seeded rollout batch
through production BF16+FlashAttention2, BF16+SDPA, and FP32+SDPA.  A fixed
raw-probe radius grid is recorded token-by-token; the report compares zero
rates, delta signs, selector IoU, damage-weighted recall, and local linearity.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import run_disturbed_continuous_next4 as core

REPO = Path("/root/OPDVR")
PY = "/root/miniconda3/envs/verl/bin/python"
PREDECESSOR = Path("/data/jiangxitai/OPDVR/disturbed_shuffle4_audit_20261006/ALL_DONE")
ROOT = Path("/data/jiangxitai/OPDVR/disturbed_precision_audit_20261006")
RECORD = Path("/root/jiangxitai/codex_log")
MONITOR_LOG = RECORD / "disturbed_precision_audit_monitor.log"

RADII = ("1e-06", "3e-06", "1e-05", "3e-05", "0.0001", "0.0003", "0.001")
ETA = {tag: float(tag) for tag in RADII}


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model: Path


@dataclass(frozen=True)
class Mode:
    name: str
    attention: str
    autocast: str
    param_dtype: str
    remove_padding: bool


CHECKPOINTS = (
    Checkpoint("initial", Path("/root/models/Qwen3-1.7B-Base")),
    Checkpoint("mid_step55", Path("/data/jiangxitai/OPDVR/direction_length_20261001/hf_exports/up_budget")),
    Checkpoint("final_step165", Path("/data/jiangxitai/OPDVR/disturbed_next3_20261001/hf_exports/up_budget_full")),
)
MODES = (
    Mode("bf16_fa2", "flash_attention_2", "bf16", "bf16", True),
    Mode("bf16_sdpa", "sdpa", "bf16", "bf16", False),
    Mode("fp32_sdpa", "sdpa", "fp32", "fp32", False),
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
            event("WAIT predecessor shuffle4-audit queue")
            last = time.monotonic()
        time.sleep(30)
    event("PREDECESSOR_COMPLETE shuffle4-audit ALL_DONE found")


def job_name(checkpoint: Checkpoint, mode: Mode) -> str:
    return f"{checkpoint.name}__{mode.name}"


def job_root(checkpoint: Checkpoint, mode: Mode) -> Path:
    return ROOT / "runs" / job_name(checkpoint, mode)


def base_env(checkpoint: Checkpoint, mode: Mode) -> dict[str, str]:
    root = job_root(checkpoint, mode)
    for path in (root / "token_audit", root / "train_rollouts"):
        path.mkdir(parents=True, exist_ok=True)
    Path("/data/tmp/pa").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    curand = Path("/root/miniconda3/envs/verl/lib/python3.12/site-packages/nvidia/curand/include")
    env["CPATH"] = str(curand) + os.pathsep + env.get("CPATH", "")
    env["LIBRARY_PATH"] = "/root/miniconda3/envs/verl/lib" + os.pathsep + env.get("LIBRARY_PATH", "")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(
        {
            "ACTOR_MODEL_PATH": str(checkpoint.model),
            "DISTURBED_ENABLED": "True",
            "ALPHA": "1.0",
            "ETA_PROBE": "1e-4",
            "KAPPA": "0.0",
            "A_MIN_QUANTILE": "0.2",
            "PROBE_MOMENTUM": "False",
            "PROBE_DIAGNOSTIC": "True",
            "SELECT": "probe",
            "DIRECTION": "both",
            "WEIGHT_MODE": "budget",
            "MASS_GROWTH_BUDGET": "0.2",
            "W_MAX": "20.0",
            "MODEL_DTYPE": "fp32",
            "ROLLOUT_DTYPE": "bfloat16",
            "DATA_SHUFFLE": "False",
            "AUDIT_DIR": str(root / "token_audit"),
            "AUDIT_POST_UPDATE": "True",
            "AUDIT_TOPK": "0",
            "AUDIT_DISTRIBUTION_KL_TOPK": "32",
            "AUDIT_EOS_TOKEN_ID": "151643",
            "ROLLOUT_DATA_DIR": str(root / "train_rollouts"),
            "VALIDATION_DATA_DIR": "",
            "WANDB_MODE": "offline",
            "VAL_BEFORE_TRAIN": "False",
            "TEST_FREQ": "-1",
            "SAVE_FREQ": "-1",
            "TOTAL_EPOCHS": "1",
            "MINI_BATCH_SIZE": "64",
            "MAX_RESP_LENGTH": "8192",
            "MAX_PROMPT_LENGTH": "1024",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "PPO_MAX_TOKEN_LEN_PER_GPU": "12288",
            "NGPUS_PER_NODE": "8",
            "ROLLOUT_SEED": "424242",
            "ACTOR_LR": "1e-6",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
            "PYTHONPATH": str(REPO / "verl"),
            "PYTHONUNBUFFERED": "1",
            "EXPERIMENT_NAME": f"precision_audit_{job_name(checkpoint, mode)}",
            "FINAL_CKPT_DIR": str(root / "unused_checkpoint"),
            "RAY_TMPDIR": "/data/ray_pa",
            "TMPDIR": "/data/tmp/pa",
        }
    )
    return env


def mode_overrides(mode: Mode) -> list[str]:
    args = [
        f"actor_rollout_ref.actor.forward_autocast_dtype={mode.autocast}",
        f"actor_rollout_ref.model.use_remove_padding={'True' if mode.remove_padding else 'False'}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={mode.attention}",
    ]
    if mode.param_dtype == "fp32":
        args.append(
            "+actor_rollout_ref.actor.fsdp_config.mixed_precision="
            "{param_dtype:fp32,reduce_dtype:fp32,buffer_dtype:fp32}"
        )
    return args


def run_job(checkpoint: Checkpoint, mode: Mode) -> None:
    root = job_root(checkpoint, mode)
    done = root / "DONE"
    log = root / "run.log"
    if not done.is_file():
        core.wait_gpus()
        command = [
            "bash",
            str(REPO / "opd_1.7b_4b_disturbed.sh"),
            "trainer.total_training_steps=1",
            *mode_overrides(mode),
        ]
        core.monitor(command, base_env(checkpoint, mode), log, f"precision:{job_name(checkpoint, mode)}")
        verify_job(checkpoint, mode)
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    else:
        verify_job(checkpoint, mode)


def verify_job(checkpoint: Checkpoint, mode: Mode) -> None:
    root = job_root(checkpoint, mode)
    rows = core.parse_rows(root / "run.log")
    if set(rows) != {1}:
        raise RuntimeError(f"{job_name(checkpoint, mode)} did not finish exactly one step")
    row = rows[1]
    for radius in RADII:
        for suffix in ("vfrac", "gain", "moved_frac"):
            key = f"diag/grad_eta{radius}/{suffix}"
            if key not in row:
                raise RuntimeError(f"{job_name(checkpoint, mode)} lacks {key}")
    shards = sorted((root / "token_audit" / "step_00001").glob("rank_*.parquet"))
    if len(shards) != 8:
        raise RuntimeError(f"{job_name(checkpoint, mode)} has {len(shards)} audit shards")
    schema = set(pq.read_schema(shards[0]).names)
    for radius in RADII:
        for suffix in ("delta", "up", "down"):
            column = f"diag_grad_eta{radius}_{suffix}"
            if column not in schema:
                raise RuntimeError(f"{job_name(checkpoint, mode)} lacks parquet column {column}")
    rollout = root / "train_rollouts" / "1.jsonl"
    if not rollout.is_file() or rollout.stat().st_size == 0:
        raise RuntimeError(f"{job_name(checkpoint, mode)} lacks fixed rollout dump")


def load_tokens(checkpoint: Checkpoint, mode: Mode) -> dict[str, np.ndarray]:
    shards = sorted((job_root(checkpoint, mode) / "token_audit" / "step_00001").glob("rank_*.parquet"))
    columns = ["audit_response_id", "token_index", "token_id", "advantage"]
    for radius in RADII:
        columns.extend(
            (
                f"diag_grad_eta{radius}_delta",
                f"diag_grad_eta{radius}_up",
                f"diag_grad_eta{radius}_down",
            )
        )
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in shards])
    data = {name: table[name].to_numpy(zero_copy_only=False) for name in columns}
    order = np.lexsort((data["token_index"], data["audit_response_id"]))
    return {name: value[order] for name, value in data.items()}


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / max(int(union), 1))


def compare(checkpoint: Checkpoint, candidate_mode: Mode, reference_mode: Mode) -> dict:
    candidate = load_tokens(checkpoint, candidate_mode)
    reference = load_tokens(checkpoint, reference_mode)
    for key in ("audit_response_id", "token_index", "token_id"):
        if not np.array_equal(candidate[key], reference[key]):
            raise RuntimeError(
                f"{checkpoint.name}: {candidate_mode.name} and {reference_mode.name} do not share fixed tokens ({key})"
            )
    output = {
        "candidate": candidate_mode.name,
        "reference": reference_mode.name,
        "tokens": int(candidate["token_id"].size),
        "advantage_sign_agreement": float(
            np.mean(np.sign(candidate["advantage"]) == np.sign(reference["advantage"]))
        ),
        "advantage_mean_abs_gap": float(np.mean(np.abs(candidate["advantage"] - reference["advantage"]))),
        "radii": {},
    }
    reference_linear_delta = reference["diag_grad_eta0.0001_delta"] / 1e-4
    candidate_linear_delta = candidate["diag_grad_eta0.0001_delta"] / 1e-4
    for radius in RADII:
        c_delta = candidate[f"diag_grad_eta{radius}_delta"].astype(np.float64)
        r_delta = reference[f"diag_grad_eta{radius}_delta"].astype(np.float64)
        c_up = candidate[f"diag_grad_eta{radius}_up"].astype(bool)
        r_up = reference[f"diag_grad_eta{radius}_up"].astype(bool)
        c_down = candidate[f"diag_grad_eta{radius}_down"].astype(bool)
        r_down = reference[f"diag_grad_eta{radius}_down"].astype(bool)
        nonzero_union = (c_delta != 0) | (r_delta != 0)
        sign_agreement = (
            float(np.mean(np.sign(c_delta[nonzero_union]) == np.sign(r_delta[nonzero_union])))
            if nonzero_union.any()
            else 1.0
        )
        damage = np.maximum(-reference["advantage"] * r_delta, 0.0)
        ref_selected = r_up | r_down
        intersection = ref_selected & (c_up | c_down)
        weighted_recall = float(damage[intersection].sum() / max(float(damage[ref_selected].sum()), 1e-30))
        eta = ETA[radius]
        output["radii"][radius] = {
            "candidate_zero_fraction": float(np.mean(c_delta == 0)),
            "reference_zero_fraction": float(np.mean(r_delta == 0)),
            "delta_sign_agreement_nonzero_union": sign_agreement,
            "delta_pearson": safe_corr(c_delta, r_delta),
            "up_iou": iou(c_up, r_up),
            "down_iou": iou(c_down, r_down),
            "damage_weighted_selector_recall": weighted_recall,
            "candidate_linearity_corr_vs_eta1e4": safe_corr(c_delta / eta, candidate_linear_delta),
            "reference_linearity_corr_vs_eta1e4": safe_corr(r_delta / eta, reference_linear_delta),
        }
    return output


def write_report() -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "checkpoints": {}}
    reference = next(mode for mode in MODES if mode.name == "fp32_sdpa")
    for checkpoint in CHECKPOINTS:
        payload["checkpoints"][checkpoint.name] = {
            mode.name: compare(checkpoint, mode, reference)
            for mode in MODES
            if mode != reference
        }
    (ROOT / "FINAL_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Disturbed-OPD precision audit",
        "",
        "Reference: FP32 parameters/forward with SDPA. Production is BF16 forward with FlashAttention2.",
        "Every comparison uses identical response token ids; a mismatch aborts the queue.",
        "",
        "| Checkpoint | Candidate | eta | zero(candidate) | zero(FP32) | sign agree | up IoU | down IoU | damage recall | linearity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint, comparisons in payload["checkpoints"].items():
        for candidate, result in comparisons.items():
            for radius, metrics in result["radii"].items():
                lines.append(
                    f"| {checkpoint} | {candidate} | {radius} | "
                    f"{100 * metrics['candidate_zero_fraction']:.2f}% | "
                    f"{100 * metrics['reference_zero_fraction']:.2f}% | "
                    f"{100 * metrics['delta_sign_agreement_nonzero_union']:.2f}% | "
                    f"{100 * metrics['up_iou']:.2f}% | {100 * metrics['down_iou']:.2f}% | "
                    f"{100 * metrics['damage_weighted_selector_recall']:.2f}% | "
                    f"{metrics['candidate_linearity_corr_vs_eta1e4']:.4f} |"
                )
    report = "\n".join(lines) + "\n"
    (ROOT / "FINAL_RESULTS.md").write_text(report)
    (RECORD / "disturbed_precision_audit_results.md").write_text(report)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    core.ROOT = ROOT
    core.RECORD = RECORD
    core.MONITOR_LOG = MONITOR_LOG
    lock = (ROOT / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another precision-audit controller is already running") from exc
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    event(
        "QUEUE_REGISTERED checkpoints="
        + ",".join(checkpoint.name for checkpoint in CHECKPOINTS)
        + " modes="
        + ",".join(mode.name for mode in MODES)
    )
    wait_for_predecessor()
    for checkpoint in CHECKPOINTS:
        if not (checkpoint.model / "config.json").is_file():
            raise RuntimeError(f"Missing checkpoint export: {checkpoint.model}")
        for mode in MODES:
            run_job(checkpoint, mode)
        event(f"COMPLETE precision checkpoint {checkpoint.name}: all three paired modes verified")
    write_report()
    (ROOT / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")
    event("QUEUE_COMPLETE precision audit and paired report verified")


if __name__ == "__main__":
    main()
