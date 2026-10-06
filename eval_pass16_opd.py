#!/usr/bin/env python3
"""Resumable, eight-GPU pass@16 evaluation based on ./eval_passk.py.

Only local inference/CPU scoring; never uploads weights or changes training.
Use run_pass16_after_filtered.sh to wait for training before using GPUs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import signal
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("OPD_EVAL_ROOT", "/data/jiangxitai/OPDVR/eval_pass16_20260920"))
REPO = Path("/root/OPDVR")
PYTHON = "/root/miniconda3/envs/verl/bin/python"
DATASET = REPO / "datasets/valid_final_unique.parquet"
EXPORTS = Path("/data/jiangxitai/OPDVR/hf_exports")
MODELS = {
    "qwen3_1.7b_base": "/root/models/Qwen3-1.7B-Base",
    "qwen3_4b_non_thinking": "/root/models/Qwen3-4B",
    "opd_1.7b_4b_baseline_step165": str(EXPORTS / "opd_1.7b_4b_baseline_step165"),
    "opd_4b_4b_baseline_step222": str(EXPORTS / "opd_4b_4b_baseline_step222"),
}
if os.environ.get("OPD_EVAL_MODEL_PATH"):
    MODELS = {
        os.environ.get("OPD_EVAL_MODEL_NAME", "qwen3_1.7b_base"): os.environ["OPD_EVAL_MODEL_PATH"]
    }
TRAIN_RUN = "opd_1.7b_4b_filtered_baseline_0_20260919_221412"
TRAIN_LOG = REPO / f"{TRAIN_RUN}.retry1.log"
TRAIN_CKPT = Path("/data/jiangxitai/OPDVR/checkpoints") / TRAIN_RUN
TRAIN_PANE = "opd_filtered_20260919_221412:0.0"
SETTINGS = {
    "n": int(os.environ.get("OPD_EVAL_N", "16")),
    "temperature": float(os.environ.get("OPD_EVAL_TEMPERATURE", "0.6")),
    "top_p": float(os.environ.get("OPD_EVAL_TOP_P", "1.0")),
    "top_k": -1,
    "max_tokens": 8192,
    "max_prompt_length": 1024,
    "max_model_len": 9216,
    "enable_thinking": False,
    "do_sample": os.environ.get("OPD_EVAL_DO_SAMPLE", "true").lower() in ("1", "true", "yes"),
    "dtype": "bfloat16",
    "repetition_penalty": 1.0,
    "seed": 20260920,
    "passk_mode": "combinatorial",
    "correct_threshold": 0.5,
    "scorer": "verl.utils.reward_score.default_compute_score",
    "decode_skip_special_tokens": True,
}
RUNTIME = {
    "tensor_parallel_size": 1,
    "data_parallel_size": 8,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 32768,
    "batch_size": 32,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "flashinfer_sampler": True,
}
SOURCES = ["aime", "aime25", "amc", "math", "minerva", "olympiad_bench"]
SOURCE_NAMES = ["AIME24", "AIME25", "AMC", "MATH500", "Minerva", "OlympiadBench"]
KS = [1, 2, 4, 8, 16]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}", flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    with open(tmp, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def manifest():
    data = read_json(ROOT / "manifest.json")
    if data["settings"] != SETTINGS or data["models"] != MODELS:
        raise RuntimeError("Manifest differs from this script; refusing to mix evaluations.")
    for path, expected in data["scorer_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"Scorer changed after manifest creation: {path}")
    return data


def prepare():
    import importlib.metadata

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source_rows = pq.read_table(DATASET).to_pylist()
    reference_ids = None
    tokenizers = {}
    model_files = {}
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        ids = [
            tok.apply_chat_template(row["prompt"], add_generation_prompt=True, enable_thinking=False)
            for row in source_rows
        ]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Chat-template/tokenization differs for {name}; inspect before evaluation.")
        tokenizers[name] = {
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "chat_template_sha256": digest(tok.chat_template),
        }
        weights = sorted(Path(path).glob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"No exported safetensors weights: {path}")
        model_files[name] = {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights}
    tok = AutoTokenizer.from_pretrained(next(iter(MODELS.values())), local_files_only=True)
    rows, excluded = [], []
    for index, (row, ids) in enumerate(zip(source_rows, reference_ids, strict=True)):
        if len(ids) > SETTINGS["max_prompt_length"]:
            excluded.append({"source_index": index, "data_source": row["data_source"], "tokens": len(ids)})
            continue
        ground_truth = row["reward_model"]["ground_truth"]
        if ground_truth is None or ground_truth == "":
            raise RuntimeError(f"Missing ground truth at row {index}")
        prompt_text = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert tok.encode(prompt_text, add_special_tokens=False) == ids
        rows.append(
            {
                "source_index": index,
                "data_source": row["data_source"],
                "messages": row["prompt"],
                "prompt_text": prompt_text,
                "prompt_token_ids": ids,
                "ground_truth": ground_truth,
                "extra_info": row.get("extra_info") or {},
            }
        )
    score_root = REPO / "verl/verl/utils/reward_score"
    score_paths = [
        score_root / "__init__.py",
        *sorted((score_root / "prime_math_refine").glob("*.py")),
        REPO / "verl/verl/utils/py_functional.py",
    ]
    data = {
        "dataset": str(DATASET),
        "dataset_sha256": file_sha256(DATASET),
        "raw_questions": len(source_rows),
        "questions": len(rows),
        "excluded": excluded,
        "data_source_counts": dict(Counter(r["data_source"] for r in rows)),
        "settings": SETTINGS,
        "runtime_defaults": RUNTIME,
        "models": MODELS,
        "model_weight_files": model_files,
        "tokenizers": tokenizers,
        "scorer_sha256": {str(p): file_sha256(p) for p in score_paths},
        "reference_script_sha256": file_sha256(Path(__file__).with_name("eval_passk.py")),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "vllm", "transformers", "pyarrow"]},
        "rows": rows,
    }
    data["fingerprint"] = digest(data)
    target = ROOT / "manifest.json"
    if target.exists() and read_json(target)["fingerprint"] != data["fingerprint"]:
        raise RuntimeError("Existing manifest is different; not overwriting.")
    atomic_json(target, data)
    log(f"PREPARED {len(rows)} questions, {data['data_source_counts']}; excluded={excluded}")


def record_path(model, row_index, scored=False):
    return ROOT / model / ("scores" if scored else "rollouts") / f"{row_index:05d}.json.gz"


def validate_record(record, data, model, row):
    assert record["fingerprint"] == data["fingerprint"]
    assert record["model"] == model
    assert record["source_index"] == row["source_index"]
    outputs = record["outputs"]
    assert len(outputs) == SETTINGS["n"]
    assert [o["index"] for o in outputs] == list(range(SETTINGS["n"]))
    for output in outputs:
        assert output["finish_reason"] in ("stop", "length"), output["finish_reason"]
        # An immediate EOS is a valid (usually incorrect) sampled completion.
        # Missing candidates are rejected by the exact-n check above.
        assert 0 <= len(output["token_ids"]) <= SETTINGS["max_tokens"]


def generate_worker(model, rank, gpu, smaller=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from vllm import LLM, SamplingParams

    data = manifest()
    for path, stat in data["model_weight_files"][model].items():
        current = Path(path).stat()
        if current.st_size != stat["size"] or current.st_mtime_ns != stat["mtime_ns"]:
            raise RuntimeError(f"Model weights changed: {path}")
    assigned = data["rows"][rank :: RUNTIME["data_parallel_size"]]
    todo = []
    for row in assigned:
        path = record_path(model, row["source_index"])
        if path.exists():
            validate_record(read_json(path), data, model, row)
        else:
            todo.append(row)
    log(f"{model} rank={rank} pending={len(todo)}/{len(assigned)}")
    if not todo:
        return
    llm = LLM(
        model=MODELS[model],
        tensor_parallel_size=1,
        dtype=SETTINGS["dtype"],
        seed=SETTINGS["seed"] + rank,
        trust_remote_code=False,
        gpu_memory_utilization=RUNTIME["gpu_memory_utilization"],
        swap_space=0,
        max_model_len=SETTINGS["max_model_len"],
        max_num_seqs=64 if smaller else RUNTIME["max_num_seqs"],
        max_num_batched_tokens=16384 if smaller else RUNTIME["max_num_batched_tokens"],
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        override_generation_config={"max_new_tokens": SETTINGS["max_tokens"], "repetition_penalty": 1.0},
    )
    tokenizer = llm.get_tokenizer()
    batch_size = RUNTIME["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(todo), batch_size):
        batch = todo[offset : offset + batch_size]
        params = [
            SamplingParams(
                n=SETTINGS["n"],
                temperature=SETTINGS["temperature"],
                top_p=SETTINGS["top_p"],
                top_k=SETTINGS["top_k"],
                max_tokens=SETTINGS["max_tokens"],
                repetition_penalty=1.0,
            )
            for row in batch
        ]
        tick = time.monotonic()
        responses = llm.generate(
            [{"prompt_token_ids": r["prompt_token_ids"]} for r in batch], sampling_params=params, use_tqdm=False
        )
        if len(responses) != len(batch):
            raise RuntimeError(f"Missing requests: {len(responses)} != {len(batch)}")
        tokens = 0
        for row, response in zip(batch, responses, strict=True):
            assert list(response.prompt_token_ids) == row["prompt_token_ids"]
            outputs = []
            for output in sorted(response.outputs, key=lambda o: o.index):
                ids = list(output.token_ids)
                tokens += len(ids)
                outputs.append(
                    {
                        "index": output.index,
                        "token_ids": ids,
                        "text": tokenizer.decode(ids, skip_special_tokens=True),
                        "text_raw": tokenizer.decode(ids, skip_special_tokens=False),
                        "finish_reason": output.finish_reason,
                        "stop_reason": output.stop_reason,
                    }
                )
            record = {
                "fingerprint": data["fingerprint"],
                "model": model,
                "source_index": row["source_index"],
                "data_source": row["data_source"],
                "engine_seed": SETTINGS["seed"] + rank,
                "request_seed": None,
                "outputs": outputs,
            }
            validate_record(record, data, model, row)
            atomic_json(record_path(model, row["source_index"]), record)
        seconds = time.monotonic() - tick
        log(
            f"BATCH model={model} rank={rank} completed={min(offset + batch_size, len(todo))}/{len(todo)} "
            f"seconds={seconds:.1f} output_tokens={tokens} tokens_per_second={tokens / seconds:.1f} "
            f"total_seconds={time.monotonic() - started:.1f}"
        )
    log(f"GENERATION_DONE model={model} rank={rank}")


def score_one(task):
    model, row, fingerprint = task
    # CPU-only fork context avoids the decorated-function pickling bug under spawn.
    # No changes to the actual answer checker or mathematical criteria.
    from verl.utils.reward_score import default_compute_score

    record = read_json(record_path(model, row["source_index"]))
    validate_record(record, {"fingerprint": fingerprint}, model, row)
    scores = []
    for output in record["outputs"]:
        result = default_compute_score(
            data_source=row["data_source"],
            solution_str=output["text"],
            ground_truth=row["ground_truth"],
            extra_info=dict(row["extra_info"]),
        )
        score = float(result["score"] if isinstance(result, dict) else result)
        if not math.isfinite(score):
            raise RuntimeError(f"Nonfinite reward at {model}/{row['source_index']}")
        scores.append(score)
    summary = {
        "fingerprint": fingerprint,
        "model": model,
        "source_index": row["source_index"],
        "data_source": row["data_source"],
        "scores": scores,
        "correct": [int(s > SETTINGS["correct_threshold"]) for s in scores],
        "response_tokens": [len(o["token_ids"]) for o in record["outputs"]],
        "finish_reasons": [o["finish_reason"] for o in record["outputs"]],
    }
    atomic_json(record_path(model, row["source_index"], scored=True), summary)
    return row["source_index"]


def score_model(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mp.set_start_method("fork", force=True)
    data = manifest()
    todo = []
    for row in data["rows"]:
        path = record_path(model, row["source_index"], scored=True)
        if path.exists():
            result = read_json(path)
            assert result["fingerprint"] == data["fingerprint"]
            assert result["model"] == model and result["source_index"] == row["source_index"]
            assert len(result["scores"]) == SETTINGS["n"]
        else:
            todo.append((model, row, data["fingerprint"]))
    log(f"SCORING {model} pending={len(todo)}")
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("fork")) as pool:
        for count, _ in enumerate(pool.map(score_one, todo, chunksize=1), 1):
            if count % 100 == 0:
                log(f"SCORED {model} {count}/{len(todo)}")
    aggregate(model)


def passk(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError((n, c, k))
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(model):
    data = manifest()
    groups = defaultdict(list)
    for row in data["rows"]:
        record = read_json(record_path(model, row["source_index"], scored=True))
        assert record["fingerprint"] == data["fingerprint"]
        assert record["model"] == model and record["source_index"] == row["source_index"]
        assert record["data_source"] == row["data_source"]
        assert len(record["scores"]) == len(record["correct"]) == SETTINGS["n"]
        assert record["correct"] == [int(s > SETTINGS["correct_threshold"]) for s in record["scores"]]
        groups[row["data_source"]].append(record)
        groups["overall_micro"].append(record)
    metrics = {}
    n = SETTINGS["n"]
    valid_ks = [k for k in KS if k <= n]
    avg_key = f"avg@{n}"
    for source, records in groups.items():
        cs = [sum(r["correct"]) for r in records]
        metrics[source] = {
            "questions": len(cs),
            "responses": len(cs) * n,
            f"solved_at_{n}": sum(c > 0 for c in cs),
            avg_key: sum(cs) / (len(cs) * n),
            **{f"pass@{k}": sum(passk(n, c, k) for c in cs) / len(cs) for k in valid_ks},
            "mean_response_tokens": sum(sum(r["response_tokens"]) for r in records) / (len(cs) * n),
            "length_finish_fraction": sum(r["finish_reasons"].count("length") for r in records) / (len(cs) * n),
        }
    metrics["macro_6sets"] = {
        key: sum(metrics[s][key] for s in SOURCES) / len(SOURCES)
        for key in [avg_key, *[f"pass@{k}" for k in valid_ks]]
    }
    result = {
        "model": model,
        "path": MODELS[model],
        "fingerprint": data["fingerprint"],
        "settings": SETTINGS,
        "metrics": metrics,
    }
    atomic_json(ROOT / model / "summary.json", result)
    log(f"COMPLETE {model}: {metrics}")
    return result


def report():
    results = [aggregate(model) for model in MODELS]
    atomic_json(ROOT / "results.json", results)
    n = SETTINGS["n"]
    valid_ks = [k for k in KS if k <= n]
    columns = ["model", "dataset", "questions", "responses", f"avg@{n}", *[f"pass@{k}" for k in valid_ks]]
    with open(ROOT / "results.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            for source, values in result["metrics"].items():
                writer.writerow({"model": result["model"], "dataset": source, **values})
    lines = [
        f"# Offline avg@{n} evaluation",
        "",
        f"All scores below are percentages. Each question has {n} independently sampled completion(s).",
        "",
        "| Model | " + " | ".join(SOURCE_NAMES) + " | Macro average |",
        "|---|" + "---:|" * 7,
    ]
    for result in results:
        values = [result["metrics"][s][f"avg@{n}"] * 100 for s in [*SOURCES, "macro_6sets"]]
        lines.append("| " + result["model"] + " | " + " | ".join(f"{x:.2f}" for x in values) + " |")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Same validation parquet, chat template, system/user messages, and answer checker as training.",
            "- 1588 questions: AIME24 30, AIME25 30, AMC 83, MATH500 500, Minerva 272, OlympiadBench 673.",
            "- Original parquet has 1590 rows; source rows 835 and 836 exceed 1024 prompt tokens and are excluded exactly as in validation.",
            f"- enable_thinking={str(SETTINGS['enable_thinking']).lower()}; n={n}; temperature={SETTINGS['temperature']}; top_p={SETTINGS['top_p']}; top_k={SETTINGS['top_k']}; max_response_len={SETTINGS['max_tokens']}; BF16.",
            f"- avg@{n} is sample accuracy. Available pass@k values, if any, are also in results.csv/json.",
            "- Primary grader: verl.utils.reward_score.default_compute_score -> prime_math_refine.compute_score.",
            "- model/rollouts/*.json.gz preserves every completion and token ID; model/scores/*.json.gz preserves all scores.",
            "- manifest.json records dataset/scorer hashes, tokenized prompts, model paths, versions, and all settings.",
            "- Runtime: eight independent single-GPU vLLM replicas, no quantization. Sampler backend is controlled by the launch environment.",
            "- RNG: engine seed = 20260920 + GPU rank; individual request seed is unset, matching the reference script. Resumed questions need not be bitwise identical to an uninterrupted run.",
            "",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"EVALUATION_COMPLETE report={ROOT / 'REPORT.md'}")


def training_status():
    with open(TRAIN_LOG, "rb") as stream:
        stream.seek(max(0, TRAIN_LOG.stat().st_size - 2_000_000))
        tail = stream.read().decode(errors="replace")
    steps = re.findall(r"training/global_step:(\d+)", tail)
    step = int(steps[-1]) if steps else 0
    pane = subprocess.run(
        ["tmux", "display-message", "-p", "-t", TRAIN_PANE, "#{pane_dead} #{pane_dead_status}"],
        capture_output=True,
        text=True,
        check=False,
    )
    alive = pane.returncode == 0 and pane.stdout.strip().startswith("0")
    checkpoint = TRAIN_CKPT / "latest_checkpointed_iteration.txt"
    saved_step = int(checkpoint.read_text().strip()) if checkpoint.exists() else 0
    errors = bool(re.search(r"CUDA out of memory|OutOfMemoryError|Error executing job", tail))
    return {"step": step, "saved_step": saved_step, "alive": alive, "recent_error": errors, "pane": pane.stdout.strip()}


def free_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    return not result.stdout.strip(), result.stdout.strip()


def wait_for_training():
    last_log = 0
    while True:
        state = training_status()
        atomic_json(
            ROOT / "status.json", {"phase": "waiting_for_training", "training": state, "updated_at": time.time()}
        )
        if time.monotonic() - last_log >= 300:
            log(f"TRAIN_MONITOR {state}")
            last_log = time.monotonic()
        if not state["alive"]:
            if state["step"] < 111 or state["saved_step"] != 111:
                raise RuntimeError(f"Training stopped before its final checkpoint: {state}")
            expected = TRAIN_CKPT / "global_step_111/actor"
            for rank in range(8):
                assert (expected / f"model_world_size_8_rank_{rank}.pt").is_file()
            available, processes = free_gpus()
            if available:
                log(f"TRAINING_COMPLETE {state}; all eight GPUs released.")
                return
            log(f"Waiting for GPU owners to exit; will not kill unrelated jobs: {processes}")
        time.sleep(30)


def launch(args, logfile, gpu=None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", buffering=1) as stream:
        return subprocess.Popen(
            [PYTHON, "-u", str(Path(__file__).resolve()), *args],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_own_process(process):
    # Only process groups created with start_new_session=True by this controller.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_suite(wait=True):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "controller.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = manifest()
        if wait:
            wait_for_training()
        else:
            available, owners = free_gpus()
            if not available:
                raise RuntimeError(f"GPUs are still in use: {owners}")
        children, graders = [], []
        try:
            for model, model_path in MODELS.items():
                if (ROOT / model / "summary.json").exists():
                    aggregate(model)
                    continue
                workers = {}
                log(f"START_MODEL {model} path={model_path}")
                for rank in range(8):
                    p = launch(
                        ["worker", "--model", model, "--rank", str(rank), "--gpu", str(rank)],
                        ROOT / "logs" / f"{model}.gpu{rank}.log",
                        gpu=rank,
                    )
                    children.append(p)
                    workers[rank] = [p, 0]
                last_log = 0
                while workers:
                    for rank, (process, retries) in list(workers.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        if code == 0:
                            del workers[rank]
                        else:
                            stop_own_process(process)
                            if retries >= 2:
                                raise RuntimeError(f"{model} GPU {rank} failed repeatedly; inspect its log.")
                            log(f"RETRY model={model} rank={rank} exit={code}; retaining completed questions")
                            p = launch(
                                ["worker", "--model", model, "--rank", str(rank), "--gpu", str(rank), "--smaller"],
                                ROOT / "logs" / f"{model}.gpu{rank}.log",
                                gpu=rank,
                            )
                            children.append(p)
                            workers[rank] = [p, retries + 1]
                    generated = len(list((ROOT / model / "rollouts").glob("*.json.gz")))
                    status = {
                        "phase": "generating",
                        "model": model,
                        "generated": generated,
                        "questions": data["questions"],
                        "active_gpu_ranks": list(workers),
                        "updated_at": time.time(),
                    }
                    atomic_json(ROOT / "status.json", status)
                    if time.monotonic() - last_log >= 300:
                        log(f"EVAL_MONITOR {status}")
                        last_log = time.monotonic()
                    for name, grader in graders:
                        if grader.poll() not in (None, 0):
                            raise RuntimeError(f"CPU scoring failed for {name}")
                    if workers:
                        time.sleep(30)
                p = launch(["score", "--model", model], ROOT / "logs" / f"{model}.score.log")
                graders.append((model, p))
                children.append(p)
                log(f"ALL_ROLLOUTS_GENERATED {model}; CPU scoring started.")
                # A finished vLLM parent should release its GPU before the next model.
                for _ in range(24):
                    if free_gpus()[0]:
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError(f"GPU processes remained after {model}: {free_gpus()[1]}")
            while any(p.poll() is None for _, p in graders):
                for name, process in graders:
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f"CPU scoring failed for {name}")
                atomic_json(ROOT / "status.json", {"phase": "scoring", "updated_at": time.time()})
                time.sleep(30)
            for name, process in graders:
                if process.returncode != 0:
                    raise RuntimeError(f"CPU scoring failed for {name}")
            report()
            atomic_json(ROOT / "status.json", {"phase": "complete", "models": list(MODELS), "updated_at": time.time()})
        finally:
            for process in children:
                if process.poll() is None:
                    stop_own_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "score", "report", "run", "status"])
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smaller", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        if args.model is None or not 0 <= args.rank < 8:
            parser.error("worker requires --model and --rank 0..7")
        generate_worker(args.model, args.rank, args.gpu, args.smaller)
    elif args.command == "score":
        if args.model is None:
            parser.error("score requires --model")
        score_model(args.model)
    elif args.command == "report":
        report()
    elif args.command == "run":
        run_suite(not args.skip_wait)
    elif args.command == "status":
        log(training_status())
        if (ROOT / "status.json").exists():
            log(read_json(ROOT / "status.json"))


if __name__ == "__main__":
    main()
