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
"""Verify every saved rollout and score in one offline avg@16 evaluation."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED_SOURCES = {
    "aime": 30,
    "aime25": 30,
    "amc": 83,
    "math": 500,
    "minerva": 272,
    "olympiad_bench": 673,
}


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify(root: Path, model: str):
    manifest = read_json(root / "manifest.json")
    status = read_json(root / "status.json")
    results = read_json(root / "results.json")
    if status != {"phase": "complete", "models": [model], "updated_at": status.get("updated_at")}:
        raise RuntimeError(f"Evaluation status is not complete for {model}: {status}")
    settings = manifest["settings"]
    required = {
        "n": 16,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 8192,
        "max_prompt_length": 1024,
        "max_model_len": 9216,
        "enable_thinking": False,
        "do_sample": True,
        "dtype": "bfloat16",
        "scorer": "verl.utils.reward_score.default_compute_score",
    }
    mismatches = {key: (settings.get(key), value) for key, value in required.items() if settings.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Evaluation settings differ: {mismatches}")
    if manifest["questions"] != 1588 or manifest["data_source_counts"] != EXPECTED_SOURCES:
        raise RuntimeError("Validation population differs from the established 1,588-question set")
    if len(results) != 1 or results[0]["model"] != model:
        raise RuntimeError(f"Unexpected result models: {[result.get('model') for result in results]}")
    result = results[0]
    if result["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("Result/manifest fingerprint mismatch")
    if read_json(root / model / "summary.json") != result:
        raise RuntimeError("Per-model summary and root results differ")

    source_counts = Counter()
    response_count = 0
    for row in manifest["rows"]:
        index = row["source_index"]
        rollout = read_json(root / model / "rollouts" / f"{index:05d}.json.gz")
        score = read_json(root / model / "scores" / f"{index:05d}.json.gz")
        for record in (rollout, score):
            if (
                record["fingerprint"] != manifest["fingerprint"]
                or record["model"] != model
                or record["source_index"] != index
                or record["data_source"] != row["data_source"]
            ):
                raise RuntimeError(f"Record identity mismatch at source row {index}")
        outputs = rollout["outputs"]
        scores = score["scores"]
        correct = score["correct"]
        if len(outputs) != 16 or len(scores) != 16 or len(correct) != 16:
            raise RuntimeError(f"Incomplete n=16 record at source row {index}")
        if [output["index"] for output in outputs] != list(range(16)):
            raise RuntimeError(f"Output indices differ at source row {index}")
        if any(not math.isfinite(value) for value in scores):
            raise RuntimeError(f"Non-finite score at source row {index}")
        if correct != [int(value > 0.5) for value in scores]:
            raise RuntimeError(f"Correctness threshold differs at source row {index}")
        source_counts[row["data_source"]] += 1
        response_count += len(outputs)

    if source_counts != Counter(EXPECTED_SOURCES) or response_count != 1588 * 16:
        raise RuntimeError(f"Incomplete population: {source_counts}, responses={response_count}")
    for source, questions in EXPECTED_SOURCES.items():
        metric = result["metrics"][source]
        if metric["questions"] != questions or metric["responses"] != questions * 16:
            raise RuntimeError(f"Aggregate count mismatch for {source}: {metric}")
    if not (root / "REPORT.md").is_file() or not (root / "results.csv").is_file():
        raise RuntimeError("Human-readable report artifacts are missing")
    print(
        json.dumps(
            {
                "model": model,
                "questions": 1588,
                "responses": response_count,
                "macro_avg16": result["metrics"]["macro_6sets"]["avg@16"],
                "micro_avg16": result["metrics"]["overall_micro"]["avg@16"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model")
    args = parser.parse_args()
    verify(args.root.resolve(), args.model)
