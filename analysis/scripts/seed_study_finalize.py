"""Verify one seed-study run, write its provenance record and its marker.

This module is invoked by ``analysis/scripts/run_seed_study.sh`` once training
and evaluation of a single run have both returned successfully. It performs the
verification that the completion marker asserts, and it writes the marker only
when every condition holds:

  1. the checkpoint directory carries an adapter configuration and at least one
     adapter weight file;
  2. the evaluation JSON exists and parses;
  3. the evaluation JSON carries a numeric ``preference_acc`` lying in the
     closed interval from zero to one;
  4. the evaluation JSON carries the expected number of scored pairs.

A run that fails any condition leaves no marker behind, so the driver re-runs it
on the next invocation rather than treating it as finished. This is the reason
the marker exists at all: an evaluation JSON on disk proves only that a file was
opened for writing, not that a usable result was produced, so resume logic that
keys on the JSON alone silently accepts a run that died during evaluation.

The provenance record written alongside the marker is
``results/eval/seed_study/<tag>.meta.json``. It is intended to be sufficient, on
its own, to establish what produced a given number: the exact command lines, the
training file with its digest and line count, every hyperparameter actually
passed, the environment and library versions, the wall time, and a digest over
the adapter weight files so that the adapter can later be shown to be the same
one that produced the reported accuracy. That last point matters because the
safety refusal evaluation cannot be run today; when
``data/raw/safety_refusal_eval.jsonl`` is recovered, the refusal metrics must be
computed against these adapters and not against retrained substitutes.

All inputs arrive through environment variables, which avoids any quoting
question at the shell boundary. The module is not intended to be run by hand.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Files whose contents constitute the trained adapter. The digest reported as
# ``adapter_sha256`` is taken over the per-file digests of these, not over the
# directory, so that logs and status files written into the same directory do
# not perturb it.
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
ADAPTER_CONFIG_NAME = "adapter_config.json"
TRAINER_STATE_NAME = "trainer_state.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"[finalize] required environment variable {name} is unset")
    return value


def parse_hyperparameters(spec: str) -> Dict[str, object]:
    """Turn the semicolon-separated key=value string into typed values."""
    out: Dict[str, object] = {}
    for item in spec.split(";"):
        if not item:
            continue
        key, _, raw = item.partition("=")
        value: object = raw
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        out[key] = value
    return out


def fail(message: str) -> None:
    print(f"[finalize] VERIFICATION FAILED: {message}", file=sys.stderr)


def main() -> int:
    tag = env("RM_TAG")
    ckpt = Path(env("RM_CKPT"))
    eval_json_path = Path(env("RM_EVAL_JSON"))
    meta_path = Path(env("RM_META"))
    done_path = Path(env("RM_DONE"))
    expected_n = int(env("RM_EXPECT_N"))

    problems: List[str] = []

    # ---- condition 1: the adapter exists on disk
    adapter_config = ckpt / ADAPTER_CONFIG_NAME
    weight_files = [ckpt / name for name in ADAPTER_WEIGHT_NAMES if (ckpt / name).is_file()]
    if not ckpt.is_dir():
        problems.append(f"checkpoint directory is absent: {ckpt}")
    else:
        if not adapter_config.is_file():
            problems.append(f"adapter configuration is absent: {adapter_config}")
        if not weight_files:
            problems.append(
                "no adapter weight file is present; expected one of "
                + ", ".join(ADAPTER_WEIGHT_NAMES)
                + f" under {ckpt}"
            )

    # ---- condition 2: the evaluation JSON exists and parses
    payload: Optional[Dict] = None
    if not eval_json_path.is_file():
        problems.append(f"evaluation JSON is absent: {eval_json_path}")
    else:
        try:
            payload = json.loads(eval_json_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"evaluation JSON does not parse: {exc}")
        else:
            if not isinstance(payload, dict):
                problems.append("evaluation JSON is not an object")
                payload = None

    # ---- conditions 3 and 4: the metric is usable and covers the expected n
    acc: Optional[float] = None
    n_scored: Optional[int] = None
    if payload is not None:
        raw_acc = payload.get("preference_acc")
        if isinstance(raw_acc, bool) or not isinstance(raw_acc, (int, float)):
            problems.append(
                f"preference_acc is not a number: {raw_acc!r}"
            )
        else:
            acc = float(raw_acc)
            if acc != acc or acc in (float("inf"), float("-inf")):
                problems.append(f"preference_acc is not finite: {acc!r}")
                acc = None
            elif not (0.0 <= acc <= 1.0):
                problems.append(
                    f"preference_acc lies outside the interval from zero to one: {acc}"
                )
                acc = None

        raw_n = payload.get("n")
        if isinstance(raw_n, bool) or not isinstance(raw_n, int):
            problems.append(f"the scored-pair count n is not an integer: {raw_n!r}")
        else:
            n_scored = int(raw_n)
            if n_scored != expected_n:
                problems.append(
                    f"the scored-pair count n is {n_scored}, expected {expected_n}"
                )

    if problems:
        for message in problems:
            fail(f"{tag}: {message}")
        print(
            f"[finalize] {tag}: no completion marker was written, so the driver "
            "will re-run this configuration on its next invocation.",
            file=sys.stderr,
        )
        return 1

    # ---- provenance record
    adapter_digests = [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(weight_files)
    ]
    combined = hashlib.sha256()
    for entry in adapter_digests:
        combined.update(f"{entry['file']}:{entry['sha256']}\n".encode())
    adapter_sha256 = combined.hexdigest()

    trainer_state = ckpt / TRAINER_STATE_NAME
    train_status = ckpt / "train_status.json"
    train_status_payload = None
    if train_status.is_file():
        try:
            train_status_payload = json.loads(train_status.read_text())
        except json.JSONDecodeError:
            train_status_payload = None

    try:
        environment = json.loads(env("RM_ENV", "{}"))
    except json.JSONDecodeError:
        environment = {}

    meta = {
        "tag": tag,
        "config": env("RM_CONFIG"),
        "seed": int(env("RM_SEED")),
        "gpu_id": env("RM_GPU"),
        "eta_nominal_pct": int(env("RM_ETA", "20")),
        "corruption": env("RM_CORR", "structured_unsafe"),
        "command_line": {
            "training": env("RM_TRAIN_CMD"),
            "evaluation": env("RM_EVAL_CMD"),
        },
        "training_file": {
            "path": env("RM_TRAIN_FILE"),
            "sha256": env("RM_TRAIN_SHA", "") or None,
            "lines": int(env("RM_TRAIN_LINES", "0")) or None,
        },
        "held_out_test": {
            "path": env("RM_HH_TEST", ""),
            "sha256": env("RM_HH_SHA", "") or None,
        },
        "base_model": env("RM_BASE"),
        "hyperparameters": parse_hyperparameters(env("RM_HYPER", "")),
        "timing": {
            "start": env("RM_START"),
            "end": env("RM_END"),
            "wall_seconds": int(env("RM_WALL")),
        },
        "environment": {
            "hostname": env("RM_HOST", environment.get("hostname", "")),
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "transformers": environment.get("transformers"),
            "trl": environment.get("trl"),
            "peft": environment.get("peft"),
            "datasets": environment.get("datasets"),
            "cuda_device_name": env("RM_CUDA_NAME", "unknown"),
        },
        "checkpoint": {
            "path": str(ckpt),
            "adapter_config": str(adapter_config) if adapter_config.is_file() else None,
            "adapter_weight_files": adapter_digests,
            "adapter_sha256": adapter_sha256,
            "adapter_sha256_definition": (
                "sha256 over the concatenation of '<file>:<sha256>\\n' for each "
                "adapter weight file, taken in filename order"
            ),
            "trainer_state_path": str(trainer_state) if trainer_state.is_file() else None,
            "train_status_path": str(train_status) if train_status.is_file() else None,
            "train_status": train_status_payload,
            "train_log": env("RM_TRAIN_LOG", ""),
            "eval_log": env("RM_EVAL_LOG", ""),
            "retention": (
                "This adapter is retained indefinitely. The safety refusal "
                "evaluation of src/evaluate_model.py cannot be run until "
                "data/raw/safety_refusal_eval.jsonl is recovered, and when it "
                "is, the refusal metrics must be computed against this adapter "
                "rather than a retrained substitute."
            ),
        },
        "evaluation": {
            "path": str(eval_json_path),
            "preference_acc": acc,
            "mean_margin": payload.get("mean_margin") if payload else None,
            "n": n_scored,
            "expected_n": expected_n,
        },
        "written_at": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    marker = {
        "tag": tag,
        "completed_at": meta["written_at"],
        "preference_acc": acc,
        "n": n_scored,
        "adapter_sha256": adapter_sha256,
        "eval_json": str(eval_json_path),
        "meta_json": str(meta_path),
        "verified": [
            "adapter configuration and weight files present",
            "evaluation JSON present and parses",
            f"preference_acc numeric and within [0, 1]: {acc}",
            f"scored-pair count equals the expected {expected_n}",
        ],
    }
    done_path.write_text(json.dumps(marker, indent=2) + "\n")

    print(f"[finalize] {tag}: verified, wrote {meta_path.name} and {done_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
