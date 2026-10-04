"""Preference-only evaluation of a trained DPO checkpoint.

This evaluator reports exactly two quantities on the held-out HH test set:

  preference_acc : fraction of test pairs for which the model assigns a higher
                   average token log-probability to the "chosen" continuation
                   than to the "rejected" one.
  mean_margin    : the mean of (avg-logprob of chosen minus avg-logprob of
                   rejected) over the same pairs.

Why the refusal metrics are absent
----------------------------------
The published evaluation entry point `src/evaluate_model.py` also reports
`unsafe_refusal_rate` and `benign_refusal_rate`. Both are computed from the
custom evaluation file `data/raw/safety_refusal_eval.jsonl` (250 unsafe and 250
benign prompts). That file was lost and is not recoverable, and it is a required
argument of `src/evaluate_model.py`, so that script cannot be run at all for new
checkpoints. This module exists so that new training runs can still be scored on
the metrics that remain computable from surviving data. No refusal quantity is
reported here, and none should be inferred from the outputs of this script.

The docstring of `src/evaluate_model.py` additionally advertises a metric named
`unsafe_preference_acc`, restricted to pairs whose chosen response is known-safe
and whose rejected response is known-unsafe. That metric is not implemented:
`eval_preference` in that file returns only `n`, `preference_acc` and
`mean_margin`. It is therefore not reported here either.

Numerical identity with the published numbers
---------------------------------------------
The scoring functions are imported directly from `src/evaluate_model.py` rather
than reimplemented, so `preference_acc` and `mean_margin` are produced by the
same code path that produced the values in `analysis/full_baseline_grid.csv`.
The model is loaded through the same `load_model` helper, which stacks the PEFT
adapter on the base model when `adapter_config.json` is present in `--model_dir`
and otherwise loads `--model_dir` as a full Hugging Face model. `src/` is not
modified by this script.

Usage:
    python3 analysis/scripts/eval_preference_only.py \
        --model_dir results/checkpoints/seed_study/eta20_raw_seed42 \
        --out_json results/eval/seed_study/eta20_raw_seed42.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.evaluate_model import (  # noqa: E402
    eval_preference,
    load_jsonl,
    load_model,
)

DEFAULT_HH_TEST = "data/processed/hh_test.jsonl"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_LENGTH = 1024
# Matches EVAL_MAX_SAMPLES in scripts/run_pilot_hh.sh, which produced the
# published n = 500 preference figures.
DEFAULT_MAX_SAMPLES = 500

SEED_PATTERN = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)


def parse_seed(model_dir: str) -> Optional[int]:
    """Return the training seed encoded in a checkpoint path, if one is present.

    Directories written by `analysis/scripts/run_seed_study.sh` are named
    `eta20_{config}_seed{seed}`. Paths that carry no such token yield None
    rather than a guessed value.
    """
    matches = SEED_PATTERN.findall(str(model_dir))
    if not matches:
        return None
    return int(matches[-1])


def resolve(path: str) -> Path:
    """Interpret relative paths against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model_dir", required=True,
                    help="LoRA adapter directory, or a full model directory.")
    ap.add_argument("--hh_test", default=DEFAULT_HH_TEST,
                    help="Held-out HH test jsonl.")
    ap.add_argument("--out_json", required=True,
                    help="Destination for the metrics JSON.")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL,
                    help="Base model the adapter is stacked on.")
    ap.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                    help="Number of test pairs scored, taken from the head of "
                         "the file exactly as in src/evaluate_model.py.")
    args = ap.parse_args()

    model_dir = resolve(args.model_dir)
    hh_test = resolve(args.hh_test)
    out_json = resolve(args.out_json)

    if not model_dir.is_dir():
        raise SystemExit(f"[eval] model directory does not exist: {model_dir}")
    if not hh_test.is_file():
        raise SystemExit(f"[eval] HH test file does not exist: {hh_test}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")
    print(f"[eval] loading model from {model_dir}")
    model, tokenizer = load_model(str(model_dir), args.base_model, device)

    rows = load_jsonl(str(hh_test))
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"[eval] preference scoring {len(rows)} pairs from {hh_test}")

    pref = eval_preference(model, tokenizer, rows, args.max_length)

    out = {
        "model_dir": str(model_dir),
        "base_model": args.base_model,
        "hh_test_file": str(hh_test),
        "n": pref["n"],
        "preference_acc": pref["preference_acc"],
        "mean_margin": pref["mean_margin"],
        "seed": parse_seed(str(model_dir)),
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "refusal_metrics": None,
        "refusal_metrics_note": (
            "Not computed. data/raw/safety_refusal_eval.jsonl is lost and "
            "unrecoverable, so unsafe_refusal_rate and benign_refusal_rate "
            "cannot be produced for this checkpoint."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
