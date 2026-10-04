"""P2: score HH pairs with a cross-backbone verifier pool.

Design: same four role prompts as the paper's pool, but each role is served by a
DIFFERENT backbone (the paper's pool serves all four roles from one shared
Qwen2.5-1.5B backbone). This isolates backbone diversity while holding the role
prompts, threshold, and data fixed.

Runs ONE (model, role) at a time, batched over rows, both directions, on a fixed
subsample of the 10K training pairs. Output format matches
hh_train.precomputed.jsonl so all downstream analyses reuse the same code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VSE = ROOT  # flattened layout
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VSE))

from src.batched_backend import BatchedLLMBackend  # noqa: E402
from src.precompute_verifiers import build_messages_for_chosen, VERIFIER_CLASSES, ROLE_NAMES  # noqa: E402

POOL = {  # role -> backbone (one original member kept for continuity)
    "safety": "google/gemma-2-9b-it",
    "helpfulness": "mistralai/Mistral-7B-Instruct-v0.3",
    "factuality": "Qwen/Qwen3-4B-Instruct-2507",
    "policy": "Qwen/Qwen2.5-1.5B-Instruct",
}
QUANT4 = {"google/gemma-2-9b-it"}  # too large for fp16 + KV on a 24GB TITAN RTX


class QuantBackend(BatchedLLMBackend):
    """BatchedLLMBackend with optional 4-bit NF4 loading for large backbones."""

    def _load(self):
        if self.model_name not in QUANT4:
            return super()._load()
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True,
                                            padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto",
            quantization_config=bnb)
        model.eval()
        self.model = model
        self.tokenizer = tok


def adapt_messages(msgs, model_name: str):
    """Gemma's chat template rejects the system role; fold system into user."""
    if "gemma" in model_name.lower() and msgs and msgs[0]["role"] == "system":
        merged = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
        return [{"role": "user", "content": merged}] + msgs[2:]
    return msgs


def parse_score(gen: str, threshold: float):
    try:
        data = BatchedLLMBackend.extract_json(gen)
        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
        rationale = str(data.get("rationale", ""))[:240]
    except Exception as e:  # noqa: BLE001
        score, rationale = 0.5, f"parse_error: {e}"
    return {"score": score, "passed": bool(score >= threshold),
            "rationale": rationale, "threshold": threshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=list(POOL))
    ap.add_argument("--n_rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_rows", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--outdir", default=str(ROOT / "analysis/outputs/diverse_pool"))
    args = ap.parse_args()

    import random
    rows = [json.loads(l) for l in open(VSE / "data/processed/hh_train.jsonl") if l.strip()]
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(rows)), args.n_rows))
    rows = [rows[i] for i in idx]

    role = args.role
    model_name = POOL[role]
    role_pos = ROLE_NAMES.index(role)
    outp = Path(args.outdir) / f"{role}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if outp.exists():
        for line in open(outp):
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        print(f"[resume] {len(done_ids)} rows already scored for {role}")
    todo = [r for r in rows if r["id"] not in done_ids]

    print(f"[diverse] role={role} model={model_name} rows={len(todo)}/{len(rows)} "
          f"(4bit={model_name in QUANT4})")
    backend = QuantBackend(model_name=model_name, max_new_tokens=96)

    t0 = time.time()
    with open(outp, "a") as fout:
        for start in range(0, len(todo), args.batch_rows):
            chunk = todo[start:start + args.batch_rows]
            # one role message per row per direction; batch = rows x directions
            msgs, keys = [], []
            for r in chunk:
                msgs.append(adapt_messages(
                    build_messages_for_chosen(r["prompt"], r["response_a"])[role_pos], model_name))
                keys.append((r["id"], "a"))
                msgs.append(adapt_messages(
                    build_messages_for_chosen(r["prompt"], r["response_b"])[role_pos], model_name))
                keys.append((r["id"], "b"))
            outs = backend.generate_batch(msgs)
            per_row = {}
            for (rid, d), gen in zip(keys, outs):
                res = parse_score(gen, args.threshold)
                res["model_name"] = model_name
                per_row.setdefault(rid, {})[d] = res
            for r in chunk:
                fout.write(json.dumps({
                    "id": r["id"], "role": role,
                    "result_a": per_row[r["id"]]["a"],
                    "result_b": per_row[r["id"]]["b"],
                }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done = start + len(chunk)
            if (start // args.batch_rows) % 10 == 0:
                rate = n_done / max(time.time() - t0, 1)
                eta_min = (len(todo) - n_done) / max(rate, 1e-9) / 60
                print(f"[{role}] {n_done}/{len(todo)} rows ({rate:.2f} rows/s, ~{eta_min:.0f} min left)",
                      flush=True)
    print(f"[done] {role} -> {outp} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
