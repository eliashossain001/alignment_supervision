"""Recover verifier scores lost to output truncation.

precompute_verifiers.py and precompute_diverse.py generate at most 96 new
tokens. A verifier that writes a long rationale has its JSON cut mid-string, the
parse fails, and parse_outputs assigns the sentinel 0.5. Because 0.5 is below the
0.7 threshold, every such truncation silently becomes a REJECT vote.

The score is emitted before the rationale, so the truncated text still contains
the model's actual score. This script recovers it, leaving genuine parse failures
(no extractable score) at the sentinel. The original file is preserved as
<name>.presalvage.jsonl and the repair is logged per role.
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?|\.\d+)')
THRESHOLD = 0.7


def repair_record(res: dict) -> tuple[dict, bool]:
    """Return (record, repaired?)."""
    if res.get("score") != 0.5:
        return res, False
    rat = str(res.get("rationale", ""))
    if not rat.startswith("parse_error"):
        return res, False
    m = SCORE_RE.search(rat)
    if not m:
        return res, False
    val = float(m.group(1))
    if not (0.0 <= val <= 1.0):
        return res, False
    res = dict(res)
    res["score"] = val
    res["passed"] = val >= THRESHOLD
    res["salvaged_from_truncation"] = True
    return res, True


def main() -> None:
    targets = [ROOT / f"analysis/outputs/diverse_pool/{r}.jsonl"
               for r in ("safety", "helpfulness", "factuality", "policy")]
    for path in targets:
        if not path.exists():
            print(f"skip (missing): {path.name}"); continue
        backup = path.with_suffix(".presalvage.jsonl")
        if not backup.exists():
            shutil.copy2(path, backup)
        rows, n_rep, n_sent = [], 0, 0
        for line in open(backup):
            d = json.loads(line)
            for k in ("result_a", "result_b"):
                if k in d:
                    d[k], rep = repair_record(d[k])
                    n_rep += rep
                    if d[k].get("score") == 0.5 and str(d[k].get("rationale","")).startswith("parse_error"):
                        n_sent += 1
            rows.append(d)
        with open(path, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        total = len(rows) * 2
        print(f"{path.stem:<13} scores={total:<6} salvaged={n_rep:<5} "
              f"({n_rep/total*100:5.2f}%)  still-sentinel={n_sent}")


if __name__ == "__main__":
    main()
