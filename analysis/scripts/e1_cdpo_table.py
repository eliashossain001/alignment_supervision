"""E1: recover the omitted cDPO (noise-aware DPO) results and validate that
our reimplementation of the filter metrics reproduces main_results.csv exactly."""
import csv
import json

import numpy as np
from common import (OUT, TAB, VSE, METHOD_LABEL, clean_mask, filter_metrics,
                    keep_consensus, keep_single, load_corrupted, load_precomputed,
                    score_matrix)

# ---- 1. validation: reproduce Table 2 filter-side numbers from raw caches ----
pre = load_precomputed()
print("== validation of metric reimplementation vs main_results.csv ==")
saved = {}
with open(VSE / "results/tables/main_results.csv") as f:
    for row in csv.DictReader(f):
        saved[(row["eta"], row["method"])] = row

for eta in (10, 20):
    rows = load_corrupted("structured_unsafe", eta)
    scores = score_matrix(rows, pre)
    ic = clean_mask(rows)
    for method, keep in [
        ("raw", np.ones(len(rows), dtype=bool)),
        ("single", keep_single(scores)),
        ("consensus_k3", keep_consensus(scores, k=3)),
        ("oracle", ic.copy()),
    ]:
        m = filter_metrics(keep, ic)
        s = saved[(str(eta), method)]
        ok = (abs(m["retention"] - float(s["retention"])) < 1e-9
              and abs(m["harmful_survival"] - float(s["harmful_survival_rate"])) < 1e-9
              and abs(m["det_f1"] - float(s["corruption_detection_f1"])) < 1e-9)
        print(f"eta={eta} {method:13s} retention={m['retention']:.4f} "
              f"harm={m['harmful_survival']:.4f} F1={m['det_f1']:.4f}  match={ok}")
        assert ok, f"MISMATCH for eta={eta} {method}: ours={m} saved={s}"
print("all filter-side numbers reproduced exactly.\n")

# ---- 2. the recovered table incl. cDPO --------------------------------------
methods = ["raw", "noise_aware", "single", "consensus_k3", "oracle"]
cols = ["preference_acc", "mean_margin", "unsafe_refusal_rate", "benign_refusal_rate",
        "retention", "harmful_survival_rate", "corruption_detection_f1"]

lines_md = ["| eta | Method | Pref. acc. | Margin | Unsafe refusal | Benign refusal | Retention | Harm. surv. | Det. F1 |",
            "|---|---|---|---|---|---|---|---|---|"]
lines_tex = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"$\eta$ & Method & Pref.\ acc. & Margin & Unsafe ref. & Benign ref. & Retention & Harm.\ surv. & Det.\ F1 \\",
             r"\midrule"]
records = []
for eta in ("0", "10", "20"):
    for method in methods:
        s = saved[(eta, method)]
        def g(c, pct=False):
            v = s[c].strip()
            if v == "":
                return "--"
            x = float(v)
            return f"{100*x:.1f}\\%" if pct else f"{x:.3f}"
        def gm(c, pct=False):
            v = s[c].strip()
            if v == "":
                return "n/a"
            x = float(v)
            return f"{100*x:.1f}%" if pct else f"{x:.3f}"
        records.append({k: (s[k] if s[k].strip() else None) for k in ["corruption", "eta", "method"] + cols})
        lines_md.append(
            f"| {eta}% | {METHOD_LABEL[method]} | {gm('preference_acc')} | {gm('mean_margin')} | "
            f"{gm('unsafe_refusal_rate')} | {gm('benign_refusal_rate')} | {gm('retention', True)} | "
            f"{gm('harmful_survival_rate', True)} | {gm('corruption_detection_f1')} |")
        lines_tex.append(
            f"{eta}\\% & {METHOD_LABEL[method]} & {g('preference_acc')} & {g('mean_margin')} & "
            f"{g('unsafe_refusal_rate')} & {g('benign_refusal_rate')} & {g('retention', True)} & "
            f"{g('harmful_survival_rate', True)} & {g('corruption_detection_f1')} \\\\")
    lines_tex.append(r"\midrule" if eta != "20" else r"\bottomrule")
lines_tex.append(r"\end{tabular}")

(TAB / "e1_full_table_with_cdpo.md").write_text("\n".join(lines_md) + "\n")
(TAB / "e1_full_table_with_cdpo.tex").write_text("\n".join(lines_tex) + "\n")
json.dump(records, open(OUT / "e1_records.json", "w"), indent=1)
print("\n".join(lines_md))
print("\n[saved] tables/e1_full_table_with_cdpo.{md,tex}")

# ---- 3. cDPO vs others: downstream deltas -----------------------------------
print("\n== cDPO vs other methods, downstream (per eta) ==")
for eta in ("0", "10", "20"):
    na = saved[(eta, "noise_aware")]
    raw = saved[(eta, "raw")]
    orc = saved[(eta, "oracle")]
    print(f"eta={eta}%: cDPO pref_acc={na['preference_acc']} vs raw={raw['preference_acc']} "
          f"oracle={orc['preference_acc']}; cDPO unsafe_refusal={na['unsafe_refusal_rate']} "
          f"vs raw={raw['unsafe_refusal_rate']} oracle={orc['unsafe_refusal_rate']}")
