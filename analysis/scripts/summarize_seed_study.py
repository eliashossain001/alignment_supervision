"""Summarise the multi-seed downstream study.

Reads every evaluation JSON written by ``analysis/scripts/eval_preference_only.py``
under ``results/eval/seed_study/`` and writes two artifacts:

    analysis/seed_study.md
    analysis/seed_study.csv

The question the summary answers is stated in one line: are the observed
differences between configurations large relative to the variation produced by
the training seed alone? Everything reported here serves that question.

For a configuration with three completed seeds the summary reports each seed
value, the mean, the sample standard deviation, the minimum, the maximum, and a
95 percent interval from the Student t distribution with two degrees of freedom.
That interval is labelled as very uncertain wherever it appears. Three runs do
not confer power, and the interval is printed so that the weakness of the
evidence is visible rather than to license a claim.

For a configuration with two completed seeds the summary reports both values,
their mean and their range, labels the configuration a partial replication,
states that the variance estimate is not stable, and does not compute an
interval. For a configuration with one seed the value is reported and labelled
unreplicated.

Differences against Raw DPO and against Oracle are computed seed by seed, seed
42 against seed 42 and so on, and the mean paired difference is reported with
its range. A paired difference isolates the method contrast within a seed and is
the quantity that speaks to the headline question. A difference of configuration
means does not, because the two means are computed over the same seeds and their
difference discards the pairing.

A one-way analysis of variance is retained as a secondary line only. With three
runs per configuration its p value is not the basis of any conclusion drawn
here, and the interpretation in this document is written from the magnitudes.

Finally, the newly computed seed 42 preference accuracy of each configuration is
compared against the published single-seed value in
``analysis/full_baseline_grid.csv``. When the deviation exceeds the tolerance the
seed 42 run is excluded from the pooled statistics of that configuration, the
pooled statistics are recomputed over the remaining seeds, and a warning naming
the likely causes is printed. A discrepant reproduction is never pooled
silently.

Only preference accuracy and mean margin are summarised. The refusal metrics of
``src/evaluate_model.py`` are not recoverable because the evaluation file
``data/raw/safety_refusal_eval.jsonl`` was lost, so no new run can be scored on
them. Nothing in this module is hard coded: configurations, seeds, the
corruption level and every published comparison value are read from the files on
disk.

Usage:
    python3 analysis/scripts/summarize_seed_study.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EVAL_DIR = ROOT / "results" / "eval" / "seed_study"
DEFAULT_GRID_CSV = ROOT / "analysis" / "full_baseline_grid.csv"
DEFAULT_OUT_MD = ROOT / "analysis" / "seed_study.md"
DEFAULT_OUT_CSV = ROOT / "analysis" / "seed_study.csv"
DEFAULT_DPO_CONFIG = ROOT / "configs" / "dpo_config.yaml"

# Evaluation JSONs are named eta{tag}_{config}_seed{seed}.json. The
# configuration token may itself contain underscores, so it is matched lazily
# between the two anchored fields.
TAG_RE = re.compile(r"^eta(?P<eta>\d+)_(?P<config>.+)_seed(?P<seed>\d+)$")

# The published grid uses its own method keys. Where a configuration name in the
# seed study differs from the published key for the same method, the equivalence
# is resolved through this table. It encodes naming only, never a value.
CONFIG_TO_PUBLISHED_KEY = {
    "cdpo": "noise_aware",
    "noise_aware": "cdpo",
}

METRICS = (
    ("preference_acc", "preference accuracy"),
    ("mean_margin", "mean margin"),
)

# The metric on which the reproduction check is decided. A discrepancy here
# excludes the seed from the pooled statistics of its configuration for every
# metric, because the cause would be a different training or scoring path rather
# than a property of one number.
REPRO_METRIC = "preference_acc"

# The two anchors against which seed-level paired differences are formed.
PAIRED_REFERENCES = ("raw", "oracle")

# Files that share the evaluation directory but are not run results.
NON_RUN_JSON = {"timing.json"}


# --------------------------------------------------------------------- stats
def t_critical(confidence: float, df: int) -> float:
    """Two-sided Student t critical value.

    SciPy is used when available. The fallback covers two degrees of freedom,
    the case that arises with three seeds, through the closed form quantile of
    the t distribution with df = 2.
    """
    p = 1.0 - (1.0 - confidence) / 2.0
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(p, df))
    except Exception:
        if df == 2:
            return (2.0 * p - 1.0) * math.sqrt(2.0 / (4.0 * p * (1.0 - p)))
        raise RuntimeError(
            f"cannot compute the t critical value for df={df} without SciPy"
        )


def anova_oneway(groups: Sequence[Sequence[float]]) -> Optional[Dict[str, float]]:
    """One-way analysis of variance across configurations.

    Reported as a secondary line only. Returns None when the design cannot
    support the test, for example when fewer than two configurations have at
    least two runs each.
    """
    usable = [list(g) for g in groups if len(g) >= 2]
    if len(usable) < 2:
        return None
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    result = stats.f_oneway(*usable)
    df_between = len(usable) - 1
    df_within = sum(len(g) for g in usable) - len(usable)
    return {
        "f": float(result.statistic),
        "p": float(result.pvalue),
        "df_between": float(df_between),
        "df_within": float(df_within),
        "k_groups": float(len(usable)),
    }


def pooled_sd(groups: Sequence[Sequence[float]]) -> Optional[float]:
    """Pooled within-configuration standard deviation.

    Each configuration contributes its own sample variance weighted by its
    degrees of freedom. Configurations with a single run contribute nothing.
    """
    num = 0.0
    den = 0
    for g in groups:
        if len(g) < 2:
            continue
        num += (len(g) - 1) * statistics.variance(g)
        den += len(g) - 1
    if den == 0:
        return None
    return math.sqrt(num / den)


REPLICATION_FULL = "replicated"
REPLICATION_PARTIAL = "partial replication"
REPLICATION_SINGLE = "unreplicated"
REPLICATION_NONE = "no completed run"


def describe(values: Sequence[float], confidence: float) -> Dict[str, Optional[object]]:
    """Summary statistics with a replication label attached.

    A t interval is produced only when three or more values are present. With
    two values the interval is deliberately withheld: an interval built from two
    points carries one degree of freedom and would misrepresent the evidence,
    and the range is the honest summary of the spread.
    """
    n = len(values)
    out: Dict[str, Optional[object]] = {
        "n": n,
        "values": list(values),
        "mean": statistics.fmean(values) if n else None,
        "sd": statistics.stdev(values) if n >= 2 else None,
        "min": min(values) if n else None,
        "max": max(values) if n else None,
        "range": (max(values) - min(values)) if n >= 2 else None,
        "ci_low": None,
        "ci_high": None,
        "t_crit": None,
        "df": None,
    }
    if n >= 3:
        tcrit = t_critical(confidence, n - 1)
        half = tcrit * out["sd"] / math.sqrt(n)  # type: ignore[operator]
        out["t_crit"] = tcrit
        out["df"] = n - 1
        out["ci_low"] = out["mean"] - half  # type: ignore[operator]
        out["ci_high"] = out["mean"] + half  # type: ignore[operator]
        out["replication"] = REPLICATION_FULL
    elif n == 2:
        out["replication"] = REPLICATION_PARTIAL
    elif n == 1:
        out["replication"] = REPLICATION_SINGLE
    else:
        out["replication"] = REPLICATION_NONE
    return out


def paired_differences(
    per_seed: Dict[int, float], ref_per_seed: Dict[int, float]
) -> Dict[str, Optional[object]]:
    """Seed-by-seed differences of a configuration against a reference.

    Only seeds present and included on both sides contribute. The pairing is by
    seed identity, seed 42 against seed 42 and so on, never by position and
    never by comparing means.
    """
    seeds = sorted(set(per_seed) & set(ref_per_seed))
    diffs = [(s, per_seed[s] - ref_per_seed[s]) for s in seeds]
    values = [d for _, d in diffs]
    return {
        "seeds": seeds,
        "per_seed": dict(diffs),
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sign_consistent": bool(values) and (
            all(v > 0 for v in values) or all(v < 0 for v in values)
        ),
    }


# ---------------------------------------------------------------- data input
def load_runs(eval_dir: Path) -> List[Dict]:
    """Collect one record per evaluation JSON found under eval_dir.

    A run is admitted only when its completion marker is present. The marker is
    written by ``analysis/scripts/seed_study_finalize.py`` after the adapter and
    the evaluation JSON have both been verified, so an evaluation JSON left
    behind by a run that died during evaluation is not summarised as though it
    were a result. Marker files are tolerated as absent when the study predates
    them, in which case the omission is reported rather than silently ignored.
    """
    runs: List[Dict] = []
    unmarked: List[str] = []
    any_marker = any(eval_dir.glob("*.done"))
    for path in sorted(eval_dir.glob("*.json")):
        # Provenance records and the timing ledger share the directory with the
        # per-run metrics and are not run results.
        if path.name.endswith(".meta.json") or path.name in NON_RUN_JSON:
            continue
        m = TAG_RE.match(path.stem)
        if not m:
            print(f"[warn] skipping {path.name}: filename does not match "
                  f"eta{{tag}}_{{config}}_seed{{seed}}.json")
            continue
        marker = path.with_suffix(".done")
        if any_marker and not marker.is_file():
            unmarked.append(path.name)
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[warn] skipping unreadable JSON {path}: {exc}")
            continue
        seed_from_payload = payload.get("seed")
        seed = int(seed_from_payload) if seed_from_payload is not None else int(m.group("seed"))
        if seed != int(m.group("seed")):
            print(f"[warn] {path.name}: seed in payload ({seed}) disagrees with "
                  f"the filename ({m.group('seed')}); the payload value is used")
        record = {
            "file": str(path),
            "eta_tag": int(m.group("eta")),
            "config": m.group("config"),
            "seed": seed,
            "n": payload.get("n"),
            "has_marker": marker.is_file(),
            "meta_file": (str(path.with_suffix(".meta.json"))
                          if path.with_suffix(".meta.json").is_file() else None),
        }
        missing = [k for k, _ in METRICS if payload.get(k) is None]
        if missing:
            print(f"[warn] skipping {path.name}: missing metric(s) {', '.join(missing)}")
            continue
        for key, _ in METRICS:
            record[key] = float(payload[key])
        runs.append(record)
    if unmarked:
        print("[warn] the following evaluation JSONs carry no completion marker "
              "and were excluded, because a missing marker means the run was not "
              "verified: " + ", ".join(unmarked))
    return runs


def load_published(grid_csv: Path) -> List[Dict[str, str]]:
    if not grid_csv.is_file():
        return []
    with open(grid_csv, newline="") as f:
        return list(csv.DictReader(f))


def published_training_seed(dpo_config: Path, fallback: Optional[int]) -> Optional[int]:
    """Recover the training seed used for the published runs.

    The value is read from the shared DPO configuration rather than assumed, so
    that the reproduction check always compares like with like. If the file is
    absent or carries no seed, the smallest seed present in the study is used.
    """
    if dpo_config.is_file():
        for line in dpo_config.read_text().splitlines():
            m = re.match(r"^\s*seed\s*:\s*(\d+)\s*(?:#.*)?$", line)
            if m:
                return int(m.group(1))
    return fallback


def published_row(rows: Sequence[Dict[str, str]], config: str, eta_tag: int) -> Optional[Dict[str, str]]:
    """Locate the published grid row for a configuration at a corruption level."""
    candidates = [config]
    alias = CONFIG_TO_PUBLISHED_KEY.get(config)
    if alias:
        candidates.append(alias)
    for key in candidates:
        for row in rows:
            try:
                row_eta = int(float(row.get("eta_nominal_pct", "")))
            except (TypeError, ValueError):
                continue
            if row_eta == eta_tag and row.get("method_key") == key:
                return row
    return None


def published_value(row: Optional[Dict[str, str]], metric_key: str) -> Optional[float]:
    if row is None or row.get(metric_key) in (None, ""):
        return None
    try:
        return float(row[metric_key])
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ renderer
def fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def signed(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{places}f}"


def resolve_baseline_seed(seeds: Sequence[int], dpo_config: Path) -> Optional[int]:
    """The seed whose runs are compared against the published grid."""
    if not seeds:
        return None
    candidate = published_training_seed(dpo_config, None)
    if candidate is not None and candidate in seeds:
        return candidate
    return min(seeds)


def reproduction_check(
    runs: Sequence[Dict],
    published_rows: Sequence[Dict[str, str]],
    baseline_seed: Optional[int],
    tolerance: float,
) -> Dict[str, Dict[str, object]]:
    """Compare the regenerated baseline-seed run against the published value.

    One record per configuration. ``verdict`` takes one of four values: exact,
    within tolerance, discrepant, or not comparable. Only a discrepant verdict
    causes the baseline seed to be excluded from that configuration's pooled
    statistics.
    """
    out: Dict[str, Dict[str, object]] = {}
    configs = sorted({r["config"] for r in runs})
    for cfg in configs:
        cfg_runs = [r for r in runs if r["config"] == cfg]
        eta_tag = cfg_runs[0]["eta_tag"]
        row = published_row(published_rows, cfg, eta_tag)
        pub = published_value(row, REPRO_METRIC)
        new_run = next((r for r in cfg_runs if r["seed"] == baseline_seed), None)
        new = new_run[REPRO_METRIC] if new_run else None
        record: Dict[str, object] = {
            "config": cfg,
            "eta_tag": eta_tag,
            "published": pub,
            "regenerated": new,
            "seed": baseline_seed,
            "deviation": None,
            "verdict": "not comparable",
            "exclude_seed": None,
        }
        if pub is not None and new is not None:
            dev = new - pub
            record["deviation"] = dev
            if dev == 0.0:
                record["verdict"] = "exact"
            elif abs(dev) <= tolerance:
                record["verdict"] = "within tolerance"
            else:
                record["verdict"] = "discrepant"
                record["exclude_seed"] = baseline_seed
        out[cfg] = record
    return out


def included_values(
    runs: Sequence[Dict], config: str, metric_key: str, excluded: Dict[str, Optional[int]]
) -> Dict[int, float]:
    """Seed to value map for one configuration after exclusions are applied."""
    drop = excluded.get(config)
    return {
        r["seed"]: r[metric_key]
        for r in runs
        if r["config"] == config and r["seed"] != drop
    }


def build_markdown(
    runs: Sequence[Dict],
    eval_dir: Path,
    grid_csv: Path,
    dpo_config: Path,
    confidence: float,
    tolerance: float,
    alpha: float,
) -> Tuple[str, Dict[str, Dict[str, object]]]:
    configs = sorted({r["config"] for r in runs})
    all_seeds = sorted({r["seed"] for r in runs})
    eta_tags = sorted({r["eta_tag"] for r in runs})
    published_rows = load_published(grid_csv)
    baseline_seed = resolve_baseline_seed(all_seeds, dpo_config)
    conf_pct = round(confidence * 100)

    repro = reproduction_check(runs, published_rows, baseline_seed, tolerance)
    excluded: Dict[str, Optional[int]] = {
        cfg: rec["exclude_seed"] for cfg, rec in repro.items()  # type: ignore[misc]
    }
    discrepant = [cfg for cfg, rec in repro.items() if rec["verdict"] == "discrepant"]

    lines: List[str] = []
    lines.append("# Training-seed variance study")
    lines.append("")
    lines.append(
        "This document asks one question: are the observed differences between "
        "configurations large relative to the variation produced by the training "
        "seed alone? It is not a comparison of method quality and it does not "
        "assert that any configuration is preferable to another."
    )
    lines.append("")

    # ---- scope
    lines.append("## Scope and limitations")
    lines.append("")
    lines.append(
        "Only two quantities are reported: preference accuracy and mean margin on "
        "the held-out HH test set. The refusal metrics of `src/evaluate_model.py`, "
        "namely the unsafe refusal rate and the benign refusal rate, are derived "
        "from `data/raw/safety_refusal_eval.jsonl`. That file was lost and is not "
        "recoverable, so no newly trained checkpoint can be scored on it. The "
        "metric named `unsafe_preference_acc` in the docstring of that module is "
        "not implemented in the module itself and is therefore also absent."
    )
    lines.append("")
    lines.append(
        f"Evaluation records were read from `{eval_dir}`. "
        f"{len(runs)} verified run(s) were found, covering {len(configs)} "
        f"configuration(s) and {len(all_seeds)} seed(s) "
        f"({', '.join(str(s) for s in all_seeds) if all_seeds else 'none'}) at "
        f"nominal corruption level(s) "
        f"{', '.join(f'{t} percent' for t in eta_tags) if eta_tags else 'none'}."
    )
    lines.append("")
    lines.append(
        "Only the training seed varies across these runs. The dataset subsample, "
        "the corruption realization, the verifier scores and every hyperparameter "
        "are fixed. The study therefore estimates training-seed uncertainty alone "
        "and does not estimate uncertainty over corruption draws, over dataset "
        "subsamples, or over verifier scoring."
    )
    lines.append("")
    if discrepant:
        lines.append(
            "**Exclusions applied.** The seed "
            f"{baseline_seed} run of the following configuration(s) failed the "
            "reproduction check against the published single-seed grid and has "
            "been excluded from every pooled statistic below: "
            + ", ".join(f"`{c}`" for c in sorted(discrepant))
            + ". The reproduction check section states the magnitude of each "
            "deviation and the candidate causes. No discrepant reproduction is "
            "pooled."
        )
        lines.append("")

    # ---- per-metric per-configuration tables
    for metric_key, metric_label in METRICS:
        places = 4 if metric_key == "preference_acc" else 6
        lines.append(f"## Seed-level {metric_label}")
        lines.append("")
        header = (["Configuration"] + [f"seed {s}" for s in all_seeds]
                  + ["n", "mean", "sd", "min", "max",
                     f"{conf_pct}% t interval", "replication"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        notes: List[str] = []
        for cfg in configs:
            per_seed_all = {r["seed"]: r[metric_key] for r in runs if r["config"] == cfg}
            per_seed = included_values(runs, cfg, metric_key, excluded)
            values = [per_seed[s] for s in sorted(per_seed)]
            d = describe(values, confidence)
            cells = [cfg]
            for s in all_seeds:
                if s in per_seed:
                    cells.append(fmt(per_seed[s], places))
                elif s in per_seed_all:
                    cells.append(f"{fmt(per_seed_all[s], places)} (excluded)")
                else:
                    cells.append("n/a")
            if d["n"] >= 3:
                interval = (f"[{fmt(d['ci_low'], places)}, {fmt(d['ci_high'], places)}]"
                            f" (very uncertain at n = {d['n']})")
                label = f"{d['replication']} ({d['n']} seeds)"
            elif d["n"] == 2:
                interval = "not reported (partial replication)"
                label = "partial replication (2 seeds)"
                notes.append(
                    f"Configuration `{cfg}` is a **partial replication**. Two "
                    f"seeds completed, giving {fmt(d['values'][0], places)} and "
                    f"{fmt(d['values'][1], places)}, a mean of "
                    f"{fmt(d['mean'], places)} and a range of "
                    f"{fmt(d['range'], places)}. A standard deviation from two "
                    "points is not a stable estimate of the seed variation, no "
                    "t interval is reported from two points, and the uncertainty "
                    "for this configuration is not resolved."
                )
            elif d["n"] == 1:
                interval = "not reported (unreplicated)"
                label = "unreplicated (1 seed)"
                notes.append(
                    f"Configuration `{cfg}` is **unreplicated**. A single seed "
                    f"completed, giving {fmt(d['values'][0], places)}. No spread "
                    "can be estimated from one run and no uncertainty statement "
                    "is available for it."
                )
            else:
                interval = "not reported"
                label = "no completed run"
            cells += [
                str(d["n"]),
                fmt(d["mean"], places),
                fmt(d["sd"], places) + (" (unstable, n = 2)" if d["n"] == 2 else ""),
                fmt(d["min"], places),
                fmt(d["max"], places),
                interval,
                label,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            f"Intervals are Student t intervals at {conf_pct} percent with n - 1 "
            "degrees of freedom. With three seeds that is two degrees of freedom "
            "and the interval is very uncertain. It is reported so that the "
            "weakness of the evidence is visible, not because three runs confer "
            "power to resolve small differences."
        )
        lines.append("")
        for note in notes:
            lines.append(note)
            lines.append("")

    # ---- paired seed-level differences
    lines.append("## Paired seed-level differences")
    lines.append("")
    lines.append(
        "Each difference below is formed within a seed, seed 42 against seed 42 "
        "and so on, and never by subtracting one configuration mean from another. "
        "Pairing removes the component of the variation that is common to a seed "
        "and leaves the method contrast, which is the quantity that speaks to the "
        "headline question."
    )
    lines.append("")
    paired_store: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for reference in PAIRED_REFERENCES:
        if reference not in configs:
            lines.append(
                f"### Against `{reference}`"
            )
            lines.append("")
            lines.append(
                f"No `{reference}` run is present among the verified results, so "
                "no paired difference against it can be formed."
            )
            lines.append("")
            continue
        lines.append(f"### Against `{reference}`")
        lines.append("")
        for metric_key, metric_label in METRICS:
            places = 4 if metric_key == "preference_acc" else 6
            ref_per_seed = included_values(runs, reference, metric_key, excluded)
            header = (["Configuration"] + [f"seed {s}" for s in all_seeds]
                      + ["paired n", "mean paired difference", "range"])
            lines.append(f"**{metric_label.capitalize()}**")
            lines.append("")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for cfg in configs:
                if cfg == reference:
                    continue
                per_seed = included_values(runs, cfg, metric_key, excluded)
                pd = paired_differences(per_seed, ref_per_seed)
                paired_store[(reference, metric_key, cfg)] = pd
                cells = [cfg]
                diffs = pd["per_seed"]  # type: ignore[assignment]
                for s in all_seeds:
                    cells.append(signed(diffs.get(s), places) if s in diffs else "n/a")  # type: ignore[union-attr]
                rng = ("n/a" if pd["n"] == 0 else
                       f"[{signed(pd['min'], places)}, {signed(pd['max'], places)}]")
                cells += [str(pd["n"]), signed(pd["mean"], places), rng]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    # ---- headline interpretation
    lines.append("## Headline interpretation")
    lines.append("")
    for metric_key, metric_label in METRICS:
        places = 4 if metric_key == "preference_acc" else 6
        groups = []
        means = {}
        for cfg in configs:
            per_seed = included_values(runs, cfg, metric_key, excluded)
            vals = [per_seed[s] for s in sorted(per_seed)]
            if vals:
                groups.append(vals)
                means[cfg] = statistics.fmean(vals)
        lines.append(f"### {metric_label.capitalize()}")
        lines.append("")
        if len(means) < 2:
            lines.append(
                "Fewer than two configurations carry results, so the question "
                "cannot be answered for this metric."
            )
            lines.append("")
            continue
        hi_cfg = max(means, key=means.get)
        lo_cfg = min(means, key=means.get)
        spread = means[hi_cfg] - means[lo_cfg]
        psd = pooled_sd(groups)
        lines.append(
            f"The highest configuration mean is {fmt(means[hi_cfg], places)} "
            f"(`{hi_cfg}`) and the lowest is {fmt(means[lo_cfg], places)} "
            f"(`{lo_cfg}`), so the largest difference between any two "
            f"configuration means is {fmt(spread, places)}."
        )
        if psd is None:
            lines.append("")
            lines.append(
                "No configuration carries two or more seeds, so the training-seed "
                "variation cannot be estimated and the question cannot be "
                "answered for this metric."
            )
            lines.append("")
            continue
        ratio = spread / psd if psd > 0 else float("inf")
        lines.append(
            f"The pooled within-configuration standard deviation across training "
            f"seeds is {fmt(psd, places)}, so the largest between-configuration "
            f"difference is {fmt(ratio, 2)} times the seed-level variation."
        )

        # paired evidence
        consistent: List[str] = []
        inconsistent: List[str] = []
        for reference in PAIRED_REFERENCES:
            for cfg in configs:
                pd = paired_store.get((reference, metric_key, cfg))
                if pd is None or pd["n"] < 2:
                    continue
                label = f"`{cfg}` against `{reference}`"
                if pd["sign_consistent"]:
                    consistent.append(label)
                else:
                    inconsistent.append(label)
        if consistent or inconsistent:
            n_keep, n_flip = len(consistent), len(inconsistent)
            lines.append(
                f"Of the paired contrasts reported above that share at least two "
                f"seeds, {n_keep} {'keeps' if n_keep == 1 else 'keep'} the same "
                f"sign across every seed and {n_flip} "
                f"{'changes' if n_flip == 1 else 'change'} sign from one seed to "
                "another."
                + (" Contrasts that change sign are: "
                   + ", ".join(sorted(inconsistent)) + "." if inconsistent else "")
            )
        lines.append("")

        exceeds_seed_noise = ratio >= 2.0
        if ratio < 1.0:
            answer = (
                f"**No.** The differences between configurations in "
                f"{metric_label} are smaller than the variation a change of "
                f"training seed produces within a single configuration. The "
                f"largest difference between two configuration means is "
                f"{fmt(spread, places)}, against a seed-level standard deviation "
                f"of {fmt(psd, places)}. On this evidence the configurations are "
                "not separated by their downstream behaviour, and no ranking of "
                "them by this metric is supported."
            )
        elif ratio < 2.0:
            answer = (
                f"**Not clearly.** The largest difference between two "
                f"configuration means in {metric_label} is {fmt(spread, places)}, "
                f"of the same order as the seed-level standard deviation of "
                f"{fmt(psd, places)}. A difference of that size is what a change "
                "of training seed alone can produce, so it does not establish a "
                "difference between the methods. No ranking of the "
                "configurations by this metric is supported by the present "
                "evidence."
            )
        else:
            answer = (
                f"**Larger than seed variation, but on thin evidence.** The "
                f"largest difference between two configuration means in "
                f"{metric_label} is {fmt(spread, places)}, which is "
                f"{fmt(ratio, 2)} times the seed-level standard deviation of "
                f"{fmt(psd, places)}. The magnitude therefore exceeds what a "
                "change of training seed alone produced here. The estimate of "
                "the seed variation itself rests on very few runs, so the "
                "comparison should be read as suggestive rather than settled."
            )
        if inconsistent:
            if exceeds_seed_noise:
                answer += (
                    " One qualification follows from the paired contrasts: "
                    f"{len(inconsistent)} of them "
                    f"{'reverses' if len(inconsistent) == 1 else 'reverse'} sign "
                    "between seeds, so the effect is not uniform across seeds "
                    "even where the aggregate magnitude is large."
                )
            else:
                answer += (
                    " The paired contrasts reinforce this reading: at least one "
                    "contrast reverses sign between seeds, which a method effect "
                    "larger than the seed variation would not do."
                )
        lines.append(answer)
        lines.append("")

        aov = anova_oneway(groups)
        if aov is not None:
            lines.append(
                f"*Secondary, not the basis of the reading above.* A one-way "
                f"analysis of variance across the {int(aov['k_groups'])} "
                f"configurations gives F({int(aov['df_between'])}, "
                f"{int(aov['df_within'])}) = {fmt(aov['f'], 3)}, p = "
                f"{fmt(aov['p'], 4)} at the {alpha:g} level. With this few runs "
                "per configuration the test has very little power and its p "
                "value is reported for completeness only."
            )
        else:
            lines.append(
                "*Secondary.* A one-way analysis of variance could not be "
                "computed for this metric, either because too few "
                "configurations carry replicated runs or because SciPy is not "
                "available."
            )
        lines.append("")

    # ---- reproduction check
    repro_metric_label = dict(METRICS)[REPRO_METRIC]
    lines.append("## Reproduction check against the published single-seed grid")
    lines.append("")
    lines.append(
        f"The published pilot values were read from `{grid_csv}`. The training "
        f"seed of the published runs was read from `{dpo_config}` and is "
        f"{baseline_seed}, so the newly trained seed {baseline_seed} run of each "
        f"configuration should reproduce the published {repro_metric_label} up "
        f"to library and hardware nondeterminism. The tolerance is "
        f"{tolerance:g} in absolute value."
    )
    lines.append("")
    if not published_rows:
        lines.append(
            "The published grid file was not found, so no reproduction check "
            "could be performed and no exclusion was applied."
        )
        lines.append("")
    else:
        header = ["Configuration", f"Published (seed {baseline_seed})",
                  f"Regenerated (seed {baseline_seed})", "Deviation", "Verdict"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for cfg in configs:
            rec = repro[cfg]
            lines.append("| " + " | ".join([
                cfg,
                fmt(rec["published"], 4),          # type: ignore[arg-type]
                fmt(rec["regenerated"], 4),        # type: ignore[arg-type]
                signed(rec["deviation"], 4),       # type: ignore[arg-type]
                str(rec["verdict"]),
            ]) + " |")
        lines.append("")
        exact = [c for c in configs if repro[c]["verdict"] == "exact"]
        within = [c for c in configs if repro[c]["verdict"] == "within tolerance"]
        if exact:
            lines.append(
                "Exact match against the published value: "
                + ", ".join(f"`{c}`" for c in exact) + "."
            )
            lines.append("")
        if within:
            lines.append(
                f"Agreement within the stated tolerance of {tolerance:g}: "
                + ", ".join(f"`{c}`" for c in within) + "."
            )
            lines.append("")
        if discrepant:
            lines.append("### Discrepant reproductions")
            lines.append("")
            for cfg in sorted(discrepant):
                rec = repro[cfg]
                lines.append(
                    f"Configuration `{cfg}` regenerated "
                    f"{fmt(rec['regenerated'], 4)} against a published "  # type: ignore[arg-type]
                    f"{fmt(rec['published'], 4)}, a deviation of "        # type: ignore[arg-type]
                    f"{signed(rec['deviation'], 4)}, which exceeds the "  # type: ignore[arg-type]
                    f"tolerance of {tolerance:g}. The seed {baseline_seed} run of "
                    "this configuration has been excluded from its pooled "
                    "statistics above, which are computed over the remaining "
                    "seeds only."
                )
                lines.append("")
            lines.append(
                "The likely causes, in the order in which they should be "
                "investigated, are the following. First, library version drift "
                "between the environment that produced the published grid and "
                "the environment in use now: the training and scoring stack "
                "includes PyTorch, Transformers, TRL and PEFT, and a change in "
                "any of them can move a preference accuracy by more than the "
                "tolerance without any error having occurred. Second, "
                "nondeterminism in batched half-precision kernels: reductions in "
                "fp16 and bf16 are not associative, so batching and kernel "
                "selection can change the last bits of a log probability, and a "
                "preference accuracy is a count of sign comparisons that can "
                "flip when two continuations score almost equally. The "
                "per-run provenance files "
                "`results/eval/seed_study/*.meta.json` record the library "
                "versions and the CUDA device of every new run and are the "
                "place to begin. Until the cause is identified the discrepant "
                "seed must not be pooled with the others."
            )
            lines.append("")
        else:
            lines.append(
                "No configuration deviates from its published value by more than "
                f"{tolerance:g}, so no run was excluded from the pooled "
                "statistics."
            )
            lines.append("")

    # ---- provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append("| Run | Configuration | Seed | n | Verified | Evaluation file | Provenance file |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["config"], r["seed"])):
        meta_name = Path(r["meta_file"]).name if r.get("meta_file") else "absent"
        lines.append(
            f"| eta{r['eta_tag']} | {r['config']} | {r['seed']} | {r['n']} | "
            f"{'yes' if r.get('has_marker') else 'no marker'} | "
            f"`{Path(r['file']).name}` | `{meta_name}` |"
        )
    lines.append("")
    return "\n".join(lines), repro


def write_csv(
    runs: Sequence[Dict],
    out_csv: Path,
    grid_csv: Path,
    dpo_config: Path,
    confidence: float,
    tolerance: float,
    repro: Dict[str, Dict[str, object]],
) -> None:
    configs = sorted({r["config"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    published_rows = load_published(grid_csv)
    base_seed = resolve_baseline_seed(seeds, dpo_config)
    excluded: Dict[str, Optional[int]] = {
        cfg: rec["exclude_seed"] for cfg, rec in repro.items()  # type: ignore[misc]
    }

    fieldnames = (
        ["eta_nominal_pct", "config", "metric"]
        + [f"seed_{s}" for s in seeds]
        + ["n", "mean", "sd", "min", "max", "range",
           "ci_low", "ci_high", "confidence", "df", "t_crit",
           "replication_label",
           "excluded_seed", "reproduction_verdict",
           "published_baseline_value", "new_baseline_seed", "new_baseline_value",
           "deviation", "flagged"]
        + [f"paired_vs_{ref}_{field}"
           for ref in PAIRED_REFERENCES
           for field in ("n", "mean", "min", "max")]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in configs:
            cfg_runs = sorted([r for r in runs if r["config"] == cfg],
                              key=lambda r: r["seed"])
            eta_tag = cfg_runs[0]["eta_tag"]
            pub_row = published_row(published_rows, cfg, eta_tag)
            for metric_key, _ in METRICS:
                per_seed = included_values(runs, cfg, metric_key, excluded)
                values = [per_seed[s] for s in sorted(per_seed)]
                d = describe(values, confidence)
                pub_val = published_value(pub_row, metric_key)
                base_val = per_seed.get(base_seed) if base_seed is not None else None
                dev = ((base_val - pub_val)
                       if (pub_val is not None and base_val is not None) else None)
                row = {
                    "eta_nominal_pct": eta_tag,
                    "config": cfg,
                    "metric": metric_key,
                    "n": d["n"],
                    "mean": d["mean"],
                    "sd": d["sd"],
                    "min": d["min"],
                    "max": d["max"],
                    "range": d["range"],
                    "ci_low": d["ci_low"],
                    "ci_high": d["ci_high"],
                    "confidence": confidence,
                    "df": d["df"],
                    "t_crit": d["t_crit"],
                    "replication_label": d["replication"],
                    "excluded_seed": excluded.get(cfg),
                    "reproduction_verdict": repro.get(cfg, {}).get("verdict"),
                    "published_baseline_value": pub_val,
                    "new_baseline_seed": base_seed,
                    "new_baseline_value": base_val,
                    "deviation": dev,
                    "flagged": "" if dev is None else int(abs(dev) > tolerance),
                }
                for s in seeds:
                    row[f"seed_{s}"] = per_seed.get(s)
                for ref in PAIRED_REFERENCES:
                    ref_per_seed = included_values(runs, ref, metric_key, excluded)
                    pd = (paired_differences(per_seed, ref_per_seed)
                          if ref in configs and ref != cfg else None)
                    for field in ("n", "mean", "min", "max"):
                        row[f"paired_vs_{ref}_{field}"] = pd[field] if pd else None
                writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval_dir", default=str(DEFAULT_EVAL_DIR))
    ap.add_argument("--grid_csv", default=str(DEFAULT_GRID_CSV))
    ap.add_argument("--dpo_config", default=str(DEFAULT_DPO_CONFIG),
                    help="Shared DPO configuration, read to recover the "
                         "training seed used for the published runs.")
    ap.add_argument("--out_md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out_csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level of the secondary analysis of "
                         "variance line.")
    ap.add_argument("--flag_threshold", type=float, default=0.02,
                    help="Absolute deviation from the published value above "
                         "which the reproduction is treated as discrepant and "
                         "the baseline seed is excluded from the pooled "
                         "statistics of that configuration.")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        raise SystemExit(f"[summary] evaluation directory not found: {eval_dir}")

    runs = load_runs(eval_dir)
    if not runs:
        raise SystemExit(f"[summary] no usable evaluation JSON found in {eval_dir}")

    grid_csv = Path(args.grid_csv)
    dpo_config = Path(args.dpo_config)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    md, repro = build_markdown(runs, eval_dir, grid_csv, dpo_config,
                               args.confidence, args.flag_threshold, args.alpha)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    write_csv(runs, out_csv, grid_csv, dpo_config, args.confidence,
              args.flag_threshold, repro)

    discrepant = [c for c, rec in repro.items() if rec["verdict"] == "discrepant"]
    if discrepant:
        seed = repro[discrepant[0]]["seed"]
        bar = "!" * 78
        print(bar)
        print("[summary] WARNING: the reproduction check failed for "
              f"{len(discrepant)} configuration(s): {', '.join(sorted(discrepant))}")
        for cfg in sorted(discrepant):
            rec = repro[cfg]
            print(f"[summary]   {cfg}: published {rec['published']}, "
                  f"regenerated {rec['regenerated']}, deviation "
                  f"{rec['deviation']:+.4f}, tolerance {args.flag_threshold:g}")
        print(f"[summary] seed {seed} has been EXCLUDED from the pooled statistics "
              "of those configurations. The pooled statistics reported for them "
              "cover the remaining seeds only.")
        print("[summary] likely causes: library version drift between the "
              "environment that produced the published grid and the environment "
              "in use now (PyTorch, Transformers, TRL, PEFT), or nondeterminism "
              "in batched fp16 and bf16 kernels, whose non-associative "
              "reductions can flip sign comparisons between near-equal "
              "continuations.")
        print("[summary] inspect results/eval/seed_study/*.meta.json for the "
              "recorded library versions and CUDA device of each run.")
        print(bar)

    partial = []
    for cfg in sorted({r["config"] for r in runs}):
        drop = repro.get(cfg, {}).get("exclude_seed")
        n = len([r for r in runs if r["config"] == cfg and r["seed"] != drop])
        if n == 2:
            partial.append(f"{cfg} (2 seeds, partial replication)")
        elif n == 1:
            partial.append(f"{cfg} (1 seed, unreplicated)")
    if partial:
        print("[summary] incomplete configurations: " + "; ".join(partial))

    print(f"[summary] {len(runs)} run(s) summarised")
    print(f"[summary] wrote {out_md}")
    print(f"[summary] wrote {out_csv}")


if __name__ == "__main__":
    main()
