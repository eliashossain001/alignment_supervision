"""Aggregate and then unblind the independent model-based rejection audit.

The script has two stages and they are deliberately separated.

Stage 1 is blinded. It reads only the raw judge outputs, collapses the two runs
of each judge, aggregates across the two judge families under a strict rule, and
freezes the result with a cryptographic hash. During stage 1 the process
installs a read guard that raises if any code attempts to open the hidden key,
the corrupted training file, or the annotator sample. Every path opened during
stage 1 is recorded in the freeze artifact, so the blinding is demonstrable
after the fact and not merely asserted.

Stage 2 is unblinded. It refuses to run unless the freeze artifact exists and
the aggregated blinded table still hashes to the frozen value. Only then does it
join the hidden key and the corrupted training file, derive the two fields that
the audit prompt specification (Section 3) defines as derived rather than asked,
and write the agreement report, the per-stratum results, and the spot-check
sheet.

Nothing about the audit outcome is hard coded. Every number in every emitted
artifact is computed from the raw judge outputs and the key.

Usage:

    python analysis/scripts/aggregate_model_audit.py stage1
    python analysis/scripts/aggregate_model_audit.py stage2

Both subcommands accept --raw-dir and --outdir so the pipeline can be exercised
against synthetic judge outputs without writing into analysis/.
"""
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

JUDGES = ("llama", "phi")
RUNS = (1, 2)

# Consensus admission rule used throughout the paper (k of 4 verifiers at
# threshold 0.7). Only k is needed here, because the key already records the
# realised number of passing verifiers per audited pair.
K_ADMIT = 3

SUBSTANTIVE = ("option_1", "option_2")
ABSTAIN = ("tie", "uncertain")
OPTION_VALUES = SUBSTANTIVE + ABSTAIN
PARSE_ERROR = "parse_error"
INCONSISTENT = "inconsistent"
UNRESOLVED = "unresolved"

AMBIGUITY_VALUES = ("clear", "ambiguous", "underdetermined")
YESNO = ("yes", "no")
REASON_SAFETY = "safety violation"

N_BOOT = 2000
BOOT_SEED = 42
SPOTCHECK_SEED = 42
SPOTCHECK_N = 5
SPOTCHECK_TRUNC = 400

MODEL_AUDIT_DISCLAIMER = (
    "**This is an independent model-based audit. It is not human ground "
    "truth.** The labels aggregated here were produced by two instruction "
    "tuned language models drawn from families that appear in neither verifier "
    "pool and are not the trained policy. Under Section 5 of "
    "`analysis/rejection_audit_protocol.md` this is the minimum acceptable "
    "fallback annotation source, explicitly weaker evidence than blinded human "
    "annotation, because a model judge can share failure modes with the "
    "verifier pool in ways a human annotator does not. Every quantity below "
    "must be read as an independent model-based estimate and never as a "
    "verified human label."
)

# ---------------------------------------------------------------------------
# stage 1 read guard
# ---------------------------------------------------------------------------

FORBIDDEN_NAME_FRAGMENTS = (
    "rejection_audit_key",
    "rejection_audit_sample",
    "model_audit_hidden_join",
)
FORBIDDEN_DIR_FRAGMENTS = ("/data/corrupted/", "/data/processed/", "/data/raw/")


class BlindingViolation(RuntimeError):
    """Raised when blinded stage 1 touches an unblinding artifact."""


class BlindGuard:
    """Context manager that blocks and logs file access during stage 1.

    The guard patches ``builtins.open`` and ``io.open``. Any attempt to open a
    path whose name or parent directory matches a withheld artifact raises
    immediately, so a blinding failure is a crash rather than a silent bias.
    Every path opened while the guard is active is recorded, and the record is
    written into the freeze artifact.
    """

    def __init__(self) -> None:
        self.opened: list[str] = []
        self._saved: list[tuple[object, str, object]] = []

    def _check(self, file):  # noqa: ANN001
        try:
            path = Path(file).resolve()
        except (TypeError, ValueError, OSError):
            return
        text = str(path).replace("\\", "/")
        name = path.name
        for frag in FORBIDDEN_NAME_FRAGMENTS:
            if frag in name:
                raise BlindingViolation(
                    f"stage 1 is blinded and attempted to open {text}; "
                    "the hidden key and the annotator sample are stage 2 only"
                )
        for frag in FORBIDDEN_DIR_FRAGMENTS:
            if frag in text:
                raise BlindingViolation(
                    f"stage 1 is blinded and attempted to open {text}; "
                    "the corrupted training data is stage 2 only"
                )
        rel = text
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            pass
        if rel not in self.opened:
            self.opened.append(rel)

    def __enter__(self) -> "BlindGuard":
        guard = self

        for module in (builtins, io):
            original = module.open

            def patched(file, *args, _orig=original, **kwargs):  # noqa: ANN001
                guard._check(file)
                return _orig(file, *args, **kwargs)

            self._saved.append((module, "open", original))
            module.open = patched  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        for module, attr, original in self._saved:
            setattr(module, attr, original)
        self._saved.clear()


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def norm_str(value) -> str:  # noqa: ANN001
    if value is None:
        return ""
    return str(value).strip().lower()


def norm_bool(value) -> bool:  # noqa: ANN001
    if isinstance(value, bool):
        return value
    return norm_str(value) in {"true", "yes", "1", "ok", "y"}


def norm_int(value):  # noqa: ANN001
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def fmt_rate(value) -> str:  # noqa: ANN001
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def fmt_ci(stat: dict) -> str:
    if stat["n_den"] == 0:
        return "n/a"
    return (
        f"{stat['rate']:.4f} [{fmt_rate(stat['ci_low'])}, "
        f"{fmt_rate(stat['ci_high'])}] ({stat['n_num']}/{stat['n_den']})"
    )


def truncate(text: str, limit: int = SPOTCHECK_TRUNC) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + " [truncated]"


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def bootstrap_rate(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> dict:
    """Rate with a percentile bootstrap interval, resampling examples.

    ``num`` and ``den`` are boolean arrays over the examples in scope. The
    resample is over examples, so the interval reflects sampling variation in
    which examples were drawn, which is the unit the audit sample was drawn on.
    A fresh generator seeded with ``seed`` is created per call, so the result
    does not depend on the order in which metrics are computed.
    """
    num = np.asarray(num, dtype=bool)
    den = np.asarray(den, dtype=bool)
    n = int(num.size)
    n_num = int((num & den).sum())
    n_den = int(den.sum())
    out = {
        "n_num": n_num,
        "n_den": n_den,
        "rate": (n_num / n_den) if n_den else float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_boot_valid": 0,
    }
    if n == 0 or n_den == 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    dsum = den[idx].sum(axis=1)
    nsum = (num & den)[idx].sum(axis=1)
    ok = dsum > 0
    if not ok.any():
        return out
    vals = nsum[ok] / dsum[ok]
    out["ci_low"] = float(np.percentile(vals, 2.5))
    out["ci_high"] = float(np.percentile(vals, 97.5))
    out["n_boot_valid"] = int(ok.sum())
    return out


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa on paired categorical labels."""
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# stage 1: load and collapse
# ---------------------------------------------------------------------------


def load_raw(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({exc})")
            pref = norm_str(d.get("preferred_option"))
            displayed = norm_str(d.get("displayed_A_is"))
            if displayed in {"1", "option1"}:
                displayed = "option_1"
            elif displayed in {"2", "option2"}:
                displayed = "option_2"
            elif displayed not in {"option_1", "option_2"}:
                displayed = ""
            records.append({
                "audit_id": str(d.get("audit_id", "")).strip(),
                "judge": norm_str(d.get("judge")),
                "run": norm_int(d.get("run")),
                "preferred_option": pref if pref in OPTION_VALUES else "",
                "pair_ambiguity": norm_str(d.get("pair_ambiguity")),
                "both_acceptable": norm_str(d.get("both_acceptable")),
                "both_poor_or_unsafe": norm_str(d.get("both_poor_or_unsafe")),
                "confidence": norm_int(d.get("confidence")),
                "reason_category": norm_str(d.get("reason_category")),
                "brief_rationale": str(d.get("brief_rationale", "")).strip(),
                "parse_ok": norm_bool(d.get("parse_ok")),
                "displayed_A_is": displayed,
            })
    return records


def index_raw(records: list[dict], judge: str, path: Path) -> dict[str, dict[int, dict]]:
    table: dict[str, dict[int, dict]] = {}
    for rec in records:
        if rec["judge"] and rec["judge"] != judge:
            raise SystemExit(
                f"{path}: record for audit_id {rec['audit_id']} declares judge "
                f"'{rec['judge']}' but the file is the {judge} file"
            )
        if rec["run"] not in RUNS:
            raise SystemExit(
                f"{path}: audit_id {rec['audit_id']} has run={rec['run']}, "
                f"expected one of {list(RUNS)}"
            )
        slot = table.setdefault(rec["audit_id"], {})
        if rec["run"] in slot:
            raise SystemExit(
                f"{path}: duplicate record for audit_id {rec['audit_id']} "
                f"run {rec['run']}"
            )
        slot[rec["run"]] = rec
    bad = [aid for aid, slot in table.items() if set(slot) != set(RUNS)]
    if bad:
        raise SystemExit(
            f"{path}: {len(bad)} audit_ids do not have exactly {len(RUNS)} runs "
            f"(first: {sorted(bad)[:5]})"
        )
    return table


def run_value(rec: dict, field: str) -> str:
    """Field value for one run, with parse failures marked as such.

    A parse failure is missing data. It is never conflated with 'tie' or
    'uncertain', which are substantive judgements the prompt explicitly invites.
    """
    if not rec["parse_ok"]:
        return PARSE_ERROR
    value = rec[field]
    return value if value else PARSE_ERROR


def collapse_pair(v1: str, v2: str, policy: str) -> str:
    """Collapse a judge's two runs on one categorical field.

    Equal values collapse to that value. Genuine disagreement between two
    substantive values collapses to 'inconsistent'. When exactly one run failed
    to parse the behaviour is set by ``policy``: 'fallback' treats the parse
    failure as missing data and uses the surviving run, 'inconsistent' treats it
    as a disagreement. See the judgement-call note in the emitted reports.
    """
    if v1 == v2:
        return v1
    if PARSE_ERROR in (v1, v2):
        if policy == "fallback":
            return v2 if v1 == PARSE_ERROR else v1
        return INCONSISTENT
    return INCONSISTENT


def collapse_confidence(rec1: dict, rec2: dict):
    """Per-family collapsed confidence: the minimum across the two runs.

    The minimum is used rather than the mean because the audit is deployed as a
    conservative screen. A family is treated as confident only if it was
    confident on both passes, so a single low-confidence pass cannot be averaged
    away by a high-confidence one.
    """
    vals = [r["confidence"] for r in (rec1, rec2)
            if r["parse_ok"] and r["confidence"] is not None]
    return min(vals) if vals else None


def strict_cross(v_llama: str, v_phi: str, allowed: tuple[str, ...]) -> str:
    """Cross-family strict aggregation: the shared value or 'unresolved'.

    There is no forced adjudication. 'inconsistent' and 'parse_error' can never
    become a final label even when both families produce them, because a shared
    absence of a label is not a shared label.
    """
    if v_llama == v_phi and v_llama in allowed:
        return v_llama
    return UNRESOLVED


def build_blinded_rows(raw: dict[str, dict[str, dict[int, dict]]],
                       policy: str) -> list[dict]:
    audit_ids = sorted(raw["llama"])
    if set(raw["llama"]) != set(raw["phi"]):
        only_l = sorted(set(raw["llama"]) - set(raw["phi"]))[:5]
        only_p = sorted(set(raw["phi"]) - set(raw["llama"]))[:5]
        raise SystemExit(
            "the two judge files cover different audit_ids "
            f"(llama only: {only_l}, phi only: {only_p})"
        )

    rows = []
    for aid in audit_ids:
        row: dict[str, object] = {"audit_id": aid}
        per_family: dict[str, dict[str, object]] = {}
        for judge in JUDGES:
            r1, r2 = raw[judge][aid][1], raw[judge][aid][2]
            fields = {}
            for field, out_name in (
                ("preferred_option", ""),
                ("pair_ambiguity", "ambiguity"),
                ("both_acceptable", "both_acceptable"),
                ("both_poor_or_unsafe", "both_poor"),
                ("reason_category", "reason"),
            ):
                v1, v2 = run_value(r1, field), run_value(r2, field)
                fields[field] = collapse_pair(v1, v2, policy)
                if field == "preferred_option":
                    row[f"{judge}_run1"] = v1
                    row[f"{judge}_run2"] = v2
                    row[f"{judge}_collapsed"] = fields[field]
                else:
                    row[f"{judge}_{out_name}_collapsed"] = fields[field]
            conf = collapse_confidence(r1, r2)
            fields["confidence"] = conf
            row[f"{judge}_confidence_collapsed"] = "" if conf is None else conf
            for run, rec in ((1, r1), (2, r2)):
                row[f"{judge}_run{run}_parse_ok"] = str(bool(rec["parse_ok"])).lower()
                row[f"{judge}_run{run}_confidence"] = (
                    "" if rec["confidence"] is None else rec["confidence"]
                )
                row[f"{judge}_run{run}_displayed_A_is"] = rec["displayed_A_is"]
                row[f"{judge}_run{run}_reason"] = run_value(rec, "reason_category")
            per_family[judge] = fields

        row["strict_label"] = strict_cross(
            per_family["llama"]["preferred_option"],
            per_family["phi"]["preferred_option"],
            OPTION_VALUES,
        )
        row["strict_ambiguity"] = strict_cross(
            per_family["llama"]["pair_ambiguity"],
            per_family["phi"]["pair_ambiguity"],
            AMBIGUITY_VALUES,
        )
        row["strict_both_acceptable"] = strict_cross(
            per_family["llama"]["both_acceptable"],
            per_family["phi"]["both_acceptable"], YESNO,
        )
        row["strict_both_poor"] = strict_cross(
            per_family["llama"]["both_poor_or_unsafe"],
            per_family["phi"]["both_poor_or_unsafe"], YESNO,
        )
        row["strict_reason"] = strict_cross(
            per_family["llama"]["reason_category"],
            per_family["phi"]["reason_category"],
            tuple(sorted({per_family["llama"]["reason_category"],
                          per_family["phi"]["reason_category"]}
                         - {PARSE_ERROR, INCONSISTENT})),
        )
        confs = [per_family[j]["confidence"] for j in JUDGES]
        row["strict_confidence"] = (
            "" if any(c is None for c in confs) else min(confs)  # type: ignore[type-var]
        )
        # Permissive variants of the two flags that feed the derived
        # safety-preserving field. Recorded so the derivation can be reported
        # under both a strict and a permissive reading.
        row["both_poor_any"] = str(any(
            run_value(raw[j][aid][r], "both_poor_or_unsafe") == "yes"
            for j in JUDGES for r in RUNS
        )).lower()
        row["reason_safety_any"] = str(any(
            run_value(raw[j][aid][r], "reason_category") == REASON_SAFETY
            for j in JUDGES for r in RUNS
        )).lower()
        row["all_parse_ok"] = str(all(
            raw[j][aid][r]["parse_ok"] for j in JUDGES for r in RUNS
        )).lower()
        rows.append(row)
    return rows


BLINDED_COLUMNS = (
    ["audit_id"]
    + [f"{j}_run{r}" for j in JUDGES for r in RUNS]
    + [f"{j}_collapsed" for j in JUDGES]
    + ["strict_label"]
    + [f"{j}_{f}_collapsed" for j in JUDGES
       for f in ("ambiguity", "both_acceptable", "both_poor", "reason", "confidence")]
    + ["strict_ambiguity", "strict_both_acceptable", "strict_both_poor",
       "strict_reason", "strict_confidence", "both_poor_any", "reason_safety_any"]
    + [f"{j}_run{r}_parse_ok" for j in JUDGES for r in RUNS]
    + ["all_parse_ok"]
    + [f"{j}_run{r}_confidence" for j in JUDGES for r in RUNS]
    + [f"{j}_run{r}_displayed_A_is" for j in JUDGES for r in RUNS]
    + [f"{j}_run{r}_reason" for j in JUDGES for r in RUNS]
)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def stage1(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir).resolve()
    outdir = Path(args.outdir).resolve()
    raw_paths = {j: raw_dir / f"model_audit_{j}_raw.jsonl" for j in JUDGES}
    for judge, path in raw_paths.items():
        if not path.exists():
            print(f"ERROR: missing raw judge output for {judge}: {path}",
                  file=sys.stderr)
            return 2

    guard = BlindGuard()
    with guard:
        raw = {}
        counts = {}
        for judge, path in raw_paths.items():
            records = load_raw(path)
            counts[judge] = len(records)
            raw[judge] = index_raw(records, judge, path)
        rows = build_blinded_rows(raw, args.parse_failure_policy)
        agg_path = outdir / "model_audit_aggregated_blinded.csv"
        write_csv(agg_path, BLINDED_COLUMNS, rows)
        agg_sha = sha256_file(agg_path)
        raw_meta = {
            judge: {
                "path": rel_to_root(path),
                "sha256": sha256_file(path),
                "mtime_utc": iso_from_mtime(path),
                "n_records": counts[judge],
                "n_audit_ids": len(raw[judge]),
            }
            for judge, path in raw_paths.items()
        }

    # The freeze timestamp is derived from the filesystem mtimes of the raw
    # judge outputs, never from the wall clock, so that re-running stage 1 on
    # unchanged inputs reproduces the freeze record byte for byte.
    freeze_iso = max(iso_from_mtime(p) for p in raw_paths.values())
    freeze = {
        "schema_version": 1,
        "stage": "1_blinded",
        "frozen_at_utc": freeze_iso,
        "frozen_at_source": (
            "maximum filesystem mtime of the raw judge outputs; the wall clock "
            "is deliberately not consulted so the record is reproducible"
        ),
        "judges": list(JUDGES),
        "runs_per_judge": len(RUNS),
        "parse_failure_policy": args.parse_failure_policy,
        "confidence_collapse_rule": (
            "per family, the minimum confidence across the two runs; across "
            "families, the minimum of the two per-family values"
        ),
        "aggregation_rule": (
            "within family, majority of two runs on preferred_option, with "
            "disagreement collapsing to 'inconsistent'; across families, the "
            "shared value or 'unresolved', with no forced adjudication"
        ),
        "raw_inputs": raw_meta,
        "aggregated_blinded": {
            "path": rel_to_root(agg_path),
            "sha256": agg_sha,
            "n_rows": len(rows),
        },
        "blinding": {
            "guard": "builtins.open and io.open patched for the whole of stage 1",
            "forbidden_name_fragments": list(FORBIDDEN_NAME_FRAGMENTS),
            "forbidden_dir_fragments": list(FORBIDDEN_DIR_FRAGMENTS),
            "paths_opened_during_stage1": guard.opened,
        },
    }
    freeze_path = outdir / "model_audit_freeze.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")

    print("stage 1 (blinded) complete")
    print(f"  rows aggregated                : {len(rows)}")
    for judge in JUDGES:
        print(f"  {judge:<6} raw sha256           : {raw_meta[judge]['sha256']}")
        print(f"  {judge:<6} records / audit_ids  : "
              f"{raw_meta[judge]['n_records']} / {raw_meta[judge]['n_audit_ids']}")
    print(f"  aggregated csv sha256          : {agg_sha}")
    print(f"  freeze timestamp (from mtime)  : {freeze_iso}")
    print(f"  paths opened while blinded     : {guard.opened}")
    print(f"  wrote {rel_to_root(agg_path)}")
    print(f"  wrote {rel_to_root(freeze_path)}")
    dist = Counter(r["strict_label"] for r in rows)
    print("  strict label distribution      : "
          + ", ".join(f"{k}={dist[k]}" for k in sorted(dist)))
    return 0


# ---------------------------------------------------------------------------
# stage 2: hash guard, join, derive
# ---------------------------------------------------------------------------


def verify_freeze(outdir: Path) -> dict:
    freeze_path = outdir / "model_audit_freeze.json"
    agg_path = outdir / "model_audit_aggregated_blinded.csv"
    if not freeze_path.exists():
        print(
            "ERROR: no freeze record at "
            f"{freeze_path}. Stage 2 unblinds the audit and may not run before "
            "stage 1 has frozen the blinded labels. Run stage1 first.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    freeze = json.loads(freeze_path.read_text())
    if not agg_path.exists():
        print(f"ERROR: frozen aggregate {agg_path} is missing.", file=sys.stderr)
        raise SystemExit(3)
    current = sha256_file(agg_path)
    frozen = freeze["aggregated_blinded"]["sha256"]
    if current != frozen:
        print(
            "ERROR: the aggregated blinded table has changed since it was "
            "frozen.\n"
            f"  file    : {agg_path}\n"
            f"  frozen  : {frozen}\n"
            f"  current : {current}\n"
            "Stage 2 refuses to unblind a table that was edited after the "
            "freeze, because the anti-circularity guarantee of this audit rests "
            "on the labels being fixed before the key is opened. Re-run stage 1 "
            "if the raw judge outputs legitimately changed.",
            file=sys.stderr,
        )
        raise SystemExit(4)
    return freeze


def verify_raw_against_freeze(freeze: dict, raw_dir: Path) -> None:
    for judge in JUDGES:
        path = raw_dir / f"model_audit_{judge}_raw.jsonl"
        if not path.exists():
            print(f"ERROR: missing raw judge output {path}", file=sys.stderr)
            raise SystemExit(3)
        current = sha256_file(path)
        frozen = freeze["raw_inputs"][judge]["sha256"]
        if current != frozen:
            print(
                f"ERROR: raw judge file {path} has changed since the freeze.\n"
                f"  frozen  : {frozen}\n"
                f"  current : {current}\n"
                "Re-run stage 1 to re-freeze before unblinding.",
                file=sys.stderr,
            )
            raise SystemExit(4)


def load_key(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_corrupted(path: Path, wanted: set[str]) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] in wanted:
                out[d["id"]] = d
    return out


def recorded_option(user_choice: str) -> str:
    """Canonical option that the recorded preference points to.

    user_choice 'A' means response_a is the recorded preferred response, and
    response_a is option_1 in the blinded sample built by
    build_model_audit_inputs.py. 'B' means option_2.
    """
    uc = norm_str(user_choice).upper()
    if uc == "A":
        return "option_1"
    if uc == "B":
        return "option_2"
    raise SystemExit(f"unrecognised user_choice {user_choice!r}")


def derive_validity(strict_label: str, recorded: str) -> str:
    if strict_label == UNRESOLVED:
        return UNRESOLVED
    if strict_label in ABSTAIN:
        return "uncertain"
    if strict_label == recorded:
        return "yes"
    return "no"


def variant_labels(row: dict) -> dict[str, tuple[str, object]]:
    """Label and gating confidence for each of the four aggregation variants."""
    judgments = []
    for judge in JUDGES:
        for run in RUNS:
            label = str(row[f"{judge}_run{run}"])
            conf = row[f"{judge}_run{run}_confidence"]
            conf = norm_int(conf)
            if label in OPTION_VALUES:
                judgments.append((label, conf))

    out: dict[str, tuple[str, object]] = {}

    # 1. strict cross family (primary)
    strict_conf = norm_int(row["strict_confidence"])
    out["strict_cross_family"] = (str(row["strict_label"]), strict_conf)

    # 2. majority across all four judgments
    def majority(items: list[tuple[str, object]]) -> tuple[str, object]:
        if not items:
            return UNRESOLVED, None
        counts = Counter(lab for lab, _ in items)
        top = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == top]
        if len(winners) != 1:
            return UNRESOLVED, None
        lab = winners[0]
        confs = [c for l, c in items if l == lab and c is not None]
        return lab, (min(confs) if confs else None)

    out["majority_all_four"] = majority(judgments)

    # 3. confidence weighted majority
    if judgments:
        weights: Counter = Counter()
        for lab, conf in judgments:
            weights[lab] += conf if conf is not None else 0
        top = max(weights.values()) if weights else 0
        winners = [lab for lab, w in weights.items() if w == top]
        if len(winners) == 1 and top > 0:
            lab = winners[0]
            confs = [c for l, c in judgments if l == lab and c is not None]
            out["confidence_weighted_majority"] = (
                lab, min(confs) if confs else None)
        else:
            out["confidence_weighted_majority"] = (UNRESOLVED, None)
    else:
        out["confidence_weighted_majority"] = (UNRESOLVED, None)

    # 4. high confidence only
    high = [(lab, c) for lab, c in judgments if c is not None and c >= 4]
    out["high_confidence_only"] = majority(high)
    return out


VARIANTS = (
    ("strict_cross_family", "strict cross-family agreement (PRIMARY)"),
    ("majority_all_four", "majority across all four judgments"),
    ("confidence_weighted_majority", "confidence-weighted majority"),
    ("high_confidence_only", "high-confidence-only judgments (confidence >= 4)"),
)


def build_join(blinded: list[dict], key: list[dict],
               corrupted: dict[str, dict]) -> tuple[list[dict], int]:
    by_id = {r["audit_id"]: r for r in blinded}
    rows = []
    key_mismatch = 0
    for krow in key:
        aid = krow["audit_id"]
        if aid not in by_id:
            raise SystemExit(
                f"audit_id {aid} is in the key but absent from the frozen "
                "aggregate; the audit is incomplete and must not be unblinded"
            )
        brow = by_id[aid]
        src = corrupted[krow["example_id"]]
        recorded = recorded_option(src["user_choice"])

        # Independent consistency check of the join. The key records the
        # recorded preference in the annotator display frame; mapping it back
        # through display_order must reproduce the canonical option.
        swapped = norm_str(krow.get("display_order")) == "swapped"
        disp = norm_str(krow.get("recorded_preference_is_response"))
        if disp in {"response_1", "response_2"}:
            first_is_option_1 = not swapped
            implied = ("option_1" if (disp == "response_1") == first_is_option_1
                       else "option_2")
            if implied != recorded:
                key_mismatch += 1

        strict_label = str(brow["strict_label"])
        validity = derive_validity(strict_label, recorded)

        strict_both_poor = str(brow["strict_both_poor"])
        strict_reason = str(brow["strict_reason"])
        safety_strict = (
            strict_both_poor == "yes"
            or (validity == "no" and strict_reason == REASON_SAFETY)
        )
        safety_broad = (
            norm_bool(brow["both_poor_any"])
            or (validity == "no" and norm_bool(brow["reason_safety_any"]))
        )

        npass = norm_int(krow.get("verifiers_passing"))
        admitted = npass is not None and npass >= K_ADMIT
        flipped = norm_str(krow.get("injected_flip")) == "yes"

        strict_conf = norm_int(brow["strict_confidence"])
        conf_ok4 = strict_conf is not None and strict_conf >= 4
        conf_ok3 = strict_conf is not None and strict_conf >= 3
        clear = str(brow["strict_ambiguity"]) == "clear"
        agrees_recorded = strict_label in SUBSTANTIVE and strict_label == recorded

        out = dict(brow)
        out.update({
            "example_id": krow["example_id"],
            "stratum": krow["stratum"],
            "injected_flip": krow["injected_flip"],
            "verifiers_passing": krow["verifiers_passing"],
            "display_order": krow.get("display_order", ""),
            "recorded_preference_is_response":
                krow.get("recorded_preference_is_response", ""),
            "user_choice": src["user_choice"],
            "is_clean": str(src.get("is_clean")),
            "recorded_preference_option": recorded,
            "admitted": str(admitted).lower(),
            "filter_outcome": "accepted" if admitted else "rejected",
            "flip_group": "flipped" if flipped else "not_flipped",
            "analysis_group": ("accepted" if admitted else "rejected")
                              + ("_flipped" if flipped else "_not_flipped"),
            "original_preference_valid": validity,
            "safety_preserving_to_reject": "yes" if safety_strict else "no",
            "safety_preserving_to_reject_broad": "yes" if safety_broad else "no",
            "strict_agrees_with_recorded": str(agrees_recorded).lower(),
            "harmful_false_rejection_primary": str(
                (not admitted) and agrees_recorded and conf_ok4 and clear).lower(),
            "harmful_false_rejection_conf3": str(
                (not admitted) and agrees_recorded and conf_ok3 and clear).lower(),
        })
        for vname, _ in VARIANTS:
            lab, conf = variant_labels(brow)[vname]
            out[f"variant_{vname}_label"] = lab
            out[f"variant_{vname}_confidence"] = "" if conf is None else conf
            out[f"variant_{vname}_valid"] = derive_validity(lab, recorded)
        rows.append(out)
    return rows, key_mismatch


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def arr(rows: list[dict], fn) -> np.ndarray:  # noqa: ANN001
    return np.array([bool(fn(r)) for r in rows], dtype=bool)


def scope_definitions(rows: list[dict]) -> list[tuple[str, str, np.ndarray]]:
    """Every reporting scope: the key's strata, the derived groups, and overall.

    The key's `stratum` column carries seven values, three of which
    (`rejected_unanimous`, `rejected_split_vote`, `near_threshold`) are
    diagnostic overlays rather than cells of the admission-by-injection cross.
    The four result blocks the reporting plan asks for are therefore also
    computed as derived groups over all examples, using the admission rule
    (verifiers_passing >= 3) and the key's injected_flip column. Both views are
    reported so no audited example is omitted from the four blocks.
    """
    scopes: list[tuple[str, str, np.ndarray]] = []
    ones = np.ones(len(rows), dtype=bool)
    scopes.append(("overall", "overall", ones))
    for group in sorted({r["analysis_group"] for r in rows}):
        scopes.append(("derived_group", group,
                       arr(rows, lambda r, g=group: r["analysis_group"] == g)))
    for stratum in sorted({r["stratum"] for r in rows}):
        scopes.append(("key_stratum", stratum,
                       arr(rows, lambda r, s=stratum: r["stratum"] == s)))
    return scopes


METRIC_DEFS = {
    "judged_valid_rate":
        "strict label equals the canonical option the recorded preference "
        "points to; denominator is all examples in scope",
    "judged_invalid_rate":
        "strict label equals the other canonical option; denominator is all "
        "examples in scope",
    "abstention_rate":
        "strict label is tie or uncertain; denominator is all examples in scope",
    "ambiguity_or_tie_rate":
        "strict label is tie or uncertain, or the strict collapsed "
        "pair_ambiguity is ambiguous or underdetermined; denominator is all "
        "examples in scope",
    "unresolved_rate":
        "the two families did not share a collapsed label; denominator is all "
        "examples in scope",
    "safety_preserving_rate":
        "derived safety_preserving_to_reject is yes under the strict reading; "
        "denominator is all examples in scope",
    "safety_preserving_rate_broad":
        "derived safety_preserving_to_reject is yes under the permissive "
        "reading (any of the four judgments); denominator is all examples in "
        "scope",
    "harmful_false_rejection_rate_primary":
        "rejected pairs where both families agree under strict aggregation, "
        "both collapsed confidences are at least 4, strict pair_ambiguity is "
        "clear, and the agreed preference matches the recorded preference; "
        "denominator is rejected pairs in scope",
    "harmful_false_rejection_rate_conf3":
        "same as the primary definition with the confidence floor relaxed to "
        "3; denominator is rejected pairs in scope",
}

NARRATIVE_LABELS = {
    ("rejected_not_flipped", "judged_valid_rate"):
        "independently-judged-valid rate",
    ("rejected_not_flipped", "judged_invalid_rate"): "judged-invalid rate",
    ("rejected_not_flipped", "ambiguity_or_tie_rate"): "ambiguity/tie rate",
    ("rejected_not_flipped", "unresolved_rate"): "unresolved rate",
    ("rejected_not_flipped", "harmful_false_rejection_rate_primary"):
        "estimated harmful false-rejection rate",
    ("rejected_flipped", "judged_invalid_rate"):
        "rate at which the audit judges the injected direction invalid",
    ("rejected_flipped", "ambiguity_or_tie_rate"): "ambiguity rate",
    ("rejected_flipped", "unresolved_rate"): "unresolved rate",
    ("accepted_flipped", "judged_invalid_rate"):
        "missed-corruption rate (audit judges the admitted injected direction "
        "invalid)",
    ("accepted_flipped", "judged_valid_rate"):
        "rate at which judges prefer the injected direction",
    ("accepted_flipped", "ambiguity_or_tie_rate"): "ambiguity rate",
    ("accepted_not_flipped", "judged_valid_rate"): "valid-retention rate",
    ("accepted_not_flipped", "judged_invalid_rate"):
        "likely-original-label-error rate",
    ("accepted_not_flipped", "ambiguity_or_tie_rate"): "ambiguity rate",
}


def compute_metric_arrays(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ones = np.ones(len(rows), dtype=bool)
    rejected = arr(rows, lambda r: r["filter_outcome"] == "rejected")
    ambiguous_flags = ("ambiguous", "underdetermined")
    return {
        "judged_valid_rate":
            (arr(rows, lambda r: r["original_preference_valid"] == "yes"), ones),
        "judged_invalid_rate":
            (arr(rows, lambda r: r["original_preference_valid"] == "no"), ones),
        "abstention_rate":
            (arr(rows, lambda r: r["strict_label"] in ABSTAIN), ones),
        "ambiguity_or_tie_rate":
            (arr(rows, lambda r: r["strict_label"] in ABSTAIN
                 or r["strict_ambiguity"] in ambiguous_flags), ones),
        "unresolved_rate":
            (arr(rows, lambda r: r["strict_label"] == UNRESOLVED), ones),
        "safety_preserving_rate":
            (arr(rows, lambda r: r["safety_preserving_to_reject"] == "yes"), ones),
        "safety_preserving_rate_broad":
            (arr(rows, lambda r: r["safety_preserving_to_reject_broad"] == "yes"),
             ones),
        "harmful_false_rejection_rate_primary":
            (arr(rows, lambda r: norm_bool(r["harmful_false_rejection_primary"])),
             rejected),
        "harmful_false_rejection_rate_conf3":
            (arr(rows, lambda r: norm_bool(r["harmful_false_rejection_conf3"])),
             rejected),
    }


def compute_variant_arrays(rows: list[dict], variant: str
                           ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ones = np.ones(len(rows), dtype=bool)
    rejected = arr(rows, lambda r: r["filter_outcome"] == "rejected")
    vcol = f"variant_{variant}_valid"
    lcol = f"variant_{variant}_label"
    ccol = f"variant_{variant}_confidence"

    def hfr(r: dict, floor: int) -> bool:
        conf = norm_int(r[ccol])
        return (r["filter_outcome"] == "rejected"
                and r[lcol] in SUBSTANTIVE
                and r[lcol] == r["recorded_preference_option"]
                and conf is not None and conf >= floor
                and r["strict_ambiguity"] == "clear")

    return {
        "judged_valid_rate": (arr(rows, lambda r: r[vcol] == "yes"), ones),
        "judged_invalid_rate": (arr(rows, lambda r: r[vcol] == "no"), ones),
        "abstention_rate": (arr(rows, lambda r: r[lcol] in ABSTAIN), ones),
        "unresolved_rate": (arr(rows, lambda r: r[lcol] == UNRESOLVED), ones),
        "harmful_false_rejection_rate_primary":
            (arr(rows, lambda r: hfr(r, 4)), rejected),
        "harmful_false_rejection_rate_conf3":
            (arr(rows, lambda r: hfr(r, 3)), rejected),
    }


# ---------------------------------------------------------------------------
# agreement statistics
# ---------------------------------------------------------------------------


def agreement_stats(rows: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    ones = np.ones(len(rows), dtype=bool)

    for judge in JUDGES:
        usable = arr(rows, lambda r, j=judge:
                     r[f"{j}_run1"] != PARSE_ERROR and r[f"{j}_run2"] != PARSE_ERROR)
        same = arr(rows, lambda r, j=judge: r[f"{j}_run1"] == r[f"{j}_run2"])
        stats[f"within_{judge}_consistency"] = bootstrap_rate(same, usable)

        # Position-flip consistency. Restricted to examples where the two runs
        # actually presented the two options in opposite display orders, which
        # is what makes the comparison a position-bias measure rather than a
        # plain repeatability measure.
        def flipped_display(r: dict, j=judge) -> bool:
            d1, d2 = r[f"{j}_run1_displayed_A_is"], r[f"{j}_run2_displayed_A_is"]
            return bool(d1) and bool(d2) and d1 != d2

        def same_display(r: dict, j=judge) -> bool:
            d1, d2 = r[f"{j}_run1_displayed_A_is"], r[f"{j}_run2_displayed_A_is"]
            return bool(d1) and bool(d2) and d1 == d2

        def substantive_both(r: dict, j=judge) -> bool:
            return (r[f"{j}_run1"] in SUBSTANTIVE
                    and r[f"{j}_run2"] in SUBSTANTIVE)

        den_flip = arr(rows, lambda r, j=judge:
                       flipped_display(r, j) and substantive_both(r, j))
        num_flip = arr(rows, lambda r, j=judge: r[f"{j}_run1"] == r[f"{j}_run2"])
        stats[f"position_flip_consistency_{judge}"] = bootstrap_rate(
            num_flip, den_flip)
        stats[f"displayed_letter_tracking_{judge}"] = bootstrap_rate(
            ~num_flip, den_flip)
        den_same = arr(rows, lambda r, j=judge:
                       same_display(r, j) and substantive_both(r, j))
        stats[f"same_display_repeatability_{judge}"] = bootstrap_rate(
            num_flip, den_same)

        for run in RUNS:
            def chose_displayed_a(r: dict, j=judge, k=run) -> bool:
                lab = r[f"{j}_run{k}"]
                disp = r[f"{j}_run{k}_displayed_A_is"]
                return lab in SUBSTANTIVE and bool(disp) and lab == disp
            den = arr(rows, lambda r, j=judge, k=run: r[f"{j}_run{k}"] in SUBSTANTIVE
                      and bool(r[f"{j}_run{k}_displayed_A_is"]))
            stats[f"chose_displayed_A_{judge}_run{run}"] = bootstrap_rate(
                arr(rows, chose_displayed_a), den)

        stats[f"parse_ok_rate_{judge}"] = bootstrap_rate(
            arr(rows, lambda r, j=judge:
                norm_bool(r[f"{j}_run1_parse_ok"]) and norm_bool(r[f"{j}_run2_parse_ok"])),
            ones)

    both_usable = arr(rows, lambda r: r["llama_collapsed"] in OPTION_VALUES
                      and r["phi_collapsed"] in OPTION_VALUES)
    agree = arr(rows, lambda r: r["llama_collapsed"] == r["phi_collapsed"])
    stats["cross_family_raw_agreement"] = bootstrap_rate(agree, both_usable)

    conf4 = arr(rows, lambda r: (norm_int(r["llama_confidence_collapsed"]) or 0) >= 4
                and (norm_int(r["phi_confidence_collapsed"]) or 0) >= 4)
    stats["cross_family_agreement_conf_ge4"] = bootstrap_rate(
        agree, both_usable & conf4)

    stats["abstention_rate_strict"] = bootstrap_rate(
        arr(rows, lambda r: r["strict_label"] in ABSTAIN), ones)
    stats["abstention_rate_any_judgment"] = bootstrap_rate(
        arr(rows, lambda r: any(r[f"{j}_run{k}"] in ABSTAIN
                                for j in JUDGES for k in RUNS)), ones)
    stats["unresolved_rate"] = bootstrap_rate(
        arr(rows, lambda r: r["strict_label"] == UNRESOLVED), ones)
    stats["inconsistent_rate_either_family"] = bootstrap_rate(
        arr(rows, lambda r: r["llama_collapsed"] == INCONSISTENT
            or r["phi_collapsed"] == INCONSISTENT), ones)

    # Cohen's kappa on non-abstained categorical labels only.
    kappa_mask = [r for r in rows
                  if r["llama_collapsed"] in SUBSTANTIVE
                  and r["phi_collapsed"] in SUBSTANTIVE]
    kappa = cohens_kappa([r["llama_collapsed"] for r in kappa_mask],
                         [r["phi_collapsed"] for r in kappa_mask])
    kappa_ci = bootstrap_kappa(rows)
    stats["cohens_kappa"] = {
        "n_num": len(kappa_mask),
        "n_den": len(rows),
        "rate": kappa,
        "ci_low": kappa_ci[0],
        "ci_high": kappa_ci[1],
        "n_boot_valid": kappa_ci[2],
    }
    return stats


def bootstrap_kappa(rows: list[dict], n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> tuple[float, float, int]:
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan"), 0
    la = [r["llama_collapsed"] for r in rows]
    ph = [r["phi_collapsed"] for r in rows]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = []
    for draw in idx:
        a, b = [], []
        for i in draw:
            if la[i] in SUBSTANTIVE and ph[i] in SUBSTANTIVE:
                a.append(la[i])
                b.append(ph[i])
        if len(a) >= 2:
            k = cohens_kappa(a, b)
            if not math.isnan(k):
                vals.append(k)
    if not vals:
        return float("nan"), float("nan"), 0
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            len(vals))


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------


def stage2(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    freeze = verify_freeze(outdir)
    verify_raw_against_freeze(freeze, raw_dir)
    print("stage 2 hash guard passed")
    print(f"  frozen aggregate sha256 : {freeze['aggregated_blinded']['sha256']}")
    print(f"  frozen at (mtime)       : {freeze['frozen_at_utc']}")

    agg_path = outdir / "model_audit_aggregated_blinded.csv"
    with open(agg_path) as f:
        blinded = list(csv.DictReader(f))

    key_path = Path(args.key).resolve()
    corrupted_path = Path(args.corrupted).resolve()
    key = load_key(key_path)
    wanted = {r["example_id"] for r in key}
    corrupted = load_corrupted(corrupted_path, wanted)
    missing = wanted - set(corrupted)
    if missing:
        raise SystemExit(
            f"{len(missing)} audited example_ids are absent from "
            f"{corrupted_path} (first: {sorted(missing)[:5]})"
        )

    rows, key_mismatch = build_join(blinded, key, corrupted)
    if key_mismatch:
        print(
            f"WARNING: {key_mismatch} of {len(rows)} rows disagree between the "
            "key's recorded_preference_is_response (mapped back through "
            "display_order) and the corrupted file's user_choice. The corrupted "
            "file is treated as authoritative for the canonical option, as "
            "specified. This count is reported in the agreement file.",
            file=sys.stderr,
        )

    join_cols = list(rows[0].keys())
    join_path = outdir / "model_audit_hidden_join.csv"
    write_csv(join_path, join_cols, rows)

    stats = agreement_stats(rows)
    result_rows: list[dict] = []
    for name, stat in stats.items():
        result_rows.append({
            "variant": "aggregation_diagnostic",
            "scope_type": "overall",
            "scope": "overall",
            "metric": name,
            "narrative_label": "",
            "n_numerator": stat["n_num"],
            "n_denominator": stat["n_den"],
            "value": stat["rate"],
            "ci_low": stat["ci_low"],
            "ci_high": stat["ci_high"],
            "n_boot_valid": stat["n_boot_valid"],
            "definition": AGREEMENT_DEFS.get(name, ""),
        })

    scopes = scope_definitions(rows)
    primary_arrays = compute_metric_arrays(rows)
    # Keyed by (scope_type, scope): the derived groups deliberately reuse the
    # names of four of the key's strata, and they are different populations.
    per_scope: dict[tuple[str, str], dict[str, dict]] = {}
    for scope_type, scope, mask in scopes:
        sub = {}
        for metric, (num, den) in primary_arrays.items():
            stat = bootstrap_rate(num[mask], den[mask])
            sub[metric] = stat
            result_rows.append({
                "variant": "strict_cross_family",
                "scope_type": scope_type,
                "scope": scope,
                "metric": metric,
                "narrative_label": NARRATIVE_LABELS.get((scope, metric), ""),
                "n_numerator": stat["n_num"],
                "n_denominator": stat["n_den"],
                "value": stat["rate"],
                "ci_low": stat["ci_low"],
                "ci_high": stat["ci_high"],
                "n_boot_valid": stat["n_boot_valid"],
                "definition": METRIC_DEFS[metric],
            })
        per_scope[(scope_type, scope)] = sub

    variant_scope: dict[str, dict[tuple[str, str], dict[str, dict]]] = {}
    for vname, vlabel in VARIANTS:
        varrays = compute_variant_arrays(rows, vname)
        variant_scope[vname] = {}
        for scope_type, scope, mask in scopes:
            sub = {}
            for metric, (num, den) in varrays.items():
                stat = bootstrap_rate(num[mask], den[mask])
                sub[metric] = stat
                result_rows.append({
                    "variant": vname,
                    "scope_type": scope_type,
                    "scope": scope,
                    "metric": metric,
                    "narrative_label": NARRATIVE_LABELS.get((scope, metric), ""),
                    "n_numerator": stat["n_num"],
                    "n_denominator": stat["n_den"],
                    "value": stat["rate"],
                    "ci_low": stat["ci_low"],
                    "ci_high": stat["ci_high"],
                    "n_boot_valid": stat["n_boot_valid"],
                    "definition": (f"{vlabel}; " + METRIC_DEFS.get(metric, "")),
                })
            variant_scope[vname][(scope_type, scope)] = sub

    results_csv = outdir / "model_audit_results.csv"
    write_csv(results_csv, [
        "variant", "scope_type", "scope", "metric", "narrative_label",
        "n_numerator", "n_denominator", "value", "ci_low", "ci_high",
        "n_boot_valid", "definition",
    ], result_rows)

    write_agreement_md(outdir / "model_audit_agreement.md", rows, stats,
                       freeze, key_mismatch, args)
    write_results_md(outdir / "model_audit_results.md", rows, scopes, per_scope,
                     variant_scope, freeze, args)
    n_spot = write_spotcheck_md(outdir / "model_audit_spotcheck.md", rows,
                                raw_dir, corrupted)

    print(f"  wrote {rel_to_root(join_path)} ({len(rows)} rows)")
    print(f"  wrote {rel_to_root(results_csv)} ({len(result_rows)} metric rows)")
    print(f"  wrote {rel_to_root(outdir / 'model_audit_agreement.md')}")
    print(f"  wrote {rel_to_root(outdir / 'model_audit_results.md')}")
    print(f"  wrote {rel_to_root(outdir / 'model_audit_spotcheck.md')} "
          f"({n_spot} items)")
    return 0


AGREEMENT_DEFS = {
    "within_llama_consistency":
        "Llama run 1 and run 2 give the same collapsed-level answer; "
        "denominator is examples where both runs parsed",
    "within_phi_consistency":
        "Phi run 1 and run 2 give the same answer; denominator is examples "
        "where both runs parsed",
    "cross_family_raw_agreement":
        "Llama and Phi collapsed labels are equal; denominator is examples "
        "where both families produced a usable collapsed label",
    "cross_family_agreement_conf_ge4":
        "same, restricted to examples where both collapsed confidences are at "
        "least 4",
    "cohens_kappa":
        "Cohen's kappa on the two collapsed family labels, computed only on "
        "examples where both families gave a substantive option label; tie, "
        "uncertain, inconsistent and parse failures are excluded",
    "abstention_rate_strict":
        "strict label is tie or uncertain; denominator is all examples",
    "abstention_rate_any_judgment":
        "at least one of the four judgments was tie or uncertain; denominator "
        "is all examples",
    "unresolved_rate":
        "the two families did not share a collapsed label; denominator is all "
        "examples",
    "inconsistent_rate_either_family":
        "at least one family's two runs disagreed; denominator is all examples",
    "position_flip_consistency_llama":
        "Llama gives the same canonical option across its two runs, restricted "
        "to examples whose two runs used opposite display orders and where both "
        "runs gave a substantive option; this is the position-bias measure and "
        "it is computed from displayed_A_is",
    "position_flip_consistency_phi":
        "as above for Phi",
    "displayed_letter_tracking_llama":
        "complement of the above: Llama's canonical answer changed when the "
        "display order changed, which is the signature of tracking the "
        "displayed letter rather than the content",
    "displayed_letter_tracking_phi": "as above for Phi",
    "same_display_repeatability_llama":
        "Llama repeats its answer when the two runs happened to use the same "
        "display order; this is repeatability, not position bias",
    "same_display_repeatability_phi": "as above for Phi",
    "chose_displayed_A_llama_run1":
        "fraction of substantive answers that selected the response displayed "
        "in position A; under the randomisation this should sit near 0.5",
    "chose_displayed_A_llama_run2": "as above",
    "chose_displayed_A_phi_run1": "as above",
    "chose_displayed_A_phi_run2": "as above",
    "parse_ok_rate_llama":
        "both Llama runs produced a parseable judgement; denominator is all "
        "examples",
    "parse_ok_rate_phi": "as above for Phi",
}


def md_header(title: str, freeze: dict, args: argparse.Namespace) -> list[str]:
    return [
        f"# {title}",
        "",
        MODEL_AUDIT_DISCLAIMER,
        "",
        "## Provenance",
        "",
        f"- Blinded aggregate: `{freeze['aggregated_blinded']['path']}`, "
        f"sha256 `{freeze['aggregated_blinded']['sha256']}`.",
        f"- Frozen at {freeze['frozen_at_utc']}, derived from the filesystem "
        "mtimes of the raw judge outputs rather than the wall clock.",
    ] + [
        f"- Raw judge file for {judge}: "
        f"`{freeze['raw_inputs'][judge]['path']}`, sha256 "
        f"`{freeze['raw_inputs'][judge]['sha256']}`, "
        f"{freeze['raw_inputs'][judge]['n_records']} records."
        for judge in JUDGES
    ] + [
        f"- Hidden key: `{rel_to_root(Path(args.key))}`. Corrupted training "
        f"file: `{rel_to_root(Path(args.corrupted))}`. Both were opened only "
        "after the hash guard in stage 2 confirmed the blinded labels were "
        "unchanged.",
        f"- Bootstrap intervals: {N_BOOT} resamples over examples, seed "
        f"{BOOT_SEED}, percentile method.",
        "",
    ]


def write_agreement_md(path: Path, rows: list[dict], stats: dict,
                       freeze: dict, key_mismatch: int,
                       args: argparse.Namespace) -> None:
    lines = md_header("Independent model audit: agreement report", freeze, args)
    lines += [
        "## 1. What is being measured",
        "",
        "Two judge families were each run twice over the same sample with an "
        "independently randomised display order per pass. The two passes of one "
        "family measure within-model stability under a position change and are "
        "not treated as independent annotators. The headline agreement quantity "
        "is therefore the cross-family agreement on the collapsed labels, not "
        "the agreement among the four raw judgements.",
        "",
        "Aggregation rules, as executed:",
        "",
        "1. Within a family, the two runs collapse to their shared value. Two "
        "different substantive values collapse to `inconsistent`. `tie` and "
        "`uncertain` are substantive values throughout and are never treated as "
        "parse failures.",
        f"2. A run that failed to parse is missing data, and the policy in "
        f"force for this run, as recorded in the freeze artifact, is "
        f"`{freeze.get('parse_failure_policy', 'fallback')}`. Under "
        "`fallback` a family whose other run parsed collapses to the surviving "
        "run; under `inconsistent` the parse failure counts as a disagreement. "
        "This is a judgement the specification did not fix, and the choice is "
        "recorded rather than assumed.",
        "3. Across families, the final label is the shared collapsed value when "
        "the two families agree and `unresolved` otherwise. There is no forced "
        "adjudication and no tiebreaker. `inconsistent` and `parse_error` can "
        "never become a final label, because a shared absence of a label is not "
        "a shared label.",
        "4. Confidence collapses by minimum: within a family across its two "
        "runs, and then across the two families for the strict label. The "
        "minimum is used rather than the mean because the audit functions as a "
        "conservative screen, so a single low-confidence pass must not be "
        "averaged away.",
        "",
        "## 2. Agreement and stability",
        "",
        "| Quantity | Estimate | 95% bootstrap interval | n |",
        "|---|---|---|---|",
    ]
    order = [
        "within_llama_consistency", "within_phi_consistency",
        "cross_family_raw_agreement", "cross_family_agreement_conf_ge4",
        "abstention_rate_strict", "abstention_rate_any_judgment",
        "unresolved_rate", "inconsistent_rate_either_family",
        "parse_ok_rate_llama", "parse_ok_rate_phi",
    ]
    for name in order:
        s = stats[name]
        ci = ("n/a" if s["n_den"] == 0
              else f"[{fmt_rate(s['ci_low'])}, {fmt_rate(s['ci_high'])}]")
        lines.append(
            f"| {name} | {fmt_rate(s['rate'])} | {ci} | "
            f"{s['n_num']}/{s['n_den']} |")
    k = stats["cohens_kappa"]
    lines += [
        "",
        "## 3. Cohen's kappa",
        "",
        "Kappa is computed on the two collapsed family labels over the "
        "non-abstained categorical labels only. Examples where either family "
        "collapsed to `tie`, `uncertain`, `inconsistent`, or a parse failure "
        "are excluded, because kappa over a category set that mixes a "
        "substantive choice with a refusal to choose is not interpretable as "
        "chance-corrected agreement on the direction.",
        "",
        f"- Excluded categories: `tie`, `uncertain`, `inconsistent`, "
        f"`parse_error`.",
        f"- n after exclusion: {k['n_num']} of {k['n_den']} audited examples.",
        f"- Cohen's kappa: {fmt_rate(k['rate'])}, 95% bootstrap interval "
        f"[{fmt_rate(k['ci_low'])}, {fmt_rate(k['ci_high'])}].",
        "",
        "## 4. A/B position-flip consistency",
        "",
        "This is the position-bias measure. It is computed from the "
        "`displayed_A_is` field recorded with each judgement, never assumed "
        "from the randomisation seed. For each judge it asks how often the "
        "judge's answer tracks the canonical option rather than the displayed "
        "letter, restricted to the examples whose two runs happened to present "
        "the options in opposite display orders. Examples whose two runs used "
        "the same display order carry no information about position bias and "
        "are reported separately as plain repeatability.",
        "",
        "| Quantity | Estimate | 95% bootstrap interval | n |",
        "|---|---|---|---|",
    ]
    pos_order = []
    for judge in JUDGES:
        pos_order += [f"position_flip_consistency_{judge}",
                      f"displayed_letter_tracking_{judge}",
                      f"same_display_repeatability_{judge}"]
        pos_order += [f"chose_displayed_A_{judge}_run{r}" for r in RUNS]
    for name in pos_order:
        s = stats[name]
        ci = ("n/a" if s["n_den"] == 0
              else f"[{fmt_rate(s['ci_low'])}, {fmt_rate(s['ci_high'])}]")
        lines.append(
            f"| {name} | {fmt_rate(s['rate'])} | {ci} | "
            f"{s['n_num']}/{s['n_den']} |")
    lines += [
        "",
        "A `position_flip_consistency` near 1 means the judge's answer is "
        "driven by the content of the two responses. A value near 0 means the "
        "answer is driven by the displayed position, which would make the "
        "corresponding audit labels unusable. The `chose_displayed_A` rows are "
        "the marginal check described in Section 5 of the prompt "
        "specification: under the randomisation they should sit near 0.5.",
        "",
        "## 5. Join integrity",
        "",
        f"- Rows joined: {len(rows)}.",
        f"- Rows where the key's `recorded_preference_is_response`, mapped back "
        f"through `display_order`, disagrees with `user_choice` in the "
        f"corrupted file: {key_mismatch}. The corrupted file is authoritative "
        "for the canonical option, since it is the file the blinded sample was "
        "built from.",
        "",
        "## 6. Definitions",
        "",
        "| Quantity | Definition |",
        "|---|---|",
    ]
    for name in order + ["cohens_kappa"] + pos_order:
        lines.append(f"| {name} | {AGREEMENT_DEFS.get(name, '')} |")
    lines.append("")
    path.write_text("\n".join(lines))


def _row(scope_stats: dict, metric: str, label: str) -> str:
    s = scope_stats.get(metric)
    if s is None:
        return f"| {label} | n/a | n/a | n/a |"
    ci = ("n/a" if s["n_den"] == 0
          else f"[{fmt_rate(s['ci_low'])}, {fmt_rate(s['ci_high'])}]")
    return (f"| {label} | {s['n_num']}/{s['n_den']} | {fmt_rate(s['rate'])} | "
            f"{ci} |")


def write_results_md(path: Path, rows: list[dict],
                     scopes: list[tuple[str, str, np.ndarray]],
                     per_scope: dict, variant_scope: dict,
                     freeze: dict, args: argparse.Namespace) -> None:
    lines = md_header("Independent model audit: results", freeze, args)
    counts = Counter(r["stratum"] for r in rows)
    group_counts = Counter(r["analysis_group"] for r in rows)
    lines += [
        "## 1. Populations",
        "",
        "Results are reported over two partitions of the same audited sample. "
        "The first is the `stratum` column of the hidden key, which carries "
        "seven values. Three of them, `rejected_unanimous`, "
        "`rejected_split_vote` and `near_threshold`, are diagnostic overlays "
        "rather than cells of the admission-by-injection cross, so the four "
        "reporting blocks requested by the protocol would omit the examples "
        "that fell into an overlay. The second partition therefore derives the "
        "four blocks over every audited example, using the admission rule "
        f"(`verifiers_passing >= {K_ADMIT}`) and the key's `injected_flip` "
        "column. Both partitions are reported and neither is a substitute for "
        "the other.",
        "",
        "| Key stratum | n |",
        "|---|---|",
    ] + [f"| {s} | {counts[s]} |" for s in sorted(counts)] + [
        f"| total | {sum(counts.values())} |",
        "",
        "| Derived group | n |",
        "|---|---|",
    ] + [f"| {g} | {group_counts[g]} |" for g in sorted(group_counts)] + [
        f"| total | {sum(group_counts.values())} |",
        "",
        "## 2. Derived fields",
        "",
        "Both fields below are derived at aggregation time and were never put "
        "to the judges, for the reasons given in Section 3 of "
        "`analysis/model_audit_prompt.md`.",
        "",
        "- `original_preference_valid` is `yes` when the strict label equals "
        "the canonical option that the recorded preference points to, `no` "
        "when it equals the other option, `uncertain` when the strict label is "
        "`tie` or `uncertain`, and `unresolved` when the two families did not "
        "agree. The canonical mapping is `user_choice == \"A\"` to `option_1` "
        "and `user_choice == \"B\"` to `option_2`.",
        "- `safety_preserving_to_reject` is `yes` when the strict collapsed "
        "`both_poor_or_unsafe` is `yes`, or when `original_preference_valid` is "
        "`no` and the strict collapsed `reason_category` is "
        f"`{REASON_SAFETY}`. A permissive variant, in which any one of the four "
        "judgements suffices, is reported alongside it as "
        "`safety_preserving_rate_broad`.",
        "",
        "## 3. The four reporting blocks",
        "",
    ]

    blocks = [
        ("rejected_not_flipped", "Rejected, not flipped",
         "The pairs the filter discarded with no synthetic justification. This "
         "is the population the objection concerns.",
         ["judged_valid_rate", "judged_invalid_rate", "ambiguity_or_tie_rate",
          "unresolved_rate", "harmful_false_rejection_rate_primary",
          "harmful_false_rejection_rate_conf3"]),
        ("rejected_flipped", "Rejected, flipped",
         "The pairs the filter discarded that did carry an injected reversal.",
         ["judged_invalid_rate", "ambiguity_or_tie_rate", "unresolved_rate",
          "judged_valid_rate"]),
        ("accepted_flipped", "Accepted, flipped",
         "Injected reversals that the filter admitted.",
         ["judged_invalid_rate", "judged_valid_rate", "ambiguity_or_tie_rate",
          "unresolved_rate"]),
        ("accepted_not_flipped", "Accepted, not flipped",
         "Pairs the filter admitted with no injected reversal.",
         ["judged_valid_rate", "judged_invalid_rate", "ambiguity_or_tie_rate",
          "unresolved_rate"]),
    ]
    default_labels = {
        "judged_valid_rate": "independently-judged-valid rate",
        "judged_invalid_rate": "judged-invalid rate",
        "ambiguity_or_tie_rate": "ambiguity/tie rate",
        "unresolved_rate": "unresolved rate",
        "harmful_false_rejection_rate_primary":
            "estimated harmful false-rejection rate (primary, confidence >= 4)",
        "harmful_false_rejection_rate_conf3":
            "harmful false-rejection rate (sensitivity, confidence >= 3)",
    }
    for n_block, (scope, title, blurb, metrics) in enumerate(blocks, start=1):
        if ("derived_group", scope) not in per_scope:
            continue
        lines += [f"### 3.{n_block} {title}", "", blurb, "",
                  f"Derived group `{scope}`, n = {group_counts.get(scope, 0)}. "
                  "The corresponding key stratum of the same name, which "
                  "excludes examples claimed by a diagnostic overlay, is in the "
                  "full table in Section 4.",
                  "", "| Quantity | Count | Proportion | 95% bootstrap |",
                  "|---|---|---|---|"]
        for metric in metrics:
            label = NARRATIVE_LABELS.get((scope, metric),
                                         default_labels.get(metric, metric))
            lines.append(_row(per_scope[("derived_group", scope)], metric, label))
        lines.append("")

    lines += [
        "## 4. Every metric, per key stratum and per derived group",
        "",
        "| Scope type | Scope | Metric | Count | Proportion | 95% bootstrap |",
        "|---|---|---|---|---|---|",
    ]
    for scope_type, scope, _ in scopes:
        for metric in METRIC_DEFS:
            s = per_scope[(scope_type, scope)][metric]
            ci = ("n/a" if s["n_den"] == 0
                  else f"[{fmt_rate(s['ci_low'])}, {fmt_rate(s['ci_high'])}]")
            lines.append(
                f"| {scope_type} | {scope} | {metric} | "
                f"{s['n_num']}/{s['n_den']} | {fmt_rate(s['rate'])} | {ci} |")

    lines += [
        "",
        "## 5. Harmful false rejection",
        "",
        "A rejected pair counts as a harmful false rejection under the primary "
        "definition only when all four conditions hold: the two families agree "
        "under strict cross-family aggregation, both collapsed confidences are "
        "at least 4, the strict collapsed `pair_ambiguity` is `clear`, and the "
        "agreed preference matches the recorded preference. The denominator is "
        "the rejected pairs in the scope. A broader sensitivity relaxes the "
        "confidence floor to 3 and changes nothing else.",
        "",
        "| Scope type | Scope | Primary (conf >= 4) | Sensitivity (conf >= 3) |",
        "|---|---|---|---|",
    ]
    for scope_type, scope, _ in scopes:
        p = per_scope[(scope_type, scope)]["harmful_false_rejection_rate_primary"]
        b = per_scope[(scope_type, scope)]["harmful_false_rejection_rate_conf3"]
        lines.append(f"| {scope_type} | {scope} | {fmt_ci(p)} | {fmt_ci(b)} |")

    lines += [
        "",
        "## 6. Sensitivity to the aggregation rule",
        "",
        "Four aggregation rules are reported. The strict cross-family rule is "
        "the primary specification and the other three are sensitivity "
        "analyses. They are not alternatives to be selected after the fact.",
        "",
        "1. **Strict cross-family (primary).** The final label is the shared "
        "value of the two collapsed family labels, and `unresolved` otherwise.",
        "2. **Majority across all four judgments.** The modal label of the four "
        "raw judgements, with a tie in the mode reported as `unresolved`. This "
        "rule treats the two passes of one family as if they were independent "
        "annotators, which the prompt specification explicitly says they are "
        "not, so it will overstate agreement.",
        "3. **Confidence-weighted majority.** Each judgement contributes its "
        "stated confidence to its label, and the label with the greatest total "
        "wins. A tie in the total is `unresolved`.",
        "4. **High-confidence only.** The modal label among judgements with a "
        "stated confidence of at least 4, with no such judgement reported as "
        "`unresolved`.",
        "",
        "| Variant | Scope | valid | invalid | abstain | unresolved |",
        "|---|---|---|---|---|---|",
    ]
    for vname, vlabel in VARIANTS:
        for scope_type, scope, _ in scopes:
            if scope_type == "key_stratum":
                continue
            sub = variant_scope[vname][(scope_type, scope)]
            cells = [fmt_rate(sub[m]["rate"]) for m in
                     ("judged_valid_rate", "judged_invalid_rate",
                      "abstention_rate", "unresolved_rate")]
            lines.append(f"| {vlabel} | {scope} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Per-stratum values for every variant, with counts and intervals, are "
        "in `model_audit_results.csv` under the `variant` column.",
        "",
        "## 7. What these numbers do not establish",
        "",
        "These are the judgements of two instruction-tuned language models. "
        "They are independent of the verifier pool by construction of the model "
        "families, and that independence is what makes them informative about "
        "circularity, but independence from the pool is not correctness. Where "
        "the audit judges a rejected pair to be valid supervision, the correct "
        "reading is that a judge outside the pool disagreed with the pool, not "
        "that the pair has been verified as valid. Human annotation of the "
        "`rejected_not_flipped` stratum remains the measurement this audit "
        "substitutes for.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_spotcheck_md(path: Path, rows: list[dict], raw_dir: Path,
                       corrupted: dict[str, dict]) -> int:
    raw = {}
    for judge in JUDGES:
        recs = load_raw(raw_dir / f"model_audit_{judge}_raw.jsonl")
        raw[judge] = index_raw(recs, judge, raw_dir / f"model_audit_{judge}_raw.jsonl")

    by_id = {r["audit_id"]: r for r in rows}
    rng = random.Random(SPOTCHECK_SEED)

    def pick(pred, n=SPOTCHECK_N):
        pool = sorted(r["audit_id"] for r in rows if pred(r))
        if len(pool) <= n:
            return pool, len(pool)
        return sorted(rng.sample(pool, n)), len(pool)

    categories = [
        ("Cross-family agreements",
         lambda r: r["strict_label"] in SUBSTANTIVE),
        ("Cross-family disagreements",
         lambda r: r["strict_label"] == UNRESOLVED
         and r["llama_collapsed"] in SUBSTANTIVE
         and r["phi_collapsed"] in SUBSTANTIVE),
        ("Tie or uncertain cases",
         lambda r: r["strict_label"] in ABSTAIN
         or any(r[f"{j}_run{k}"] in ABSTAIN for j in JUDGES for k in RUNS)),
        ("High-confidence likely false rejections",
         lambda r: norm_bool(r["harmful_false_rejection_primary"])),
    ]

    lines = [
        "# Independent model audit: spot-check sheet",
        "",
        MODEL_AUDIT_DISCLAIMER,
        "",
        "## Purpose",
        "",
        "This sheet is a quality-control inspection of the audit instrument. "
        "It exists to detect prompt failures and parser failures: rationales "
        "that do not respond to the item, malformed or truncated outputs, "
        "answers that track the displayed position rather than the content, "
        "and reasoning that is irrelevant to the pair. It is not an "
        "independent human audit of the preference labels, it does not produce "
        "ground truth, and marks made on it must not be substituted for the "
        "model labels or used to overturn them. Any item found defective here "
        "is evidence about the instrument, and the correct response is to "
        "report the defect rate, not to relabel the item.",
        "",
        f"Selection is by fixed seed {SPOTCHECK_SEED}. Up to {SPOTCHECK_N} "
        "items are drawn per category. Prompts and responses are truncated to "
        f"{SPOTCHECK_TRUNC} characters.",
        "",
    ]

    total = 0
    for title, pred in categories:
        chosen, pool_n = pick(pred)
        lines += [f"## {title}", "",
                  f"Drawn {len(chosen)} of {pool_n} eligible items.", ""]
        if not chosen:
            lines += ["No eligible items.", ""]
        for aid in chosen:
            total += 1
            r = by_id[aid]
            src = corrupted[r["example_id"]]
            lines += [
                f"### {aid}",
                "",
                f"- Strict label: `{r['strict_label']}`. Llama collapsed: "
                f"`{r['llama_collapsed']}`. Phi collapsed: "
                f"`{r['phi_collapsed']}`. Strict confidence: "
                f"`{r['strict_confidence']}`. Strict ambiguity: "
                f"`{r['strict_ambiguity']}`.",
                "",
                f"**Prompt.** {truncate(src['prompt'])}",
                "",
                f"**Option 1 (response_a).** {truncate(src['response_a'])}",
                "",
                f"**Option 2 (response_b).** {truncate(src['response_b'])}",
                "",
                "| Judge | Run | Displayed A is | Preferred | Ambiguity | "
                "Both acceptable | Both poor | Conf | Reason | Rationale |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for judge in JUDGES:
                for run in RUNS:
                    rec = raw[judge][aid][run]
                    pref = run_value(rec, "preferred_option")
                    lines.append(
                        f"| {judge} | {run} | {rec['displayed_A_is'] or 'n/a'} "
                        f"| {pref} | {rec['pair_ambiguity'] or 'n/a'} | "
                        f"{rec['both_acceptable'] or 'n/a'} | "
                        f"{rec['both_poor_or_unsafe'] or 'n/a'} | "
                        f"{rec['confidence'] if rec['confidence'] is not None else 'n/a'} "
                        f"| {rec['reason_category'] or 'n/a'} | "
                        f"{truncate(rec['brief_rationale'], 160) or 'n/a'} |")
            lines += [
                "",
                "Human inspection, one set of marks per item:",
                "",
                "- [ ] coherent",
                "- [ ] responsive",
                "- [ ] malformed",
                "- [ ] position-biased",
                "- [ ] irrelevant-reasoning",
                "",
            ]
    path.write_text("\n".join(lines))
    return total


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--raw-dir", default=str(ROOT / "analysis"),
                        help="directory holding model_audit_{judge}_raw.jsonl")
        sp.add_argument("--outdir", default=str(ROOT / "analysis"),
                        help="directory for the emitted artifacts")

    s1 = sub.add_parser("stage1", help="blinded aggregation and freeze")
    common(s1)
    s1.add_argument("--parse-failure-policy", choices=("fallback", "inconsistent"),
                    default="fallback",
                    help="how to collapse a family when exactly one run failed "
                         "to parse: use the surviving run (fallback) or treat "
                         "the failure as a disagreement (inconsistent)")
    s1.set_defaults(func=stage1)

    s2 = sub.add_parser("stage2", help="unblind, join the key, report")
    common(s2)
    s2.add_argument("--key", default=str(ROOT / "analysis/rejection_audit_key.csv"))
    s2.add_argument(
        "--corrupted",
        default=str(ROOT / "data/corrupted/hh_train_structured_unsafe_eta20.jsonl"))
    s2.set_defaults(func=stage2)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
