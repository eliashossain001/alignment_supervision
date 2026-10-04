#!/usr/bin/env bash
# Training-seed variance study at eta = 20% (structured_unsafe).
#
# Purpose: quantify how much of the between-method spread in held-out preference
# accuracy is attributable to the training seed alone. Five configurations are
# each trained under three training seeds (42, 43, 44), giving fifteen runs. The
# study is NOT intended to show that any configuration is better than another.
#
# Every hyperparameter below is fixed to the published pilot value so that the
# seed-42 runs are directly comparable to the numbers already in
# analysis/full_baseline_grid.csv. Sources:
#   src/train_dpo.py           lines 44 to 58 (argument-parser defaults)
#   scripts/run_pilot_hh.sh    lines 15 to 23 (pilot environment defaults)
#   discussion/run_configuration.md sections 3 and 4
#
# Evaluation reports preference accuracy and mean margin only. The refusal
# metrics of src/evaluate_model.py cannot be recomputed because
# data/raw/safety_refusal_eval.jsonl was lost; see the module docstring of
# analysis/scripts/eval_preference_only.py.
#
# Scheduling: a single shared work queue is consumed by one serial worker per
# GPU. A worker claims an item by creating its claim directory, which mkdir
# performs atomically, so no item is ever taken twice. Exactly two runs are in
# flight at any moment, one per GPU, and never more.
#
# Idempotence: a run is skipped only when its completion marker
# results/eval/seed_study/<tag>.done exists. The marker is written after the
# adapter files and the evaluation JSON have both been verified, so a run that
# was interrupted part way through evaluation is re-run rather than silently
# treated as finished. The script may therefore be re-run after any
# interruption.
#
# Retention: no path in this script deletes, prunes or overwrites a trained
# adapter. Checkpoints are retained indefinitely, and deliberately so. The
# safety refusal evaluation of src/evaluate_model.py cannot be run today because
# data/raw/safety_refusal_eval.jsonl is lost; if that file is recovered, the
# refusal metrics must be computable against exactly the adapters that produced
# the preference numbers reported here, which is only possible if those adapters
# still exist. Any future cleanup of results/checkpoints/seed_study/ destroys
# that possibility and must not be added.
#
# Provenance: every run writes results/eval/seed_study/<tag>.meta.json recording
# the exact command lines, the hyperparameters actually passed, the training
# file with its sha256 and line count, the environment and library versions, the
# wall time, and a digest over the adapter weight files. The training log is
# preserved inside the checkpoint directory as train.log.
#
# Usage:
#   bash analysis/scripts/run_seed_study.sh
#
# Environment overrides (all optional):
#   SEEDS, CONFIGS, GPUS, ETA_TAG, CORR, BASE_MODEL, EVAL_MAX_SAMPLES,
#   MIN_GPU_FREE_MIB, MIN_DISK_GB, HF_CACHE_DIR, EST_PER_RUN_MB,
#   FORCE (set to 1 to bypass the concurrent-job guard, which is unsafe while
#   other work occupies the GPUs).
# The threshold overrides exist so that the driver can be exercised against a
# stub harness without a GPU. Lowering them for a real launch removes the
# protection they provide and should not be done.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- parameters
ETA_TAG="${ETA_TAG:-20}"
CORR="${CORR:-structured_unsafe}"
FILT_DIR="${FILT_DIR:-$ROOT/data/filtered/eta${ETA_TAG}}"

SEEDS=(${SEEDS:-42 43 44})
CONFIGS=(${CONFIGS:-raw cdpo single consensus_k3 oracle})
GPUS=(${GPUS:-0 1})

# Published hyperparameters. Do not change these without also restating that the
# runs are no longer comparable to the published seed-42 values.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"  # train_dpo.py line 44
EPOCHS="${EPOCHS:-1.0}"                                  # train_dpo.py line 45
BATCH_SIZE="${BATCH_SIZE:-2}"                            # train_dpo.py line 46
GRAD_ACCUM="${GRAD_ACCUM:-8}"                            # train_dpo.py line 47
LR="${LR:-5e-6}"                                         # train_dpo.py line 48
BETA="${BETA:-0.1}"                                      # train_dpo.py line 49
MAX_LENGTH="${MAX_LENGTH:-1024}"                         # train_dpo.py line 50
LORA_R="${LORA_R:-16}"                                   # train_dpo.py line 52
LORA_ALPHA="${LORA_ALPHA:-32}"                           # train_dpo.py line 53
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"                     # train_dpo.py line 54
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"              # run_pilot_hh.sh line 22

HH_TEST="$ROOT/data/processed/hh_test.jsonl"
HH_TEST_EXPECTED_LINES="${HH_TEST_EXPECTED_LINES:-1000}"

CKPT_ROOT="$ROOT/results/checkpoints/seed_study"
EVAL_ROOT="$ROOT/results/eval/seed_study"
LOG_ROOT="$ROOT/analysis/outputs/seed_study"
DRIVER_LOG="$LOG_ROOT/driver.log"
MANIFEST="$ROOT/analysis/seed_study_manifest.json"
TIMING_JSON="$EVAL_ROOT/timing.json"

# Preflight thresholds.
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-20480}"   # 20 GiB of free memory per GPU
MIN_DISK_GB="${MIN_DISK_GB:-40}"                # free space on the results filesystem
HF_CACHE_DIR="${HF_CACHE_DIR:-/shared/models/huggingface/hub}"
# Per-run on-disk footprint estimate. A LoRA adapter of rank 16 over the four
# attention projections of Qwen2.5-1.5B carries about 4.36 million parameters,
# roughly 9 MB in bfloat16; the tokenizer files that trainer.save_model writes
# alongside it add about 11 MB, and the retained logs a few hundred kilobytes.
EST_PER_RUN_MB="${EST_PER_RUN_MB:-25}"

# Expected line count of each configuration's training file. These are the
# retained-pair counts recorded in data/filtered/eta20/filter_summary.json and
# restated in analysis/seed_study_training_exposure.md. A mismatch means the
# filtering stage was re-run and the study is no longer comparable to the
# published grid, so it is a hard preflight failure rather than a warning.
declare -A EXPECTED_LINES=(
    [raw]=10000
    [cdpo]=10000
    [single]=5905
    [consensus_k3]=4020
    [oracle]=7969
)
declare -A TRAIN_SHA=()
declare -A TRAIN_LINES=()

FORCE="${FORCE:-0}"

log() { echo "[seed-study $(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }
iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------- guarding
# The guard runs before any directory is created or any job is launched.
# Both GPUs may be occupied by the cross-backbone precompute or the model audit.
# Starting on top of those would evict them, so refuse unless FORCE=1.
GUARD_PATTERNS=(precompute_diverse.py run_model_audit.py)
BUSY=0
for pat in "${GUARD_PATTERNS[@]}"; do
    if pgrep -f "$pat" > /dev/null 2>&1; then
        echo "REFUSING TO START: a '$pat' process is currently running." >&2
        pgrep -af "$pat" >&2
        BUSY=1
    fi
done
if [[ "$BUSY" -eq 1 ]]; then
    if [[ "$FORCE" == "1" ]]; then
        echo "WARNING: FORCE=1 is set, proceeding despite the running job(s) above." >&2
    else
        cat >&2 <<'MSG'

Those processes occupy the GPUs and must not be disturbed. Wait for them to
finish, then re-run this script. It is idempotent, so nothing is lost by
waiting. Set FORCE=1 only if you are certain the GPUs are free.
MSG
        exit 1
    fi
fi

# ---------------------------------------------------------- training file map
train_file_for() {
    case "$1" in
        raw|cdpo)      echo "$FILT_DIR/raw.train.jsonl" ;;
        single)        echo "$FILT_DIR/single.train.jsonl" ;;
        consensus_k3)  echo "$FILT_DIR/consensus_k3.train.jsonl" ;;
        oracle)        echo "$FILT_DIR/oracle.train.jsonl" ;;
        *) return 1 ;;
    esac
}

# The only knob that differs between configurations is the cDPO label smoothing.
extra_args_for() {
    case "$1" in
        cdpo) echo "--label_smoothing 0.1" ;;
        *)    echo "" ;;
    esac
}

# ---------------------------------------------------------------- path safety
# Every artifact this driver writes must land inside results/checkpoints/seed_study/
# or results/eval/seed_study/. The published pilot artifacts live in
# results/eval/*.json and results/tables/, and nothing here may reach them.
CKPT_ROOT_CANON="$(readlink -m "$CKPT_ROOT")"
EVAL_ROOT_CANON="$(readlink -m "$EVAL_ROOT")"
FORBIDDEN_ROOTS=("$(readlink -m "$ROOT/results/tables")" "$(readlink -m "$ROOT/results/figures")")

path_is_under() {
    # path_is_under CHILD PARENT
    local child parent
    child="$(readlink -m "$1")"
    parent="$(readlink -m "$2")"
    [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}

assert_output_path() {
    # assert_output_path PATH  -> refuses unless PATH is inside one of the two
    # permitted output roots and outside every protected root.
    local p canon
    p="$1"
    canon="$(readlink -m "$p")"
    if ! path_is_under "$canon" "$CKPT_ROOT_CANON" && ! path_is_under "$canon" "$EVAL_ROOT_CANON"; then
        echo "REFUSING: output path escapes the seed-study output roots: $canon" >&2
        return 1
    fi
    local bad
    for bad in "${FORBIDDEN_ROOTS[@]}"; do
        if path_is_under "$canon" "$bad"; then
            echo "REFUSING: output path lands in a protected published tree: $canon" >&2
            return 1
        fi
    done
    # results/eval/*.json at the top level are published pilot artifacts.
    if [[ "$(dirname "$canon")" == "$(readlink -m "$ROOT/results/eval")" ]]; then
        echo "REFUSING: output path would write into the published results/eval root: $canon" >&2
        return 1
    fi
    return 0
}

# ------------------------------------------------------------------ preflight
# Every check below runs before any directory is created and before any job is
# launched. Each prints a single PASS or FAIL line. Any failure aborts.
PREFLIGHT_FAIL=0
pf_pass() { printf '[preflight] PASS  %s\n' "$*"; }
pf_fail() { printf '[preflight] FAIL  %s\n' "$*" >&2; PREFLIGHT_FAIL=1; }
pf_info() { printf '[preflight] info  %s\n' "$*"; }

printf '\n===== preflight =====\n'

# 1. Output-path containment, evaluated for every artifact the study will write.
PATH_OK=1
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tag="eta${ETA_TAG}_${cfg}_seed${seed}"
        for candidate in \
            "$CKPT_ROOT/$tag" \
            "$CKPT_ROOT/$tag/train.log" \
            "$CKPT_ROOT/$tag/eval.log" \
            "$EVAL_ROOT/${tag}.json" \
            "$EVAL_ROOT/${tag}.meta.json" \
            "$EVAL_ROOT/${tag}.done"
        do
            assert_output_path "$candidate" || PATH_OK=0
        done
    done
done
assert_output_path "$TIMING_JSON" || PATH_OK=0
if [[ "$PATH_OK" -eq 1 ]]; then
    pf_pass "all $(( ${#CONFIGS[@]} * ${#SEEDS[@]} * 6 + 1 )) output paths lie inside results/checkpoints/seed_study/ or results/eval/seed_study/; results/eval/*.json and results/tables/ are unreachable from this driver"
else
    pf_fail "at least one output path escapes the permitted seed-study roots"
fi

# 2. GPU availability: no compute processes and sufficient free memory.
if ! command -v nvidia-smi > /dev/null 2>&1; then
    pf_fail "nvidia-smi is not available, so GPU state cannot be verified"
else
    for gpu in "${GPUS[@]}"; do
        apps="$(nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
        if [[ -n "${apps//[[:space:]]/}" ]]; then
            pf_fail "GPU$gpu carries compute processes: $(echo "$apps" | tr '\n' ';')"
        else
            pf_pass "GPU$gpu reports no compute processes"
        fi
        free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
        gpu_name="$(nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null)"
        if [[ -z "$free_mib" ]]; then
            pf_fail "GPU$gpu free memory could not be read from nvidia-smi"
        elif (( free_mib < MIN_GPU_FREE_MIB )); then
            pf_fail "GPU$gpu ($gpu_name) has ${free_mib} MiB free, below the required ${MIN_GPU_FREE_MIB} MiB"
        else
            pf_pass "GPU$gpu ($gpu_name) has ${free_mib} MiB free, at or above the required ${MIN_GPU_FREE_MIB} MiB"
        fi
    done
fi

# 3. Concurrent-job guard. Re-stated here so that the preflight report is
#    complete; the hard refusal itself is above and runs first.
if [[ "$BUSY" -eq 1 ]]; then
    pf_pass "concurrent-job guard triggered on ${GUARD_PATTERNS[*]} and was overridden by FORCE=1"
else
    pf_pass "no precompute_diverse.py or run_model_audit.py process is running"
fi

# 4. Training files: existence, expected line count, recorded sha256.
for cfg in "${CONFIGS[@]}"; do
    tf="$(train_file_for "$cfg")" || { pf_fail "unknown configuration: $cfg"; continue; }
    if [[ ! -f "$tf" ]]; then
        pf_fail "training file for '$cfg' is missing: $tf"
        continue
    fi
    lines="$(wc -l < "$tf" | tr -d '[:space:]')"
    want="${EXPECTED_LINES[$cfg]:-}"
    sha="$(sha256sum "$tf" | awk '{print $1}')"
    TRAIN_SHA[$cfg]="$sha"
    TRAIN_LINES[$cfg]="$lines"
    if [[ -z "$want" ]]; then
        pf_fail "no expected line count is declared for configuration '$cfg'"
    elif [[ "$lines" != "$want" ]]; then
        pf_fail "training file for '$cfg' has $lines lines, expected $want: $tf"
    else
        pf_pass "training file for '$cfg' has the expected $want lines, sha256 ${sha:0:16} ($(basename "$tf"))"
    fi
done

# 5. Held-out test file.
if [[ ! -f "$HH_TEST" ]]; then
    pf_fail "held-out test file is missing: $HH_TEST"
else
    hh_lines="$(wc -l < "$HH_TEST" | tr -d '[:space:]')"
    HH_TEST_SHA="$(sha256sum "$HH_TEST" | awk '{print $1}')"
    if [[ "$hh_lines" != "$HH_TEST_EXPECTED_LINES" ]]; then
        pf_fail "held-out test file has $hh_lines lines, expected $HH_TEST_EXPECTED_LINES: $HH_TEST"
    else
        pf_pass "held-out test file has the expected $HH_TEST_EXPECTED_LINES lines, sha256 ${HH_TEST_SHA:0:16}"
    fi
fi
HH_TEST_SHA="${HH_TEST_SHA:-}"

# 6. Base model resolves from the local Hugging Face cache with no network call.
BASE_MODEL_CACHE_DIR="$HF_CACHE_DIR/models--${BASE_MODEL//\//--}"
if [[ ! -d "$BASE_MODEL_CACHE_DIR" ]]; then
    pf_fail "base model $BASE_MODEL is not present in the local cache at $BASE_MODEL_CACHE_DIR"
else
    resolved="$(HF_HUB_OFFLINE=1 HF_HUB_CACHE="$HF_CACHE_DIR" HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR" \
        python3 - "$BASE_MODEL" <<'PY' 2>/dev/null
import sys
try:
    from huggingface_hub import try_to_load_from_cache
except Exception:
    sys.exit(2)
repo = sys.argv[1]
paths = []
for fname in ("config.json", "tokenizer.json"):
    p = try_to_load_from_cache(repo, fname)
    if not isinstance(p, str):
        sys.exit(3)
    paths.append(p)
print(paths[0])
PY
)"
    if [[ -n "$resolved" ]]; then
        pf_pass "base model $BASE_MODEL resolves offline from the local cache: $resolved"
    else
        pf_fail "base model $BASE_MODEL did not resolve offline from $HF_CACHE_DIR; a launch would attempt a network fetch"
    fi
fi

# 7. Disk space on the filesystem holding results/, and the footprint estimate.
DISK_PROBE="$ROOT/results"
[[ -d "$DISK_PROBE" ]] || DISK_PROBE="$ROOT"
avail_gb="$(df -BG --output=avail "$DISK_PROBE" 2>/dev/null | tail -1 | tr -dc '0-9')"
n_runs=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
est_mb=$(( n_runs * EST_PER_RUN_MB ))
pf_info "estimated footprint of $n_runs LoRA adapters plus retained logs: about ${est_mb} MB (about $(awk -v m="$est_mb" 'BEGIN{printf "%.2f", m/1024}') GB), at roughly ${EST_PER_RUN_MB} MB per run"
if [[ -z "$avail_gb" ]]; then
    pf_fail "free disk space on the filesystem holding $DISK_PROBE could not be determined"
elif (( avail_gb < MIN_DISK_GB )); then
    pf_fail "filesystem holding results/ has ${avail_gb} GB free, below the required ${MIN_DISK_GB} GB (the requirement is deliberate headroom, far above the ${est_mb} MB the study itself needs)"
else
    pf_pass "filesystem holding results/ has ${avail_gb} GB free, at or above the required ${MIN_DISK_GB} GB"
fi

if [[ "$PREFLIGHT_FAIL" -eq 1 ]]; then
    printf '\n[preflight] one or more checks failed. Nothing was launched, no directory was created and no file was written.\n' >&2
    exit 1
fi
printf '[preflight] all checks passed\n\n'

# --------------------------------------------------------------- directories
mkdir -p "$CKPT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

log "root=$ROOT eta=$ETA_TAG corr=$CORR base=$BASE_MODEL"
log "configs=${CONFIGS[*]} seeds=${SEEDS[*]} gpus=${GPUS[*]}"
log "hyperparameters: epochs=$EPOCHS bs=$BATCH_SIZE ga=$GRAD_ACCUM lr=$LR beta=$BETA max_length=$MAX_LENGTH lora_r=$LORA_R lora_alpha=$LORA_ALPHA lora_dropout=$LORA_DROPOUT"

# ------------------------------------------------------------------ manifest
# Written once at launch. If a manifest already exists its substantive content
# is compared against the current one and any difference is reported loudly,
# because a difference means the study inputs changed between invocations.
write_manifest() {
    local cfg tf entries=""
    for cfg in "${CONFIGS[@]}"; do
        tf="$(train_file_for "$cfg")"
        entries+="${cfg}|${tf}|${TRAIN_SHA[$cfg]:-}|${TRAIN_LINES[$cfg]:-}|$(extra_args_for "$cfg")"$'\n'
    done
    SM_ENTRIES="$entries" \
    SM_SEEDS="${SEEDS[*]}" \
    SM_ETA="$ETA_TAG" \
    SM_CORR="$CORR" \
    SM_BASE="$BASE_MODEL" \
    SM_HH_TEST="$HH_TEST" \
    SM_HH_SHA="$HH_TEST_SHA" \
    SM_HH_LINES="$HH_TEST_EXPECTED_LINES" \
    SM_OUT="$MANIFEST" \
    SM_HYPER="epochs=$EPOCHS;batch_size=$BATCH_SIZE;grad_accum=$GRAD_ACCUM;lr=$LR;beta=$BETA;max_length=$MAX_LENGTH;lora_r=$LORA_R;lora_alpha=$LORA_ALPHA;lora_dropout=$LORA_DROPOUT;eval_max_samples=$EVAL_MAX_SAMPLES" \
    python3 - <<'PY'
import json, os, datetime, pathlib

entries = []
for line in os.environ["SM_ENTRIES"].splitlines():
    if not line.strip():
        continue
    cfg, path, sha, lines, extra = line.split("|", 4)
    entries.append({
        "config": cfg,
        "train_file": path,
        "train_file_sha256": sha,
        "train_file_lines": int(lines) if lines else None,
        "extra_training_args": extra.strip() or None,
    })

hyper = {}
for kv in os.environ["SM_HYPER"].split(";"):
    if not kv:
        continue
    k, raw = kv.split("=", 1)
    try:
        hyper[k] = int(raw)
    except ValueError:
        try:
            hyper[k] = float(raw)
        except ValueError:
            hyper[k] = raw

manifest = {
    "study": "training-seed variance study",
    "written_at": datetime.datetime.now(datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "seeds": [int(s) for s in os.environ["SM_SEEDS"].split()],
    "eta_nominal_pct": int(os.environ["SM_ETA"]),
    "corruption": os.environ["SM_CORR"],
    "base_model": os.environ["SM_BASE"],
    "hyperparameters": hyper,
    "configurations": entries,
    "held_out_test": {
        "path": os.environ["SM_HH_TEST"],
        "sha256": os.environ["SM_HH_SHA"] or None,
        "lines": int(os.environ["SM_HH_LINES"]),
    },
    "fixed_data_provenance": {
        "subsample": {
            "producer": "src/load_datasets.py",
            "seed": 42,
            "note": "The HH-RLHF train, dev and test subsamples are drawn once "
                    "at seed 42 and are identical for every run in this study.",
        },
        "corruption": {
            "producer": "src/create_corruption.py",
            "seed": 42,
            "corruption": "structured_unsafe",
            "eta_nominal_pct": 20,
            "corrupted_rows": 2031,
            "note": "The corruption realization is drawn once at seed 42 and is "
                    "identical for every run in this study. It injects 2031 "
                    "corrupted rows into the 10000-row training set, an "
                    "effective eta of 0.2031.",
        },
        "filtering": {
            "note": "Verifier scoring and the retained-pair sets are fixed "
                    "artifacts on disk. No filtering step is re-run by the "
                    "seed study.",
        },
    },
    "what_varies": "Only the training seed passed to src/train_dpo.py varies "
                   "across runs. The dataset subsample, the corruption "
                   "realization, the verifier scores, the retained-pair sets, "
                   "the base model, the held-out test set and every "
                   "hyperparameter are held fixed.",
    "what_this_estimates": "This study estimates training-seed uncertainty "
                           "alone. It does not estimate uncertainty over "
                           "corruption draws, over dataset subsamples, or over "
                           "verifier scoring, and it must not be described as "
                           "though it did.",
}

out = pathlib.Path(os.environ["SM_OUT"])
comparable = {k: v for k, v in manifest.items() if k != "written_at"}
if out.is_file():
    try:
        prior = json.loads(out.read_text())
    except Exception:
        prior = None
    prior_comparable = ({k: v for k, v in prior.items() if k != "written_at"}
                        if isinstance(prior, dict) else None)
    if prior_comparable == comparable:
        print(f"[manifest] unchanged, retained: {out}")
    else:
        alt = out.with_suffix(".new.json")
        alt.write_text(json.dumps(manifest, indent=2) + "\n")
        print("[manifest] WARNING: the existing manifest differs from the "
              "current study inputs. The existing file was NOT overwritten; "
              f"the current inputs were written to {alt}. Reconcile them "
              "before trusting any pooled result.")
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] wrote {out}")
PY
}
write_manifest | tee -a "$DRIVER_LOG"

# ------------------------------------------------------- environment capture
# Collected once, outside the run loop, so that fifteen runs do not pay for
# fifteen interpreter starts. The values describe the environment the runs
# execute in and are copied verbatim into every per-run meta JSON.
ENV_JSON="$(python3 - <<'PY'
import json, platform, sys
info = {"python": sys.version.split()[0], "hostname": platform.node()}
for mod in ("torch", "transformers", "trl", "peft", "datasets"):
    try:
        info[mod] = __import__(mod).__version__
    except Exception as exc:
        info[mod] = f"unavailable: {exc.__class__.__name__}"
print(json.dumps(info))
PY
)"
HOSTNAME_STR="$(hostname)"
log "environment: $ENV_JSON"

# --------------------------------------------------------------- run one item
run_one() {
    local gpu="$1" cfg="$2" seed="$3"
    local tag="eta${ETA_TAG}_${cfg}_seed${seed}"
    local ckpt="$CKPT_ROOT/$tag"
    local eval_json="$EVAL_ROOT/${tag}.json"
    local meta_json="$EVAL_ROOT/${tag}.meta.json"
    local done_marker="$EVAL_ROOT/${tag}.done"
    local train_log="$ckpt/train.log"
    local eval_log="$ckpt/eval.log"
    local time_file="$LOG_ROOT/${tag}.walltime"
    local train_file extra start end elapsed rc

    # Defence in depth: the preflight already checked these paths, and they are
    # checked again here so that no future edit to the tag scheme can write
    # outside the two permitted roots.
    local p
    for p in "$ckpt" "$train_log" "$eval_log" "$eval_json" "$meta_json" "$done_marker"; do
        assert_output_path "$p" || { log "GPU$gpu REFUSED $tag (unsafe output path $p)"; return 1; }
    done

    # Resume keys on the completion marker, never on the evaluation JSON alone.
    # An evaluation JSON without a marker means the previous attempt died part
    # way through evaluation or produced an unusable file, so it is redone.
    if [[ -f "$done_marker" ]]; then
        log "GPU$gpu skip $tag (completion marker present)"
        return 0
    fi
    if [[ -f "$eval_json" ]]; then
        log "GPU$gpu $tag has an evaluation JSON but no completion marker; the previous attempt did not verify, so evaluation is repeated"
        rm -f "$eval_json"
    fi

    train_file="$(train_file_for "$cfg")"
    extra="$(extra_args_for "$cfg")"
    local -a extra_arr=()
    [[ -n "$extra" ]] && read -r -a extra_arr <<< "$extra"

    mkdir -p "$ckpt"
    start=$(date +%s)
    local start_iso
    start_iso="$(iso_now)"

    local -a train_cmd=(
        python3 -m src.train_dpo
        --train_file "$train_file"
        --output_dir "$ckpt"
        --base_model "$BASE_MODEL"
        --epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE"
        --grad_accum "$GRAD_ACCUM"
        --lr "$LR"
        --beta "$BETA"
        --max_length "$MAX_LENGTH"
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --seed "$seed"
        "${extra_arr[@]}"
    )
    local -a eval_cmd=(
        python3 "$ROOT/analysis/scripts/eval_preference_only.py"
        --model_dir "$ckpt"
        --hh_test "$HH_TEST"
        --out_json "$eval_json"
        --base_model "$BASE_MODEL"
        --max_length "$MAX_LENGTH"
        --max_samples "$EVAL_MAX_SAMPLES"
    )
    local train_cmd_str eval_cmd_str
    train_cmd_str="CUDA_VISIBLE_DEVICES=$gpu $(printf '%q ' "${train_cmd[@]}")"
    eval_cmd_str="CUDA_VISIBLE_DEVICES=$gpu $(printf '%q ' "${eval_cmd[@]}")"

    local did_train=1
    if [[ -f "$ckpt/train_status.json" ]]; then
        did_train=0
        log "GPU$gpu $tag already trained; evaluating only"
    else
        log "GPU$gpu train $tag  file=$(basename "$train_file") extra='${extra:-none}'"
        CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}" > "$train_log" 2>&1
        rc=$?
        if [[ $rc -ne 0 || ! -f "$ckpt/train_status.json" ]]; then
            log "GPU$gpu TRAIN FAILED $tag rc=$rc (see $train_log)"
            return 1
        fi
    fi

    log "GPU$gpu eval $tag"
    CUDA_VISIBLE_DEVICES="$gpu" "${eval_cmd[@]}" > "$eval_log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -f "$eval_json" ]]; then
        log "GPU$gpu EVAL FAILED $tag rc=$rc (see $eval_log)"
        return 1
    fi

    end=$(date +%s)
    elapsed=$(( end - start ))
    local end_iso
    end_iso="$(iso_now)"

    local cuda_name
    cuda_name="$(nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null | tr -d '\n')"
    [[ -n "$cuda_name" ]] || cuda_name="unknown"

    # Verification and provenance. The completion marker is written by this step
    # and only if every condition holds.
    RM_TAG="$tag" RM_CONFIG="$cfg" RM_SEED="$seed" RM_GPU="$gpu" \
    RM_ETA="$ETA_TAG" RM_CORR="$CORR" \
    RM_CKPT="$ckpt" RM_EVAL_JSON="$eval_json" RM_META="$meta_json" \
    RM_DONE="$done_marker" RM_TRAIN_LOG="$train_log" RM_EVAL_LOG="$eval_log" \
    RM_TRAIN_FILE="$train_file" \
    RM_TRAIN_SHA="${TRAIN_SHA[$cfg]:-}" RM_TRAIN_LINES="${TRAIN_LINES[$cfg]:-}" \
    RM_BASE="$BASE_MODEL" RM_HH_TEST="$HH_TEST" RM_HH_SHA="$HH_TEST_SHA" \
    RM_TRAIN_CMD="$train_cmd_str" RM_EVAL_CMD="$eval_cmd_str" \
    RM_START="$start_iso" RM_END="$end_iso" RM_WALL="$elapsed" \
    RM_HOST="$HOSTNAME_STR" RM_ENV="$ENV_JSON" RM_CUDA_NAME="$cuda_name" \
    RM_EXPECT_N="$EVAL_MAX_SAMPLES" \
    RM_HYPER="epochs=$EPOCHS;batch_size=$BATCH_SIZE;grad_accum=$GRAD_ACCUM;lr=$LR;beta=$BETA;max_length=$MAX_LENGTH;lora_r=$LORA_R;lora_alpha=$LORA_ALPHA;lora_dropout=$LORA_DROPOUT;seed=$seed;label_smoothing=$([[ "$cfg" == cdpo ]] && echo 0.1 || echo 0.0);loss_type=sigmoid;eval_max_samples=$EVAL_MAX_SAMPLES" \
    python3 "$ROOT/analysis/scripts/seed_study_finalize.py"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        log "GPU$gpu VERIFICATION FAILED $tag; no completion marker was written (see $eval_log)"
        return 1
    fi

    # The wall time is recorded only when this invocation actually trained. An
    # evaluation-only rerun takes a few minutes rather than a few hours, and
    # letting it overwrite the recorded time would corrupt the timing model.
    if [[ "$did_train" -eq 1 ]]; then
        echo "$elapsed" > "$time_file"
    fi
    log "GPU$gpu done $tag in ${elapsed}s (trained=$did_train, marker $done_marker)"
    report_eta_if_ready
    return 0
}

# ------------------------------------------------------------ measured timing
# After the first raw run and the first consensus_k3 run have finished, the
# remaining queue is re-estimated from their measured wall times. Wall time is
# modelled as affine in the number of training pairs, which the two anchors
# determine exactly. This is reporting only. No cutoff and no automatic stop is
# derived from it: the decision to stop the study is a human checkpoint on
# 2 August and is not delegated to this script.
ETA_MARKER="$LOG_ROOT/.eta_reported"

report_eta_if_ready() {
    local raw_t cons_t
    raw_t=""
    cons_t=""
    local s
    for s in "${SEEDS[@]}"; do
        [[ -z "$raw_t" && -f "$LOG_ROOT/eta${ETA_TAG}_raw_seed${s}.walltime" ]] && \
            raw_t="$(cat "$LOG_ROOT/eta${ETA_TAG}_raw_seed${s}.walltime")"
        [[ -z "$cons_t" && -f "$LOG_ROOT/eta${ETA_TAG}_consensus_k3_seed${s}.walltime" ]] && \
            cons_t="$(cat "$LOG_ROOT/eta${ETA_TAG}_consensus_k3_seed${s}.walltime")"
    done
    [[ -n "$raw_t" && -n "$cons_t" ]] || return 0
    mkdir "$ETA_MARKER" 2>/dev/null || return 0   # atomic: report exactly once

    local remaining=""
    local cfg seed tag
    for cfg in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="eta${ETA_TAG}_${cfg}_seed${seed}"
            [[ -f "$EVAL_ROOT/${tag}.done" ]] && continue
            remaining+="${cfg}|${TRAIN_LINES[$cfg]:-0}"$'\n'
        done
    done

    ET_RAW="$raw_t" ET_CONS="$cons_t" \
    ET_RAW_N="${TRAIN_LINES[raw]:-10000}" ET_CONS_N="${TRAIN_LINES[consensus_k3]:-4020}" \
    ET_REMAINING="$remaining" ET_NGPU="${#GPUS[@]}" ET_OUT="$TIMING_JSON" \
    python3 - <<'PY' | tee -a "$DRIVER_LOG"
import json, os, datetime, pathlib

t_raw = float(os.environ["ET_RAW"])
t_cons = float(os.environ["ET_CONS"])
n_raw = float(os.environ["ET_RAW_N"])
n_cons = float(os.environ["ET_CONS_N"])
ngpu = max(1, int(os.environ["ET_NGPU"]))

fallback = None
if n_raw != n_cons:
    slope = (t_raw - t_cons) / (n_raw - n_cons)
else:
    slope = 0.0
intercept = t_raw - slope * n_raw
if slope <= 0.0 or intercept < 0.0:
    # The two anchors do not admit an increasing affine fit, which happens when
    # the measured times are too close together to separate. Fall back to a
    # purely proportional model anchored on the longer of the two runs.
    fallback = ("the two anchors do not admit an increasing affine fit, so a "
                "proportional model anchored on the longer run is used instead")
    if t_raw / max(n_raw, 1.0) >= t_cons / max(n_cons, 1.0):
        slope = t_raw / max(n_raw, 1.0)
    else:
        slope = t_cons / max(n_cons, 1.0)
    intercept = 0.0

remaining = []
for line in os.environ["ET_REMAINING"].splitlines():
    if not line.strip():
        continue
    cfg, n = line.split("|", 1)
    remaining.append((cfg, float(n)))

def predict(n):
    return max(0.0, intercept + slope * n)

total = sum(predict(n) for _, n in remaining)
eta_seconds = total / ngpu

record = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "anchors": {
        "raw": {"train_pairs": n_raw, "wall_seconds": t_raw},
        "consensus_k3": {"train_pairs": n_cons, "wall_seconds": t_cons},
    },
    "model": {
        "form": "wall_seconds = intercept + slope * train_pairs",
        "intercept_seconds": intercept,
        "slope_seconds_per_pair": slope,
        "fallback": fallback,
    },
    "remaining_runs": len(remaining),
    "remaining_gpu_seconds": total,
    "gpus": ngpu,
    "eta_seconds": eta_seconds,
    "eta_hours": eta_seconds / 3600.0,
    "per_config_predicted_seconds": {
        cfg: predict(n) for cfg, n in dict(remaining).items()
    },
    "note": "Reporting only. No automatic cutoff and no automatic stop is "
            "derived from this estimate. The decision to stop is a human "
            "checkpoint on 2 August.",
}

out = pathlib.Path(os.environ["ET_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
history = []
if out.is_file():
    try:
        prior = json.loads(out.read_text())
        history = prior if isinstance(prior, list) else [prior]
    except Exception:
        history = []
history.append(record)
out.write_text(json.dumps(history, indent=2) + "\n")

print("[timing] measured anchors: raw {:.0f}s over {:.0f} pairs, "
      "consensus_k3 {:.0f}s over {:.0f} pairs"
      .format(t_raw, n_raw, t_cons, n_cons))
print("[timing] fitted wall time = {:.1f}s + {:.4f}s per training pair"
      .format(intercept, slope))
if fallback:
    print("[timing] note: {}".format(fallback))
print("[timing] {} run(s) remain, {:.0f} GPU-seconds in total, "
      "about {:.2f} hours of wall time on {} GPU(s)"
      .format(len(remaining), total, eta_seconds / 3600.0, ngpu))
print("[timing] appended to {}".format(out))
print("[timing] this is an estimate only; no automatic cutoff is applied")
PY
}

# ------------------------------------------------------------------ work list
# Ordered configuration-major so that, if the study is interrupted, whole seeds
# of the earlier configurations are complete rather than fragments of all five.
WORK=()
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        WORK+=("${cfg}:${seed}")
    done
done
log "${#WORK[@]} runs queued"

# One serial worker per GPU, drawing from a single shared queue. A worker claims
# an item by creating its claim directory, which mkdir performs atomically, so
# no item is ever taken twice and no GPU ever holds two runs. Concurrency is
# therefore exactly ${#GPUS[@]}, and because the queue is shared rather than
# pre-dealt, a GPU that finishes a short configuration immediately picks up the
# next outstanding run instead of idling.
NGPU=${#GPUS[@]}
CLAIM_DIR="$LOG_ROOT/.claims"
# The claim directory is bookkeeping under analysis/outputs and is reset on each
# invocation. This is the only removal the driver performs. It never touches
# results/checkpoints/seed_study/, so no trained adapter is ever deleted.
if [[ "$CLAIM_DIR" == "$LOG_ROOT/.claims" ]]; then
    rm -rf "$CLAIM_DIR"
fi
mkdir -p "$CLAIM_DIR"
rm -rf "$ETA_MARKER"

worker() {
    local slot="$1" gpu="${GPUS[$1]}" item cfg seed
    for item in "${WORK[@]}"; do
        cfg="${item%%:*}"
        seed="${item##*:}"
        if ! mkdir "$CLAIM_DIR/eta${ETA_TAG}_${cfg}_seed${seed}" 2>/dev/null; then
            continue
        fi
        run_one "$gpu" "$cfg" "$seed"
    done
    log "GPU$gpu worker finished"
}

STUDY_START=$(date +%s)
for (( s = 0; s < NGPU; s++ )); do
    worker "$s" &
done
wait
STUDY_ELAPSED=$(( $(date +%s) - STUDY_START ))
log "all workers finished in ${STUDY_ELAPSED}s"

# --------------------------------------------------------------- summary table
printf '\n'
printf '%-14s %-6s %-9s %-10s %-12s %s\n' CONFIG SEED STATUS WALL_S PREF_ACC MEAN_MARGIN
printf '%s\n' "--------------------------------------------------------------------------"
N_DONE=0
N_MISSING=0
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tag="eta${ETA_TAG}_${cfg}_seed${seed}"
        eval_json="$EVAL_ROOT/${tag}.json"
        done_marker="$EVAL_ROOT/${tag}.done"
        time_file="$LOG_ROOT/${tag}.walltime"
        wall="-"
        [[ -f "$time_file" ]] && wall="$(cat "$time_file")"
        if [[ -f "$done_marker" && -f "$eval_json" ]]; then
            read -r acc margin < <(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"{d['preference_acc']:.4f}\", f\"{d['mean_margin']:.6f}\")
" "$eval_json")
            printf '%-14s %-6s %-9s %-10s %-12s %s\n' "$cfg" "$seed" OK "$wall" "$acc" "$margin"
            N_DONE=$((N_DONE + 1))
        elif [[ -f "$eval_json" ]]; then
            printf '%-14s %-6s %-9s %-10s %-12s %s\n' "$cfg" "$seed" UNVERIFIED "$wall" "-" "-"
            N_MISSING=$((N_MISSING + 1))
        else
            printf '%-14s %-6s %-9s %-10s %-12s %s\n' "$cfg" "$seed" MISSING "$wall" "-" "-"
            N_MISSING=$((N_MISSING + 1))
        fi
    done
done
printf '%s\n' "--------------------------------------------------------------------------"
printf '%d of %d runs complete, %d missing. Total wall time this invocation: %ds\n' \
    "$N_DONE" "$((N_DONE + N_MISSING))" "$N_MISSING" "$STUDY_ELAPSED"
printf 'Checkpoints under %s are retained. Nothing in this driver deletes them.\n' "$CKPT_ROOT"

if [[ "$N_MISSING" -gt 0 ]]; then
    printf 'Re-run this script to resume; runs carrying a completion marker will be skipped.\n'
    exit 1
fi

printf 'Next step: python3 analysis/scripts/summarize_seed_study.py\n'
