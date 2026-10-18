#!/usr/bin/env bash
# LLM-as-judge scoring of every answer set run_all_answers.sh produced for the
# current experiment (the spec below). Discovers each predictions.jsonl under the
# experiment's _answers root and writes metrics.<judge-slug>.json next to it (so
# different judges' scores coexist instead of clobbering).
#
# Usage:
#   bash experiments/run_all_judge.sh
#   FORCE=1 bash experiments/run_all_judge.sh             # re-judge sets already scored by THIS judge
#   JUDGE_MODEL=gpt-4o bash experiments/run_all_judge.sh  # a LABELLED sensitivity check only
#                                                         # (gpt-4o-mini is the fixed project judge)
#   JUDGE_MODEL=gpt-5.5 JUDGE_REASONING_EFFORT=low bash experiments/run_all_judge.sh  # GPT-5 judge, sensitivity check
#   BATCH=1 bash experiments/run_all_judge.sh             # grade via OpenAI Batch API (cheaper, no rate limit)
#   INCLUDE_ORACLE=0 FORCE=1 bash experiments/run_all_judge.sh   # redo the sweep, leave oracle alone
#   ORACLE_ONLY=1 bash experiments/run_all_judge.sh       # oracle cells only
#   LIST_ONLY=1 bash experiments/run_all_judge.sh         # show what would be judged, run nothing
#
# Note: no `set -e` so one failing set doesn't kill the rest.
set -uo pipefail

# ---- Shared config from the experiment spec (SYSTEM/DATASET/MEM_MODEL/...) ----
# Override with SPEC=path to judge a different experiment without editing the
# default (e.g. SPEC=experiments/user_turns/longmemeval.yaml bash experiments/run_all_judge.sh).
SPEC="${SPEC:-$(dirname "$0")/lightmem/longmemeval.yaml}"
source "$(dirname "$0")/load_spec.sh"

# ---- judge-only knobs ----
# JUDGE_MODEL comes from the spec (judge_model:) via load_spec.sh, overridable
# with JUDGE_MODEL=... ; the literal is a fallback. Keep it DIFFERENT from the
# generator (self-grading bias). gpt-4o-mini is the project's judge, held fixed
# across systems; JUDGE_REASONING_EFFORT/JUDGE_VERBOSITY below are ignored by it
# (openai_gen only sends them to gpt-5/o-series models).
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-}     # empty = judge's OpenAI default
# GPT-5/o-series judge config (ignored by models that don't support them). Lower
# reasoning effort is cheaper/faster; a seed makes judging more reproducible.
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-medium}   # none|low|medium|high|xhigh
JUDGE_VERBOSITY=${JUDGE_VERBOSITY:-low}                  # low|medium|high
JUDGE_SEED=${JUDGE_SEED:-42}                            # integer
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.0}            # sampling temperature; 0.0 = greedy (gpt-5 ignores it)
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}             # per-call output budget; raise for reasoning judges
# Which cells to judge. Same vocabulary as run_all_answers.sh, so a judge run can
# be scoped exactly like the answer run that produced it.
#   INCLUDE_ORACLE=0   judge the retriever cells, SKIP oracle. What you want with
#                      FORCE=1 when oracle is already scored and only the sweep
#                      needs redoing — FORCE alone would re-judge oracle too.
#   ORACLE_ONLY=1      judge ONLY oracle.
INCLUDE_ORACLE=${INCLUDE_ORACLE:-1}
ORACLE_ONLY=${ORACLE_ONLY:-0}
if [[ "$INCLUDE_ORACLE" == "0" && "$ORACLE_ONLY" == "1" ]]; then
  echo "!!! INCLUDE_ORACLE=0 and ORACLE_ONLY=1 contradict each other. Pick one." >&2
  exit 1
fi
# List what would be judged and exit. No API calls, no writes.
LIST_ONLY=${LIST_ONLY:-0}
BATCH=${BATCH:-0}                      # 1 = grade via OpenAI Batch API (cheaper, no rate limit; OpenAI judge only)
BATCH_POLL_INTERVAL=${BATCH_POLL_INTERVAL:-30}         # seconds between batch status checks
FORCE=${FORCE:-0}                      # 1 = re-judge even if metrics.<judge>.json exists
# ---- Finishing the items that came back without a verdict -------------------
# A judge call can fail — a 429, a 500, a content filter, or a reply truncated at
# --judge-max-tokens. Those items are recorded as UNJUDGED and excluded from the
# accuracy (see Judge.build_metrics); they are NOT counted wrong. This pass
# finishes them: it re-judges ONLY the unjudged items, LIVE, and rewrites the
# metrics file complete. It is on by default because an incomplete metrics file
# is the failure mode this whole mechanism exists to make visible, and leaving
# one lying around defeats that.
#
# LIVE even under BATCH=1, deliberately: the stragglers are a handful of calls,
# and a second Batch job would mean another queue wait (up to 24h) for them.
RETRY_FAILED=${RETRY_FAILED:-1}

# Space-separated question_types to drop from scoring. Empty = score all 500, as
# every published LongMemEval table does (see run_all_retrieval.sh). Keep in sync
# with run_all_answers.sh so judged metrics cover the same question set.
EXCLUDE_QUESTION_TYPES=${EXCLUDE_QUESTION_TYPES:-""}

LOG_DIR=logs/judge
mkdir -p "$LOG_DIR"

# One --exclude-question-type per type (string form for set -u safety).
EXCLUDE_FLAGS=""
for qt in $EXCLUDE_QUESTION_TYPES; do EXCLUDE_FLAGS+=" --exclude-question-type $qt"; done

# Path-safe slug, matching evalgrid.lib.paths.model_slug.
slug() { echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'; }

# 0 when a metrics file scored every item, 1 when some came back without a
# verdict. Metrics written before `complete` existed have no such key and are
# treated as complete, so this never re-judges an older run's finished file.
# A malformed/half-written file reports incomplete — the safe direction.
metrics_complete() {
  python - "$1" <<'PYEOF' 2>/dev/null || return 1
import json, sys
try:
    with open(sys.argv[1]) as f:
        m = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if m.get("complete", True) else 1)
PYEOF
}

# Answers root for this experiment, mirroring evalgrid.lib.paths.answers_dir's first levels:
#   outputs/_answers/<system>/<dataset>/<mem-model>/
# Retriever answers live directly under here (no labeler segment — they don't
# depend on the labeler); oracle answers are keyed by labeler under
# oracle/<labeler-segment>/, so we judge the retriever sets plus only THIS
# experiment's oracle.
labeler_seg=$(slug "$LABELER_MODEL")
[[ "$CANDIDATE_SOURCE" == "retrieval" ]] && labeler_seg="${labeler_seg}-retrieval"
# <mem-seg> = mem-model slug plus "--<mem-variant>" when the spec carries one,
# matching evalgrid.lib.paths.mem_segment.
#
# ANSWERS_ROOT is overridable so a SMOKE build can be judged in its own tree.
# The answer jobs write there when OUT_ROOT is set — outputs/_smoke4_answers/
# <system>/<retriever>/<cut>/ — which is a flatter layout than the keyed one
# below, but this script only ever globs "*/predictions.jsonl" beneath the root,
# so the depth does not matter:
#
#   ANSWERS_ROOT=outputs/_smoke4_answers/all_turns \
#     SPEC=experiments/all_turns/longmemeval.yaml bash experiments/run_all_judge.sh
#
# Judging a smoke build in the keyed tree is what this exists to avoid: the
# metrics file would land beside a cell a full run will later fill, labelled by
# nothing that says it scored four questions.
ANSWERS_ROOT="${ANSWERS_ROOT:-outputs/_answers/$SYSTEM/$DATASET/$(slug "$MEM_MODEL")${MEM_VARIANT:+--$(slug "$MEM_VARIANT")}}"
ORACLE_ROOT="$ANSWERS_ROOT/oracle"

if [[ ! -d "$ANSWERS_ROOT" ]]; then
  echo "No answers at $ANSWERS_ROOT — run experiments/run_all_answers.sh first." >&2
  exit 1
fi

echo "Judging answers under $ANSWERS_ROOT (oracle: $labeler_seg)"
echo "Judge model: $JUDGE_MODEL"

# run_judge.py writes metrics.<judge-slug>.json; key the skip-check + logs by the
# same slug so re-judging with a different JUDGE_MODEL doesn't look "already done".
judge_slug=$(slug "$JUDGE_MODEL")

# Judge flags shared by every invocation (everything but --predictions and the
# batch-phase flags, which differ per call).
COMMON_JFLAGS=(--dataset "$DATASET" --judge-model "$JUDGE_MODEL")
[[ -n "$JUDGE_BASE_URL" ]] && COMMON_JFLAGS+=(--judge-base-url "$JUDGE_BASE_URL")
[[ -n "$JUDGE_REASONING_EFFORT" ]] && COMMON_JFLAGS+=(--judge-reasoning-effort "$JUDGE_REASONING_EFFORT")
[[ -n "$JUDGE_VERBOSITY" ]] && COMMON_JFLAGS+=(--judge-verbosity "$JUDGE_VERBOSITY")
[[ -n "$JUDGE_SEED" ]] && COMMON_JFLAGS+=(--judge-seed "$JUDGE_SEED")
[[ -n "$JUDGE_TEMPERATURE" ]] && COMMON_JFLAGS+=(--judge-temperature "$JUDGE_TEMPERATURE")
[[ -n "$JUDGE_MAX_TOKENS" ]] && COMMON_JFLAGS+=(--judge-max-tokens "$JUDGE_MAX_TOKENS")

# Run judge.py for one predictions file (extra flags = batch phase, if any),
# logging by the file's slug. Echoes nothing; returns the python exit code.
run_judge_file() {
  local pred="$1"; shift
  local safe_name
  safe_name=$(echo "${pred%/predictions.jsonl}" | sed "s#^$ANSWERS_ROOT/##; s#/#_#g")
  python -m evalgrid.pipeline.judge --predictions "$pred" "${COMMON_JFLAGS[@]}" "$@" $EXCLUDE_FLAGS \
    2>&1 | tee "$LOG_DIR/judge_${judge_slug}_${safe_name}.log"
  return "${PIPESTATUS[0]}"
}

# ---- Gather the predictions files to judge (oracle filter + skip-already-done) ----
fail=0 judged=0 skipped=0
listed=0 retried=0
preds=()
retries=()
while IFS= read -r pred; do
  # Oracle answers are labeler-keyed; judge only this experiment's labeler_seg,
  # skipping other labelers' oracle sets that share this mem-model root.
  #
  # The second pattern also has to admit the GOLD-LEVEL suffix. EVERY oracle now
  # carries one — oracle/<labeler_seg>-gold-<level>/ — which matches "under
  # ORACLE_ROOT" but NOT "under ORACLE_ROOT/<labeler_seg>/", so a bare
  # two-pattern test would drop the upper bound from the table with nothing
  # saying why: exactly the failure the block above warns about. Levels are
  # judged together on purpose: strict and turn are two rows of one sensitivity
  # check, not two experiments. The unsuffixed pattern is kept for trees not yet
  # run through evalgrid.tools.migrate_gold_names.
  if [[ "$pred" == "$ORACLE_ROOT"/* \
        && "$pred" != "$ORACLE_ROOT/$labeler_seg"/* \
        && "$pred" != "$ORACLE_ROOT/$labeler_seg"-gold-*/* ]]; then
    continue
  fi
  # Oracle in/out. Applied AFTER the labeler_seg guard above so that guard still
  # decides WHICH oracle dir is the current one; this only decides whether oracle
  # is in scope at all.
  if [[ "$pred" == "$ORACLE_ROOT"/* ]]; then
    [[ "$INCLUDE_ORACLE" == "0" ]] && continue
  else
    [[ "$ORACLE_ONLY" == "1" ]] && continue
  fi
  dir=$(dirname "$pred")
  rel=${dir#"$ANSWERS_ROOT/"}
  if [[ "$LIST_ONLY" == "1" ]]; then
    printf '  %s%s\n' "$rel" \
      "$([[ -f "$dir/metrics.${judge_slug}.json" ]] && echo '   [scored]')"
    listed=$((listed + 1)); continue
  fi
  if [[ "$FORCE" != "1" && -f "$dir/metrics.${judge_slug}.json" ]]; then
    # An INCOMPLETE metrics file is not "already judged". Skipping it here is
    # what used to make a partial score permanent: the run that produced it
    # reported the failures on stdout, in a log named after a job id, and every
    # later run saw the file and moved on — so the incomplete number is the one
    # that reaches the table. Queue it for the retry pass instead.
    if [[ "$RETRY_FAILED" == "1" ]] && ! metrics_complete "$dir/metrics.${judge_slug}.json"; then
      echo ">>> [retry] $rel (metrics.${judge_slug}.json is INCOMPLETE — finishing its unjudged items)"
      retries+=("$pred")
      continue
    fi
    echo ">>> [skip] $rel (metrics.${judge_slug}.json exists; FORCE=1 to redo)"
    skipped=$((skipped + 1))
    continue
  fi
  # FORCE re-judge under --batch: drop any stale batch state so submit starts a
  # fresh job instead of resuming a previous (now mismatched) one.
  [[ "$BATCH" == "1" && "$FORCE" == "1" ]] && rm -f "$dir/_batch_judge_${judge_slug}/judge_state.json" \
                                                    "$dir/_batch_judge_${judge_slug}/judge_input.jsonl"
  preds+=("$pred")
done < <(find "$ANSWERS_ROOT" -name predictions.jsonl | sort)

if [[ "$LIST_ONLY" == "1" ]]; then
  echo "  ($listed cell(s) would be judged; INCLUDE_ORACLE=$INCLUDE_ORACLE ORACLE_ONLY=$ORACLE_ONLY FORCE=$FORCE)"
  exit 0
fi

# ---- Dispatch ----
if [[ ${#preds[@]} -eq 0 && ${#retries[@]} -eq 0 ]]; then
  if [[ $skipped -eq 0 ]]; then
    echo "No predictions.jsonl found under $ANSWERS_ROOT." >&2
    exit 1
  fi
  echo "Nothing to judge ($skipped already scored; FORCE=1 to redo)."
  exit 0
fi

if [[ ${#preds[@]} -eq 0 ]]; then
  echo ">>> Nothing new to judge; ${#retries[@]} incomplete file(s) to finish."
elif [[ "$BATCH" == "1" ]]; then
  # Concurrent: submit EVERY file's batch first (non-blocking), so the jobs run
  # in parallel on OpenAI's side, then collect them all (each collect blocks on
  # its own job).
  echo ">>> Submitting ${#preds[@]} batch job(s)..."
  for pred in "${preds[@]}"; do
    run_judge_file "$pred" --batch --batch-phase submit || { echo "!!! [submit] FAILED: $pred" >&2; fail=1; }
  done
  echo ">>> Collecting ${#preds[@]} batch job(s)..."
  for pred in "${preds[@]}"; do
    echo "==================================================================="
    echo ">>> [collect] ${pred#"$ANSWERS_ROOT/"}"
    if run_judge_file "$pred" --batch --batch-phase collect --batch-poll-interval "$BATCH_POLL_INTERVAL"; then
      judged=$((judged + 1))
    else
      echo "!!! [collect] FAILED: $pred" >&2
      fail=1
    fi
  done
else
  for pred in "${preds[@]}"; do
    echo "==================================================================="
    echo ">>> [judge] ${pred#"$ANSWERS_ROOT/"}"
    if run_judge_file "$pred"; then
      judged=$((judged + 1))
    else
      echo "!!! [judge] FAILED: ${pred#"$ANSWERS_ROOT/"}" >&2
      fail=1
    fi
  done
fi

# ---- Finish the unjudged items -------------------------------------------
# Two sources feed this: files that were ALREADY incomplete when this run
# started (queued above), and files this run just judged that came back with
# failures. Both are handled the same way — re-judge only the items with no
# verdict, live, and rewrite the metrics complete.
#
# Checked after the fact rather than trusted from the exit code, because a batch
# whose lines errored still exits 0: it did its job, some lines just have no
# answer in them. The metrics file's own `complete` flag is the only thing that
# knows, which is exactly why it exists.
if [[ "$RETRY_FAILED" == "1" ]]; then
  for pred in "${preds[@]+"${preds[@]}"}"; do
    dir=$(dirname "$pred")
    [[ -f "$dir/metrics.${judge_slug}.json" ]] || continue
    metrics_complete "$dir/metrics.${judge_slug}.json" || retries+=("$pred")
  done
fi
if [[ ${#retries[@]} -gt 0 ]]; then
  echo "==================================================================="
  echo ">>> Finishing ${#retries[@]} file(s) with unjudged items (live, not batched)"
  for pred in "${retries[@]}"; do
    echo "-------------------------------------------------------------------"
    echo ">>> [retry-failed] ${pred#"$ANSWERS_ROOT/"}"
    # No --batch here whatever BATCH is: a handful of stragglers is not worth a
    # second Batch queue wait.
    if run_judge_file "$pred" --retry-failed; then
      retried=$((retried + 1))
    else
      echo "!!! [retry-failed] FAILED: ${pred#"$ANSWERS_ROOT/"}" >&2
      fail=1
    fi
  done
fi

# ---- Report anything STILL incomplete -------------------------------------
# The last word, because a metrics file that is still short of verdicts after a
# retry is one whose accuracy must not be quoted — and unlike the per-file
# output, this survives at the bottom of the log where it will be read.
incomplete=()
while IFS= read -r m; do
  metrics_complete "$m" || incomplete+=("$m")
done < <(find "$ANSWERS_ROOT" -name "metrics.${judge_slug}.json" | sort)

echo "==================================================================="
echo "Judged $judged, skipped $skipped, finished $retried. Logs in $LOG_DIR/."
if [[ ${#incomplete[@]} -gt 0 ]]; then
  echo "!!! ${#incomplete[@]} metrics file(s) STILL have unjudged items — their accuracy" >&2
  echo "!!!   is computed over fewer questions than the row claims. Do not table them" >&2
  echo "!!!   without saying so. Reasons are in judge_failures.${judge_slug}.jsonl beside each:" >&2
  for m in "${incomplete[@]}"; do echo "!!!     ${m#"$ANSWERS_ROOT/"}" >&2; done
  fail=1
fi
[[ $fail -ne 0 ]] && { echo "Some sets failed — check $LOG_DIR/." >&2; exit 1; }
exit 0
