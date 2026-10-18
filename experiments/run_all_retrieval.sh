#!/usr/bin/env bash
# Run + eval every retriever in the --retrievers string, one entry at a time.
#
# Usage:
#   bash experiments/run_all_retrieval.sh           # sequential
#   PARALLEL=1 bash experiments/run_all_retrieval.sh # dense runs in parallel (separate endpoints)
#   EVAL=0 bash experiments/run_all_retrieval.sh      # skip the eval pass
#
# Note: we do NOT use `set -e` so one failing retriever doesn't kill the rest.
set -uo pipefail

# ---- Shared config from the experiment spec (SYSTEM/DATASET/MEM_MODEL/...) ----
# Override with SPEC=path to run a different experiment without editing the
# default (e.g. SPEC=experiments/user_turns/longmemeval.yaml bash experiments/run_all_retrieval.sh).
SPEC="${SPEC:-$(dirname "$0")/lightmem/longmemeval.yaml}"
source "$(dirname "$0")/load_spec.sh"

# ---- run/eval-only knobs ----
K_VALUES=${K_VALUES:-3,5,10,20}

# Space-separated question_types to skip in both run + eval. Empty = keep all,
# which is the default and what every published LongMemEval table does.
#
# This used to default to "single-session-assistant" on the belief that the
# LightMem study excluded those 56 questions. It does not: its Table 7 scores
# every method on them (LightMem 32.14, A-MEM 96.43, Naive RAG 98.21). Excluding
# them moves LightMem +4.6 and A-Mem -4.3 — an 8.9-point swing between two rows
# of the same table. Set it only for a deliberately labelled subset analysis.
EXCLUDE_QUESTION_TYPES=${EXCLUDE_QUESTION_TYPES:-""}

PARALLEL=${PARALLEL:-0}   # 1 = run dense retrievers concurrently
EVAL=${EVAL:-1}           # 1 = run eval_retrieval.py after each run
# Which evidence set a ranking is scored against (evalgrid/benchmarks/gold.py).
# "turn" is the evaluation default: "strict" scores the 94 evidence sessions
# LongMemEval never flags as distractors, and for 21 questions leaves the gold
# set EMPTY (recall undefined, not zero). Each level writes its own metrics file
# — metrics.json at strict, metrics-gold-<level>.json otherwise — so levels
# never overwrite each other and a ranking is never re-run to change one.
GOLD=${GOLD:-turn}

LOG_DIR=logs/retrieval
mkdir -p "$LOG_DIR"

# ---- Parse one spec into the retrieval flags ----
# Echoes the --retriever ... flags for a single comma-separated entry.
build_flags() {
  local spec="$1"
  if [[ "$spec" == "bm25" ]]; then
    printf -- '--retriever bm25'
  elif [[ "$spec" == dense:* ]]; then
    local rest="${spec#dense:}"
    local emb_model="${rest%%@*}"
    if [[ "$rest" == *@* ]]; then
      printf -- '--retriever dense --emb-model %s --emb-base-url %s' "$emb_model" "${rest#*@}"
    else
      printf -- '--retriever dense --emb-model %s' "$emb_model"
    fi
  elif [[ "$spec" == fusion:* ]]; then
    # Hybrid BM25 + dense; same MODEL[@url] syntax as dense (BM25 needs no server).
    local rest="${spec#fusion:}"
    local emb_model="${rest%%@*}"
    if [[ "$rest" == *@* ]]; then
      printf -- '--retriever fusion --emb-model %s --emb-base-url %s' "$emb_model" "${rest#*@}"
    else
      printf -- '--retriever fusion --emb-model %s' "$emb_model"
    fi
  elif [[ "$spec" == rrf:* ]]; then
    # BM25 + dense fused by reciprocal rank fusion; same MODEL[@url] syntax as
    # fusion:, and the same two legs — a different combiner and its own run dir.
    local rest="${spec#rrf:}"
    local emb_model="${rest%%@*}"
    if [[ "$rest" == *@* ]]; then
      printf -- '--retriever rrf --emb-model %s --emb-base-url %s' "$emb_model" "${rest#*@}"
    else
      printf -- '--retriever rrf --emb-model %s' "$emb_model"
    fi
  elif [[ "$spec" == sparse:* ]]; then
    # Learned sparse (naver/splade-v3, promptreps:<llm>) — loads locally, no @url.
    printf -- '--retriever sparse --emb-model %s' "${spec#sparse:}"
  elif [[ "$spec" == hybrid:* ]]; then
    # Dense + learned sparse fusion (PromptReps-style); needs a
    # dual-representation model (promptreps:<llm>) — local, no @url.
    printf -- '--retriever hybrid --emb-model %s' "${spec#hybrid:}"
  else
    echo "!!! Unknown retriever spec: $spec" >&2
    return 1
  fi
}

# ---- Run (and optionally eval) a single retriever ----
run_one() {
  local spec="$1"
  local flags safe_name
  flags=$(build_flags "$spec") || return 1
  safe_name=$(echo "$spec" | tr '/:@.' '____')

  # One --exclude-question-type flag per type in EXCLUDE_QUESTION_TYPES.
  local exclude_flags=""
  for qt in $EXCLUDE_QUESTION_TYPES; do
    exclude_flags+=" --exclude-question-type $qt"
  done

  # EVAL_ONLY=1 skips retrieval and only (re)scores the existing run.tsv — no
  # embedding servers needed, since eval reads run.tsv + qrels (not the model).
  if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
    echo ">>> [run] $spec"
    python -m evalgrid.pipeline.retrieve \
      --system "$SYSTEM" --dataset "$DATASET" \
      --mem-model "$MEM_MODEL" --mem-variant "$MEM_VARIANT" \
      --labeler-model "$LABELER_MODEL" \
      --candidate-source "$CANDIDATE_SOURCE" \
      $exclude_flags \
      $flags 2>&1 | tee "$LOG_DIR/run_${safe_name}.log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      echo "!!! [run] FAILED ($rc): $spec" >&2
      return $rc
    fi
  fi

  if [[ "$EVAL" == "1" || "${EVAL_ONLY:-0}" == "1" ]]; then
    # eval_retrieval.py takes --emb-model but not --emb-base-url
    local eval_flags="${flags%% --emb-base-url*}"
    echo ">>> [eval] $spec"
    python -m evalgrid.pipeline.eval_retrieval \
      --system "$SYSTEM" --dataset "$DATASET" \
      --mem-model "$MEM_MODEL" --mem-variant "$MEM_VARIANT" \
      --labeler-model "$LABELER_MODEL" \
      --candidate-source "$CANDIDATE_SOURCE" \
      --k-values "$K_VALUES" --gold "$GOLD" \
      $eval_flags 2>&1 | tee "$LOG_DIR/eval_${safe_name}.log"
  fi
}

# ---- Dispatch ----
IFS=',' read -ra SPECS <<< "$RETRIEVERS"
pids=()
for spec in "${SPECS[@]}"; do
  echo "==================================================================="
  if [[ "$PARALLEL" == "1" && ( "$spec" == dense:*@* || "$spec" == fusion:*@* ) ]]; then
    # only the vLLM-endpoint (dense/fusion) models are safe to parallelize
    run_one "$spec" &
    pids+=("$!")
  else
    run_one "$spec"
  fi
done

# wait for any backgrounded parallel runs
fail=0
for pid in "${pids[@]:-}"; do
  [[ -z "$pid" ]] && continue
  wait "$pid" || fail=1
done

echo "==================================================================="
if [[ $fail -ne 0 ]]; then
  echo "Done, but at least one parallel run failed — check $LOG_DIR/." >&2
else
  echo "All retrievers done. Logs in $LOG_DIR/."
fi
