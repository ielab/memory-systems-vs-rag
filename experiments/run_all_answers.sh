#!/usr/bin/env bash
# Generate answers for every retriever's top-k retrieval run, plus oracle.
# Reads the same experiment spec as run_all_retrieval.sh so the
# labeler segment lines up and answers read the run.tsv those scripts wrote.
# Score the results separately with experiments/run_all_judge.sh.
#
# Usage:
#   bash experiments/run_all_answers.sh
#   TOP_K_VALUES=5,10,20 bash experiments/run_all_answers.sh   # sweep top-k
#   CONTEXT_BUDGETS=250,500,1000 bash experiments/run_all_answers.sh  # sweep token budgets instead
#   INCLUDE_ORACLE=0 bash experiments/run_all_answers.sh       # skip the oracle upper bound
#   INCLUDE_FULL=1 bash experiments/run_all_answers.sh         # also run the full-context baseline (whole corpus)
#   ORACLE_ONLY=1 bash experiments/run_all_answers.sh          # ONLY the oracle upper bound (no retrieval needed)
#   CONTEXT_WINDOW=65536 bash experiments/run_all_answers.sh   # generator is served at 64k, cut prompts to fit
#
# CONTEXT_BUDGETS matches systems on EVIDENCE VOLUME rather than depth, which is
# the axis that survives a change of memory unit: equal k is not equal evidence
# when one system's unit is twice another's. When set it REPLACES the top-k
# sweep over the retriever grid (oracle/full are uncapped either way and still
# run once). For both grids, run the script twice — they write to separate
# directories (budget<T>/ vs top<k>/), so neither overwrites the other.
#
# Note: no `set -e` so one failing retriever doesn't kill the rest. Every answer
# hits the single generator endpoint, so runs are sequential (no PARALLEL knob).
set -uo pipefail

# ---- Shared config from the experiment spec (SYSTEM/DATASET/MEM_MODEL/...) ----
# Override with SPEC=path to answer a different experiment without editing the
# default (e.g. SPEC=experiments/user_turns/longmemeval.yaml bash experiments/run_all_answers.sh).
SPEC="${SPEC:-$(dirname "$0")/lightmem/longmemeval.yaml}"
source "$(dirname "$0")/load_spec.sh"

# ---- answer-only knobs ----
# GEN_MODEL comes from the spec (gen_model:) via load_spec.sh, overridable with
# GEN_MODEL=... ; the literal below is only a fallback for a spec without it.
GEN_MODEL=${GEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
GEN_BASE_URL=${GEN_BASE_URL:-http://localhost:8000/v1}
TOP_K_VALUES=${TOP_K_VALUES:-10}          # comma-separated; oracle/full ignore k
CONTEXT_BUDGETS=${CONTEXT_BUDGETS:-}      # comma-separated token budgets; empty = depth sweep only
INCLUDE_ORACLE=${INCLUDE_ORACLE:-1}       # 1 = also answer from gold qrels (upper bound)
# WHICH evidence set the oracle answers from (evalgrid/benchmarks/gold.py).
# "turn" is the oracle default: "strict" leaves the 21 abstention questions with
# an EMPTY prompt and drops a hop from 32 multi-hop chains, so it is not an upper
# bound on this benchmark. Every level writes its own "-gold-<level>" path, so
# levels never overwrite each other and can sit side by side in one table.
GOLD=${GOLD:-turn}
INCLUDE_FULL=${INCLUDE_FULL:-0}           # 1 = also answer from the WHOLE corpus (full-context baseline, no retrieval)
FULL_HISTORY_TOKENS=${FULL_HISTORY_TOKENS-100000}   # full-context history cap, newest turns kept (paper: 100K)
FULL_ONLY=${FULL_ONLY:-0}                 # 1 = ONLY the full-context baseline: skip the retriever sweep and oracle
# ORACLE_ONLY is FULL_ONLY's counterpart: the perfect-retrieval upper bound and
# nothing else. It exists because the oracle is the one cell that needs NO
# retrieval run — its facts come straight from the labeler's qrels — so it can
# be run for a system whose retrieval grid has not been answered, and re-run
# when the labeler changes without touching the retriever cells.
ORACLE_ONLY=${ORACLE_ONLY:-0}             # 1 = ONLY the oracle: skip the retriever sweep and full context
if [[ "$FULL_ONLY" == "1" && "$ORACLE_ONLY" == "1" ]]; then
  # They are mutually exclusive by construction: each says "only me". Refusing
  # is better than picking one, because the loser would be silently absent from
  # a table that looks complete.
  echo "!!! FULL_ONLY and ORACLE_ONLY both set — each means 'only this row'. Pick one." >&2
  exit 1
fi
# Each implies its own row is on and nothing else runs.
[[ "$FULL_ONLY" == "1" ]] && { INCLUDE_FULL=1; INCLUDE_ORACLE=0; }
[[ "$ORACLE_ONLY" == "1" ]] && { INCLUDE_ORACLE=1; INCLUDE_FULL=0; }

# Generation knobs — pinned here (not left to run_answer.py defaults) so the
# generator config is reproducible from this file, mirroring run_all_judge.sh.
# GPT-5/o-series-only knobs (reasoning-effort, verbosity) are ignored by other
# models; empty = omit the flag.
GEN_SEED=${GEN_SEED:-42}                   # best-effort reproducibility (matches judge)
GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.0}    # sampling temperature; 0.0 = greedy (gpt-5 ignores it)
GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-2000}     # per-answer output budget
# The generator's SERVING window — the --max-model-len it is actually up at.
# Passed as --context-window so a prompt that would not fit is CUT to fit rather
# than sent and rejected with a 400 (which would silently shorten the
# denominator: a rejected question lands in errors.jsonl, not predictions.jsonl).
# It normally never binds — a top-10 prompt is a few thousand tokens. It exists
# for the full-context row, whose prompt is the whole rendered haystack:
# ~110-114k tokens for all_turns on LongMemEval-S, which clears 131072 and does
# NOT clear 65536. SET IT TO WHAT THE SERVER IS ACTUALLY SERVING — too high and
# the cut never fires when it should, too low and evidence is dropped for
# nothing. Empty = no cap (the pre-2026-08 behaviour).
CONTEXT_WINDOW=${CONTEXT_WINDOW:-131072}
GEN_REASONING_EFFORT=${GEN_REASONING_EFFORT:-}   # none|low|medium|high|xhigh (gpt-5/o only)
GEN_VERBOSITY=${GEN_VERBOSITY:-}                 # low|medium|high (gpt-5 only)

# Space-separated question_types to skip. One --exclude-question-type per type
# is passed to run_answer.py. Empty = answer all 500, which is what every
# published LongMemEval table reports — see the note in run_all_retrieval.sh for
# why the old "single-session-assistant" default was wrong.
EXCLUDE_QUESTION_TYPES=${EXCLUDE_QUESTION_TYPES:-""}

# ---- Reading a NON-DEFAULT input tree (a smoke build) ----
# Both default to empty = the keyed tree the Python stages derive from the spec.
# Set them to answer a corpus/retrieval built under a prefix such as
# outputs/_smoke4_{qrels,retrieval}/.
# Note these move only the INPUTS: predictions still land in the keyed
# outputs/_answers/ tree, which is where run_all_judge.sh and the checkers look.
QRELS_DIR=${QRELS_DIR:-}            # a <root>_qrels/<system> dir
RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-}  # a <root>_retrieval/<system> dir; the per-retriever
                                    # subdir is appended, named by paths.retriever_name

# ---- Writing to a NON-DEFAULT output tree (a smoke build) -------------------
# Empty (the default) = predictions land in the keyed outputs/_answers/ tree,
# which is where run_all_judge.sh and evalgrid.tools.check_answers look.
#
# Set it and they land in <root>_answers/<system>/<retriever>/<cut>/ instead —
# the same flattened layout scripts/query.sh writes for the memory
# systems, so a smoke build's three answer jobs (memory systems, naive RAG, full
# context) end up in ONE tree that can be read together and deleted together.
#
# THIS IS THE KNOB THAT KEEPS A 4-SAMPLE SMOKE OUT OF THE REAL TABLE. Without
# it a smoke run writes four questions into the cell a full run will later use,
# and because the answer stage resumes BY QUESTION_ID the full run keeps them:
# four answers generated against a throwaway store, silently mixed into 500.
OUT_ANSWERS_ROOT=${OUT_ANSWERS_ROOT:-}

# The retriever's run-dir name, asked of evalgrid.lib.paths rather than
# reproduced here: model_slug/retriever_name key every run.tsv on disk, and a
# bash copy of them would drift silently into "no run.tsv for this sample".
# Only called when RETRIEVAL_ROOT is set, so the default path spawns nothing.
rname_of() {
  python - "$1" "${2:-}" <<'PYEOF'
import sys
from evalgrid.lib.paths import retriever_name
print(retriever_name(sys.argv[1], sys.argv[2] or None))
PYEOF
}

LOG_DIR=logs/answers
mkdir -p "$LOG_DIR"

# Build the repeated --exclude-question-type flags once (string form is safe
# under set -u in bash 3.2, unlike empty-array expansion). Types have no spaces.
EXCLUDE_FLAGS=""
for qt in $EXCLUDE_QUESTION_TYPES; do EXCLUDE_FLAGS+=" --exclude-question-type $qt"; done

# Generation flags pinned above; the gpt-5-only knobs are appended only when set.
GEN_FLAGS="--gen-seed $GEN_SEED --gen-temperature $GEN_TEMPERATURE --max-tokens $GEN_MAX_TOKENS"
[[ -n "$CONTEXT_WINDOW" ]] && GEN_FLAGS+=" --context-window $CONTEXT_WINDOW"
# Empty = answer.py's default (8). CONCURRENCY=1 for a latency-grade rerun: at 8,
# each call's latency_sec includes sharing the server with the other seven.
[[ -n "${CONCURRENCY:-}" ]] && GEN_FLAGS+=" --concurrency $CONCURRENCY"
[[ -n "$GEN_REASONING_EFFORT" ]] && GEN_FLAGS+=" --gen-reasoning-effort $GEN_REASONING_EFFORT"
[[ -n "$GEN_VERBOSITY" ]] && GEN_FLAGS+=" --gen-verbosity $GEN_VERBOSITY"

# ---- Answer one retriever config ----
# args: <safe_name> <label> <retriever-dir-name> <cut-seg> <run_answer flags...>
answer_one() {
  local safe_name="$1"; shift
  local label="$1"; shift
  local rdir_name="$1"; shift
  local cut_seg="$1"; shift
  # --out is the predictions.jsonl PATH on this path (the file, not the dir),
  # and pipeline/answer.py takes it LITERALLY — it appends no segment of its
  # own. So the retriever and the cut both have to be spelled here, or every
  # cell of the sweep writes one file and the second cell is skipped
  # question-by-question as already answered.
  local out=()
  [[ -n "$OUT_ANSWERS_ROOT" ]] && \
    out=(--out "$OUT_ANSWERS_ROOT/$SYSTEM/$rdir_name/$cut_seg/predictions.jsonl")
  echo "==================================================================="
  echo ">>> [answer] $label${OUT_ANSWERS_ROOT:+  -> ${out[1]}}"
  python -m evalgrid.pipeline.answer \
    --system "$SYSTEM" --dataset "$DATASET" --mem-model "$MEM_MODEL" \
    --mem-variant "$MEM_VARIANT" \
    --labeler-model "$LABELER_MODEL" --candidate-source "$CANDIDATE_SOURCE" \
    --gen-model "$GEN_MODEL" --gen-base-url "$GEN_BASE_URL" \
    ${QRELS_DIR:+--qrels-dir "$QRELS_DIR"} \
    "${out[@]+"${out[@]}"}" \
    $GEN_FLAGS \
    $EXCLUDE_FLAGS \
    "$@" 2>&1 | tee "$LOG_DIR/answer_${safe_name}.log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 ]] && echo "!!! [answer] FAILED ($rc): $label" >&2
  return 0
}

# ---- Dispatch over RETRIEVERS x TOP_K_VALUES, then oracle ----
IFS=',' read -ra SPECS <<< "$RETRIEVERS"
IFS=',' read -ra KS <<< "$TOP_K_VALUES"
# Specs no branch below could dispatch. Empty is the only acceptable end state;
# see the `else` arm for what a silent one costs.
UNPARSED=()
BUDGETS=()
[[ -n "$CONTEXT_BUDGETS" ]] && IFS=',' read -ra BUDGETS <<< "$CONTEXT_BUDGETS"

for spec in "${SPECS[@]}"; do
  # Either "only" mode skips every retriever cell; both rows answer from the
  # corpus or the qrels, not from a run.tsv.
  [[ "$FULL_ONLY" == "1" || "$ORACLE_ONLY" == "1" ]] && break
  # Answering reads run.tsv and never embeds, so dense needs --emb-model (to
  # locate its run dir) but NOT --emb-base-url.
  # Cleared per iteration: bm25 sets no emb_model, and a stale one from the
  # previous spec would name the wrong run dir.
  emb_model=""
  if [[ "$spec" == "bm25" ]]; then
    rflags=(--retriever bm25); base="bm25"
  elif [[ "$spec" == dense:* ]]; then
    rest="${spec#dense:}"; emb_model="${rest%%@*}"
    rflags=(--retriever dense --emb-model "$emb_model")
    base="dense_$(echo "$emb_model" | tr '/:@.' '____')"
  elif [[ "$spec" == fusion:* ]]; then
    rest="${spec#fusion:}"; emb_model="${rest%%@*}"
    rflags=(--retriever fusion --emb-model "$emb_model")
    base="fusion_$(echo "$emb_model" | tr '/:@.' '____')"
  elif [[ "$spec" == rrf:* ]]; then
    # BM25 + dense fused by reciprocal rank fusion — same two legs as `fusion:`
    # under a different combiner, and lib/paths gives it its own `rrf-` prefix
    # so the two cannot share a run dir. Carries an @url like dense: does, and
    # it is stripped for the same reason: answering reads a run.tsv and embeds
    # nothing, so the model name is only ever used to locate the dir.
    rest="${spec#rrf:}"; emb_model="${rest%%@*}"
    rflags=(--retriever rrf --emb-model "$emb_model")
    base="rrf_$(echo "$emb_model" | tr '/:@.' '____')"
  elif [[ "$spec" == multi_vector:* ]]; then
    rest="${spec#multi_vector:}"; emb_model="${rest%%@*}"
    rflags=(--retriever multi_vector --emb-model "$emb_model")
    base="mv_$(echo "$emb_model" | tr '/:@.' '____')"
  elif [[ "$spec" == sparse:* ]]; then
    # Learned sparse (naver/splade-v3, promptreps:<llm>) — local, no @url.
    emb_model="${spec#sparse:}"
    rflags=(--retriever sparse --emb-model "$emb_model")
    base="sparse_$(echo "$emb_model" | tr '/:@.' '____')"
  elif [[ "$spec" == hybrid:* ]]; then
    # Dense + learned-sparse fusion (promptreps:<llm>) — local, no @url.
    emb_model="${spec#hybrid:}"
    rflags=(--retriever hybrid --emb-model "$emb_model")
    base="hybrid_$(echo "$emb_model" | tr '/:@.' '____')"
  else
    # NOT just a warning. A spec this loop cannot parse is a column that never
    # existed — no directory, no log, nothing for check_grid.py to call PARTIAL
    # — and the sweep used to finish by printing "All answers done" over the
    # hole. That is how the raw-turn baselines ended up with five answered
    # retrievers against eight retrieved ones: `rrf:` had no branch here while
    # lib/paths, lib/cli and pipeline/answer all supported it, so the three rrf
    # columns were dropped on stderr in the middle of a job log nobody re-reads.
    # Recorded and re-reported at the end, where it is still on screen.
    echo "!!! Unknown retriever spec: $spec" >&2
    UNPARSED+=("$spec"); continue
  fi
  # rflags[1] is the --retriever value set just above.
  #
  # ONE derivation of the run-dir name, used for both the input override and the
  # output path — asked of evalgrid.lib.paths so the two cannot disagree, and so
  # an answers dir is always named after the run.tsv it read. Derived only when
  # something actually needs it (each call spawns a python), which is why this
  # is a conditional and not an unconditional assignment.
  rname=""
  [[ -n "$RETRIEVAL_ROOT$OUT_ANSWERS_ROOT" ]] && rname=$(rname_of "${rflags[1]}" "$emb_model")
  rdir=()
  [[ -n "$RETRIEVAL_ROOT" ]] && rdir=(--retrieval-dir "$RETRIEVAL_ROOT/$rname")
  rflags+=("${rdir[@]+"${rdir[@]}"}")
  if [[ -n "$CONTEXT_BUDGETS" ]]; then
    # Budget sweep: --top-k is ignored, and each budget writes to budget<T>/ so
    # it cannot collide with the depth grid. Run the script twice to get both.
    for t in "${BUDGETS[@]}"; do
      answer_one "${base}_budget${t}" "${spec}@budget${t}" "$rname" "budget${t}" "${rflags[@]}" --context-budget "$t"
    done
  else
    for k in "${KS[@]}"; do
      answer_one "${base}_top${k}" "${spec}@top${k}" "$rname" "top${k}" "${rflags[@]}" --top-k "$k"
    done
  fi
done

if [[ "$INCLUDE_ORACLE" == "1" ]]; then
  # Oracle answers from gold qrels; top-k is irrelevant (path segment "all").
  #
  # ORACLE IS THE ONE SET KEYED BY THE LABELER, and the <labeler-segment> level
  # has to be here for the same reason evalgrid.lib.paths.answers_dir puts it
  # there: its facts ARE the labeler's output, so two labelers produce two
  # different oracle rows and must not share a predictions.jsonl. Retriever
  # answers carry no such level — they depend on the run.tsv and the corpus, not
  # on who labelled it.
  #
  # It is also load-bearing for JUDGING. experiments/run_all_judge.sh filters
  # oracle sets to this experiment's labeler with
  #   [[ $pred == $ANSWERS_ROOT/oracle/* && $pred != $ANSWERS_ROOT/oracle/$labeler_seg/* ]] && continue
  # so an oracle written to oracle/all/ instead of oracle/<labeler>/all/ matches
  # the first test, fails the second, and is SILENTLY SKIPPED — the upper bound
  # missing from the table with nothing saying why.
  #
  # The gold level is part of that key for the SAME reason, and must be appended
  # HERE as well: pipeline/answer.py adds gold_suffix() to the path it computes,
  # so an oracle at --gold turn lands in oracle/<labeler>-gold-turn/. Leaving it
  # off here would make the OUT_ANSWERS_ROOT branch below write a turn-gold
  # oracle over a strict one at the shared path, and would desync this script's
  # bookkeeping from the directory answer.py actually wrote. Mirrors
  # evalgrid.benchmarks.gold.gold_suffix: empty for strict, "-gold-<level>" else.
  oracle_seg=$(echo "$LABELER_MODEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')
  [[ "$CANDIDATE_SOURCE" == "retrieval" ]] && oracle_seg="${oracle_seg}-retrieval"
  # Unconditional: no level holds an empty suffix any more, so every oracle —
  # strict included — is named by the evidence set it answered from.
  oracle_seg="${oracle_seg}-gold-${GOLD}"
  answer_one "oracle" "oracle" "oracle/$oracle_seg" "all" --retriever oracle --gold "$GOLD"
fi

if [[ "$INCLUDE_FULL" == "1" ]]; then
  # Full-context baseline: answer from the ENTIRE corpus, no retrieval; top-k is
  # irrelevant (path segment "all"). The prompt is the whole corpus, so the
  # generator needs a large enough context window (--max-model-len) — this is the
  # one row CONTEXT_WINDOW exists for. If it binds, the run says so on its last
  # line and answer_summary.json carries the count: a cut full-context row is not
  # "the whole haystack in the window" and must be reported as truncated.
  # FULL_HISTORY_TOKENS: the paper caps the supplied history at 100K tokens and
  # keeps the most recent turns that fit. Empty = no cap beyond CONTEXT_WINDOW.
  answer_one "full" "full" "full" "all" --retriever full \
    ${FULL_HISTORY_TOKENS:+--max-history-tokens "$FULL_HISTORY_TOKENS"}
fi

echo "==================================================================="
if (( ${#UNPARSED[@]} )); then
  echo "!!! ${#UNPARSED[@]} retriever spec(s) were NOT answered — this dispatch has no" >&2
  echo "!!! branch for them, so they produced no cell at all:" >&2
  printf '!!!   %s\n' "${UNPARSED[@]}" >&2
  echo "!!! Add a branch above, or drop them from RETRIEVERS so the sweep's" >&2
  echo "!!! shape matches what it claims to cover." >&2
  exit 1
fi
echo "All answers done. Logs in $LOG_DIR/. Score them with experiments/run_all_judge.sh."
