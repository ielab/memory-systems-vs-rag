#!/bin/bash
# THE ORACLE UPPER BOUND, per memory system — answering from the units the
# labeler judged relevant, with no retrieval in the way.
#
# What it measures, and why it is not the same row as the naive-RAG oracle:
# the gold set is drawn from THAT SYSTEM'S OWN corpus, i.e. the memory units its
# ingest produced. So the number is the ceiling of the system's MEMORY
# REPRESENTATION, not of retrieval — if ingestion never wrote the answer-bearing
# fact, the oracle cannot recover it and the cell scores wrong with a perfect
# retriever. An oracle well below 100% is therefore a finding about ingest, and
# it is the denominator every retriever row in that system should be read
# against.
#
# WHY THIS IS ITS OWN JOB. Oracle needs no retrieval run at all — its facts come
# from qrels/test.tsv, not from a run.tsv (pipeline/answer.py, `if oracle:`) —
# so it does not belong in scripts/native_grid.sh, which exists to
# replay saved rankings and has no oracle concept. It needs only the qrels the
# query stage already built, plus a generator.
#
# The gold set is NOT comparable in size across systems, and that is the point
# to carry into the table. Measured on outputs/_smoke4 (4 samples, cl100k):
#
#   system      gold units (med)   rendered tokens (med / max)
#   a_mem                      1        355 /   941
#   memorybank                 2        865 / 1,396
#   mem0                       3         73 /   170
#   lightmem                   3         97 /   114
#   hypermem                  19      1,644 / 1,937
#   simplemem                 21      1,274 / 1,756
#
# 1 to 21 units and 59 to 1,644 tokens — a "unit" means something different in
# each system, which is the same caveat that applies to
# cross-system recall. Read the oracle WITHIN a system, as its ceiling; a
# cross-system oracle column is comparing different amounts of evidence.
#
# NO BUDGET APPLIES. --context-budget is refused for oracle (and for full):
# both are deliberately uncapped conditions, and truncating a gold set by an
# order that means nothing is an upper bound that is not one. The cut that does
# apply is --context-window, the serving limit, which at 131072 never fires here.
#
# The prompt is the UNIFIED one — the same shared_answer_prompt and the same
# --context-window as every other row — so the oracle sits on one axis with the
# retriever cells, the naive-RAG cells and the full-context row.
#
# Run from the repo root:
#   OUT_ROOT=outputs/_smoke4 bash scripts/oracle.sh
#   bash scripts/oracle.sh                         # the keyed tree, full dataset
#
# Knobs:
#   SYSTEMS           space-separated; default is the six memory systems. The
#                     raw-turn baselines are NOT here — their oracle already
#                     comes with ORACLE_ONLY=1 experiments/run_all_answers.sh,
#                     and running it twice would just re-answer the same cell.
#   OUT_ROOT=""       EMPTY (default) = the keyed outputs/_qrels tree, writing to
#                     outputs/_answers. Non-empty must match what the query
#                     stage was run with, and moves inputs AND outputs:
#                     ${OUT_ROOT}_qrels -> ${OUT_ROOT}_answers.
#   (no LIMIT)        THE SAMPLE COUNT IS WHATEVER THE QRELS TREE HOLDS.
#                     experiments/run_all_answers.sh passes no --limit, and the
#                     answer stage iterates the sample dirs under --qrels-dir —
#                     so a qrels tree built with --limit 4 answers four samples
#                     and there is nothing to set here. Point OUT_ROOT at the
#                     smoke tree to get a smoke run; that IS the limit.
#   GEN_MODEL         defaults to MEM_MODEL (the one served endpoint)
#   plus everything scripts/lib/vllm.sh reads (CTX_LONG, GPU_UTIL_*, DRY_RUN,
#   SKIP_SERVER, VLLM_PORT, CONDA_ENV, ...)
#
# GPU: one card, generator only. No memory system is constructed (the oracle is
# a corpus + a qrels file) and no embedder is served.
#

# No `set -e`: one system failing must not skip the other five.
set -uo pipefail

SYSTEMS=${SYSTEMS:-"lightmem a_mem mem0 memorybank simplemem hypermem"}
OUT_ROOT=${OUT_ROOT:-}

REPO_DIR=${REPO_DIR:-$PWD}
cd "$REPO_DIR" || { echo "!!! repo dir not found: $REPO_DIR" >&2; exit 1; }
source "$REPO_DIR/scripts/lib/vllm.sh"
mkdir -p logs/answer logs/vllm logs/answers

JOB=${JOB:-$$}
GEN_MODEL=${GEN_MODEL:-$MEM_MODEL}
# One window for the whole job, and the long one: the generator is the only
# model on this card, and 131072 is what every other answer row is served at, so
# --context-window means the same number across the table. It cannot bind here
# anyway — the largest oracle prompt measured is ~1,937 tokens.
CTX=${MAX_MODEL_LEN:-$CTX_LONG}

echo "==================================================================="
echo "Oracle answering            job $JOB on $(hostname)"
echo "Systems    : $SYSTEMS"
echo "Qrels root : ${OUT_ROOT:-none (keyed default: outputs/_qrels)}"
echo "Answers    : $([[ -n "$OUT_ROOT" ]] && echo "${OUT_ROOT}_answers" || echo "outputs/_answers (keyed)")"
echo "Generator  : $GEN_MODEL @ max-model-len $CTX (--context-window $CTX)"
echo "Samples    : however many the qrels tree holds (no --limit on this path)"
echo "==================================================================="

activate_python

# Check every corpus BEFORE starting a server that takes 10-25 minutes to come
# up.
qrels_of() {
  if [[ -n "$OUT_ROOT" ]]; then echo "${OUT_ROOT}_qrels/$1"; return; fi
  python - "$1" "$SPEC_NAME" <<'PYEOF'
import sys
from evalgrid.lib.paths import ExperimentPaths, labeler_segment
# mem_model is read from the spec by the answer stage itself; here we only need
# the tree, and the query stage keys it by the same mem-model as the spec.
import yaml
spec = yaml.safe_load(open(f"experiments/{sys.argv[1]}/{sys.argv[2]}"))
print(ExperimentPaths(system=sys.argv[1], dataset=spec["dataset"],
                      mem_model=spec["mem_model"]).qrels(labeler_segment("has_answer")))
PYEOF
}

missing=0
for s in $SYSTEMS; do
  [[ -f "$(spec_of "$s")" ]] || {
    echo "!!! no spec for '$s'" >&2; missing=1; continue; }
  d=$(qrels_of "$s")
  [[ -d "$d" ]] || { echo "!!! no qrels for $s at $d" >&2; missing=1; }
done
if [[ $missing -ne 0 ]]; then
  echo "!!! build them first — the query stage writes qrels on every run:" >&2
  echo "!!!   SYSTEMS=<sys> STAGE=retrieve MEM_ROOT=${OUT_ROOT:-} bash scripts/query.sh" >&2
  exit 1
fi

if [[ "${SKIP_SERVER:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  MEM_MODEL=$GEN_MODEL
  start_server "$CTX" || { echo "!!! generator never came up" >&2; exit 1; }
  trap stop_server EXIT
fi

export GEN_MODEL
export GEN_BASE_URL="http://localhost:${VLLM_PORT}/v1"
# The served window and the flag are one number by construction.
export CONTEXT_WINDOW="$CTX"
export ORACLE_ONLY=1

fail=0
declare -a RESULTS=()
for s in $SYSTEMS; do
  export SPEC="$REPO_DIR/$(spec_of "$s")"
  # Inputs and outputs both follow OUT_ROOT, so a smoke build stays out of the
  # cell a full run will later fill (the answer stage resumes by question_id, so
  # a smoke that landed in the keyed tree would be silently kept).
  export QRELS_DIR=$([[ -n "$OUT_ROOT" ]] && echo "${OUT_ROOT}_qrels/$s" || echo "")
  export OUT_ANSWERS_ROOT=$([[ -n "$OUT_ROOT" ]] && echo "${OUT_ROOT}_answers" || echo "")
  echo "==================================================================="
  echo "### oracle — $s"
  start=$SECONDS
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "(dry run) would run: SPEC=$SPEC ORACLE_ONLY=1 QRELS_DIR=${QRELS_DIR:-<keyed>} bash experiments/run_all_answers.sh"
    rc=0
  else
    bash experiments/run_all_answers.sh; rc=$?
  fi
  elapsed=$((SECONDS - start))
  if (( rc == 0 )); then RESULTS+=("OK|$s|${elapsed}s")
  else RESULTS+=("FAIL($rc)|$s|${elapsed}s"); fail=1; fi
done

echo "==================================================================="
echo "Oracle summary (job $JOB)"
printf '  %s\n' "${RESULTS[@]+"${RESULTS[@]}"}"
echo "Score them with:"
for s in $SYSTEMS; do
  echo "  ${OUT_ROOT:+ANSWERS_ROOT=${OUT_ROOT}_answers/$s }SPEC=$(spec_of "$s") bash experiments/run_all_judge.sh"
done
echo "==================================================================="
exit $fail
