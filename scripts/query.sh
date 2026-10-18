#!/bin/bash
# The QUERY path — retrieval and/or answering — over memory that
# scripts/ingest.sh built.
#
# One engine for both halves because they share everything that matters: which
# store to read, which context window that system needs, how to start vLLM, how
# to name output dirs so the two halves line up. STAGE picks which half runs.
#
# Runs, per system, the three stages that turn a saved store into scoreable
# predictions:
#
#   1. qrels_has_answer   CPU. The gold set: every saved unit is a corpus doc,
#                         the ones carrying has_answer are relevant. Cheap and
#                         deterministic, so it is rebuilt each run rather than
#                         assumed.
#   2. the answer stage   GPU. MODE picks which:
#                           native — the system's OWN retrieval loop and its own
#                                    answer prompt (--prompt native). Emits BOTH
#                                    predictions and the run.tsv of unit ids its
#                                    retrieval touched, in one pass.
#                           bm25   — retrieve then answer with the SHARED
#                           dense    retriever and the shared prompt, which is
#                                    what holds everything but the memory
#                                    representation fixed.
#   3. eval_retrieval     CPU. Scores whichever run.tsv step 2 wrote against the
#                         step 1 qrels, so native and bm25 land on one axis.
#
# JUDGING IS NOT HERE. evalgrid.pipeline.judge calls an external API (gpt-4o-mini),
# needs no GPU, and would hold the GPU idle while it waited on rate limits. Run
# it afterwards with experiments/run_all_judge.sh.
#
# Run from the repo root, one or more systems:
#
#   SYSTEMS=a_mem bash scripts/query.sh                       # native, 3 samples
#   SYSTEMS=a_mem LIMIT= bash scripts/query.sh                # native, full dataset
#   SYSTEMS=a_mem LIMIT= STAGE=retrieve bash scripts/query.sh # retrieval only
#
# Knobs (environment variables):
#   MODE=native           native | bm25 | dense
#   SYSTEMS="..."         space-separated; default is all six
#   SPEC_NAME=longmemeval.yaml
#                         WHICH EXPERIMENT SPEC, by basename — the specs are
#                         per-system, so this picks experiments/<system>/<name>.
#                         DATASET is then read from that spec, so this one knob
#                         moves the whole run (memory, qrels, retrieval, answers)
#                         onto another benchmark:
#                           SPEC_NAME=beam_100k.yaml SYSTEMS=lightmem bash scripts/query.sh
#                         Setting DATASET to something the spec does not name is
#                         refused — five stages here pass only --exp, so the two
#                         must agree or the run splits across two trees.
#   MEM_ROOT=""           where the stores ARE. Empty = the keyed default
#                         outputs/_memory/<system>/<dataset>/<mem-seg>/. Set it
#                         to whatever ingest used: a run over MEM_ROOT stores
#                         MUST pass the same value here or it will silently
#                         answer from a different (or absent) store.
#   OUT_ROOT=outputs      root for _qrels / _retrieval / _answers. Under
#                         MEM_ROOT this defaults to MEM_ROOT's parent so a
#                         throwaway ingest keeps its scoring beside it.
#   LIMIT=""              samples per system; empty = the whole dataset
#   TOP_K=10              answer depth for bm25/dense (shared path)
#   EMB_SERVE=auto        auto = if NATIVE_RETRIEVER names a localhost embedder
#                         (…@http://localhost:8002/v1), serve that model on that
#                         port IN THIS JOB, at EMB_UTIL (0.1) of the GPU. It has
#                         to be this job: "localhost" is this
#                         machine. 0 = do not, you are serving it yourself.
#   NATIVE_RETRIEVER=""   MODE=native: swap the retriever INSIDE the system's own
#                         loop. Same spec as --native-retriever: bm25 |
#                         dense:MODEL[@url] | sparse:MODEL | rrf:MODEL[@url] |
#                         fusion:MODEL[@url] | hybrid:MODEL | multi_vector:MODEL.
#                         Empty = the system retrieves exactly as published, and
#                         every path below is byte-identical to what it was
#                         before this knob existed.
#                         A served embedder (@url) must be started SEPARATELY —
#                         this script manages one endpoint, the memory/generator.
#   NATIVE_VARIANT=""     names the setting in the run dir. Use `grid` for a
#                         retriever sweep: for a_mem/lightmem/mem0/memorybank the
#                         MiniLM row IS their published encoder (Qwen3-0.6B for
#                         simplemem), so without a variant that one cell derives
#                         the published run's directory and is then SKIPPED as
#                         already-done, silently reporting the published run.
#   CONFIG=""             --config passed to the native stages. REQUIRED for a
#                         hypermem swap: evalgrid/configs/hypermem/grid.yaml sets
#                         retrieval_type: keyword (without it the swap is refused)
#                         and reranker.enabled: false.
#   NATIVE_TOP_K=auto     RETRIEVAL depth for MODE=native. DEFAULTS TO "auto" —
#                         each spec's native_top_k, deep enough that the saved
#                         retrieval can serve the answer sweep. Set it EMPTY
#                         (NATIVE_TOP_K=) for each config's published depth, the
#                         "as published" control, or to a number for one depth
#                         everywhere. A deep run writes to
#                         <retrieval>/native-top<k>/, never over the control.
#   PROMPT=shared         WHICH ANSWER PROMPT (STAGE=answer, MODE=native).
#                         shared (default) = the UNIFIED prompt every system and
#                         every raw-turn baseline answers under; native = the
#                         system's own, a labelled control that lands under a
#                         "--native-prompt" path suffix. The two are different
#                         code paths over the SAME saved run.tsv — see the block
#                         beside FROM_RETRIEVAL below.
#   CONCURRENCY=8         PROMPT=shared only: concurrent generator calls. That
#                         path makes no query-time memory call, so questions are
#                         independent; it does not shard (NUM_SHARDS>1 is
#                         refused) and parallelises here instead.
#   ANSWER_TOP_K=""       ANSWER depth on STAGE=answer — k MEMORIES of the saved
#                         retrieval that reach the prompt
#   CONTEXT_BUDGET=""     ANSWER token budget on STAGE=answer
#   MAX_TOKENS=""         OUTPUT budget for the generated answer (--max-tokens).
#                         Empty = answer.py's default of 2000, so nothing moves.
#                         On the native path --max-tokens OVERRIDES the config's
#                         answer_max_tokens, so a system asking for more is cut
#                         to 2000 unless this is set. hypermem is the only one
#                         that asks for more (4096, for its 7-step CoT prompt):
#                           MAX_TOKENS=4096 SYSTEMS=hypermem bash scripts/query.sh
#   TRACE_LEVEL=full      off|calls|steps|full — record what the system's own
#                         retrieval/answer loop did, step by step. Default full
#                         (untruncated payloads), matching ingest.sh.
#                         NATIVE ONLY: the flat bm25/dense path takes no trace
#                         flags, so this is ignored there rather than passed.
#   TRACE_SAMPLES=""      how many samples to trace when TRACE_LEVEL != off.
#                         Empty (the default) = ALL of them. Set a number to cap
#                         the cost — a `full` sample can be tens of MB, so
#                         TRACE_SAMPLES=1 writes one trajectory per shard.
#   GEN_MODEL             defaults to MEM_MODEL — same served endpoint
#   NUM_SHARDS=1          split samples across processes; set SHARD_ID=0..N-1 per process
#   EMB_MODEL             embedder for MODE=dense
#   SKIP_QRELS=0          1 = reuse existing qrels instead of rebuilding
#   ANSWER_OVERWRITE=0    1 = regenerate answers already in predictions.jsonl,
#                         instead of skipping them. STAGE=answer. REQUIRED when
#                         the re-run exists to change the answers (a new
#                         MAX_TOKENS, GEN_MODEL, prompt): without it the stage
#                         skips every question and rewrites config.json anyway.
#   RETRIEVE_OVERWRITE=0  1 = re-run samples that already have a run.tsv,
#                         instead of resuming past them. STAGE=retrieve only.
#                         Needed to back-fill trajectories into a cell that
#                         already finished — a resumed sample writes none.
#   RETRIEVAL_OUT=""      MODE=native STAGE=retrieve only: a PREFIX for a side
#                         tree, for a LATENCY/COST re-run that must not touch
#                         the published cell. The keyed path is kept and only
#                         its root moves:
#                           outputs/_retrieval/<keyed...>  ->  ${RETRIEVAL_OUT}_retrieval/<keyed...>
#                         and the qrels rebuild and the eval are SKIPPED — the
#                         keyed dirs come from lib/paths and ignore OUT_ROOT, so
#                         without this knob a re-run rebuilds the published
#                         qrels (--overwrite) and rewrites the published
#                         metrics.
#   plus everything scripts/lib/vllm.sh reads (CTX_*, GPU_UTIL_*, DRY_RUN,
#   SKIP_SERVER, MEM_MODEL, VLLM_PORT, CONDA_ENV, ...)
#

# No `set -e`: one system failing must not skip the rest.
set -uo pipefail

# ---- Shared stage plumbing -------------------------------------------------
REPO_DIR=${REPO_DIR:-$PWD}
source "$REPO_DIR/scripts/lib/vllm.sh"

# ---- Answer knobs ----------------------------------------------------------
# Which half (or halves) of the query path to run.
#
#   both      retrieve and answer in one pass — one answer per question. The
#             default, and what the native answer path has always done.
#   retrieve  the system's retrieval loop only, saved to disk. No generation.
#   answer    answer from a retrieval this stage saved earlier, via
#             --from-retrieval. Pays no retrieval cost at all.
#
# Splitting is worth it exactly when you want several answers from one
# retrieval: a sweep over budgets, generators, or answer prompts. SimpleMem
# spends 5-8 LLM calls retrieving and one answering, so a four-point sweep run
# as `both` pays for retrieval four times to vary the cheap half.
STAGE=${STAGE:-both}
case "$STAGE" in
  both|retrieve|answer) ;;
  *) echo "!!! STAGE must be both, retrieve or answer — got '$STAGE'" >&2; exit 1 ;;
esac

# Where a split run keeps its saved retrieval. Defaults to the same directory
# the retrieve stage writes, so the two halves line up without being told.
FROM_RETRIEVAL=${FROM_RETRIEVAL:-}

# ---- Which ANSWER PROMPT the generator is given ----------------------------
#
#   shared  THE DEFAULT AND THE UNIFIED SETTING. evalgrid.systems.base's
#           shared_answer_prompt — one prompt for all six systems AND for the
#           raw-turn baselines (naive RAG, full context), which have no other.
#           That is what makes a memory-system row and a baseline row differ by
#           memory representation alone rather than by prompt wording.
#   native  each system's own answer prompt (SystemPlugin.answer_prompt). Five
#           of six call shared_answer_prompt internally; LightMem is the one
#           system whose prompt really differs, so this is a labelled control,
#           not the comparison.
#
# The two are DIFFERENT CODE PATHS in pipeline/answer.py, not a flag on one:
#
#   --prompt native  replays the system's own trace   (--from-retrieval)
#   --prompt shared  re-ranks the qrels corpus by the saved run.tsv
#                    (--qrels-dir + --retrieval-dir)
#
# Both read the SAME saved retrieval — the run.tsv the retrieve stage wrote —
# so a shared-prompt cell costs generation only, exactly like a native replay,
# and needs no re-retrieval. Two further consequences of the path split, both
# in the shared path's favour here:
#   * NO MEMORY SYSTEM IS LOADED, so no store and no memory LLM are touched;
#     the job needs the generator alone.
#   * it writes answer_stats.jsonl and answer_summary.json (the native replay
#     writes neither), so every budget/truncation
#     field worth reporting is on disk without re-parsing
#     traces.jsonl.
#
# On disk the two never collide: the path segment carries "--native-prompt" for
# the non-default one, the same rule lib/paths.gen_segment follows.
# Whether the caller ASKED for a prompt, before the default is applied. The
# default is shared, and it applies at every stage — so "PROMPT is shared" alone
# says nothing about intent and cannot be the trigger for the STAGE check below.
# Without this, the default made every STAGE=retrieve job an error.
PROMPT_EXPLICIT=${PROMPT:+1}
PROMPT=${PROMPT:-shared}
case "$PROMPT" in
  shared|native) ;;
  *) echo "!!! PROMPT must be shared or native — got '$PROMPT'" >&2; exit 1 ;;
esac
# (The STAGE/MODE compatibility check lives below, where MODE is resolved.)
#
# The shared path takes --concurrency instead of --num-shards/--shard-id: it
# makes no query-time memory call, so its questions are independent generator
# calls and run concurrently rather than being split across array tasks.
CONCURRENCY=${CONCURRENCY:-8}

# ---- The two answer-budget axes, and the retrieval depth that must serve them
#
# NATIVE_TOP_K   RETRIEVAL depth — how deep the system's own loop goes, applied
#                before it runs. On this path k is retriever *identity* (A-Mem's
#                link expansion, MemoryBank's regrouping and HyperMem's funnel
#                all operate on it), so a deep run is a DIFFERENT retriever, not
#                a prefix of a shallow one, and it gets its own output dir.
#                  auto   = each spec's native_top_k (THE DEFAULT)
#                  empty  = each config's published depth — the "as published"
#                           control. Note the single dash in the assignment
#                           below: NATIVE_TOP_K= from the submitting shell means
#                           "published depth", not "use the default", the same
#                           convention LIMIT= follows.
#                  <n>    = that depth for every system
#
#                AUTO IS THE DEFAULT because the published depths are too
#                shallow to replay from: lightmem and mem0 return 10 units
#                (~285 and ~189 tokens), so a top-50 cell is unreachable and a
#                4000-token budget fills to 7%. Both cells come back
#                context_underfilled rather than failing. The auto depths are
#                measured for ~2x headroom on both answer axes; the published
#                run is still one submission away and lands in its own dir.
#
# ANSWER_TOP_K   ANSWER depth — how many memories of a SAVED retrieval reach the
# CONTEXT_BUDGET prompt. Both are post-hoc cuts of a finished ranking, so one
#                deep retrieval serves a whole sweep over either. Replay only
#                (STAGE=answer); a budget on a live run would pay for retrieval
#                and then throw part of it away.
#
# MAX_TOKENS     OUTPUT budget for the generated answer, passed straight through
#                as --max-tokens. Empty (the default) leaves pipeline/answer.py's
#                own default of 2000 in place, so no existing cell moves.
#
#                IT IS NOT ONLY A CAP, it is an OVERRIDE. On the native path
#                answer.py:791 writes it over the config's `answer_max_tokens`
#                unconditionally (systems/base.apply_answer_max_tokens), so a
#                config asking for more than 2000 is silently downgraded unless
#                this is set. Only hypermem asks for more (4096, sized for its
#                7-step CoT prompt); every other system's config declares 2000,
#                which is what the default already writes, so this knob is a
#                no-op for them. The warning below fires when the two disagree.
NATIVE_TOP_K=${NATIVE_TOP_K-auto}
ANSWER_TOP_K=${ANSWER_TOP_K:-}
CONTEXT_BUDGET=${CONTEXT_BUDGET:-}
MAX_TOKENS=${MAX_TOKENS:-}
# Both cut a SAVED ranking, so only the replay half reads them — and the check
# exists because answers_dir() names the cell after them either way. Without it,
# STAGE=both plus ANSWER_TOP_K=10 retrieves at the full native depth, applies no
# answer-side cut, and files the result under `top10`: a cell named for a
# truncation that never happened, which is exactly the class of mislabeled
# directory every other segment in this script exists to prevent. Refusing is
# better than ignoring: a knob that is silently dropped reads as "I set the
# budget and nothing changed".
if [[ "$STAGE" != "answer" && -n "$ANSWER_TOP_K$CONTEXT_BUDGET" ]]; then
  echo "!!! ANSWER_TOP_K / CONTEXT_BUDGET cut a SAVED retrieval and apply to" >&2
  echo "!!!   STAGE=answer only (got STAGE=$STAGE)." >&2
  echo "!!! Run the retrieval once, then sweep either axis over it:" >&2
  echo "!!!   SYSTEMS=<sys> STAGE=retrieve MEM_ROOT=\$MEM_ROOT bash scripts/query.sh" >&2
  echo "!!!   SYSTEMS=<sys> STAGE=answer MEM_ROOT=\$MEM_ROOT CONTEXT_BUDGET=4000 bash scripts/query.sh" >&2
  exit 1
fi

# Per-system retrieval depth giving ~2x headroom on BOTH answer axes (top-50
# memories and a 4000-token budget), measured from outputs/_smoke2_retrieval.
#
# One number cannot serve all six, which is why this is a table and not a
# constant: units-per-k spans 1.0 (lightmem, mem0, memorybank) to 4.3 (a_mem's
# link expansion), and tokens-per-k spans 24 (mem0) to 2,032 (a_mem) — 85x. A
# uniform 150 would be 6x too deep for a_mem (~305k tokens per question, from a
# ~500-unit corpus) and too shallow for mem0 (3,630 tokens, never reaching a
# 4000-token budget). Note which axis binds differs: tokens for the small-unit
# systems, depth for the large-unit ones.
native_top_k() {
  [[ -z "$NATIVE_TOP_K" ]] && { echo ""; return; }
  [[ "$NATIVE_TOP_K" != "auto" ]] && { echo "$NATIVE_TOP_K"; return; }
  case "$1" in
    a_mem)      echo 25  ;;   # 4.3 units/k, 2,032 tok/k — depth binds
    hypermem)   echo 95  ;;   # 2.1 units/k,    82 tok/k — tokens bind
    lightmem)   echo 270 ;;   # 1.0 units/k,    30 tok/k — tokens bind
    mem0)       echo 330 ;;   # 1.0 units/k,    24 tok/k — tokens bind
    memorybank) echo 100 ;;   # 1.0 units/k,   509 tok/k — depth binds
    simplemem)  echo 45  ;;   # 2.6 units/k,   171 tok/k — tokens bind
    *)          echo ""  ;;
  esac
}

# Systems whose retrieval calls the MEMORY model, and therefore need vLLM even
# in STAGE=retrieve. The other four rank with a local embedder or BM25, so their
# retrieval jobs skip the server entirely — which saves the 10-25 minute vLLM
# start, the dominant cost of a short retrieval run.
#
# HYPERMEM IS NOT IN THIS LIST AND STILL NEEDS AN EMBEDDING SERVER ON A NATIVE
# RUN. This variable is about the *memory* model, which hypermem's funnel never
# prompts — that part is still true. But its config ranks with
# `retrieval_type: rrf`, so MODE=native hypermem reaches for an embedder:
#
#   vllm serve Qwen/Qwen3-Embedding-4B --task embed --port 11810
#
# The cross-encoder (:12810) is needed for ONE config and no other:
# configs/hypermem/naive_query.yaml, the published-setting row, sets
# `reranker.enabled: true`. Every other hypermem config here keeps it false
#, so every other run needs the embedder alone.
#
# Start it with RERANK_SERVE (see its block below), which this script does
# manage — unlike the embedder, which predates that plumbing:
#
#   RERANK_SERVE=Qwen/Qwen3-Reranker-4B@12810
#
# This script does not start the embedder — it manages a single endpoint, the
# memory/generator model. Start it yourself before submitting a native hypermem
# job, or use evalgrid/configs/hypermem/grid.yaml (`retrieval_type: keyword`),
# which needs no server and is a labelled ablation of the published method.
# Both failure modes are loud: the adapter raises rather than degrading to
# BM25.
NEEDS_MEM_LLM=${NEEDS_MEM_LLM:-"a_mem simplemem"}

MODE=${MODE:-native}
case "$MODE" in
  native|bm25|dense) ;;
  *) echo "!!! MODE must be native, bm25 or dense — got '$MODE'" >&2; exit 1 ;;
esac

# A shared-prompt answer is a cut of a SAVED ranking, so one has to exist first.
# STAGE=both retrieves and answers in a single native pass — there is no run.tsv
# on disk yet when that pass builds its prompt, so the two are incompatible by
# construction rather than by policy.
# Only when the caller SET it: PROMPT is inert at retrieve (retrieval has no
# prompt) and the default carries through to whatever answer job follows, which
# is the contract native_grid.sh documents and relies on.
if [[ -n "$PROMPT_EXPLICIT" && "$PROMPT" == "shared" && "$MODE" == "native" && "$STAGE" != "answer" ]]; then
  echo "!!! PROMPT=shared reads a SAVED ranking (run.tsv), so it is STAGE=answer only" >&2
  echo "!!!   (got STAGE=$STAGE). Retrieve once, then answer over it:" >&2
  echo "!!!   SYSTEMS=<sys> STAGE=retrieve MEM_ROOT=\$MEM_ROOT bash scripts/query.sh" >&2
  echo "!!!   SYSTEMS=<sys> STAGE=answer PROMPT=shared MEM_ROOT=\$MEM_ROOT bash scripts/query.sh" >&2
  exit 1
fi

SYSTEMS=${SYSTEMS:-"lightmem a_mem mem0 memorybank simplemem hypermem"}
MEM_ROOT=${MEM_ROOT:-}
# Scoring artifacts live beside the memory they score. For a throwaway ingest
# (MEM_ROOT=outputs/_smoke2) that means outputs/_smoke2_{qrels,retrieval,...},
# which keeps a test run from being mistaken for the real tree later.
#
# THE TRAILING SLASH ON THE FALLBACK IS LOAD-BEARING. OUT_ROOT is a string
# PREFIX, not a directory: the
# paths are "${OUT_ROOT}_retrieval", so `outputs` yields outputs_retrieval/ —
# a SIBLING of outputs/ — while `outputs/` yields outputs/_retrieval/, the keyed
# tree lib/paths.py defines and every other stage reads. A real run sets
# MEM_ROOT empty, so this fallback IS what a real run gets, and with `outputs`
# it could never land in the tree the rest of the pipeline uses. Job 27475495
# wrote a whole simplemem grid into the sibling and then failed scoring on qrels
# that were sitting in the right tree the whole time.
OUT_ROOT=${OUT_ROOT:-${MEM_ROOT:-outputs/}}
LIMIT=${LIMIT-}
TOP_K=${TOP_K:-10}
NATIVE_RETRIEVER=${NATIVE_RETRIEVER:-}
NATIVE_VARIANT=${NATIVE_VARIANT:-}
CONFIG=${CONFIG:-}
if [[ -n "$NATIVE_RETRIEVER$NATIVE_VARIANT" && "$MODE" != "native" ]]; then
  echo "!!! NATIVE_RETRIEVER/NATIVE_VARIANT apply to MODE=native only (got MODE=$MODE)" >&2
  exit 1
fi
# Built once and reused, so every stage passes the identical flags. Empty when
# nothing is swapped, which is what keeps an unswapped run unchanged.
SWAPFLAG=()
[[ -n "$NATIVE_RETRIEVER" ]] && SWAPFLAG+=(--native-retriever "$NATIVE_RETRIEVER")
[[ -n "$NATIVE_VARIANT"   ]] && SWAPFLAG+=(--native-variant "$NATIVE_VARIANT")
[[ -n "$CONFIG"           ]] && SWAPFLAG+=(--config "$CONFIG")
GEN_MODEL=${GEN_MODEL:-$MEM_MODEL}
GEN_BASE_URL=${GEN_BASE_URL:-$MEM_BASE_URL}
EMB_MODEL=${EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B}
SKIP_QRELS=${SKIP_QRELS:-0}
RETRIEVAL_OUT=${RETRIEVAL_OUT:-}
if [[ -n "$RETRIEVAL_OUT" ]]; then
  if [[ "$MODE" != "native" || "$STAGE" != "retrieve" ]]; then
    echo "!!! RETRIEVAL_OUT is for MODE=native STAGE=retrieve only (got MODE=$MODE STAGE=$STAGE)" >&2
    exit 1
  fi
  # The side run ranks against the published store; its gold set is the
  # published one and is not rebuilt (and not scored — see the eval step).
  SKIP_QRELS=1
fi
# Which evidence set eval_retrieval scores a ranking against
# (evalgrid/benchmarks/gold.py). "turn" is the evaluation default: "strict"
# treats the 94 evidence sessions LongMemEval names but never flags as
# distractors, and for 21 questions leaves the gold set EMPTY. Rankings do not
# depend on labels, so changing this re-scores without re-retrieving, and each
# level writes its own metrics file.
GOLD=${GOLD:-turn}
# RETRIEVE stage only, and named for it: the answer stage has its OWN resume
# (by question-id) and must not be swept along by one variable.
#
# retrieve.py resumes by sample — a sample holding a run.tsv is skipped — which
# is what makes a walltime-killed cell cheap to finish. The cost is that a
# RE-run of a COMPLETE cell is a no-op: it re-derives nothing, and in particular
# writes no trajectory for a sample it skips. So a cell that finished before
# the trace scope widened cannot be back-filled by resubmitting it.
# RETRIEVE_OVERWRITE=1 passes --overwrite and re-runs every sample in the cell.
RETRIEVE_OVERWRITE=${RETRIEVE_OVERWRITE:-0}
OVERWRITEFLAG=()
[[ "$RETRIEVE_OVERWRITE" == "1" ]] && OVERWRITEFLAG=(--overwrite)
# The SAME no-op-on-resume story on the answer side, and it needed its own knob:
# OVERWRITEFLAG above reaches only the retrieve invocation, so an answer cell
# that already has every question in predictions.jsonl could not be regenerated
# through this script at all — the stage skips by question_id and reports
# "Nothing to answer".
#
# That is the right default (it is what makes a walltime-killed sweep cheap to
# finish), but it makes a SETTINGS change invisible: re-submitting the same cell
# with a different MAX_TOKENS, GEN_MODEL or CONTEXT_BUDGET regenerates nothing
# and then rewrites config.json from the new args, labelling old answers with
# settings they were never generated under (answer.py's own warning at the end
# of a no-op run says as much). Set this whenever the point of the re-run is
# that the ANSWERS should change.
ANSWER_OVERWRITE=${ANSWER_OVERWRITE:-0}
ANSWER_OVERWRITEFLAG=()
[[ "$ANSWER_OVERWRITE" == "1" ]] && ANSWER_OVERWRITEFLAG=(--overwrite)
# ---- The Python environment, BEFORE ANYTHING CALLS PYTHON ------------------
# Everything below this line that resolves a NAME runs python: the dataset read
# off the spec (spec_dataset, immediately below) and every output directory,
# derived by evalgrid.tools.stage_dir and evalgrid.tools.native_dir. So the env
# has to be active HERE. This call used to sit ~250 lines down, just above the
# stage loop, and those derivations ran on whatever interpreter the job
# inherited.
#
# --export=ALL copies the submitting shell, so that worked when the shell had
# `conda activate MemGrid` in it and failed when it did not — with a traceback
# that names neither the cause nor the fix:
#
#   ModuleNotFoundError: No module named 'tqdm'
#   !!! could not derive the run directory for 'simplemem'
#   !!! a hypermem swap requires CONFIG=...    <- a canned hint, not the reason
#
# The dataset read cleared on the way past because system python3.9 carries
# PyYAML but not tqdm, which is why the job died on the directory rather than
# on the spec. A batch job must not depend on the shell that submitted it: the
# env it needs is the one named by CONDA_ENV, activated on the node that runs.
#
# The `cd` moves up with it, and belongs here for the same reason: spec_of()
# yields repo-relative paths (experiments/<sys>/<spec>.yaml), so the spec read
# below requires this working directory.
cd "$REPO_DIR" || { echo "!!! repo dir not found: $REPO_DIR" >&2; exit 1; }
activate_python

# ---- Which dataset ---------------------------------------------------------
# SPEC_NAME and spec_of() come from scripts/lib/vllm.sh, sourced above.
#
# DATASET DEFAULTS TO THE SPEC'S OWN `dataset:` rather than to a literal. It has
# to: five of this script's stages pass only --exp, so a DATASET that disagreed
# with the spec would move the scoring paths and leave the native
# retrieve/answer paths behind — half a run in each of two trees. Reading it from
# the spec makes SPEC_NAME alone sufficient and keeps the two in step.
_SPEC_DS="$(spec_dataset)"

# A spec that EXISTS but whose dataset could not be read must not fall back.
# The fallback is "longmemeval_s", so under SPEC_NAME=beam_100k.yaml a failed
# read would score into the LongMemEval tree while the --exp-only stages ran
# BEAM — the very split the check below exists to prevent, arrived at silently
# because the check is skipped when _SPEC_DS is empty.
if [[ -z "$_SPEC_DS" ]]; then
  for _s in $SYSTEMS; do
    if [[ -f "$(spec_of "$_s")" ]]; then
      echo "!!! $(spec_of "$_s") exists but its 'dataset:' could not be read." >&2
      echo "!!! Refusing to fall back to longmemeval_s — that would score this" >&2
      echo "!!! run into the wrong tree. Check the spec parses, and that python" >&2
      echo "!!! with pyyaml is on PATH at this point in the job." >&2
      exit 1
    fi
  done
fi
DATASET=${DATASET:-${_SPEC_DS:-longmemeval_s}}

# A DATASET explicitly set to something the spec does not name is the silent
# half-run above, so it is refused rather than resolved one way or the other.
if [[ -n "$_SPEC_DS" && "$DATASET" != "$_SPEC_DS" ]]; then
  echo "!!! DATASET=$DATASET but $SPEC_NAME says dataset: $_SPEC_DS" >&2
  echo "!!! The --exp-only stages (native retrieve/answer) would use '$_SPEC_DS'" >&2
  echo "!!! while the scoring stages used '$DATASET' — two half-filled trees." >&2
  echo "!!! Pick the spec, not the dataset: SPEC_NAME=<dataset>.yaml" >&2
  exit 1
fi

NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}
# --num-shards/--shard-id are native-path flags and the shared path's parser
# rejects them. Refused here rather than dropped: a shard silently ignored makes
# every array task answer the WHOLE dataset into one predictions.jsonl, and the
# per-question resumability check would make all but the first look instant and
# correct. The shared path parallelises with --concurrency instead, in one task.
if [[ "$PROMPT" == "shared" ]] && (( NUM_SHARDS > 1 )); then
  echo "!!! PROMPT=shared does not shard — the shared answer path has no --num-shards." >&2
  echo "!!!   It runs $CONCURRENCY questions concurrently in ONE task instead (no" >&2
  echo "!!!   query-time memory call is made, so the questions are independent)." >&2
  echo "!!!   Drop NUM_SHARDS/ARRAY and raise CONCURRENCY if it is not fast enough." >&2
  exit 1
fi

# Logs mirror the submit dirs, so a retrieval log is where you would look for
# one rather than under "answer".
LOG_DIR=$([[ "$STAGE" == "retrieve" ]] && echo logs/retrieve || echo logs/answer)
mkdir -p "$LOG_DIR" logs/vllm logs/query
JOB=${JOB:-$$}

# Where each system's artifacts go. Kept as functions rather than inlined so the
# three stages cannot disagree about a path — the failure mode that produces an
# eval scoring one run against another run's qrels.
mem_dir()       { [[ -n "$MEM_ROOT" ]] && echo "$MEM_ROOT/$1" || echo ""; }
# Keyed output dirs come from evalgrid/lib/paths.py via
# evalgrid.tools.stage_dir, NOT from string-building here. They used to be built
# here, and the four functions between them carried the system and the retriever
# and dropped <dataset>, <mem-seg>, <labeler-seg> and <gen-seg> — four of the
# five keys lib/paths defines. The whole 2026-08 grid landed in
# outputs/_retrieval/<sys>/<retriever>/ and every checker reported it MISSING.
#
# That is not a tidiness problem. Those segments exist to stop silent overwrite:
# retrieve resumes per sample on run.tsv and answer resumes per question_id, so
# a second mem-variant / dataset / labeler / generator written to the same flat
# cell is SKIPPED as already-done and the FIRST run's output is served under the
# second's name. lightmem ships three configs and hypermem two, so that is live.
#
# Derived ONCE per system into a table, for the reason the swap table above
# gives: these are called from $( ), where a failure would otherwise become an
# empty string pasted into a path.
STAGE_DIRS=""
_stage_dir() {   # _stage_dir <stage> <system> [extra flags...]
  local stage=$1 sys=$2; shift 2
  python -m evalgrid.tools.stage_dir --relative \
    --exp "$(spec_of "$sys")" --stage "$stage" \
    --system "$sys" --dataset "$DATASET" --mem-model "$MEM_MODEL" \
    ${MEM_VARIANT:+--mem-variant "$MEM_VARIANT"} \
    ${MEM_ROOT:+--root "$MEM_ROOT"} "$@"
}

# The <retriever> segment. The hand-built forms below are right — `native-top45`
# is what lib/paths.native_segment produces, byte for byte — so they stay here
# rather than being re-derived in Python, which would make a third copy of the
# MODE/NATIVE_RETRIEVER/TOP_K logic that the two stages already agree on.
#
# EXCEPT when a variant or a swap names the run. Both put the ENCODER in the
# segment (`native-emb-<model>--naive`), which bash cannot know — it is read off
# the resolved config by the plugin — so those cases are derived, via
# evalgrid.tools.native_dir, exactly as pipeline/retrieve derives it internally.
#
# NATIVE_VARIANT and CONFIG used to be missing from this decision, and the cost
# was not cosmetic: a run differing only in those fell to the bare `native`
# below, so lightmem under default.yaml and lightmem under naive_query.yaml
# wrote to ONE directory. Retrieval resumes per sample on run.tsv and answering
# per question_id, so the second config would have been skipped sample by sample
# and the first one's rankings and predictions served under its name — with
# check_grid reporting ok 500/500 and nothing but the config.json the second run
# overwrote to say which config the numbers came from.
#
# CONFIG counts because for HYPERMEM the retriever NAME is a function of the
# config and nothing else (SystemPlugin.native_retriever_name):
#
#   default.yaml      native-rrf-norerank      rrf first stage, cross-encoder off
#   grid.yaml         native-bm25-norerank     retrieval_type: keyword
#   naive_query.yaml  native-rrf               cross-encoder back on
#
# Three retrieval conditions, no swap and no variant between them. Keyed on
# NATIVE_RETRIEVER/NATIVE_VARIANT alone, all three collide on one directory —
# the very failure this function was just fixed for everywhere else.
#
# Passing CONFIG explicitly therefore always derives, even when the path IS the
# default config: naming a config is a statement that this run is configured,
# and the derived name is the fully explicit one. A run that names no config
# keeps the historical short name, which is what leaves finished runs in place.
retriever_seg() {
  local k; k=$(native_top_k "$1")
  if [[ "$MODE" == "native" && -n "$NATIVE_RETRIEVER$NATIVE_VARIANT$CONFIG" ]]; then
    derived_seg "$1"
  elif [[ "$MODE" == "native" && -n "$k" ]]; then
    echo "$MODE-top$k"
  else
    echo "$MODE"
  fi
}

# The answers <topk-seg>: unchanged logic, lifted out of answers_dir so the
# table below can build every stage's path in one pass.
depth_seg() {
  if [[ "$MODE" != "native" ]]; then
    echo "top$TOP_K"
  elif [[ -n "$CONTEXT_BUDGET" ]]; then
    echo "budget$CONTEXT_BUDGET"   # a budget supersedes k, as in answer.py
  elif [[ -n "$ANSWER_TOP_K" ]]; then
    echo "top$ANSWER_TOP_K"
  else
    echo all
  fi
}

_lookup() {   # _lookup <system> <column 2..5>
  printf '%s' "$STAGE_DIRS" | awk -F'\t' -v s="$1" -v c="$2" '$1==s {print $c; found=1} END {exit !found}'
}
qrels_dir()     { _lookup "$1" 2; }
# A native run at an imposed depth is a different retriever, so it must not
# land on the published-depth control. Mirrors what lib/paths.native_segment
# does when --out is not overridden.
#
# A SWAPPED OR VARIANT run is named by lib/paths itself, via
# evalgrid.tools.native_dir — the `native-<slug>-top<k>[--variant]` vocabulary,
# which encodes the substitute and the named setting and so cannot collide. The
# hand-built `$MODE-top$k` form is kept for the plain case (no swap, no variant)
# so that every pre-existing run, answer and eval directory stays exactly where
# it is: this knob must not move a GPU-day of finished runs. The two forms
# therefore differ, deliberately — a derived run carries the encoder slug, a
# plain one does not. scripts/native_grid.sh relies on this: it
# clears NATIVE_VARIANT for GRID_RETRIEVERS=native precisely so the published
# control keeps its historical bare `native` directory.
# Derived ONCE for every system, at startup, into a "sys<TAB>seg" table.
#
# Not on demand inside the dir functions, because those are called from
# $( ... ) — and `exit` inside a command substitution kills only the SUBSHELL.
# A failing derivation there returned an empty string that the caller happily
# pasted into a path, producing outputs/..._retrieval/<sys>/ and an eval dir of
# <sys>/<sys>. Up front, a failure stops the job before it starts a server.
DERIVED_SEGS=""
if [[ -n "$NATIVE_RETRIEVER$NATIVE_VARIANT$CONFIG" ]]; then
  for _s in $SYSTEMS; do
    _k=$(native_top_k "$_s")
    # An EMPTY k is "the config's published depth", not "no opinion". Omitting
    # the flag made native_dir fall back to the spec's native_top_k and print a
    # -top<k> the run does not write, so a published-depth run derived a
    # directory nothing would ever fill. --published-depth says it outright.
    if [[ -n "$_k" ]]; then _DEPTH=(--top-k "$_k"); else _DEPTH=(--published-depth); fi
    _seg=$(python -m evalgrid.tools.native_dir --system "$_s" \
             "${_DEPTH[@]}" "${SWAPFLAG[@]+"${SWAPFLAG[@]}"}")
    if [[ $? -ne 0 || -z "$_seg" ]]; then
      echo "!!! could not derive the run directory for '$_s'" >&2
      echo "!!!   NATIVE_RETRIEVER=$NATIVE_RETRIEVER NATIVE_VARIANT=$NATIVE_VARIANT CONFIG=$CONFIG" >&2
      echo "!!! a hypermem swap requires CONFIG=evalgrid/configs/hypermem/grid.yaml" >&2
      echo "!!! (without it retrieval_type is rrf and the swap is refused)" >&2
      exit 1
    fi
    DERIVED_SEGS+="$_s	$_seg"$'\n'
  done
  echo "Native dirs:"
  printf '%s' "$DERIVED_SEGS" | while IFS=$'\t' read -r a b; do echo "   $a -> $b"; done
fi
derived_seg() {
  printf '%s' "$DERIVED_SEGS" | awk -F'\t' -v s="$1" '$1==s {print $2; found=1} END {exit !found}'
}

# NOW the per-system dir table can be built: retriever_seg() calls derived_seg()
# for a swapped or variant run, so this MUST come after DERIVED_SEGS is populated above. It
# did not, and every derivation came back empty — stage_dir was handed
# --retriever-seg "" and refused, which is the argument for it refusing rather
# than defaulting.
for _s in $SYSTEMS; do
  _rseg=$(retriever_seg "$_s"); _dseg=$(depth_seg)
  _q=$(_stage_dir qrels     "$_s") || _q=""
  _r=$(_stage_dir retrieval "$_s" --retriever-seg "$_rseg") || _r=""
  _e=$(_stage_dir eval      "$_s" --retriever-seg "$_rseg") || _e=""
  _a=$(_stage_dir answers   "$_s" --retriever-seg "$_rseg" --depth-seg "$_dseg" \
         --gen-model "$GEN_MODEL" --prompt "$PROMPT") || _a=""
  if [[ -z "$_q" || -z "$_r" || -z "$_e" || -z "$_a" ]]; then
    echo "!!! could not derive output dirs for '$_s' (stage_dir failed)" >&2
    echo "!!!   DATASET=$DATASET MEM_MODEL=$MEM_MODEL GEN_MODEL=$GEN_MODEL" >&2
    exit 1
  fi
  STAGE_DIRS+="$_s	$_q	$_r	$_e	$_a"$'\n'
done
retrieval_dir() { _lookup "$1" 3; }
# Keyed by the ANSWER depth/budget as well as MODE, mirroring what
# lib/paths.answer_depth_segment appends when --out is not overridden:
#   shared path        top<TOP_K>          the depth the prompt was cut to
#   native, replay     top<k> | budget<T>  the cut applied to a saved ranking
#   native, live       all                 nothing was truncated afterwards; k
#                                          is already in the retriever segment
# pipeline/answer.py takes --out LITERALLY and appends no segment of its own, so
# without this every cell of an ANSWER_TOP_K / CONTEXT_BUDGET sweep wrote one
# predictions.jsonl — and since the answer stage resumes by question_id, the
# second cell skipped every question as already answered and filed the first
# cell's predictions under its own name.
# The answers dir, from lib/paths.answers_dir via the table above. Three axes
# key it, and each was learned the hard way — the notes are kept because the
# reasons are not obvious from the path:
#
#   <retriever>  the RETRIEVAL CONDITION. Two runs reading different saved
#                retrievals would otherwise share one predictions.jsonl, and the
#                answer stage — which resumes by question_id — would skip every
#                question of the second and file the first's predictions under
#                its name. Covers the swap (native-rrf-...), the retrieval depth
#                (native-top270 vs native) and plain native at once.
#   <gen-seg>    the GENERATOR, plus the prompt as a suffix when it is not the
#                shared one. Same failure: a two-prompt run lands twice on one
#                predictions.jsonl and the second is skipped question by
#                question. NOTE this moved — the old hand-built path suffixed
#                the DEPTH segment with --<prompt>-prompt; lib/paths.gen_segment
#                suffixes the GENERATOR segment, which is the spelling every
#                other reader expects.
#   <topk-seg>   the answer depth or budget the prompt was cut to, so a
#                ANSWER_TOP_K / CONTEXT_BUDGET sweep does not write one
#                predictions.jsonl per cell.
answers_dir()   { _lookup "$1" 5; }
# Keyed by the retrieval segment, not by MODE: a native-top<k> run and the
# published-depth control are different retrievers and must not share a
# metrics file.
eval_dir()      { _lookup "$1" 4; }
# The retrieve stage's --out: the keyed dir, re-rooted under RETRIEVAL_OUT when
# that is set. Refuses a keyed dir outside outputs/_retrieval/ rather than
# guessing, since a wrong guess would write into the published tree.
side_retrieval_dir() {
  local rd; rd=$(retrieval_dir "$1")
  if [[ -z "$RETRIEVAL_OUT" ]]; then echo "$rd"; return; fi
  [[ "$rd" == outputs/_retrieval/* ]] || { echo "!!! cannot re-root $rd under RETRIEVAL_OUT" >&2; return 1; }
  echo "${RETRIEVAL_OUT}_retrieval/${rd#outputs/_retrieval/}"
}

echo "==================================================================="
echo "Query stage                job $JOB on $(hostname)"
echo "Mode       : $MODE"
echo "Stage      : $STAGE"
[[ "$MODE" == "native" && "$STAGE" == "answer" ]] && \
  echo "Prompt     : $PROMPT$([[ "$PROMPT" == "shared" ]] \
       && echo " (the unified prompt — same as the naive-RAG/full-context rows)" \
       || echo " (the system's own — a labelled control, path suffix --native-prompt)")"
echo "Systems    : $SYSTEMS"
echo "Dataset    : $DATASET"
echo "Mem model  : $MEM_MODEL"
echo "Gen model  : $GEN_MODEL"
echo "Memory dir : ${MEM_ROOT:-outputs/_memory (keyed default)}"
echo "Out root   : $OUT_ROOT"
echo "Limit      : ${LIMIT:-none (full dataset)}"
[[ "$MODE" != "native" ]] && echo "Top-k      : $TOP_K"
# Not "auto -> <n>": a multi-system run has one depth PER system, so a single
# number in the banner would be the first system's and wrong for the rest. The
# per-system value is visible on each stage's own command line below.
if [[ "$MODE" == "native" ]]; then
  case "$NATIVE_TOP_K" in
    "")     echo "Native k   : published depth (control)" ;;
    auto)   echo "Native k   : auto (each spec's native_top_k)" ;;
    *)      echo "Native k   : $NATIVE_TOP_K (imposed on every system)" ;;
  esac
fi
[[ -n "$ANSWER_TOP_K" ]]  && echo "Answer k   : $ANSWER_TOP_K (post-hoc cut of the saved retrieval)"
[[ -n "$CONTEXT_BUDGET" ]] && echo "Budget     : $CONTEXT_BUDGET tok"
# The shared answer path records no trajectory and cannot: add_trace_args is
# attached to the NATIVE parser group, because a trace is a record of a system's
# own loop and this path runs none — it re-ranks a corpus by a saved run.tsv.
# What it writes instead is answer_stats.jsonl / answer_summary.json, which is
# the per-question budget and truncation record. The
# retrieval trajectory is not lost either: it was recorded once, by the retrieve
# stage, in the retrieval.jsonl this run replays.
if [[ "$MODE" == "native" && "$PROMPT" == "shared" && "$STAGE" == "answer" ]]; then
  echo "Trace      : n/a on the shared path — answer_stats.jsonl + answer_summary.json instead"
elif [[ "$MODE" == "native" ]]; then
  echo "Trace      : $TRACE_LEVEL$([[ "$TRACE_LEVEL" != "off" ]] && echo " ($TRACE_SCOPE)")"
fi
[[ "$ANSWER_OVERWRITE" == "1" && "$STAGE" == "answer" ]] && \
  echo "Overwrite  : yes — every question re-answered, resume disabled"
[[ "$RETRIEVE_OVERWRITE" == "1" && "$STAGE" == "retrieve" ]] && \
  echo "Overwrite  : yes — every sample re-retrieved, resume disabled"
(( NUM_SHARDS > 1 )) && echo "Shard      : $SHARD_ID of $NUM_SHARDS"
echo "Started    : $(date)"
[[ "$DRY_RUN" == "1" ]] && echo "DRY RUN    : resolving knobs only"
echo "==================================================================="

# (activate_python ran at the top, before the first derivation — see the block
# above "---- Which dataset". Every path printed above came from that env.)

# ---- Group by context, exactly as ingest does ------------------------------
# The same per-system windows apply: a_mem and simplemem call the MEMORY model
# at query time (keyword generation, query planning), so the endpoint has to be
# the one that system expects, not a single compromise window.
#
# EXCEPT under PROMPT=shared at STAGE=answer, where the per-system split has
# nothing left to key on: no memory system is loaded and no memory model is
# prompted, so the only model in the job is the GENERATOR and it is the same
# model for all six. One window therefore serves everything, and it is the LONG
# one (131072) — the same window the raw-turn baselines are served at, so a
# memory-system answer and a full-context answer are generated under identical
# serving conditions and --context-window means the same number in both. It also
# costs nothing to be generous: a top20 or budget1000 prompt is ~1k tokens, so
# the window here is a guard that should never fire.
# One server start for the whole job instead of two.
declare -a SPECS=() CTXS=()
ANSWER_CTX=""
[[ "$PROMPT" == "shared" && "$MODE" == "native" && "$STAGE" == "answer" ]] && \
  ANSWER_CTX=${MAX_MODEL_LEN:-$CTX_LONG}
for s in $SYSTEMS; do
  spec="$(spec_of "$s")"
  if [[ -f "$spec" ]]; then
    SPECS+=("$s"); CTXS+=("${ANSWER_CTX:-$(ctx_for "$s")}")
  else
    echo "!!! no spec for '$s' at $spec — skipping" >&2
  fi
done
[[ ${#SPECS[@]} -eq 0 ]] && { echo "!!! nothing to run" >&2; exit 1; }
mapfile -t CTX_GROUPS < <(printf '%s\n' "${CTXS[@]}" | sort -n -u)

echo "Context plan (one vLLM start per group):"
for ctx in "${CTX_GROUPS[@]}"; do
  m=""; for i in "${!SPECS[@]}"; do [[ "${CTXS[$i]}" == "$ctx" ]] && m+=" ${SPECS[$i]}"; done
  printf '  max-model-len %-7s gpu-util %-5s :%s\n' "$ctx" "$(gpu_util_for "$ctx")" "$m"
done

# ---- Run ------------------------------------------------------------------
declare -a RESULTS=()
fail=0

run_stage() {   # $1=label  $2..=command
  local label=$1; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  [%s] would run:' "$label"; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  echo "  [$label] $*"
  "$@"
}

# Does any system in this run need the memory/generator LLM for this stage?
# STAGE=retrieve needs it only for the two systems that call the memory model at
# query time; the other four rank locally, and starting vLLM for them would
# spend 10-25 minutes to serve zero requests.
needs_server() {
  local s
  [[ "$STAGE" != "retrieve" ]] && return 0      # answering always generates
  for s in $SYSTEMS; do
    case " $NEEDS_MEM_LLM " in *" $s "*) return 0 ;; esac
  done
  return 1
}
# ---- served embedder -------------------------------------------------------
# The Qwen3-Embedding settings address their server as
# http://localhost:<port>/v1, and inside a job "localhost" is THIS node — an
# embedder submitted separately lands elsewhere and is unreachable. So the model
# and port are read straight out of NATIVE_RETRIEVER and served here.
#
# Derived rather than configured on purpose: the spec already names the model
# and the port, and a separate EMB_MODEL knob could disagree with it — serving
# 0.6B while the run ranks with 4B produces a complete, wrong table with nothing
# in the output to show it. EMB_SERVE=0 opts out when you are running your own.
#
# EMB_SERVE=MODEL@PORT is the explicit form, for the one case derivation cannot
# reach: an UNSWAPPED run whose embedder is named in the system's config rather
# than in NATIVE_RETRIEVER. That is hypermem at its own setting —
# `retrieval_type: rrf` reads `embedding.model` / `:11810` out of
# configs/hypermem/default.yaml, and NATIVE_RETRIEVER is empty because nothing
# is being swapped. Keep the two halves in agreement by hand there: this starts
# a server, it does not tell the config what to rank with.
#
# It runs even under SKIP_SERVER=1 (the block below checks EMB_SPEC_MODEL only),
# which is what a hypermem retrieval job wants — no memory LLM, one embedder.
EMB_SERVE=${EMB_SERVE:-auto}
EMB_SPEC_MODEL=""; EMB_SPEC_PORT=""
if [[ "$EMB_SERVE" == *"@"* ]]; then
  EMB_SPEC_MODEL=${EMB_SERVE%@*}
  EMB_SPEC_PORT=${EMB_SERVE##*@}
  if [[ ! "$EMB_SPEC_PORT" =~ ^[0-9]+$ ]]; then
    echo "!!! EMB_SERVE must be MODEL@PORT (got '$EMB_SERVE')" >&2; exit 1
  fi
  echo ">>> EMB_SERVE explicit: $EMB_SPEC_MODEL on :$EMB_SPEC_PORT"
elif [[ "$EMB_SERVE" != "0" && "$NATIVE_RETRIEVER" == *"@"* ]]; then
  _m=${NATIVE_RETRIEVER#*:}; _m=${_m%%@*}
  _u=${NATIVE_RETRIEVER#*@}
  if [[ "$_u" == *localhost* || "$_u" == *127.0.0.1* ]]; then
    EMB_SPEC_PORT=$(sed -E 's#.*:([0-9]+).*#\1#' <<<"$_u")
    EMB_SPEC_MODEL="$_m"
  else
    echo ">>> NATIVE_RETRIEVER points at $_u — not localhost, so not started here"
  fi
fi

# ---- served cross-encoder reranker -----------------------------------------
# RERANK_SERVE=MODEL@PORT starts HyperMem's cross-encoder. There is no `auto`
# form and deliberately so: the embedder can be derived from NATIVE_RETRIEVER
# because a swap names its model on the command line, but the reranker is named
# only inside the system's YAML, which this script does not parse. Asking for it
# explicitly keeps the script from guessing wrong and starting 8 GB of the wrong
# model.
#
# Needed by exactly one config — configs/hypermem/naive_query.yaml, which sets
# `reranker.enabled: true` to reproduce paper §4.1. Empty (the default) starts
# nothing, which is right for every other run in this project.
#
#   RERANK_SERVE=Qwen/Qwen3-Reranker-4B@12810
#
# THE CONFIG STILL DECIDES WHETHER IT IS USED. This starts a server; it does not
# turn the reranker on. Setting this against a config with `enabled: false`
# wastes a model load and changes no result — and the reverse, `enabled: true`
# with no server, fails the run rather than silently falling back, which is the
# behaviour evalgrid/systems/hypermem/reranker.py's no-retry policy buys.
RERANK_SERVE=${RERANK_SERVE:-}
RERANK_SPEC_MODEL=""; RERANK_SPEC_PORT=""
if [[ -n "$RERANK_SERVE" && "$RERANK_SERVE" != "0" ]]; then
  if [[ "$RERANK_SERVE" != *"@"* ]]; then
    echo "!!! RERANK_SERVE must be MODEL@PORT (got '$RERANK_SERVE')" >&2; exit 1
  fi
  RERANK_SPEC_MODEL=${RERANK_SERVE%@*}
  RERANK_SPEC_PORT=${RERANK_SERVE##*@}
  if [[ ! "$RERANK_SPEC_PORT" =~ ^[0-9]+$ ]]; then
    echo "!!! RERANK_SERVE must be MODEL@PORT (got '$RERANK_SERVE')" >&2; exit 1
  fi
  echo ">>> RERANK_SERVE explicit: $RERANK_SPEC_MODEL on :$RERANK_SPEC_PORT"
fi

if ! needs_server; then
  echo ">>> STAGE=retrieve and no system here calls the memory LLM"
  echo "    ($SYSTEMS) — skipping vLLM entirely."
  SKIP_SERVER=1
fi

# The two servers share one card and vLLM's gpu-memory-utilization is a fraction
# of the WHOLE card, not of what is free — so the memory LLM's share and the
# embedder's must leave headroom between them. Checked here because the failure
# is otherwise a 20-minute wait followed by "died during startup", and because
# the default pairing is genuinely bad for one system: a_mem is in
# LONG_CTX_SYSTEMS, so its memory LLM takes GPU_UTIL_LONG (0.95) and the 0.1 the
# embedder wants does not exist.
#
# At STAGE=retrieve a_mem's only LLM call is short keyword extraction — the
# 131072 window is sized for ingest and answering, not for this — so the fix is
# to shrink the memory LLM rather than starve the embedder:
#   MAX_MODEL_LEN=32768,GPU_UTIL=0.78,EMB_UTIL=0.18
#
# 0.78 IS A FLOOR, NOT A PREFERENCE. Qwen3-30B-A3B-Instruct-2507 is 56.93 GiB of
# weights — measured, logs/vllm/27204949.log "Model loading took 56.93 GiB" —
# which is 0.72 of an 80 GB H100 before one KV block. This line used to advise
# GPU_UTIL=0.6; that is 47.8 GiB and cannot hold the weights at all, so it turned
# a caught misconfiguration into an uncaught "died during startup" 20 minutes
# later. 0.78 leaves ~4.3 GiB of KV (ample: one short prompt per question) and
# ~17 GiB for the embedder, hence 0.18 — enough for Qwen3-Embedding-4B's ~8 GiB.
# The corollary: the memory LLM and BOTH served embedders do not fit at once.
# Cycle them — see scripts/native_grid.sh, which groups the grid's
# eight cells by embedder for exactly this reason.
if [[ -n "$EMB_SPEC_MODEL" && "$SKIP_SERVER" != "1" ]]; then
  for _s in $SYSTEMS; do
    case " $NEEDS_MEM_LLM " in *" $_s "*) ;; *) continue ;; esac
    _u=$(gpu_util_for "$(ctx_for "$_s")")
    # The reranker is a THIRD claim on the same card when it is asked for, so it
    # enters the same sum. In practice this branch and a reranker never meet —
    # hypermem is not in NEEDS_MEM_LLM, so the loop skips it — but a future
    # system that both prompts a memory LLM and reranks would otherwise get the
    # 20-minute "died during startup" instead of this message.
    _side=$EMB_UTIL
    [[ -n "$RERANK_SPEC_MODEL" ]] && _side=$(awk "BEGIN{print $_side + $RERANK_UTIL}")
    if awk "BEGIN{exit !($_u + $_side > 0.98)}"; then
      echo "!!! $_s: memory LLM wants $_u of the GPU and the side servers want $_side" >&2
      echo "!!!   — that is more than the card has, and one of them will fail to start." >&2
      echo "!!! Shrink the memory LLM; at STAGE=retrieve it needs far less than it asks:" >&2
      echo "!!!   MAX_MODEL_LEN=32768 GPU_UTIL=0.78 EMB_UTIL=0.18 bash scripts/query.sh" >&2
      echo "!!! Do not go below 0.78: the weights alone are 56.93 GiB of an 80 GB card." >&2
      exit 1
    fi
  done
fi

# Started ONCE, before the context loop, and left running across it: stop_server
# only kills the memory LLM, so the embedder survives a context-group switch
# instead of being reloaded per group. The EXIT trap in vllm.sh stops it.
if [[ -n "$EMB_SPEC_MODEL" && "$DRY_RUN" != "1" ]]; then
  if ! start_embed_server "$EMB_SPEC_MODEL" "$EMB_SPEC_PORT"; then
    echo "!!! the retriever spec needs that embedder; nothing here can run without it" >&2
    exit 1
  fi
elif [[ -n "$EMB_SPEC_MODEL" ]]; then
  echo "(dry run) would serve $EMB_SPEC_MODEL on :$EMB_SPEC_PORT"
fi

# Started after the embedder and, like it, once before the context loop:
# stop_server only kills the memory LLM, so this survives a context-group
# switch. The EXIT trap in vllm.sh stops it.
if [[ -n "$RERANK_SPEC_MODEL" && "$DRY_RUN" != "1" ]]; then
  if ! start_rerank_server "$RERANK_SPEC_MODEL" "$RERANK_SPEC_PORT"; then
    echo "!!! reranker.enabled: true needs that cross-encoder; a hypermem run" >&2
    echo "!!!   configured for it cannot fall back to first-stage order" >&2
    exit 1
  fi
elif [[ -n "$RERANK_SPEC_MODEL" ]]; then
  echo "(dry run) would serve $RERANK_SPEC_MODEL on :$RERANK_SPEC_PORT"
fi

for ctx in "${CTX_GROUPS[@]}"; do
  if [[ "$DRY_RUN" != "1" && "$SKIP_SERVER" != "1" ]]; then
    stop_server
    if ! start_server "$ctx"; then
      for i in "${!SPECS[@]}"; do
        [[ "${CTXS[$i]}" == "$ctx" ]] && RESULTS+=("NO-SERVER|${SPECS[$i]}|-")
      done
      fail=1; continue
    fi
  fi

  for i in "${!SPECS[@]}"; do
    [[ "${CTXS[$i]}" == "$ctx" ]] || continue
    s=${SPECS[$i]}
    log="$LOG_DIR/${s}_${MODE}_${STAGE}_${JOB}.log"
    echo "==================================================================="
    echo ">>> [$MODE] $s   max-model-len=$ctx"
    echo ">>> log: $log"

    # SCORING is the subset every stage takes: it names WHICH run to look at.
    # COMMON adds --limit, which names how much of it to RUN — eval_retrieval
    # runs nothing (it scores whatever step 2 already wrote, against step 1's
    # qrels, both already limited) and its parser rejects the flag. Passing
    # COMMON there aborted the eval of every cell with "unrecognized arguments:
    # --limit", so a limited grid wrote rankings and zero metrics.
    # MEM_VARIANT has to reach the Python stages too, not only the output dirs
    # (_stage_dir): without it they resolve the memory store from --mem-model
    # alone, so a variant cell (e.g. ingest-gpt-4o-mini) silently read the
    # default store while writing under the variant's name.
    MEMVAR=(); [[ -n "${MEM_VARIANT:-}" ]] && MEMVAR=(--mem-variant "$MEM_VARIANT")
    SCORING=(--system "$s" --dataset "$DATASET" --mem-model "$MEM_MODEL"
             --labeler-model has_answer "${MEMVAR[@]+"${MEMVAR[@]}"}")
    COMMON=("${SCORING[@]}")
    [[ -n "$LIMIT" ]] && COMMON+=(--limit "$LIMIT")
    MEMFLAG=(); [[ -n "$(mem_dir "$s")" ]] && MEMFLAG=(--mem-dir "$(mem_dir "$s")")
    # The calls that take MEMFLAG (qrels, native retrieve/answer) never take
    # SCORING/COMMON, so the variant rides here for them without duplication.
    MEMFLAG+=("${MEMVAR[@]+"${MEMVAR[@]}"}")
    SHARDFLAG=(); (( NUM_SHARDS > 1 )) && SHARDFLAG=(--num-shards "$NUM_SHARDS" --shard-id "$SHARD_ID")
    # Native only: add_trace_args is attached to the native parser group in both
    # retrieve.py and answer.py, so passing these on the flat path would be a
    # parser error (the two halves take disjoint flags, and that is enforced).
    # The MODE test stays; the "!= off" one is GONE. TRACE_FLAGS now always
    # carries an explicit --trace-level, and dropping it at "off" would leave the
    # native path on the Python default, which is "full". The flat path is
    # unaffected either way: with no trace args registered, its recorder falls
    # back to "off" in build_recorder_from_args.
    TRACEFLAG=(); [[ "$MODE" == "native" ]] && TRACEFLAG=("${TRACE_FLAGS[@]}")

    start=$SECONDS
    # rc is assigned INSIDE the block below, and `{ ... } | tee` runs that block
    # in a subshell — so every `|| rc=$?` in it dies with the subshell and an
    # `rc=0` set out here would survive untouched. That reported OK for cells
    # that had failed outright (eight lightmem cells that raised before
    # retrieving a single sample all printed OK). The block therefore ends in
    # `exit $rc` and the status is read back off PIPESTATUS. pipefail (set at
    # the top) is no help here: it governs the pipeline's own status, not a
    # variable assigned inside one stage of it.
    rc=0
    {
      # 1. qrels — no LLM, so it runs even under SKIP_SERVER.
      if [[ "$SKIP_QRELS" != "1" ]]; then
        run_stage qrels python -m evalgrid.pipeline.qrels_has_answer \
          --system "$s" --dataset "$DATASET" --mem-model "$MEM_MODEL" \
          "${MEMFLAG[@]+"${MEMFLAG[@]}"}" --out "$(qrels_dir "$s")" \
          ${LIMIT:+--limit "$LIMIT"} --overwrite || rc=$?
      fi

      # 2. answer (+ retrieval)
      if [[ "$MODE" == "native" && "$STAGE" == "retrieve" ]]; then
        # Retrieval only, saved for later replay. No generator is constructed,
        # and the stage raises if anything reaches for one.
        # --top-k here is RETRIEVAL depth: it goes into the system's own
        # retriever before it runs. Omitted = the config's published depth.
        # NATIVE_TOP_K empty = the published-depth control, and it has to say so:
        # the spec carries native_top_k, which the stage applies unless told
        # otherwise. `auto` (the default) and a number read/use that depth and
        # pass it explicitly, so DRY_RUN shows it and retrieval_dir() names the
        # dir with it.
        NK=(--published-depth); k=$(native_top_k "$s"); [[ -n "$k" ]] && NK=(--top-k "$k")
        # Resolved up front: an empty --out would send retrieve.py to its keyed
        # default, i.e. the published cell RETRIEVAL_OUT exists to protect.
        rout=$(side_retrieval_dir "$s") && [[ -n "$rout" ]] || exit 1
        run_stage retrieve_native python -m evalgrid.pipeline.retrieve \
          --exp "$(spec_of "$s")" \
          --retriever native \
          --mem-base-url "$MEM_BASE_URL" \
          "${MEMFLAG[@]+"${MEMFLAG[@]}"}" "${NK[@]+"${NK[@]}"}" "${SWAPFLAG[@]+"${SWAPFLAG[@]}"}" \
          --out "$rout" \
          "${TRACEFLAG[@]+"${TRACEFLAG[@]}"}" "${OVERWRITEFLAG[@]+"${OVERWRITEFLAG[@]}"}" \
          ${LIMIT:+--limit "$LIMIT"} "${SHARDFLAG[@]+"${SHARDFLAG[@]}"}" || rc=$?

      elif [[ "$MODE" == "native" && "$STAGE" == "answer" ]]; then
        # Replay. The store is still loaded (rendering needs it) but no
        # query-time LLM call is made, so query_retrieval_* reads 0 here — that
        # cost belongs to the retrieve run that produced the directory.
        from_dir=${FROM_RETRIEVAL:-$(retrieval_dir "$s")}
        # The existence check is skipped under DRY_RUN on purpose: a dry run
        # exists to show the command a real run would issue, and the retrieval
        # it consumes legitimately does not exist yet when you are previewing
        # the pair.
        if [[ "$DRY_RUN" != "1" && ! -d "$from_dir" ]]; then
          echo "!!! STAGE=answer but no saved retrieval at $from_dir — run STAGE=retrieve first" >&2
          echo "!!!   SYSTEMS=$s STAGE=retrieve MEM_ROOT=$MEM_ROOT bash scripts/query.sh" >&2
          rc=1
        elif [[ "$PROMPT" == "shared" ]]; then
          # THE UNIFIED SETTING. Same saved ranking, different code path: this
          # one re-reads the run.tsv against the qrels corpus and renders it with
          # evalgrid.systems.base.shared_answer_prompt — the prompt the raw-turn
          # baselines answer under, which is what puts a memory-system row and a
          # naive-RAG row on one axis.
          #
          # No memory system is constructed here, so:
          #   * no --mem-dir / --mem-base-url (the parser rejects both — no
          #     native loop exists to configure), and no store is read;
          #   * no --native-retriever either. The swap is already baked into the
          #     ranking this reads; $from_dir is what names it (answers_dir
          #     inherits that basename), so passing it again would be both
          #     meaningless and refused.
          #   * --top-k counts CORPUS DOCS, i.e. one memory each, for all six
          #     systems — including memorybank, whose date blocks exist only
          #     inside its own loop. So the memory_atoms machinery the native
          #     replay needs is not in play: k is exact by
          #     construction on this path.
          #
          # --out is the predictions.jsonl PATH here, not a directory: the shared
          # path takes --out literally as the file (the native path takes it as
          # the dir). Getting this wrong writes a file named after the cell.
          run_stage answer_shared python -m evalgrid.pipeline.answer \
            --retriever native --prompt shared \
            "${COMMON[@]}" \
            --gen-model "$GEN_MODEL" --gen-base-url "$GEN_BASE_URL" \
            --qrels-dir "$(qrels_dir "$s")" --retrieval-dir "$from_dir" \
            ${ANSWER_TOP_K:+--top-k "$ANSWER_TOP_K"} \
            ${CONTEXT_BUDGET:+--context-budget "$CONTEXT_BUDGET"} \
            ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"} \
            "${ANSWER_OVERWRITEFLAG[@]+"${ANSWER_OVERWRITEFLAG[@]}"}" \
            --context-window "$ctx" --concurrency "$CONCURRENCY" \
            --out "$(answers_dir "$s")/predictions.jsonl" || rc=$?
        else
          # --qrels-dir: a cut to k MEMORIES or to a token budget needs the
          # per-memory texts behind a GROUPED unit, and memorybank's blocks are
          # several memories concatenated. The corpus reconstructs them exactly
          # (pipeline/answer.memory_atoms). Passed because OUT_ROOT moves the
          # qrels tree, so the stage's own default would look in outputs/_qrels
          # while step 1 above wrote them under $OUT_ROOT.
          run_stage answer_replay python -m evalgrid.pipeline.answer --retriever native --prompt native \
            --exp "$(spec_of "$s")" \
            --gen-model "$GEN_MODEL" --gen-base-url "$GEN_BASE_URL" \
            --mem-base-url "$MEM_BASE_URL" \
            "${MEMFLAG[@]+"${MEMFLAG[@]}"}" "${SWAPFLAG[@]+"${SWAPFLAG[@]}"}" \
            --from-retrieval "$from_dir" --qrels-dir "$(qrels_dir "$s")" \
            ${ANSWER_TOP_K:+--top-k "$ANSWER_TOP_K"} \
            ${CONTEXT_BUDGET:+--context-budget "$CONTEXT_BUDGET"} \
            ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"} \
            "${ANSWER_OVERWRITEFLAG[@]+"${ANSWER_OVERWRITEFLAG[@]}"}" \
            --out "$(answers_dir "$s")" --retrieval-out "" \
            "${TRACEFLAG[@]+"${TRACEFLAG[@]}"}" \
            ${LIMIT:+--limit "$LIMIT"} "${SHARDFLAG[@]+"${SHARDFLAG[@]}"}" || rc=$?
        fi

      elif [[ "$MODE" == "native" ]]; then
        # One pass: predictions AND the run.tsv of what its own retrieval
        # touched. --mem-base-url is required here and nowhere else — a_mem and
        # simplemem call the memory model at query time.
        # NATIVE_TOP_K empty = the published-depth control, and it has to say so:
        # the spec carries native_top_k, which the stage applies unless told
        # otherwise. `auto` (the default) and a number read/use that depth and
        # pass it explicitly, so DRY_RUN shows it and retrieval_dir() names the
        # dir with it.
        NK=(--published-depth); k=$(native_top_k "$s"); [[ -n "$k" ]] && NK=(--top-k "$k")
        run_stage answer_native python -m evalgrid.pipeline.answer --retriever native --prompt native \
          --exp "$(spec_of "$s")" \
          --gen-model "$GEN_MODEL" --gen-base-url "$GEN_BASE_URL" \
          --mem-base-url "$MEM_BASE_URL" \
          "${MEMFLAG[@]+"${MEMFLAG[@]}"}" "${NK[@]+"${NK[@]}"}" "${SWAPFLAG[@]+"${SWAPFLAG[@]}"}" \
          ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"} \
          "${ANSWER_OVERWRITEFLAG[@]+"${ANSWER_OVERWRITEFLAG[@]}"}" \
          --out "$(answers_dir "$s")" --retrieval-out "$(retrieval_dir "$s")" \
          "${TRACEFLAG[@]+"${TRACEFLAG[@]}"}" \
          ${LIMIT:+--limit "$LIMIT"} "${SHARDFLAG[@]+"${SHARDFLAG[@]}"}" || rc=$?
      else
        EMB=(); [[ "$MODE" == "dense" ]] && EMB=(--emb-model "$EMB_MODEL" --emb-base-url "$MEM_BASE_URL")
        # No --top-k here: a flat retrieval run.tsv is a FULL ranking, and the
        # depth is applied by the answer stage when it builds the prompt. That
        # is what lets one retrieval serve every k in a sweep.
        run_stage retrieve python -m evalgrid.pipeline.retrieve \
          "${COMMON[@]}" --retriever "$MODE" \
          "${EMB[@]+"${EMB[@]}"}" \
          --qrels-dir "$(qrels_dir "$s")" --out "$(retrieval_dir "$s")" || rc=$?
        run_stage answer python -m evalgrid.pipeline.answer \
          "${COMMON[@]}" --retriever "$MODE" --top-k "$TOP_K" \
          --gen-model "$GEN_MODEL" --gen-base-url "$GEN_BASE_URL" \
          --qrels-dir "$(qrels_dir "$s")" --retrieval-dir "$(retrieval_dir "$s")" \
          --out "$(answers_dir "$s")" || rc=$?
      fi

      # 3. retrieval metrics — CPU, and scored against the step-1 qrels so a
      # native row and a bm25 row are the same measurement.
      #
      # Skipped on STAGE=answer: that half writes no ranking (--retrieval-out "")
      # because the ranking already exists and re-scoring it would produce a
      # second identical metrics file under a different run.
      #
      # `rc -eq 0` is load-bearing, not tidiness: step 2 crashing partway leaves
      # a run dir holding the samples it reached, and scoring that publishes a
      # number under the failed cell's name that reads exactly like a finished
      # one. Two cells of the 2026-08-21 smoke grid did that after CUDA OOM —
      # 2 of 4 samples, metrics.json written, FAILED in the same log. The stage
      # refuses such a dir on its own now (no retrieval_usage.json); this stops
      # the driver asking in the first place, and covers a step-1 qrels failure
      # too, where the gold set itself is what is half-built.
      if [[ -n "$RETRIEVAL_OUT" ]]; then
        echo ">>> [eval] skipped — RETRIEVAL_OUT set; the published metrics stand"
      elif [[ "$STAGE" != "answer" && $rc -eq 0 ]]; then
        # --out is a metrics FILE path, and eval_retrieval does not create its
        # parent — passing the directory raises FileNotFoundError.
        [[ "$DRY_RUN" == "1" ]] || mkdir -p "$(eval_dir "$s")"
        # The gold level is part of the metrics FILENAME, and --out overrides the
        # name eval_retrieval would have picked — so the suffix has to be spelled
        # here too. Without it two runs scored at different levels land on one
        # metrics.json and the second silently replaces the first, which is the
        # one thing gold_suffix exists to prevent. Mirrors
        # evalgrid.benchmarks.gold.gold_suffix, which is never empty.
        run_stage eval_retrieval python -m evalgrid.pipeline.eval_retrieval \
          "${SCORING[@]}" --retriever "$MODE" --gold "$GOLD" \
          --qrels-dir "$(qrels_dir "$s")" --retrieval-dir "$(retrieval_dir "$s")" \
          --out "$(eval_dir "$s")/metrics-gold-$GOLD.json" || rc=$?
      elif [[ "$STAGE" != "answer" ]]; then
        echo "!!! skipping eval_retrieval: an earlier stage failed (rc=$rc), so this" >&2
        echo "!!!   cell's run dir is partial and any metrics from it would be a" >&2
        echo "!!!   different measurement wearing the same run name." >&2
      fi
      exit $rc
    } 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    elapsed=$((SECONDS - start))
    if [[ $rc -eq 0 ]]; then
      RESULTS+=("OK|$s|${elapsed}s"); echo ">>> OK  $s in ${elapsed}s"
    else
      RESULTS+=("FAIL($rc)|$s|${elapsed}s")
      echo "!!! FAILED ($rc): $s after ${elapsed}s — see $log" >&2
      fail=1
    fi
  done
done
[[ "$DRY_RUN" != "1" ]] && stop_server

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==================================================================="
  echo "Dry run complete — nothing executed."
  exit 0
fi

# ---- Summary ---------------------------------------------------------------
echo "==================================================================="
echo "Answer results — job $JOB, mode $MODE, finished $(date)"
printf '%-12s %-16s %8s\n' "STATUS" "SYSTEM" "WALL"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r st nm el <<< "$r"; printf '%-12s %-16s %8s\n' "$st" "$nm" "$el"
done

echo
echo "Retrieval metrics:"
for s in $SYSTEMS; do
  f="$(eval_dir "$s")/metrics.json"
  [[ -f "$f" ]] && echo "  $s: $f" || echo "  $s: (none written)"
done

# A trace run's whole product is the trajectories, so list them rather than
# leaving the reader to guess which dirs got one — "full" writes untruncated
# payloads and a single sample can be tens of MB. Mirrors ingest.sh.
if [[ "$MODE" == "native" && "$TRACE_LEVEL" != "off" ]]; then
  echo
  echo "Trajectories (trajectory-${TRACE_LEVEL}.jsonl, $TRACE_SCOPE):"
  for s in $SYSTEMS; do
    find "$(retrieval_dir "$s")" "$(answers_dir "$s")" \
        -name "trajectory-${TRACE_LEVEL}.jsonl" -exec du -h {} + 2>/dev/null | sort -k2
  done | sort -u || echo "  (none written)"
  echo "  Read one: python -m evalgrid.tools.inspect_trajectory <path>"
fi

echo
echo "Next — judge the answers. No GPU needed, so run it on a LOGIN node:"
for s in $SYSTEMS; do
  echo "  python -m evalgrid.pipeline.judge \\"
  echo "      --predictions $(answers_dir "$s")/predictions.jsonl \\"
  echo "      --dataset $DATASET --judge-model gpt-4o-mini"
done

[[ $fail -ne 0 ]] && { echo; echo "At least one system FAILED — logs in $LOG_DIR/" >&2; exit 1; }
exit 0
