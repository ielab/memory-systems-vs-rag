#!/bin/bash
# The native retriever grid for ONE system — all eight settings in one
# allocation.
#
# WHY ONE JOB PER SYSTEM AND NOT ONE PER CELL. The retrieval itself is seconds:
# measured on outputs/_smoke4 at published depth, 7s (a_mem) to 41s (hypermem)
# for four samples. Everything else is server startup — 10-25 min for the memory
# LLM, 1-5 min for an embedder. Submitted one cell at a time, the grid is 48
# allocations that spend ~99% of their GPU time loading models they could have
# shared. Here the memory LLM loads ONCE and the eight cells run against it.
#
# THE CELLS ARE GROUPED BY EMBEDDER, because the card cannot hold everything at
# once (see GPU budget below):
#
#   group 0  bm25, sparse:splade-v3, dense:MiniLM, rrf:MiniLM   no server
#   group 1  dense/rrf Qwen3-Embedding-0.6B                     :8001
#   group 2  dense/rrf Qwen3-Embedding-4B                       :8002
#
# The memory LLM stays up across all three; only the embedder is cycled, and a
# 4B embedder loads in a minute or two. Same shape as query.sh's CTX_GROUPS,
# on the embedder axis instead of the context axis.
#
# One system per run:
#
#   SYS=hypermem MEM_ROOT= LIMIT= bash scripts/native_grid.sh
#
# ANSWERING THE GRID. STAGE=answer replays each cell's saved retrieval, and
# ANSWER_CELLS fans each one out over the answer-side cuts — so eight retrievers
# x two cuts is sixteen answer cells in ONE allocation, still one model load:
#
#   STAGE=answer ANSWER_CELLS=top5+budget1000 ...
#
# Two things differ from the retrieve stage and are handled below rather than
# left to the caller. Answering ALWAYS needs the generator, including for the
# four systems that never prompt the memory model at query time — NEEDS_MEM_LLM
# is a statement about retrieval. And answering needs NO EMBEDDER at all: a
# replay ranks nothing, it re-renders a saved ranking, so the two embedder
# groups below serve no request and are skipped.
#
# THE E1 ROW — the system's own retrieval, no swap. GRID_RETRIEVERS=native is
# one unswapped cell, and it is what the two answering budgets
# (top-k and tokens) are measured on:
#
#   STAGE=answer GRID_RETRIEVERS=native ANSWER_CELLS=top3+top5+top10+top20 ...
#
# Knobs: SYS (required), MEM_ROOT, LIMIT, STAGE (retrieve), GRID_RETRIEVERS
# (default: the spec's native_retrievers; `native` = no swap), ANSWER_CELLS
# (STAGE=answer only), NATIVE_VARIANT (grid; empty = the published run),
# NATIVE_TOP_K, TRACE_SAMPLES (all at retrieve, 1 at answer),
# SKIP_QRELS (1 = reuse an existing gold set; required when splitting the
# grid across concurrent jobs), RETRIEVE_OVERWRITE (1 = re-run finished samples
# rather than resume past them; inherited by query.sh), DRY_RUN.
#

# No `set -e`: one cell failing must not skip the other seven.
set -uo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
cd "$REPO_DIR" || { echo "!!! repo dir not found: $REPO_DIR" >&2; exit 1; }

SYS=${SYS:?set SYS=<system>}
STAGE=${STAGE:-retrieve}

# SYS may name SEVERAL systems, and ONLY at STAGE=answer. That is not caution
# for its own sake — it is exactly where the GPU sizing stops depending on which
# system is running:
#
#   STAGE=retrieve  the LLM window is sized by what THAT system SENDS at query
#                   time (a_mem 8192, simplemem 24576) and the embedder window
#                   by ITS unit lengths, so two systems in one job would run one
#                   of them under the other's budget.
#   STAGE=answer    nothing prompts the memory model — install_replay makes no
#                   query-time call — and every system generates through the one
#                   shared generator at 32768/0.8. One server serves all six.
#
# So the whole answering grid is ONE allocation with ONE model load, instead of
# six jobs each paying 10-25 min of startup to run minutes of generation.
read -ra SYS_LIST <<< "$SYS"
if (( ${#SYS_LIST[@]} > 1 )) && [[ "$STAGE" != "answer" ]]; then
  echo "!!! SYS names ${#SYS_LIST[@]} systems, which is STAGE=answer only (got STAGE=$STAGE)." >&2
  echo "!!!   At STAGE=retrieve the LLM window, the embedder window and the GPU" >&2
  echo "!!!   fractions are all sized per system — see the GPU budget block below." >&2
  echo "!!!   Run retrieval one system at a time: SYS=<one> bash $0" >&2
  exit 1
fi
# Single dash, matching LIMIT below: MEM_ROOT= (explicitly empty) means the
# keyed default tree, which is what a real run wants. `:-` would silently send
# it to the smoke tree.
MEM_ROOT=${MEM_ROOT-outputs/_smoke4}
LIMIT=${LIMIT-4}
# Single dash: NATIVE_VARIANT= (explicitly empty) must mean "no variant", which
# is what the system's OWN published run is keyed under. With `:-` an empty value
# silently became `grid` and the job looked for native-...-top270--grid.
NATIVE_VARIANT=${NATIVE_VARIANT-grid}
DRY_RUN=${DRY_RUN:-0}

# How many samples each cell records a trajectory for, forwarded to query.sh
# (which resolves it into --trace-samples; empty there means ALL, expressed by
# not passing the flag at all).
#
# The default is per stage, because the two stages record objects of very
# different size. A RETRIEVE trajectory is the ranking and the calls that
# produced it — kilobytes, and the whole point of the grid is that every cell is
# inspectable afterwards, so the retrieve stage traces EVERY sample. An ANSWER
# trajectory carries the rendered prompt, so at `full` a single sample can be
# tens of MB and 500 samples x 16 cells is tens of GB; that stage stays capped
# at the first sample per shard.
#
# Single dash on both, so TRACE_SAMPLES= on the submit line means "all" rather
# than "use the default" — the same convention as MEM_ROOT and LIMIT above.
_TRACE_SAMPLES_DEFAULT=
[[ "$STAGE" == "answer" ]] && _TRACE_SAMPLES_DEFAULT=1
TRACE_SAMPLES=${TRACE_SAMPLES-$_TRACE_SAMPLES_DEFAULT}

# Empty (the default) = the per-cell rule below: build the gold set on the first
# cell of each system, reuse it for the rest. Set SKIP_QRELS=1 to reuse an
# EXISTING gold set for every cell, including the first.
#
# That knob is what makes a SPLIT grid safe. Two jobs over disjoint retrievers —
# one per embedder group, sized to their own walltimes — would each rebuild the
# same qrels dir with --overwrite on their first cell, and two writers over the
# directory a third process is reading is a torn gold set, not a redundant one.
# Build the qrels once (any single job, or one cell), then submit the splits
# with SKIP_QRELS=1 and they can run at the same time.
SKIP_QRELS=${SKIP_QRELS:-}

# The answer-side cut applied to each cell's saved ranking: top<k> (memories),
# budget<T> (tokens), or a bare number (a budget). Empty = one uncut cell per
# retriever, which is what the `all` segment means.
#
# Separated by `+`, `:`, space or comma, e.g. ANSWER_CELLS=top5+budget1000.
#
# WHICH PROMPT the answer cells are generated under. `shared` (the default) is
# the unified prompt — the one evalgrid.systems.base defines, and the only one
# the raw-turn baselines have — so a memory-system cell and a naive-RAG cell
# differ by memory representation, not by wording. `native` is each system's own
# and is a labelled control; it lands under a `--native-prompt` path suffix, so
# the two can be run over the same rankings without colliding. Forwarded to
# query.sh, which owns the two code paths.
PROMPT=${PROMPT:-shared}
case "$PROMPT" in
  shared|native) ;;
  *) echo "!!! PROMPT must be shared or native — got '$PROMPT'" >&2; exit 1 ;;
esac
# Inert at STAGE=retrieve (retrieval has no prompt), so it is not an error there
# — the default simply carries through to whatever answer job follows.
#
# The shared path answers questions concurrently rather than in shards — it
# makes no query-time memory call, so nothing serialises them.
CONCURRENCY=${CONCURRENCY:-8}

ANSWER_CELLS=${ANSWER_CELLS:-}
ANSWER_CELLS=${ANSWER_CELLS//[,+:]/ }
if [[ -n "$ANSWER_CELLS" && "$STAGE" != "answer" ]]; then
  echo "!!! ANSWER_CELLS cuts a SAVED ranking and applies to STAGE=answer only" >&2
  echo "!!!   (got STAGE=$STAGE). Retrieve the grid first, then answer over it:" >&2
  echo "!!!   SYS=<one> STAGE=retrieve            bash $0" >&2
  echo "!!!   SYS=<one> STAGE=answer ANSWER_CELLS=top5+budget1000 bash $0" >&2
  exit 1
fi
for _c in $ANSWER_CELLS; do
  [[ "$_c" =~ ^(top|budget)?[0-9]+$ ]] && continue
  echo "!!! ANSWER_CELLS takes top<k>, budget<T> or a bare number — got '$_c'" >&2
  exit 1
done
# "No cut" is ONE uncut pass, not zero passes — and it has to be an ARRAY
# holding one empty element to stay that way. A string of whitespace looks like
# one element and word-splits to none, which silently turned STAGE=retrieve into
# a job that started its servers and ran nothing.
read -ra ANSWER_CUTS <<< "$ANSWER_CELLS"
(( ${#ANSWER_CUTS[@]} == 0 )) && ANSWER_CUTS=("")

mkdir -p logs/{retrieve,answer,query,vllm}
JOB=${JOB:-$$}

# Captured BEFORE sourcing vllm.sh, which hard-defaults EMB_UTIL to 0.1. A
# `${EMB_UTIL:-0.18}` written after the source would always find 0.1 already
# there and silently keep it — the GPU sizing below would then be a comment
# describing something that never happened.
WANT_MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
WANT_GPU_UTIL=${GPU_UTIL:-}
WANT_EMB_UTIL=${EMB_UTIL:-}

source "$REPO_DIR/scripts/lib/vllm.sh"
activate_python

# GEN_MODEL / MEM_MODEL / DATASET come from the spec and are identical across
# all six, so one source is enough for the server config below. The per-system
# CELL LIST is read separately rather than by re-sourcing: load_spec exports
# with `${VAR:-value}` precedence, so a second source inside a system loop would
# silently keep the FIRST system's native_retrievers.
SPEC="$(spec_of "${SYS_LIST[0]}")"
source "$REPO_DIR/experiments/load_spec.sh" || exit 1

# The cell list comes from the spec, never from here, so it cannot drift from
# the flat control column beside it (experiments/<sys>/$SPEC_NAME).
# Fails LOUDLY. A spec that cannot be read, or one with no native_retrievers,
# used to come back as an empty list — and an empty list means "run nothing",
# which a job reports as success. In a multi-system grid that is six systems
# silently skipped.
cells_for() {
  local out
  out=$(python -c 'import sys,yaml; print(" ".join(yaml.safe_load(open(sys.argv[1])).get("native_retrievers") or []))' \
        "$REPO_DIR/$(spec_of "$1")") || {
    echo "!!! could not read native_retrievers from $(spec_of "$1")" >&2
    exit 1
  }
  [[ -n "${out// /}" ]] || {
    echo "!!! $(spec_of "$1") lists no native_retrievers, so this" >&2
    echo "!!!   job would run zero cells for $1 and exit 0. Add the key, or name" >&2
    echo "!!!   the cells with GRID_RETRIEVERS." >&2
    exit 1
  }
  echo "$out"
}

# GRID_RETRIEVERS=native is not a retriever name — it means "no swap at all":
# ONE cell, the system's own published query loop, which is the row the two
# answering budgets are measured on. It is a sentinel rather
# than an empty GRID_RETRIEVERS because empty already means "fall back to the
# spec's eight", and the two must not be confused.
#
# That cell's NATIVE_RETRIEVER is the empty string, so query.sh derives
# `native-top<k>` — no encoder slug, no variant — and the answers land beside
# the published retrieval instead of under a --grid sibling that never ran.
#
# `native+grid` is BOTH: the published row and all eight swaps, which is what
# makes one submission cover every retriever a system has.
[[ "${GRID_RETRIEVERS:-}" == "native" ]] && NATIVE_VARIANT=""
cell_list_for() {          # $1 = system -> one cell per line ("" = no swap)
  case "${GRID_RETRIEVERS:-spec}" in
    native)      echo "" ;;
    native+grid) echo ""; cells_for "$1" | tr ' ' '\n' ;;
    spec)        cells_for "$1" | tr ' ' '\n' ;;
    *)           echo "${GRID_RETRIEVERS//,/ }" | tr ' ' '\n' ;;
  esac
}

# hypermem's swap REQUIRES retrieval_type: keyword — under the shipped `rrf`
# check_supported refuses it and the cell dies deriving its own output dir
#.
CONFIG=${CONFIG:-}          # explicit override; the hypermem default is per-system

# ---- GPU budget ------------------------------------------------------------
# Qwen3-30B-A3B-Instruct-2507 is 56.93 GiB of weights, measured
# (logs/vllm/27204949.log: "Model loading took 56.93 GiB"). On an 80 GB H100
# that is 0.72 of the card before a single KV block, which is why:
#
#   * 0.6 (what query.sh used to suggest) is 47.8 GiB and cannot hold the
#     weights at all. 0.78 holds them and was assumed to leave ~4.3 GiB of KV.
#     It does not. MEASURED at 0.78 on an H100 80GB (logs/vllm/27418634_ctx32768
#     .log), the accounting is:
#
#         0.78 x 79.65 GiB   = 62.13 GiB budget
#         - weights            56.93
#         - CUDA graphs         0.65   (profiled since vLLM 0.21; it says 0.78
#                                       is "equivalent to 0.7718" without them)
#         - activations etc.    ~2.8
#         = KV cache            1.74 GiB
#
#     and 32768 needs 3.0 GiB of KV to admit ONE request. So vLLM refused to
#     start: "estimated maximum model length is 19040". Both a_mem and simplemem
#     died there, before a single cell ran.
#
#     The window cannot simply be cut to fit, because it is not sized by the
#     PROMPT. simplemem sends max_tokens 16000 on every call including its
#     query-time planning ones (configs/simplemem/default.yaml:55, and it is
#     explicit for the reason in lib/vllm.sh:96-103), so its window has to cover
#     ~3k of prompt plus that 16k budget. Even 24576 needs 2.25 GiB and still
#     does not fit at 0.78. Raising the fraction is the only lever.
#
#     0.80 buys (0.80 - 0.78) x 79.65 = 1.59 GiB more, so KV = 3.33 GiB, which
#     clears the 3.0 GiB that 32768 demands.
#
#     0.80 rather than a number picked to maximise headroom: it is ALREADY this
#     repo's pairing for a 32768 window — lib/vllm.sh sets GPU_UTIL_SHORT=0.8
#     and gpu_util_for() hands it to every CTX_SHORT server. The grid running
#     0.78 was the anomaly, and a third fraction here would have been a fourth
#     thing to keep in sync for no gain.
#   * the embedder takes the rest. EMB_UTIL drops 0.18 -> 0.16 to keep the pair
#     at the 0.96 total that already ran (0.78 + 0.18): 0.16 x 79.65 = 12.7 GiB,
#     still ample for Qwen3-Embedding-4B (~8 GiB of weights), and the reason the
#     two embedders are cycled rather than both held.
#
#     If the memory LLM still will not start, the next lever is MAX_MODEL_LEN,
#     not the fraction: 20480 covers both systems (a_mem needs ~5k — max_tokens
#     4000 plus a short keyword prompt — and simplemem ~19k).
#
# The four systems that never prompt the memory LLM start no memory LLM, so
# their embedder has the whole card.
# NEEDS_MEM_LLM is about RETRIEVAL — which systems prompt the memory model
# inside their query loop. Answering prompts it by definition, for every system,
# so the stage decides this before the system does. Without that, a STAGE=answer
# grid for lightmem/mem0/memorybank/hypermem started no server and sent every
# generation to a port with nothing behind it.
NEEDS_MEM_LLM=${NEEDS_MEM_LLM:-"a_mem simplemem"}
case " $NEEDS_MEM_LLM " in
  *" $SYS "*) MEM_LLM=1 ;;
  *)          MEM_LLM=0 ;;
esac
[[ "$STAGE" != "retrieve" ]] && MEM_LLM=1

# A replay ranks nothing — install_replay reads the saved run — so the embedder
# groups are dead weight when answering: two model loads and a share of the card
# to serve zero requests.
SERVE_EMB=1
[[ "$STAGE" == "answer" ]] && SERVE_EMB=0

# THREE consumers share this card, not two, and every number below is measured
# off a failure or a startup log rather than assumed.
#
#   memory LLM     56.93 GiB of weights + 0.65 CUDA graphs + activations. At
#                  util 0.78 vLLM reported KV 1.74, so its non-KV footprint is
#                  0.78 x 79.17 - 1.74 = 60.01 GiB, plus ~0.97 of process
#                  context that sits OUTSIDE the fraction.
#   embedder       Qwen3-Embedding-4B: 7.56 weights + 1.28 overhead + 144 KiB
#                  per token of KV. 0.6B: 1.12 + 0.80 + 109 KiB/token.
#   the cell ITSELF  a_mem and simplemem build their own SentenceTransformer
#                  in-process. a_mem's is all-MiniLM-L6-v2 (~0.35 GiB);
#                  simplemem's is Qwen3-Embedding-0.6B (~1.80) — a second copy
#                  of the model already being served, because its dispatch tests
#                  name.startswith("qwen3") against "Qwen/Qwen3-Embedding-0.6B"
#                  and loses. simplemem's is now ~0 IN THIS JOB: a declared swap
#                  substitutes _UnusedEncoder for it (evalgrid/systems/simplemem/
#                  system.py), since the swap replaces the only call that reads
#                  its vectors. It was the last 1.80 GiB that made the sums
#                  below fail — the 0.6B cells of the 2026-08-21 grid died in
#                  add_entries with CUDA OOM. An UNSWAPPED cell
#                  (GRID_RETRIEVERS=native) still pays it, so the figures below
#                  are left at the worst case.
#
# What failed at 0.8 / EMB_UTIL 0.16: 64.31 + 13.07 + 1.76 = 79.14 of 79.17,
# i.e. "5.25 MiB is free" and a 12 MiB allocation short. And the 4B never got
# that far — at its native 40960 window its floor is 14.47 GiB (EMB_UTIL 0.183),
# so it died at 0.16 while the 0.6B beside it started (floor 0.067).
#
# TWO levers, and they are not interchangeable:
#
#   max-model-len sets the FLOOR (KV to admit one request at that length).
#   gpu-memory-utilization sets the TAKE — vLLM fills whatever fraction it is
#   given, spending the surplus on extra KV. So shrinking a window frees nothing
#   on its own; it frees memory only by making a lower fraction viable, and the
#   fraction has to come down with it.
#
# The LLM's window is therefore sized per system, by what that system SENDS,
# which is what makes room for a bigger embedder window than 32768/0.8 allowed:
#
#   a_mem      max_tokens 4000 + a short keyword prompt -> 8192  (floor .767)
#   simplemem  max_tokens 16000 on EVERY call, including its query-time
#              planners (configs/simplemem/default.yaml:55), + ~3k of prompt
#              -> 24576 (floor .786). This one cannot go much lower: no window
#              under ~20k can hold prompt+budget however short the prompt is.
#
# Resulting budgets, both verified against the floors above:
#
#   a_mem      LLM 8192 @0.78 = 62.72 | EMB 16384 @0.15 = 12.28 | own 0.35
#                                                       -> 75.35, spare 3.82
#   simplemem  LLM 24576 @0.79 = 63.51 | EMB 16384 @0.15 = 12.28 | own 1.80
#                                                       -> 77.59, spare 1.58
#
# The EMBEDDER's window is sized per system too, and for the same reason: it is
# the units of THAT system that get embedded, and the two distributions are not
# remotely alike (approx tokens, chars/3.5, over outputs/_smoke4 qrels corpora):
#
#            units    p50    p95    p99    max
#   a_mem     2180    585   1224   1491   4906     <- notes that can be a session
#   simplemem 2941     55     96    126    206     <- one-fact restatements
#
# So 16384 is 3x a_mem's worst case and 80x simplemem's. Sizing each to its own
# corpus costs simplemem nothing and buys back the margin where the card is
# actually tight — its own duplicate encoder is what leaves it thinnest:
#
#   a_mem      16384 (3x)   4B floor 11.05, util .15 -> +0.82 slack, card +3.82
#   simplemem   4096 (20x)  4B floor  9.36, util .13 -> +0.93 slack, card +3.16
#
# Both are set well clear of the observed maximum on purpose: vLLM REJECTS an
# over-long embed rather than truncating it, so a window that is merely typical
# turns one long unit into a dead cell. These maxima come from 4 questions —
# a_mem's 3x is the one to revisit if a full run starts losing 4B cells, and
# lib/vllm.sh will name the ceiling in the failure.
if (( MEM_LLM )); then
  if [[ "$STAGE" == "retrieve" ]]; then
    case "$SYS" in
      a_mem)     MAX_MODEL_LEN=${WANT_MAX_MODEL_LEN:-8192}
                 GPU_UTIL=${WANT_GPU_UTIL:-0.78}
                 _EMB_WIN=16384 ; _EMB_UTIL=0.15 ;;
      simplemem) MAX_MODEL_LEN=${WANT_MAX_MODEL_LEN:-24576}
                 GPU_UTIL=${WANT_GPU_UTIL:-0.79}
                 _EMB_WIN=4096  ; _EMB_UTIL=0.13 ;;
      # Unreachable while NEEDS_MEM_LLM is its default, but that is a knob — so
      # it takes the conservative pair, not either system's tuned one.
      *)         MAX_MODEL_LEN=${WANT_MAX_MODEL_LEN:-32768}
                 GPU_UTIL=${WANT_GPU_UTIL:-0.8}
                 _EMB_WIN=16384 ; _EMB_UTIL=0.15 ;;
    esac
  else
    # ANSWERING generates for every system and the prompt is a whole cell's
    # ranking, so the per-system retrieve windows above do not apply. And the
    # GENERATOR IS THE ONLY MODEL ON THE CARD here: SERVE_EMB=0 just above (a
    # replay ranks nothing), and under PROMPT=shared no memory system is even
    # constructed. So it takes the long pairing — 131072/0.95, lib/vllm.sh's
    # probed CTX_LONG — rather than the 32768/0.8 this used to pin.
    #
    # Three reasons that is the right default and not merely a safe one:
    #   * it is the SAME window the raw-turn baselines are served at
    #     (LONG_CTX_SYSTEMS), so a memory-system row and a full-context row are
    #     generated under identical serving conditions;
    #   * --context-window is passed as the same number, so the context-window
    #     guard means one thing across every row of the table;
    #   * nothing here is close to binding — a top20 or budget1000 prompt is
    #     ~1k tokens, and hypermem's uncut block ~9k — which is what makes an
    #     over-generous window free rather than wasteful.
    MAX_MODEL_LEN=${WANT_MAX_MODEL_LEN:-${CTX_LONG:-131072}}
    GPU_UTIL=${WANT_GPU_UTIL:-${GPU_UTIL_LONG:-0.95}}
    _EMB_WIN=16384 ; _EMB_UTIL=0.15
  fi
  # Set only on this branch: a system with no memory LLM has the card to itself
  # at EMB_UTIL 0.4 and should keep the model's own window.
  if (( SERVE_EMB )); then
    EMB_UTIL=${WANT_EMB_UTIL:-$_EMB_UTIL}
    EMB_MAX_LEN=${EMB_MAX_LEN:-$_EMB_WIN}
  fi
else
  EMB_UTIL=${WANT_EMB_UTIL:-0.4}
fi

# ---- Server lifecycle ------------------------------------------------------
# vllm.sh tracks ONE embedder pid, which is all we need — the groups are
# sequential, so at most one embedder is ever up. Its EXIT trap covers both.
serve_group_embedder() {   # $1=model  $2=port ; "" "" = no embedder for this group
  stop_embed_server
  (( SERVE_EMB )) || return 0
  [[ -z "$1" ]] && return 0
  if [[ "$DRY_RUN" == "1" ]]; then echo "(dry run) would serve $1 on :$2 at $EMB_UTIL"; return 0; fi
  # The window the client must cut to, from the same variable this server is
  # started with. VLLMEmbedder asks /v1/models first and only falls back to
  # this, but not every vLLM reports max_model_len on its card — and when it
  # does not, an over-long document 400s and costs a sample. Exported here
  # because this is the one place that KNOWS the number.
  export EVALGRID_EMB_MAX_INPUT_TOKENS="${EMB_MAX_LEN:-}"
  start_embed_server "$1" "$2" "$EMB_UTIL"
}

# WHICH MODEL the one server holds. start_server serves $MEM_MODEL, but at
# STAGE=answer under PROMPT=shared nothing prompts the memory model — the only
# client is the generator — so a GEN_MODEL override (a generator sweep) must be
# what is served, or every request names a model the server does not have.
# MEM_MODEL itself is left alone: it keys the store/retrieval/answers paths,
# and changing it would send the replay looking for a retrieval that never ran.
SERVE_MODEL=$MEM_MODEL
if [[ "$STAGE" == "answer" && -n "${GEN_MODEL:-}" && "$GEN_MODEL" != "$MEM_MODEL" ]]; then
  if [[ "$PROMPT" != "shared" ]]; then
    echo "!!! GEN_MODEL=$GEN_MODEL differs from MEM_MODEL=$MEM_MODEL, which needs PROMPT=shared:" >&2
    echo "!!!   the native replay talks to the memory model too, and this job serves one model." >&2
    exit 1
  fi
  SERVE_MODEL=$GEN_MODEL
fi
export GEN_MODEL=${GEN_MODEL:-$MEM_MODEL}

if (( MEM_LLM )); then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "(dry run) would serve $SERVE_MODEL at max-model-len $MAX_MODEL_LEN, util $GPU_UTIL"
  elif ! MEM_MODEL="$SERVE_MODEL" start_server "$MAX_MODEL_LEN"; then
    echo "!!! ${SYS_LIST[*]} needs the memory model; nothing here can run without it" >&2
    exit 1
  fi
fi

echo "==================================================================="
echo "Native retriever grid      job $JOB on $(hostname)"
echo "Stage      : $STAGE"
echo "Memory dir : $MEM_ROOT"
echo "Limit      : ${LIMIT:-none (full dataset)}"
echo "Variant    : $NATIVE_VARIANT"
echo "Config     : ${CONFIG:-<system default>}"
echo "Systems    : ${SYS_LIST[*]}"
echo "Memory LLM : $( ((MEM_LLM)) && echo "yes, :$VLLM_PORT at $GPU_UTIL" || echo "no (no system here prompts it at query time)")"
[[ "$STAGE" == "answer" ]] && echo "Generator  : $GEN_MODEL (served: $SERVE_MODEL)"
echo "Cells      : ${GRID_RETRIEVERS:-per-spec native_retrievers}$([[ "$STAGE" == "answer" ]] && echo " x ${#ANSWER_CUTS[@]} answer cut(s)")"
echo "Trace      : $([[ -n "$TRACE_SAMPLES" ]] && echo "first $TRACE_SAMPLES sample(s) per shard" || echo "every sample")"
[[ -n "$SKIP_QRELS" ]] && echo "Qrels      : reusing existing (SKIP_QRELS=$SKIP_QRELS) — not rebuilt"
[[ "$STAGE" == "answer" ]] && echo "Answer cut : ${ANSWER_CELLS:-none (uncut, 'all')}"
[[ "$STAGE" == "answer" ]] && echo "Prompt     : $PROMPT$([[ "$PROMPT" == "shared" ]] && echo " (unified)" || echo " (control, --native-prompt suffix)")"
[[ "$STAGE" == "answer" ]] && echo "Embedders  : none (a replay ranks nothing)"
echo "==================================================================="

# ---- The three groups ------------------------------------------------------
RESULTS=(); fail=0; first=1
# Systems OUTERMOST, so one server load covers the whole grid. At STAGE=retrieve
# SYS_LIST holds exactly one system (enforced at the top), and this loop is a
# no-op wrapper around the behaviour that was here before.
for S in "${SYS_LIST[@]}"; do
 # Assignment, NOT process substitution: `< <(...)` runs the producer in a
 # subshell, so a cells_for failure exits that subshell and the job carries on
 # with whatever partial list it got. `$( )` propagates the status here.
 CELLS_S=$(cell_list_for "$S") || exit 1
 CELL_LIST=()
 while IFS= read -r _cell; do CELL_LIST+=("$_cell"); done <<< "$CELLS_S"
 # Per system, not once: hypermem's swap REQUIRES retrieval_type: keyword, and
 # in a multi-system job the other five must NOT be handed its config.
 CONFIG_S=$CONFIG
 [[ "$S" == hypermem && -z "$CONFIG_S" ]] && CONFIG_S=evalgrid/configs/hypermem/grid.yaml
 # qrels are deterministic and rebuilt with --overwrite, so they are built on
 # the first cell of each SYSTEM — not of the whole job, which would leave every
 # system after the first without a gold set to score against.
 first=1
 echo
 echo "==================================================================="
 echo "### $S — ${#CELL_LIST[@]} cell(s) x ${#ANSWER_CUTS[@]} cut(s)"
 for GROUP in "" "8001" "8002"; do
  # Partition the cell list by which port, if any, it addresses.
  GROUP_CELLS=(); EMB_MODEL_G=""
  for R in "${CELL_LIST[@]}"; do
    if [[ "$R" == *"@http://localhost:${GROUP:-__none__}/v1"* ]]; then
      GROUP_CELLS+=("$R"); _m=${R#*:}; EMB_MODEL_G=${_m%%@*}
    elif [[ -z "$GROUP" && "$R" != *"@"* ]]; then
      GROUP_CELLS+=("$R")
    fi
  done
  (( ${#GROUP_CELLS[@]} == 0 )) && continue

  echo
  echo ">>> group ${GROUP:-local}: ${#GROUP_CELLS[@]} cell(s)${EMB_MODEL_G:+  embedder $EMB_MODEL_G on :$GROUP}"
  if ! serve_group_embedder "$EMB_MODEL_G" "$GROUP"; then
    for R in "${GROUP_CELLS[@]}"; do RESULTS+=("NO-EMBEDDER|$R"); done
    fail=1; continue
  fi

  for R in "${GROUP_CELLS[@]}"; do
   for CUT in "${ANSWER_CUTS[@]}"; do
    # Exactly ONE axis per pass: answers_dir() names the cell after whichever is
    # set, and a budget supersedes k there, so a stale top-k carried into a
    # budget pass would apply both cuts under the budget's name.
    # The variant names a SWAP setting, so the unswapped cell must not carry it
    # — under GRID_RETRIEVERS=native+grid the two kinds of cell run side by side
    # and only the swapped ones belong under `--grid`.
    VARIANT_R=$NATIVE_VARIANT
    [[ -z "$R" ]] && VARIANT_R=""
    AK=""; CB=""
    case "$CUT" in
      top*)    AK=${CUT#top}    ;;
      budget*) CB=${CUT#budget} ;;
      [0-9]*)  CB=$CUT          ;;
    esac
    echo "-------------------------------------------------------------------"
    echo ">>> $S  <-  ${R:-<no swap>}${CUT:+  [$CUT]}"
    start=$SECONDS
    # query.sh owns qrels, the retrieve stage, output-dir derivation and
    # eval_retrieval. Called as a child rather than reimplemented so a cell of
    # this grid is the same code path as a single hand-submitted cell.
    #
    # SKIP_SERVER/EMB_SERVE=0: the servers are already up and are OURS. The
    # child's own EXIT trap cannot touch them — vllm.sh resets VLLM_PID/EMB_PID
    # to "" at source time, so a child that started nothing kills nothing.
    #
    # qrels is deterministic and rebuilt with --overwrite, so eight rebuilds of
    # one gold set is eight times nothing useful. Built on the first cell only.
    # PROMPT names an ANSWER path, so it is forwarded at STAGE=answer only.
    # Forwarding it at retrieve makes the child read it as an EXPLICIT request
    # for the shared answer path, which it refuses at a non-answer stage — the
    # default is inert here and carries through to the answer job by itself.
    # -u, not merely "not passed": under --export=ALL a PROMPT set on the submit
    # line is already in this job's environment and would reach the child anyway.
    PROMPT_ENV=(-u PROMPT)
    [[ "$STAGE" == "answer" ]] && PROMPT_ENV=(PROMPT="$PROMPT")
    # FIRST, before any NAME=VALUE: env stops parsing options at the first
    # operand, so `env SYSTEMS=x -u PROMPT ...` reads -u as the PROGRAM to run
    # and exits 127 without running anything (job 27466089, both cells, 0s).
    # An assignment is a plain operand, so the answer branch is happy here too.
    env ${PROMPT_ENV[@]+"${PROMPT_ENV[@]}"} \
        SYSTEMS="$S" STAGE="$STAGE" MODE=native \
        CONCURRENCY="$CONCURRENCY" \
        MEM_ROOT="$MEM_ROOT" ${LIMIT:+LIMIT="$LIMIT"} \
        NATIVE_RETRIEVER="$R" NATIVE_VARIANT="$VARIANT_R" \
        ${CONFIG_S:+CONFIG="$CONFIG_S"} \
        SKIP_SERVER=1 EMB_SERVE=0 SKIP_QRELS="${SKIP_QRELS:-$(( first ? 0 : 1 ))}" \
        ANSWER_TOP_K="$AK" CONTEXT_BUDGET="$CB" \
        MEM_BASE_URL="$MEM_BASE_URL" VLLM_PORT="$VLLM_PORT" \
        REPO_DIR="$REPO_DIR" JOB="$JOB" DRY_RUN="$DRY_RUN" \
        TRACE_SAMPLES="$TRACE_SAMPLES" \
        bash "$REPO_DIR/scripts/query.sh"
    rc=$?
    first=0
    elapsed=$((SECONDS - start))
    if (( rc == 0 )); then RESULTS+=("OK|$S ${R:-<no swap>}${CUT:+ [$CUT]}|${elapsed}s")
    else RESULTS+=("FAIL($rc)|$S ${R:-<no swap>}${CUT:+ [$CUT]}|${elapsed}s"); fail=1; fi
   done
  done
 done
done

echo
echo "==================================================================="
echo "Grid summary  ${SYS_LIST[*]}  (job $JOB)"
printf '  %s\n' "${RESULTS[@]+"${RESULTS[@]}"}"
echo "==================================================================="
exit $fail
