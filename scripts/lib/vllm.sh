#!/bin/bash
# scripts/lib/vllm.sh — shared stage plumbing. SOURCE this, do not run it.
#
# Everything here is identical whether a job is building memory or answering
# from it: which Python env, how to start vLLM at the
# right context for a system, and how to stop it cleanly. It lives once because
# the alternative was two copies of ~200 lines that would drift the first time
# a server-startup bug got fixed in only one of them.
#
# A caller sources this, then uses:
#   ctx_for <system>        -> the --max-model-len that system needs
#   gpu_util_for <ctx>      -> the --gpu-memory-utilization for that window
#   start_server <ctx>      -> launch vLLM, wait for /health; non-zero on failure
#   stop_server             -> kill it and let the GPU settle
#   activate_python         -> optional conda activate + import check
#
# and reads: MEM_MODEL, MEM_BASE_URL, VLLM_PORT, REPO_DIR, DRY_RUN, SKIP_SERVER,
#            VLLM_EXTRA_ARGS.
#
# Callers: scripts/ingest.sh, scripts/query.sh

# ---- Where things live -----------------------------------------------------
REPO_DIR=${REPO_DIR:-$PWD}
# Empty CONDA_ENV (the default) = run the stages in the active interpreter.
CONDA_ENV=${CONDA_ENV:-}

# ---- Ingest knobs ----------------------------------------------------------
MEM_MODEL=${MEM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
# Port first, base URL derived from it. The other order let the two disagree:
# VLLM_PORT=8001 started the server on 8001 while every ingest kept talking to
# the 8000 baked into MEM_BASE_URL's default, and the failure surfaced as a
# connection error 20 minutes into the run rather than at startup.
#
# Set VLLM_PORT when two of these jobs can land on the same node — a node with
# several GPUs will happily schedule both, and the second server then fails to
# bind. An explicit MEM_BASE_URL still wins, for pointing at an endpoint that
# this job did not start (see SKIP_SERVER).
#
VLLM_PORT=${VLLM_PORT:-8000}
MEM_BASE_URL=${MEM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}

# ---- Context length, per system --------------------------------------------
# One number for all six systems was wrong in both directions. Four of them send
# prompts that a 32k window cannot hold once their own max_tokens budget is
# subtracted, and the failure is not a warning: the endpoint returns a 400, four
# of the six upstream extractors retry a 400 in an unbounded `while True:`, and
# the stage burns its whole timeout before giving up and dropping that unit's
# memory. HyperMem lost one topic's entire fact set that way (six of a sample's
# 53 sessions ended with no facts) after 331 rejected requests.
#
# So the long-context group gets a window big enough that a rejection is a real
# bug rather than an arithmetic certainty, and the two systems that chunk their
# input small keep the cheap window (a bigger one costs KV cache and buys them
# nothing).
#
#   CTX_LONG    hypermem  max_tokens 20000 reserved out of every request
#               a_mem     evolution prompt carries k neighbour notes, grows with the store
#               mem0      neighbour set + ADD/UPDATE resolution over the whole store
#               memorybank whole-session summarisation in one call
#   CTX_SHORT   lightmem  fixed 16k extraction budget over one topic segment
#               simplemem fixed 16k extraction budget over one 20-dialogue
#                         window (overlap 5), so the prompt does not grow with
#                         the store the way a_mem's does
## The raw-turn baselines are in the long group for a different reason: they have
# no ingest at all, so the window is not about a construction prompt — it is
# about the ANSWER prompt of the full-context row, which is the whole rendered
# haystack. Measured on LongMemEval-S, an all_turns haystack is ~110k tokens
# (o200k) / ~112k (cl100k), so 131072 clears it with roughly 15k of headroom and
# 65536 does not clear it at all.
#
#   CTX_LONG    all_turns   full-context prompt = the whole haystack, ~110-114k
#               pair_turns  same turns, same total (only the doc boundary moves)
#               user_turns  ~16k, so it does not NEED the long window — it is
#                           listed with the others so that one naive-turns job
#                           can answer all three roles against one server
#
# Both CTX_SHORT systems earn the cheap window by CAPPING their own output, and
# that is the whole of the argument — a system in this group whose max_tokens is
# null does not belong in it. simplemem shipped null until 2026-08-19 on the
# theory that "no cap" avoided truncating its JSON array; what it actually did
# was promote this window to the cap, so four calls in outputs/_smoke4 clipped
# at exactly 32768 and one window was lost. Its budget is now
# explicit, which is what makes 32768 the right window rather than an accident:
# 10,046 largest prompt + 16,000 budget = 26k, inside the window with headroom.
#
# WARNING: 131072 has been probed on one H100 at GPU_UTIL=0.95 and FAILED to
# start (logs/probe/27197567.out: "VERDICT: FAIL — vLLM died during startup
# at max-model-len 131072"). If the long group will not come up, lower CTX_LONG
# rather than editing this script — 65536 still clears every prompt observed so
# far by 5x.
CTX_LONG=${CTX_LONG:-131072}
CTX_SHORT=${CTX_SHORT:-32768}
LONG_CTX_SYSTEMS=${LONG_CTX_SYSTEMS:-"hypermem memorybank mem0 a_mem all_turns pair_turns user_turns"}
# Non-empty pins ONE window for every system, as this script used to. Kept so an
# existing "MAX_MODEL_LEN=... bash ..." line still means what it said.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}

# The window a system's ingest must run against. A system not named in
# LONG_CTX_SYSTEMS gets CTX_SHORT — the safe default is the cheap one, so a
# system added to SYSTEMS without being classified here cannot silently consume
# a 128k allocation.
ctx_for() {
  local sys=$1 s
  if [[ -n "$MAX_MODEL_LEN" ]]; then echo "$MAX_MODEL_LEN"; return; fi
  for s in $LONG_CTX_SYSTEMS; do
    [[ "$s" == "$sys" ]] && { echo "$CTX_LONG"; return; }
  done
  echo "$CTX_SHORT"
}

# ---- GPU memory fraction, per context group --------------------------------
# Paired with the window, because the two are one decision. KV cache is what a
# long window spends, and it comes out of whatever the weights leave behind:
# Qwen3-30B-A3B is ~61 GB in bf16, so on an 80 GB H100 the difference between
# 0.8 and 0.95 (~64 GB vs ~76 GB) is roughly 3 GB of cache versus 15 GB. The
# short group does not need that and should not take the OOM risk for it; the
# long group cannot start without it.
#
# These are also the exact pairings that have been probed, which is the reason
# to keep them together rather than let a run mix a 128k window with 0.8:
#   32768  @ 0.8   started fine and ingested every system (jobs 27195053-27253282)
#   131072 @ 0.95  FAILED during model load (logs/probe/27197567.out)
# So the long pairing is the one still unproven — see the WARNING above.
GPU_UTIL_LONG=${GPU_UTIL_LONG:-0.95}
GPU_UTIL_SHORT=${GPU_UTIL_SHORT:-0.8}
# Non-empty pins ONE fraction for every group, mirroring MAX_MODEL_LEN.
GPU_UTIL=${GPU_UTIL:-}

# The fraction to serve ``$1`` context at. Matched on CTX_SHORT rather than
# CTX_LONG so that setting the two windows equal resolves to the SAFER value:
# too little KV cache reports "this context does not fit", which is diagnosable,
# while too much dies in the middle of loading weights.
gpu_util_for() {
  local ctx=$1
  if [[ -n "$GPU_UTIL" ]]; then echo "$GPU_UTIL"; return; fi
  [[ "$ctx" == "$CTX_SHORT" ]] && { echo "$GPU_UTIL_SHORT"; return; }
  echo "$GPU_UTIL_LONG"
}
# First start on a cold torch.compile cache spends 10-25 silent minutes compiling
# MoE kernels after the weights are resident — 1800 was not enough headroom.
SERVER_TIMEOUT=${SERVER_TIMEOUT:-2400}   # seconds to wait for /health

# Extra flags appended to the memory LLM's `vllm serve`, word-split. Empty keeps
# the probed Qwen3-30B-A3B launch unchanged. Hybrid-attention models (Qwen3.5
# family, e.g. Qwen3.8-27B) need "--max-num-seqs 256": vLLM 0.22 defaults to
# 1024 sequences, each needs one Mamba state block, and 0.95 of an H100 leaves
# only ~397 of them after 51 GiB of weights (logs/vllm/28609667_ctx131072.log).
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

LIMIT=${LIMIT-3}                 # note the single dash: LIMIT="" means "no limit"
OVERWRITE=${OVERWRITE:-0}

# Shard this job's slice of the dataset. NUM_SHARDS=N with SHARD_ID=0..N-1 gives
# N jobs disjoint subsets of the same dataset, all writing into the same memory
# dir — the split is index % num_shards, so no coordination is needed beyond the
# filesystem (see ingest.py::ShardedAdapter).
#
# A full ingest of a_mem or hypermem is long, so run N processes (or machines),
# each with the same NUM_SHARDS and its own SHARD_ID in 0..N-1:
#
#   NUM_SHARDS=8 SHARD_ID=0 SYSTEMS=a_mem LIMIT= bash scripts/ingest.sh
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}
# Trajectories are on by default at the level that keeps prompts verbatim, for
# every sample: a run you cannot read back is a run you have to repeat, and every
# bug fixed in this project was found in a `full` trajectory. The cost is disk, not GPU —
# roughly 4 bytes per recorded token, so a full LongMemEval-S ingest of all six
# systems is ~10 GB. TRACE_LEVEL=off restores the NULL_RECORDER path.
TRACE_LEVEL=${TRACE_LEVEL:-full}
# Empty = every sample, which is the CLI's own default (--trace-samples None) and
# so is expressed by NOT passing the flag rather than by a sentinel number. Note
# the single dash: TRACE_SAMPLES="" from the submitting shell means "all", not
# "use the default". Set a number to cap it (TRACE_SAMPLES=1 = first per shard).
TRACE_SAMPLES=${TRACE_SAMPLES-}

# Resolved once here so the two engines cannot disagree about what they recorded:
# TRACE_FLAGS is what goes on the command line, TRACE_SCOPE is how the banner and
# the closing trajectory listing describe it.
#
# --trace-level is passed ALWAYS, including "off", and that is load-bearing: the
# Python default is now "full" (evalgrid/lib/cli.py), so an omitted flag means
# "full", not "off". This block used to express off by emitting no flags, which
# under the new default would turn TRACE_LEVEL=off into a ~10 GB full trace —
# silently reversing the one knob a caller sets to avoid exactly that.
TRACE_FLAGS=(--trace-level "$TRACE_LEVEL")
TRACE_SCOPE="off"
if [[ "$TRACE_LEVEL" != "off" ]]; then
  if [[ -n "$TRACE_SAMPLES" ]]; then
    TRACE_FLAGS+=(--trace-samples "$TRACE_SAMPLES")
    TRACE_SCOPE="first $TRACE_SAMPLES sample(s) per shard"
  else
    TRACE_SCOPE="every sample"
  fi
fi
MEM_ROOT=${MEM_ROOT:-}           # empty = the default keyed outputs/_memory paths
SKIP_SERVER=${SKIP_SERVER:-0}
# 1 = resolve every knob, print the exact ingest command per system, run nothing.
# The point is that "did my env vars arrive?" is answerable in two seconds
# instead of after a 20-minute vLLM start. No server, no GPU, no writes.
DRY_RUN=${DRY_RUN:-0}

# Every system with a construction stage. The raw-turn baselines (all_turns /
# user_turns / pair_turns) are deliberately absent: they have no plugin and no
# ingest — that is the point of those rows.
SYSTEMS=${SYSTEMS:-"lightmem a_mem mem0 memorybank simplemem hypermem"}

# ---- Which experiment spec, and therefore which benchmark ------------------
# Experiment specs are per-system (experiments/<system>/<SPEC_NAME>), so the knob
# is the spec's BASENAME rather than a path. Both engines named
# "longmemeval.yaml" literally before this existed, so an unset SPEC_NAME leaves
# every LongMemEval run byte-identical.
#
#   SPEC_NAME=beam_100k.yaml SYSTEMS=lightmem bash scripts/query.sh
#
# Defined here, beside SYSTEMS, because the ingest and query engines must agree
# on it: a memory built from one spec and queried under another is two different
# benchmarks sharing an output tree.
SPEC_NAME=${SPEC_NAME:-longmemeval.yaml}

spec_of() {   # spec_of <system> -> experiments/<system>/<SPEC_NAME>
  echo "experiments/$1/$SPEC_NAME"
}

spec_dataset() {   # the `dataset:` of the first spec in $SYSTEMS that exists
  local s
  for s in $SYSTEMS; do
    [[ -f "$(spec_of "$s")" ]] || continue
    python - "$(spec_of "$s")" <<'PYEOF'
import sys, yaml
print((yaml.safe_load(open(sys.argv[1])) or {}).get("dataset", ""))
PYEOF
    return
  done
}


# ---- Python environment ----------------------------------------------------
# A dry run answers "did my env vars arrive?", which is true or false regardless
# of whether this machine can run the job — so an unusable env is a warning
# there and fatal everywhere else. That is what lets DRY_RUN work on a login
# node, or a laptop, without the conda env the real run needs.
die_or_warn() { if [[ "$DRY_RUN" == "1" ]]; then echo "(dry run) $1" >&2; else echo "!!! $1" >&2; exit 1; fi; }

# A function rather than top-level code, because sourcing this file must not
# have side effects: a caller that only wants ctx_for() to print a plan should
# not have its shell's conda env swapped underneath it.
activate_python() {
  if [[ -n "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        || die_or_warn "no conda on PATH"
    conda activate "$CONDA_ENV" || die_or_warn "cannot activate conda env $CONDA_ENV"
  fi
  echo ">>> python: $(command -v python 2>/dev/null || echo none)"
  _check_python_deps "${CONDA_ENV:-the active environment}"
}

_check_python_deps() {
  python - "$1" <<'PY' || die_or_warn "$1 is missing pipeline dependencies"
import importlib.util as u, sys
env = sys.argv[1]
def missing(mods):
    out = []
    for m in mods:
        try:
            if u.find_spec(m) is None:
                out.append(m)
        except (ImportError, ValueError):
            out.append(m)
    return out
# Imported unconditionally by every stage — absent means nothing can run.
hard = missing(("evalgrid", "yaml", "tqdm"))
# Needed by most systems/retrievers but not all, so a warning: a lightmem
# ingest legitimately runs without faiss-shaped extras.
soft = missing(("numpy", "torch", "dotenv"))
if hard:
    print(f"!!! {env} is missing: {', '.join(hard)}", file=sys.stderr)
    print(f"!!!   fix: conda activate {env} && pip install --no-deps -r requirements.txt", file=sys.stderr)
    sys.exit(1)
if soft:
    print(f"    warning: {env} lacks {', '.join(soft)} — some systems will fail later")
print(f"    python: {sys.executable}")
PY
}

# ---- Memory LLM ------------------------------------------------------------
VLLM_PID=""
# Run a `vllm serve ...` command line with the vllm on PATH.
_vllm_exec() { "$@"; }

_vllm_available() {
  command -v vllm >/dev/null 2>&1 || {
    echo "!!! no vllm on PATH (install vLLM 0.22.1), or start the servers yourself" >&2
    echo "!!!   and pass SKIP_SERVER=1" >&2
    return 1; }
}

stop_server() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ">>> stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null
    # The next server must be able to claim the port and the whole GPU. Without
    # the settle the successor's memory profiling races the predecessor's
    # teardown and it fails to allocate its KV cache — which looks exactly like
    # "this context length does not fit", the one diagnosis that must stay
    # trustworthy here.
    sleep 20
  fi
  VLLM_PID=""
}
trap 'stop_server; stop_embed_server; stop_rerank_server' EXIT INT TERM

# Start the memory LLM at ``$1`` context. Returns non-zero instead of exiting:
# one context group failing to come up must still leave the other group's
# results on the table, same reasoning as "no set -e" above.
start_server() {
  local ctx=$1 waited=0
  local util; util=$(gpu_util_for "$ctx")
  if [[ "$SKIP_SERVER" == "1" ]]; then
    echo ">>> SKIP_SERVER=1 — not starting $MEM_MODEL; expecting it on :$VLLM_PORT (max-model-len $ctx)"
    return 0
  fi
  _vllm_available || return 1
  VLLM_LOG="logs/vllm/${JOB}_ctx${ctx}.log"
  echo ">>> starting vLLM: $MEM_MODEL on :$VLLM_PORT  max-model-len=$ctx" \
       " gpu-memory-utilization=$util${VLLM_EXTRA_ARGS:+  extra: $VLLM_EXTRA_ARGS}  (log: $VLLM_LOG)"
  _vllm_exec vllm serve "$MEM_MODEL" \
          --host 0.0.0.0 --port "$VLLM_PORT" \
          --max-model-len "$ctx" \
          --gpu-memory-utilization "$util" \
          $VLLM_EXTRA_ARGS \
          > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  echo -n ">>> waiting for /health "
  until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo
      echo "!!! vLLM died during startup at max-model-len $ctx, gpu-memory-utilization" \
           "$util — last 40 lines of $VLLM_LOG:" >&2
      tail -40 "$VLLM_LOG" >&2
      # The 40-line tail is the API-server unwind — the same stack whatever
      # killed it. vLLM's OWN diagnosis is ~50 lines further up, in the
      # EngineCore block, and it is exact ("estimated maximum model length is
      # 19040"). Surfacing it here is the difference between a reader who knows
      # which lever to pull and one who reads a torch traceback and guesses.
      if grep -q "_check_enough_kv_cache_memory\|is needed, which is larger than the available KV cache" \
              "$VLLM_LOG" 2>/dev/null; then
        echo "!!!" >&2
        echo "!!! ROOT CAUSE: KV cache, not weights — the weights loaded fine and there" >&2
        echo "!!!   was not enough left to admit one max-length request:" >&2
        grep -h "is needed, which is larger than the available KV cache" "$VLLM_LOG" \
          | tail -1 | sed 's/^.*ValueError: /!!!   /' >&2
        echo "!!!   Raise the utilization fraction (GPU_UTIL / GPU_UTIL_LONG / EMB_UTIL" >&2
        echo "!!!   share one card and must sum to <= ~0.96), or lower max-model-len to" >&2
        echo "!!!   under the estimate above. Careful: a system that sends a large" >&2
        echo "!!!   max_tokens needs prompt+budget to FIT — simplemem asks for 16000 on" >&2
        echo "!!!   every call, so no window under ~20k works for it however short its" >&2
        echo "!!!   prompts are." >&2
      elif grep -q "exceeds available Mamba cache blocks" "$VLLM_LOG" 2>/dev/null; then
        echo "!!!" >&2
        echo "!!! ROOT CAUSE: hybrid-attention model, too many sequences for its state cache:" >&2
        grep -h "exceeds available Mamba cache blocks" "$VLLM_LOG" \
          | tail -1 | sed 's/^.*ValueError: /!!!   /' >&2
        echo "!!!   Resubmit with VLLM_EXTRA_ARGS=\"--max-num-seqs 256\" (or the cap it names)." >&2
      else
        # Anything else: vLLM's own last error line, not a guess. Weights that
        # loaded show up as "Model loading took", so say which side it died on.
        echo "!!!" >&2
        echo "!!! ROOT CAUSE (vLLM's last error line):" >&2
        grep -hE "^\(EngineCore[^)]*\) [A-Za-z.]*(Error|Exception): " "$VLLM_LOG" \
          | tail -1 | sed 's/^([^)]*) /!!!   /' >&2
        if grep -q "Model loading took" "$VLLM_LOG" 2>/dev/null; then
          grep -h "Model loading took" "$VLLM_LOG" | tail -1 \
            | sed 's/^.*INFO.*\] /!!!   weights DID load: /' >&2
        else
          echo "!!!   weights never finished loading — check the model is staged" \
               "and its weights are complete." >&2
        fi
      fi
      VLLM_PID=""
      return 1
    fi
    if (( waited >= SERVER_TIMEOUT )); then
      echo; echo "!!! vLLM not ready after ${SERVER_TIMEOUT}s — see $VLLM_LOG" >&2
      stop_server
      return 1
    fi
    sleep 10; waited=$((waited + 10)); echo -n "."
  done
  echo " ready after ${waited}s"
  return 0
}

# ---- Embedding model -------------------------------------------------------
# A SECOND vLLM, for the served embedders in the retriever grid
# (Qwen3-Embedding-0.6B / 4B). Separate from the memory LLM above because the
# two coexist: a_mem and simplemem call the memory model at query time AND rank
# with the embedder, so both servers must be up at once on the same GPU.
#
# WHY IT IS STARTED HERE. The retriever spec addresses the embedder as
# http://localhost:<port>/v1, so it must run on the same machine as the stage.
#
# GPU SHARE. The default 0.1 leaves the memory LLM its normal allocation. It is
# a FRACTION OF THE WHOLE CARD, not of what is free, so raising it while a
# 131072-context memory model is up will make one of the two fail to allocate
# its KV cache. 0.1 of an H100 is ~8GB — ample for 0.6B, tight but workable for
# 4B; raise EMB_UTIL and lower GPU_UTIL_* together if 4B will not start.
EMB_UTIL=${EMB_UTIL:-0.1}

# Window for the embedding server. EMPTY = let vLLM use the model's own default,
# which is what every caller that has the card to itself should do.
#
# It matters because the comment that used to sit on the serve command below —
# "an embedding model's context is its own and there is no KV cache to size" —
# is FALSE, and expensively so. vLLM allocates a KV cache for an embedding model
# exactly as for a generator, sized to admit one request at the model's full
# native window. Measured (logs/vllm/27418633_embed_8002.log, EMB_UTIL=0.4):
#
#   Qwen3-Embedding-4B    native window 40960   KV 5.63 GiB   floor 14.47 GiB
#   Qwen3-Embedding-0.6B  native window 32768   KV 3.50 GiB   floor  5.42 GiB
#
# So the 4B needs EMB_UTIL >= 0.183 to start AT ALL, whatever it is asked to
# embed — which is why it died at 0.16 while the 0.6B beside it came up fine.
# Nearly all of that is KV for a context length nothing here uses: the longest
# unit in any system's corpus measures ~4,900 tokens (a_mem and memorybank
# session dumps; lightmem/mem0/simplemem are under 350). Capping the window
# spends the difference on whatever else has to share the card.
EMB_MAX_LEN=${EMB_MAX_LEN:-}
# Extra `vllm serve` flags for the embedder only, word-split. Empty = none. Used
# for BEAM pair_turns, whose 100K_5 holds a 26k-token unit: at vLLM 0.22.1's
# default --max-num-batched-tokens 8192 that unit is prefilled in chunks, and (it
# appears) the pooling path then indexes rotary positions past 32768 and dies.
#   EMB_EXTRA_ARGS="--max-num-batched-tokens 32768"
EMB_EXTRA_ARGS=${EMB_EXTRA_ARGS:-}
EMB_PID=""

# --- cross-encoder reranker (HyperMem, published setting) -------------------
# A SECOND side server, needed only by HyperMem with `reranker.enabled: true`
# (evalgrid/configs/hypermem/naive_query.yaml). Off everywhere else, so no other row
# in this project has a second stage.
#
# Qwen3-Reranker-4B is a GENERATIVE model scored through /v1/completions with a
# one-token yes/no template (evalgrid/systems/hypermem/reranker.py), not an
# embedding model. vLLM infers that from the model config, so the serve command
# is the same one the embedder uses and no --task is passed either way.
#
# GPU SHARE. 4B weights plus a KV cache sized to the model's native window, so
# the same arithmetic as EMB_UTIL applies: measured floors put a 4B at >= 0.183
# of an H100 before it will start at all. 0.2 is that with a little room.
#
# THIS ONLY FITS BECAUSE THE MEMORY LLM IS NOT UP. HyperMem is absent from
# NEEDS_MEM_LLM (its funnel prompts no model), so a STAGE=retrieve job for it
# sets SKIP_SERVER=1 and the card holds only these two: embedder 0.2 + reranker
# 0.2. Ask for a reranker beside a 131072-context memory LLM and one of the
# three will fail to allocate.
RERANK_UTIL=${RERANK_UTIL:-0.2}
# Window for the reranker. Empty = the model's own, as with EMB_MAX_LEN, and for
# the same reason: nearly all of a 4B's floor is KV for a context nothing here
# uses. A rerank prompt is one query plus one candidate — the longest unit in
# any corpus measures ~4,900 tokens — so capping this is the cheapest way to
# free room if the pair will not co-start.
RERANK_MAX_LEN=${RERANK_MAX_LEN:-}
RERANK_PID=""
stop_embed_server() {
  # ${EMB_PID:-} rather than $EMB_PID, for the reason stop_rerank_server gives:
  # the EXIT trap is installed well above the line that initialises this, and
  # query.sh runs under `set -u`, so a failure in between would fire the trap
  # on an unbound variable and report that instead of what actually went wrong.
  if [[ -n "${EMB_PID:-}" ]] && kill -0 "$EMB_PID" 2>/dev/null; then
    echo ">>> stopping embedding vLLM (pid $EMB_PID)"
    kill "$EMB_PID" 2>/dev/null
    wait "$EMB_PID" 2>/dev/null
  fi
  EMB_PID=""
}

# _start_side_vllm <model> <port> <util> <tag> [max-model-len]
#
# Starts a vLLM BESIDE the main one and waits for /health. Sets SIDE_PID and
# SIDE_LOG; the caller copies both into its own pair of variables, which is the
# whole reason this is factored out — two side servers run at once on a native
# HyperMem run (the embedder for the dense leg, the cross-encoder for rerank),
# and a single shared EMB_PID would leave the first of them unkillable by the
# EXIT trap. SIDE_PID is "" on failure, so the caller's variable clears too.
#
# Same image and binds as start_server; see there for why both binds are
# required. VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the engine in-process — the
# V1 multiprocess path spawns a worker that does not inherit this container's
# environment cleanly when a second server is already running on the card.
# --max-model-len only when the caller passes one; unset keeps the model's own
# window. No --task: vLLM infers `embed` or `generate` from the model's config,
# which is what lets one function serve both an embedder and a reranker.
_start_side_vllm() {
  local model=$1 port=$2 util=$3 tag=$4 max_len=${5:-} extra=${6:-} waited=0
  SIDE_PID=""
  SIDE_LOG=""
  if [[ "$SKIP_SERVER" == "1" ]]; then
    echo ">>> SKIP_SERVER=1 — not starting $tag $model; expecting it on :$port"
    return 0
  fi
  _vllm_available || return 1
  SIDE_LOG="logs/vllm/${JOB}_${tag}_${port}.log"
  echo ">>> starting $tag vLLM: $model on :$port  gpu-memory-utilization=$util" \
       "${max_len:+  max-model-len=$max_len}  (log: $SIDE_LOG)"
  _vllm_exec env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      vllm serve "$model" \
          --host 0.0.0.0 --port "$port" \
          ${max_len:+--max-model-len "$max_len"} \
          --gpu-memory-utilization "$util" \
          $extra \
          > "$SIDE_LOG" 2>&1 &
  SIDE_PID=$!

  echo -n ">>> waiting for $tag /health "
  until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
    if ! kill -0 "$SIDE_PID" 2>/dev/null; then
      echo
      echo "!!! $tag vLLM died during startup — last 40 lines of $SIDE_LOG:" >&2
      tail -40 "$SIDE_LOG" >&2
      echo "!!! if it ran out of memory, something else already holds most of the" >&2
      echo "!!!   card: lower GPU_UTIL_* or raise EMB_UTIL/RERANK_UTIL, not both upward." >&2
      SIDE_PID=""
      return 1
    fi
    if (( waited >= SERVER_TIMEOUT )); then
      echo; echo "!!! $tag not ready after ${SERVER_TIMEOUT}s — see $SIDE_LOG" >&2
      kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null
      SIDE_PID=""
      return 1
    fi
    sleep 10; waited=$((waited + 10)); echo -n "."
  done
  echo " ready after ${waited}s"
  return 0
}

# start_embed_server <model> <port> [gpu-util]
# Contract unchanged: sets EMB_PID and EMB_LOG, returns non-zero on failure with
# EMB_PID cleared. EMB_LOG is assigned on failure too, as it was before, so a
# caller can still point at the log after a bad start.
start_embed_server() {
  _start_side_vllm "$1" "$2" "${3:-$EMB_UTIL}" embed "$EMB_MAX_LEN" "$EMB_EXTRA_ARGS"
  local rc=$?
  EMB_LOG=$SIDE_LOG
  EMB_PID=$SIDE_PID
  return $rc
}

stop_rerank_server() {
  # ${RERANK_PID:-} rather than $RERANK_PID: the EXIT trap is installed well
  # above the line that initialises this, and query.sh runs under `set -u`,
  # so an early failure would otherwise fire the trap on an unbound variable and
  # report that instead of whatever actually went wrong.
  if [[ -n "${RERANK_PID:-}" ]] && kill -0 "$RERANK_PID" 2>/dev/null; then
    echo ">>> stopping reranker vLLM (pid $RERANK_PID)"
    kill "$RERANK_PID" 2>/dev/null
    wait "$RERANK_PID" 2>/dev/null
  fi
  RERANK_PID=""
}

# start_rerank_server <model> <port> [gpu-util]
# HyperMem's cross-encoder. Mirrors start_embed_server exactly; see RERANK_UTIL
# above for why it only fits when the memory LLM is not up.
start_rerank_server() {
  _start_side_vllm "$1" "$2" "${3:-$RERANK_UTIL}" rerank "$RERANK_MAX_LEN"
  local rc=$?
  RERANK_LOG=$SIDE_LOG
  RERANK_PID=$SIDE_PID
  return $rc
}

if [[ "$SKIP_SERVER" == "1" ]]; then
  echo ">>> SKIP_SERVER=1 — using existing endpoint $MEM_BASE_URL"
  echo "!!! warning: one endpoint serves every system, so the per-system context" >&2
  echo "!!!   split below is NOT applied — make sure it was started at $CTX_LONG." >&2
  curl -sf "${MEM_BASE_URL%/v1}/health" > /dev/null 2>&1 \
      || echo "!!! warning: no /health at $MEM_BASE_URL — continuing anyway"
fi

