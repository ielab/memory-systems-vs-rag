#!/bin/bash
# Ingest ONE system with an OpenAI API model as the memory LLM (no vLLM).
#
# scripts/ingest.sh starts a local vLLM server and takes --mem-model
# from each spec, so neither half fits an API model. This calls the pipeline
# directly: no GPU needed, OPENAI_API_KEY from .env, --mem-model on the command line.
# The key reaches each system through SystemPlugin.apply_mem_model ->
# route_mem_llm_to_openai (evalgrid/systems/base.py).
#
# Run from the repo root. Smoke test (1 conversation, throwaway dir):
#
#   SYSTEM=a_mem LIMIT=1 MEM_ROOT=outputs/_smoke_openai_ingest bash scripts/ingest_openai.sh
#
# lightmem runs LLMLingua-2 on CUDA, so it needs a GPU; the other systems do not.
#
# Full run: leave MEM_ROOT and LIMIT empty (optionally NUM_SHARDS=N SHARD_ID=i
# across processes) — the store then lands at the keyed
# outputs/_memory/<system>/<dataset>/gpt-4o-mini/.
#
# Knobs: SYSTEM (required), SPEC_NAME=beam_100k.yaml, MEM_MODEL=gpt-4o-mini,
#        LIMIT="", NUM_SHARDS=1, MEM_ROOT="", OVERWRITE=0, TRACE_LEVEL=full
#

set -uo pipefail

: "${SYSTEM:?set SYSTEM=<a_mem|mem0|hypermem|memorybank|simplemem|lightmem>}"
SPEC_NAME=${SPEC_NAME:-beam_100k.yaml}
export MEM_MODEL=${MEM_MODEL:-gpt-4o-mini}
MEM_BASE_URL=https://api.openai.com/v1
LIMIT=${LIMIT-}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}
MEM_ROOT=${MEM_ROOT-}
OVERWRITE=${OVERWRITE:-0}
export TRACE_LEVEL=${TRACE_LEVEL:-full}

REPO_DIR=$PWD
source "$REPO_DIR/scripts/lib/vllm.sh"
cd "$REPO_DIR" || { echo "!!! repo dir not found: $REPO_DIR" >&2; exit 1; }
mkdir -p logs/ingest
activate_python

if [[ -z "${OPENAI_API_KEY:-}" && -r .env ]]; then
  set -a; . ./.env; set +a
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "!!! OPENAI_API_KEY unset and not in .env" >&2; exit 1; }

# Fail in seconds, not on the first ingest call: no route or a bad key.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
       -H "Authorization: Bearer $OPENAI_API_KEY" "$MEM_BASE_URL/models")
[[ "$code" == "200" ]] || { echo "!!! $MEM_BASE_URL/models returned HTTP $code" >&2; exit 1; }

spec=experiments/$SYSTEM/$SPEC_NAME
[[ -f "$spec" ]] || { echo "!!! no spec $spec" >&2; exit 1; }

FLAGS=(--exp "$spec" --mem-model "$MEM_MODEL" --mem-base-url "$MEM_BASE_URL"
       "${TRACE_FLAGS[@]}")
[[ -n "$LIMIT" ]] && FLAGS+=(--limit "$LIMIT")
[[ -n "$MEM_ROOT" ]] && FLAGS+=(--mem-dir "$MEM_ROOT/$SYSTEM")
[[ "$OVERWRITE" == "1" ]] && FLAGS+=(--overwrite)
(( NUM_SHARDS > 1 )) && FLAGS+=(--num-shards "$NUM_SHARDS" --shard-id "$SHARD_ID")

echo "==================================================================="
echo "Ingest via OpenAI    on $(hostname)"
echo "System     : $SYSTEM   spec $spec"
echo "Mem model  : $MEM_MODEL @ $MEM_BASE_URL"
echo "Limit      : ${LIMIT:-all}   shard $SHARD_ID/$NUM_SHARDS   mem-root ${MEM_ROOT:-<keyed>}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "==================================================================="

start=$SECONDS
python -m evalgrid.pipeline.ingest "${FLAGS[@]}"
rc=$?
echo ">>> [ingest] $SYSTEM exit $rc after $((SECONDS - start))s"
exit $rc
