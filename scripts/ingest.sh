#!/bin/bash
# Stage 1 — build memory for one or more systems on one GPU.
#
# Starts the memory LLM with vLLM once per context group, then runs
# evalgrid.pipeline.ingest for each system's spec against that endpoint, and
# prints a pass/fail table with each store's construction cost.
#
# The default is a 3-sample smoke run (LIMIT=3); LIMIT= (empty) is the full
# dataset. LIMIT is read with a single dash below so an empty value is honoured.
#
# Run from the repo root:
#
#   bash scripts/ingest.sh                                  # 3 samples, all six systems
#   SYSTEMS=a_mem LIMIT= bash scripts/ingest.sh             # full dataset, one system
#   SYSTEMS=a_mem LIMIT= NUM_SHARDS=4 SHARD_ID=0 bash scripts/ingest.sh   # one of 4 shards
#
# Logs are grouped by artifact:
#   logs/ingest/   the per-system tee
#   logs/vllm/     one server log per context group, <job>_ctx<N>.log
#
# Knobs (environment variables):
#   LIMIT=3               samples per system; empty ("") = the full 500
#   SYSTEMS="..."         space-separated subset of the systems below
#   SPEC_NAME=longmemeval.yaml
#                         WHICH EXPERIMENT SPEC, by basename — the specs are
#                         per-system, so this picks experiments/<system>/<name>,
#                         and with it the dataset the memory is built from:
#                           SPEC_NAME=beam_100k.yaml bash scripts/ingest.sh
#                         Query the result with the SAME SPEC_NAME, or you are
#                         querying one benchmark's memory as another's.
#   OVERWRITE=0           1 = re-ingest samples already saved (default: resume)
#   TRACE_LEVEL=full      off|calls|steps|full — record what each pipeline did.
#                         Default full: untruncated payloads, every sample (see
#                         TRACE_SAMPLES). Pass off for the old no-recorder path.
#   TRACE_SAMPLES=""      how many samples to trace when TRACE_LEVEL != off.
#                         Empty (the default) = ALL of them. A number caps it,
#                         e.g. TRACE_SAMPLES=1 for the first sample of each shard.
#   MEM_ROOT=""           write memory under <MEM_ROOT>/<system>/ instead of the
#                         default outputs/_memory/<system>/<dataset>/<mem-seg>/
#                         key. Use for throwaway runs (a trace test) so they
#                         neither resume over nor overwrite a real ingest:
#                           LIMIT=2 TRACE_LEVEL=full TRACE_SAMPLES=2 \
#                           MEM_ROOT=outputs/_trace_smoke \
#                           bash scripts/ingest.sh
#                         These dirs are NOT the keyed location, so every
#                         downstream stage needs the same --mem-dir to find them.
#   DRY_RUN=0             1 = print the resolved knobs + the exact ingest command
#                         per system and exit (no GPU needed):
#                           LIMIT=2 TRACE_LEVEL=full DRY_RUN=1 \
#                           bash scripts/ingest.sh
#   SKIP_SERVER=0         1 = don't launch vLLM; use MEM_BASE_URL as-is. Note
#                         this defeats the per-system context split below —
#                         start that endpoint at CTX_LONG or systems will 400.
#   CTX_LONG=131072       window for LONG_CTX_SYSTEMS (see "Context length" below)
#   CTX_SHORT=32768       window for every other system
#   LONG_CTX_SYSTEMS      default "hypermem memorybank mem0 a_mem"
#   MAX_MODEL_LEN=""      non-empty pins ONE window for every system (old behaviour)
#   GPU_UTIL_LONG=0.95    --gpu-memory-utilization for the CTX_LONG server
#   GPU_UTIL_SHORT=0.8    --gpu-memory-utilization for the CTX_SHORT server
#   GPU_UTIL=""           non-empty pins ONE fraction for every group
#   MEM_BASE_URL          default http://localhost:8000/v1
#   MEM_MODEL             default Qwen/Qwen3-30B-A3B-Instruct-2507
#   CONDA_ENV=""          conda env to activate; empty = the active interpreter
#
# For a real run, ingest one system at a time and shard it (NUM_SHARDS/SHARD_ID);
# hypermem in particular is a few hundred LLM calls per sample.

# No `set -e`: one system's ingest failing must not skip the rest — finding out
# which systems break is the whole job.
set -uo pipefail

# ---- Shared stage plumbing -------------------------------------------------
# Paths, conda, the per-system context/GPU-fraction table, and start/stop_server
# all live in one place so this engine and the answer engine cannot drift.
REPO_DIR=${REPO_DIR:-$PWD}
source "$REPO_DIR/scripts/lib/vllm.sh"

cd "$REPO_DIR" || { echo "!!! repo dir not found: $REPO_DIR" >&2; exit 1; }
LOG_DIR=logs/ingest
# Every log dir this script writes to.
mkdir -p "$LOG_DIR" logs/vllm
JOB=${JOB:-$$}

echo "==================================================================="
echo "Ingest smoke test          job $JOB on $(hostname)"
echo "Repo       : $REPO_DIR"
echo "Systems    : $SYSTEMS"
echo "Mem model  : $MEM_MODEL"
if [[ -n "$MAX_MODEL_LEN" ]]; then
  echo "Context    : $MAX_MODEL_LEN for every system (MAX_MODEL_LEN pinned)"
else
  echo "Context    : $CTX_LONG for [$LONG_CTX_SYSTEMS], $CTX_SHORT for the rest"
fi
if [[ -n "$GPU_UTIL" ]]; then
  echo "GPU util   : $GPU_UTIL for every group (GPU_UTIL pinned)"
elif [[ -n "$MAX_MODEL_LEN" ]]; then
  # One window means one group, so name the fraction it actually resolves to
  # rather than a split that cannot happen.
  echo "GPU util   : $(gpu_util_for "$MAX_MODEL_LEN") (single group at $MAX_MODEL_LEN)"
else
  echo "GPU util   : $GPU_UTIL_LONG at $CTX_LONG, $GPU_UTIL_SHORT at $CTX_SHORT"
fi
echo "Limit      : ${LIMIT:-none (full dataset)}"
if (( NUM_SHARDS > 1 )); then
  echo "Shard      : $SHARD_ID of $NUM_SHARDS   (samples where index % $NUM_SHARDS == $SHARD_ID)"
fi
echo "Trace      : $TRACE_LEVEL$([[ "$TRACE_LEVEL" != "off" ]] && echo " ($TRACE_SCOPE)")"
if [[ -n "$MEM_ROOT" ]]; then
  echo "Memory dir : $MEM_ROOT/<system>   (override — downstream stages need the same --mem-dir)"
else
  echo "Memory dir : outputs/_memory/<system>/<dataset>/<mem-seg>   (default keyed paths)"
fi
echo "Started    : $(date)"
[[ "$DRY_RUN" == "1" ]] && echo "DRY RUN    : resolving knobs only — no server, no ingest, no writes"
echo "==================================================================="

activate_python

# ---- Build the spec list ---------------------------------------------------
SPECS=()
# spec_of()/SPEC_NAME come from scripts/lib/vllm.sh — one definition shared with
# the query engine, because a memory built from one spec and queried under
# another is two benchmarks sharing an output tree.
for s in $SYSTEMS; do
  spec="$(spec_of "$s")"
  if [[ -f "$spec" ]]; then
    SPECS+=("$spec")
  else
    echo "!!! no spec for system '$s' at $spec — skipping" >&2
  fi
done
[[ ${#SPECS[@]} -eq 0 ]] && { echo "!!! no specs to run" >&2; exit 1; }

# Flags shared by every ingest call. --mem-model comes from each spec; passing
# --mem-base-url here is what points them all at the one server above.
COMMON_FLAGS=(--mem-base-url "$MEM_BASE_URL")
[[ -n "$LIMIT" ]] && COMMON_FLAGS+=(--limit "$LIMIT")
[[ "$OVERWRITE" == "1" ]] && COMMON_FLAGS+=(--overwrite)
# Only passed when actually sharding: --num-shards 1 is the default and adding
# it to every command line would make the DRY_RUN output noisier for no reason.
if (( NUM_SHARDS > 1 )); then
  COMMON_FLAGS+=(--num-shards "$NUM_SHARDS" --shard-id "$SHARD_ID")
fi
# Unconditional: TRACE_FLAGS now always carries an explicit --trace-level (see
# scripts/lib/vllm.sh). Skipping it at "off" would fall through to the Python
# default, which is "full".
COMMON_FLAGS+=("${TRACE_FLAGS[@]}")

# ---- Group the specs by the context length their system needs ---------------
# Ascending, so the cheap window runs first: if the long-context server is the
# one that cannot allocate its KV cache, lightmem and simplemem are already
# ingested rather than blocked behind it.
declare -a CTXS=()
for spec in "${SPECS[@]}"; do
  CTXS+=("$(ctx_for "$(basename "$(dirname "$spec")")")")
done
mapfile -t CTX_GROUPS < <(printf '%s\n' "${CTXS[@]}" | sort -n -u)

echo "==================================================================="
echo "Context plan (one vLLM start per group):"
for ctx in "${CTX_GROUPS[@]}"; do
  members=""
  for i in "${!SPECS[@]}"; do
    [[ "${CTXS[$i]}" == "$ctx" ]] && members+=" $(basename "$(dirname "${SPECS[$i]}")")"
  done
  printf '  max-model-len %-7s gpu-util %-5s :%s\n' "$ctx" "$(gpu_util_for "$ctx")" "$members"
done

# ---- Ingest each system, one context group at a time ------------------------
declare -a RESULTS=()
fail=0
for ctx in "${CTX_GROUPS[@]}"; do
  if [[ "$DRY_RUN" != "1" && "$SKIP_SERVER" != "1" ]]; then
    stop_server
    if ! start_server "$ctx"; then
      for i in "${!SPECS[@]}"; do
        [[ "${CTXS[$i]}" == "$ctx" ]] || continue
        RESULTS+=("NO-SERVER|$(basename "$(dirname "${SPECS[$i]}")")|-")
      done
      fail=1
      continue
    fi
  fi

  for i in "${!SPECS[@]}"; do
    [[ "${CTXS[$i]}" == "$ctx" ]] || continue
    spec=${SPECS[$i]}
    name=$(basename "$(dirname "$spec")")
    # A spec other than SPEC_NAME gets a suffixed row name.
    [[ "$(basename "$spec")" != "$SPEC_NAME" ]] && \
        name="$name/$(basename "$spec" .yaml | sed "s/^$(basename "$SPEC_NAME" .yaml)_//")"
    safe=${name//\//_}
    log="$LOG_DIR/${safe}_${JOB}.log"

    echo "==================================================================="
    echo ">>> [ingest] $name   ($spec)   max-model-len=$ctx"
    echo ">>> log: $log"
    # One dir per spec under MEM_ROOT — a single shared --mem-dir would have six
    # systems writing their stores on top of each other.
    MEM_FLAGS=()
    [[ -n "$MEM_ROOT" ]] && MEM_FLAGS+=(--mem-dir "$MEM_ROOT/$safe")

    if [[ "$DRY_RUN" == "1" ]]; then
      # printf %q so the printed line is the literal command — copy-pasteable, and
      # an empty or word-split value shows up as such instead of looking correct.
      printf '>>> would run (ctx %s): python -m evalgrid.pipeline.ingest --exp %q' "$ctx" "$spec"
      printf ' %q' "${COMMON_FLAGS[@]}" "${MEM_FLAGS[@]+"${MEM_FLAGS[@]}"}"
      printf '\n'
      continue
    fi

    start=$SECONDS
    python -m evalgrid.pipeline.ingest --exp "$spec" "${COMMON_FLAGS[@]}" "${MEM_FLAGS[@]+"${MEM_FLAGS[@]}"}" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    elapsed=$((SECONDS - start))

    if [[ $rc -eq 0 ]]; then
      RESULTS+=("OK|$name|${elapsed}s")
      echo ">>> [ingest] OK  $name in ${elapsed}s"
    else
      RESULTS+=("FAIL($rc)|$name|${elapsed}s")
      echo "!!! [ingest] FAILED ($rc): $name after ${elapsed}s — see $log" >&2
      fail=1
    fi
  done
done
[[ "$DRY_RUN" != "1" ]] && stop_server

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==================================================================="
  echo "Dry run complete — nothing was executed. If the knobs above are what you"
  echo "intended, re-submit the same line without DRY_RUN=1."
  exit 0
fi

# ---- Summary ---------------------------------------------------------------
echo "==================================================================="
echo "Ingest results — job $JOB, finished $(date)"
printf '%-12s %-30s %8s\n' "STATUS" "SYSTEM" "WALL"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status name elapsed <<< "$r"
  printf '%-12s %-30s %8s\n' "$status" "$name" "$elapsed"
done

echo
echo "Construction cost (${MEM_ROOT:-outputs/_memory}/**/ingest_usage.json):"
MEM_ROOT="$MEM_ROOT" python - <<'PY'
import glob, json, os

root = os.environ.get("MEM_ROOT") or ""
rows = []
# Default: outputs/_memory/<system>/<dataset>/<mem-seg>/ingest_usage.json — the
# mem-seg carries "--<variant>" when the spec set one, so variants list
# separately. Under MEM_ROOT the tree is flat (<root>/<system>/), so the dir
# name IS the label and the dataset is read back out of config.json.
pattern = f"{root}/*/ingest_usage.json" if root else "outputs/_memory/*/*/*/ingest_usage.json"
for path in sorted(glob.glob(pattern)):
    parts = path.split(os.sep)
    if root:
        label, mem_dir = parts[-2], os.path.dirname(path)
        try:
            with open(os.path.join(mem_dir, "config.json")) as f:
                dataset = json.load(f).get("dataset", "?")
        except Exception:
            dataset = "?"
    else:
        system, dataset, mem_seg = parts[2], parts[3], parts[4]
        label = f"{system}--{mem_seg.split('--', 1)[1]}" if "--" in mem_seg else system
    with open(path) as f:
        u = json.load(f)
    n = u.get("samples", 0) or 0
    rows.append((label, dataset, n, u.get("llm_calls", 0),
                 u.get("tokens", 0), u.get("seconds", 0.0)))

if not rows:
    print("  (none written yet)")
else:
    print(f"  {'system':<28}{'dataset':<16}{'samples':>8}{'calls':>10}{'tokens':>12}"
          f"{'s/sample':>10}{'calls/smp':>11}")
    for system, dataset, n, calls, tokens, secs in rows:
        per_s = f"{secs / n:.1f}" if n else "-"
        per_c = f"{calls / n:.1f}" if n else "-"
        print(f"  {system:<28}{dataset:<16}{n:>8}{calls:>10}{tokens:>12}{per_s:>10}{per_c:>11}")
    print("\n  Multiply s/sample and calls/sample by 500 to project a full LongMemEval-S run.")
PY

# A trace run's whole product is the trajectories, so list them rather than
# leaving the reader to guess which dirs got one — "full" writes untruncated
# payloads and a single sample can be tens of MB.
if [[ "$TRACE_LEVEL" != "off" ]]; then
  echo
  echo "Trajectories (trajectory-${TRACE_LEVEL}.jsonl, $TRACE_SCOPE):"
  find "${MEM_ROOT:-outputs/_memory}" -name "trajectory-${TRACE_LEVEL}.jsonl" \
      -exec du -h {} + 2>/dev/null | sort -k2 || echo "  (none written)"
  echo "  Read one: python -m evalgrid.tools.inspect_trajectory <path>"
fi

echo
if [[ $fail -ne 0 ]]; then
  echo "At least one system FAILED — logs in $LOG_DIR/" >&2
  exit 1
fi
echo "All systems ingested. Next: python -m evalgrid.pipeline.qrels_has_answer --exp <spec>"
exit 0
