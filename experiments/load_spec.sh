# Sourced helper — turns an experiment YAML into the bash vars the .sh drivers
# loop over, so the YAML stays the single source of truth (the Python stages
# read the same file via --exp).
#
#   SPEC=experiments/lightmem/longmemeval.yaml
#   source "$(dirname "$0")/load_spec.sh"
#   echo "$SYSTEM $DATASET $MEM_MODEL $LABELER_MODEL $CANDIDATE_SOURCE"
#   IFS=',' read -ra SPECS <<< "$RETRIEVERS"
#
# Exports: SYSTEM DATASET MEM_MODEL MEM_VARIANT LABELER_MODEL CANDIDATE_SOURCE
#          RETRIEVERS GEN_MODEL JUDGE_MODEL
# MEM_VARIANT may be empty (default ingest config) — the Python stages treat
# --mem-variant "" as no variant, so drivers can pass it unconditionally.
#
# Precedence is env > spec: a var already set in the environment is kept (so
# GEN_MODEL=... / DATASET=... overrides still work), otherwise the spec value is
# used. Model names / urls / comma lists carry no $, " or } so they're safe in
# the "${VAR:-value}" form below.
: "${SPEC:?set SPEC=path/to/experiment.yaml before sourcing load_spec.sh}"
[[ -f "$SPEC" ]] || { echo "!!! spec not found: $SPEC" >&2; return 1 2>/dev/null || exit 1; }

eval "$(python - "$SPEC" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
def emit(var, key, default=""):
    v = d.get(key, default)
    if isinstance(v, list):
        v = ",".join(str(x) for x in v)
    print(f'export {var}="${{{var}:-{v}}}"')   # env wins, else spec value
emit("SYSTEM", "system")
emit("DATASET", "dataset")
emit("MEM_MODEL", "mem_model")
emit("MEM_VARIANT", "mem_variant")
emit("LABELER_MODEL", "labeler_model")
emit("CANDIDATE_SOURCE", "candidate_source", "sweep")
emit("RETRIEVERS", "retrievers")
# Deliberately SPEC_NATIVE_RETRIEVERS, not NATIVE_RETRIEVERS: query.sh reads
# NATIVE_RETRIEVER (singular) as "install this one swap", and a var differing by
# one letter, inherited through --export=ALL, is a trap. Loop it in a shell:
#   for R in ${SPEC_NATIVE_RETRIEVERS//,/ }; do ... --native-retriever "$R"; done
emit("SPEC_NATIVE_RETRIEVERS", "native_retrievers")
emit("GEN_MODEL", "gen_model")
emit("JUDGE_MODEL", "judge_model")
PY
)"
