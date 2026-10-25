# scripts/

Shell scripts for the GPU stages. Each one starts the vLLM servers it needs, runs the stages,
and stops the servers at the end. Run them from the repo root, setting options as environment
variables.

| Script | Does | Paper |
|---|---|---|
| `ingest.sh` | builds memory | all RQs |
| `query.sh` | each system's own retrieval and answer prompt | RQ1 |
| `native_grid.sh` | the 8 retrievers inside each system's pipeline, then the answer budgets | RQ2 |
| `oracle.sh` | answers from each store's gold units | RQ3 |
| `ingest_openai.sh` | builds memory with gpt-4o-mini through the OpenAI API | appendix |
| `lib/vllm.sh` | shared helpers, sourced by the scripts above | — |

The README's [Run the experiments](../README.md#4-run-the-experiments) section has the exact command for each step.

## Common options

| Variable | Default | Meaning |
|---|---|---|
| `LIMIT` | 3 (4 for `native_grid.sh`) | number of samples; `LIMIT=` (empty) means the full dataset |
| `SPEC_NAME` | `longmemeval.yaml` | `beam_100k.yaml` for BEAM-100K |
| `MEM_ROOT` | `outputs/` (`outputs/_smoke4` for `native_grid.sh`) | where the stores are; `MEM_ROOT=` means `outputs/` |
| `NUM_SHARDS`, `SHARD_ID` | 1, 0 | split one run over N processes, each with its own `SHARD_ID` |
| `SKIP_SERVER` | 0 | 1 = start no servers and use ones already running |
| `CONDA_ENV` | empty | a conda env to activate; empty means the active Python |
| `DRY_RUN` | 0 | 1 = print the commands without running them |

Each script's header lists the rest of its options.

## Servers

| Model | Port | Context window |
|---|---|---|
| Memory model / generator | 8000 (`VLLM_PORT`) | 131,072 for HyperMem, MemoryBank, Mem0, A-Mem and the raw-history baselines; 32,768 for LightMem and SimpleMem |
| Qwen3-Embedding-0.6B / -4B | 8001 / 8002 | model default |

If 131,072 does not fit on your GPU, lower `CTX_LONG` or `GPU_UTIL_LONG` (set in
`lib/vllm.sh`). Logs are written to `logs/`.
