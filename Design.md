# Code design

This page is for readers who want to understand or extend the code. To run the experiments, see
the [README](README.md).

The code has three parts:

- **Pipeline stages** (`evalgrid/pipeline/`): the same five steps for every method.
- **System plugins** (`evalgrid/systems/<system>/`): everything specific to one memory system.
- **Experiment specs** (`experiments/<system>/<dataset>.yaml`): which system, dataset, models and
  retrievers a run uses.

The stages contain no system-specific code, so a new system needs only a plugin and a spec.

**Two terms used throughout the code:**

- *Native* means a system's own pipeline: its own retrieval logic, and, with `--prompt native`, its
  own answer prompt.
- *Shared* is the code's name for the paper's *unified* answer prompt, used by every method in
  RQ2 and RQ3.

## Pipeline stages

| Stage | Module | Output |
|---|---|---|
| 1. Build memory | `ingest.py` | `outputs/_memory/` |
| 2. Build corpus and relevance labels | `qrels_has_answer.py` (memory systems), `qrels_turns.py` (raw-history RAG) | `outputs/_qrels/` |
| 3. Retrieve, then score retrieval | `retrieve.py`, `eval_retrieval.py` | `outputs/_retrieval/`, `outputs/_eval/` |
| 4. Answer | `answer.py` | `outputs/_answers/` |
| 5. Judge | `judge.py` | `metrics.<judge>.json` next to each `predictions.jsonl` |

Each stage reads only the previous stage's files, so any stage can be rerun alone.

Raw-history RAG skips stage 1, because its corpus is the conversation turns themselves.
Full-context and Oracle are not separate systems; they are answer-stage conditions
(`--retriever full`, `--retriever oracle`).

## System plugins

Each system implements two classes.

**`SystemPlugin`** (`evalgrid/systems/base.py`) tells the stages how to handle the system:

| Method | What it does |
|---|---|
| `build` | creates the system. Its dependencies are imported lazily, so one system's missing packages never break another. |
| `apply_mem_model` | points the system's memory-construction LLM at a given model |
| `iter_units` | turns a saved store into a list of memory units, each with an id, text, timestamp and the turns it was built from |
| `answer_prompt` | the system's own answer prompt; the default is the unified prompt |
| `native_unit_ids` | the memory units the system's own retrieval returned, in rank order |
| `apply_retrieval_budget`, `native_retriever_name` | set the retrieval depth, and name the retriever used inside the system's pipeline |

**`MemorySystem`** (`evalgrid/benchmarks/types.py`) is the system itself, with `reset`, `ingest`,
`answer`, `save_memory` and `load_memory`. Its `answer` runs the system's real query-time
pipeline.

**Why `iter_units` matters.** Each memory unit records the conversation turns it was built
from, and a unit counts as relevant when one of those turns is annotated evidence. This gives
every system relevance labels on the same scale. The labels score the system's own retrieval
and any swapped-in retriever, and they define the oracle context (paper Appendix D).

## Paper conditions in the code

| Condition | Retrieval | Answer prompt |
|---|---|---|
| RQ1: original configuration | the system's own pipeline at its published depth | the system's own (MemoryBank and Mem0 have none and use the unified one) |
| RQ2: controlled retrievers | the system's own pipeline, with one of the 8 retrievers in its search step | unified |
| Raw-history RAG | a retriever over turns or turn pairs | unified |
| Full-context | none: the most recent 100K tokens of history | unified |
| RQ3: Oracle | none: units built from an evidence turn | unified |

The underlying commands are below. The scripts in `scripts/` and `experiments/` build them for
you.

```bash
# RQ1: retrieval and answering in one pass
answer.py --retriever native --prompt native --published-depth \
          --config evalgrid/configs/<sys>/naive_query.yaml --native-variant naive
# RQ2: save the ranking once, then answer from it at each budget
retrieve.py --retriever native --native-retriever <retriever> --native-variant grid
answer.py   --retriever native --prompt shared --retrieval-dir <ranking> --context-budget <B>   # or --top-k <k>
# Raw-history RAG
retrieve.py --retriever bm25|sparse|dense|rrf [--emb-model <model>]
answer.py   --retriever <same> --top-k <k>   # or --context-budget <B>
# Full-context and Oracle
answer.py --retriever full --max-history-tokens 100000
answer.py --retriever oracle --gold turn
```

**Swapping the retriever (RQ2).** `--native-retriever` replaces only the system's search step.
Query planning, link expansion, regrouping, reranking and reflection stay as the system defines
them (paper Appendix C).

**Scoring a system's own retrieval.** The system's pipeline returns ranked units, not scores. The
code writes them as a `run.tsv` with score `1/rank`, which is enough for recall and NDCG. The
`run.tsv` is scored against the same labels as every other retriever.

**Context budgets.** A top-k budget keeps the first k units. A token budget adds units in rank
order until the next one would exceed B tokens, counted with the generator's tokenizer; only an
oversized first unit is truncated. Each answer cell's `answer_summary.json` records how often
the budget truncated or was not filled.

## Configs and specs

- `evalgrid/configs/<system>/default.yaml` holds the construction settings used for every store.
- `evalgrid/configs/<system>/naive_query.yaml` has the same construction settings, plus the
  system's published query-time settings, for RQ1.
- `experiments/<system>/<dataset>.yaml` names the system, dataset, models and the 8 retrievers:
  `retrievers:` for raw-history RAG, `native_retrievers:` for retrievers inside a system.
- Two variant options add a suffix to output directory names. `--mem-variant` marks a different
  store, so it also changes `outputs/_memory/`. `--native-variant` marks different retrieval
  settings on the same store, so it changes only the retrieval and answer directories.

## Output layout

```
outputs/_memory/<system>/<dataset>/<mem-model>/<sample>/
outputs/_qrels/<system>/<dataset>/<mem-model>/has_answer/
outputs/_retrieval/<system>/<dataset>/<mem-model>/<retriever>/run.tsv
outputs/_eval/<system>/<dataset>/<mem-model>/has_answer/<retriever>/metrics-gold-turn.json
outputs/_answers/<system>/<dataset>/<mem-model>/<retriever>/<generator>/<budget>/predictions.jsonl
```

- `<mem-model>` is the construction model, or `none` for raw-history RAG.
- `<budget>` is `top<k>` or `budget<B>`.
- A `<retriever>` inside a system reads like `native-emb-Qwen_Qwen3-Embedding-0.6B-top270--grid`:
  the swapped-in retriever, the retrieval depth, and the variant.

## Adding a system

1. Copy the upstream code into `vendor/<name>/` unchanged, and add it to
   [vendor/VERSIONS.md](vendor/VERSIONS.md).
2. In `evalgrid/systems/<name>/`, write a `SystemPlugin` and a `MemorySystem` subclass.
3. Add `evalgrid/configs/<name>/default.yaml`, plus `naive_query.yaml` if the system's published
   query-time settings differ.
4. Add one line to `_REGISTRY` in `evalgrid/systems/registry.py`.
5. Add `experiments/<name>/<dataset>.yaml`, and a test case in
   `evalgrid/tools/check_system_contract.py`.

After changing a plugin, a prompt or `lib/paths.py`, run
`python -m evalgrid.tools.check_system_contract`.
