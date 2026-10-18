# Vendored systems

Each memory system runs from a copy of its upstream code, pinned to the commit below. Each copy
keeps its upstream licence in `vendor/<system>/LICENSE`. Most adaptations are made in the adapters
(`evalgrid/systems/<system>/`) and are described in the paper's Appendix A.1 and B.1. The edits
to the vendored code itself are listed here. SimpleMem's unused `integrations/` folder is not
included.

| System | Upstream | Commit | Licence | Edited |
|---|---|---|---|---|
| LightMem | [zjunlp/LightMem](https://github.com/zjunlp/LightMem) | `8a9bab3` | MIT | yes, see below |
| A-Mem | [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) | `0c8039f` | MIT | no |
| SimpleMem | [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem) | `db80b6a` | MIT | no |
| HyperMem | [EverMind-AI/HyperMem](https://github.com/EverMind-AI/HyperMem) | `15c7009` | Apache-2.0 | no |
| MemoryBank | [zhongwanjun/MemoryBank-SiliconFriend](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | `cf61c41` | MIT | no |
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | `v0.1.118` | Apache-2.0 | one line, see below |

## LightMem

- **Provenance:** each saved memory records the session and turn it came from, which the qrels
  and the oracle need. The extraction prompt also states the valid `source_id` range, because
  models produced out-of-range ids.
- **Settings:** the short-term-memory size and the update similarity threshold were hard-coded
  upstream (512, 0.8). They are now config values.
- **LLMLingua-2:** the compressor calls the LLMLingua-2 entry point the paper describes, and no
  longer re-compresses text in a loop. The topic segmenter truncates sentences longer than the
  model's 512-token limit, and works under transformers ≥ 5.
- **Crash fixes:** an over-long single message no longer causes an infinite loop, and an odd
  number of messages no longer raises an error.

## Mem0

Mem0 is pinned to 0.1.118, the release that matches the Mem0 paper; `pip install mem0ai` now
installs a different 2.x system. One line in `__init__.py` changed: the version is a literal
instead of being read from pip metadata, which fails for a copy that is not pip-installed.
