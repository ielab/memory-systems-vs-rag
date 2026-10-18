"""
evalgrid/systems/a_mem/
-----------------------
This repo's adapters for A-Mem (Xu et al., NeurIPS 2025, arXiv 2502.12110).

Upstream lives at vendor/a_mem/, byte-identical to WujiangXu/A-mem @ 0c8039f.
Nothing here edits it: the adapters change upstream behaviour only by assigning
over attributes or binding names into its modules, each marked ADAPTER: at the
site. system.py's docstring lists them.
"""
