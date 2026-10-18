"""
evalgrid/systems/simplemem/
---------------------------
This repo's adapters for SimpleMem (Liu et al., arXiv 2601.02553).

Upstream lives at vendor/simplemem/, byte-identical to aiming-lab/SimpleMem.
Nothing here edits it: the adapters change upstream behaviour only by assigning
over attributes or driving its components directly, each marked ADAPTER: at the
site. system.py's docstring lists them.

Only ``simplemem.core.*`` and ``simplemem.text.system`` are used. The vendored
``multimodal/``, ``evolver/`` and ``integrations/`` trees are separate products
(Omni-SimpleMem, EvolveMem, the MCP server) and are never imported — in
particular ``evolver/benchmarks/longmemeval.py``, whose answer prompt is
conditioned on LongMemEval's gold ``question_type`` field.
"""
