"""
evalgrid/systems/hypermem/
--------------------------
This repo's adapters for HyperMem (Yue et al., ACL 2026, arXiv 2604.08256).

Upstream lives at vendor/hypermem/, byte-identical to EverMind-AI/HyperMem.
Nothing here edits it: the adapters change upstream behaviour only by
substituting a provider, driving its extractors directly, or overriding config
class attributes — each marked ADAPTER: at the site. system.py's docstring lists
them.

HyperMem ships no library entry point at all: no ``HyperMem`` class, no
``add``/``search``, just six ``python stageN_*.py`` scripts wired to
``locomo10.json`` through ``results/<experiment>/`` directories. What this
package provides is the orchestration those scripts' ``main()`` functions hold,
re-expressed against evalgrid's MemorySystem interface, calling upstream's
extractors (and therefore upstream's prompts) unchanged.
"""
