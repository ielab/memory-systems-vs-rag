"""
evalgrid/systems/memorybank/
----------------------------
This repo's adapters for MemoryBank / SiliconFriend (Zhong et al., AAAI 2024,
arXiv 2305.10250).

**This package is the one exception to the vendor-verbatim rule**, and the
exception is forced. Upstream is frozen at 2023-05-24 against `openai==0.27.6`
(`openai.ChatCompletion.create`, removed in the 1.x SDK), `langchain==0.0.146`
(`langchain.vectorstores.FAISS`, `UnstructuredFileLoader` — moved or deleted)
and `llama-index==0.5.23` (`GPTSimpleVectorIndex` — deleted). Nothing in
`vendor/memory_bank/requirement.txt` co-installs on a 2026 Python, so
`local_doc_qa.py` and `forget_memory.py` cannot be imported at all.

What is therefore reused and what is rewritten:

  imported verbatim   the four summarisation prompt builders
                      (`summarize_memory.summarize_content_prompt`,
                      `summarize_person_prompt`, `summarize_overall_prompt`,
                      `summarize_overall_personality`) — that module's only
                      unrunnable part is its LLM client, which we replace, so
                      the prompts themselves import cleanly. They are the
                      highest-value artifact in the repo and are never copied
                      here, so they cannot drift.

  rewritten           the store (store.py), the forgetting curve, the index and
                      the retrieval loop. Each rewritten piece names the
                      upstream file and line it reproduces, and store.py's unit
                      rendering is byte-identical to what upstream's loaders
                      emit — the strings are the part that matters for
                      retrieval.

  deliberately dropped  the llama-index path (`build_memory_index.py`), whose
                      query call is itself an LLM synthesis step and so cannot
                      be expressed as a retrieval run; the Chinese path; the
                      ChatGLM/BELLE LoRA generators; and the AI-companion /
                      psychological-counsellor answer prompts, which live in
                      `utils/prompt_utils.py` and are not vendored here.
"""
