"""
models/embedders/promptreps_embed.py
--------------------------------------
PromptReps (Zhuang et al. 2024, arXiv:2404.18424): prompt
an instruction-tuned LLM to summarize a text as a single word, then read both
retrieval representations off the SAME forward pass at the last token
position (the one that would generate the word):

  * dense  — the final hidden layer's state, L2-normalized  -> EmbedderModel
  * sparse — log(1 + relu(next-token logits)), restricted to the token ids of
    non-stopword words that actually appear in the text, top
    ``sparse_top_k``                                        -> SparseEncoderModel

One class implements both interfaces, so the same model id works with
--retriever dense and --retriever sparse (each retriever run re-encodes and
keeps its own on-disk cache).

The reference implementation hardcodes per-model prompt files; here the same
system/user/assistant-prefix wording is rendered through the model's own chat
template, so any HF chat model works (e.g. meta-llama/Meta-Llama-3-8B-Instruct,
Qwen/Qwen2.5-7B-Instruct). Two intentional deviations from the reference:
in-text words come from bm25.py-style regex tokenization + an inlined stopword
list instead of nltk (avoids nltk's runtime data downloads; it only selects
which token ids are eligible), and sparse weights stay float instead of
round(w*100) (that quantization is a Pyserini impact-index artifact; the
dot-product scoring in evalgrid/benchmarks/retrievers/sparse.py doesn't need it).

Runs locally via HF transformers (no vllm server — the OpenAI API can't
return hidden states/logits). An 8B model in bf16 needs ~17 GB VRAM; note the
full-vocab logits tensor also scales with batch_size, so keep batch_size and
--concurrency moderate.

meta-llama/* checkpoints are gated HF repos — accept the terms on the Hub and
authenticate (huggingface-cli login) before first download.
"""

from __future__ import annotations

import asyncio
import re
import threading
from functools import partial

import numpy as np

from evalgrid.models.base import EmbedderModel, SparseEncoderModel

# Same wording as the reference prompts/, minus the model-specific chat markup
# (which the tokenizer's chat template supplies instead).
_SYSTEM_PROMPT = "You are an AI assistant that can understand human language."
_USER_TEMPLATE = (
    '{kind_title}: "{text}". Use one word to represent the {kind} in a '
    "retrieval task. Make sure your word is in lowercase."
)
_ASSISTANT_PREFIX = 'The word is: "'

# Placeholder split out of the rendered template so corpus text can be
# tokenized separately (and truncated) without re-rendering per text.
_TEXT_MARKER = "<<PROMPTREPS_TEXT>>"

_TOKEN_RE = re.compile(r"\w+")

# nltk's English stopword list (the reference filters these words, plus
# punctuation — which _TOKEN_RE already excludes).
_STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren aren't as at
be because been before being below between both but by can couldn couldn't d
did didn didn't do does doesn doesn't doing don don't down during each few for
from further had hadn hadn't has hasn hasn't have haven haven't having he her
here hers herself him himself his how i if in into is isn isn't it it's its
itself just ll m ma me mightn mightn't more most mustn mustn't my myself needn
needn't no nor not now o of off on once only or other our ours ourselves out
over own re s same shan shan't she she's should should've shouldn shouldn't so
some such t than that that'll the their theirs them themselves then there
these they this those through to too under until up ve very was wasn wasn't we
were weren weren't what when where which while who whom why will with won
won't wouldn wouldn't y you you'd you'll you're you've your yours yourself
yourselves
""".split())


class PromptRepsModel(EmbedderModel, SparseEncoderModel):
    """
    Args:
        model: HF model id of an instruction-tuned causal LM,
            e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
        batch_size: texts per forward pass
        max_text_tokens: truncate each text to this many tokens before it is
            wrapped in the prompt (the reference's passage_max_len)
        sparse_top_k: max in-text token ids kept in the sparse map
        device: e.g. "cpu", "cuda" — None picks cuda when available
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        batch_size: int = 8,
        max_text_tokens: int = 512,
        sparse_top_k: int = 128,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.batch_size = batch_size
        self.max_text_tokens = max_text_tokens
        self.sparse_top_k = sparse_top_k
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Right padding: the last real token is found via attention_mask.sum,
        # keeping RoPE positions correct without custom position_ids.
        self._tokenizer.padding_side = "right"

        self._model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        self._model.to(self._device)
        self._model.eval()

        # (prefix_ids, suffix_ids) around the text slot, per side.
        self._affixes = {
            True: self._render_affixes(kind="query"),
            False: self._render_affixes(kind="passage"),
        }

        # Recent (is_query, texts) -> (dense, sparse) results; see _encode_all.
        self._memo: dict[tuple, tuple] = {}
        self._MEMO_MAX = 32

        # retrieve.py runs samples concurrently and offloads each encode to a
        # thread (loop.run_in_executor). The HF fast tokenizer, the model
        # forward, and the memo dict are all single-instance and not
        # thread-safe (the Rust tokenizer raises "Already borrowed" on
        # concurrent .encode), so serialize the whole encode. One local GPU
        # gains nothing from parallel forwards anyway; asyncio still overlaps
        # everything else while this lock queues the GPU work.
        self._lock = threading.Lock()

    def _render_affixes(self, kind: str) -> tuple[list[int], list[int]]:
        rendered = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    kind_title=kind.title(), kind=kind, text=_TEXT_MARKER)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ) + _ASSISTANT_PREFIX
        prefix, suffix = rendered.split(_TEXT_MARKER)
        encode = partial(self._tokenizer.encode, add_special_tokens=False)
        return encode(prefix), encode(suffix)

    def _text_token_ids(self, text: str) -> set[int]:
        """Token ids of the text's non-stopword words — the ids eligible for
        its sparse representation (encoded word-by-word, as in the reference)."""
        words = {w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS}
        ids: set[int] = set()
        for w in words:
            ids.update(self._tokenizer.encode(w, add_special_tokens=False))
        return ids

    def _sparse_map(self, text: str, weights) -> dict[int, float]:
        """Top-k in-text entries of one text's vocab weight vector; falls back
        to the global top-10 when no in-text token survives the filters."""
        ids = sorted(self._text_token_ids(text))
        if ids:
            import torch
            vals = weights[torch.tensor(ids, device=weights.device)]
            top_vals, top_pos = vals.topk(min(len(ids), self.sparse_top_k))
            pairs = ((ids[p], v) for p, v in zip(top_pos.tolist(), top_vals.tolist()))
        else:
            top_vals, top_ids = weights.topk(10)
            pairs = zip(top_ids.tolist(), top_vals.tolist())
        return {int(t): float(v) for t, v in pairs if v > 0.0}

    def _forward_batch(self, texts: list[str], is_query: bool):
        """One prompted forward pass -> (dense [n, hidden] float32 ndarray,
        sparse [{token_id: weight}, ...])."""
        import torch

        prefix_ids, suffix_ids = self._affixes[is_query]
        seqs = [
            prefix_ids
            + self._tokenizer.encode(
                t, add_special_tokens=False, truncation=True,
                max_length=self.max_text_tokens)
            + suffix_ids
            for t in texts
        ]
        batch = self._tokenizer.pad(
            {"input_ids": seqs}, padding=True, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            out = self._model(**batch, output_hidden_states=True, return_dict=True)

        rows = torch.arange(len(seqs), device=self._device)
        last = batch["attention_mask"].sum(dim=1) - 1
        dense = out.hidden_states[-1][rows, last].float()
        dense = torch.nn.functional.normalize(dense, p=2, dim=-1).cpu().numpy()
        weights = torch.log1p(torch.relu(out.logits[rows, last].float()))
        sparse = [self._sparse_map(t, w) for t, w in zip(texts, weights)]
        return dense, sparse

    def _encode_all(self, texts: list[str], is_query: bool):
        # Both representations come from one forward pass, but embed() and
        # encode_sparse() are separate calls — memoize recent encodes so a
        # hybrid retriever asking for both sides of the same texts pays for
        # one LLM pass, not two. Sized for several samples in flight
        # (retrieve.py --concurrency); worst case on a race/eviction is a
        # recompute, never a wrong result.
        key = (is_query, tuple(texts))
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None:
                return hit

            dense_parts: list[np.ndarray] = []
            sparse: list[dict[int, float]] = []
            for i in range(0, len(texts), self.batch_size):
                d, s = self._forward_batch(texts[i:i + self.batch_size], is_query)
                dense_parts.append(d)
                sparse.extend(s)
            result = (np.concatenate(dense_parts), sparse)

            self._memo[key] = result
            while len(self._memo) > self._MEMO_MAX:
                del self._memo[next(iter(self._memo))]
            return result

    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        dense, _ = await loop.run_in_executor(
            None, partial(self._encode_all, texts, is_query))
        return [v.tolist() for v in dense]

    async def encode_sparse(self, texts: list[str], *, is_query: bool = False) -> list[dict[int, float]]:
        loop = asyncio.get_event_loop()
        _, sparse = await loop.run_in_executor(
            None, partial(self._encode_all, texts, is_query))
        return sparse
