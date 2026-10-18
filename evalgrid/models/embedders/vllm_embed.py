"""
models/embedders/vllm_embed.py
--------------------------------
Text embedding via a vllm server started with --task embed.

Start the server before using:
    vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001

Uses the OpenAI-compatible /v1/embeddings endpoint.

Many embedding model families are trained asymmetrically: queries and
documents need different (or one-sided) text prefixes to hit the
representation the model was actually tuned for — see
models/embedders/instructions.py for the per-family rules (Qwen3-Embedding,
E5, BGE, ...). Skipping this can cost several points of retrieval quality.
The right rule is auto-detected from `model`'s name; override with
query_prefix/doc_prefix if a model isn't recognized or you want to try a
different instruction.

TRUNCATION. A served embedder REFUSES an input longer than its window — a 400
that aborts whatever was ranking — where the local sentence-transformers backend
silently cuts to the model's `max_seq_length` (256 tokens for MiniLM-L6-v2, so
most a_mem notes are already being cut there). One a_mem note of 16,385 tokens
against a 16,384-token server therefore killed four cells of the retriever grid
at the same sample while the four local-embedder cells finished.

So this backend truncates too, and the point of the code below is that it says
so: the cut is made at the server's OWN window (never lower, so a doc that fits
is never touched and a resumed run stays consistent with what it already
wrote), it is counted, and the totals are printed at exit. A silent cut here
would be the same defect as the local backend's, just better hidden.
"""

from __future__ import annotations

import atexit
import os
import re
import sys

import openai

from evalgrid.models.base import EmbedderModel
from evalgrid.models.embedders.instructions import default_rule

#: The server states its own window when it refuses a request:
#:   "This model's maximum context length is 16384 tokens. However, you
#:    requested 0 output tokens and your prompt contains at least 16385 ..."
#: That sentence is the last and most reliable source of the number — it comes
#: from the process that will accept or reject the next request, so it cannot
#: disagree with it the way a model card or an env var can.
_LIMIT_FROM_400 = re.compile(r"maximum context length is (\d+) tokens")


def _limit_in(exc: BaseException) -> int | None:
    m = _LIMIT_FROM_400.search(str(exc))
    return int(m.group(1)) if m else None


#: Set by whatever started the embedding server, from the same value it passed
#: to --max-model-len. The fallback for a vLLM whose /v1/models card does not
#: report max_model_len — see _probe_window.
_WINDOW_ENV = "EVALGRID_EMB_MAX_INPUT_TOKENS"

#: Tokens left free under the server's window. vLLM counts the prompt AFTER any
#: special tokens the model's template adds, and the client cannot see those, so
#: the cut leaves room for them. Negligible against a 16k window; the failure it
#: prevents is a 400 that ends a run.
_WINDOW_MARGIN = 16

#: Process-wide, so the summary covers every embedder a run built rather than
#: whichever one happened to be last. {model: [docs, tokens_dropped]}
_TRUNCATED: dict[str, list[int]] = {}


def _report_truncation() -> None:
    if not _TRUNCATED:
        return
    print("\n!!! embedder input truncated to the served window:", file=sys.stderr)
    for model, (docs, dropped) in sorted(_TRUNCATED.items()):
        print(f"!!!   {model}: {docs} doc(s), {dropped:,} token(s) dropped",
              file=sys.stderr)
    print("!!!   Those documents were indexed SHORT. The local sentence-transformers "
          "backend does the same cut without counting it.", file=sys.stderr)


atexit.register(_report_truncation)



def _prompt_tokens(response) -> int | None:
    """Tokens the server says it processed, or None if it did not say.

    vLLM's OpenAI-compatible /v1/embeddings returns a usage block, but not
    across every version, and the client leaves it None when absent. An absent
    count must read as "not counted", never as zero: a zero would understate
    the encoder in the one file whose purpose is to state it.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        n = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)
        return int(n) if n else None
    except Exception:
        return None


class VLLMEmbedder(EmbedderModel):
    """
    Args:
        model: model name as registered in the vllm server
        base_url: vllm server URL (default: http://localhost:8001/v1)
        api_key: vllm accepts any non-empty string
        batch_size: max texts per request
        query_prefix: literal text prepended to every query. Defaults to the
            known convention for `model`'s family (models/embedders/
            instructions.py), or "" (no prefix) if unrecognized. Pass "" to
            force-disable.
        doc_prefix: literal text prepended to every document. Same
            defaulting as query_prefix — most families need none (E5 is the
            notable exception: it prefixes both sides differently).
        max_input_tokens: the served window, which is what an over-long input is
            cut to. "auto" (default) reads it from the server's own
            /v1/models once, so the cut always matches the server actually
            answering rather than a number pasted into a launch script. An int pins
            it; None disables truncation and restores the 400.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "token",
        batch_size: int = 64,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        max_input_tokens: int | str | None = "auto",
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.max_input_tokens = max_input_tokens
        self._window: int | None = None if max_input_tokens == "auto" else max_input_tokens
        self._window_probed = max_input_tokens != "auto"
        self._tokenizer = None
        self._tokenizer_tried = False

        rule = default_rule(model)
        self.query_prefix = query_prefix if query_prefix is not None else (rule.query_prefix if rule else "")
        self.doc_prefix = doc_prefix if doc_prefix is not None else (rule.doc_prefix if rule else "")

    def _format(self, texts: list[str], is_query: bool) -> list[str]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    async def _probe_window(self) -> int | None:
        """The window of the server that will answer, read once from /v1/models.

        Asked of the server first because a number pasted into a launch script and the
        server actually answering can disagree, and only the latter prevents the
        400. But `max_model_len` is not in every vLLM's ModelCard, so a listing
        that simply lacks the field is the ORDINARY case, not an error — hence
        the env fallback, which the job that started the server exports from the
        same variable it passed to --max-model-len.

        Every outcome is announced. The first version of this warned only when
        the listing RAISED, so a server whose card had no `max_model_len` turned
        truncation off and said nothing — a silent no-op, which is the same
        defect this whole file exists to remove. That is how a_mem's 400 came
        back after the fix shipped.
        """
        if self._window_probed:
            return self._window
        self._window_probed = True
        why = ""
        try:
            listing = await self.client.models.list()
            ids = [c.id for c in listing.data]
            card = next((c for c in listing.data if c.id == self.model), None)
            if card is None:
                why = (f"the server serves {ids} — {self.model!r} is not among them, "
                       f"so its window is unknown")
            else:
                extra = getattr(card, "model_extra", None) or {}
                n = extra.get("max_model_len") or getattr(card, "max_model_len", None)
                if n:
                    self._window = int(n)
                else:
                    why = "this vLLM's /v1/models card carries no max_model_len"
        except Exception as e:
            why = f"/v1/models could not be read ({type(e).__name__}: {e})"

        if self._window is None:
            env = os.environ.get(_WINDOW_ENV)
            if env and env.isdigit() and int(env) > 0:
                self._window = int(env)
                print(f"!!! embedder window from {_WINDOW_ENV}={self._window} "
                      f"({why}); truncation is ON at that value", file=sys.stderr)
            else:
                print(f"!!! TRUNCATION IS OFF for {self.model}: {why}, and "
                      f"{_WINDOW_ENV} is unset. An input longer than the served "
                      f"window will come back as a 400 and cost that sample.",
                      file=sys.stderr)
        return self._window

    def _cut(self, text: str, limit: int) -> str:
        """`text` shortened to `limit` tokens, or unchanged if it already fits.

        The pre-check counts UTF-8 BYTES, and the difference is not pedantry.
        Qwen3 tokenises byte-level BPE, where one multi-byte character can
        become several tokens — so "a token is at least one character", which an
        earlier version of this bounded on, is FALSE, and any text of non-ASCII
        content could sail past the check and 400 anyway. Every token covers at
        least one byte in byte-level BPE and in sentencepiece alike, so bytes
        are a sound bound where characters are not. For ASCII the two are equal,
        which is why this still keeps the tokenizer off the path for everything
        but the rare outlier — and that matters, because this backend re-embeds
        the whole corpus on every query.
        """
        if len(text.encode("utf-8")) < limit:
            return text
        if not self._tokenizer_tried:
            self._tokenizer_tried = True
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            except Exception as e:
                print(f"!!! no tokenizer for {self.model} ({type(e).__name__}: {e}); "
                      f"falling back to a character cut, which is approximate",
                      file=sys.stderr)
        keep = max(1, limit - _WINDOW_MARGIN)
        if self._tokenizer is None:
            # One character per token is the worst case, so cutting at `keep`
            # characters cannot leave more than `keep` tokens.
            return text[:keep]
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= keep:
            return text
        counter = _TRUNCATED.setdefault(self.model, [0, 0])
        counter[0] += 1
        counter[1] += len(ids) - keep
        return self._tokenizer.decode(ids[:keep])

    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        texts = self._format(texts, is_query)
        # After _format: the prefix counts toward the window too, and an input
        # one token over is exactly the case this exists for.
        if self.max_input_tokens is not None:
            limit = await self._probe_window()
            if limit:
                texts = [self._cut(t, limit) for t in texts]
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model, input=batch
                )
            except openai.BadRequestError as e:
                # LAST RESORT, and the only one that cannot be misconfigured.
                # /v1/models may not carry max_model_len and the env var may not
                # be set — both were true on the a_mem grid, and truncation
                # stayed off through two attempted fixes. The
                # 400 itself names the window, so a request refused for length
                # is re-shaped to what the server just said it accepts and sent
                # ONCE more.
                #
                # On the repo's no-retry rule:
                # that rule exists to stop upstream's `while True` loops
                # resending an identical prompt that cannot ever fit. This is
                # the opposite — one send of a DIFFERENT, corrected request,
                # only for a length refusal, only once, and counted. If the
                # re-shaped request is refused too, the error propagates and the
                # sample is skipped and recorded like any other failure.
                limit = _limit_in(e)
                if limit is None or self.max_input_tokens is None:
                    raise
                if self._window != limit:
                    print(f"!!! the embedding server refused a {self.model} request for "
                          f"length and reports its window as {limit}; truncating to that "
                          f"and sending once more", file=sys.stderr)
                self._window, self._window_probed = limit, True
                cut = [self._cut(t, limit) for t in batch]
                if cut == batch:
                    # Nothing was over the limit by our count, so re-sending
                    # would be the identical request the rule forbids.
                    raise
                response = await self.client.embeddings.create(
                    model=self.model, input=cut
                )
            self._record_encode(len(batch), _prompt_tokens(response))
            results.extend(item.embedding for item in response.data)
        return results
