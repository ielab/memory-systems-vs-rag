# Backends are imported lazily in factory.py to avoid hard deps at import time.


def count_processed_tokens(model, texts: list[str]) -> int | None:
    """Tokens a sentence-transformers model actually processes for ``texts``.

    ``attention_mask``, not the raw id count: sentence-transformers cuts every
    input at the model's ``max_seq_length`` (256 for MiniLM-L6-v2) and says
    nothing about it — our development notes record that silent cut — so tokens FED IN
    and tokens the model SAW differ by whatever was dropped. This reports what
    the model saw, which is the quantity a cost column is asking about.

    Excludes the few tokens a prompt/instruction prefix adds, which is not worth
    a second tokenization to recover. None on any failure, so a model whose
    tokenizer does not answer is uncounted rather than zero.

    Shared by the dense and sparse local backends because it is the same
    question for both; only the caller's ``model`` differs.
    """
    try:
        # `preprocess` on current sentence-transformers, `tokenize` on older
        # ones — the rename is live (3.x deprecates the old name and warns on
        # every call), and this repo is pinned to neither.
        prepare = getattr(model, "preprocess", None) or model.tokenize
        features = prepare(texts)
        mask = features.get("attention_mask")
        if mask is not None:
            return int(mask.sum().item())
        ids = features.get("input_ids")
        return int(ids.numel()) if ids is not None else None
    except Exception:
        return None
