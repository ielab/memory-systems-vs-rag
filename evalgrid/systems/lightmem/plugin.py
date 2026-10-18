"""
evalgrid/systems/lightmem/plugin.py
-----------------------------------
LightMem's SystemPlugin — the reference implementation of the contract.

Holds the three things the pipeline used to hardcode about LightMem: how to
build it (including the config surgery that was inlined in ingest.py), and how
to read its saved memory back as a flat corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from evalgrid.systems.base import MemoryUnit, SystemPlugin, route_mem_llm_to_openai

# Defaults for the system's own retrieval/answer budget. These used to be dead
# ("_UNUSED_TOP_K"): the staged pipeline retrieves separately, so LightMemory's
# own top_k never fired. The native answer path DOES use
# it, so both are now read from the config and settable with
# --top-k / --max-tokens, like every other system's.
_DEFAULT_TOP_K = 10
_DEFAULT_ANSWER_MAX_TOKENS = 2000

# Keys this plugin consumes itself. They are popped before the rest of the YAML
# is handed to BaseMemoryConfigs, which does not know them.
_ADAPTER_KEYS = ("locomo_compress_rate", "retrieve_k", "answer_max_tokens")


class LightMemPlugin(SystemPlugin):

    name = "lightmem"

    # Per-fact source_id back to one turn. Note the corpus is user turns
    # only, so an assistant-side evidence turn is unreachable rather than
    # mislabelled.
    has_answer_granularity = "turn"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, config: dict, *, dataset: str, generator, phase: str):
        # Imported here, not at module scope: LightMem pulls in qdrant_client
        # and torch, which must not be a prerequisite for running some other
        # system (or for merely listing the registry).
        from lightmem.configs.base import BaseMemoryConfigs

        config = json.loads(json.dumps(config))  # don't mutate the caller's dict

        adapter = {k: config.pop(k) for k in _ADAPTER_KEYS if k in config}

        # BaseMemoryConfigs is a plain pydantic BaseModel, i.e. extra="ignore":
        # a misspelt key is dropped in silence and the run proceeds on the
        # default. Checked here instead, where the file name is still known.
        unknown = sorted(set(config) - set(BaseMemoryConfigs.model_fields))
        if unknown:
            raise ValueError(
                f"lightmem config has keys LightMem does not define: {unknown}. "
                f"pydantic would drop them silently and run on the defaults. "
                f"Adapter-level keys are {list(_ADAPTER_KEYS)}; everything else "
                f"must be a BaseMemoryConfigs field."
            )

        # Each parallel process needs its own Qdrant SQLite path — concurrent
        # writes to one file raise "readonly database". PID makes it unique even
        # for non-sharded parallel runs; all of them nest under the configured
        # base dir (qdrant_data/<pid>/).
        qdrant_cfg = config.setdefault("embedding_retriever", {}).setdefault("configs", {})
        base = qdrant_cfg.get("path", "qdrant_data").rstrip("/")
        qdrant_cfg["path"] = os.path.join(base, str(os.getpid()))

        # LoCoMo: flat extraction matching the reference LightMem baseline
        # (zjunlp/LightMem/experiments/locomo, --extraction_mode flat). The
        # reference uses messages_use="user_only": every real utterance is a
        # "user" message (see LightMemLoCoMo.ingest), so both speakers are
        # captured. LongMemEval also uses user_only (yaml default).
        if dataset == "locomo":
            config["extraction_mode"] = "flat"
            config["messages_use"] = "user_only"
            # The two datasets use different compression rates in the reference:
            # locomo 0.6, longmemeval 0.8 (it omits compress_config entirely and
            # takes the default). `rate` in the YAML is therefore the LongMemEval
            # rate, and the LoCoMo one is named separately so a variant can move
            # both. Until 2026-08-12 this line forced 0.6 unconditionally, which
            # collapsed every rate variant to 0.6 on LoCoMo while the output dir
            # still carried the variant's name — a rate0.4-stm768 LoCoMo store
            # built before that date is mislabelled and was really built at 0.6.
            cc = (config.setdefault("pre_compressor", {})
                        .setdefault("configs", {})
                        .setdefault("compress_config", {}))
            cc["rate"] = adapter.get("locomo_compress_rate", 0.6)

        lm_config = BaseMemoryConfigs(**config)

        # LightMem's own retrieval/answer budget. Adapter-level, not
        # BaseMemoryConfigs fields, hence popped above. Only the native path
        # reads them — the staged pipeline imposes its own --top-k/--max-tokens.
        kwargs = {
            "top_k": int(adapter.get("retrieve_k", _DEFAULT_TOP_K)),
            "max_tokens": int(adapter.get("answer_max_tokens", _DEFAULT_ANSWER_MAX_TOKENS)),
        }

        # Per-dataset subclass: each owns its ingest policy.
        if dataset == "locomo":
            from evalgrid.systems.lightmem.locomo import LightMemLoCoMo
            return LightMemLoCoMo(lm_config, generator, **kwargs)
        if dataset.startswith("beam"):
            from evalgrid.systems.lightmem.beam import LightMemBEAM
            return LightMemBEAM(lm_config, generator, **kwargs)
        from evalgrid.systems.lightmem.longmemeval import LightMemLongMemEval
        return LightMemLongMemEval(lm_config, generator, **kwargs)

    def native_retriever_name(self, config: dict) -> str:
        """LightMem keeps its encoder under text_embedder.configs.model.

        Its native retrieval is one dense search over that encoder's vectors —
        so this name is the SAME one the flat dense run gets, and the two
        rankings should be identical at equal k.
        """
        from evalgrid.lib.paths import retriever_name
        # A substituted retriever wins, exactly as in the base implementation:
        # overriding this method must not lose that, or a swapped run would be
        # named after the published encoder and land on top of it.
        from evalgrid.systems.native_retrieval import retriever_slug

        swapped = retriever_slug(config)
        if swapped:
            return swapped

        model = ((config.get("text_embedder") or {}).get("configs") or {}).get("model")
        if not model:
            raise ValueError(
                "lightmem config has no text_embedder.configs.model, so a native run dir "
                "could not name the encoder it ranked with.")
        return retriever_name("dense", model)

    def apply_mem_model(self, config: dict, mem_model: str,
                        mem_base_url: Optional[str] = None) -> dict:
        """LightMem's memory_manager is what turns dialogue into memory entries,
        so it is the model that keys outputs/_memory/."""
        mm = config.setdefault("memory_manager", {}).setdefault("configs", {})
        mm["model"] = mem_model
        if mem_base_url:
            mm["openai_base_url"] = mem_base_url
        route_mem_llm_to_openai(mm, mm.get("openai_base_url"), mem_model)
        return config

    # ------------------------------------------------------------------
    # Flatten saved memory into a retrieval corpus
    # ------------------------------------------------------------------

    def iter_units(self, sample_mem_dir: str) -> Iterable[MemoryUnit]:
        """One unit per Qdrant point dumped by LightMemSystem.save_memory.

        LightMem's retrievable state lives entirely in its embedding store: one
        point per extracted memory, carrying the memory text plus a payload
        (timestamps, tags, has_answer). save_memory dumps them already sorted by
        float_time_stamp, so reading the file in order preserves chronology —
        which is what the full-context baseline feeds the generator.

        Points predating the has_answer-aware ingester have no such key. That is
        not detectable per unit (absent and False look alike), so the qrels
        builder checks the sample as a whole; see units_carry_has_answer.

        Provenance comes from ``provenance.jsonl`` when that sidecar is present,
        and from the point payload when it is not.

        The payload is not trustworthy on its own. LightMem prices a fact to a
        single source turn: the extractor returns a ``source_id`` naming the
        turn-pair the fact came from, and utils.py:338-339 reads
        ``turn_id_list[source_id * 2]``. That inverse holds only while every user
        message sits at an EVEN global sequence number, and our pairing step
        breaks exactly that — it skips leading non-user turns and drops the odd
        tail, 2,020 turns over LongMemEval-S, and each drop flips the parity of
        everything after it. Measured on the shipped stores, 28.15% of user
        messages land on an odd sequence number, so their facts are charged to
        the neighbouring ASSISTANT turn — which is why ~26% of saved units carry
        an assistant turn id under ``messages_use: user_only``, a combination
        that should be impossible. ``has_answer`` reads the same broken index and
        is wrong in the same places, so neither field can check the other.

        Since 2026-09-02 the sidecar is written by the INGEST itself:
        ``LightMemSystem`` recomputes the label the extraction prompt printed and
        joins it to the ``source_id`` the model returned, while both halves are
        still in hand (``_capture_provenance``), and ``save_memory`` writes the
        result beside ``points.jsonl``. So a store built by a current ingest
        always has one, and no post-hoc step has to be remembered.

        ``evalgrid.tools.lightmem_rebuild_provenance`` reconstructs the same
        mapping from the ingest trajectory for stores built BEFORE that. It is a
        repair tool for old stores, not part of the pipeline; it resolves fewer
        units than the ingest-time capture (no pre-update text to join through,
        and it parses the extractor's replies concatenated), so do not run it
        over a store that already has a sidecar.

        The payload branch below stays for stores older still, which have no
        sidecar at all, so they keep reading exactly as they did.
        """
        path = os.path.join(sample_mem_dir, "points.jsonl")
        if not os.path.isfile(path):
            return
        fixed = self._rebuilt_provenance(sample_mem_dir)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                point = json.loads(line)
                payload = point.get("payload") or {}
                record = fixed.get(point["id"])
                if record is not None:
                    source_turn_ids = list(record.get("source_turn_ids") or [])
                    has_answer = bool(record.get("has_answer"))
                else:
                    # "turn_id" is what vendor/lightmem/memory/lightmem.py:439
                    # writes. This used to read "source_ids", a key nothing in
                    # the tree has ever written, so source_turn_ids was silently
                    # [] on every lightmem unit — on the one system whose docs
                    # advertise per-turn provenance.
                    turn_id = payload.get("turn_id")
                    source_turn_ids = [turn_id] if turn_id else []
                    has_answer = bool(payload.get("has_answer"))
                yield MemoryUnit(
                    unit_id=point["id"],
                    text=payload["memory"],
                    timestamp=payload.get("time_stamp"),
                    has_answer=has_answer,
                    source_turn_ids=source_turn_ids,
                )

    @staticmethod
    def _rebuilt_provenance(sample_mem_dir: str) -> dict:
        """unit_id -> rebuilt provenance record, or {} when no sidecar exists."""
        path = os.path.join(sample_mem_dir, "provenance.jsonl")
        if not os.path.isfile(path):
            return {}
        out = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                out[record["unit_id"]] = record
        return out

    def records_has_answer(self, sample_mem_dir: str) -> bool:
        """True if this sample's saved points carry the has_answer key.

        Stores built before the has_answer-aware ingester have no such key, and
        would otherwise silently produce empty qrels for every sample.
        """
        path = os.path.join(sample_mem_dir, "points.jsonl")
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and "has_answer" in (json.loads(line).get("payload") or {}):
                    return True
        return False

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def answer_prompt(self, question: str, question_date: Optional[str],
                      facts: list[str]) -> list[dict]:
        """LightMem's published LongMemEval prompt (--prompt native).

        The one system here with a prompt worth reproducing verbatim: the shared
        prompt was derived from it, so running LightMem under both measures
        exactly what the whitespace re-flow costs — which is the control for
        every other system's shared-prompt number.
        """
        from evalgrid.systems.lightmem.longmemeval import native_answer_prompt

        return native_answer_prompt(question, question_date, facts)
