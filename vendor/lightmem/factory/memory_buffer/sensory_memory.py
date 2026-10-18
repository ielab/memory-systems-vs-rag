import logging
import numpy as np
from typing import List, Dict, Optional, Any

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class SenMemBufferManager:
    def __init__(self, max_tokens: int = 512, tokenizer = None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.buffer: List[Dict] = []
        self.big_buffer: List[Dict] = []
        self.token_count: int = 0

    def _encode_len(self, text: str) -> int:
        """Token count without triggering the model_max_length warning."""
        return len(self.tokenizer.encode(text, truncation=False, verbose=False))

    def _recount_tokens(self) -> None:
        self.token_count = sum(self._encode_len(m["content"]) for m in self.buffer if m["role"] == "user")

    def add_messages(self, messages: List[Dict], segmenter, text_embedder) -> None:
        all_segments = []
        self.big_buffer.extend(messages)

        while self.big_buffer:
            processed_messages = []
            for msg in self.big_buffer:
                if msg["role"] == "user":
                    cur_token_count = self._encode_len(msg["content"])
                    if self.token_count + cur_token_count <= self.max_tokens or self.token_count == 0:
                        # Force-add when buffer is empty to guarantee progress even if a
                        # single message exceeds max_tokens (otherwise infinite loop).
                        self.buffer.append(msg)
                        self.token_count += cur_token_count
                        processed_messages.append(msg)
                    else:
                        segments = self.cut_with_segmenter(segmenter, text_embedder)
                        all_segments.extend(segments)
                        break
                else:
                    self.buffer.append(msg)
                    processed_messages.append(msg)
            for msg in processed_messages:
                self.big_buffer.remove(msg)

        return all_segments

    def should_trigger(self) -> bool:
        return self.token_count >= self.max_tokens

    def cut_with_segmenter(self, segmenter, text_embedder, force_segment: bool=False) -> List:
        """
        Cut buffer into segments using a two-stage strategy:
        1. Coarse boundaries from segmenter.
        2. Fine-grained adjustment based on semantic similarity.
        """
        segments = []
        buffer_texts = [m["content"] for m in self.buffer if m["role"] == "user"]
        boundaries = segmenter.propose_cut(buffer_texts)

        if not boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        turns = []
        for i in range(0, len(self.buffer) - 1, 2):
            user_msg = self.buffer[i]["content"]
            assistant_msg = self.buffer[i + 1]["content"]
            turns.append(user_msg + " " + assistant_msg)

        if not turns:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        embeddings = []
        for turn in turns:
            emb = text_embedder.embed(turn)
            embeddings.append(np.array(emb, dtype=np.float32))
        embeddings = np.vstack(embeddings)

        fine_boundaries = []
        threshold = 0.2
        while threshold <= 0.5 and not fine_boundaries:
            for i in range(len(turns) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                if sim < threshold:
                    fine_boundaries.append(i + 1)
            if not fine_boundaries:
                threshold += 0.05
        
        if not fine_boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        adjusted_boundaries = []
        for fb in fine_boundaries:
            for cb in boundaries:
                if abs(fb - cb) <= 3:
                    adjusted_boundaries.append(fb)
                    break
        if not adjusted_boundaries:
            adjusted_boundaries = fine_boundaries

        boundaries = sorted(set(adjusted_boundaries))

        start_idx = 0
        for i, boundary in enumerate(boundaries):
            end_idx = 2 * boundary
            seg = self.buffer[start_idx:end_idx]
            segments.append(seg)
            start_idx = 2 * boundary

        if force_segment:
            segments.append(self.buffer[start_idx:])
            start_idx = len(boundaries)

        if start_idx > 0: 
            del self.buffer[:start_idx]
            self._recount_tokens()

        return segments

    def _cosine_similarity(self, vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
