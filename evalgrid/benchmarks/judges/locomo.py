"""
evalgrid/benchmarks/judges/locomo.py
----------------------------
LoCoMo judge: a single generous CORRECT/WRONG prompt applied uniformly across
categories (criteria do not change per category), with explicit temporal
leniency. Prompt is verbatim from zjunlp/LightMem/experiments/locomo/llm_judge.py
(the Mem0-popularised LoCoMo evaluation).
"""

from __future__ import annotations

from evalgrid.benchmarks.types import Query
from evalgrid.benchmarks.judges.base import Judge, extract_json, gold_to_str

# LoCoMo category ids -> readable names (category 5 is excluded upstream).
_CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}

ACCURACY_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


class LoCoMoJudge(Judge):

    def build_messages(self, query: Query, hypothesis: str) -> list[dict]:
        prompt = ACCURACY_PROMPT.format(
            question=query.question,
            gold_answer=gold_to_str(query.answer),
            generated_answer=hypothesis,
        )
        return [{"role": "user", "content": prompt}]

    def parse_label(self, response: str) -> bool:
        data = extract_json(response)
        if data and "label" in data:
            return str(data["label"]).strip().upper() == "CORRECT"
        # Fallback for non-JSON replies: trust an unambiguous CORRECT/WRONG.
        up = response.upper()
        return ("CORRECT" in up) and ("WRONG" not in up)

    def category_of(self, query: Query) -> str:
        cat = str(query.question_type)
        return _CATEGORY_NAMES.get(cat, cat)
