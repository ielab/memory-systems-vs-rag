"""
evalgrid/benchmarks/judges/longmemeval.py
---------------------------------
LongMemEval judge: the official per-question-type prompts (yes/no) from
LongMemEval's evaluate_qa.py. Different question types use different correctness
criteria by design (abstention, temporal off-by-one tolerance, knowledge-update
recency, preference rubric). Verbatim from xiaowu0162/LongMemEval.
"""

from __future__ import annotations

import re

from evalgrid.benchmarks.types import Query
from evalgrid.benchmarks.judges.base import Judge, gold_to_str


def get_anscheck_prompt(task: str, question: str, answer: str, response: str,
                        abstention: bool = False) -> str:
    """Official LongMemEval per-type judge prompt (verbatim)."""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unknown LongMemEval task: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


class LongMemEvalJudge(Judge):

    def _task_and_abstention(self, query: Query) -> tuple[str, bool]:
        # The adapter appends "/abstention" to question_type and uses a "_abs"
        # query_id suffix for abstention questions.
        qt = query.question_type or ""
        abstention = qt.endswith("/abstention") or query.query_id.endswith("_abs")
        task = qt.replace("/abstention", "")
        return task, abstention

    def build_messages(self, query: Query, hypothesis: str) -> list[dict]:
        task, abstention = self._task_and_abstention(query)
        prompt = get_anscheck_prompt(
            task, query.question, gold_to_str(query.answer), hypothesis,
            abstention=abstention,
        )
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    def parse_label(self, response: str) -> bool:
        # Official parsing is `'yes' in response.lower()`, which assumes a terse
        # "yes"/"no" verdict. That substring test misfires on a verbose/reasoning
        # judge: "yesterday" or a "yes" buried in <think> reasoning both count as
        # correct. Strip any <think>...</think> block (vLLM thinking models emit
        # it inline in the content) and match a standalone 'yes' token instead.
        verdict = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL | re.IGNORECASE)
        return re.search(r"\byes\b", verdict.lower()) is not None
