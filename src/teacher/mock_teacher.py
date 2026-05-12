from __future__ import annotations

from vlm_step_by_step import format_reasoning_answer_target


def generate_mock_teacher_output(example: dict, use_gold_answer: bool = True) -> dict:
    reasoning = _reasoning_from_example(example)
    if use_gold_answer:
        teacher_answer = example["answer_letter"]
    else:
        # Real teacher inference will generate the answer itself later.
        teacher_answer = "A"

    raw_output = format_reasoning_answer_target(reasoning, teacher_answer)
    return {
        "teacher_reasoning": reasoning,
        "teacher_answer": teacher_answer,
        "teacher_raw_output": raw_output,
    }


def _reasoning_from_example(example: dict) -> str:
    return (
        example.get("solution")
        or example.get("lecture")
        or example.get("hint")
        or "This is a mock reasoning for debugging."
    )
