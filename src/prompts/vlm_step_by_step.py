from __future__ import annotations


def format_scienceqa_prompt(
    example: dict,
    dataset_config: dict,
    mode: str = "reasoning_answer",
    task_prefix: str | None = None,
) -> str:
    question = _get_value(example, "question", dataset_config.get("question_column", "question"))
    choices = _get_value(example, "choices", dataset_config.get("choices_column", "choices")) or []
    choices_text = _format_choices(choices)

    if mode == "multitask_label":
        task_prefix = task_prefix or "[label]"
        prompt_parts = _base_prompt_parts(task_prefix, "Answer the science multiple-choice question using the image.", question, choices_text)
    elif mode == "multitask_rationale":
        task_prefix = task_prefix or "[rationale]"
        prompt_parts = _base_prompt_parts(
            task_prefix,
            "Explain the reasoning needed to answer the science multiple-choice question using the image.",
            question,
            choices_text,
        )
    else:
        prompt_parts = _base_prompt_parts(None, "Answer the science multiple-choice question using the image.", question, choices_text)

    if mode == "answer_only":
        prompt_parts.extend(
            [
                "Give the final answer as a single letter.",
                "Use this format:",
                "Answer: <letter>",
            ]
        )
    elif mode == "multitask_label":
        prompt_parts.extend(
            [
                "Give the final answer as a single letter.",
                "Use this format:",
                "Answer: <letter>",
            ]
        )
    elif mode == "multitask_rationale":
        prompt_parts.extend(
            [
                "Use this format:",
                "Reasoning: ...",
            ]
        )
    elif mode == "reasoning_answer":
        prompt_parts.extend(
            [
                "Explain briefly and then give the final answer as a single letter.",
                "Use this format:",
                "Reasoning: ...",
                "Answer: <letter>",
            ]
        )
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    return "\n".join(prompt_parts)


def format_answer_target(answer_letter: str) -> str:
    return f"Answer: {answer_letter}"


def format_reasoning_answer_target(reasoning: str, answer_letter: str) -> str:
    return f"Reasoning: {reasoning}\nAnswer: {answer_letter}"


def format_rationale_target(reasoning: str) -> str:
    return f"Reasoning: {reasoning}"


def answer_index_to_letter(answer_index: int) -> str:
    return chr(65 + int(answer_index))


def _base_prompt_parts(task_prefix: str | None, instruction: str, question: str, choices_text: str) -> list[str]:
    parts = []
    if task_prefix:
        parts.extend([task_prefix, instruction])
    else:
        parts.append(instruction)
    parts.extend(
        [
            "",
            f"Question: {question}",
            "",
            "Choices:",
            choices_text,
            "",
        ]
    )
    return parts


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))


def _get_value(example: dict, normalized_key: str, raw_key: str):
    if normalized_key in example:
        return example[normalized_key]
    return example.get(raw_key)
