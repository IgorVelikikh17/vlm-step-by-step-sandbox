from __future__ import annotations


def format_scienceqa_prompt(example: dict, dataset_config: dict, mode: str = "reasoning_answer") -> str:
    question = _get_value(example, "question", dataset_config.get("question_column", "question"))
    choices = _get_value(example, "choices", dataset_config.get("choices_column", "choices")) or []
    choices_text = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))

    prompt_parts = [
        "Answer the science multiple-choice question using the image.",
        "",
        f"Question: {question}",
        "",
        "Choices:",
        choices_text,
        "",
    ]

    if mode == "answer_only":
        prompt_parts.extend(
            [
                "Give the final answer as a single letter.",
                "Use this format:",
                "Answer: <letter>",
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


def answer_index_to_letter(answer_index: int) -> str:
    return chr(65 + int(answer_index))


def _get_value(example: dict, normalized_key: str, raw_key: str):
    if normalized_key in example:
        return example[normalized_key]
    return example.get(raw_key)
