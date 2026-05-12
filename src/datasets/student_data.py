from __future__ import annotations

from scienceqa import normalize_scienceqa_example
from vlm_step_by_step import (
    format_answer_target,
    format_reasoning_answer_target,
    format_scienceqa_prompt,
)


def build_student_rows(
    split_examples,
    dataset_config: dict,
    teacher_rows: list[dict] | None,
    split_name: str,
    mode: str,
    label_source: str,
    max_samples: int | None = None,
) -> list[dict]:
    _validate_args(mode, label_source)

    teacher_by_cache_id = {row.get("cache_id"): row for row in teacher_rows or []}
    row_count = len(split_examples) if max_samples is None else min(max_samples, len(split_examples))

    rows = []
    for local_index in range(row_count):
        cache_id = f"{split_name}_{local_index}"
        teacher_row = teacher_by_cache_id.get(cache_id)

        if _needs_teacher(mode, label_source) and teacher_row is None:
            continue

        example = normalize_scienceqa_example(split_examples[local_index], dataset_config)
        prompt = format_scienceqa_prompt(example, dataset_config, mode=mode)
        target = _build_target(example, teacher_row, mode, label_source)
        if target is None:
            continue

        rows.append(
            {
                "cache_id": cache_id,
                "split": split_name,
                "local_index": local_index,
                "example_id": example["id"],
                "image": example["image"],
                "question": example["question"],
                "choices": example["choices"],
                "gold_answer": example["answer_letter"],
                "teacher_answer": teacher_row.get("teacher_answer") if _needs_teacher(mode, label_source) else None,
                "prompt_mode": mode,
                "label_source": label_source,
                "prompt": prompt,
                "target": target,
            }
        )

    return rows


def _validate_args(mode: str, label_source: str) -> None:
    if mode not in ["answer_only", "reasoning_answer"]:
        raise ValueError(f"Unsupported mode: {mode}")
    if label_source not in ["gold", "teacher"]:
        raise ValueError(f"Unsupported label_source: {label_source}")
    if mode == "reasoning_answer" and label_source != "teacher":
        raise ValueError("reasoning_answer mode requires label_source='teacher'")


def _needs_teacher(mode: str, label_source: str) -> bool:
    return mode == "reasoning_answer" or label_source == "teacher"


def _build_target(example: dict, teacher_row: dict | None, mode: str, label_source: str) -> str | None:
    if mode == "answer_only":
        if label_source == "gold":
            return format_answer_target(example["answer_letter"])
        teacher_answer = teacher_row.get("teacher_answer") if teacher_row else None
        return format_answer_target(teacher_answer) if teacher_answer else None

    teacher_reasoning = teacher_row.get("teacher_reasoning") if teacher_row else None
    teacher_answer = teacher_row.get("teacher_answer") if teacher_row else None
    if not teacher_reasoning or not teacher_answer:
        return None
    return format_reasoning_answer_target(teacher_reasoning, teacher_answer)
