from __future__ import annotations

from scienceqa import normalize_scienceqa_example
from vlm_step_by_step import (
    format_answer_target,
    format_rationale_target,
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
    filter_rationale_by_gold_answer: bool = False,
) -> list[dict]:
    _validate_args(mode, label_source)

    teacher_by_cache_id = {row.get("cache_id"): row for row in teacher_rows or []}
    if mode in ["multitask", "multitask_label"]:
        return _build_multitask_rows(
            split_examples=split_examples,
            dataset_config=dataset_config,
            teacher_by_cache_id=teacher_by_cache_id,
            split_name=split_name,
            mode=mode,
            label_source=label_source,
            max_samples=max_samples,
            filter_rationale_by_gold_answer=filter_rationale_by_gold_answer,
        )

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
                "source_index": example["source_index"],
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
    if mode not in ["answer_only", "reasoning_answer", "multitask", "multitask_label"]:
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


def _build_multitask_rows(
    split_examples,
    dataset_config: dict,
    teacher_by_cache_id: dict,
    split_name: str,
    mode: str,
    label_source: str,
    max_samples: int | None,
    filter_rationale_by_gold_answer: bool,
) -> list[dict]:
    rows = []
    for local_index in range(len(split_examples)):
        base_cache_id = f"{split_name}_{local_index}"
        teacher_row = teacher_by_cache_id.get(base_cache_id)
        if teacher_row is None and (mode == "multitask" or label_source == "teacher") and not filter_rationale_by_gold_answer:
            continue

        example = normalize_scienceqa_example(split_examples[local_index], dataset_config)
        rationale_skip_reason = _rationale_skip_reason(example, teacher_row) if filter_rationale_by_gold_answer else None
        label_row = _build_multitask_label_row(
            example=example,
            dataset_config=dataset_config,
            teacher_row=teacher_row,
            split_name=split_name,
            local_index=local_index,
            label_source=label_source,
            rationale_skip_reason=rationale_skip_reason,
        )
        if label_row is not None:
            rows.append(label_row)
            if _enough_rows(rows, max_samples):
                return rows

        if mode == "multitask" and rationale_skip_reason is None:
            rationale_row = _build_multitask_rationale_row(example, dataset_config, teacher_row, split_name, local_index)
            if rationale_row is not None:
                rows.append(rationale_row)
                if _enough_rows(rows, max_samples):
                    return rows

    return rows


def _build_multitask_label_row(
    example: dict,
    dataset_config: dict,
    teacher_row: dict | None,
    split_name: str,
    local_index: int,
    label_source: str,
    rationale_skip_reason: str | None,
) -> dict | None:
    teacher_answer = teacher_row.get("teacher_answer") if teacher_row else None
    answer_letter = example["answer_letter"] if label_source == "gold" else teacher_answer
    if not answer_letter:
        return None

    base_cache_id = f"{split_name}_{local_index}"
    return _student_row(
        cache_id=f"{base_cache_id}_label",
        base_cache_id=base_cache_id,
        task="label",
        split_name=split_name,
        local_index=local_index,
        example=example,
        teacher_answer=teacher_answer,
        prompt_mode="multitask_label",
        label_source=label_source,
        prompt=format_scienceqa_prompt(example, dataset_config, mode="multitask_label"),
        target=format_answer_target(answer_letter),
        rationale_skip_reason=rationale_skip_reason,
    )


def _build_multitask_rationale_row(
    example: dict,
    dataset_config: dict,
    teacher_row: dict,
    split_name: str,
    local_index: int,
) -> dict | None:
    teacher_reasoning = teacher_row.get("teacher_reasoning")
    teacher_answer = teacher_row.get("teacher_answer")
    if not teacher_reasoning:
        return None

    base_cache_id = f"{split_name}_{local_index}"
    return _student_row(
        cache_id=f"{base_cache_id}_rationale",
        base_cache_id=base_cache_id,
        task="rationale",
        split_name=split_name,
        local_index=local_index,
        example=example,
        teacher_answer=teacher_answer,
        prompt_mode="multitask_rationale",
        label_source="teacher",
        prompt=format_scienceqa_prompt(example, dataset_config, mode="multitask_rationale"),
        target=format_rationale_target(teacher_reasoning),
        rationale_skip_reason=None,
    )


def _student_row(
    cache_id: str,
    base_cache_id: str,
    task: str,
    split_name: str,
    local_index: int,
    example: dict,
    teacher_answer: str | None,
    prompt_mode: str,
    label_source: str,
    prompt: str,
    target: str,
    rationale_skip_reason: str | None,
) -> dict:
    return {
        "cache_id": cache_id,
        "base_cache_id": base_cache_id,
        "task": task,
        "split": split_name,
        "local_index": local_index,
        "example_id": example["id"],
        "source_index": example["source_index"],
        "image": example["image"],
        "question": example["question"],
        "choices": example["choices"],
        "gold_answer": example["answer_letter"],
        "teacher_answer": teacher_answer,
        "prompt_mode": prompt_mode,
        "label_source": label_source,
        "prompt": prompt,
        "target": target,
        "rationale_used": task == "rationale" or (task == "label" and rationale_skip_reason is None),
        "rationale_skip_reason": rationale_skip_reason,
    }


def _enough_rows(rows: list[dict], max_samples: int | None) -> bool:
    return max_samples is not None and len(rows) >= max_samples


def _rationale_skip_reason(example: dict, teacher_row: dict | None) -> str | None:
    if teacher_row is None:
        return "missing_teacher_row"

    teacher_reasoning = teacher_row.get("teacher_reasoning")
    if not teacher_reasoning:
        return "missing_teacher_reasoning"

    teacher_answer = teacher_row.get("teacher_answer")
    if teacher_answer != example["answer_letter"]:
        return "teacher_answer_mismatch"

    return None
