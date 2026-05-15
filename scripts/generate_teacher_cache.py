from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from io_utils import load_yaml, resolve_project_path, write_jsonl  # noqa: E402
from mock_teacher import generate_mock_teacher_output  # noqa: E402
from qwen_vl_teacher import (  # noqa: E402
    format_qwen_teacher_prompt,
    format_qwen_teacher_retry_prompt,
    generate_qwen_teacher_output,
    load_qwen_vl_teacher,
)
from scienceqa import load_scienceqa_image_examples, normalize_scienceqa_example  # noqa: E402
from seed import set_seed  # noqa: E402
from vlm_step_by_step import format_scienceqa_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a ScienceQA teacher cache.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_gold_answer", action="store_true")
    parser.add_argument("--teacher_type", type=str, choices=["mock", "qwen"], default="mock")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--teacher_device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--teacher_dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--teacher_max_new_tokens", type=int, default=256)
    parser.add_argument("--retry_on_parse_failure", action="store_true")
    parser.add_argument("--preview_count", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_count < 0:
        raise ValueError("--preview_count must be >= 0")

    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))

    set_seed(experiment_config["seed"])
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    split = splits[args.split]

    max_samples = args.max_samples if args.max_samples is not None else len(split)
    max_samples = min(max_samples, len(split))

    model = None
    processor = None
    if args.teacher_type == "qwen" and not args.dry_run:
        model, processor = load_qwen_vl_teacher(
            model_name=args.teacher_model_name,
            device=args.teacher_device,
            dtype=args.teacher_dtype,
        )

    rows = []
    for local_index in range(max_samples):
        example = normalize_scienceqa_example(split[local_index], dataset_config)
        prompt = _teacher_prompt(example, dataset_config, args.teacher_type)

        if args.dry_run:
            print(f"dry_run: true")
            print(f"teacher_type: {args.teacher_type}")
            print(f"split: {args.split}")
            print(f"max_samples: {max_samples}")
            print("first teacher prompt:")
            print(prompt)
            return

        teacher_output = _generate_teacher_output(
            args=args,
            example=example,
            prompt=prompt,
            model=model,
            processor=processor,
        )

        rows.append(
            {
                "cache_id": f"{args.split}_{local_index}",
                "split": args.split,
                "local_index": local_index,
                "example_id": example["id"],
                "question": example["question"],
                "choices": example["choices"],
                "gold_answer": example["answer_letter"],
                "prompt_mode": "reasoning_answer",
                "prompt": prompt,
                "teacher_type": args.teacher_type,
                "teacher_model_name": _teacher_model_name(args),
                "teacher_reasoning": teacher_output["teacher_reasoning"],
                "teacher_answer": teacher_output["teacher_answer"],
                "teacher_raw_output": teacher_output["teacher_raw_output"],
                "teacher_retry_used": teacher_output.get("teacher_retry_used", False),
                "teacher_first_raw_output": teacher_output.get("teacher_first_raw_output"),
            }
        )

    output_path = resolve_project_path(ROOT, args.output_path)
    write_jsonl(rows, output_path)

    print(f"output path: {output_path}")
    print(f"saved examples: {len(rows)}")
    print(f"parse failures: {_count_parse_failures(rows)}")
    _print_row_previews(rows, args.preview_count)


def _teacher_prompt(example: dict, dataset_config: dict, teacher_type: str) -> str:
    if teacher_type == "qwen":
        return format_qwen_teacher_prompt(example)
    return format_scienceqa_prompt(example, dataset_config, mode="reasoning_answer")


def _generate_teacher_output(
    args: argparse.Namespace,
    example: dict,
    prompt: str,
    model,
    processor,
) -> dict:
    if args.teacher_type == "qwen":
        teacher_output = generate_qwen_teacher_output(
            model=model,
            processor=processor,
            image=example["image"],
            prompt=prompt,
            max_new_tokens=args.teacher_max_new_tokens,
        )
        teacher_output["teacher_retry_used"] = False

        if args.retry_on_parse_failure and teacher_output.get("teacher_answer") is None:
            retry_prompt = format_qwen_teacher_retry_prompt(example)
            retry_output = generate_qwen_teacher_output(
                model=model,
                processor=processor,
                image=example["image"],
                prompt=retry_prompt,
                max_new_tokens=args.teacher_max_new_tokens,
            )
            retry_output["teacher_retry_used"] = True
            retry_output["teacher_first_raw_output"] = teacher_output["teacher_raw_output"]
            return retry_output

        return teacher_output
    return generate_mock_teacher_output(example, use_gold_answer=args.use_gold_answer)


def _teacher_model_name(args: argparse.Namespace) -> str:
    if args.teacher_type == "qwen":
        return args.teacher_model_name
    return "mock"


def _count_parse_failures(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("teacher_answer") is None)


def _print_row_previews(rows: list[dict], preview_count: int) -> None:
    preview_count = min(preview_count, len(rows))
    for index in range(preview_count):
        print(f"--- preview row {index} ---")
        print(json.dumps(rows[index], ensure_ascii=False, indent=2))

    if len(rows) > preview_count:
        print(f"... skipped preview for remaining {len(rows) - preview_count} rows ...")


if __name__ == "__main__":
    main()
