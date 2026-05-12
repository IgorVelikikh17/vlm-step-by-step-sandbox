from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from io_utils import load_yaml, resolve_project_path, write_jsonl  # noqa: E402
from mock_teacher import generate_mock_teacher_output  # noqa: E402
from scienceqa import load_scienceqa_image_examples, normalize_scienceqa_example  # noqa: E402
from seed import set_seed  # noqa: E402
from vlm_step_by_step import format_scienceqa_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mock ScienceQA teacher cache.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_gold_answer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))

    set_seed(experiment_config["seed"])
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    split = splits[args.split]

    max_samples = args.max_samples if args.max_samples is not None else len(split)
    max_samples = min(max_samples, len(split))

    rows = []
    for local_index in range(max_samples):
        example = normalize_scienceqa_example(split[local_index], dataset_config)
        prompt = format_scienceqa_prompt(example, dataset_config, mode="reasoning_answer")
        teacher_output = generate_mock_teacher_output(example, use_gold_answer=args.use_gold_answer)

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
                "teacher_reasoning": teacher_output["teacher_reasoning"],
                "teacher_answer": teacher_output["teacher_answer"],
                "teacher_raw_output": teacher_output["teacher_raw_output"],
            }
        )

    output_path = resolve_project_path(ROOT, args.output_path)
    write_jsonl(rows, output_path)

    print(f"output path: {output_path}")
    print(f"saved examples: {len(rows)}")
    if rows:
        print("first saved example preview:")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
