from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402
from seed import set_seed  # noqa: E402
from student_data import build_student_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect student training rows built from ScienceQA and teacher cache.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["answer_only", "reasoning_answer", "multitask", "multitask_label"],
        default="answer_only",
    )
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preview_count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    teacher_cache_path = resolve_project_path(ROOT, args.teacher_cache_path)

    set_seed(experiment_config["seed"])
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    teacher_rows = read_teacher_cache(teacher_cache_path)
    student_rows = build_student_rows(
        split_examples=splits[args.split],
        dataset_config=dataset_config,
        teacher_rows=teacher_rows,
        split_name=args.split,
        mode=args.mode,
        label_source=args.label_source,
        max_samples=args.max_samples,
    )

    print(f"config path: {ROOT / args.config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"split: {args.split}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"built student rows: {len(student_rows)}")

    preview_count = min(args.preview_count, len(student_rows))
    for index in range(preview_count):
        row = student_rows[index]
        print()
        print(f"--- student row {index} ---")
        print(f"cache_id: {row['cache_id']}")
        print(f"base_cache_id: {row.get('base_cache_id')}")
        print(f"source_index: {row.get('source_index')}")
        print(f"task: {row.get('task')}")
        print(f"prompt_mode: {row.get('prompt_mode')}")
        print(f"question: {row['question']}")
        print("choices:")
        for choice_index, choice in enumerate(row["choices"]):
            print(f"  {chr(65 + choice_index)}. {choice}")
        print(f"gold_answer: {row['gold_answer']}")
        print(f"teacher_answer: {row['teacher_answer']}")
        print("prompt:")
        print(row["prompt"])
        print("target:")
        print(row["target"])


if __name__ == "__main__":
    main()
