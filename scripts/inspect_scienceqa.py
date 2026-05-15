from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/prompts", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from io_utils import load_yaml, resolve_project_path  # noqa: E402
from scienceqa import load_scienceqa_image_examples, normalize_scienceqa_example  # noqa: E402
from seed import set_seed  # noqa: E402
from vlm_step_by_step import (  # noqa: E402
    format_answer_target,
    format_reasoning_answer_target,
    format_scienceqa_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ScienceQA image examples for the VLM skeleton.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))

    set_seed(experiment_config["seed"])
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)

    print(f"experiment: {experiment_config['experiment_name']}")
    print(f"shuffle_train: {bool(experiment_config.get('shuffle_train', False))}")
    print(f"shuffle_eval: {bool(experiment_config.get('shuffle_eval', False))}")
    print(f"shuffle_seed: {experiment_config.get('shuffle_seed', experiment_config.get('seed'))}")
    print(f"train image examples: {len(splits['train'])}")
    print(f"validation image examples: {len(splits['validation'])}")
    _print_split_preview("train", splits["train"], dataset_config)
    _print_split_preview("validation", splits["validation"], dataset_config)

    preview_count = min(experiment_config.get("prompt_preview_count", 1), len(splits["validation"]))
    for index in range(preview_count):
        raw_example = splits["validation"][index]
        example = normalize_scienceqa_example(raw_example, dataset_config)
        display_id = example["id"] if example["id"] is not None else index

        print()
        print(f"--- validation example {index} ---")
        print(f"id: {display_id}")
        print(f"has image: {example['image'] is not None}")
        print(f"question: {example['question']}")
        print("choices:")
        for choice_index, choice in enumerate(example["choices"]):
            print(f"  {chr(65 + choice_index)}. {choice}")
        print(f"gold answer index: {example['answer']}")
        print(f"gold answer letter: {example['answer_letter']}")

        print()
        print("answer_only prompt:")
        print(format_scienceqa_prompt(example, dataset_config, mode="answer_only"))

        print()
        print("reasoning_answer prompt:")
        print(format_scienceqa_prompt(example, dataset_config, mode="reasoning_answer"))

        print()
        print("answer_only target:")
        print(format_answer_target(example["answer_letter"]))

        reasoning = _reasoning_text(example)
        if reasoning:
            print()
            print("reasoning_answer target:")
            print(format_reasoning_answer_target(reasoning, example["answer_letter"]))


def _reasoning_text(example: dict) -> str | None:
    return example.get("solution") or example.get("lecture") or example.get("hint")


def _print_split_preview(split_name: str, split, dataset_config: dict) -> None:
    print()
    print(f"{split_name} first questions:")
    for index in range(min(3, len(split))):
        example = normalize_scienceqa_example(split[index], dataset_config)
        print(f"  {split_name}_{index}: {example['answer_letter']} | {example['question']}")

    distribution = _gold_distribution(split, dataset_config)
    print(f"{split_name} gold distribution:")
    for letter in ["A", "B", "C", "D", "E"]:
        if distribution.get(letter, 0):
            print(f"  {letter}: {distribution[letter]}")


def _gold_distribution(split, dataset_config: dict) -> Counter:
    distribution = Counter()
    for index in range(len(split)):
        example = normalize_scienceqa_example(split[index], dataset_config)
        if example["answer_letter"]:
            distribution[example["answer_letter"]] += 1
    return distribution


if __name__ == "__main__":
    main()
