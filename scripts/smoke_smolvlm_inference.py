from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/model", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path  # noqa: E402
from parsing import parse_answer_letter  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402
from seed import set_seed  # noqa: E402
from smolvlm_student import generate_smolvlm_answer, load_smolvlm  # noqa: E402
from student_data import build_student_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SmolVLM inference smoke test.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["answer_only", "reasoning_answer", "multitask_label"],
        default="answer_only",
    )
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    model_config = load_yaml(ROOT / args.model_config)
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

    model_name = resolve_model_name_or_path(
        ROOT,
        args.model_name or model_config.get("pretrained_name", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    )
    print(f"config path: {ROOT / args.config}")
    print(f"model config path: {ROOT / args.model_config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"model name: {model_name}")
    print(f"split: {args.split}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"student rows for smoke test: {len(student_rows)}")

    if args.dry_run:
        _print_dry_run_rows(student_rows)
        return

    if is_local_model_path(args.model_name) and not Path(model_name).exists():
        raise FileNotFoundError(f"Local model checkpoint does not exist: {model_name}")

    model, processor = load_smolvlm(
        model_name=model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    for index, row in enumerate(student_rows):
        print()
        print(f"--- smolvlm generation row {index} ---")
        print(f"cache_id: {row['cache_id']}")
        print(f"question: {row['question']}")
        print(f"gold_answer: {row['gold_answer']}")
        print(f"teacher_answer: {row['teacher_answer']}")
        print("target:")
        print(row["target"])

        raw_output = generate_smolvlm_answer(
            model=model,
            processor=processor,
            image=row["image"],
            prompt=row["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        parsed_answer = parse_answer_letter(raw_output)
        print("raw model output:")
        print(raw_output)
        print(f"parsed model answer: {parsed_answer}")
        print(f"is_correct: {parsed_answer == row['gold_answer']}")


def _print_dry_run_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        print()
        print(f"--- dry run row {index} ---")
        print(f"cache_id: {row['cache_id']}")
        print(f"image exists: {row['image'] is not None}")
        print(f"question: {row['question']}")
        print(f"gold_answer: {row['gold_answer']}")
        print(f"teacher_answer: {row['teacher_answer']}")
        print("prompt:")
        print(row["prompt"])
        print("target:")
        print(row["target"])


def resolve_model_name_or_path(root: Path, model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path)

    if model_path.parts and model_path.parts[0] in [".", "..", "outputs", "checkpoints", "artifacts"]:
        return str(root / model_path)

    candidate = root / model_path
    if candidate.exists():
        return str(candidate)

    return model_name


def is_local_model_path(model_name: str | None) -> bool:
    if model_name is None:
        return False
    model_path = Path(model_name)
    if model_path.is_absolute():
        return True
    return bool(model_path.parts and model_path.parts[0] in [".", "..", "outputs", "checkpoints", "artifacts"])


if __name__ == "__main__":
    main()
