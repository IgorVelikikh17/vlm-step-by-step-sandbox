from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/model", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path, write_jsonl  # noqa: E402
from parsing import parse_answer_letter  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402
from seed import set_seed  # noqa: E402
from smolvlm_student import generate_smolvlm_answer, load_smolvlm  # noqa: E402
from student_data import build_student_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SmolVLM student on a small ScienceQA split.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="validation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["answer_only", "reasoning_answer", "multitask_label"],
        default="answer_only",
    )
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument("--max_eval_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--preview_count", type=int, default=3)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_dir", type=str, default="results/eval_smoke")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_count < 0:
        raise ValueError("--preview_count must be >= 0")

    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    model_config = load_yaml(ROOT / args.model_config)
    teacher_cache_path = resolve_project_path(ROOT, args.teacher_cache_path)
    output_dir = resolve_project_path(ROOT, args.output_dir)

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
        max_samples=args.max_eval_samples,
    )

    model_name_or_path = resolve_model_name_or_path(
        ROOT,
        args.model_name or model_config.get("pretrained_name", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    )
    print(f"config path: {ROOT / args.config}")
    print(f"model config path: {ROOT / args.model_config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"model name or path: {model_name_or_path}")
    print(f"split: {args.split}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"student rows for evaluation: {len(student_rows)}")
    print(f"preview_count: {args.preview_count}")
    print(f"output dir: {output_dir}")

    if args.dry_run:
        _print_dry_run(student_rows)
        return

    if not student_rows:
        raise RuntimeError("No student rows were built for evaluation.")

    if is_local_model_path(args.model_name) and not Path(model_name_or_path).exists():
        raise FileNotFoundError(f"Local model checkpoint does not exist: {model_name_or_path}")

    model, processor = load_smolvlm(
        model_name_or_path=model_name_or_path,
        device=args.device,
        dtype=args.dtype,
    )

    predictions = []
    for index, row in enumerate(student_rows):
        raw_output = generate_smolvlm_answer(
            model=model,
            processor=processor,
            image=row["image"],
            prompt=row["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        pred_answer = parse_answer_letter(raw_output)
        is_correct = pred_answer == row["gold_answer"]

        prediction = _prediction_row(row, raw_output, pred_answer, is_correct)
        predictions.append(prediction)

        if index < args.preview_count:
            print()
            print(f"--- evaluation row {index} ---")
            print(f"cache_id: {row['cache_id']}")
            print(f"gold_answer: {row['gold_answer']}")
            print(f"teacher_answer: {row.get('teacher_answer')}")
            print(f"pred_answer: {pred_answer}")
            print(f"is_correct: {is_correct}")
            print("raw output:")
            print(raw_output)

    if len(student_rows) > args.preview_count:
        skipped = len(student_rows) - args.preview_count
        print()
        print(f"... skipped detailed printing for remaining {skipped} examples ...")

    metrics = _compute_metrics(predictions, model_name_or_path, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, output_dir / "predictions.jsonl")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print()
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    print(f"saved predictions: {output_dir / 'predictions.jsonl'}")
    print("metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _print_dry_run(rows: list[dict]) -> None:
    print("dry_run: true")
    if not rows:
        print("no student rows built")
        return

    first = rows[0]
    print(f"image exists: {first['image'] is not None}")
    print(f"gold_answer: {first['gold_answer']}")
    print("first prompt:")
    print(first["prompt"])
    print("first target:")
    print(first["target"])


def _prediction_row(row: dict, raw_output: str, pred_answer: str | None, is_correct: bool) -> dict:
    return {
        "cache_id": row["cache_id"],
        "base_cache_id": row.get("base_cache_id"),
        "task": row.get("task"),
        "split": row["split"],
        "local_index": row["local_index"],
        "example_id": row["example_id"],
        "prompt_mode": row["prompt_mode"],
        "label_source": row["label_source"],
        "question": row["question"],
        "choices": row["choices"],
        "gold_answer": row["gold_answer"],
        "teacher_answer": row.get("teacher_answer"),
        "target": row["target"],
        "raw_output": raw_output,
        "pred_answer": pred_answer,
        "is_correct": is_correct,
    }


def _compute_metrics(predictions: list[dict], model_name_or_path: str, args: argparse.Namespace) -> dict:
    num_examples = len(predictions)
    num_correct = sum(1 for row in predictions if row["is_correct"])
    num_parse_failures = sum(1 for row in predictions if row["pred_answer"] is None)
    return {
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": num_correct / num_examples if num_examples else 0.0,
        "num_parse_failures": num_parse_failures,
        "parse_failure_rate": num_parse_failures / num_examples if num_examples else 0.0,
        "model_name": model_name_or_path,
        "split": args.split,
        "mode": args.mode,
        "label_source": args.label_source,
    }


def resolve_model_name_or_path(root: Path, model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path)
    if model_path.is_absolute():
        return str(model_path)

    if model_path.parts and model_path.parts[0] in [".", "..", "outputs", "checkpoints", "artifacts"]:
        return str(root / model_path)

    candidate = root / model_path
    if candidate.exists():
        return str(candidate)

    return model_name_or_path


def is_local_model_path(model_name_or_path: str | None) -> bool:
    if model_name_or_path is None:
        return False
    model_path = Path(model_name_or_path)
    if model_path.is_absolute():
        return True
    return bool(model_path.parts and model_path.parts[0] in [".", "..", "outputs", "checkpoints", "artifacts"])


if __name__ == "__main__":
    main()
