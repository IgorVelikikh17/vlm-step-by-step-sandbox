from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for relative in ["src/datasets", "src/evaluation", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small labeled reducing-data comparison.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--rationale_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_root", type=str, default="outputs/labeled_reducing_data")
    parser.add_argument("--results_dir", type=str, default="results/labeled_reducing_data")
    parser.add_argument("--multitask_method_name", type=str, default="multitask_gold")
    parser.add_argument("--filter_rationale_by_gold_answer", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    teacher_cache_path = resolve_project_path(ROOT, args.teacher_cache_path)
    if not teacher_cache_path.exists():
        _raise_missing_teacher_cache(args, teacher_cache_path, max(args.train_sizes))

    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    _check_split_sizes(args, splits)

    teacher_rows = read_teacher_cache(teacher_cache_path)
    _check_teacher_cache_coverage(args, teacher_rows)

    runs = _build_runs(args)
    if args.dry_run:
        _print_dry_run(runs)
        return

    results = []
    for run in runs:
        print()
        print(f"=== train train_size={run['train_size']} method={run['method']['name']} ===", flush=True)
        _run_command(run["train_command"])

        print()
        print(f"=== evaluate train_size={run['train_size']} method={run['method']['name']} ===", flush=True)
        _run_command(run["eval_command"])

        metrics = _read_json(run["eval_dir"] / "metrics.json")
        results.append(_result_row(args, run, metrics))

    results_dir = resolve_project_path(ROOT, args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(results, results_dir / "results.csv")
    _write_json(results, results_dir / "results.json")
    print()
    print(f"saved results csv: {results_dir / 'results.csv'}")
    print(f"saved results json: {results_dir / 'results.json'}")


def _check_split_sizes(args: argparse.Namespace, splits) -> None:
    max_train_size = max(args.train_sizes)
    num_train_examples = len(splits["train"])
    if max_train_size > num_train_examples:
        raise ValueError(
            f"Requested max train_size={max_train_size}, but only {num_train_examples} train examples are available.\n"
            f"Increase max_train_samples in {args.config} or use another config.\n"
            "Then regenerate teacher cache with enough examples."
        )

    num_eval_examples = len(splits["validation"])
    if args.max_eval_samples > num_eval_examples:
        raise ValueError(
            f"Requested max_eval_samples={args.max_eval_samples}, but only {num_eval_examples} "
            "validation examples are available.\n"
            f"Increase max_eval_samples in {args.config} or use a larger config."
        )


def _check_teacher_cache_coverage(args: argparse.Namespace, teacher_rows: list[dict]) -> None:
    max_train_size = max(args.train_sizes)
    cache_ids = {row.get("cache_id") for row in teacher_rows}
    required_ids = [f"train_{index}" for index in range(max_train_size)]
    missing_ids = [cache_id for cache_id in required_ids if cache_id not in cache_ids]
    if not missing_ids:
        return

    preview = ", ".join(missing_ids[:10])
    if len(missing_ids) > 10:
        preview += ", ..."
    command = (
        "python scripts/generate_teacher_cache.py "
        f"--config {args.config} "
        "--split train "
        f"--output_path {args.teacher_cache_path} "
        f"--max_samples {max_train_size} "
        "--use_gold_answer"
    )
    raise ValueError(
        f"Teacher cache does not cover requested train_size={max_train_size}.\n"
        f"Missing cache ids include: {preview}\n"
        "Regenerate teacher cache with:\n"
        f"{command}"
    )


def _build_runs(args: argparse.Namespace) -> list[dict]:
    methods = [
        {
            "name": "answer_only_gold",
            "train_mode": "answer_only",
            "eval_mode": "answer_only",
            "rationale_loss_weight": "",
        },
        {
            "name": args.multitask_method_name,
            "train_mode": "multitask",
            "eval_mode": "multitask_label",
            "rationale_loss_weight": args.rationale_loss_weight,
            "filter_rationale_by_gold_answer": args.filter_rationale_by_gold_answer,
        },
    ]

    runs = []
    for train_size in args.train_sizes:
        max_steps = train_size * args.num_epochs
        for method in methods:
            checkpoint_dir = resolve_project_path(ROOT, args.output_root) / f"train_size_{train_size}" / method["name"]
            eval_dir = resolve_project_path(ROOT, args.results_dir) / f"train_size_{train_size}" / method["name"]
            runs.append(
                {
                    "train_size": train_size,
                    "max_steps": max_steps,
                    "method": method,
                    "checkpoint_dir": checkpoint_dir,
                    "eval_dir": eval_dir,
                    "train_command": _train_command(args, method, train_size, max_steps, checkpoint_dir),
                    "eval_command": _eval_command(args, method, checkpoint_dir, eval_dir),
                }
            )
    return runs


def _train_command(
    args: argparse.Namespace,
    method: dict,
    train_size: int,
    max_steps: int,
    checkpoint_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/dev/train_student.py",
        "--config",
        args.config,
        "--model_config",
        args.model_config,
        "--teacher_cache_path",
        args.teacher_cache_path,
        "--split",
        "train",
        "--mode",
        method["train_mode"],
        "--label_source",
        "gold",
        "--train_size",
        str(train_size),
        "--max_steps",
        str(max_steps),
        "--batch_size",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--output_dir",
        str(checkpoint_dir.relative_to(ROOT)),
    ]
    if args.model_name:
        command.extend(["--model_name", args.model_name])
    if method["train_mode"] == "multitask":
        command.extend(["--rationale_loss_weight", str(args.rationale_loss_weight)])
    if method.get("filter_rationale_by_gold_answer"):
        command.append("--filter_rationale_by_gold_answer")
    return command


def _eval_command(args: argparse.Namespace, method: dict, checkpoint_dir: Path, eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_student.py",
        "--config",
        args.config,
        "--model_config",
        args.model_config,
        "--model_name",
        str(checkpoint_dir.relative_to(ROOT)),
        "--teacher_cache_path",
        args.teacher_cache_path,
        "--split",
        "validation",
        "--mode",
        method["eval_mode"],
        "--label_source",
        "gold",
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--output_dir",
        str(eval_dir.relative_to(ROOT)),
    ]


def _run_command(command: list[str]) -> None:
    print(_format_command(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _print_dry_run(runs: list[dict]) -> None:
    print("dry_run: true")
    for run in runs:
        print()
        print(f"train_size: {run['train_size']}")
        print(f"num_epochs: {run['max_steps'] // run['train_size']}")
        print(f"max_steps: {run['max_steps']}")
        print(f"method: {run['method']['name']}")
        print("train command:")
        print(_format_command(run["train_command"]))
        print("evaluation command:")
        print(_format_command(run["eval_command"]))


def _format_command(command: list[str]) -> str:
    display = ["python" if item == sys.executable else item for item in command]
    return " ".join(display)


def _result_row(args: argparse.Namespace, run: dict, metrics: dict) -> dict:
    method = run["method"]
    return {
        "train_size": run["train_size"],
        "num_epochs": args.num_epochs,
        "max_steps": run["max_steps"],
        "method": method["name"],
        "learning_rate": args.learning_rate,
        "rationale_loss_weight": method["rationale_loss_weight"],
        "filter_rationale_by_gold_answer": method.get("filter_rationale_by_gold_answer", False),
        "accuracy": metrics["accuracy"],
        "parse_failure_rate": metrics["parse_failure_rate"],
        "num_examples": metrics["num_examples"],
        "num_correct": metrics["num_correct"],
        "checkpoint_dir": str(run["checkpoint_dir"].relative_to(ROOT)),
        "eval_dir": str(run["eval_dir"].relative_to(ROOT)),
    }


def _write_results_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "train_size",
        "num_epochs",
        "max_steps",
        "method",
        "learning_rate",
        "rationale_loss_weight",
        "filter_rationale_by_gold_answer",
        "accuracy",
        "parse_failure_rate",
        "num_examples",
        "num_correct",
        "checkpoint_dir",
        "eval_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be >= 1")
    if not args.train_sizes:
        raise ValueError("--train_sizes must contain at least one value")
    if any(train_size < 1 for train_size in args.train_sizes):
        raise ValueError("--train_sizes values must be >= 1")


def _raise_missing_teacher_cache(args: argparse.Namespace, path: Path, max_train_size: int) -> None:
    output_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if "qwen" in str(path).lower():
        command = (
            "python scripts/generate_teacher_cache.py "
            f"--config {args.config} "
            "--split train "
            f"--output_path {output_path} "
            f"--max_samples {max_train_size} "
            "--teacher_type qwen "
            "--teacher_model_name Qwen/Qwen2.5-VL-3B-Instruct "
            "--teacher_device cuda "
            "--teacher_dtype bfloat16 "
            "--teacher_max_new_tokens 256 "
            "--retry_on_parse_failure "
            "--preview_count 2"
        )
    else:
        command = (
            "python scripts/generate_teacher_cache.py "
            f"--config {args.config} "
            "--split train "
            f"--output_path {output_path} "
            f"--max_samples {max_train_size} "
            "--use_gold_answer"
        )
    raise FileNotFoundError(
        f"Teacher cache does not exist: {path}\n"
        "Generate it first with:\n"
        f"{command}"
    )


if __name__ == "__main__":
    main()
