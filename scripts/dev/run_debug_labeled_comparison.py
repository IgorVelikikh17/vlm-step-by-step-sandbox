from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small answer-only vs multitask labeled comparison.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--rationale_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_root", type=str, default="outputs/debug_labeled_comparison")
    parser.add_argument("--results_dir", type=str, default="results/debug_labeled_comparison")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_cache_path = _resolve_project_path(args.teacher_cache_path)
    if not teacher_cache_path.exists():
        _raise_missing_teacher_cache(teacher_cache_path)

    methods = [
        {
            "name": "answer_only_gold",
            "train_mode": "answer_only",
            "eval_mode": "answer_only",
            "rationale_loss_weight": "",
        },
        {
            "name": "multitask_gold",
            "train_mode": "multitask",
            "eval_mode": "multitask_label",
            "rationale_loss_weight": args.rationale_loss_weight,
        },
    ]

    runs = []
    for method in methods:
        checkpoint_dir = _resolve_project_path(args.output_root) / method["name"]
        eval_dir = _resolve_project_path(args.results_dir) / method["name"]
        train_command = _train_command(args, method, checkpoint_dir)
        eval_command = _eval_command(args, method, checkpoint_dir, eval_dir)
        runs.append(
            {
                "method": method,
                "checkpoint_dir": checkpoint_dir,
                "eval_dir": eval_dir,
                "train_command": train_command,
                "eval_command": eval_command,
            }
        )

    if args.dry_run:
        _print_dry_run(runs)
        return

    results = []
    for run in runs:
        method_name = run["method"]["name"]
        print()
        print(f"=== train {method_name} ===", flush=True)
        _run_command(run["train_command"])

        print()
        print(f"=== evaluate {method_name} ===", flush=True)
        _run_command(run["eval_command"])

        metrics = _read_json(run["eval_dir"] / "metrics.json")
        results.append(_result_row(args, run, metrics))

    results_dir = _resolve_project_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(results, results_dir / "results.csv")
    _write_json(results, results_dir / "results.json")
    print()
    print(f"saved comparison csv: {results_dir / 'results.csv'}")
    print(f"saved comparison json: {results_dir / 'results.json'}")


def _train_command(args: argparse.Namespace, method: dict, checkpoint_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_student.py",
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
        str(args.train_size),
        "--max_steps",
        str(args.max_steps),
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
        "method": method["name"],
        "train_size": args.train_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "rationale_loss_weight": method["rationale_loss_weight"],
        "accuracy": metrics["accuracy"],
        "parse_failure_rate": metrics["parse_failure_rate"],
        "num_examples": metrics["num_examples"],
        "num_correct": metrics["num_correct"],
        "checkpoint_dir": str(run["checkpoint_dir"].relative_to(ROOT)),
        "eval_dir": str(run["eval_dir"].relative_to(ROOT)),
    }


def _write_results_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "method",
        "train_size",
        "max_steps",
        "learning_rate",
        "rationale_loss_weight",
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


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _raise_missing_teacher_cache(path: Path) -> None:
    command = (
        "python scripts/generate_teacher_cache.py "
        "--config src/configs/experiment/debug.yaml "
        "--split train "
        "--output_path data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl "
        "--max_samples 8 "
        "--use_gold_answer"
    )
    raise FileNotFoundError(
        f"Teacher cache does not exist: {path}\n"
        "Generate it first with:\n"
        f"{command}"
    )


if __name__ == "__main__":
    main()
