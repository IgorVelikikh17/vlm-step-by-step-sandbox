from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/model", "src/prompts", "src/teacher", "src/training", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path, write_jsonl  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402
from seed import set_seed  # noqa: E402
from smolvlm_batching import build_smolvlm_training_batch  # noqa: E402
from smolvlm_student import load_smolvlm  # noqa: E402
from student_data import build_student_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SmolVLM student on prepared ScienceQA rows.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument("--mode", type=str, choices=["answer_only", "multitask"], default="answer_only")
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--rationale_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/train_debug_answer_only_gold")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("train_student.py currently supports batch_size=1 for simple task-level loss weighting.")

    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    model_config = load_yaml(ROOT / args.model_config)
    teacher_cache_path = resolve_project_path(ROOT, args.teacher_cache_path)
    output_dir = resolve_project_path(ROOT, args.output_dir)

    set_seed(experiment_config["seed"])
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    original_examples = _take_original_examples(splits[args.split], args.train_size)
    teacher_rows = read_teacher_cache(teacher_cache_path)
    train_rows = build_student_rows(
        split_examples=original_examples,
        dataset_config=dataset_config,
        teacher_rows=teacher_rows,
        split_name=args.split,
        mode=args.mode,
        label_source=args.label_source,
        max_samples=None,
    )

    model_name_or_path = resolve_model_name_or_path(
        ROOT,
        args.model_name or model_config.get("pretrained_name", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    )
    num_original_examples = len(original_examples)

    print(f"config path: {ROOT / args.config}")
    print(f"model config path: {ROOT / args.model_config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"model name or path: {model_name_or_path}")
    print(f"split: {args.split}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"train_size original examples: {num_original_examples}")
    print(f"student rows ready: {len(train_rows)}")
    print(f"max_steps: {args.max_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"rationale_loss_weight: {args.rationale_loss_weight}")
    print(f"output dir: {output_dir}")

    if args.dry_run:
        _print_dry_run(num_original_examples, train_rows)
        return

    if not train_rows:
        raise RuntimeError("No student rows were built for training.")

    if is_local_model_path(args.model_name) and not Path(model_name_or_path).exists():
        raise FileNotFoundError(f"Local model checkpoint does not exist: {model_name_or_path}")

    training_config = _training_config(args, model_name_or_path, num_original_examples, len(train_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(training_config, output_dir / "training_config.json")

    import torch

    model, processor = load_smolvlm(
        model_name_or_path=model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        eval_mode=False,
    )
    train_device = str(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    loss_rows = []
    saved_checkpoint = False
    stopped_reason = None

    for step_index in range(args.max_steps):
        row = train_rows[step_index % len(train_rows)]
        batch = build_smolvlm_training_batch([row], processor, device=train_device)
        outputs = model(**batch)
        raw_loss = outputs.loss
        loss_weight = _loss_weight(row, args)
        loss = raw_loss * loss_weight

        if not torch.isfinite(loss):
            stopped_reason = "non_finite_loss"
            loss_rows.append(_loss_row(step_index + 1, row, raw_loss, loss, loss_weight))
            print(f"stopping: non-finite loss at step {step_index + 1}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        loss_row = _loss_row(step_index + 1, row, raw_loss, loss, loss_weight)
        loss_rows.append(loss_row)
        print(
            f"step: {loss_row['step']} "
            f"loss: {loss_row['loss']:.6f} "
            f"raw_loss: {loss_row['raw_loss']:.6f} "
            f"task: {loss_row['task']} "
            f"loss_weight: {loss_row['loss_weight']} "
            f"cache_id: {loss_row['cache_id']}"
        )

    if stopped_reason is None:
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        saved_checkpoint = True
        print(f"saved checkpoint: {output_dir}")

    metrics = _train_metrics(
        args=args,
        num_original_examples=num_original_examples,
        num_student_rows=len(train_rows),
        loss_rows=loss_rows,
        saved_checkpoint=saved_checkpoint,
        stopped_reason=stopped_reason,
    )
    write_jsonl(loss_rows, output_dir / "train_losses.jsonl")
    _write_json(metrics, output_dir / "train_metrics.json")
    print(f"saved train losses: {output_dir / 'train_losses.jsonl'}")
    print(f"saved train metrics: {output_dir / 'train_metrics.json'}")


def _take_original_examples(split_examples, train_size: int):
    row_count = min(train_size, len(split_examples))
    if hasattr(split_examples, "select"):
        return split_examples.select(range(row_count))
    return split_examples[:row_count]


def _loss_weight(row: dict, args: argparse.Namespace) -> float:
    if args.mode == "multitask" and row.get("task") == "rationale":
        return args.rationale_loss_weight
    return 1.0


def _loss_row(step: int, row: dict, raw_loss, loss, loss_weight: float) -> dict:
    return {
        "step": step,
        "loss": float(loss.detach().cpu().item()),
        "raw_loss": float(raw_loss.detach().cpu().item()),
        "task": row.get("task", "label"),
        "loss_weight": loss_weight,
        "cache_id": row["cache_id"],
    }


def _print_dry_run(num_original_examples: int, train_rows: list[dict]) -> None:
    print("dry_run: true")
    print(f"train_size original examples: {num_original_examples}")
    print(f"number of student rows: {len(train_rows)}")
    if not train_rows:
        print("no student rows built")
        return

    first = train_rows[0]
    print(f"first row cache_id: {first['cache_id']}")
    print(f"first row task: {first.get('task', 'label')}")
    print("first prompt:")
    print(first["prompt"])
    print("first target:")
    print(first["target"])


def _training_config(
    args: argparse.Namespace,
    model_name_or_path: str,
    num_original_examples: int,
    num_student_rows: int,
) -> dict:
    return {
        "model_name": model_name_or_path,
        "config": args.config,
        "model_config": args.model_config,
        "teacher_cache_path": args.teacher_cache_path,
        "split": args.split,
        "mode": args.mode,
        "label_source": args.label_source,
        "train_size": args.train_size,
        "num_original_examples": num_original_examples,
        "num_student_rows": num_student_rows,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rationale_loss_weight": args.rationale_loss_weight,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "dtype": args.dtype,
        "output_dir": args.output_dir,
    }


def _train_metrics(
    args: argparse.Namespace,
    num_original_examples: int,
    num_student_rows: int,
    loss_rows: list[dict],
    saved_checkpoint: bool,
    stopped_reason: str | None,
) -> dict:
    return {
        "num_original_examples": num_original_examples,
        "num_student_rows": num_student_rows,
        "max_steps": args.max_steps,
        "completed_steps": len(loss_rows),
        "final_loss": loss_rows[-1]["loss"] if loss_rows else None,
        "mode": args.mode,
        "label_source": args.label_source,
        "rationale_loss_weight": args.rationale_loss_weight,
        "saved_checkpoint": saved_checkpoint,
        "stopped_reason": stopped_reason,
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
