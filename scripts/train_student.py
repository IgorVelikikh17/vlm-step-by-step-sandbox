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
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/train_debug_answer_only_gold")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("train_student.py currently supports batch_size=1 for simple task-level loss weighting.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1")

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
    print(f"log_every: {args.log_every}")
    print(f"rationale_loss_weight: {args.rationale_loss_weight}")
    print(f"output dir: {output_dir}")

    if args.dry_run:
        _print_dry_run(num_original_examples, train_rows)
        return

    if not train_rows:
        raise RuntimeError("No student rows were built for training.")
    if args.mode == "multitask":
        _validate_multitask_pairs(train_rows)

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

    saved_checkpoint = False
    if args.mode == "multitask":
        loss_rows, stopped_reason = _run_multitask_training(
            train_rows=train_rows,
            model=model,
            processor=processor,
            optimizer=optimizer,
            train_device=train_device,
            args=args,
            torch=torch,
        )
    else:
        loss_rows, stopped_reason = _run_answer_only_training(
            train_rows=train_rows,
            model=model,
            processor=processor,
            optimizer=optimizer,
            train_device=train_device,
            args=args,
            torch=torch,
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


def _run_answer_only_training(
    train_rows: list[dict],
    model,
    processor,
    optimizer,
    train_device: str,
    args: argparse.Namespace,
    torch,
) -> tuple[list[dict], str | None]:
    loss_rows = []
    for step_index in range(args.max_steps):
        row = train_rows[step_index % len(train_rows)]
        batch = build_smolvlm_training_batch([row], processor, device=train_device)
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            loss_rows.append(_answer_only_loss_row(step_index + 1, row, loss))
            print(f"stopping: non-finite loss at step {step_index + 1}")
            return loss_rows, "non_finite_loss"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        loss_row = _answer_only_loss_row(step_index + 1, row, loss)
        loss_rows.append(loss_row)
        if _should_log_step(loss_row["step"], args.max_steps, args.log_every):
            print(
                f"step: {loss_row['step']} "
                f"loss: {loss_row['loss']:.6f} "
                f"task: {loss_row['task']} "
                f"cache_id: {loss_row['cache_id']}"
            )

    return loss_rows, None


def _run_multitask_training(
    train_rows: list[dict],
    model,
    processor,
    optimizer,
    train_device: str,
    args: argparse.Namespace,
    torch,
) -> tuple[list[dict], str | None]:
    loss_rows = []
    num_pairs = len(train_rows) // 2
    for step_index in range(args.max_steps):
        pair_index = step_index % num_pairs
        label_row = train_rows[pair_index * 2]
        rationale_row = train_rows[pair_index * 2 + 1]

        label_batch = build_smolvlm_training_batch([label_row], processor, device=train_device)
        label_outputs = model(**label_batch)
        label_loss = label_outputs.loss

        rationale_batch = build_smolvlm_training_batch([rationale_row], processor, device=train_device)
        rationale_outputs = model(**rationale_batch)
        rationale_loss = rationale_outputs.loss

        total_loss = label_loss + args.rationale_loss_weight * rationale_loss
        loss_row = _multitask_loss_row(
            step=step_index + 1,
            label_row=label_row,
            rationale_row=rationale_row,
            label_loss=label_loss,
            rationale_loss=rationale_loss,
            total_loss=total_loss,
            rationale_loss_weight=args.rationale_loss_weight,
        )

        if not _all_finite(torch, label_loss, rationale_loss, total_loss):
            loss_rows.append(loss_row)
            print(f"stopping: non-finite multitask loss at step {step_index + 1}")
            return loss_rows, "non_finite_loss"

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        loss_rows.append(loss_row)
        if _should_log_step(loss_row["step"], args.max_steps, args.log_every):
            print(
                f"step: {loss_row['step']} "
                f"total_loss: {loss_row['total_loss']:.6f} "
                f"label_loss: {loss_row['label_loss']:.6f} "
                f"rationale_loss: {loss_row['rationale_loss']:.6f} "
                f"rationale_loss_weight: {loss_row['rationale_loss_weight']} "
                f"base_cache_id: {loss_row['base_cache_id']}"
            )

    return loss_rows, None


def _should_log_step(step: int, max_steps: int, log_every: int) -> bool:
    return step == 1 or step % log_every == 0 or step == max_steps


def _validate_multitask_pairs(train_rows: list[dict]) -> None:
    if len(train_rows) < 2 or len(train_rows) % 2 != 0:
        raise ValueError("multitask training requires an even number of label/rationale rows.")

    for pair_start in range(0, len(train_rows), 2):
        label_row = train_rows[pair_start]
        rationale_row = train_rows[pair_start + 1]
        if label_row.get("task") != "label":
            raise ValueError(f"Expected label row at index {pair_start}, got: {label_row.get('cache_id')}")
        if rationale_row.get("task") != "rationale":
            raise ValueError(f"Expected rationale row at index {pair_start + 1}, got: {rationale_row.get('cache_id')}")
        if label_row.get("base_cache_id") != rationale_row.get("base_cache_id"):
            raise ValueError(
                "Mismatched multitask pair: "
                f"{label_row.get('cache_id')} and {rationale_row.get('cache_id')}"
            )


def _all_finite(torch, *losses) -> bool:
    return all(bool(torch.isfinite(loss).item()) for loss in losses)


def _answer_only_loss_row(step: int, row: dict, loss) -> dict:
    return {
        "step": step,
        "loss": float(loss.detach().cpu().item()),
        "task": row.get("task", "label"),
        "cache_id": row["cache_id"],
    }


def _multitask_loss_row(
    step: int,
    label_row: dict,
    rationale_row: dict,
    label_loss,
    rationale_loss,
    total_loss,
    rationale_loss_weight: float,
) -> dict:
    return {
        "step": step,
        "base_cache_id": label_row["base_cache_id"],
        "label_cache_id": label_row["cache_id"],
        "rationale_cache_id": rationale_row["cache_id"],
        "label_loss": float(label_loss.detach().cpu().item()),
        "rationale_loss": float(rationale_loss.detach().cpu().item()),
        "rationale_loss_weight": rationale_loss_weight,
        "total_loss": float(total_loss.detach().cpu().item()),
    }


def _print_dry_run(num_original_examples: int, train_rows: list[dict]) -> None:
    print("dry_run: true")
    print(f"train_size original examples: {num_original_examples}")
    print(f"number of student rows: {len(train_rows)}")
    if not train_rows:
        print("no student rows built")
        return

    if len(train_rows) >= 2 and train_rows[0].get("task") == "label" and train_rows[1].get("task") == "rationale":
        label_row = train_rows[0]
        rationale_row = train_rows[1]
        print("first label row:")
        print(f"cache_id: {label_row['cache_id']}")
        print("target:")
        print(label_row["target"])
        print()
        print("first rationale row:")
        print(f"cache_id: {rationale_row['cache_id']}")
        print("target:")
        print(rationale_row["target"])
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
        "log_every": args.log_every,
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
    num_optimizer_steps = _num_optimizer_steps(loss_rows, stopped_reason)
    return {
        "num_original_examples": num_original_examples,
        "num_student_rows": num_student_rows,
        "max_steps": args.max_steps,
        "completed_steps": num_optimizer_steps,
        "final_loss": _final_loss(loss_rows),
        "mode": args.mode,
        "label_source": args.label_source,
        "rationale_loss_weight": args.rationale_loss_weight,
        "loss_formula": _loss_formula(args.mode),
        "num_optimizer_steps": num_optimizer_steps,
        "num_label_rows_used": _num_label_rows_used(args.mode, num_optimizer_steps),
        "num_rationale_rows_used": _num_rationale_rows_used(args.mode, num_optimizer_steps),
        "saved_checkpoint": saved_checkpoint,
        "stopped_reason": stopped_reason,
    }


def _num_optimizer_steps(loss_rows: list[dict], stopped_reason: str | None) -> int:
    if stopped_reason is None:
        return len(loss_rows)
    return max(0, len(loss_rows) - 1)


def _final_loss(loss_rows: list[dict]) -> float | None:
    if not loss_rows:
        return None
    if "total_loss" in loss_rows[-1]:
        return loss_rows[-1]["total_loss"]
    return loss_rows[-1]["loss"]


def _loss_formula(mode: str) -> str:
    if mode == "multitask":
        return "label_loss + rationale_loss_weight * rationale_loss"
    return "loss"


def _num_label_rows_used(mode: str, num_optimizer_steps: int) -> int:
    if mode == "multitask":
        return num_optimizer_steps
    return num_optimizer_steps


def _num_rationale_rows_used(mode: str, num_optimizer_steps: int) -> int:
    if mode == "multitask":
        return num_optimizer_steps
    return 0


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
