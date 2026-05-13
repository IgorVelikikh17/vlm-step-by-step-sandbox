from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/model", "src/prompts", "src/teacher", "src/training", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path  # noqa: E402
from scienceqa import load_scienceqa_image_examples  # noqa: E402
from seed import set_seed  # noqa: E402
from smolvlm_batching import build_smolvlm_training_batch  # noqa: E402
from smolvlm_student import load_smolvlm  # noqa: E402
from student_data import build_student_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny SmolVLM student training smoke test.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="train")
    parser.add_argument("--mode", type=str, choices=["answer_only", "reasoning_answer"], default="answer_only")
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument("--max_train_samples", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/smoke_smolvlm_answer_only_gold")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        max_samples=args.max_train_samples,
    )

    model_name = args.model_name or model_config.get("pretrained_name", "HuggingFaceTB/SmolVLM-500M-Instruct")
    print(f"config path: {ROOT / args.config}")
    print(f"model config path: {ROOT / args.model_config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"model name: {model_name}")
    print(f"split: {args.split}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"student rows ready: {len(student_rows)}")
    print(f"max_steps: {args.max_steps}")
    print(f"batch_size: {args.batch_size}")

    if args.dry_run:
        _print_dry_run(student_rows)
        return

    if not student_rows:
        raise RuntimeError("No student rows were built for training smoke test.")

    import torch

    model, processor = load_smolvlm(
        model_name=model_name,
        device=args.device,
        dtype=args.dtype,
        eval_mode=False,
    )
    train_device = str(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    step = 0
    while step < args.max_steps:
        start = (step * args.batch_size) % len(student_rows)
        batch_rows = student_rows[start : start + args.batch_size]
        if len(batch_rows) < args.batch_size:
            batch_rows = batch_rows + student_rows[: args.batch_size - len(batch_rows)]

        batch = build_smolvlm_training_batch(batch_rows, processor, device=train_device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        cache_ids = [row["cache_id"] for row in batch_rows]
        print(f"step: {step} loss: {loss.item():.6f} cache_ids: {cache_ids}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    _save_training_config(args, model_name, output_dir, len(student_rows))
    print(f"saved smoke checkpoint: {output_dir}")


def _print_dry_run(student_rows: list[dict]) -> None:
    print(f"dry_run: true")
    if not student_rows:
        print("no student rows built")
        return

    first = student_rows[0]
    print(f"image exists: {first['image'] is not None}")
    print("first prompt:")
    print(first["prompt"])
    print("first target:")
    print(first["target"])


def _save_training_config(args: argparse.Namespace, model_name: str, output_dir: Path, num_rows: int) -> None:
    payload = {
        "model_name": model_name,
        "config": args.config,
        "model_config": args.model_config,
        "teacher_cache_path": args.teacher_cache_path,
        "split": args.split,
        "mode": args.mode,
        "label_source": args.label_source,
        "max_train_samples": args.max_train_samples,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "dtype": args.dtype,
        "num_student_rows": num_rows,
    }
    with (output_dir / "training_smoke_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
