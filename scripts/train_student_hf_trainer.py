from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ["src/datasets", "src/evaluation", "src/model", "src/prompts", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache  # noqa: E402
from io_utils import load_yaml, resolve_project_path  # noqa: E402
from scienceqa import load_scienceqa_image_examples, normalize_scienceqa_example  # noqa: E402
from seed import set_seed  # noqa: E402
from smolvlm_student import load_smolvlm  # noqa: E402
from vlm_step_by_step import format_answer_target, format_rationale_target, format_scienceqa_prompt  # noqa: E402
from transformers import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SmolVLM student with Hugging Face Trainer.")
    parser.add_argument("--config", type=str, default="src/configs/experiment/debug.yaml")
    parser.add_argument("--model_config", type=str, default="src/configs/model/smolvlm_500m.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--mode", type=str, choices=["answer_only", "filtered_multitask"], default="answer_only")
    parser.add_argument("--label_source", type=str, choices=["gold", "teacher"], default="gold")
    parser.add_argument(
        "--rationale_filter",
        type=str,
        choices=["teacher_matches_gold", "none"],
        default="teacher_matches_gold",
    )
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/train_hf_trainer_answer_only")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--rationale_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    experiment_config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(resolve_project_path(ROOT, experiment_config["dataset_config"]))
    model_config = load_yaml(ROOT / args.model_config)
    teacher_cache_path = resolve_project_path(ROOT, args.teacher_cache_path)
    output_dir = resolve_project_path(ROOT, args.output_dir)
    seed = args.seed if args.seed is not None else int(experiment_config.get("seed", 42))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size

    set_seed(seed)
    splits = load_scienceqa_image_examples(dataset_config, experiment_config)
    train_split = splits["train"]
    selected_indices = stratified_train_indices(train_split, dataset_config, args.train_size, seed)
    teacher_rows = read_teacher_cache(teacher_cache_path)
    train_examples, build_stats = build_trainer_examples(
        split_examples=train_split,
        selected_indices=selected_indices,
        dataset_config=dataset_config,
        teacher_rows=teacher_rows,
        mode=args.mode,
        label_source=args.label_source,
        rationale_filter=args.rationale_filter,
    )

    model_name_or_path = resolve_model_name_or_path(
        ROOT,
        args.model_name or model_config.get("pretrained_name", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    )

    stats = training_data_stats(train_examples, build_stats)
    print(f"config path: {ROOT / args.config}")
    print(f"model config path: {ROOT / args.model_config}")
    print(f"teacher cache path: {teacher_cache_path}")
    print(f"model name or path: {model_name_or_path}")
    print(f"mode: {args.mode}")
    print(f"label_source: {args.label_source}")
    print(f"rationale_filter: {args.rationale_filter}")
    print(f"full train image split size: {len(train_split)}")
    print(f"train size: {len(train_examples)}")
    print(f"class distribution: {dict(stats['class_distribution'])}")
    print(f"num skipped missing teacher_answer: {stats['num_skipped_missing_teacher_answer']}")
    print(f"num usable rationales: {stats['num_usable_rationales']}")
    print(f"num missing teacher rows: {stats['num_missing_teacher_rows']}")
    print(f"per_device_train_batch_size: {args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"world_size: {world_size}")
    print(f"effective_batch_size: {effective_batch_size}")
    print(f"output dir: {output_dir}")

    if not train_examples:
        raise RuntimeError("No training examples were built.")

    if args.dry_run:
        print("dry_run: true")
        print(f"first label target: {train_examples[0]['label_target']}")
        first_rationale_row = next((row for row in train_examples if row.get("use_rationale")), None)
        if first_rationale_row:
            print(f"first rationale target: {first_rationale_row['rationale_target']}")
        else:
            print("first rationale target: none")
        processor = load_processor(model_name_or_path)
        collator = StepByStepDataCollator(processor=processor, mode=args.mode)
        preview_features = train_examples[: args.per_device_train_batch_size]
        preview_batch = collator(preview_features)
        print(f"preview batch examples: {len(preview_features)}")
        print(f"label input_ids shape: {tuple(preview_batch['label_batch']['input_ids'].shape)}")
        print(f"label labels shape: {tuple(preview_batch['label_batch']['labels'].shape)}")
        if "rationale_batch" in preview_batch:
            print(f"rationale input_ids shape: {tuple(preview_batch['rationale_batch']['input_ids'].shape)}")
            print(f"rationale labels shape: {tuple(preview_batch['rationale_batch']['labels'].shape)}")
        else:
            print("rationale batch: none")
        return

    from transformers import TrainingArguments

    dtype = _training_dtype(args)
    model, processor = load_smolvlm(
        model_name_or_path=model_name_or_path,
        device="cpu",
        dtype=dtype,
        eval_mode=False,
    )
    collator = StepByStepDataCollator(processor=processor, mode=args.mode)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=seed,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = StepByStepTrainer(
        rationale_loss_weight=args.rationale_loss_weight,
        mode=args.mode,
        model=model,
        args=training_args,
        train_dataset=StepByStepDataset(train_examples),
        data_collator=collator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_config(args, model_name_or_path, seed, world_size, effective_batch_size, stats),
        output_dir / "training_config.json",
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    write_json(train_result.metrics, output_dir / "train_metrics.json")
    print(f"saved model and processor: {output_dir}")
    print(f"saved training config: {output_dir / 'training_config.json'}")
    print(f"saved train metrics: {output_dir / 'train_metrics.json'}")


class StepByStepDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class StepByStepDataCollator:
    def __init__(self, processor, mode: str):
        self.processor = processor
        self.mode = mode

    def __call__(self, features: list[dict]) -> dict:
        batch = {
            "label_batch": self._build_batch(
                features,
                prompt_key="label_prompt",
                target_key="label_target",
            )
        }
        if self.mode == "filtered_multitask":
            rationale_features = [feature for feature in features if feature["use_rationale"]]
            if rationale_features:
                batch["rationale_batch"] = self._build_batch(
                    rationale_features,
                    prompt_key="rationale_prompt",
                    target_key="rationale_target",
                )
        return batch

    def _build_batch(self, features: list[dict], prompt_key: str, target_key: str):
        prompt_texts = []
        full_texts = []
        images = []
        for feature in features:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": feature[prompt_key]},
                ],
            }
            assistant_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": feature[target_key]},
                ],
            }
            prompt_texts.append(self.processor.apply_chat_template([user_message], add_generation_prompt=True))
            full_texts.append(
                self.processor.apply_chat_template([user_message, assistant_message], add_generation_prompt=False)
            )
            images.append(feature["image"])

        inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        for row_index in range(len(features)):
            prompt_length = int(prompt_inputs["attention_mask"][row_index].sum().item())
            labels[row_index, :prompt_length] = -100

        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
        return inputs


class StepByStepTrainer(Trainer):
    def __init__(self, rationale_loss_weight: float, mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rationale_loss_weight = rationale_loss_weight
        self.mode = mode

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        label_batch = inputs["label_batch"]
        label_outputs = model(**label_batch)
        label_loss = mean_example_loss(label_outputs.logits, label_batch["labels"])

        rationale_outputs = None
        if self.mode == "filtered_multitask" and "rationale_batch" in inputs:
            rationale_batch = inputs["rationale_batch"]
            rationale_outputs = model(**rationale_batch)
            rationale_loss = mean_example_loss(rationale_outputs.logits, rationale_batch["labels"])
            total_loss = label_loss + self.rationale_loss_weight * rationale_loss
        else:
            total_loss = label_loss

        if return_outputs:
            return total_loss, {"label_outputs": label_outputs, "rationale_outputs": rationale_outputs}
        return total_loss


def mean_example_loss(logits, labels):
    import torch

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    token_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    token_mask = shift_labels.ne(-100)
    token_counts = token_mask.sum(dim=1).clamp_min(1)
    example_losses = (token_losses * token_mask).sum(dim=1) / token_counts
    example_mask = token_mask.any(dim=1)
    if not bool(example_mask.any().item()):
        return token_losses.sum() * 0.0
    return example_losses[example_mask].mean()


def build_trainer_examples(
    split_examples,
    selected_indices: list[int],
    dataset_config: dict,
    teacher_rows: list[dict],
    mode: str,
    label_source: str,
    rationale_filter: str,
) -> tuple[list[dict], dict]:
    teacher_by_cache_id = {row.get("cache_id"): row for row in teacher_rows}
    rows = []
    num_skipped_missing_teacher_answer = 0
    for local_index in selected_indices:
        example = normalize_scienceqa_example(split_examples[local_index], dataset_config)
        gold_answer = example["answer_letter"]
        if not gold_answer:
            continue

        cache_id = f"train_{local_index}"
        teacher_row = teacher_by_cache_id.get(cache_id)
        teacher_answer = teacher_row.get("teacher_answer") if teacher_row else None
        teacher_reasoning = teacher_row.get("teacher_reasoning") if teacher_row else None
        label_answer = gold_answer
        if label_source == "teacher":
            if not teacher_answer:
                num_skipped_missing_teacher_answer += 1
                continue
            label_answer = teacher_answer

        use_rationale = (
            mode == "filtered_multitask"
            and should_use_rationale(
                teacher_reasoning=teacher_reasoning,
                teacher_answer=teacher_answer,
                gold_answer=gold_answer,
                rationale_filter=rationale_filter,
            )
        )

        row = {
            "cache_id": cache_id,
            "local_index": local_index,
            "example_id": example["id"],
            "source_index": example["source_index"],
            "image": example["image"],
            "gold_answer": gold_answer,
            "teacher_answer": teacher_answer,
            "label_source": label_source,
            "label_prompt": format_scienceqa_prompt(example, dataset_config, mode=_label_prompt_mode(mode)),
            "label_target": format_answer_target(label_answer),
            "use_rationale": use_rationale,
            "has_teacher_row": teacher_row is not None,
        }
        if mode == "filtered_multitask":
            row.update(
                {
                    "rationale_prompt": format_scienceqa_prompt(example, dataset_config, mode="multitask_rationale"),
                    "rationale_target": format_rationale_target(teacher_reasoning) if teacher_reasoning else "",
                }
            )
        rows.append(row)
    return rows, {"num_skipped_missing_teacher_answer": num_skipped_missing_teacher_answer}


def should_use_rationale(
    teacher_reasoning: str | None,
    teacher_answer: str | None,
    gold_answer: str,
    rationale_filter: str,
) -> bool:
    if not teacher_reasoning:
        return False
    if rationale_filter == "teacher_matches_gold":
        return teacher_answer == gold_answer
    if rationale_filter == "none":
        return True
    raise ValueError(f"Unsupported rationale_filter: {rationale_filter}")


def stratified_train_indices(split_examples, dataset_config: dict, train_size: int | None, seed: int) -> list[int]:
    groups = defaultdict(list)
    for index in range(len(split_examples)):
        example = normalize_scienceqa_example(split_examples[index], dataset_config)
        if example["answer_letter"]:
            groups[example["answer_letter"]].append(index)

    full_size = sum(len(indices) for indices in groups.values())
    if train_size is None or train_size >= full_size:
        return sorted(index for indices in groups.values() for index in indices)
    if train_size < 1:
        raise ValueError("--train_size must be >= 1 when provided")

    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    allocations = {}
    remainders = []
    for answer, indices in groups.items():
        exact = train_size * len(indices) / full_size
        count = min(len(indices), int(math.floor(exact)))
        allocations[answer] = count
        remainders.append((exact - count, answer))

    remaining = train_size - sum(allocations.values())
    for _, answer in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[answer] < len(groups[answer]):
            allocations[answer] += 1
            remaining -= 1

    if remaining > 0:
        for answer in sorted(groups):
            if remaining <= 0:
                break
            available = len(groups[answer]) - allocations[answer]
            extra = min(available, remaining)
            allocations[answer] += extra
            remaining -= extra

    selected = []
    for answer, count in allocations.items():
        selected.extend(groups[answer][:count])
    return sorted(selected)


def training_data_stats(rows: list[dict], build_stats: dict) -> dict:
    return {
        "class_distribution": dict(sorted(Counter(row["gold_answer"] for row in rows).items())),
        "num_skipped_missing_teacher_answer": build_stats["num_skipped_missing_teacher_answer"],
        "num_usable_rationales": sum(1 for row in rows if row.get("use_rationale")),
        "num_missing_teacher_rows": sum(1 for row in rows if not row.get("has_teacher_row")),
    }


def run_config(
    args: argparse.Namespace,
    model_name_or_path: str,
    seed: int,
    world_size: int,
    effective_batch_size: int,
    stats: dict,
) -> dict:
    return {
        "config": args.config,
        "model_config": args.model_config,
        "model_name": model_name_or_path,
        "teacher_cache_path": args.teacher_cache_path,
        "mode": args.mode,
        "label_source": args.label_source,
        "rationale_filter": args.rationale_filter,
        "train_size": args.train_size,
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "rationale_loss_weight": args.rationale_loss_weight,
        "seed": seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "data_stats": stats,
    }


def _label_prompt_mode(mode: str) -> str:
    if mode == "filtered_multitask":
        return "multitask_label"
    return "answer_only"


def _training_dtype(args: argparse.Namespace) -> str:
    if args.bf16:
        return "bfloat16"
    if args.fp16:
        return "float16"
    return "auto"


def _validate_args(args: argparse.Namespace) -> None:
    if args.bf16 and args.fp16:
        raise ValueError("Use only one precision flag: --bf16 or --fp16.")
    if args.per_device_train_batch_size < 1:
        raise ValueError("--per_device_train_batch_size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be > 0")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max_steps must be >= 1 when provided")


def load_processor(model_name_or_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name_or_path)


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


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
