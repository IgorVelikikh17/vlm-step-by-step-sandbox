# Minimal VLM Architecture

The project is a small, educational VLM step-by-step distillation sandbox. It
is still intentionally simple, but it now includes the full pilot path from
ScienceQA examples to teacher cache, student training, and evaluation.

## Intended Flow

```text
configs
-> ScienceQA image examples
-> prompt: image + question + choices
-> teacher VLM cache writes reasoning + final answer letter
-> student VLM trains on answer-only or multitask rows
-> evaluation compares final answer letters
```

## What Exists Now

- `src/configs/dataset/scienceqa.yaml`: ScienceQA image-only dataset settings
- `src/configs/model/smolvlm_500m.yaml`: first student model config
- `src/configs/model/smolvlm_256m.yaml`: smaller student model config
- `src/configs/teacher/qwen_vl.yaml`: Qwen teacher model config
- `src/configs/experiment/debug_qwen_labeled_reducing_256.yaml`: current
  256-example shuffled pilot config
- `src/datasets/scienceqa.py`: dataset loading, image filtering, source index,
  deterministic shuffle, and subset selection
- `src/datasets/student_data.py`: answer-only, multitask, and filtered
  multitask student rows
- `src/prompts/vlm_step_by_step.py`: answer-only, `[label]`, and
  `[rationale]` prompt formatting
- `src/teacher/qwen_vl_teacher.py`: minimal Qwen teacher wrapper
- `src/model/smolvlm_student.py`: minimal SmolVLM load/generate helper
- `src/training/smolvlm_batching.py`: processor batching and label masking
- `src/evaluation/parsing.py`: answer parsing utilities
- `scripts/generate_teacher_cache.py`: teacher cache generation
- `scripts/train_student_hf_trainer.py`: main Hugging Face Trainer-based
  student training
- `scripts/evaluate_student.py`: evaluation and prediction saving
- `scripts/dev/train_student.py`: legacy batch-size-1 pilot training
- `scripts/dev/run_labeled_reducing_data.py`: development reducing-data
  comparison runner

## Why This Is Minimal

- No model registry or factory layer
- No callback system
- No experiment framework
- No Comet/W&B integration
- No LoRA/QLoRA
- No quantization-specific training setup

Generated artifacts such as teacher caches, checkpoints, predictions, and
plots are intentionally kept out of git. The repository should contain the
code, configs, and short human-readable summaries rather than heavy outputs.
