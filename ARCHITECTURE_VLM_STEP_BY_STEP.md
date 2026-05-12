# Minimal VLM Architecture

The current project is only a skeleton for a future VLM step-by-step
distillation pipeline.

## Intended Flow

```text
configs
-> ScienceQA image examples
-> prompt: image + question + choices
-> teacher VLM writes reasoning + final answer letter
-> student VLM learns from answer and reasoning
-> evaluation compares final answer letters
```

## What Exists Now

- `src/configs/dataset/scienceqa.yaml`: ScienceQA image-only dataset settings
- `src/configs/model/smolvlm_500m.yaml`: first student model config
- `src/configs/model/smolvlm_256m.yaml`: later smaller-student comparison config
- `src/configs/teacher/qwen_vl.yaml`: planned teacher model config
- `src/configs/experiment/debug.yaml`: tiny debug experiment config
- `scripts/inspect_scienceqa.py`: checks image examples and prompt shape
- `src/datasets/scienceqa.py`: small dataset loading/filtering helper
- `src/prompts/vlm_step_by_step.py`: simple prompt formatting

## Why This Is Minimal

- No training loop yet
- No teacher-generation cache yet
- No evaluation runner yet
- No model registry or factory layer
- No callback system
- No quantization-specific dependencies yet

The next safe step is usually teacher-data generation for a tiny debug subset,
saved as JSONL, before adding any student training code.
