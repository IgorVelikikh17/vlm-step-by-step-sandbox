# VLM Step-by-Step Distillation Sandbox

This is a minimal, educational sandbox for a VLM version of the paper idea
“Distilling Step-by-Step”.

The old `minimal_q1_q2_hf_main_style_sandbox` studied text robustness on
SST-2. This new copy keeps the same small-project style, but changes the topic
to multimodal multiple-choice question answering on ScienceQA.

## Goal

Build a small, readable VLM version of the "Distilling Step-by-Step" idea for
ScienceQA image examples.

The current task is:

```text
image + question + choices -> final answer letter
```

The project compares answer-only fine-tuning with multitask rationale training:

```text
[label] + input -> gold answer
[rationale] + input -> teacher rationale
```

The teacher is `Qwen/Qwen2.5-VL-3B-Instruct`. The student models are
`HuggingFaceTB/SmolVLM-500M-Instruct` and
`HuggingFaceTB/SmolVLM-256M-Instruct`.

## Current Scope

Implemented now:

- ScienceQA image-only data loading with reproducible shuffled subsets
- prompt formatting for answer-only and multitask training
- Qwen teacher cache generation
- SmolVLM inference smoke tests
- SmolVLM student training
- answer-only baseline
- multitask rationale training
- filtered multitask rationale training
- evaluation with `metrics.json` and `predictions.jsonl`
- prediction analysis and reducing-data runners

Current pilot results are summarized in [EXPERIMENTS.md](EXPERIMENTS.md).

Not implemented yet:

- full-scale benchmark runs
- Comet/W&B experiment tracking
- LoRA/QLoRA
- unlabeled or pseudo-labeled setup
- large model-size experiment beyond pilot runs

## Project Style

- `src/configs/` holds readable experiment, dataset, model, and teacher configs
- `scripts/` contains the three main reproducible pipeline entrypoints
- `scripts/dev/` contains inspection, smoke-test, and environment-check
  utilities that are not required for the main pipeline
- `src/datasets/` contains the small ScienceQA loader helper
- `src/prompts/` contains prompt formatting for VLM step-by-step answers
- `src/utils/` keeps simple shared helpers

The code intentionally avoids registries, factories, dataclass-heavy config
objects, and custom experiment frameworks.

## Main Pipeline

The reproducible pipeline has exactly three executable entrypoints:

1. `scripts/generate_teacher_cache.py`: generates Qwen teacher answers and
   rationales and saves them into a JSONL teacher cache.
2. `scripts/train_student_hf_trainer.py`: trains the SmolVLM student model. It
   supports gold-label fine-tuning, teacher-label distillation, and
   step-by-step multitask distillation with rationales.
3. `scripts/evaluate_student.py`: evaluates raw or trained SmolVLM models on
   the ScienceQA validation split and saves metrics and predictions.

Minimal flow:

```text
generate teacher cache -> train student -> evaluate student
```

## Development Utilities

`scripts/dev/` contains legacy and development utilities such as inspection,
smoke-test, analysis, plotting, and old pilot-training scripts. They are kept
for debugging and historical pilot runs, but they are not required for the main
reproducible pipeline.

## Verification Utilities

Useful development utilities include:

- `scripts/dev/inspect_scienceqa.py`
- `scripts/dev/inspect_teacher_cache.py`
- `scripts/dev/inspect_student_data.py`
- `scripts/dev/smoke_smolvlm_inference.py`
- `scripts/dev/train_student_smoke.py`
- `scripts/dev/check_vlm_environment.py`
- `scripts/dev/run_debug_labeled_comparison.py`
- `scripts/dev/analyze_predictions.py`
- `scripts/dev/plot_labeled_reducing_data.py`
- `scripts/dev/run_labeled_reducing_data.py`
- `scripts/dev/train_student.py`

## Install

```bash
cd ../vlm_step_by_step_sandbox
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First Debug Command

This command only checks that the ScienceQA config and image filtering shape
make sense. It does not train or evaluate a model.

```bash
python scripts/dev/inspect_scienceqa.py --config src/configs/experiment/debug.yaml
```

To inspect the current 256-example shuffled experiment config:

```bash
python scripts/dev/inspect_scienceqa.py \
  --config src/configs/experiment/debug_qwen_labeled_reducing_256.yaml
```

If Hugging Face cannot download the dataset in your environment, rerun later
with network access or a prepared local cache.

## Secrets

Do not save `HF_TOKEN`, API keys, or model access tokens in repository files.
Use environment variables when gated model access is needed.
