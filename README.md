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
- `scripts/` contains direct command-line entrypoints
- `src/datasets/` contains the small ScienceQA loader helper
- `src/prompts/` contains prompt formatting for VLM step-by-step answers
- `src/utils/` keeps simple shared helpers

The code intentionally avoids registries, factories, dataclass-heavy config
objects, callback stacks, and trainer frameworks.

## Main Scripts

- `scripts/generate_teacher_cache.py`: generate mock or Qwen teacher outputs
  and save them as JSONL.
- `scripts/train_student.py`: train SmolVLM with answer-only or multitask
  objectives.
- `scripts/evaluate_student.py`: evaluate base models or saved checkpoints.
- `scripts/run_labeled_reducing_data.py`: run answer-only vs multitask
  comparisons for selected train sizes.
- `scripts/analyze_predictions.py`: inspect prediction distributions,
  parse failures, and common errors.
- `scripts/plot_labeled_reducing_data.py`: plot accuracy vs train size.

## Verification Utilities

These scripts are kept intentionally. They make the pipeline easier to inspect
and debug:

- `scripts/inspect_scienceqa.py`
- `scripts/inspect_teacher_cache.py`
- `scripts/inspect_student_data.py`
- `scripts/smoke_smolvlm_inference.py`
- `scripts/train_student_smoke.py`
- `scripts/check_vlm_environment.py`

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
python scripts/inspect_scienceqa.py --config src/configs/experiment/debug.yaml
```

To inspect the current 256-example shuffled experiment config:

```bash
python scripts/inspect_scienceqa.py \
  --config src/configs/experiment/debug_qwen_labeled_reducing_256.yaml
```

If Hugging Face cannot download the dataset in your environment, rerun later
with network access or a prepared local cache.

## Secrets

Do not save `HF_TOKEN`, API keys, or model access tokens in repository files.
Use environment variables when gated model access is needed.
