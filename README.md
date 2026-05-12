# VLM Step-by-Step Distillation Sandbox

This is a minimal, educational sandbox for a VLM version of the paper idea
“Distilling Step-by-Step”.

The old `minimal_q1_q2_hf_main_style_sandbox` studied text robustness on
SST-2. This new copy keeps the same small-project style, but changes the topic
to multimodal multiple-choice question answering on ScienceQA.

## Goal

Build the first safe skeleton for a future VLM distillation coursework project.

The intended full project is:

- teacher: a larger VLM, planned as `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
  with a smaller Qwen-VL fallback later
- student: a smaller VLM, starting with
  `HuggingFaceTB/SmolVLM-500M-Instruct`
- later model-size experiment: SmolVLM 256M vs 500M
- dataset: ScienceQA examples that contain images
- task: image + question + choices -> reasoning + final answer letter

In simple terms: the teacher should write both the answer and a short
explanation. The student later learns not only “A/B/C/D”, but also the
intermediate reasoning text. That is the step-by-step distillation idea.

## Current Scope

This milestone only prepares the project skeleton.

Implemented now:

- VLM-specific README and architecture notes
- minimal YAML configs under `src/configs/`
- a small ScienceQA image-example inspection script
- simple prompt formatting for reasoning plus final answer letter
- reused lightweight YAML/seed utilities from the previous sandbox

Not implemented yet:

- teacher generation
- student fine-tuning
- evaluation loop
- model-size comparison
- AWQ or quantized inference setup
- Comet/W&B experiment tracking

## Project Style

- `src/configs/` holds readable experiment, dataset, model, and teacher configs
- `scripts/` contains direct command-line entrypoints
- `src/datasets/` contains the small ScienceQA loader helper
- `src/prompts/` contains prompt formatting for VLM step-by-step answers
- `src/utils/` keeps simple shared helpers

The code intentionally avoids registries, factories, dataclass-heavy config
objects, callback stacks, and trainer frameworks.

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

If Hugging Face cannot download the dataset in your environment, rerun later
with network access or a prepared local cache.

## Secrets

Do not save `HF_TOKEN`, API keys, or model access tokens in repository files.
Use environment variables when gated model access is needed.
