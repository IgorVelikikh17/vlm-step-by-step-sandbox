# Experiments

This file summarizes the current pilot experiments for the VLM
Step-by-Step Distillation sandbox. These are debug-scale experiments, not
final benchmark results.

## Goal

The project adapts the idea from "Distilling Step-by-Step" to multimodal
multiple-choice question answering. The input is a ScienceQA image example:

```text
image + question + choices
```

The model must output the final answer letter. In the multitask setup, the
student also learns from teacher rationales during training.

## Methods

Current methods:

- `base`: evaluate the original SmolVLM checkpoint without fine-tuning.
- `answer_only_gold`: supervised fine-tuning on gold answers only.
- `multitask_gold_teacher_rationale`: multitask training with gold labels and
  Qwen teacher rationales.
- `multitask_gold_teacher_rationale_filtered`: same as multitask training, but
  rationale loss is used only when `teacher_answer == gold_answer`.

The multitask prompts follow this format:

```text
[label] + image + question + choices -> Answer: B
[rationale] + image + question + choices -> Reasoning: ...
```

The answer-only baseline does not use a `[label]` prefix:

```text
image + question + choices -> Answer: B
```

## Current Pilot Results

Teacher quality on 256 train examples:

| Teacher | Correct | Accuracy |
| --- | ---: | ---: |
| Qwen/Qwen2.5-VL-3B-Instruct | 193 / 256 | 75.39% |

SmolVLM-500M, train=256, eval=256:

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| base | 173 / 256 | 67.58% |
| answer_only_gold | 173 / 256 | 67.58% |
| filtered multitask | 168 / 256 | 65.63% |

SmolVLM-256M, train=256, eval=256, 3 epochs:

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| base | 125 / 256 | 48.83% |
| answer_only_gold | 129 / 256 | 50.39% |
| filtered multitask | 131 / 256 | 51.17% |

Parse failures for SmolVLM-256M:

| Method | Parse failure rate |
| --- | ---: |
| base | 20.31% |
| answer_only_gold | 16.02% |
| filtered multitask | 14.06% |

Prediction comparison for SmolVLM-256M, `answer_only_gold` vs filtered
multitask:

| Category | Count |
| --- | ---: |
| same predictions | 229 |
| different predictions | 27 |
| answer_only better | 8 |
| multitask better | 10 |
| both wrong but changed | 9 |

## Interpretation

For SmolVLM-500M, rationale supervision has not helped yet in the current
pilot setup. The filtered multitask run is slightly worse than the base and
answer-only runs.

For SmolVLM-256M, filtered multitask gives a small positive signal after
3 epochs: it improves by 2 correct answers over answer-only and by 6 correct
answers over the base model. It also reduces parse failures compared with the
base and answer-only setups.

The effect is still small, so it should not be treated as a final conclusion.

## Limitations

- The experiments use only 256 train and 256 evaluation examples.
- The teacher is imperfect: Qwen teacher accuracy is about 75% on the
  256-example train subset.
- Some teacher rationales can contradict the gold label.
- The current filtered multitask mode skips rationale loss when the teacher
  answer does not match the gold answer, but this is still a simple heuristic.
- Results are from pilot/debug-scale runs and need more seeds and larger
  subsets.

## Next Experiments

- Run larger train sizes, such as 512 and 1024 examples.
- Repeat key runs with multiple seeds.
- Compare SmolVLM-256M and SmolVLM-500M under the same settings.
- Sweep `rationale_loss_weight`, especially values below 1.0.
- Analyze prediction changes and parse failures for the filtered multitask
  setup.
