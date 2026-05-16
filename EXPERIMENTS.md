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

### SmolVLM-256M, train=512, eval=512, 2 epochs

Setup:

- student: `HuggingFaceTB/SmolVLM-256M-Instruct`
- teacher: `Qwen/Qwen2.5-VL-3B-Instruct`
- train size: 512 ScienceQA image examples
- eval size: 512 ScienceQA image examples
- learning rate: `1e-6`
- batch size: 1
- filtered rationale mode: enabled
- filter rule: use rationale loss only when `teacher_answer == gold_answer`
- compared rationale weights: `lambda=0.3` and `lambda=0.5`

Teacher cache quality:

| Teacher | Rows | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: | ---: |
| Qwen/Qwen2.5-VL-3B-Instruct | 512 | 381 / 512 | 74.41% | 0.0% |

Main accuracy results:

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| base | 232 / 512 | 45.31% |
| answer_only_gold | 249 / 512 | 48.63% |
| filtered multitask, lambda=0.3 | 260 / 512 | 50.78% |
| filtered multitask, lambda=0.5 | 261 / 512 | 50.98% |

Parse failure rates:

| Method | Parse failure rate |
| --- | ---: |
| base | 22.46% |
| answer_only_gold | 17.97% |
| filtered multitask, lambda=0.3 | 12.70% |
| filtered multitask, lambda=0.5 | 12.89% |

Prediction-level comparison:

| Comparison | Different predictions | First method better | Second method better | Both wrong but changed |
| --- | ---: | ---: | ---: | ---: |
| answer_only vs lambda=0.3 | 58 | 12 | 23 | 23 |
| answer_only vs lambda=0.5 | 60 | 12 | 24 | 24 |
| lambda=0.3 vs lambda=0.5 | 14 | 3 | 4 | 7 |

## Interpretation

For SmolVLM-500M, rationale supervision has not helped yet in the current
pilot setup. The filtered multitask run is slightly worse than the base and
answer-only runs.

For SmolVLM-256M, filtered multitask gives a small positive signal after
3 epochs: it improves by 2 correct answers over answer-only and by 6 correct
answers over the base model. It also reduces parse failures compared with the
base and answer-only setups.

The effect is still small, so it should not be treated as a final conclusion.

On the 512-example SmolVLM-256M pilot run, filtered multitask training is more
clearly better than both answer-only fine-tuning and the base model. It improves
over answer-only by 11 correct examples for `lambda=0.3` and by 12 correct
examples for `lambda=0.5`.

The two rationale weights are nearly equivalent in this run. `lambda=0.5` is
higher by only 1 correct example out of 512, so the current evidence supports
filtered multitask rationale supervision for SmolVLM-256M but does not clearly
prefer `lambda=0.5` over `lambda=0.3`.

## Limitations

- The experiments are still pilot-scale. The largest current run uses 512 train
  and 512 evaluation examples.
- The teacher is imperfect: Qwen teacher accuracy is about 75% on the
  current 256- and 512-example train subsets.
- Some teacher rationales can contradict the gold label.
- The current filtered multitask mode skips rationale loss when the teacher
  answer does not match the gold answer, but this is still a simple heuristic.
- Results are from pilot/debug-scale runs and need more seeds and larger
  subsets.

## Next Experiments

- Run larger train sizes, such as 1024 examples.
- Repeat key runs with multiple seeds.
- Compare SmolVLM-256M and SmolVLM-500M under the same settings.
- Sweep `rationale_loss_weight`, especially values below 1.0.
- Analyze prediction changes and parse failures for the filtered multitask
  setup.
