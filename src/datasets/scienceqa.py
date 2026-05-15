from __future__ import annotations

import random

from datasets import DatasetDict, load_dataset


def load_scienceqa_image_examples(dataset_config: dict, experiment_config: dict | None = None) -> DatasetDict:
    dataset_name = dataset_config["hf_name"]
    subset = dataset_config.get("subset")
    if subset is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    train_split_name = dataset_config.get("train_split", "train")
    eval_split_name = dataset_config.get("eval_split", "validation")
    image_column = dataset_config.get("image_column", "image")

    result = DatasetDict(
        {
            "train": _prepare_split(
                dataset[train_split_name],
                image_column=image_column,
                only_with_image=dataset_config.get("only_with_image", True),
                shuffle=_shuffle_enabled(experiment_config, "shuffle_train"),
                shuffle_seed=_shuffle_seed(experiment_config),
                max_samples=_sample_limit(dataset_config, experiment_config, "max_train_samples"),
            ),
            "validation": _prepare_split(
                dataset[eval_split_name],
                image_column=image_column,
                only_with_image=dataset_config.get("only_with_image", True),
                shuffle=_shuffle_enabled(experiment_config, "shuffle_eval"),
                shuffle_seed=_shuffle_seed(experiment_config),
                max_samples=_sample_limit(dataset_config, experiment_config, "max_eval_samples"),
            ),
        }
    )
    return result


def normalize_scienceqa_example(example: dict, dataset_config: dict) -> dict:
    answer_column = dataset_config.get("answer_column", "answer")
    answer = example.get(answer_column)

    return {
        "id": _example_id(example),
        "source_index": example.get("source_index"),
        "image": example.get(dataset_config.get("image_column", "image")),
        "question": example.get(dataset_config.get("question_column", "question")),
        "choices": list(example.get(dataset_config.get("choices_column", "choices")) or []),
        "answer": answer,
        "answer_letter": _answer_index_to_letter(answer) if answer is not None else None,
        "hint": _optional_value(example, dataset_config.get("hint_column", "hint")),
        "lecture": _optional_value(example, dataset_config.get("lecture_column", "lecture")),
        "solution": _optional_value(example, dataset_config.get("solution_column", "solution")),
    }


def _example_id(example: dict):
    for key in ["id", "qid", "question_id", "pid", "index"]:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _optional_value(example: dict, column: str | None):
    if not column:
        return None
    value = example.get(column)
    if value == "":
        return None
    return value


def _answer_index_to_letter(answer_index: int) -> str:
    return chr(65 + int(answer_index))


def _sample_limit(dataset_config: dict, experiment_config: dict | None, key: str) -> int | None:
    if experiment_config and experiment_config.get(key) is not None:
        return experiment_config[key]
    return dataset_config.get(key)


def _shuffle_enabled(experiment_config: dict | None, key: str) -> bool:
    return bool(experiment_config and experiment_config.get(key, False))


def _shuffle_seed(experiment_config: dict | None) -> int:
    if not experiment_config:
        return 42
    return int(experiment_config.get("shuffle_seed", experiment_config.get("seed", 42)))


def _prepare_split(
    split,
    image_column: str,
    only_with_image: bool,
    shuffle: bool,
    shuffle_seed: int,
    max_samples: int | None,
):
    split = _add_source_index(split)
    if only_with_image:
        split = split.filter(lambda example: example[image_column] is not None)
    if shuffle:
        split = _shuffle_split(split, shuffle_seed)
    if max_samples is not None:
        split = split.select(range(min(max_samples, len(split))))
    return split


def _add_source_index(split):
    return split.map(lambda example, index: {"source_index": index}, with_indices=True)


def _shuffle_split(split, shuffle_seed: int):
    indices = list(range(len(split)))
    random.Random(shuffle_seed).shuffle(indices)
    return split.select(indices)
