from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


LETTERS = ["A", "B", "C", "D", "E"]
PREDICTION_COLUMNS = LETTERS + ["None"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one predictions.jsonl file.")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--max_errors", type=int, default=5)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions_path)
    rows = _read_jsonl(predictions_path)
    summary = _build_summary(rows)

    _print_summary(predictions_path, summary, rows, max_errors=args.max_errors)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print()
        print(f"saved analysis json: {output_path}")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_summary(rows: list[dict]) -> dict:
    num_examples = len(rows)
    num_correct = 0
    num_parse_failures = 0
    gold_distribution = Counter()
    prediction_distribution = Counter()
    confusion_matrix = {
        gold_letter: {pred_letter: 0 for pred_letter in PREDICTION_COLUMNS}
        for gold_letter in LETTERS
    }

    for row in rows:
        gold_answer = _answer_or_none(row.get("gold_answer"))
        pred_answer = _answer_or_none(row.get("pred_answer"))
        pred_key = pred_answer if pred_answer is not None else "None"

        if gold_answer is not None:
            gold_distribution[gold_answer] += 1
        prediction_distribution[pred_key] += 1

        if pred_answer is None:
            num_parse_failures += 1
        if pred_answer is not None and pred_answer == gold_answer:
            num_correct += 1

        if gold_answer in confusion_matrix and pred_key in PREDICTION_COLUMNS:
            confusion_matrix[gold_answer][pred_key] += 1

    return {
        "num_examples": num_examples,
        "accuracy": num_correct / num_examples if num_examples else 0.0,
        "num_correct": num_correct,
        "num_parse_failures": num_parse_failures,
        "parse_failure_rate": num_parse_failures / num_examples if num_examples else 0.0,
        "gold_distribution": _ordered_nonzero_counts(gold_distribution, LETTERS),
        "prediction_distribution": _ordered_nonzero_counts(prediction_distribution, PREDICTION_COLUMNS),
        "confusion_matrix": confusion_matrix,
    }


def _answer_or_none(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "None":
        return None
    return value


def _ordered_nonzero_counts(counter: Counter, keys: list[str]) -> dict:
    return {key: counter[key] for key in keys if counter[key] > 0}


def _print_summary(predictions_path: Path, summary: dict, rows: list[dict], max_errors: int) -> None:
    num_examples = summary["num_examples"]
    print(f"predictions path: {predictions_path}")
    print(f"num examples: {num_examples}")
    print(f"accuracy: {summary['accuracy']}")
    print(
        "parse failures: "
        f"{summary['num_parse_failures']} / {num_examples} = {summary['parse_failure_rate']}"
    )

    print()
    print("gold distribution:")
    _print_distribution(summary["gold_distribution"])

    print()
    print("prediction distribution:")
    _print_distribution(summary["prediction_distribution"])

    print()
    print("confusion matrix:")
    _print_confusion_matrix(summary["confusion_matrix"])

    print()
    print("first errors:")
    _print_first_errors(rows, max_errors)


def _print_distribution(distribution: dict) -> None:
    if not distribution:
        print("(empty)")
        return
    for key, count in distribution.items():
        print(f"{key}: {count}")


def _print_confusion_matrix(confusion_matrix: dict) -> None:
    header = ["gold\\pred"] + PREDICTION_COLUMNS
    print(_format_table_row(header))
    for gold_letter in LETTERS:
        row = [gold_letter] + [str(confusion_matrix[gold_letter][pred]) for pred in PREDICTION_COLUMNS]
        print(_format_table_row(row))


def _format_table_row(values: list[str]) -> str:
    return "  ".join(str(value).rjust(9) for value in values)


def _print_first_errors(rows: list[dict], max_errors: int) -> None:
    errors = []
    for row in rows:
        gold_answer = _answer_or_none(row.get("gold_answer"))
        pred_answer = _answer_or_none(row.get("pred_answer"))
        if pred_answer != gold_answer:
            errors.append(row)
        if len(errors) >= max_errors:
            break

    if not errors:
        print("(none)")
        return

    for index, row in enumerate(errors, start=1):
        print(f"{index}. cache_id: {row.get('cache_id')}")
        print(f"   gold: {row.get('gold_answer')}")
        print(f"   pred: {row.get('pred_answer')}")
        print(f"   question: {row.get('question')}")
        print(f"   raw_output: {_preview_text(row.get('raw_output'))}")


def _preview_text(value, max_chars: int = 500) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


if __name__ == "__main__":
    main()
