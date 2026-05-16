from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for relative in ["src/evaluation", "src/teacher", "src/utils"]:
    sys.path.insert(0, str(ROOT / relative))

from cache import read_teacher_cache, validate_teacher_cache_rows  # noqa: E402
from io_utils import resolve_project_path  # noqa: E402
from parsing import parse_answer_letter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ScienceQA teacher cache JSONL file.")
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/processed/teacher_cache/scienceqa_mock_train_debug.jsonl",
    )
    parser.add_argument("--preview_count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = resolve_project_path(ROOT, args.cache_path)
    rows = read_teacher_cache(cache_path)
    stats = validate_teacher_cache_rows(rows)

    print(f"cache path: {cache_path}")
    print("validation stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    preview_count = min(args.preview_count, len(rows))
    for index in range(preview_count):
        row = rows[index]
        print()
        print(f"--- cache row {index} ---")
        print(f"cache_id: {row.get('cache_id')}")
        print(f"source_index: {row.get('source_index')}")
        print(f"teacher_type: {row.get('teacher_type')}")
        print(f"teacher_model_name: {row.get('teacher_model_name')}")
        print(f"question: {row.get('question')}")
        print("choices:")
        for choice_index, choice in enumerate(row.get("choices", [])):
            print(f"  {chr(65 + choice_index)}. {choice}")
        print(f"gold_answer: {row.get('gold_answer')}")
        print(f"teacher_answer: {row.get('teacher_answer')}")
        print(f"parsed_answer_from_raw_output: {parse_answer_letter(row.get('teacher_raw_output', ''))}")
        print("teacher_reasoning preview:")
        print(_preview_text(row.get("teacher_reasoning", "")))


def _preview_text(text: str, max_chars: int = 300) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


if __name__ == "__main__":
    main()
