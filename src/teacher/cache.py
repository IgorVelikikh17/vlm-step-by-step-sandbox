from __future__ import annotations

from pathlib import Path

from io_utils import read_jsonl
from parsing import parse_answer_letter


def read_teacher_cache(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def validate_teacher_cache_rows(rows: list[dict]) -> dict:
    stats = {
        "num_rows": len(rows),
        "num_missing_cache_id": 0,
        "num_missing_teacher_answer": 0,
        "num_parse_failures": 0,
    }

    for row in rows:
        if not row.get("cache_id"):
            stats["num_missing_cache_id"] += 1
        if not row.get("teacher_answer"):
            stats["num_missing_teacher_answer"] += 1
        if parse_answer_letter(row.get("teacher_raw_output", "")) is None:
            stats["num_parse_failures"] += 1

    return stats
