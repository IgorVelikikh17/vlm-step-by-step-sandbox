from __future__ import annotations

import re


def parse_answer_letter(text: str) -> str | None:
    if not text:
        return None

    patterns = [
        r"(?im)^\s*Answer\s*:\s*([A-E])\b",
        r"(?im)^\s*Final answer\s*:\s*([A-E])\b",
        r"(?i)\bThe answer is\s+([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None
