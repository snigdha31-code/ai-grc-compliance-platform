from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_string(text: str) -> str:
    """
    Extracts a JSON object string from:
    - raw JSON
    - ```json ... ``` fenced blocks
    - text that contains a JSON object somewhere inside
    """
    if not text:
        raise ValueError("Empty LLM output")

    s = text.strip()

    # Remove markdown fences if present
    # Examples:
    # ```json\n{...}\n```
    # ```\n{...}\n```
    if s.startswith("```"):
        # remove first fence line
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        # remove trailing fence
        s = re.sub(r"\s*```$", "", s).strip()

    # Now try to find the first {...} JSON object in the remaining text
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    return m.group(0).strip()


def parse_llm_json(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Returns (parsed_json, extracted_json_string).
    Raises ValueError if cannot parse.
    """
    json_str = extract_json_string(text)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON after extraction: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object/dict")
    return obj, json_str
