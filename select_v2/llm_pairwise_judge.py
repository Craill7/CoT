"""
LLM Pairwise CoT Judge — single-call evaluation of two CoTs for the same problem.

Evaluates: problem difficulty, quality tags, redundancy issues, winner selection.
Does NOT judge mathematical correctness (verified_correct comes from math_verify).
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests

from .config import VALID_TAGS, VALID_ISSUES, VALID_DIFFICULTY

# ── Config ──────────────────────────────────────────────
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "pairwise_cot_judge.txt"


def load_prompt_template() -> str:
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


def build_prompt(problem: str, cot_a: str, cot_b: str,
                 verified_a: bool, verified_b: bool) -> str:
    """Fill the prompt template with actual data."""
    template = load_prompt_template()
    return template.format(
        problem=problem,
        verified_a=str(verified_a),
        verified_b=str(verified_b),
        cot_a=cot_a,
        cot_b=cot_b,
    )


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM output. Handles markdown fences and LaTeX artifacts."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    for fence in ["```json", "```"]:
        if fence in text:
            parts = text.split(fence)
            if len(parts) >= 2:
                inner = parts[1].split("```")[0].strip()
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    continue

    # Fallback: try to find JSON-like braces
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i, ch in enumerate(text[brace_start:], brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def validate_result(result: Dict[str, Any]) -> Optional[str]:
    """Validate the LLM output structure. Returns error message or None if valid."""

    if "problem_difficulty" not in result or result["problem_difficulty"] not in VALID_DIFFICULTY:
        return f"Invalid/missing problem_difficulty: {result.get('problem_difficulty')}"

    for key in ("cot_a", "cot_b"):
        if key not in result:
            return f"Missing {key}"
        cot = result[key]
        tags = set(cot.get("quality_tags", []))
        if not tags.issubset(VALID_TAGS):
            return f"{key}.quality_tags invalid: {cot.get('quality_tags')}"
        issues = set(cot.get("issues", []))
        if not issues.issubset(VALID_ISSUES):
            return f"{key}.issues invalid: {cot.get('issues')}"

    if result.get("winner") not in ("a", "b"):
        return f"Invalid winner: {result.get('winner')}"

    return None  # Valid


def call_judge(problem: str, cot_a: str, cot_b: str,
               verified_a: bool, verified_b: bool,
               model: str = MODEL_NAME,
               vllm_url: str = VLLM_URL,
               max_tokens: int = 1024,
               temperature: float = 0.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call the LLM to judge two CoTs for the same problem.

    Returns (result_dict, error_msg).
    On success: (result_dict, None)
    On failure after all retries: (None, error_msg)
    """
    prompt = build_prompt(problem, cot_a, cot_b, verified_a, verified_b)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(vllm_url, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            result = extract_json(content)
            if result is None:
                last_error = f"JSON parse failed. Raw: {content[:300]}"
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                continue

            validation_err = validate_result(result)
            if validation_err:
                last_error = f"Validation failed: {validation_err}"
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                continue

            return result, None

        except requests.exceptions.Timeout:
            last_error = f"Request timeout ({REQUEST_TIMEOUT}s)"
            if attempt < MAX_RETRIES:
                time.sleep(3)
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(5)
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(2)

    return None, last_error
