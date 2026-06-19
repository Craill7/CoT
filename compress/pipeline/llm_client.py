#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM API client with retry logic and structured JSON output for compress pipeline."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """Thin wrapper around OpenAI-compatible API with retry and JSON extraction."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2.5-32B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except requests.exceptions.Timeout as e:
                last_error = e
                wait = (attempt + 1) * 30
                print(f"  [LLM] timeout (attempt {attempt+1}/{self.max_retries}), waiting {wait}s...")
                time.sleep(wait)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait = (attempt + 1) * 10
                print(f"  [LLM] connection error (attempt {attempt+1}/{self.max_retries}), waiting {wait}s...")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                last_error = e
                if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
                    wait = (attempt + 1) * 30
                    print(f"  [LLM] rate limited (attempt {attempt+1}/{self.max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    wait = (attempt + 1) * 5
                    print(f"  [LLM] error (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(wait)

        raise RuntimeError(f"LLM request failed after {self.max_retries} retries. Last error: {last_error}")

    def extract_json(self, response: str) -> Dict[str, Any]:
        """Robust JSON extraction from LLM response.

        Handles:
        - Pure JSON
        - ```json fences
        - ``` fences without language tag
        - Leading/trailing text around JSON
        """
        # Try direct parse first
        response = response.strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` fence
        fence_patterns = [
            (r"```json\s*\n?(.*?)\n?```", "```json fence"),
            (r"```\s*\n?(.*?)\n?```", "``` fence"),
        ]
        for pattern, _ in fence_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Try find JSON object boundaries
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = response.find(start_char)
            end = response.rfind(end_char)
            if start != -1 and end > start:
                candidate = response[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try progressively trimming
                    depth = 0
                    cut = start
                    for i, ch in enumerate(candidate):
                        if ch == start_char:
                            depth += 1
                        elif ch == end_char:
                            depth -= 1
                            if depth == 0:
                                cut = i + 1
                                break
                    try:
                        return json.loads(candidate[:cut])
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Failed to extract valid JSON from LLM response:\n{response[:500]}")

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send chat request and return parsed JSON.

        Adds "Respond ONLY with valid JSON." to system prompt to encourage clean output.
        """
        messages = [
            {"role": "system", "content": system + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
            {"role": "user", "content": user},
        ]
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._request(messages, **kwargs)
        return self.extract_json(response)
