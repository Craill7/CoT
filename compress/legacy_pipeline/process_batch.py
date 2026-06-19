#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch processing entry point for CoT compression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .utils import compress_sample, extract_cot_text, iter_jsonl, load_json, write_json, write_jsonl
except ImportError:
    from utils import compress_sample, extract_cot_text, iter_jsonl, load_json, write_json, write_jsonl


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for _, record in iter_jsonl(path):
            yield record
        return

    if suffix in {".txt", ".md"}:
        yield {"sample_id": path.stem, "cot": path.read_text(encoding="utf-8", errors="replace")}
        return

    payload = load_json(path)
    if isinstance(payload, list):
        for idx, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {idx} is not a JSON object.")
            yield record
        return
    if isinstance(payload, dict):
        yield payload
        return
    raise ValueError("Input must be JSON object, JSON list, JSONL, or text.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compress CoT samples into schema-compatible JSON objects."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl, .json, .txt, or .md file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .jsonl or .json file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of usable records to process.")
    parser.add_argument("--skip-errors", action="store_true", help="Continue after per-record errors.")
    parser.add_argument("--compact", action="store_true", help="Use compact JSON for .json output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed = 0

    for idx, record in enumerate(iter_records(input_path)):
        if args.limit is not None and processed >= args.limit:
            break

        cot_text, _ = extract_cot_text(record)
        if not cot_text:
            message = "No usable CoT text found."
            if not args.skip_errors:
                raise ValueError(f"Record {idx}: {message}")
            errors.append({"index": idx, "error": message})
            continue

        try:
            results.append(compress_sample(record, fallback_id=str(idx)))
            processed += 1
        except Exception as exc:
            if not args.skip_errors:
                raise
            errors.append({"index": idx, "error": str(exc)})

    if output_path.suffix.lower() == ".jsonl":
        write_jsonl(output_path, results)
    else:
        payload: Any = results
        if errors:
            payload = {"results": results, "errors": errors}
        write_json(output_path, payload, pretty=not args.compact)

    print(
        f"Processed {processed} samples -> {output_path}"
        + (f" ({len(errors)} errors skipped)" if errors else ""),
        file=sys.stderr,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
