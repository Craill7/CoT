#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-file entry point for CoT dependency-oriented compression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from .utils import compress_sample, extract_cot_text, iter_jsonl, load_json, write_json
except ImportError:
    from utils import compress_sample, extract_cot_text, iter_jsonl, load_json, write_json


def load_record(input_path: Path, record_index: int) -> Dict[str, Any]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        for idx, (_, record) in enumerate(iter_jsonl(input_path)):
            if idx == record_index:
                return record
        raise IndexError(f"record_index {record_index} is out of range for {input_path}")

    payload = load_json(input_path)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        try:
            record = payload[record_index]
        except IndexError as exc:
            raise IndexError(f"record_index {record_index} is out of range for {input_path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {record_index} is not a JSON object.")
        return record
    raise ValueError("Input JSON must be an object or a list of objects.")


def record_from_text(text: str, sample_id: str) -> Dict[str, Any]:
    return {"sample_id": sample_id, "cot": text}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress a CoT by keeping only blocks that support extracted targets."
    )
    parser.add_argument("--input", type=Path, help="Input .json, .jsonl, or plain text file.")
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    parser.add_argument("--record-index", type=int, default=0, help="Zero-based index for JSONL/list inputs.")
    parser.add_argument("--sample-id", default="0", help="Fallback sample id for raw text inputs.")
    parser.add_argument("--text", default=None, help="Raw CoT text. Used when --input is omitted.")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of indented JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.text is None and args.input is None:
        parser.error("Provide --input or --text.")

    if args.text is not None:
        record = record_from_text(args.text, args.sample_id)
        fallback_id = args.sample_id
    else:
        input_path = args.input.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if input_path.suffix.lower() in {".txt", ".md"}:
            record = record_from_text(input_path.read_text(encoding="utf-8", errors="replace"), args.sample_id)
        else:
            record = load_record(input_path, args.record_index)
        fallback_id = str(args.record_index)

    cot_text, field = extract_cot_text(record)
    if not cot_text:
        raise ValueError("No usable CoT text found in the input record.")

    result = compress_sample(record, fallback_id=fallback_id)
    if args.output:
        write_json(args.output.resolve(), result, pretty=not args.compact)
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=None if args.compact else 2)
        sys.stdout.write("\n")

    print(
        "Compressed sample "
        f"{result['sample_id']} from field={field or 'raw'}: "
        f"{result['statistics']['num_supporting']}/"
        f"{result['statistics']['num_blocks']} supporting blocks.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
