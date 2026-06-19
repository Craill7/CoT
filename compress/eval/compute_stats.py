#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute aggregate statistics from compressed output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_stats(results_path: Path, failures_path: Path) -> Dict[str, Any]:
    results = load_jsonl(results_path) if results_path.exists() else []
    failures = load_jsonl(failures_path) if failures_path.exists() else []

    if not results:
        return {"error": "No results", "num_success": 0, "num_failures": len(failures)}

    cr_blocks = []
    cr_tokens = []
    target_coverages = []
    dep_open_rates = []
    num_blocks_list = []
    num_supporting_list = []

    for r in results:
        s = r.get("statistics", {})
        cr_blocks.append(s.get("compression_ratio_blocks", 0))
        cr_tokens.append(s.get("compression_ratio_tokens", 0))
        target_coverages.append(s.get("target_coverage", 0))
        dep_open_rates.append(s.get("dependency_open_rate", 0))
        num_blocks_list.append(s.get("num_blocks", 0))
        num_supporting_list.append(s.get("num_supporting", 0))

    def _mean(vals): return sum(vals) / len(vals)
    def _median(vals):
        svals = sorted(vals)
        n = len(svals)
        return svals[n // 2] if n % 2 else (svals[n // 2 - 1] + svals[n // 2]) / 2
    def _pct(vals, p):
        svals = sorted(vals)
        return svals[min(int(len(svals) * p / 100), len(svals) - 1)]

    failure_types: Dict[str, int] = {}
    for f in failures:
        etype = f.get("type", "unknown")
        failure_types[etype] = failure_types.get(etype, 0) + 1

    return {
        "num_success": len(results),
        "num_failures": len(failures),
        "compression_ratio_blocks": {
            "mean": round(_mean(cr_blocks), 4),
            "median": round(_median(cr_blocks), 4),
            "p10": round(_pct(cr_blocks, 10), 4),
            "p90": round(_pct(cr_blocks, 90), 4),
        },
        "compression_ratio_tokens": {
            "mean": round(_mean(cr_tokens), 4),
            "median": round(_median(cr_tokens), 4),
        },
        "target_coverage": {
            "mean": round(_mean(target_coverages), 4),
            "median": round(_median(target_coverages), 4),
        },
        "dependency_open_rate": {
            "mean": round(_mean(dep_open_rates), 4),
        },
        "num_blocks": {
            "mean": round(_mean(num_blocks_list), 1),
            "median": round(_median(num_blocks_list), 1),
        },
        "num_supporting": {
            "mean": round(_mean(num_supporting_list), 1),
            "median": round(_median(num_supporting_list), 1),
        },
        "all_supporting": sum(1 for sr in [s.get("num_supporting", 0) / max(s.get("num_blocks", 1), 1) for s in [r.get("statistics", {}) for r in results]] if sr >= 0.99),
        "all_non_supporting": sum(1 for sr in [s.get("num_supporting", 0) / max(s.get("num_blocks", 1), 1) for s in [r.get("statistics", {}) for r in results]] if sr <= 0.01),
        "failure_breakdown": failure_types,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="compressed.jsonl")
    parser.add_argument("--failures", type=Path, default=None, help="failure_cases.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output stats JSON")
    args = parser.parse_args()

    failures_path = args.failures or (args.results.parent / "failure_cases.jsonl")
    stats = compute_stats(args.results, failures_path)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    else:
        json.dump(stats, sys.stdout, ensure_ascii=False, indent=2)
        print()

    # Quick summary
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Success: {stats.get('num_success', 0)} | Failures: {stats.get('num_failures', 0)}", file=sys.stderr)
    cr = stats.get("compression_ratio_blocks", {})
    print(f"Compression (blocks): mean={cr.get('mean')}, median={cr.get('median')}", file=sys.stderr)
    tc = stats.get("target_coverage", {})
    print(f"Target coverage: mean={tc.get('mean')}", file=sys.stderr)
    print(f"All-SUPPORTING: {stats.get('all_supporting', 0)} | All-NON-SUPPORTING: {stats.get('all_non_supporting', 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
