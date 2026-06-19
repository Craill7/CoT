#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze compression distribution patterns and flag anomalies."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def analyze(results_path: Path, failures_path: Path) -> Dict[str, Any]:
    results = load_jsonl(results_path) if results_path.exists() else []
    failures = load_jsonl(failures_path) if failures_path.exists() else []

    # Bin compression ratios
    cr_blocks = []
    cr_tokens = []
    block_counts = []
    supporting_rates = []

    for r in results:
        s = r.get("statistics", {})
        nb = s.get("num_blocks", 0)
        ns = s.get("num_supporting", 0)
        cr_blocks.append(ns / nb if nb > 0 else 0)
        cr_tokens.append(s.get("compression_ratio_tokens", 0))
        block_counts.append(nb)
        supporting_rates.append(ns / nb if nb > 0 else 0)

    # Histogram bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = Counter()
    for cr in cr_blocks:
        for i in range(len(bins) - 1):
            if bins[i] <= cr < bins[i + 1]:
                hist[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] += 1
                break
        else:
            hist["1.0"] += 1

    # Block count bins
    bc_bins = [0, 5, 10, 20, 30, 50, 100]
    bc_hist = Counter()
    for bc in block_counts:
        for i in range(len(bc_bins) - 1):
            if bc_bins[i] <= bc < bc_bins[i + 1]:
                bc_hist[f"{bc_bins[i]}-{bc_bins[i+1]}"] += 1
                break
        else:
            bc_hist["100+"] += 1

    # Anomaly flags
    anomalies = {
        "very_high_compression": [i for i, cr in enumerate(cr_blocks) if cr < 0.1],
        "very_low_compression": [i for i, cr in enumerate(cr_blocks) if cr > 0.95],
        "zero_supporting": [i for i, ns in enumerate([r["statistics"]["num_supporting"] for r in results]) if ns == 0],
        "full_supporting": [i for i, ns in enumerate([r["statistics"]["num_supporting"] for r in results])
                            if ns == results[i]["statistics"]["num_blocks"]],
    }

    # Failure analysis
    failure_types = Counter(f.get("type", "unknown") for f in failures)

    return {
        "distribution_histogram": dict(hist),
        "block_count_histogram": dict(bc_hist),
        "anomalies": {k: len(v) for k, v in anomalies.items()},
        "anomaly_sample_ids": {k: [results[i]["sample_id"] for i in v[:10]] for k, v in anomalies.items()},
        "failure_breakdown": dict(failure_types),
        "summary": {
            "total_samples": len(cr_blocks),
            "cr_blocks_mean": sum(cr_blocks) / len(cr_blocks) if cr_blocks else 0,
            "cr_blocks_median": sorted(cr_blocks)[len(cr_blocks)//2] if cr_blocks else 0,
            "samples_with_no_supporting": anomalies["zero_supporting"],
            "samples_with_all_supporting": anomalies["full_supporting"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--failures", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    failures_path = args.failures or (args.results.parent / "failure_cases.jsonl")
    analysis = analyze(args.results, failures_path)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    else:
        json.dump(analysis, sys.stdout, ensure_ascii=False, indent=2)
        print()

    # Print summary
    s = analysis["summary"]
    print(f"\n--- Distribution Analysis ---", file=sys.stderr)
    print(f"Samples: {s['total_samples']}", file=sys.stderr)
    print(f"CR blocks mean={s['cr_blocks_mean']:.3f} median={s['cr_blocks_median']:.3f}", file=sys.stderr)
    print(f"Histogram: {analysis['distribution_histogram']}", file=sys.stderr)
    print(f"Anomalies: {analysis['anomalies']}", file=sys.stderr)
    print(f"Failures: {analysis['failure_breakdown']}", file=sys.stderr)


if __name__ == "__main__":
    main()
