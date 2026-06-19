#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CoT Compression Pipeline — rule-based segmentation + LLM dependency labeling.

Architecture:
  Step 1: extract_targets()      — rule-based (\\boxed{} regex) — instant
  Step 2: segment_blocks()       — rule-based (paragraph/sentence) — instant
  Step 3: LLM dependency labeling — ONE call per sample, output=labels only
  Step 4: compute_statistics()

This minimizes LLM output tokens (labels only, no block text regurgitation).
Estimated throughput: ~15-20s per sample on Qwen2.5-32B.
"""

from __future__ import annotations

import argparse, json, sys, time, traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from utils import (
    compute_statistics, extract_cot_text, extract_targets, iter_jsonl,
    load_json, sample_id_from_record, segment_blocks, write_json,
)
from llm_client import LLMClient


# ── Prompt (dependency labeling only, blocks from rule-based segmentation) ──

LABELING_SYSTEM = """You are a dependency analyzer for CoT compression. Given reasoning blocks and final answer targets, label each block as SUPPORTING or NON_SUPPORTING.

## DEFINITIONS

**SUPPORTING**: Block provides reasoning that helps reach at least one target. If removed, the reasoning path to the target breaks. Must be non-redundant.

**NON_SUPPORTING**:
- Self-correction / mistake fixing
- Dead-end reasoning not leading to any target
- Repeated verification of already-proven results
- Narration / transition / commentary ("let me think", "now I will...")
- Calculations not participating in reaching any target
- Redundant restatements

**CRITICAL**: A mathematically correct calculation that does NOT support any target is STILL NON_SUPPORTING.

## RULES
- Every block gets exactly one label
- SUPPORTING blocks MUST have at least 1 target in supports[]
- NON_SUPPORTING blocks MUST have empty supports[]
- When uncertain → NON_SUPPORTING (conservative compression)
- dependency_graph lists every (block_id, target_id) support pair

## OUTPUT — ONLY this JSON, nothing else:
```json
{
  "block_labels": [
    {"block_id": 0, "type": "SUPPORTING", "supports": [0]},
    {"block_id": 1, "type": "NON_SUPPORTING", "supports": []}
  ],
  "dependency_graph": [
    {"block": 0, "target": 0}
  ]
}
```"""


def _build_labeling_user(blocks: List[Dict], targets: List[Dict]) -> str:
    """Build user message with blocks and targets."""
    targets_str = json.dumps(targets, ensure_ascii=False, indent=2)

    # Build compact blocks representation
    blocks_parts = []
    for b in blocks:
        text = b["text"]
        # Truncate very long blocks
        if len(text) > 1500:
            text = text[:750] + "\n[...]\n" + text[-750:]
        blocks_parts.append(f"[BLOCK {b['block_id']}]\n{text}")

    # If total is too long, truncate number of blocks
    max_blocks_chars = 12000
    blocks_str = "\n\n".join(blocks_parts)
    if len(blocks_str) > max_blocks_chars:
        # Keep first N blocks that fit
        kept = []
        total = 0
        for part in blocks_parts:
            if total + len(part) > max_blocks_chars:
                kept.append(f"\n[... {len(blocks_parts) - len(kept)} more blocks truncated ...]")
                break
            kept.append(part)
            total += len(part) + 2
        blocks_str = "\n\n".join(kept)

    return f"""## TARGETS
{targets_str}

## BLOCKS
{blocks_str}

---

Label each block. Output ONLY the JSON."""


# ── Core ──

def process_sample(
    client: LLMClient, record: Dict[str, Any], fallback_id: str = "0",
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process one sample. Returns (result, error)."""
    sid = sample_id_from_record(record, fallback_id)
    cot_text, field = extract_cot_text(record)

    if not cot_text:
        return None, {"index": fallback_id, "sample_id": sid, "error": "empty CoT", "type": "empty_cot"}

    try:
        # Step 1-2: Rule-based target extraction + block segmentation
        targets = extract_targets(cot_text)
        if not targets:
            return None, {"index": fallback_id, "sample_id": sid, "error": "no targets", "type": "empty_target"}

        blocks = segment_blocks(cot_text)
        if not blocks:
            return None, {"index": fallback_id, "sample_id": sid, "error": "no blocks", "type": "empty_blocks"}

        # Step 3: LLM dependency labeling
        user_msg = _build_labeling_user(blocks, targets)
        llm_result = client.chat_json(LABELING_SYSTEM, user_msg, temperature=0.0, max_tokens=4096)
        block_labels = llm_result.get("block_labels", [])
        dep_graph = llm_result.get("dependency_graph", [])

        # Fix & validate
        block_labels = _fix_labels(block_labels, blocks, targets)

        # Step 4: Statistics
        sup_ids = [lb["block_id"] for lb in block_labels if lb["type"] == "SUPPORTING"]
        red_ids = [lb["block_id"] for lb in block_labels if lb["type"] == "NON_SUPPORTING"]

        nb = len(blocks)
        ns = len(sup_ids)
        nr = nb - ns
        cr_blocks = ns / nb if nb else 0.0

        sup_set = set(sup_ids)
        total_tok = sum(len(b["text"].split()) for b in blocks)
        sup_tok = sum(len(b["text"].split()) for b in blocks if b["block_id"] in sup_set)
        cr_tokens = sup_tok / total_tok if total_tok else 0.0

        supported_targets = set()
        for lb in block_labels:
            for t in lb.get("supports", []):
                supported_targets.add(t)
        tgt_cov = len(supported_targets) / len(targets) if targets else 0.0

        labeled_ids = {lb["block_id"] for lb in block_labels}
        all_ids = {b["block_id"] for b in blocks}
        dep_open = len(all_ids - labeled_ids)

        stats = {
            "num_blocks": nb, "num_supporting": ns, "num_redundant": nr,
            "compression_ratio_blocks": round(cr_blocks, 4),
            "compression_ratio": round(cr_blocks, 4),
            "compression_ratio_tokens": round(cr_tokens, 4),
            "num_tokens_total": total_tok, "num_tokens_supporting": sup_tok,
            "target_coverage": round(tgt_cov, 4),
            "dependency_open": dep_open,
            "dependency_open_rate": round(dep_open / nb, 4) if nb else 0.0,
            "num_targets": len(targets),
        }

        return {
            "sample_id": sid, "cot_field": field,
            "targets": targets, "blocks": blocks,
            "block_labels": block_labels, "dependency_graph": dep_graph,
            "redundant_blocks": red_ids, "supporting_blocks": sup_ids,
            "statistics": stats,
        }, None

    except Exception as e:
        return None, {"index": fallback_id, "sample_id": sid,
                       "error": str(e), "traceback": traceback.format_exc(), "type": "exception"}


def _fix_labels(labels: List[Dict], blocks: List[Dict], targets: List[Dict]) -> List[Dict]:
    block_ids = {int(b["block_id"]) for b in blocks}
    target_ids = {int(t["target_id"]) for t in targets}
    fixed = []

    for lb in labels:
        bid = lb.get("block_id")
        if bid is None or int(bid) not in block_ids:
            continue
        bid = int(bid)
        ltype = lb.get("type", "NON_SUPPORTING")
        supports = [int(s) for s in lb.get("supports", []) if int(s) in target_ids]

        if ltype not in ("SUPPORTING", "NON_SUPPORTING"):
            ltype = "NON_SUPPORTING"
        if ltype == "NON_SUPPORTING" or not supports:
            ltype = "NON_SUPPORTING"
            supports = []

        fixed.append({"block_id": bid, "type": ltype, "supports": supports})

    labeled = {lb["block_id"] for lb in fixed}
    for bid in block_ids - labeled:
        fixed.append({"block_id": bid, "type": "NON_SUPPORTING", "supports": []})

    fixed.sort(key=lambda x: x["block_id"])
    return fixed


# ── Batch runner ──

def run_batch(
    input_path: Path, output_dir: Path,
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen2.5-32B-Instruct",
    limit: Optional[int] = None, resume: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_path = output_dir / "compressed.jsonl"
    fail_path = output_dir / "failure_cases.jsonl"
    stats_path = output_dir / "stats.json"
    ckpt_path = output_dir / "checkpoint.jsonl"

    client = LLMClient(base_url=base_url, model=model, timeout=300, max_retries=2)

    # Resume
    done_ids: set = set()
    if resume and ckpt_path.exists():
        for _, rec in iter_jsonl(ckpt_path):
            done_ids.add(rec.get("sample_id", ""))
        if done_ids:
            print(f"[resume] skipping {len(done_ids)} done samples")

    # Collect records
    records: List[Tuple[int, Dict]] = []
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        for ln, rec in iter_jsonl(input_path):
            records.append((ln, rec))
    else:
        p = load_json(input_path)
        for i, rec in enumerate(p if isinstance(p, list) else [p]):
            records.append((i, rec))

    if limit:
        records = records[:limit]

    succ = fail = 0
    t0 = time.time()

    for ln, rec in records:
        sid = sample_id_from_record(rec, str(ln))
        if sid in done_ids:
            continue

        result, error = process_sample(client, rec, fallback_id=str(ln))
        if result:
            succ += 1
            with open(comp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": sid, "status": "ok"}, ensure_ascii=False) + "\n")
        else:
            fail += 1
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": sid, "status": "fail", "type": error.get("type", "?")}, ensure_ascii=False) + "\n")

        done = succ + fail
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(records) - len(done_ids) - done
        eta = f"{remaining / rate / 60:.0f}m" if rate > 0 and remaining > 0 else "?"

        if done <= 10 or done % 100 == 0:
            s = result["statistics"] if result else {}
            print(f"[{done}] {sid} | b={s.get('num_blocks','?')}/{s.get('num_supporting','?')} "
                  f"cr={s.get('compression_ratio_blocks','?'):.4f} | {rate:.1f}/s ETA={eta}")

    return _finalize(stats_path, comp_path, fail_path, succ, fail, t0)


def _finalize(stats_path, comp_path, fail_path, succ, fail, t0):
    # Read results for aggregate
    results = []
    if comp_path.exists():
        for _, r in iter_jsonl(comp_path):
            results.append(r)
    failures = []
    if fail_path.exists():
        for _, f in iter_jsonl(fail_path):
            failures.append(f)

    cr = [r["statistics"]["compression_ratio_blocks"] for r in results] if results else []
    sr_vals = []
    tgt_cov = []
    for r in results:
        s = r["statistics"]
        sr_vals.append(s["num_supporting"] / max(s["num_blocks"], 1))
        tgt_cov.append(s.get("target_coverage", 0))

    _mn = lambda v: round(sum(v) / len(v), 4) if v else 0
    _md = lambda v: round(sorted(v)[len(v) // 2], 4) if v else 0
    _p = lambda v, pct: round(sorted(v)[min(int(len(v) * pct / 100), len(v) - 1)], 4) if v else 0

    ftypes: Dict[str, int] = {}
    for f in failures:
        t = f.get("type", "unknown")
        ftypes[t] = ftypes.get(t, 0) + 1

    agg = {
        "timestamp": datetime.now().isoformat(),
        "num_success": succ, "num_failed": fail,
        "elapsed_seconds": round(time.time() - t0, 0),
        "mean_cr_blocks": _mn(cr), "median_cr_blocks": _md(cr),
        "p10_cr_blocks": _p(cr, 10), "p90_cr_blocks": _p(cr, 90),
        "mean_target_coverage": _mn(tgt_cov), "median_target_coverage": _md(tgt_cov),
        "mean_supporting_rate": _mn(sr_vals),
        "all_supporting": sum(1 for v in sr_vals if v >= 0.99),
        "all_non_supporting": sum(1 for v in sr_vals if v <= 0.01),
        "very_high_compression": sum(1 for v in cr if v < 0.2),
        "very_low_compression": sum(1 for v in cr if v > 0.9),
        "failure_breakdown": ftypes,
    }
    write_json(stats_path, agg, pretty=True)

    print(f"\nDONE: {succ} ok / {fail} fail in {agg['elapsed_seconds']:.0f}s")
    print(f"  CR blocks: mean={agg['mean_cr_blocks']} median={agg['median_cr_blocks']}")
    print(f"  target_coverage: mean={agg['mean_target_coverage']}")
    print(f"  all-SUPPORTING: {agg['all_supporting']} all-NON: {agg['all_non_supporting']}")
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen2.5-32B-Instruct")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    run_batch(input_path=args.input, output_dir=args.output_dir,
              base_url=args.base_url, model=args.model,
              limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
