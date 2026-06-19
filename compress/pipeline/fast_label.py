#!/usr/bin/env python3
"""Ultra-fast compress: rule-based segmentation + minimal LLM labeling (SUPPORTING IDs only)."""

import json, sys, time, traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (extract_cot_text, extract_targets, segment_blocks,
                   sample_id_from_record, iter_jsonl, load_json, write_json)
from llm_client import LLMClient

FAST_SYSTEM = """You label reasoning blocks for CoT compression. Given blocks and final answer targets, output ONLY the list of SUPPORTING block IDs.

A block is SUPPORTING if removing it would affect the correctness or completeness of reaching a target. It must provide NON-REDUNDANT reasoning toward at least one target.

A block is NON_SUPPORTING if it is:
- Self-correction or mistake fixing (the model changing its mind)
- Dead-end reasoning not leading to any target
- Repeated verification of already-established results
- Narration, transition, or commentary
- Calculations not participating in reaching any target
- Redundant restatements of already-covered information

CRITICAL: Even if a calculation is mathematically correct, if it does not support any target, it is STILL NON_SUPPORTING. When uncertain, do NOT include (conservative compression)."""

def _build_user(blocks, targets):
    targets_str = json.dumps(targets, ensure_ascii=False)
    parts = []
    for b in blocks:
        parts.append("[{}] {}".format(b["block_id"], b["text"]))
    blocks_str = "\n\n".join(parts)
    return "TARGETS:\n{}\n\nBLOCKS:\n{}\n\nList SUPPORTING block IDs ONLY. JSON: {{\"supporting\": [0,2,5]}}".format(targets_str, blocks_str)

def process_one(client, record, fb_id="0"):
    sid = sample_id_from_record(record, fb_id)
    cot_text, field = extract_cot_text(record)
    if not cot_text:
        return None, {"sample_id": sid, "error": "empty cot", "type": "empty_cot"}
    try:
        targets = extract_targets(cot_text)
        if not targets:
            return None, {"sample_id": sid, "error": "no targets", "type": "empty_target"}
        blocks = segment_blocks(cot_text)
        if not blocks:
            return None, {"sample_id": sid, "error": "no blocks", "type": "empty_blocks"}

        # LLM call
        user_msg = _build_user(blocks, targets)
        llm_out = client.chat_json(FAST_SYSTEM, user_msg, temperature=0.0, max_tokens=1024)
        sup_ids = llm_out.get("supporting", [])
        block_ids = {int(b["block_id"]) for b in blocks}
        sup_ids = [int(x) for x in sup_ids if int(x) in block_ids]
        sup_set = set(sup_ids)

        # Build labels
        block_labels = []
        dep_graph = []
        all_target_ids = [int(t["target_id"]) for t in targets]
        for b in blocks:
            bid = int(b["block_id"])
            if bid in sup_set:
                block_labels.append({"block_id": bid, "type": "SUPPORTING", "supports": all_target_ids})
                for tid in all_target_ids:
                    dep_graph.append({"block": bid, "target": tid})
            else:
                block_labels.append({"block_id": bid, "type": "NON_SUPPORTING", "supports": []})

        red_ids = sorted(list(block_ids - sup_set))
        sup_ids_sorted = sorted(list(sup_set))

        # Stats
        nb = len(blocks); ns = len(sup_ids_sorted); nr = nb - ns
        cr_blocks = ns / nb if nb else 0.0
        total_tok = sum(len(b["text"].split()) for b in blocks)
        sup_tok = sum(len(b["text"].split()) for b in blocks if b["block_id"] in sup_set)
        cr_tokens = sup_tok / total_tok if total_tok else 0.0
        tgt_cov = 1.0 if sup_ids else 0.0

        stats = {
            "num_blocks": nb, "num_supporting": ns, "num_redundant": nr,
            "compression_ratio_blocks": round(cr_blocks, 4),
            "compression_ratio": round(cr_blocks, 4),
            "compression_ratio_tokens": round(cr_tokens, 4),
            "num_tokens_total": total_tok, "num_tokens_supporting": sup_tok,
            "target_coverage": round(tgt_cov, 4),
            "dependency_open": 0, "dependency_open_rate": 0.0,
            "num_targets": len(targets),
        }
        return {
            "sample_id": sid, "cot_field": field,
            "targets": targets, "blocks": blocks,
            "block_labels": block_labels, "dependency_graph": dep_graph,
            "redundant_blocks": red_ids, "supporting_blocks": sup_ids_sorted,
            "statistics": stats,
        }, None
    except Exception as e:
        return None, {"sample_id": sid, "error": str(e), "traceback": traceback.format_exc(), "type": "exception"}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    odir = args.output_dir; odir.mkdir(parents=True, exist_ok=True)
    comp_path = odir / "compressed.jsonl"
    fail_path = odir / "failure_cases.jsonl"
    stats_path = odir / "stats.json"
    ckpt_path = odir / "checkpoint.jsonl"

    client = LLMClient(base_url="http://localhost:8000/v1", model="Qwen2.5-32B-Instruct", timeout=300, max_retries=2)

    done_ids = set()
    if not args.no_resume and ckpt_path.exists():
        for _, rec in iter_jsonl(ckpt_path):
            done_ids.add(rec.get("sample_id", ""))
        if done_ids:
            print("[resume] skip {}".format(len(done_ids)))

    records = []
    if args.input.suffix == ".jsonl":
        for ln, rec in iter_jsonl(args.input):
            records.append((ln, rec))
    else:
        pld = load_json(args.input)
        for i, rec in enumerate(pld if isinstance(pld, list) else [pld]):
            records.append((i, rec))
    if args.limit:
        records = records[:args.limit]

    succ = fail = 0; t0 = time.time()
    for ln, rec in records:
        sid = sample_id_from_record(rec, str(ln))
        if sid in done_ids:
            continue
        result, error = process_one(client, rec, str(ln))
        done = succ + fail + 1
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(records) - len(done_ids) - done
        eta = "{:.0f}m".format(remaining / rate / 60) if rate > 0 and remaining > 0 else "?"

        if result:
            succ += 1
            with open(comp_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            with open(ckpt_path, "a") as f:
                f.write(json.dumps({"sample_id": sid, "status": "ok"}, ensure_ascii=False) + "\n")
            s = result["statistics"]
            print("[{}] OK {} | {}b/{}s cr={:.3f} | {:.2f}/s ETA={}".format(done, sid, s["num_blocks"], s["num_supporting"], s["compression_ratio_blocks"], rate, eta))
        else:
            fail += 1
            with open(fail_path, "a") as f:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")
            with open(ckpt_path, "a") as f:
                f.write(json.dumps({"sample_id": sid, "status": "fail"}, ensure_ascii=False) + "\n")
            print("[{}] FAIL {} | {} | ETA={}".format(done, sid, error.get("type", "?"), eta))

    # Aggregate stats
    results = []
    if comp_path.exists():
        for _, r in iter_jsonl(comp_path):
            results.append(r)
    failures = []
    if fail_path.exists():
        for _, f in iter_jsonl(fail_path):
            failures.append(f)

    cr = [r["statistics"]["compression_ratio_blocks"] for r in results]
    sr_vals = [r["statistics"]["num_supporting"] / max(r["statistics"]["num_blocks"], 1) for r in results]
    tgt_cov = [r["statistics"].get("target_coverage", 0) for r in results]
    _mn = lambda v: round(sum(v)/len(v),4) if v else 0
    _md = lambda v: round(sorted(v)[len(v)//2],4) if v else 0
    ft: Dict[str,int] = {}
    for f in failures:
        t = f.get("type","unknown"); ft[t] = ft.get(t,0) + 1
    agg = {
        "timestamp": datetime.now().isoformat(),
        "num_success": succ, "num_failed": fail,
        "elapsed_seconds": round(time.time()-t0,0),
        "mean_cr_blocks": _mn(cr), "median_cr_blocks": _md(cr),
        "mean_target_coverage": _mn(tgt_cov), "median_target_coverage": _md(tgt_cov),
        "mean_supporting_rate": _mn(sr_vals),
        "all_supporting": sum(1 for v in sr_vals if v>=0.99),
        "all_non_supporting": sum(1 for v in sr_vals if v<=0.01),
        "very_high_compression": sum(1 for v in cr if v<0.2),
        "very_low_compression": sum(1 for v in cr if v>0.9),
        "failure_breakdown": ft,
    }
    write_json(stats_path, agg, pretty=True)
    print("\nDONE: {} ok / {} fail in {:.0f}s".format(succ, fail, agg["elapsed_seconds"]))
    print("  CR: mean={} median={}".format(agg["mean_cr_blocks"], agg["median_cr_blocks"]))
    print("  coverage: mean={}".format(agg["mean_target_coverage"]))

if __name__ == "__main__":
    main()
