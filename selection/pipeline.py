"""
Select pipeline — Step 0 to Step 2: data preprocessing, pairwise judge, in-problem selection.

Usage:
    python -m select.pipeline \\
        --input /path/to/data_sample_3k.jsonl \\
        --output-dir /path/to/output/ \\
        --vllm-url http://127.0.0.1:8000/v1/chat/completions

Step 0: Filter N=1 problems, pick 2 median-length CoTs per problem.
Step 1: LLM pairwise judge (concurrent with checkpoint/resume).
Step 2: In-problem winner selection via scoring functions.
"""

import json
import os
import hashlib
import argparse
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .llm_pairwise_judge import call_judge
from .scoring import selection_score, pick_winner

# ── Constants ────────────────────────────────────────────
MAX_CHAR_LIMIT = None  # disabled — length does not correlate with dead loops
MIN_LEN_RATIO = 0.5           # exclude CoTs shorter than 0.5 * median_len
MAX_WORKERS = 4               # concurrent LLM requests
REQUEST_TIMEOUT = 300          # per-request timeout (seconds)


# ══════════════════════════════════════════════════════════
# Step 0: data preprocessing
# ══════════════════════════════════════════════════════════

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_verified_label(item: Dict[str, Any], idx: int) -> str:
    """Map is_reasoning_complete / correctness_math_verify → verified_correct or unverified_or_wrong.

    Handles both legacy field names.  True → verified_correct, False/missing → unverified_or_wrong.
    """
    for field in ("is_reasoning_complete", "correctness_math_verify"):
        arr = item.get(field)
        if arr is not None and isinstance(arr, list) and idx < len(arr):
            return "verified_correct" if arr[idx] else "unverified_or_wrong"
    return "unverified_or_wrong"


def pick_pair(
    generations: List[str],
) -> Optional[Tuple[int, int]]:
    """Pick the two generations closest to the median length.

    Returns (idx_a, idx_b) or None if fewer than 2 valid generations.
    """
    # Filter out super-long generations (dead-loop guard)
    valid = [
        (i, g) for i, g in enumerate(generations)
        if MAX_CHAR_LIMIT is None or len(g) <= MAX_CHAR_LIMIT
    ]
    if len(valid) < 2:
        return None

    # Sort by length
    valid.sort(key=lambda x: len(x[1]))
    lengths = [len(g) for _, g in valid]
    n = len(lengths)
    median_len = lengths[n // 2]

    # Exclude too-short generations
    valid = [
        (i, g) for i, g in valid
        if len(g) >= MIN_LEN_RATIO * median_len
    ]
    if len(valid) < 2:
        return None

    # Re-sort after filtering
    valid.sort(key=lambda x: len(x[1]))
    lengths = [len(g) for _, g in valid]
    n = len(lengths)
    median_len = lengths[n // 2]

    # Pick the two closest to median
    by_distance = sorted(valid, key=lambda x: abs(len(x[1]) - median_len))
    idx_a, _ = by_distance[0]
    idx_b, _ = by_distance[1]
    return idx_a, idx_b


def problem_hash(problem_text: str) -> str:
    return hashlib.md5(problem_text.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════
# Step 1 + 2: judge + select (per-item worker)
# ══════════════════════════════════════════════════════════

def process_one(
    item: Dict[str, Any],
    vllm_url: str,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Run Step 0→1→2 for a single problem. Returns the selected result or None on skip/failure."""
    problem = item["problem"]
    generations = item.get("generations", [])

    # ── Step 0: preprocess ──
    if len(generations) < 2:
        return None  # skip N=1

    pair = pick_pair(generations)
    if pair is None:
        return None

    idx_a, idx_b = pair
    cot_a = generations[idx_a]
    cot_b = generations[idx_b]
    label_a = get_verified_label(item, idx_a)
    label_b = get_verified_label(item, idx_b)

    # Map verified_correct → True for the judge prompt
    verified_a = label_a == "verified_correct"
    verified_b = label_b == "verified_correct"

    # ── Step 1: LLM judge ──
    judge_result, error = call_judge(
        problem=problem,
        cot_a=cot_a,
        cot_b=cot_b,
        verified_a=verified_a,
        verified_b=verified_b,
        model=model_name,
        vllm_url=vllm_url,
    )
    if judge_result is None:
        # Log failure but don't crash the whole batch
        return {
            "problem_hash": problem_hash(problem),
            "status": "judge_failed",
            "error": error,
            "problem": problem,
        }

    # ── Step 2: in-problem selection ──
    score_a = selection_score(
        quality_tags=judge_result["cot_a"].get("quality_tags", []),
        issues=judge_result["cot_a"].get("issues", []),
        answer_label=label_a,
    )
    score_b = selection_score(
        quality_tags=judge_result["cot_b"].get("quality_tags", []),
        issues=judge_result["cot_b"].get("issues", []),
        answer_label=label_b,
    )

    winner = pick_winner(
        score_a=score_a,
        score_b=score_b,
        len_a=len(cot_a),
        len_b=len(cot_b),
    )
    winner_idx = idx_a if winner == "a" else idx_b
    winner_cot = cot_a if winner == "a" else cot_b
    winner_label = label_a if winner == "a" else label_b
    winner_score = score_a if winner == "a" else score_b

    problem_difficulty = judge_result.get("problem_difficulty", "Medium")

    return {
        "problem_hash": problem_hash(problem),
        "status": "ok",
        "instruction": problem,
        "output": winner_cot,
        "solution": item.get("solution", ""),
        "answer": item.get("answer", ""),
        "problem_difficulty": problem_difficulty,
        "winner": winner,
        "winner_score": winner_score,
        "score_a": score_a,
        "score_b": score_b,
        "verified_label": winner_label,
        "metadata": {
            "source_idx": winner_idx,
            "cot_a_idx": idx_a,
            "cot_b_idx": idx_b,
            "cot_a_length": len(cot_a),
            "cot_b_length": len(cot_b),
            "judge_winner": judge_result.get("winner"),
            "judge_reason": judge_result.get("brief_reason", ""),
            "cot_a_tags": judge_result["cot_a"].get("quality_tags", []),
            "cot_a_issues": judge_result["cot_a"].get("issues", []),
            "cot_b_tags": judge_result["cot_b"].get("quality_tags", []),
            "cot_b_issues": judge_result["cot_b"].get("issues", []),
            "label_a": label_a,
            "label_b": label_b,
        },
    }


# ══════════════════════════════════════════════════════════
# Batch runner
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Select pipeline Step 0–2")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "selected.jsonl")
    failure_path = os.path.join(args.output_dir, "judge_failures.jsonl")

    # ── Load data ──
    print(f"Loading: {args.input}")
    all_items = load_jsonl(args.input)
    print(f"  {len(all_items)} problems loaded")

    # ── Resume / checkpoint ──
    done_hashes = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_hashes.add(json.loads(line)["problem_hash"])
                    except (KeyError, json.JSONDecodeError):
                        pass
        print(f"  Resuming: {len(done_hashes)} already processed")

    # Also skip previously failed items
    if args.resume and os.path.exists(failure_path):
        with open(failure_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_hashes.add(json.loads(line)["problem_hash"])
                    except (KeyError, json.JSONDecodeError):
                        pass

    pending = [it for it in all_items if problem_hash(it["problem"]) not in done_hashes]
    print(f"  {len(pending)} remaining to process")

    if not pending:
        print("All done — nothing to process.")
        return

    # ── Process concurrently ──
    ok_count = 0
    fail_count = 0
    skipped_count = 0

    with open(output_path, "a", encoding="utf-8") as f_out, \
         open(failure_path, "a", encoding="utf-8") as f_fail:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_one, item, args.vllm_url, args.model): item
                for item in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
                result = future.result()
                if result is None:
                    skipped_count += 1
                elif result.get("status") == "judge_failed":
                    f_fail.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_fail.flush()
                    fail_count += 1
                else:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    ok_count += 1

    # ── Summary ──
    print(f"\nDone.  OK: {ok_count}  Failed: {fail_count}  Skipped (N<2): {skipped_count}")
    print(f"Selected: {output_path}")
    print(f"Failures: {failure_path}")


if __name__ == "__main__":
    main()
