"""
Select pipeline — Step 3: global clustering + Top-K selection.

Usage:
    python -m select.clustering \\
        --input results/selected.jsonl \\
        --output-dir results/ \\
        --model-path /ky200t/models/Qwen/Qwen2.5-32B-Instruct \\
        --num-clusters 40 \\
        --k-ratio 0.8

Loads the winner CoTs from Step 0–2, computes IFD + embeddings, clusters, and
selects the top-K samples per cluster ranked by global_score (with IFD tiebreaker).
"""

import os
import json
import argparse
import hashlib
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM

from .scoring import global_score as compute_global_score


# ══════════════════════════════════════════════════════════
# IFD + embedding
# ══════════════════════════════════════════════════════════

def compute_ifd_and_embedding(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    instruction: str,
    output: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> tuple:
    """Compute IFD (PPL) and mean-pooled last-hidden-state embedding in one pass."""
    full_text = f"{instruction}\n{output}"
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
        loss = outputs.loss
        # Mean-pool the last hidden state
        last_hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_dim)
        embedding = last_hidden.float().mean(dim=1).cpu()  # (1, hidden_dim)

    ifd = torch.exp(loss).item()
    return ifd, embedding


# ══════════════════════════════════════════════════════════
# Clustering + Top-K
# ══════════════════════════════════════════════════════════

def cluster_and_select(
    data_list: List[Dict[str, Any]],
    embeddings: List[torch.Tensor],
    num_clusters: int,
    k_ratio: float,
) -> tuple:
    """KMeans cluster → per-cluster global_score ranking → Top-K split.

    Returns (high_quality, low_quality).
    """
    features = torch.cat(embeddings, dim=0).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(features)

    for idx, label in enumerate(kmeans.labels_):
        data_list[idx]["_cluster"] = int(label)

    # Group by cluster
    clusters: Dict[int, List[Dict]] = {}
    for item in data_list:
        cid = item["_cluster"]
        clusters.setdefault(cid, []).append(item)

    high, low = [], []
    for cid, items in clusters.items():
        # Sort: higher global_score → higher IFD → shorter output
        items.sort(
            key=lambda x: (
                x.get("global_score", 0.0),
                x.get("IFD_Score", 0.0),
                -(len(x.get("output", ""))),
            ),
            reverse=True,
        )
        target_count = max(1, int(len(items) * k_ratio))
        high.extend(items[:target_count])
        low.extend(items[target_count:])

    return high, low


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Select pipeline Step 3: clustering")
    parser.add_argument("--input", required=True, help="selected.jsonl from Step 0–2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True, help="HF model path for IFD + embeddings")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-clusters", type=int, default=40)
    parser.add_argument("--k-ratio", type=float, default=0.8)
    parser.add_argument("--force-recompute", action="store_true", help="Ignore embedding cache")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "embeddings_cache.pt")

    # ── Load data ──
    data_list = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("status") != "ok":
                continue
            data_list.append(item)

    print(f"Loaded {len(data_list)} valid entries")

    # ── Compute or load embeddings ──
    if os.path.exists(cache_path) and not args.force_recompute:
        print(f"Loading cached embeddings from {cache_path}")
        checkpoint = torch.load(cache_path)
        data_list = checkpoint["data_list"]
        embeddings = checkpoint["embeddings"]
        print(f"  {len(data_list)} entries restored")
    else:
        print(f"Loading model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        embeddings = []
        for item in tqdm(data_list, desc="Computing IFD+emb"):
            ifd, emb = compute_ifd_and_embedding(
                tokenizer, model,
                instruction=item["instruction"],
                output=item["output"],
                max_length=args.max_length,
            )
            item["IFD_Score"] = ifd
            embeddings.append(emb)

        print("Saving cache ...")
        torch.save({"data_list": data_list, "embeddings": embeddings}, cache_path)

        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ── Compute global_score ──
    for item in data_list:
        sel_score = item.get("winner_score", 0.0)
        diff = item.get("problem_difficulty", "Medium")
        item["global_score"] = compute_global_score(sel_score, diff)

    # ── Cluster & select ──
    print(f"Clustering (k={args.num_clusters}, ratio={args.k_ratio}) ...")
    high, low = cluster_and_select(data_list, embeddings, args.num_clusters, args.k_ratio)

    # ── Clean internal keys before saving ──
    for item in data_list:
        item.pop("_cluster", None)

    high_path = os.path.join(args.output_dir, "high_quality.json")
    low_path = os.path.join(args.output_dir, "low_quality.json")

    with open(high_path, "w", encoding="utf-8") as f:
        json.dump(high, f, ensure_ascii=False, indent=2)
    with open(low_path, "w", encoding="utf-8") as f:
        json.dump(low, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    high_scores = [it["global_score"] for it in high]
    low_scores = [it["global_score"] for it in low] if low else [0.0]
    high_ifds = [it.get("IFD_Score", 0) for it in high]

    print(f"\nClustering done.")
    print(f"  High quality: {len(high)}  (avg global_score: {np.mean(high_scores):.2f}, avg IFD: {np.mean(high_ifds):.2f})")
    print(f"  Low quality:  {len(low)}  (avg global_score: {np.mean(low_scores):.2f})")
    print(f"  High: {high_path}")
    print(f"  Low:  {low_path}")


if __name__ == "__main__":
    main()
