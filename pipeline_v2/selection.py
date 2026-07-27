from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .common import JsonLLM, answer_gate


class CandidateSelector(Protocol):
    name: str

    def select(self, record: Dict[str, Any]) -> Dict[str, Any]: ...


class GlobalScorer(Protocol):
    name: str

    def score(self, record: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]: ...


class Clusterer(Protocol):
    name: str

    def cluster(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...


class IdentitySelector:
    name = "identity_selector"

    def select(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if len(record.get("candidates", [])) != 1:
            raise ValueError("identity selector requires exactly one candidate")
        cot = record["candidates"][0]
        return {
            "selection_mode": "identity",
            "selected_index": 0,
            "selected_cot": cot,
            "score_components": {},
        }


class PairwiseSelector:
    name = "pairwise_selector"

    def __init__(self, vllm_url: str, model: str):
        self.vllm_url = vllm_url
        self.model = model

    def select(self, record: Dict[str, Any]) -> Dict[str, Any]:
        from select_v2.llm_pairwise_judge import call_judge
        from select_v2.pipeline import get_verified_label, pick_pair
        from select_v2.scoring import pick_winner, selection_score

        candidates = record.get("candidates", [])
        pair = pick_pair(candidates)
        if pair is None:
            raise ValueError("pairwise selector found fewer than two eligible candidates")
        idx_a, idx_b = pair
        source = record.get("source_record", {})
        label_a = get_verified_label(source, idx_a)
        label_b = get_verified_label(source, idx_b)
        result, error = call_judge(
            problem=record["question"],
            cot_a=candidates[idx_a],
            cot_b=candidates[idx_b],
            verified_a=label_a == "verified_correct",
            verified_b=label_b == "verified_correct",
            model=self.model,
            vllm_url=self.vllm_url,
        )
        if result is None:
            raise RuntimeError(error or "pairwise judge failed")

        score_a = selection_score(
            result["cot_a"].get("quality_tags", []),
            result["cot_a"].get("issues", []),
            label_a,
        )
        score_b = selection_score(
            result["cot_b"].get("quality_tags", []),
            result["cot_b"].get("issues", []),
            label_b,
        )
        winner = pick_winner(
            score_a,
            score_b,
            len_a=len(candidates[idx_a]),
            len_b=len(candidates[idx_b]),
        )
        selected_index = idx_a if winner == "a" else idx_b
        return {
            "selection_mode": "pairwise",
            "selected_index": selected_index,
            "selected_cot": candidates[selected_index],
            "score_components": {
                "winner_score": score_a if winner == "a" else score_b,
                "score_a": score_a,
                "score_b": score_b,
                "problem_difficulty": result.get("problem_difficulty", "Medium"),
                "judge_winner": result.get("winner"),
                "judge_reason": result.get("brief_reason", ""),
                "candidate_indices": [idx_a, idx_b],
            },
        }


class MockPairwiseSelector:
    """Deterministic multi-candidate selector for no-model integration tests."""

    name = "mock_pairwise_selector"

    def select(self, record: Dict[str, Any]) -> Dict[str, Any]:
        candidates = record.get("candidates", [])
        if len(candidates) < 2:
            raise ValueError("mock pairwise selector requires at least two candidates")
        selected_index = max(range(len(candidates)), key=lambda index: len(candidates[index]))
        return {
            "selection_mode": "pairwise",
            "selected_index": selected_index,
            "selected_cot": candidates[selected_index],
            "score_components": {
                "winner_score": 2.0,
                "problem_difficulty": "Medium",
                "judge_winner": "mock",
                "candidate_indices": list(range(len(candidates))),
            },
        }


class SelectionRouter:
    def __init__(
        self,
        identity: CandidateSelector,
        pairwise: CandidateSelector,
        mode: str = "auto",
    ):
        if mode not in {"auto", "identity", "pairwise"}:
            raise ValueError(f"unsupported select mode: {mode}")
        self.identity = identity
        self.pairwise = pairwise
        self.mode = mode

    def select(self, record: Dict[str, Any]) -> Dict[str, Any]:
        count = len(record.get("candidates", []))
        if count == 0:
            raise ValueError("record has no candidate CoT")
        mode = self.mode
        if mode == "auto":
            mode = "identity" if count == 1 else "pairwise"
        if mode == "identity":
            if count != 1:
                raise ValueError("forced identity mode requires one candidate per record")
            return self.identity.select(record)
        if count < 2:
            raise ValueError("forced pairwise mode requires at least two candidates")
        return self.pairwise.select(record)


class ReuseOrAbsoluteGlobalScorer:
    name = "absolute_global_scorer"

    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def score(self, record: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
        from select_v2.pipeline import get_verified_label
        from select_v2.scoring import global_score, selection_score

        components = dict(selection.get("score_components", {}))
        if "winner_score" in components:
            base_score = float(components["winner_score"])
            difficulty = str(components.get("problem_difficulty", "Medium"))
            score_source = "pairwise_reuse"
        else:
            response = self.llm.chat_json(
                system=(
                    "Score one mathematical Chain-of-Thought for data selection. "
                    "Do not solve the problem again. Return quality_tags, issues, "
                    "problem_difficulty, and a short reason."
                ),
                user=(
                    f"QUESTION:\n{record['question']}\n\n"
                    f"COT:\n{selection['selected_cot']}\n\n"
                    "Allowed quality_tags: Deep, Present, Exploratory, Cohesive, Concise.\n"
                    "Allowed issues: forced_verification_alignment, hallucinated_context, "
                    "unnecessary_cross_validation, redundant_verification, redundant_restatement.\n"
                    "problem_difficulty must be Hard, Medium, or Easy."
                ),
                temperature=0.0,
                max_tokens=768,
            )
            source = record.get("source_record", {})
            selected_index = int(selection["selected_index"])
            label = get_verified_label(source, selected_index)
            base_score = selection_score(
                list(response.get("quality_tags", [])),
                list(response.get("issues", [])),
                label,
            )
            difficulty = str(response.get("problem_difficulty", "Medium"))
            components.update(
                {
                    "winner_score": base_score,
                    "problem_difficulty": difficulty,
                    "quality_tags": list(response.get("quality_tags", [])),
                    "issues": list(response.get("issues", [])),
                    "absolute_reason": str(response.get("reason", ""))[:500],
                }
            )
            score_source = "absolute_llm"

        result = dict(selection)
        result["score_components"] = components
        result["global_score"] = global_score(base_score, difficulty)
        result["global_score_source"] = score_source
        result["answer_gate"] = answer_gate(
            record,
            int(selection["selected_index"]),
            str(selection["selected_cot"]),
        )
        return result


class MockGlobalScorer:
    """Deterministic scorer that still emits the production score schema."""

    name = "mock_global_scorer"

    def score(self, record: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
        cot = str(selection["selected_cot"])
        score = 0.5 if ("MOCK_GAP" in cot or "MOCK_CONTAMINATED" in cot) else 3.0
        result = dict(selection)
        result["score_components"] = {
            **selection.get("score_components", {}),
            "winner_score": score,
            "problem_difficulty": "Medium",
            "quality_tags": ["Cohesive"],
            "issues": ["mock_low_quality"] if score < 1.0 else [],
        }
        result["global_score"] = score
        result["global_score_source"] = "mock"
        result["answer_gate"] = answer_gate(
            record,
            int(selection["selected_index"]),
            cot,
        )
        return result


@dataclass
class ScoreRatioClusterer:
    """Cheap deterministic plugin for tests and framework smoke runs."""

    k_ratio: float = 0.8
    name: str = "score_ratio_clusterer"

    def cluster(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ordered = sorted(
            records,
            key=lambda item: (float(item.get("global_score", 0.0)), -len(item.get("selected_cot", ""))),
            reverse=True,
        )
        if not ordered:
            return [], []
        high_count = max(1, int(len(ordered) * self.k_ratio))
        return ordered[:high_count], ordered[high_count:]


@dataclass
class IfdKMeansClusterer:
    model_path: str
    num_clusters: int = 40
    k_ratio: float = 0.8
    max_length: int = 2048
    device: str = "cuda:0"
    name: str = "ifd_kmeans_clusterer"

    def cluster(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not records:
            return [], []
        import gc

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from select_v2.clustering import cluster_and_select, compute_ifd_and_embedding

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        model.eval()
        embeddings = []
        prepared: List[Dict[str, Any]] = []
        for record in records:
            item = dict(record)
            ifd, embedding = compute_ifd_and_embedding(
                tokenizer,
                model,
                instruction=item["question"],
                output=item["selected_cot"],
                max_length=self.max_length,
                device=self.device,
            )
            item["IFD_Score"] = ifd
            item["output"] = item["selected_cot"]
            embeddings.append(embedding)
            prepared.append(item)

        cluster_count = max(1, min(self.num_clusters, len(prepared)))
        high, low = cluster_and_select(prepared, embeddings, cluster_count, self.k_ratio)
        for item in high + low:
            item.pop("_cluster", None)
            item.pop("output", None)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return high, low
