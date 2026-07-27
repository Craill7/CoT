from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

from .common import final_answer


_V42_SEGMENTER_LOCK = threading.Lock()


class CotAnalyzer(Protocol):
    def analyze(
        self,
        record: Dict[str, Any],
        cot: str,
        run_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: ...


class V42Analyzer:
    """Adapter over the project's existing Compress V4.2 implementation."""

    def __init__(
        self,
        client: Any,
        marker_dict: Optional[Path] = None,
        max_atomic_segments: int = 160,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        parse_retries: int = 2,
    ):
        from compress.pipeline import joint_label as v4
        from compress.pipeline import joint_label_v42 as v42
        from .atomic_segmenter import segment_atomic_v2

        self.v4 = v4
        self.v42 = v42
        self.segment_atomic = segment_atomic_v2
        self.client = client
        self.marker_dict = v42.resolve_marker_dict(
            marker_dict or Path("legacy_v1/logic_markers_dict.json")
        )
        self.max_atomic_segments = max_atomic_segments
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_retries = parse_retries
        self.logger = logging.getLogger("pipeline_v2.v42")
        self.prompts = {
            "target_fallback": v4.read_prompt("target_fallback.txt"),
            "refine_blocks": (
                v4.read_prompt("joint_refine_blocks.txt")
                + "\n\nV2.1 exclusivity constraint: if an atomic segment is used as "
                "parent_segment_id by any internal_split block, that segment id MUST NOT "
                "appear in any segment_group. Internal splits for one parent must be "
                "non-overlapping exact substrings."
            ),
            "block_labeling": v4.read_prompt("joint_block_labeling.txt"),
            "dependency_graph": v4.read_prompt("joint_dependency_graph.txt"),
        }

    def analyze(
        self,
        record: Dict[str, Any],
        cot: str,
        run_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        payload = {
            "sample_id": run_id,
            "problem": record["question"],
            "answer": record.get("ground_truth", ""),
            "generations": [cot],
        }
        # V4.2 imports its segmenter as a module global. Swap it only while the
        # serial adapter call is active so the baseline module remains unchanged.
        with _V42_SEGMENTER_LOCK:
            original_segmenter = self.v42.segment_atomic
            self.v42.segment_atomic = self.segment_atomic
            try:
                return self.v42.process_one(
                    client=self.client,
                    prompts=self.prompts,
                    record=payload,
                    fallback_id=run_id,
                    marker_dict=self.marker_dict,
                    max_atomic_segments=self.max_atomic_segments,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    parse_retries=self.parse_retries,
                    logger=self.logger,
                )
            finally:
                self.v42.segment_atomic = original_segmenter


class MockAnalyzer:
    """Deterministic analyzer used by unit tests and no-GPU smoke runs."""

    def analyze(
        self,
        record: Dict[str, Any],
        cot: str,
        run_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", cot) if part.strip()]
        if not paragraphs:
            return None, {
                "sample_id": run_id,
                "stage": "mock_analysis",
                "error_type": "empty_cot",
                "error": "No CoT text",
            }
        blocks = [
            {
                "block_id": idx,
                "text": text,
                "type": "final" if idx == len(paragraphs) - 1 else "derivation",
                "source": "segment_group",
            }
            for idx, text in enumerate(paragraphs)
        ]
        target_text = record.get("ground_truth") or final_answer(cot) or "unclear or absent"
        contaminated = "MOCK_CONTAMINATED" in cot or "translate the above" in record["question"].lower()
        open_gap = "MOCK_GAP" in cot
        keep_ids = [] if contaminated else list(range(len(blocks)))
        labels = [
            {
                "block_id": block["block_id"],
                "type": "SUPPORTING" if block["block_id"] in keep_ids else "NON_SUPPORTING",
                "label_subtype": (
                    "FINAL_ANSWER"
                    if block["block_id"] == len(blocks) - 1
                    else "DERIVATION_SUPPORT"
                ),
                "keep": block["block_id"] in keep_ids,
                "supports": [0] if block["block_id"] in keep_ids else [],
                "reason": "mock",
            }
            for block in blocks
        ]
        coverage = 0.0 if contaminated else 1.0
        quality_flags = ["proof_gap_risk"] if open_gap or contaminated else []
        if contaminated:
            quality_flags.append("task_contaminated")
        return {
            "sample_id": run_id,
            "targets": [{"target_id": 0, "description": target_text, "type": "final_answer"}],
            "blocks": blocks,
            "block_labels": labels,
            "dependency_graph": [
                {"block": idx, "target": 0} for idx in keep_ids
            ],
            "supporting_blocks": keep_ids,
            "redundant_blocks": [idx for idx in range(len(blocks)) if idx not in keep_ids],
            "quality_flags": quality_flags,
            "engineering_flags": [],
            "statistics": {
                "num_blocks": len(blocks),
                "num_supporting": len(keep_ids),
                "target_coverage": coverage,
                "dependency_open": 1 if contaminated else 0,
                "dependency_open_rate": 1.0 if contaminated else 0.0,
            },
            "target_answer_alignment": "match" if record.get("ground_truth") else "unknown",
            "pipeline_version": "mock_v42",
            "raw_responses": {},
        }, None
