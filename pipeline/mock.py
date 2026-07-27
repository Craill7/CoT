from __future__ import annotations

import re
from typing import Any, Dict, Optional


class MockJsonLLM:
    """Predictable JSON LLM used for framework integration tests."""

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        lowered = system.lower()
        if "score one mathematical" in lowered:
            return {
                "quality_tags": ["Cohesive", "Concise"],
                "issues": [],
                "problem_difficulty": "Medium",
                "reason": "mock absolute score",
            }
        if "build directed block-to-block" in lowered:
            block_ids = [int(value) for value in re.findall(r"'block_id':\s*(\d+)", user)]
            block_ids = list(dict.fromkeys(block_ids))
            dependencies = [
                {"from_block": left, "to_block": right, "relation": "derives"}
                for left, right in zip(block_ids, block_ids[1:])
            ]
            if "proof_gap_risk" in user:
                upstream = block_ids[:1]
                downstream = block_ids[1:2]
                return {
                    "block_dependencies": dependencies,
                    "dependency_state": "open",
                    "gaps": [
                        {
                            "gap_id": "g0",
                            "gap_type": "missing_derivation",
                            "upstream_blocks": upstream,
                            "downstream_blocks": downstream,
                            "target_ids": [0],
                            "missing_claim": "A local derivation step is missing.",
                            "suggested_action": "rewrite_span",
                            "confidence": 0.9,
                        }
                    ],
                    "diagnosis_reason": "mock proof gap",
                }
            return {
                "block_dependencies": dependencies,
                "dependency_state": "closed",
                "gaps": [],
                "diagnosis_reason": "mock closed graph",
            }
        if "repair only the diagnosed gaps" in lowered:
            return {
                "rewrite_mode": "rewrite_span",
                "edits": [
                    {
                        "after_block": None,
                        "before_block": None,
                        "replace_blocks": [0],
                        "new_text": "The missing derivation is supplied explicitly and preserves the stated answer.",
                        "reason": "mock local repair",
                    }
                ],
            }
        if "verify a refined mathematical" in lowered:
            return {
                "verdict": "pass",
                "checks": {
                    "answer": "pass",
                    "format": "pass",
                    "dependency": "pass",
                    "faithfulness": "pass",
                    "redundancy": "pass",
                },
                "failure_codes": [],
                "repairable": False,
            }
        raise ValueError(f"MockJsonLLM received unknown prompt: {system[:80]}")
