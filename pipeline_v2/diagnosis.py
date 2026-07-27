from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .common import JsonLLM, is_contaminated


VALID_RELATIONS = {"supports", "derives", "validates"}
VALID_GAP_TYPES = {
    "missing_premise",
    "missing_derivation",
    "incomplete_case",
    "missing_bound",
    "missing_uniqueness",
    "contradiction",
    "format_answer",
    "contaminated",
    "uncertain",
}
VALID_ACTIONS = {"insert_patch", "rewrite_span", "format_fix", "reject"}
REPAIRABLE_ACTIONS = {"insert_patch", "rewrite_span", "format_fix"}
UNLOCATED_REJECT_TYPES = {"contaminated", "contradiction", "uncertain"}


def _ints(value: Any, valid: Set[int]) -> List[int]:
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed in valid and parsed not in out:
            out.append(parsed)
    return out


def _has_cycle(edges: Sequence[Dict[str, Any]]) -> bool:
    graph: Dict[int, List[int]] = {}
    for edge in edges:
        graph.setdefault(int(edge["from_block"]), []).append(int(edge["to_block"]))
    visiting: Set[int] = set()
    visited: Set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def sanitize_diagnosis(
    raw: Dict[str, Any],
    baseline: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    block_ids = {int(block["block_id"]) for block in baseline.get("blocks", [])}
    target_ids = {int(target["target_id"]) for target in baseline.get("targets", [])}
    dependencies: List[Dict[str, Any]] = []
    seen = set()
    for edge in raw.get("block_dependencies", []) if isinstance(raw.get("block_dependencies"), list) else []:
        if not isinstance(edge, dict):
            continue
        try:
            source = int(edge.get("from_block"))
            target = int(edge.get("to_block"))
        except (TypeError, ValueError):
            continue
        relation = str(edge.get("relation", "supports"))
        key = (source, target, relation)
        if source in block_ids and target in block_ids and source != target and relation in VALID_RELATIONS and key not in seen:
            dependencies.append(
                {"from_block": source, "to_block": target, "relation": relation}
            )
            seen.add(key)

    gaps: List[Dict[str, Any]] = []
    for idx, gap in enumerate(raw.get("gaps", []) if isinstance(raw.get("gaps"), list) else []):
        if not isinstance(gap, dict):
            continue
        gap_type = str(gap.get("gap_type", "uncertain"))
        action = str(gap.get("suggested_action", "reject"))
        if gap_type not in VALID_GAP_TYPES:
            gap_type = "uncertain"
        if action not in VALID_ACTIONS:
            action = "reject"
        try:
            confidence = max(0.0, min(1.0, float(gap.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        gaps.append(
            {
                "gap_id": str(gap.get("gap_id") or f"g{idx}"),
                "gap_type": gap_type,
                "upstream_blocks": _ints(gap.get("upstream_blocks"), block_ids),
                "downstream_blocks": _ints(gap.get("downstream_blocks"), block_ids),
                "target_ids": _ints(gap.get("target_ids"), target_ids),
                "missing_claim": str(gap.get("missing_claim", ""))[:1000],
                "suggested_action": action,
                "confidence": confidence,
            }
        )

    stats = baseline.get("statistics", {})
    coverage = float(stats.get("target_coverage", 0.0) or 0.0)
    dependency_open = int(stats.get("dependency_open", 0) or 0)
    target_texts = [str(item.get("description", "")) for item in baseline.get("targets", [])]
    block_texts = [str(item.get("text", "")) for item in baseline.get("blocks", [])]
    baseline_flags = {str(flag) for flag in baseline.get("quality_flags", [])}
    supporting = {int(value) for value in baseline.get("supporting_blocks", [])}
    target_edges = [
        edge
        for edge in baseline.get("dependency_graph", [])
        if isinstance(edge, dict)
        and edge.get("block") is not None
        and edge.get("target") is not None
        and int(edge["block"]) in supporting
        and int(edge["target"]) in target_ids
    ]
    supported_targets = {int(edge["target"]) for edge in target_edges}
    missing_targets = sorted(target_ids - supported_targets)
    if missing_targets and not any(gap["target_ids"] for gap in gaps):
        gaps.append(
            {
                "gap_id": f"g{len(gaps)}",
                "gap_type": "missing_derivation",
                "upstream_blocks": [],
                "downstream_blocks": [],
                "target_ids": missing_targets,
                "missing_claim": "One or more targets have no retained supporting block.",
                "suggested_action": "insert_patch",
                "confidence": 1.0,
            }
        )

    incoming = {
        int(edge["to_block"])
        for edge in dependencies
        if int(edge["from_block"]) in supporting and int(edge["to_block"]) in supporting
    }
    terminal_support = {int(edge["block"]) for edge in target_edges}
    if (
        len(supporting) > 1
        and terminal_support
        and not baseline_flags.intersection({"proof_gap_risk", "task_contaminated", "contaminated"})
        and all(block_id not in incoming for block_id in terminal_support - {min(supporting)})
        and not gaps
    ):
        gaps.append(
            {
                "gap_id": f"g{len(gaps)}",
                "gap_type": "missing_derivation",
                "upstream_blocks": [min(supporting)],
                "downstream_blocks": sorted(terminal_support),
                "target_ids": sorted(target_ids),
                "missing_claim": "The target-support block has no continuous incoming reasoning path.",
                "suggested_action": "insert_patch",
                "confidence": 0.8,
            }
        )

    label_by_block = {
        int(label["block_id"]): str(label.get("label_subtype", "")).upper()
        for label in baseline.get("block_labels", [])
        if isinstance(label, dict) and label.get("block_id") is not None
    }
    if (
        len(supporting) == 1
        and next(iter(supporting)) in terminal_support
        and label_by_block.get(next(iter(supporting))) == "FINAL_ANSWER"
        and not gaps
    ):
        only_block = next(iter(supporting))
        gaps.append(
            {
                "gap_id": f"g{len(gaps)}",
                "gap_type": "missing_derivation",
                "upstream_blocks": [],
                "downstream_blocks": [only_block],
                "target_ids": sorted(target_ids),
                "missing_claim": "Only a final answer remains; the necessary derivation is absent.",
                "suggested_action": "insert_patch",
                "confidence": 0.9,
            }
        )

    contaminated = (
        is_contaminated(question, *target_texts, *block_texts)
        or "contaminated" in baseline_flags
        or "task_contaminated" in baseline_flags
    )
    unclear_target = not target_ids or any(
        token in text.lower()
        for text in target_texts
        for token in ("unclear or absent", "final answer is unclear", "is unclear or absent")
    )
    if contaminated or unclear_target:
        gaps.append(
            {
                "gap_id": f"g{len(gaps)}",
                "gap_type": "contaminated",
                "upstream_blocks": [],
                "downstream_blocks": [],
                "target_ids": sorted(target_ids),
                "missing_claim": "The task or extracted target is contaminated or unclear.",
                "suggested_action": "reject",
                "confidence": 1.0,
            }
        )
    elif (coverage < 1.0 or dependency_open > 0) and not gaps:
        gaps.append(
            {
                "gap_id": "g0",
                "gap_type": "missing_derivation",
                "upstream_blocks": [],
                "downstream_blocks": [],
                "target_ids": sorted(target_ids),
                "missing_claim": "At least one target has no complete supporting path.",
                "suggested_action": "insert_patch",
                "confidence": 1.0,
            }
        )
    elif "proof_gap_risk" in baseline_flags and not gaps:
        gaps.append(
            {
                "gap_id": "g0",
                "gap_type": "uncertain",
                "upstream_blocks": [],
                "downstream_blocks": [],
                "target_ids": sorted(target_ids),
                "missing_claim": "The baseline flagged a proof gap, but no reliable location was found.",
                "suggested_action": "reject",
                "confidence": 1.0,
            }
        )

    if _has_cycle(dependencies):
        gaps.append(
            {
                "gap_id": f"g{len(gaps)}",
                "gap_type": "uncertain",
                "upstream_blocks": [],
                "downstream_blocks": [],
                "target_ids": sorted(target_ids),
                "missing_claim": "Block dependency graph contains a cycle.",
                "suggested_action": "reject",
                "confidence": 1.0,
            }
        )

    requested_state = str(raw.get("dependency_state", "uncertain"))
    if contaminated or any(gap["suggested_action"] == "reject" for gap in gaps):
        state = "uncertain"
    elif gaps or coverage < 1.0 or dependency_open > 0:
        state = "open"
    elif requested_state == "closed":
        state = "closed"
    else:
        state = "uncertain"

    return {
        "block_dependencies": dependencies,
        "dependency_graph": list(baseline.get("dependency_graph", [])),
        "dependency_state": state,
        "gaps": gaps,
        "repairable": bool(gaps) and all(gap["suggested_action"] in REPAIRABLE_ACTIONS for gap in gaps),
        "baseline_quality_flags": list(baseline.get("quality_flags", [])),
        "diagnosis_reason": str(raw.get("diagnosis_reason", ""))[:1000],
    }


def validate_diagnosis_response(
    raw: Dict[str, Any],
    baseline: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    state = str(raw.get("dependency_state", ""))
    if state not in {"closed", "open", "uncertain"}:
        errors.append("dependency_state must be closed, open, or uncertain")
    gaps = raw.get("gaps")
    if not isinstance(gaps, list):
        errors.append("gaps must be a list")
        return errors
    if state in {"open", "uncertain"} and not gaps:
        errors.append(f"{state} diagnosis requires at least one structured gap")

    valid_blocks = {int(block["block_id"]) for block in baseline.get("blocks", [])}
    valid_targets = {int(target["target_id"]) for target in baseline.get("targets", [])}
    for index, gap in enumerate(gaps):
        prefix = f"gap[{index}]"
        if not isinstance(gap, dict):
            errors.append(f"{prefix} must be an object")
            continue
        gap_type = str(gap.get("gap_type", ""))
        action = str(gap.get("suggested_action", ""))
        if gap_type not in VALID_GAP_TYPES:
            errors.append(f"{prefix}.gap_type is invalid")
        if action not in VALID_ACTIONS:
            errors.append(f"{prefix}.suggested_action is invalid")
        if not str(gap.get("missing_claim", "")).strip():
            errors.append(f"{prefix}.missing_claim is required")
        try:
            confidence = float(gap.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not 0.0 < confidence <= 1.0:
            errors.append(f"{prefix}.confidence must be in (0,1]")

        upstream = _ints(gap.get("upstream_blocks"), valid_blocks)
        downstream = _ints(gap.get("downstream_blocks"), valid_blocks)
        targets = _ints(gap.get("target_ids"), valid_targets)
        localized_type = gap_type not in UNLOCATED_REJECT_TYPES
        if localized_type and not (upstream or downstream):
            errors.append(f"{prefix} requires an upstream or downstream block anchor")
        if localized_type and not targets:
            errors.append(f"{prefix} requires at least one valid target_id")
        if action in REPAIRABLE_ACTIONS and not (upstream or downstream):
            errors.append(f"{prefix} repair action requires a block anchor")
    return errors


class DependencyDiagnoser:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def diagnose(
        self,
        record: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        compact_blocks = [
            {
                "block_id": int(block["block_id"]),
                "text": str(block.get("text", ""))[:1800],
                "type": block.get("type"),
                "kept": int(block["block_id"]) in set(baseline.get("supporting_blocks", [])),
            }
            for block in baseline.get("blocks", [])
        ]
        system = (
            "Inspect a compressed mathematical reasoning graph. Build directed block-to-block "
            "dependencies and identify concrete logical gaps. A proof_gap must have a structured "
            "gap record. Do not rewrite the CoT."
        )
        base_user = (
                f"QUESTION:\n{record['question']}\n\n"
                f"GROUND_TRUTH:\n{record.get('ground_truth') or '(unknown)'}\n\n"
                f"TARGETS:\n{baseline.get('targets', [])}\n\n"
                f"BLOCKS:\n{compact_blocks}\n\n"
                f"BLOCK_TO_TARGET_EDGES:\n{baseline.get('dependency_graph', [])}\n\n"
                f"BASELINE_FLAGS:\n{baseline.get('quality_flags', [])}\n\n"
                "Return exactly this JSON shape:\n"
                "{\n"
                '  "block_dependencies": [{"from_block": 0, "to_block": 1, '
                '"relation": "supports|derives|validates"}],\n'
                '  "dependency_state": "closed|open|uncertain",\n'
                '  "gaps": [{\n'
                '    "gap_id": "g0",\n'
                '    "gap_type": "missing_premise|missing_derivation|incomplete_case|'
                'missing_bound|missing_uniqueness|contradiction|format_answer|'
                'contaminated|uncertain",\n'
                '    "upstream_blocks": [0],\n'
                '    "downstream_blocks": [1],\n'
                '    "target_ids": [0],\n'
                '    "missing_claim": "concrete missing logical claim",\n'
                '    "suggested_action": "insert_patch|rewrite_span|format_fix|reject",\n'
                '    "confidence": 0.8\n'
                "  }],\n"
                '  "diagnosis_reason": "short explanation"\n'
                "}\n"
                "For dependency_state=closed, gaps MUST be []. For every open repairable gap, "
                "use the exact key suggested_action, choose exactly one allowed action, include "
                "at least one valid upstream_blocks or downstream_blocks anchor, at least one "
                "valid target_id, a non-empty missing_claim, and confidence in (0,1]. "
                "Do not emit aliases such as repair, patch, keep, remove, or rewrite."
        )
        attempts: List[Dict[str, Any]] = []
        validation_errors: List[str] = []
        previous: Dict[str, Any] = {}
        for attempt in range(2):
            retry_context = ""
            if attempt:
                retry_context = (
                    f"\n\nPREVIOUS_RESPONSE:\n{previous}\n\n"
                    f"VALIDATION_ERRORS:\n{validation_errors}\n\n"
                    "Return a corrected diagnosis JSON using the exact schema above. Every gap "
                    "must use the literal key suggested_action with exactly one of "
                    "insert_patch, rewrite_span, format_fix, reject. Do not omit valid block "
                    "anchors, target_ids, missing_claim, or confidence."
                )
            response = self.llm.chat_json(
                system=system,
                user=base_user + retry_context,
                temperature=0.0,
                max_tokens=4096,
            )
            validation_errors = validate_diagnosis_response(response, baseline)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "validation_errors": list(validation_errors),
                    "raw_response": response,
                }
            )
            if not validation_errors:
                result = sanitize_diagnosis(response, baseline, record["question"])
                result["diagnosis_attempts"] = attempts
                return result
            previous = response

        fallback = {
            "block_dependencies": previous.get("block_dependencies", []),
            "dependency_state": "uncertain",
            "gaps": [
                {
                    "gap_id": "diagnosis_invalid",
                    "gap_type": "uncertain",
                    "upstream_blocks": [],
                    "downstream_blocks": [],
                    "target_ids": [
                        int(target["target_id"])
                        for target in baseline.get("targets", [])
                    ],
                    "missing_claim": (
                        "Diagnosis remained structurally invalid after one retry: "
                        + "; ".join(validation_errors)
                    ),
                    "suggested_action": "reject",
                    "confidence": 1.0,
                }
            ],
            "diagnosis_reason": "diagnosis_invalid_after_retry",
        }
        result = sanitize_diagnosis(fallback, baseline, record["question"])
        result["diagnosis_attempts"] = attempts
        return result
