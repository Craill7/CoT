from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .common import JsonLLM
from .diagnosis import REPAIRABLE_ACTIONS


def supporting_text(analysis: Dict[str, Any]) -> str:
    block_map = {int(block["block_id"]): block for block in analysis.get("blocks", [])}
    texts = [
        str(block_map[block_id].get("text", "")).strip()
        for block_id in analysis.get("supporting_blocks", [])
        if block_id in block_map and str(block_map[block_id].get("text", "")).strip()
    ]
    return "\n\n".join(texts)


def sanitize_rewrite(
    raw: Dict[str, Any],
    analysis: Dict[str, Any],
    allowed_modes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    mode = str(raw.get("rewrite_mode", ""))
    if mode not in REPAIRABLE_ACTIONS:
        raise ValueError(f"invalid rewrite_mode: {mode}")
    if allowed_modes is not None and mode not in allowed_modes:
        raise ValueError(f"rewrite_mode {mode} is not authorized by the diagnosis")
    block_ids = {int(block["block_id"]) for block in analysis.get("blocks", [])}
    edits: List[Dict[str, Any]] = []
    replaced: Set[int] = set()
    for idx, edit in enumerate(raw.get("edits", []) if isinstance(raw.get("edits"), list) else []):
        if not isinstance(edit, dict):
            continue
        replace_blocks: List[int] = []
        for value in edit.get("replace_blocks", []) if isinstance(edit.get("replace_blocks"), list) else []:
            try:
                block_id = int(value)
            except (TypeError, ValueError):
                continue
            if block_id in block_ids and block_id not in replace_blocks:
                replace_blocks.append(block_id)
        if replaced.intersection(replace_blocks):
            raise ValueError("rewrite edits overlap")
        replaced.update(replace_blocks)
        def valid_anchor(value: Any) -> Optional[int]:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed in block_ids else None

        after = valid_anchor(edit.get("after_block"))
        before = valid_anchor(edit.get("before_block"))
        new_text = str(edit.get("new_text", "")).strip()
        if not new_text:
            raise ValueError(f"rewrite edit {idx} has empty new_text")
        if not replace_blocks and after is None and before is None:
            raise ValueError(f"rewrite edit {idx} has no authorized anchor")
        edits.append(
            {
                "after_block": after,
                "before_block": before,
                "replace_blocks": replace_blocks,
                "new_text": new_text,
                "reason": str(edit.get("reason", ""))[:500],
            }
        )
    if not edits:
        raise ValueError("rewriter returned no usable edits")
    return {"rewrite_mode": mode, "edits": edits}


def apply_rewrite(analysis: Dict[str, Any], rewrite: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    block_map = {int(block["block_id"]): block for block in analysis.get("blocks", [])}
    base_ids = [
        int(block_id)
        for block_id in analysis.get("supporting_blocks", [])
        if int(block_id) in block_map
    ]
    if not base_ids:
        base_ids = sorted(block_map)
    replacement_by_start: Dict[int, Dict[str, Any]] = {}
    replaced_ids: Set[int] = set()
    insert_before: Dict[int, List[Dict[str, Any]]] = {}
    insert_after: Dict[int, List[Dict[str, Any]]] = {}
    for edit in rewrite["edits"]:
        replace_blocks = [bid for bid in edit["replace_blocks"] if bid in base_ids]
        if replace_blocks:
            ordered_positions = sorted(base_ids.index(bid) for bid in replace_blocks)
            expected = list(range(ordered_positions[0], ordered_positions[-1] + 1))
            if ordered_positions != expected:
                raise ValueError("replace_blocks must be contiguous in the kept CoT")
            start_id = base_ids[ordered_positions[0]]
            replacement_by_start[start_id] = edit
            replaced_ids.update(replace_blocks)
        elif edit["before_block"] in base_ids:
            insert_before.setdefault(edit["before_block"], []).append(edit)
        elif edit["after_block"] in base_ids:
            insert_after.setdefault(edit["after_block"], []).append(edit)
        else:
            raise ValueError("rewrite insert has no retained block anchor")

    output: List[str] = []
    applied: List[Dict[str, Any]] = []
    for block_id in base_ids:
        for edit in insert_before.get(block_id, []):
            output.append(edit["new_text"])
            applied.append(edit)
        if block_id in replacement_by_start:
            edit = replacement_by_start[block_id]
            output.append(edit["new_text"])
            applied.append(edit)
        elif block_id not in replaced_ids:
            output.append(str(block_map[block_id].get("text", "")).strip())
        for edit in insert_after.get(block_id, []):
            output.append(edit["new_text"])
            applied.append(edit)
    final_cot = "\n\n".join(part for part in output if part.strip())
    return final_cot, {
        "scope_valid": True,
        "base_block_ids": base_ids,
        "applied_edits": applied,
        "rewrite_mode": rewrite["rewrite_mode"],
    }


class ConstrainedRewriter:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def rewrite(
        self,
        record: Dict[str, Any],
        analysis: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        blocks = [
            {
                "block_id": int(block["block_id"]),
                "text": str(block.get("text", ""))[:2200],
                "kept": int(block["block_id"]) in set(analysis.get("supporting_blocks", [])),
            }
            for block in analysis.get("blocks", [])
        ]
        response = self.llm.chat_json(
            system=(
                "Repair only the diagnosed gaps in a mathematical Chain-of-Thought. Return "
                "structured edits, never a freely regenerated full answer. Preserve the final "
                "answer and every block outside authorized edits."
            ),
            user=(
                f"QUESTION:\n{record['question']}\n\n"
                f"GROUND_TRUTH:\n{record.get('ground_truth') or '(unknown)'}\n\n"
                f"BLOCKS:\n{blocks}\n\n"
                f"GAPS:\n{diagnosis.get('gaps', [])}\n\n"
                "Return rewrite_mode (insert_patch, rewrite_span, or format_fix) and edits. "
                "Each edit has after_block, before_block, replace_blocks, new_text, and reason. "
                "Use the smallest possible edit and do not change the answer."
            ),
            temperature=0.1,
            max_tokens=4096,
        )
        allowed_modes = {
            str(gap.get("suggested_action"))
            for gap in diagnosis.get("gaps", [])
            if str(gap.get("suggested_action")) in REPAIRABLE_ACTIONS
        }
        return sanitize_rewrite(response, analysis, allowed_modes or None)
