#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggressive marker-based atomic segmentation for V4.2.

Atomic segments are local, exact spans from the original CoT. They are not the
final compression blocks; V4.2 asks the LLM to group or split these spans, and
the pipeline reconstructs final block text from the original spans only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MARKER_CATEGORIES = {
    "setup",
    "derivation",
    "verification",
    "correction",
    "final",
    "narration",
    "conclusion",
    "unknown",
}

PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class MarkerSpec:
    marker: str
    category: str
    priority: str


@dataclass(frozen=True)
class Boundary:
    pos: int
    marker: Optional[str]
    category: str
    priority: str


DEFAULT_HIGH_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("Therefore, the final answer", "final"),
    ("Thus, the final answer", "final"),
    ("Final Answer", "final"),
    ("The answer is", "final"),
    ("Answer:", "final"),
    ("\\boxed{", "final"),
    ("Wait, no", "correction"),
    ("But this is wrong", "correction"),
    ("This doesn't help", "correction"),
    ("I made a mistake", "correction"),
    ("Let me check", "verification"),
    ("Let's check", "verification"),
    ("Let's verify", "verification"),
    ("Verification", "verification"),
    ("Check:", "verification"),
    ("Actually", "correction"),
    ("Wait", "correction"),
    ("Verify", "verification"),
)

DEFAULT_MEDIUM_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("Alternatively", "derivation"),
    ("Another way", "derivation"),
    ("Suppose", "setup"),
    ("Assume", "setup"),
    ("Case 1", "derivation"),
    ("Case 2", "derivation"),
    ("Case 3", "derivation"),
    ("First", "setup"),
    ("Second", "derivation"),
    ("Finally", "conclusion"),
    ("Therefore", "conclusion"),
    ("Thus", "conclusion"),
    ("Hence", "conclusion"),
    ("Next", "derivation"),
    ("Now", "derivation"),
    ("So", "conclusion"),
)

DEFAULT_LOW_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("Let's tackle this problem step by step", "narration"),
    ("Let's solve this step by step", "narration"),
    ("Let me think", "narration"),
    ("This is confusing", "narration"),
    ("time constraints", "narration"),
    ("I'm not sure", "narration"),
    ("Let's see", "narration"),
    ("It seems", "narration"),
    ("I think", "narration"),
    ("Maybe", "narration"),
    ("Okay", "narration"),
    ("Hmm", "narration"),
)

GENERIC_MEDIUM_MARKERS = {"therefore", "thus", "hence", "so", "now", "then", "next"}


def _dedup_markers(markers: Iterable[MarkerSpec]) -> List[MarkerSpec]:
    best: Dict[str, MarkerSpec] = {}
    for marker in markers:
        key = marker.marker.lower()
        old = best.get(key)
        if old is None or PRIORITY_RANK[marker.priority] > PRIORITY_RANK[old.priority]:
            best[key] = marker
    return sorted(best.values(), key=lambda item: len(item.marker), reverse=True)


def load_marker_specs(marker_dict: Optional[Path] = None) -> List[MarkerSpec]:
    markers: List[MarkerSpec] = []
    for marker, category in DEFAULT_HIGH_MARKERS:
        markers.append(MarkerSpec(marker, category, "high"))
    for marker, category in DEFAULT_MEDIUM_MARKERS:
        markers.append(MarkerSpec(marker, category, "medium"))
    for marker, category in DEFAULT_LOW_MARKERS:
        markers.append(MarkerSpec(marker, category, "low"))

    if marker_dict and marker_dict.exists():
        try:
            payload = json.loads(marker_dict.read_text(encoding="utf-8"))
            logic = payload.get("logic_markers", {})
            for marker in logic.get("A_derivation_markers", []):
                category = "final" if "answer" in marker.lower() else "derivation"
                priority = "high" if "answer" in marker.lower() else "medium"
                markers.append(MarkerSpec(str(marker), category, priority))
            for marker in logic.get("B_verification_reflection_markers", []):
                lowered = str(marker).lower()
                category = "correction" if any(word in lowered for word in ("wait", "wrong", "hold on")) else "verification"
                priority = "high" if any(word in lowered for word in ("wait", "check", "verify")) else "medium"
                markers.append(MarkerSpec(str(marker), category, priority))
            for marker in logic.get("C_starting_transition_markers", []):
                lowered = str(marker).lower()
                category = "narration" if any(word in lowered for word in ("let's see", "start", "tackle")) else "setup"
                priority = "low" if category == "narration" else "medium"
                markers.append(MarkerSpec(str(marker), category, priority))
        except Exception:
            pass
    return _dedup_markers(markers)


def _protected_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    patterns = [
        r"\\begin\{(?:aligned|align|cases|array|matrix|pmatrix|bmatrix)\}.*?\\end\{(?:aligned|align|cases|array|matrix|pmatrix|bmatrix)\}",
        r"\\\[.*?\\\]",
        r"\$\$.*?\$\$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            spans.append((match.start(), match.end()))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _in_spans(pos: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start < pos < end for start, end in spans)


def _line_boundaries(text: str, protected: Sequence[Tuple[int, int]]) -> List[Boundary]:
    boundaries: List[Boundary] = []
    for match in re.finditer(r"\n\s*\n+|\n", text):
        pos = match.end()
        if not _in_spans(pos, protected):
            boundaries.append(Boundary(pos, None, "unknown", "low"))
    return boundaries


def _sentence_boundaries(text: str, protected: Sequence[Tuple[int, int]]) -> List[Boundary]:
    boundaries: List[Boundary] = []
    for match in re.finditer(r"(?<=[.!?。！？])\s+", text):
        pos = match.end()
        if not _in_spans(pos, protected):
            boundaries.append(Boundary(pos, None, "unknown", "low"))
    return boundaries


def _marker_boundaries(
    text: str,
    markers: Sequence[MarkerSpec],
    protected: Sequence[Tuple[int, int]],
    min_priority: str,
    drop_generic_medium: bool,
) -> List[Boundary]:
    boundaries: List[Boundary] = []
    min_rank = PRIORITY_RANK[min_priority]
    for spec in markers:
        if PRIORITY_RANK[spec.priority] < min_rank:
            continue
        if drop_generic_medium and spec.priority == "medium" and spec.marker.lower().strip(" ,.:;") in GENERIC_MEDIUM_MARKERS:
            continue
        pattern = re.compile(re.escape(spec.marker), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            pos = match.start()
            if pos <= 0 or _in_spans(pos, protected):
                continue
            boundaries.append(Boundary(pos, spec.marker, spec.category, spec.priority))
    return boundaries


def _is_equationish(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        re.search(r"[=<>≤≥]|\\frac|\\sqrt|\\sum|\\int|\\begin|\\end", stripped)
        or re.match(r"^\s*(?:[-+*/=<>≤≥\d\s_a-zA-Z{}^\\().,]+)$", stripped)
    )


def _segment_marker(text: str, markers: Sequence[MarkerSpec]) -> Tuple[Optional[str], str, str]:
    stripped = text.lstrip()
    for spec in markers:
        if stripped.lower().startswith(spec.marker.lower()):
            return spec.marker, spec.category, spec.priority
    if "\\boxed{" in text:
        return "\\boxed{", "final", "high"
    return None, "unknown", "medium"


def _raw_segments_from_boundaries(
    text: str,
    boundaries: Sequence[Boundary],
    markers: Sequence[MarkerSpec],
) -> List[Dict[str, Any]]:
    boundary_by_pos: Dict[int, Boundary] = {}
    for boundary in boundaries:
        if boundary.pos <= 0 or boundary.pos >= len(text):
            continue
        old = boundary_by_pos.get(boundary.pos)
        if old is None or PRIORITY_RANK[boundary.priority] > PRIORITY_RANK[old.priority]:
            boundary_by_pos[boundary.pos] = boundary

    cuts = [0] + sorted(boundary_by_pos) + [len(text)]
    raw: List[Dict[str, Any]] = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        segment_text = text[start:end]
        marker, category, priority = _segment_marker(segment_text, markers)
        boundary = boundary_by_pos.get(start)
        if boundary and boundary.marker:
            marker, category, priority = boundary.marker, boundary.category, boundary.priority
        raw.append(
            {
                "text": segment_text,
                "char_start": start,
                "char_end": end,
                "marker": marker,
                "marker_category": category if category in MARKER_CATEGORIES else "unknown",
                "priority": priority if priority in PRIORITY_RANK else "medium",
            }
        )
    return raw


def _merge_two(left: Dict[str, Any], right: Dict[str, Any], original: str) -> Dict[str, Any]:
    start = int(left["char_start"])
    end = int(right["char_end"])
    marker = left.get("marker") or right.get("marker")
    category = left.get("marker_category")
    if category == "unknown":
        category = right.get("marker_category", "unknown")
    priority = left.get("priority", "medium")
    if PRIORITY_RANK.get(right.get("priority", "medium"), 1) > PRIORITY_RANK.get(priority, 1):
        priority = right.get("priority", "medium")
    return {
        "text": original[start:end],
        "char_start": start,
        "char_end": end,
        "marker": marker,
        "marker_category": category,
        "priority": priority,
    }


def _merge_short_segments(segments: List[Dict[str, Any]], original: str, min_chars: int = 28) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for segment in segments:
        strong = segment.get("priority") == "high"
        if (
            merged
            and not strong
            and len(str(segment.get("text", "")).strip()) < min_chars
            and segment.get("marker_category") not in {"final", "correction", "verification"}
        ):
            merged[-1] = _merge_two(merged[-1], segment, original)
        else:
            merged.append(segment)
    return merged


def _merge_equation_runs(segments: List[Dict[str, Any]], original: str) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged: List[Dict[str, Any]] = [segments[0]]
    for segment in segments[1:]:
        prev = merged[-1]
        if (
            prev.get("priority") != "high"
            and segment.get("priority") != "high"
            and _is_equationish(str(prev.get("text", "")))
            and _is_equationish(str(segment.get("text", "")))
        ):
            merged[-1] = _merge_two(prev, segment, original)
        else:
            merged.append(segment)
    return merged


def _limit_segments(segments: List[Dict[str, Any]], original: str, max_segments: int) -> Tuple[List[Dict[str, Any]], bool]:
    if max_segments <= 0 or len(segments) <= max_segments:
        return segments, False
    limited = list(segments)
    applied = False
    while len(limited) > max_segments:
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[int, int]] = None
        for idx in range(len(limited) - 1):
            left = limited[idx]
            right = limited[idx + 1]
            if left.get("priority") == "high" or right.get("priority") == "high":
                continue
            score = (
                PRIORITY_RANK.get(left.get("priority", "medium"), 1) + PRIORITY_RANK.get(right.get("priority", "medium"), 1),
                len(str(left.get("text", ""))) + len(str(right.get("text", ""))),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            best_idx = 0
        limited[best_idx] = _merge_two(limited[best_idx], limited[best_idx + 1], original)
        del limited[best_idx + 1]
        applied = True
    return limited, applied


def _assign_ids(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        row = {
            "seg_id": idx,
            "text": segment["text"],
            "char_start": int(segment["char_start"]),
            "char_end": int(segment["char_end"]),
            "marker": segment.get("marker"),
            "marker_category": segment.get("marker_category", "unknown"),
            "priority": segment.get("priority", "medium"),
        }
        assigned.append(row)
    return assigned


def segment_atomic(
    cot_text: str,
    marker_dict: Optional[Path] = None,
    max_atomic_segments: int = 160,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not cot_text:
        return [], ["empty_cot"]
    engineering_flags: List[str] = []
    markers = load_marker_specs(marker_dict)
    protected = _protected_spans(cot_text)

    def build(min_priority: str, drop_generic_medium: bool) -> List[Dict[str, Any]]:
        boundaries: List[Boundary] = []
        boundaries.extend(_line_boundaries(cot_text, protected))
        if min_priority == "low":
            boundaries.extend(_sentence_boundaries(cot_text, protected))
        boundaries.extend(_marker_boundaries(cot_text, markers, protected, min_priority, drop_generic_medium))
        raw = _raw_segments_from_boundaries(cot_text, boundaries, markers)
        raw = _merge_short_segments(raw, cot_text)
        raw = _merge_equation_runs(raw, cot_text)
        return raw

    segments = build("low", False)
    if len(segments) > max_atomic_segments:
        segments = build("medium", True)
        engineering_flags.append("segment_limit_relaxed_markers")
    segments, limited = _limit_segments(segments, cot_text, max_atomic_segments)
    if limited:
        engineering_flags.append("segment_limit_applied")

    assigned = _assign_ids(segments)
    for segment in assigned:
        start = int(segment["char_start"])
        end = int(segment["char_end"])
        if cot_text[start:end] != segment["text"]:
            raise ValueError(f"Atomic segment {segment['seg_id']} is not an exact original span.")
    return assigned, engineering_flags


__all__ = ["segment_atomic", "load_marker_specs"]
