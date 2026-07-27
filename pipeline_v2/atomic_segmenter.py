from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from compress.pipeline import atomic_segmenter as legacy


def _english_marker_pattern(marker: str) -> re.Pattern[str]:
    escaped = re.escape(marker)
    left = r"(?<![A-Za-z])" if marker[:1].isalpha() else ""
    right = r"(?![A-Za-z])" if marker[-1:].isalpha() else ""
    return re.compile(left + escaped + right, flags=re.IGNORECASE)


def _strict_marker_boundaries(
    text: str,
    markers: Sequence[legacy.MarkerSpec],
    protected: Sequence[Tuple[int, int]],
    min_priority: str,
    drop_generic_medium: bool,
) -> List[legacy.Boundary]:
    boundaries: List[legacy.Boundary] = []
    min_rank = legacy.PRIORITY_RANK[min_priority]
    for spec in markers:
        if legacy.PRIORITY_RANK[spec.priority] < min_rank:
            continue
        if (
            drop_generic_medium
            and spec.priority == "medium"
            and spec.marker.lower().strip(" ,.:;") in legacy.GENERIC_MEDIUM_MARKERS
        ):
            continue
        pattern = (
            _english_marker_pattern(spec.marker)
            if any(char.isascii() and char.isalpha() for char in spec.marker)
            else re.compile(re.escape(spec.marker), flags=re.IGNORECASE)
        )
        for match in pattern.finditer(text):
            pos = match.start()
            if pos <= 0 or legacy._in_spans(pos, protected):
                continue
            boundaries.append(
                legacy.Boundary(pos, spec.marker, spec.category, spec.priority)
            )
    return boundaries


def _strict_segment_marker(
    text: str,
    markers: Sequence[legacy.MarkerSpec],
) -> Tuple[Optional[str], str, str]:
    stripped = text.lstrip()
    for spec in markers:
        pattern = (
            _english_marker_pattern(spec.marker)
            if any(char.isascii() and char.isalpha() for char in spec.marker)
            else re.compile(re.escape(spec.marker), flags=re.IGNORECASE)
        )
        match = pattern.match(stripped)
        if match:
            return spec.marker, spec.category, spec.priority
    if r"\boxed{" in text:
        return r"\boxed{", "final", "high"
    return None, "unknown", "medium"


def _raw_segments(
    text: str,
    boundaries: Sequence[legacy.Boundary],
    markers: Sequence[legacy.MarkerSpec],
) -> List[Dict[str, Any]]:
    raw = legacy._raw_segments_from_boundaries(text, boundaries, markers)
    for segment in raw:
        marker, category, priority = _strict_segment_marker(
            str(segment["text"]),
            markers,
        )
        segment["marker"] = marker
        segment["marker_category"] = category
        segment["priority"] = priority
    return raw


def segment_atomic_v2(
    cot_text: str,
    marker_dict: Optional[Path] = None,
    max_atomic_segments: int = 160,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """V2.1 exact-span segmentation with lexical marker boundaries.

    Unlike Compress V4.2, this version never joins sentence-level segments
    merely because both contain equations. Multi-line LaTeX environments are
    still protected as atomic spans.
    """

    if not cot_text:
        return [], ["empty_cot"]
    engineering_flags: List[str] = []
    markers = legacy.load_marker_specs(marker_dict)
    protected = legacy._protected_spans(cot_text)

    def build(min_priority: str, drop_generic_medium: bool) -> List[Dict[str, Any]]:
        boundaries: List[legacy.Boundary] = []
        boundaries.extend(legacy._line_boundaries(cot_text, protected))
        if min_priority == "low":
            boundaries.extend(legacy._sentence_boundaries(cot_text, protected))
        boundaries.extend(
            _strict_marker_boundaries(
                cot_text,
                markers,
                protected,
                min_priority,
                drop_generic_medium,
            )
        )
        raw = _raw_segments(cot_text, boundaries, markers)
        return legacy._merge_short_segments(raw, cot_text)

    segments = build("low", False)
    if len(segments) > max_atomic_segments:
        segments = build("medium", True)
        engineering_flags.append("segment_limit_relaxed_markers")
    segments, limited = legacy._limit_segments(
        segments,
        cot_text,
        max_atomic_segments,
    )
    if limited:
        engineering_flags.append("segment_limit_applied")

    assigned = legacy._assign_ids(segments)
    for segment in assigned:
        start = int(segment["char_start"])
        end = int(segment["char_end"])
        if cot_text[start:end] != segment["text"]:
            raise ValueError(
                f"Atomic segment {segment['seg_id']} is not an exact original span."
            )
    return assigned, engineering_flags


__all__ = ["segment_atomic_v2"]
