"""
CoT scoring functions — in-problem winner selection and global ranking.

Design principle:
  selection_score = verified_bonus + quality_sum - issue_penalty  (with severe cap)
  global_score    = selection_score + difficulty_bonus

selection_score decides the winner *within* a problem (A vs B).
global_score decides priority *across* problems during clustering / Top-K.
"""

from typing import List, Optional

from .config import (
    QUALITY_WEIGHT,
    ISSUE_PENALTY,
    SEVERE_CAPS,
    VERIFIED_BONUS,
    DIFFICULTY_BONUS,
)


def apply_issue_cap(score: float, issues: List[str]) -> float:
    """Cap the score when severe issues are present.

    Prevents a CoT with a serious quality problem from out-scoring a clean
    CoT simply because it accumulated many quality-tag points.
    """
    for issue, cap in SEVERE_CAPS:
        if issue in issues:
            return min(score, cap)
    return score


def selection_score(
    quality_tags: List[str],
    issues: List[str],
    answer_label: str = "unverified_or_wrong",
) -> float:
    """Compute the in-problem selection score for a single CoT.

    Parameters
    ----------
    quality_tags : list[str]
        Subset of {"Deep", "Present", "Exploratory", "Cohesive", "Concise"}.
    issues : list[str]
        Subset of the 5 issue types.
    answer_label : str
        Either "verified_correct" or "unverified_or_wrong".

    Returns
    -------
    float
        Higher = better CoT for this problem.
    """
    quality_sum = sum(
        QUALITY_WEIGHT[tag] for tag in quality_tags if tag in QUALITY_WEIGHT
    )
    issue_penalty = sum(
        ISSUE_PENALTY[issue] for issue in issues if issue in ISSUE_PENALTY
    )
    verified_bonus = VERIFIED_BONUS if answer_label == "verified_correct" else 0.0

    raw = verified_bonus + quality_sum - issue_penalty
    return apply_issue_cap(raw, issues)


def global_score(
    selection: float,
    problem_difficulty: str,
) -> float:
    """Add difficulty bonus for cross-problem ranking.

    selection_score measures CoT quality within a problem.
    global_score measures how valuable this sample is to the training set.
    """
    bonus = DIFFICULTY_BONUS.get(problem_difficulty, 0.0)
    return selection + bonus


def pick_winner(
    score_a: float,
    score_b: float,
    ifd_a: Optional[float] = None,
    ifd_b: Optional[float] = None,
    len_a: Optional[int] = None,
    len_b: Optional[int] = None,
) -> str:
    """Pick the winner between two CoTs.

    Priority: higher score → higher IFD → shorter length.
    IFD and length are only consulted when scores are equal.
    """
    if score_a > score_b:
        return "a"
    if score_b > score_a:
        return "b"

    # Score tie — IFD tiebreaker (higher IFD = harder to learn = more valuable)
    if ifd_a is not None and ifd_b is not None and ifd_a != ifd_b:
        return "a" if ifd_a > ifd_b else "b"

    # IFD tie — length tiebreaker (shorter is better, all else equal)
    if len_a is not None and len_b is not None and len_a != len_b:
        return "a" if len_a < len_b else "b"

    return "a"  # ultimate fallback
