"""
Scoring constants for CoT pairwise judge and selection pipeline.

All weights, penalties, and caps are defined here so the scoring functions
and prompt design stay in sync with a single source of truth.
"""

# ── Quality tag weights ──────────────────────────────────
# Higher = more important for a good CoT.
QUALITY_WEIGHT = {
    "Present": 3.0,
    "Cohesive": 3.0,
    "Deep": 2.0,
    "Concise": 2.0,
    "Exploratory": 1.0,
}

# ── Issue penalties ──────────────────────────────────────
# Higher = more severe.  These are subtracted from the score.
ISSUE_PENALTY = {
    "redundant_restatement": 1.0,
    "redundant_verification": 1.5,
    "unnecessary_cross_validation": 3.0,
    "hallucinated_context": 6.0,
    "forced_verification_alignment": 8.0,
}

# ── Severe-issue score caps ──────────────────────────────
# Prevents a CoT with a severe issue from out-scoring a clean one
# just because it has many quality tags.
SEVERE_CAPS = [
    ("forced_verification_alignment", 3.0),
    ("hallucinated_context", 5.0),
    ("unnecessary_cross_validation", 8.0),
]

# ── Verified-correct bonus ───────────────────────────────
VERIFIED_BONUS = 10.0

# ── Difficulty bonus (global scoring only) ───────────────
DIFFICULTY_BONUS = {
    "Hard": 2.0,
    "Medium": 0.0,
    "Easy": -1.0,
}

# ── Valid enum sets (must match prompt) ──────────────────
VALID_TAGS = {"Deep", "Present", "Exploratory", "Cohesive", "Concise"}
VALID_ISSUES = {
    "redundant_restatement",
    "redundant_verification",
    "unnecessary_cross_validation",
    "forced_verification_alignment",
    "hallucinated_context",
}
VALID_DIFFICULTY = {"Hard", "Medium", "Easy"}
