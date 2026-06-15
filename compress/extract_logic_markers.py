import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path(
    r"D:\study\科研\大模型与智能体\backup\home\zhouyan\share\cyh\CoT\3k_test_data\data_sample_3k.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\study\科研\大模型与智能体\backup\home\zhouyan\share\cyh\CoT\compress"
)
DEFAULT_OUTPUT_DICT = DEFAULT_OUTPUT_DIR / "logic_markers_dict.json"


# Field names are checked in this order. The data sample currently stores long
# reasoning traces in "generations", but this keeps the script usable on nearby
# CoT datasets with different schemas.
TEXT_FIELD_PRIORITY = (
    "original_cot",
    "cot",
    "chain_of_thought",
    "reasoning",
    "rationale",
    "response",
    "output",
    "generation",
    "generations",
    "solution",
    "messages",
)


SEED_MARKERS: Dict[str, Sequence[str]] = {
    "A_derivation_markers": (
        "Putting it all together",
        "This is equivalent to",
        "Which is equivalent to",
        "This implies that",
        "This means that",
        "Which means that",
        "From this we get",
        "Substituting this back",
        "Substituting into",
        "Solving for",
        "Rearranging gives",
        "Therefore, we have",
        "Therefore",
        "Consequently",
        "It follows that",
        "Thus, we get",
        "Thus",
        "Hence",
        "So, we have",
        "So this gives",
        "So",
        "Using the fact that",
        "By substitution",
        "Combining these",
        "Simplifying gives",
        "After simplification",
        "Then",
        "Therefore, the answer is",
    ),
    "B_verification_reflection_markers": (
        "Wait, let me check that",
        "Wait, let me check",
        "Let me check this",
        "Let me verify this",
        "Let's verify this",
        "Let's check this",
        "Let me double-check",
        "Double-checking",
        "To verify",
        "Check original equations",
        "Checking the conditions",
        "But wait",
        "Wait",
        "Hold on",
        "However",
        "But",
        "This seems",
        "That seems",
        "Need to check",
        "We need to check",
        "Let's test",
        "As a check",
    ),
    "C_starting_transition_markers": (
        "Let's tackle this problem step by step",
        "Let's solve this step by step",
        "Let's start by",
        "Let me start by",
        "First, let me",
        "First,",
        "At first",
        "Initially",
        "Now, let's",
        "Next,",
        "Then,",
        "Finally",
        "In summary",
        "To summarize",
        "Now",
        "Also",
        "Alternatively",
        "Another way",
        "Case 1",
        "Case 2",
        "Suppose that",
        "Assume that",
        "Let's assume",
        "Let's denote",
        "Let me denote",
        "Let me recall",
        "Let's see",
        "Moving on",
    ),
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] Skip invalid JSON at line {line_no}: {exc}")
    return records


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        parts: List[str] = []
        for key in ("content", "text", "response", "output", "reasoning"):
            if key in value:
                parts.append(flatten_text(value[key]))
        if parts:
            return "\n".join(parts)
        return "\n".join(flatten_text(v) for v in value.values())
    return str(value)


def extract_cot_text(record: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for field in TEXT_FIELD_PRIORITY:
        if field in record:
            text = flatten_text(record[field]).strip()
            if text:
                return text, field
    return "", None


def position_stratified_sample(
    records: Sequence[Dict[str, Any]], sample_size: int, strata: int, seed: int
) -> List[Tuple[int, Dict[str, Any]]]:
    if not records:
        return []

    rng = random.Random(seed)
    n = len(records)
    sample_size = min(sample_size, n)
    strata = max(1, min(strata, sample_size, n))
    base = sample_size // strata
    extra = sample_size % strata
    selected: List[int] = []

    for i in range(strata):
        start = (n * i) // strata
        end = (n * (i + 1)) // strata
        bucket = list(range(start, end))
        if not bucket:
            continue
        k = base + (1 if i < extra else 0)
        k = min(k, len(bucket))
        selected.extend(rng.sample(bucket, k))

    if len(selected) < sample_size:
        remaining = [idx for idx in range(n) if idx not in set(selected)]
        selected.extend(rng.sample(remaining, sample_size - len(selected)))

    selected = sorted(set(selected))
    return [(idx, records[idx]) for idx in selected[:sample_size]]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def is_clean_marker(phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase or len(phrase) > 80:
        return False
    if re.search(r"[$\\=<>^_{}]|\d", phrase):
        return False
    if re.search(r"[.;:!?]{2,}", phrase):
        return False
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", phrase)
    return 1 <= len(words) <= 8


def count_seed_markers(texts: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    corpus = "\n".join(normalize_text(t) for t in texts)
    for markers in SEED_MARKERS.values():
        for marker in markers:
            pattern = re.compile(r"(?<![A-Za-z])" + re.escape(marker) + r"(?![A-Za-z])", re.IGNORECASE)
            count = len(pattern.findall(corpus))
            if count:
                counts[marker] += count
    return counts


def harvest_candidate_markers(texts: Iterable[str]) -> Counter:
    """Collect only short, generic sentence-leading discourse markers.

    The extractor deliberately avoids open-ended captures such as
    "Therefore, the maximum ..." because those phrases leak problem-specific
    mathematical content into the dictionary.
    """
    counts: Counter = Counter()
    controlled = (
        "But wait",
        "Wait",
        "So",
        "Therefore",
        "Thus",
        "Hence",
        "Then",
        "Now",
        "First",
        "Next",
        "Finally",
        "Alternatively",
        "However",
        "In summary",
        "To summarize",
        "Let's see",
        "Let's check",
        "Let's verify",
        "Let's assume",
        "Let's denote",
        "Let me check",
        "Let me verify",
        "Let me compute",
        "Let me denote",
        "Let me recall",
        "Suppose",
        "Assume",
    )
    leading_pattern = re.compile(
        r"(?:^|[\n.!?]\s+)(" + "|".join(re.escape(p) for p in controlled) + r")(?=[,\s:.])",
        re.IGNORECASE,
    )
    for text in texts:
        clean = normalize_text(text)
        for match in leading_pattern.finditer(clean):
            phrase = match.group(1).strip(" ,:\n\t")
            phrase = re.sub(r"\s+", " ", phrase)
            if is_clean_marker(phrase):
                counts[phrase] += 1
    return counts


def classify_marker(marker: str) -> str:
    lower = marker.lower()
    verification_heads = (
        "wait",
        "but wait",
        "let me check",
        "let me double-check",
        "let me verify",
        "let's verify",
        "let's check",
        "double-check",
        "to verify",
        "check",
        "checking",
        "hold on",
        "however",
        "need to check",
        "we need to check",
        "let's test",
        "as a check",
        "this seems",
        "that seems",
        "but",
    )
    transition_heads = (
        "first",
        "at first",
        "initially",
        "next",
        "finally",
        "in summary",
        "to summarize",
        "let's start",
        "let me start",
        "let's tackle",
        "let's solve",
        "let's assume",
        "assume",
        "suppose",
        "let's denote",
        "let me denote",
        "let me recall",
        "let's see",
        "moving on",
        "case ",
        "another way",
        "alternatively",
        "now, let's",
        "also",
    )
    if lower.startswith(verification_heads):
        return "B_verification_reflection_markers"
    if lower.startswith(transition_heads):
        return "C_starting_transition_markers"
    return "A_derivation_markers"


def sort_markers(markers: Iterable[str]) -> List[str]:
    unique = sorted(set(m.strip() for m in markers if is_clean_marker(m)), key=lambda x: (-len(x), x.lower()))
    return unique


def build_dictionary(texts: Sequence[str], min_count: int = 2) -> Dict[str, List[str]]:
    seed_counts = count_seed_markers(texts)
    harvested_counts = harvest_candidate_markers(texts)
    merged_counts = seed_counts + harvested_counts

    grouped: Dict[str, List[str]] = defaultdict(list)
    for category, markers in SEED_MARKERS.items():
        grouped[category].extend(markers)

    for marker, count in merged_counts.items():
        if count >= min_count:
            grouped[classify_marker(marker)].append(marker)

    return {category: sort_markers(grouped[category]) for category in SEED_MARKERS}


def optional_llm_extract(texts: Sequence[str]) -> Optional[Dict[str, List[str]]]:
    """Optional OpenAI API hook.

    This task is completed locally by default. If deeper semantic extraction is
    needed later, set USE_LLM_EXTRACTION=1 and OPENAI_API_KEY, install the
    official openai package, and adapt the model name below.
    """
    if os.getenv("USE_LLM_EXTRACTION") != "1":
        return None
    if not os.getenv("OPENAI_API_KEY"):
        print("[warn] USE_LLM_EXTRACTION=1 but OPENAI_API_KEY is not set; falling back to local extraction.")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[warn] openai package is not installed; falling back to local extraction.")
        return None

    client = OpenAI()
    prompt = (
        "Extract clean English logical marker phrases from these math CoTs. "
        "Return strict JSON with keys A_derivation_markers, "
        "B_verification_reflection_markers, C_starting_transition_markers. "
        "Do not include formulas, variables, numbers, or punctuation-only text. "
        "Sort each list by string length descending.\n\n"
        + "\n\n---\n\n".join(texts)
    )
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )
    try:
        data = json.loads(response.output_text)
    except json.JSONDecodeError:
        print("[warn] LLM response was not valid JSON; falling back to local extraction.")
        return None
    return {category: sort_markers(data.get(category, [])) for category in SEED_MARKERS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract logic markers from math CoT JSONL data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DICT)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--strata", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.input)
    sampled = position_stratified_sample(records, args.sample_size, args.strata, args.seed)

    texts: List[str] = []
    field_counts: Counter = Counter()
    sampled_indices: List[int] = []
    for idx, record in sampled:
        text, field = extract_cot_text(record)
        if text:
            texts.append(text)
            sampled_indices.append(idx)
            field_counts[field or "<unknown>"] += 1

    llm_dict = optional_llm_extract(texts)
    marker_dict = llm_dict if llm_dict is not None else build_dictionary(texts, min_count=args.min_count)

    payload = {
        "metadata": {
            "input_path": str(args.input),
            "sample_size_requested": args.sample_size,
            "sample_size_used": len(texts),
            "sampling_strategy": "position_stratified_random",
            "strata": args.strata,
            "seed": args.seed,
            "text_fields_used": dict(field_counts),
            "sampled_record_indices_zero_based": sampled_indices,
            "ordering_rule": "Each marker list is sorted by string length descending.",
        },
        "logic_markers": marker_dict,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote logic marker dictionary to: {args.output}")
    print(f"Sampled {len(texts)} records from {len(records)} total records.")
    print(f"Text fields used: {dict(field_counts)}")


if __name__ == "__main__":
    main()
