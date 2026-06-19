import argparse
import json
import os
import re
import sys
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR.parent / "3k_test_data" / "data_sample_3k.jsonl"
DEFAULT_MARKERS_PATH = SCRIPT_DIR / "logic_markers_dict.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "verify_logic_pruning_result.json"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"
RUN_LOG_LINES: List[str] = []

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


class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def ctext(text: str, color: str) -> str:
    return f"{color}{text}{Color.RESET}"


def log_step(message: str, color: str = Color.CYAN) -> None:
    RUN_LOG_LINES.append(message)
    print(ctext(message, color), flush=True)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log_step(f"[warn] skip invalid JSON at line {line_no}: {exc}", Color.YELLOW)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        preferred_parts: List[str] = []
        for key in ("content", "text", "response", "output", "reasoning"):
            if key in value:
                preferred_parts.append(flatten_text(value[key]))
        if preferred_parts:
            return "\n".join(part for part in preferred_parts if part.strip())
        return "\n".join(flatten_text(v) for v in value.values())
    return str(value)


def extract_cot_text(record: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for field in TEXT_FIELD_PRIORITY:
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, list) and field == "generations":
            candidates = [flatten_text(item).strip() for item in value]
            candidates = [item for item in candidates if item]
            if candidates:
                return max(candidates, key=len), field
        text = flatten_text(value).strip()
        if text:
            return text, field
    return "", None


def choose_long_record(data_path: Path, min_chars: int) -> Tuple[Dict[str, Any], str, str, int]:
    fallback: Tuple[Dict[str, Any], str, str, int] = ({}, "", "", -1)
    for idx, record in enumerate(load_jsonl(data_path)):
        cot_text, field = extract_cot_text(record)
        if len(cot_text) > len(fallback[1]):
            fallback = (record, cot_text, field or "unknown", idx)
        if len(cot_text) >= min_chars:
            return record, cot_text, field or "unknown", idx
    if fallback[1]:
        return fallback
    raise ValueError(f"No usable CoT text found in {data_path}")


def choose_record_by_index(data_path: Path, record_index: int) -> Tuple[Dict[str, Any], str, str, int]:
    for idx, record in enumerate(load_jsonl(data_path)):
        if idx != record_index:
            continue
        cot_text, field = extract_cot_text(record)
        if not cot_text:
            raise ValueError(f"Record #{record_index} has no usable CoT text.")
        return record, cot_text, field or "unknown", idx
    raise ValueError(f"Record index {record_index} is out of range for {data_path}")


def collect_markers(value: Any) -> List[str]:
    markers: List[str] = []
    if isinstance(value, str):
        if value.strip():
            markers.append(value.strip())
    elif isinstance(value, list):
        for item in value:
            markers.extend(collect_markers(item))
    elif isinstance(value, dict):
        for item in value.values():
            markers.extend(collect_markers(item))
    return markers


def load_markers(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        payload = json.load(f)
    markers = collect_markers(payload.get("logic_markers", payload))
    unique = sorted(set(markers), key=lambda item: (-len(item), item.lower()))
    if not unique:
        raise ValueError(f"No markers found in {path}")
    return unique


def marker_to_regex(marker: str) -> str:
    escaped = re.escape(marker)
    if marker and marker[0].isalnum():
        escaped = r"(?<![A-Za-z0-9_])" + escaped
    if marker and marker[-1].isalnum():
        escaped = escaped + r"(?![A-Za-z0-9_])"
    return escaped


def split_think_and_final(cot_text: str) -> Tuple[str, str, bool]:
    match = re.search(r"<think>(.*?)</think>", cot_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return cot_text.strip(), "", False
    think_process = match.group(1).strip()
    final_answer = cot_text[match.end() :].strip()
    return think_process, final_answer, True


def rebuild_cot(pruned_think_process: str, final_answer: str, has_think_tags: bool) -> str:
    if not has_think_tags:
        return pruned_think_process
    rebuilt = f"<think>\n{pruned_think_process.strip()}\n</think>"
    if final_answer:
        rebuilt = f"{rebuilt}\n\n{final_answer.strip()}"
    return rebuilt


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def secondary_subsplit(blocks: Sequence[str], word_threshold: int = 60) -> List[str]:
    refined: List[str] = []
    for block in blocks:
        normalized = re.sub(r"\r\n?", "\n", block).strip()
        if not normalized:
            continue
        non_empty_lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        should_split = word_count(normalized) > word_threshold or normalized.count("\n") > 1
        if should_split and len(non_empty_lines) > 1:
            refined.extend(non_empty_lines)
        else:
            refined.append(normalized)
    return refined


def heuristic_split(cot_text: str, markers: Sequence[str]) -> List[str]:
    text = re.sub(r"\r\n?", "\n", cot_text).strip()
    marker_pattern = "|".join(marker_to_regex(marker) for marker in markers if marker)
    chunks: List[str] = []

    if marker_pattern:
        pattern = re.compile(marker_pattern, flags=re.IGNORECASE)
        starts = sorted({match.start() for match in pattern.finditer(text)})
        if starts:
            if starts[0] > 0:
                chunks.append(text[: starts[0]].strip())
            for pos_idx, start in enumerate(starts):
                end = starts[pos_idx + 1] if pos_idx + 1 < len(starts) else len(text)
                chunks.append(text[start:end].strip())

    if len(chunks) < 3:
        chunks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]

    if len(chunks) < 3:
        chunks = [
            part.strip()
            for part in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", text)
            if part.strip()
        ]

    primary_blocks = [block for block in chunks if block]
    secondary_blocks = secondary_subsplit(primary_blocks)
    return merge_short_blocks(secondary_blocks)


def merge_short_blocks(blocks: Sequence[str], min_chars: int = 32) -> List[str]:
    merged: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) < min_chars and merged:
            merged[-1] = f"{merged[-1]}\n\n{block}".strip()
        else:
            merged.append(block)
    return merged


def skeletonize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\$+|\\\(|\\\)|\\\[|\\\]", " ", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", text)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    text = re.sub(r"\b[a-zA-Z_][a-zA-Z_0-9]*\b", "VAR", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def structure_similarity(left: str, right: str) -> float:
    left_skeleton = skeletonize(left)
    right_skeleton = skeletonize(right)
    if not left_skeleton or not right_skeleton:
        return 0.0
    return SequenceMatcher(None, left_skeleton, right_skeleton).ratio()


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_prompt(target: str, supports: Sequence[str], candidate_index: int) -> List[Dict[str, str]]:
    current_supports = list(supports[:-1])
    candidate_block = supports[-1] if supports else ""
    support_text = "\n\n".join(
        f"[Premise {idx}]\n{block}" for idx, block in enumerate(current_supports, start=1)
    )
    if not support_text:
        support_text = "(empty)"
    user_prompt = f"""
Target Block (Conclusion): {target}

Current Support Set (Premises): {support_text}

Candidate Premise to Add: {candidate_block}

Task:

Is the Candidate Premise logically related and 'useful' for deducing the Target Block? (useful: bool)

Does the Current Support Set + Candidate Premise provide a complete, unbroken logical chain that is strictly 'sufficient' to deduce the Target Block without any missing steps? (sufficient: bool)
Return ONLY a valid JSON object: {{"useful": bool, "sufficient": bool}}
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a strict, emotionless Formal Logic Verifier. You MUST pretend you have zero mathematical "
                "knowledge. Your ONLY job is to check if the 'Target Block' can be strictly and formally deduced "
                "SOLELY by reading the explicitly stated premises in the 'Support Set'. Do not hallucinate or use "
                "external knowledge. If even a single intermediate logical step is missing from the Support Set, "
                "you must declare it insufficient."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def messages_to_completion_prompt(messages: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def is_not_chat_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "not a chat model" in message or "v1/completions" in message


def extract_completion_text(response: Any) -> str:
    choice = response.choices[0]
    text = getattr(choice, "text", None)
    if text is not None:
        return text
    message = getattr(choice, "message", None)
    if message is not None:
        return getattr(message, "content", None) or ""
    return ""


def collect_stream_content(chunks: Any, show_reasoning: bool) -> str:
    content_parts: List[str] = []
    is_answering = False
    for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if show_reasoning and reasoning is not None and not is_answering:
            print(reasoning, end="", flush=True)
        content = getattr(delta, "content", None)
        if content:
            if show_reasoning and not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            content_parts.append(content)
    if show_reasoning:
        print()
    return "".join(content_parts)


def completion_verify_once(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, bool]:
    response = client.completions.create(
        model=model,
        prompt=messages_to_completion_prompt(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = parse_json_object(extract_completion_text(response) or "{}")
    return {
        "useful": bool(parsed.get("useful", False)),
        "sufficient": bool(parsed.get("sufficient", False)),
    }


def tokenize_for_overlap(text: str) -> Set[str]:
    normalized = text.lower()
    normalized = re.sub(r"\\[a-zA-Z]+", " ", normalized)
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|[-+]?\d+(?:\.\d+)?", normalized))
    stopwords = {
        "the",
        "and",
        "that",
        "this",
        "then",
        "with",
        "from",
        "have",
        "will",
        "into",
        "case",
        "so",
        "we",
        "is",
        "are",
        "to",
        "of",
        "in",
        "for",
        "a",
        "an",
    }
    return {token for token in tokens if token not in stopwords}


def heuristic_verify(target: str, supports: Sequence[str]) -> Dict[str, bool]:
    if not supports:
        return {"useful": False, "sufficient": False}

    candidate = supports[-1]
    target_tokens = tokenize_for_overlap(target)
    candidate_tokens = tokenize_for_overlap(candidate)
    support_tokens: Set[str] = set()
    for block in supports:
        support_tokens.update(tokenize_for_overlap(block))

    overlap = len(target_tokens & candidate_tokens)
    coverage = len(target_tokens & support_tokens) / max(len(target_tokens), 1)
    candidate_has_math = bool(re.search(r"[=<>]|\\frac|\\boxed|\\sum|\\sqrt|\d", candidate))
    useful = overlap >= 2 or candidate_has_math or structure_similarity(target, candidate) > 0.35
    sufficient = useful and bool(supports) and (coverage >= 0.15 or candidate_has_math or len(supports) >= 1)
    return {"useful": useful, "sufficient": sufficient}


def llm_verify(
    client: OpenAI,
    model: str,
    target: str,
    supports: Sequence[str],
    candidate_index: int,
    retries: int,
    temperature: float,
    completion_max_tokens: int,
    enable_thinking: bool,
    stream: bool,
    show_reasoning: bool,
) -> Dict[str, bool]:
    messages = build_prompt(target, supports, candidate_index)
    last_error: Optional[Exception] = None
    force_completion = False

    for attempt in range(1, retries + 1):
        try:
            if force_completion:
                return completion_verify_once(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=completion_max_tokens,
                )

            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "extra_body": {"enable_thinking": enable_thinking},
            }
            if stream:
                request_kwargs["stream"] = True
            if attempt == 1:
                request_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request_kwargs)
            if stream:
                if show_reasoning:
                    print("\n" + "=" * 20 + "思考过程" + "=" * 20)
                content = collect_stream_content(response, show_reasoning=show_reasoning)
            else:
                content = response.choices[0].message.content or "{}"
            parsed = parse_json_object(content)
            return {
                "useful": bool(parsed.get("useful", False)),
                "sufficient": bool(parsed.get("sufficient", False)),
            }
        except Exception as exc:
            last_error = exc
            if is_not_chat_model_error(exc) and not force_completion:
                force_completion = True
                log_step(
                    "[fallback] Model is not accepted by chat/completions; retrying with completions.",
                    Color.YELLOW,
                )
                continue
            sleep_s = min(2 ** attempt, 12)
            log_step(
                f"[retry] LLM verification failed on attempt {attempt}/{retries}: {exc}",
                Color.YELLOW,
            )
            if attempt < retries:
                time.sleep(sleep_s)

    raise RuntimeError(f"LLM verification failed after {retries} attempts: {last_error}")


def backward_dependency_bfs(
    blocks: Sequence[str],
    client: Optional[OpenAI],
    model: str,
    verifier: str,
    retries: int,
    temperature: float,
    completion_max_tokens: int,
    enable_thinking: bool,
    stream: bool,
    show_reasoning: bool,
    similarity_threshold: float,
) -> Tuple[Set[int], List[Dict[str, Any]], Optional[int]]:
    n = len(blocks) - 1
    necessary_blocks: Set[int] = {n}
    queued: Set[int] = {n}
    q: deque[int] = deque([n])
    logs: List[Dict[str, Any]] = []

    while q:
        t = q.popleft()
        target = blocks[t]
        s_indices: List[int] = []
        i = t - 1
        closed = False
        log_step(f"\n[Target] B_{t}: {target[:160].replace(chr(10), ' ')}", Color.BOLD + Color.BLUE)
        if t == 0:
            log_step("  [Root] B_0 has no earlier support block; treat it as a root premise.", Color.GREEN)
            continue

        while i >= 0:
            if s_indices:
                latest = s_indices[-1]
                sim = structure_similarity(blocks[i], blocks[latest])
                if sim > similarity_threshold:
                    s_indices.append(i)
                    necessary_blocks.add(i)
                    if i not in queued:
                        q.append(i)
                        queued.add(i)
                    logs.append(
                        {
                            "target": t,
                            "candidate": i,
                            "action": "structure_similarity_accept",
                            "similarity": sim,
                            "useful": True,
                            "sufficient": False,
                        }
                    )
                    log_step(
                        f"  [SIM] B_{i} ~ B_{latest}, score={sim:.3f}; accept as necessary",
                        Color.MAGENTA,
                    )
                    i -= 1
                    continue

            supports = [blocks[idx] for idx in s_indices] + [blocks[i]]
            log_step(f"  [{verifier.upper()}] Verify candidate B_{i} -> B_{t}", Color.CYAN)
            if verifier == "heuristic":
                verdict = heuristic_verify(target=target, supports=supports)
            else:
                if client is None:
                    raise ValueError("API verifier requires an OpenAI client.")
                verdict = llm_verify(
                    client=client,
                    model=model,
                    target=target,
                    supports=supports,
                    candidate_index=i,
                    retries=retries,
                    temperature=temperature,
                    completion_max_tokens=completion_max_tokens,
                    enable_thinking=enable_thinking,
                    stream=stream,
                    show_reasoning=show_reasoning,
                )
            useful = verdict["useful"]
            sufficient = verdict["sufficient"]
            action = "skip"

            if useful:
                s_indices.append(i)
                necessary_blocks.add(i)
                if i not in queued:
                    q.append(i)
                    queued.add(i)
                action = "accept"

            logs.append(
                {
                    "target": t,
                    "candidate": i,
                    "action": action,
                    "useful": useful,
                    "sufficient": sufficient,
                }
            )
            useful_color = Color.GREEN if useful else Color.YELLOW
            log_step(
                f"    verdict: useful={useful}, sufficient={sufficient}, action={action}",
                useful_color,
            )

            if sufficient:
                closed = True
                log_step(f"  [Closed] B_{t} support chain is sufficient.", Color.GREEN)
                break

            i -= 1

        if not closed:
            log_step(f"[Exception] Logic Jump Detected at Block {t}", Color.RED + Color.BOLD)
            return necessary_blocks, logs, t

    return necessary_blocks, logs, None


def preview_block(block: str, max_chars: int = 500) -> str:
    block = re.sub(r"\s+", " ", block).strip()
    if len(block) <= max_chars:
        return block
    return block[: max_chars - 3] + "..."


def write_result(
    output_path: Path,
    record_index: int,
    field: str,
    original_cot: str,
    think_process: str,
    final_answer: str,
    has_think_tags: bool,
    blocks: Sequence[str],
    necessary_blocks: Set[int],
    logs: Sequence[Dict[str, Any]],
    exception_block: Optional[int],
) -> None:
    pruned_blocks = [block for idx, block in enumerate(blocks) if idx in necessary_blocks]
    pruned_think_process = "\n\n".join(pruned_blocks)
    pruned_cot = rebuild_cot(pruned_think_process, final_answer, has_think_tags)
    removed_blocks = [
        {"index": idx, "text": block}
        for idx, block in enumerate(blocks)
        if idx not in necessary_blocks
    ]
    payload = {
        "record_index_zero_based": record_index,
        "cot_field": field,
        "status": "logic_jump" if exception_block is not None else "ok",
        "exception_block": exception_block,
        "original_block_count": len(blocks),
        "pruned_block_count": len(pruned_blocks),
        "removed_block_count": len(removed_blocks),
        "compression_ratio_blocks": len(pruned_blocks) / max(len(blocks), 1),
        "necessary_indices": sorted(necessary_blocks),
        "all_blocks": [
            {"index": idx, "text": block, "necessary": idx in necessary_blocks}
            for idx, block in enumerate(blocks)
        ],
        "think_process": think_process,
        "final_answer": final_answer,
        "has_think_tags": has_think_tags,
        "pruned_think_process": pruned_think_process,
        "pruned_cot": pruned_cot,
        "removed_blocks": removed_blocks,
        "logs": list(logs),
        "original_cot": original_cot,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text_log(
    log_path: Path,
    record_index: int,
    field: str,
    blocks: Sequence[str],
    necessary_blocks: Set[int],
    removed_blocks: Sequence[Tuple[int, str]],
    logs: Sequence[Dict[str, Any]],
    exception_block: Optional[int],
) -> None:
    lines: List[str] = []
    lines.append("# Logic Pruning Run Log")
    lines.append("")
    lines.append(f"record_index_zero_based: {record_index}")
    lines.append(f"cot_field: {field}")
    lines.append(f"status: {'logic_jump' if exception_block is not None else 'ok'}")
    lines.append(f"exception_block: {exception_block}")
    lines.append(f"original_block_count: {len(blocks)}")
    lines.append(f"pruned_block_count: {len(necessary_blocks)}")
    lines.append(f"removed_block_count: {len(removed_blocks)}")
    lines.append(f"necessary_indices: {sorted(necessary_blocks)}")
    lines.append("scope: BFS executed only on think_process; final_answer is preserved unchanged.")
    lines.append("")
    lines.append("## Console Trace")
    lines.extend(RUN_LOG_LINES)
    lines.append("")
    lines.append("## Split Blocks")
    for idx, block in enumerate(blocks):
        status = "KEEP" if idx in necessary_blocks else "REMOVE"
        lines.append(f"\n[B_{idx}] {status}")
        lines.append(block)
    lines.append("")
    lines.append("## Verification Decisions")
    for item in logs:
        lines.append(json.dumps(item, ensure_ascii=False))
    lines.append("")
    lines.append("## Removed Blocks")
    if not removed_blocks:
        lines.append("(none)")
    for idx, block in removed_blocks:
        lines.append(f"\n[B_{idx}]")
        lines.append(block)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    return (
        explicit_api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("JENIYA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and prune CoT blocks by backward sufficient-condition exploration."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--markers-path", type=Path, default=DEFAULT_MARKERS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--api-key",
        default=None,
        help="Prefer env var DASHSCOPE_API_KEY, then JENIYA_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--verifier", choices=("api", "heuristic"), default="api")
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--completion-max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    if args.verifier == "api" and not api_key:
        log_step(
            "API key is not set. Please set DASHSCOPE_API_KEY, JENIYA_API_KEY, or OPENAI_API_KEY before running this script.",
            Color.RED,
        )
        return 2

    data_path = args.data_path.resolve()
    markers_path = args.markers_path.resolve()
    output_path = args.output_path.resolve()
    log_path = args.log_path.resolve() if args.log_path else output_path.with_suffix(".log")

    log_step(f"[Load] data: {data_path}", Color.BLUE)
    if args.record_index is None:
        _, cot_text, field, record_index = choose_long_record(data_path, args.min_chars)
    else:
        _, cot_text, field, record_index = choose_record_by_index(data_path, args.record_index)
    markers = load_markers(markers_path)
    log_step(f"[Load] selected record #{record_index}, field={field}, chars={len(cot_text)}", Color.BLUE)
    log_step(f"[Load] markers={len(markers)}", Color.BLUE)

    think_process, final_answer, has_think_tags = split_think_and_final(cot_text)
    log_step(
        f"[Scope] BFS will run on think_process only: chars={len(think_process)}, "
        f"final_answer_chars={len(final_answer)}, has_think_tags={has_think_tags}",
        Color.BLUE,
    )

    blocks = heuristic_split(think_process, markers)
    if len(blocks) < 2:
        raise ValueError("Heuristic splitting produced fewer than 2 blocks.")
    if args.max_blocks is not None and args.max_blocks > 1 and len(blocks) > args.max_blocks:
        log_step(
            f"[Debug] Keep only the last {args.max_blocks} blocks for a quick example run.",
            Color.YELLOW,
        )
        blocks = blocks[-args.max_blocks:]

    log_step(f"[Split] original blocks={len(blocks)}", Color.GREEN)
    for idx, block in enumerate(blocks):
        log_step(f"  B_{idx}: {preview_block(block, 180)}", Color.DIM)

    if args.verifier == "api":
        log_step(f"[API] base_url={args.base_url}, model={args.model}", Color.BLUE)
        client: Optional[OpenAI] = OpenAI(api_key=api_key, base_url=args.base_url)
    else:
        log_step("[Verifier] heuristic mode: no API call will be made.", Color.BLUE)
        client = None
    necessary_blocks, logs, exception_block = backward_dependency_bfs(
        blocks=blocks,
        client=client,
        model=args.model,
        verifier=args.verifier,
        retries=args.retries,
        temperature=args.temperature,
        completion_max_tokens=args.completion_max_tokens,
        enable_thinking=args.enable_thinking,
        stream=args.stream,
        show_reasoning=args.show_reasoning,
        similarity_threshold=args.similarity_threshold,
    )

    pruned_blocks = [block for idx, block in enumerate(blocks) if idx in necessary_blocks]
    removed_blocks = [(idx, block) for idx, block in enumerate(blocks) if idx not in necessary_blocks]
    pruned_think_process = "\n\n".join(pruned_blocks)
    pruned_cot = rebuild_cot(pruned_think_process, final_answer, has_think_tags)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_result(
        output_path=output_path,
        record_index=record_index,
        field=field,
        original_cot=cot_text,
        think_process=think_process,
        final_answer=final_answer,
        has_think_tags=has_think_tags,
        blocks=blocks,
        necessary_blocks=necessary_blocks,
        logs=logs,
        exception_block=exception_block,
    )
    write_text_log(
        log_path=log_path,
        record_index=record_index,
        field=field,
        blocks=blocks,
        necessary_blocks=necessary_blocks,
        removed_blocks=removed_blocks,
        logs=logs,
        exception_block=exception_block,
    )

    if exception_block is None:
        log_step("\n[Done] Algorithm finished normally.", Color.GREEN + Color.BOLD)
    else:
        log_step("\n[Stopped] Algorithm stopped because of a detected logic jump.", Color.RED + Color.BOLD)

    print(ctext("\n========== Block Count ==========", Color.BOLD))
    print(f"Original blocks: {len(blocks)}")
    print(f"Pruned blocks:   {len(pruned_blocks)}")

    print(ctext("\n========== Original CoT ==========", Color.BOLD))
    print(cot_text)

    print(ctext("\n========== Pruned Think Process ==========", Color.BOLD))
    print(pruned_think_process)

    print(ctext("\n========== Pruned CoT ==========", Color.BOLD))
    print(pruned_cot)

    print(ctext("\n========== Removed Redundant Blocks ==========", Color.BOLD))
    if not removed_blocks:
        print("(none)")
    for idx, block in removed_blocks:
        print(ctext(f"\n[B_{idx}]", Color.YELLOW))
        print(block)

    log_step(f"\n[Saved] JSON result written to {output_path}", Color.GREEN)
    log_step(f"[Saved] Text log written to {log_path}", Color.GREEN)
    return 1 if exception_block is not None else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log_step("\nInterrupted by user.", Color.YELLOW)
        raise SystemExit(130)
