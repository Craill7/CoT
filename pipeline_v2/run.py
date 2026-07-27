from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import MockAnalyzer, V42Analyzer
from .answers import LLMAnswerJudge, MockAnswerJudge
from .common import PipelineConfig
from .diagnosis import DependencyDiagnoser
from .mock import MockJsonLLM
from .orchestrator import PipelineV2
from .rewrite import ConstrainedRewriter
from .selection import (
    IdentitySelector,
    IfdKMeansClusterer,
    MockGlobalScorer,
    MockPairwiseSelector,
    PairwiseSelector,
    ReuseOrAbsoluteGlobalScorer,
    ScoreRatioClusterer,
    SelectionRouter,
)
from .verifier import CombinedVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT Pipeline V2")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="/ky200t/models/Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--select-mode", choices=("auto", "identity", "pairwise"), default="auto")
    parser.add_argument("--clusterer", choices=("ifd_kmeans", "score"), default="ifd_kmeans")
    parser.add_argument("--model-path", default="/ky200t/models/Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--num-clusters", type=int, default=40)
    parser.add_argument("--k-ratio", type=float, default=0.8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--max-rewrite-rounds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def build_pipeline(config: PipelineConfig) -> PipelineV2:
    if config.mock:
        llm = MockJsonLLM()
        analyzer = MockAnalyzer()
        clusterer = ScoreRatioClusterer(config.k_ratio)
        pairwise = MockPairwiseSelector()
        scorer = MockGlobalScorer()
        answer_judge = MockAnswerJudge()
    else:
        from compress.pipeline.llm_client import LLMClient

        llm = LLMClient(
            base_url=config.base_url,
            model=config.model,
            temperature=0.0,
            max_tokens=config.llm_max_tokens,
            timeout=300,
            max_retries=2,
        )
        analyzer = V42Analyzer(llm, max_tokens=config.llm_max_tokens)
        clusterer = (
            IfdKMeansClusterer(
                model_path=config.model_path,
                num_clusters=config.num_clusters,
                k_ratio=config.k_ratio,
                max_length=config.max_length,
                device=f"cuda:{config.gpu_id}",
            )
            if config.clusterer == "ifd_kmeans"
            else ScoreRatioClusterer(config.k_ratio)
        )
        pairwise = PairwiseSelector(
            vllm_url=f"{config.base_url.rstrip('/')}/chat/completions",
            model=config.model,
        )
        scorer = ReuseOrAbsoluteGlobalScorer(llm)
        answer_judge = LLMAnswerJudge(llm)

    selector = SelectionRouter(IdentitySelector(), pairwise, config.select_mode)
    return PipelineV2(
        config=config,
        selector=selector,
        scorer=scorer,
        clusterer=clusterer,
        answer_judge=answer_judge,
        analyzer=analyzer,
        diagnoser=DependencyDiagnoser(llm),
        rewriter=ConstrainedRewriter(llm),
        verifier=CombinedVerifier(llm),
    )


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        select_mode=args.select_mode,
        clusterer=args.clusterer,
        model_path=args.model_path,
        num_clusters=args.num_clusters,
        k_ratio=args.k_ratio,
        max_length=args.max_length,
        gpu_id=args.gpu_id,
        llm_max_tokens=args.llm_max_tokens,
        max_rewrite_rounds=args.max_rewrite_rounds,
        limit=args.limit,
        resume=args.resume,
        mock=args.mock,
    )
    stats = build_pipeline(config).run()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
