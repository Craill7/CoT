from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import CotAnalyzer
from .answers import AnswerJudge
from .common import JsonlStage, PipelineConfig, final_answer, iter_jsonl, normalize_record, stage_row
from .diagnosis import DependencyDiagnoser
from .rewrite import ConstrainedRewriter, apply_rewrite, supporting_text
from .selection import Clusterer, GlobalScorer, SelectionRouter
from .verifier import CombinedVerifier


class PipelineV2:
    def __init__(
        self,
        config: PipelineConfig,
        selector: SelectionRouter,
        scorer: GlobalScorer,
        clusterer: Clusterer,
        answer_judge: AnswerJudge,
        analyzer: CotAnalyzer,
        diagnoser: DependencyDiagnoser,
        rewriter: ConstrainedRewriter,
        verifier: CombinedVerifier,
    ):
        self.config = config
        self.selector = selector
        self.scorer = scorer
        self.clusterer = clusterer
        self.answer_judge = answer_judge
        self.analyzer = analyzer
        self.diagnoser = diagnoser
        self.rewriter = rewriter
        self.verifier = verifier
        self.fingerprint = config.fingerprint()
        output = Path(config.output_dir)
        self.stages = {
            name: JsonlStage(output / f"{name}.jsonl")
            for name in (
                "normalized",
                "selected",
                "high_quality",
                "low_quality",
                "answer_audit",
                "diagnosed",
                "rewritten",
                "accepted",
                "rejected",
                "lineage",
            )
        }
        self.stats_path = output / "stats.json"

    def _flat_stage(self, name: str, record: Dict[str, Any]) -> None:
        self.stages[name].append(stage_row(record, self.fingerprint, **record))

    def _terminal(self, sample_id: str) -> Optional[Dict[str, Any]]:
        return self.stages["accepted"].get(sample_id, self.fingerprint) or self.stages["rejected"].get(
            sample_id, self.fingerprint
        )

    def _write_lineage(self, record: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
        self.stages["lineage"].append(
            stage_row(
                record,
                self.fingerprint,
                terminal_state=events[-1]["state"] if events else "unknown",
                events=events,
            )
        )

    def _accept(
        self,
        record: Dict[str, Any],
        final_cot: str,
        route: str,
        events: List[Dict[str, Any]],
        **details: Any,
    ) -> None:
        events.append({"state": "accepted", "time": time.time()})
        row = stage_row(
            record,
            self.fingerprint,
            route=route,
            final_cot=final_cot,
            lineage=events,
            **details,
        )
        self.stages["accepted"].append(row)
        self._write_lineage(record, events)

    def _reject(
        self,
        record: Dict[str, Any],
        reason: str,
        events: List[Dict[str, Any]],
        **details: Any,
    ) -> None:
        events.append({"state": "rejected", "reason": reason, "time": time.time()})
        row = stage_row(
            record,
            self.fingerprint,
            reason=reason,
            lineage=events,
            **details,
        )
        self.stages["rejected"].append(row)
        self._write_lineage(record, events)

    def _select_all(self) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for line_no, source in iter_jsonl(Path(self.config.input_path)):
            if self.config.limit is not None and len(selected) >= self.config.limit:
                break
            normalized = normalize_record(source, str(line_no))
            cached = self.stages["selected"].get(normalized["sample_id"], self.fingerprint)
            if cached:
                selected.append({key: value for key, value in cached.items() if key != "config_fingerprint"})
                continue
            self._flat_stage("normalized", normalized)
            try:
                started = time.perf_counter()
                selection = self.selector.select(normalized)
                scored = self.scorer.score(normalized, selection)
                merged = {
                    **normalized,
                    **scored,
                    "original_final_answer": (
                        normalized.get("original_final_answer")
                        or final_answer(str(scored["selected_cot"]))
                    ),
                    "stage_timings": {"selection_and_global_score_s": time.perf_counter() - started},
                }
                self._flat_stage("selected", merged)
                selected.append(merged)
            except Exception as exc:
                events = [{"state": "selection_failed", "error": str(exc), "time": time.time()}]
                self._reject(normalized, "selection_failed", events, error=str(exc))
        return selected

    def _route_all(self, selected: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cached_high: List[Dict[str, Any]] = []
        cached_low: List[Dict[str, Any]] = []
        fully_cached = True
        for record in selected:
            high = self.stages["high_quality"].get(record["sample_id"], self.fingerprint)
            low = self.stages["low_quality"].get(record["sample_id"], self.fingerprint)
            audit = self.stages["answer_audit"].get(record["sample_id"], self.fingerprint)
            if high:
                cached_high.append({key: value for key, value in high.items() if key != "config_fingerprint"})
            elif low:
                cached_low.append({key: value for key, value in low.items() if key != "config_fingerprint"})
            elif audit:
                if not self._terminal(record["sample_id"]):
                    events = [
                        {
                            "state": "high_answer_gate",
                            "verdict": audit.get("answer_judgment", {}).get("verdict", "uncertain"),
                            "time": time.time(),
                        }
                    ]
                    self._reject(
                        record,
                        "high_quality_answer_mismatch",
                        events,
                        answer_judgment=audit.get("answer_judgment", {}),
                    )
            else:
                fully_cached = False
                break
        if fully_cached:
            return cached_high, cached_low

        high, low = self.clusterer.cluster(selected)
        routed_high: List[Dict[str, Any]] = []
        routed_low: List[Dict[str, Any]] = []
        for record in high:
            if self._terminal(record["sample_id"]):
                continue
            judgment = self.answer_judge.compare(record, str(record["selected_cot"]))
            verdict = judgment.get("verdict", "uncertain")
            if verdict == "equivalent":
                item = {
                    **record,
                    "answer_gate": "pass",
                    "answer_judgment": judgment,
                    "quality_route": "high",
                }
                self._flat_stage("high_quality", item)
                routed_high.append(item)
                continue
            reason = (
                "high_quality_answer_mismatch"
                if verdict == "not_equivalent"
                else "high_quality_answer_uncertain"
            )
            item = {
                **record,
                "answer_gate": "fail" if verdict == "not_equivalent" else "uncertain",
                "answer_judgment": judgment,
                "quality_route": "audit",
                "audit_reason": reason,
            }
            self._flat_stage("answer_audit", item)
            events = [
                {
                    "state": "high_answer_gate",
                    "verdict": verdict,
                    "time": time.time(),
                }
            ]
            self._reject(
                item,
                reason,
                events,
                answer_judgment=judgment,
            )
        for record in low:
            item = {**record, "quality_route": "low"}
            self._flat_stage("low_quality", item)
            routed_low.append(item)
        return routed_high, routed_low

    @staticmethod
    def _synthetic_gap(analysis: Dict[str, Any], verifier: Dict[str, Any]) -> Dict[str, Any]:
        kept = [int(value) for value in analysis.get("supporting_blocks", [])]
        return {
            "dependency_state": "open",
            "repairable": True,
            "block_dependencies": [],
            "dependency_graph": list(analysis.get("dependency_graph", [])),
            "baseline_quality_flags": list(analysis.get("quality_flags", [])),
            "diagnosis_reason": "Verifier requested one constrained repair.",
            "gaps": [
                {
                    "gap_id": "verify_g0",
                    "gap_type": "missing_derivation",
                    "upstream_blocks": kept[:1],
                    "downstream_blocks": kept[-1:] if len(kept) > 1 else [],
                    "target_ids": [
                        int(target["target_id"]) for target in analysis.get("targets", [])
                    ],
                    "missing_claim": ", ".join(verifier.get("failure_codes", [])) or "Verifier failure",
                    "suggested_action": "rewrite_span",
                    "confidence": 0.5,
                }
            ],
        }

    @staticmethod
    def _preserve_required_answer_suffix(original_cot: str, candidate_cot: str) -> str:
        """Restore a boxed final answer excluded by think-only graph analysis."""
        if r"\boxed{" not in original_cot or r"\boxed{" in candidate_cot:
            return candidate_cot
        answer = final_answer(original_cot)
        if not answer:
            return candidate_cot
        return candidate_cot.rstrip() + "\n\nFinal Answer: \\boxed{" + answer + "}"

    def _process_low(self, record: Dict[str, Any]) -> None:
        if self._terminal(record["sample_id"]):
            return
        events: List[Dict[str, Any]] = [{"state": "low_selected", "time": time.time()}]
        original_cot = str(record["selected_cot"])
        started = time.perf_counter()
        analysis, failure = self.analyzer.analyze(record, original_cot, f"{record['sample_id']}:initial")
        analysis_s = time.perf_counter() - started
        if analysis is None:
            self._reject(record, "compress_analysis_failed", events, failure=failure)
            return
        started = time.perf_counter()
        diagnosis = self.diagnoser.diagnose(record, analysis)
        diagnosis_s = time.perf_counter() - started
        events.append(
            {
                "state": "diagnosed",
                "dependency_state": diagnosis["dependency_state"],
                "gap_types": [gap["gap_type"] for gap in diagnosis["gaps"]],
                "duration_s": analysis_s + diagnosis_s,
                "time": time.time(),
            }
        )
        self.stages["diagnosed"].append(
            stage_row(
                record,
                self.fingerprint,
                analysis=analysis,
                diagnosis=diagnosis,
            )
        )
        if diagnosis["dependency_state"] == "uncertain" or (
            diagnosis["dependency_state"] == "open" and not diagnosis["repairable"]
        ):
            self._reject(record, "unrepairable_dependency", events, diagnosis=diagnosis)
            return

        rewrite_rounds = 0
        rewrite_meta: Optional[Dict[str, Any]] = None
        candidate_cot = supporting_text(analysis)
        current_analysis = analysis
        current_diagnosis = diagnosis
        if diagnosis["dependency_state"] == "open":
            try:
                started = time.perf_counter()
                rewrite = self.rewriter.rewrite(record, analysis, diagnosis)
                candidate_cot, rewrite_meta = apply_rewrite(analysis, rewrite)
                rewrite_rounds = 1
                events.append(
                    {
                        "state": "rewritten",
                        "round": 1,
                        "duration_s": time.perf_counter() - started,
                        "time": time.time(),
                    }
                )
                self.stages["rewritten"].append(
                    stage_row(
                        record,
                        self.fingerprint,
                        rewritten_cot=candidate_cot,
                        rewrite=rewrite,
                        rewrite_meta=rewrite_meta,
                    )
                )
            except Exception as exc:
                self._reject(record, "rewrite_failed", events, error=str(exc), diagnosis=diagnosis)
                return

        candidate_cot = self._preserve_required_answer_suffix(original_cot, candidate_cot)
        while True:
            started = time.perf_counter()
            current_analysis, failure = self.analyzer.analyze(
                record,
                candidate_cot,
                f"{record['sample_id']}:verify:{rewrite_rounds}",
            )
            if current_analysis is None:
                self._reject(record, "post_rewrite_analysis_failed", events, failure=failure)
                return
            current_diagnosis = self.diagnoser.diagnose(record, current_analysis)
            verification = self.verifier.verify(
                record,
                original_cot,
                candidate_cot,
                current_analysis,
                current_diagnosis,
                rewrite_meta,
            )
            events.append(
                {
                    "state": "verified",
                    "round": rewrite_rounds,
                    "verdict": verification["verdict"],
                    "failure_codes": verification.get("failure_codes", []),
                    "duration_s": time.perf_counter() - started,
                    "time": time.time(),
                }
            )
            if verification["verdict"] == "pass":
                self._accept(
                    record,
                    candidate_cot,
                    "low_refined" if rewrite_rounds else "low_compressed",
                    events,
                    analysis=current_analysis,
                    diagnosis=current_diagnosis,
                    verification=verification,
                    rewrite_meta=rewrite_meta,
                )
                return
            if (
                rewrite_rounds >= self.config.max_rewrite_rounds
                or not verification.get("repairable")
            ):
                self._reject(
                    record,
                    "verification_failed",
                    events,
                    analysis=current_analysis,
                    diagnosis=current_diagnosis,
                    verification=verification,
                )
                return
            repair_diagnosis = (
                current_diagnosis
                if current_diagnosis.get("gaps")
                else self._synthetic_gap(current_analysis, verification)
            )
            try:
                started = time.perf_counter()
                rewrite = self.rewriter.rewrite(record, current_analysis, repair_diagnosis)
                candidate_cot, rewrite_meta = apply_rewrite(current_analysis, rewrite)
                candidate_cot = self._preserve_required_answer_suffix(
                    original_cot,
                    candidate_cot,
                )
                rewrite_rounds += 1
                events.append(
                    {
                        "state": "rewritten",
                        "round": rewrite_rounds,
                        "duration_s": time.perf_counter() - started,
                        "time": time.time(),
                    }
                )
                self.stages["rewritten"].append(
                    stage_row(
                        record,
                        self.fingerprint,
                        rewritten_cot=candidate_cot,
                        rewrite=rewrite,
                        rewrite_meta=rewrite_meta,
                    )
                )
            except Exception as exc:
                self._reject(record, "rewrite_failed", events, error=str(exc))
                return

    def _repair_missing_lineage(self) -> None:
        for stage_name in ("accepted", "rejected"):
            for row in self.stages[stage_name].records.values():
                if not self.stages["lineage"].get(row["sample_id"], self.fingerprint):
                    self.stages["lineage"].append(
                        stage_row(
                            row,
                            self.fingerprint,
                            terminal_state=stage_name,
                            events=row.get("lineage", []),
                        )
                    )

    def _write_stats(self) -> Dict[str, Any]:
        def rows(name: str) -> List[Dict[str, Any]]:
            return [
                row for (_, fingerprint), row in self.stages[name].records.items()
                if fingerprint == self.fingerprint
            ]

        selected = rows("selected")
        accepted = rows("accepted")
        rejected = rows("rejected")
        stats = {
            "pipeline_version": self.config.pipeline_version,
            "config_fingerprint": self.fingerprint,
            "num_selected": len(selected),
            "num_high": len(rows("high_quality")),
            "num_low": len(rows("low_quality")),
            "num_answer_audit": len(rows("answer_audit")),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "selection_modes": dict(Counter(row.get("selection_mode", "unknown") for row in selected)),
            "accepted_routes": dict(Counter(row.get("route", "unknown") for row in accepted)),
            "answer_judge_verdicts": dict(
                Counter(
                    row.get("answer_judgment", {}).get("verdict", "unknown")
                    for row in rows("high_quality") + rows("answer_audit")
                )
            ),
            "rejection_reasons": dict(Counter(row.get("reason", "unknown") for row in rejected)),
            "gap_types": dict(
                Counter(
                    gap.get("gap_type", "unknown")
                    for row in rows("diagnosed")
                    for gap in row.get("diagnosis", {}).get("gaps", [])
                )
            ),
            "rewrite_actions": dict(
                Counter(
                    row.get("rewrite", {}).get("rewrite_mode", "unknown")
                    for row in rows("rewritten")
                )
            ),
            "verifier_failure_codes": dict(
                Counter(
                    code
                    for row in accepted + rejected
                    for code in row.get("verification", {}).get("failure_codes", [])
                )
            ),
            "config": asdict(self.config),
        }
        self.stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        return stats

    def run(self) -> Dict[str, Any]:
        self._repair_missing_lineage()
        selected = self._select_all()
        high, low = self._route_all(selected)
        for record in high:
            if self._terminal(record["sample_id"]):
                continue
            events = [{"state": "high_selected", "time": time.time()}]
            self._accept(
                record,
                str(record["selected_cot"]),
                "high_direct",
                events,
                verification={
                    "verdict": "pass",
                    "source": "llm_answer_gate",
                    "answer_judgment": record.get("answer_judgment", {}),
                },
            )
        for record in low:
            self._process_low(record)
        return self._write_stats()
