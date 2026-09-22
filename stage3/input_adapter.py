from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Stage3InputSummary:
    image_id: str
    image_path: Optional[str]
    source_pipeline_version: Optional[str]
    fracture_candidate_status: Optional[str]
    screening_statement: Optional[str]
    fracture_suspected: Optional[bool]
    accepted_candidate_count: int
    raw_candidate_count: int
    rejected_candidate_count: int
    artifact_rejected_candidate_count: int
    primary_candidate_confidence: Optional[float]
    primary_candidate_bbox_xyxy: Optional[List[float]]
    anatomy_label: Optional[str]
    anatomy_parent_label: Optional[str]
    anatomy_output_type: Optional[str]
    anatomy_confidence: Optional[float]
    anatomy_probability_ranking: List[Dict[str, Any]]
    stage2c_decision: Optional[str]
    stage2c_was_run: Optional[bool]
    stage2c_mean_suspicious_probability: Optional[float]
    stage2c_suspicious_votes: Optional[int]
    stage2c_uncertain_votes: Optional[int]
    heatmap_available: bool
    global_uncertainty_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_stage3_input(pipeline_json: Dict[str, Any]) -> Stage3InputSummary:
    """Extract a stable Stage 3 input summary from a locked pipeline JSON."""
    final = pipeline_json.get("final_assessment") or {}
    anatomy = pipeline_json.get("stage2b_anatomy") or {}
    anatomy_policy = anatomy.get("policy") or {}
    fine_classifier = anatomy.get("fine_classifier") or {}
    parent_classifier = anatomy.get("parent_classifier") or {}
    stage2c = pipeline_json.get("stage2c_safety_check") or {}
    stage2c_prob = stage2c.get("probability_summary") or {}
    primary = final.get("primary_fracture_detection") or {}

    candidate_status = final.get("fracture_candidate_status")
    anatomy_ranking = fine_classifier.get("probability_ranking") or parent_classifier.get("probability_ranking") or []

    heatmap_available = (
        stage2c.get("decision") in {"suspicious", "uncertain"}
        and bool(stage2c.get("stage2c_was_run"))
    )

    flags = list(final.get("reliability_flags") or [])
    if anatomy_policy.get("output_type") == "parent_fallback":
        flags.append("stage3_anatomy_parent_fallback")
    if candidate_status in {"no_candidate_detected", "no_accepted_candidate_stage2c_suspicious", "no_accepted_candidate_stage2c_uncertain"}:
        flags.append("stage3_no_accepted_fracture_box")
    if stage2c.get("decision") in {"suspicious", "uncertain"}:
        flags.append("stage3_stage2c_warning_context")

    return Stage3InputSummary(
        image_id=str(pipeline_json.get("image_id") or ""),
        image_path=pipeline_json.get("image_path"),
        source_pipeline_version=pipeline_json.get("pipeline_version"),
        fracture_candidate_status=candidate_status,
        screening_statement=final.get("screening_statement"),
        fracture_suspected=final.get("fracture_suspected"),
        accepted_candidate_count=_safe_int(final.get("fracture_detection_count")),
        raw_candidate_count=_safe_int(final.get("raw_fracture_candidate_count")),
        rejected_candidate_count=_safe_int(final.get("rejected_fracture_candidate_count")),
        artifact_rejected_candidate_count=_safe_int(final.get("artifact_rejected_fracture_candidate_count")),
        primary_candidate_confidence=_safe_float(primary.get("confidence")),
        primary_candidate_bbox_xyxy=primary.get("xyxy_pixels"),
        anatomy_label=anatomy_policy.get("label"),
        anatomy_parent_label=anatomy_policy.get("parent_label"),
        anatomy_output_type=anatomy_policy.get("output_type"),
        anatomy_confidence=_safe_float(anatomy_policy.get("confidence")),
        anatomy_probability_ranking=anatomy_ranking,
        stage2c_decision=stage2c.get("decision"),
        stage2c_was_run=stage2c.get("stage2c_was_run"),
        stage2c_mean_suspicious_probability=_safe_float(stage2c_prob.get("mean_suspicious_probability")),
        stage2c_suspicious_votes=stage2c_prob.get("suspicious_votes"),
        stage2c_uncertain_votes=stage2c_prob.get("uncertain_votes"),
        heatmap_available=heatmap_available,
        global_uncertainty_flags=sorted(set(flags)),
    )

