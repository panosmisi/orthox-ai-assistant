from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


def _pct(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.1f}%"


def _claim_by_id(claims: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    return [claim for claim in claims if str(claim.get("claim_id", "")).startswith(prefix)]


def _display_label(label: str | None) -> str:
    if not label:
        return "unspecified anatomy"
    display_map = {
        "knee_lower_leg": "knee/lower leg",
        "wrist_hand": "wrist/hand",
        "ankle_foot": "ankle/foot",
        "pelvis_hip_femur": "pelvis/hip/femur",
        "shoulder_upper_arm": "shoulder/upper arm",
        "leg": "lower limb",
        "hand": "hand/wrist region",
    }
    return display_map.get(label, label.replace("_", " "))


def _source_lookup(sources: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(source.get("source_id")): source for source in sources if source.get("source_id")}


def _citation(source_id: str | None, sources_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source = sources_by_id.get(str(source_id))
    if not source:
        return {"source_id": source_id, "title": None, "url": None, "evidence_level": None}
    return {
        "source_id": source.get("source_id"),
        "title": source.get("title"),
        "url": source.get("url"),
        "evidence_level": source.get("evidence_level"),
        "source_type": source.get("source_type"),
    }


def _finding_level(summary: Stage3InputSummary) -> str:
    if summary.accepted_candidate_count > 0:
        return "fracture_candidate_retained"
    if summary.stage2c_decision == "suspicious":
        return "image_level_suspicion_without_box"
    if summary.stage2c_decision == "uncertain":
        return "image_level_uncertainty_without_box"
    return "no_retained_candidate_with_limitations"


def _evidence_note(claim: Dict[str, Any], sources_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "text": claim.get("claim_text"),
        "claim_id": claim.get("claim_id"),
        "citation": _citation(claim.get("source_id"), sources_by_id),
        "evidence_level": claim.get("evidence_level"),
        "claim_type": claim.get("claim_type", "approved_context"),
    }


def build_clinician_summary(
    summary: Stage3InputSummary,
    clinical_support: Dict[str, Any],
    allowed_claims: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    source_relevance_audit: Dict[str, Any] | None = None,
    max_evidence_notes: int = 5,
) -> Dict[str, Any]:
    """Build a deterministic clinician-facing summary from approved material only."""
    sources_by_id = _source_lookup(sources)
    anatomy = _display_label(summary.anatomy_label or summary.anatomy_parent_label)
    anatomy_conf = _pct(summary.anatomy_confidence)
    primary_conf = _pct(summary.primary_candidate_confidence)
    stage2c_prob = _pct(summary.stage2c_mean_suspicious_probability)
    finding_level = _finding_level(summary)

    if summary.accepted_candidate_count > 0:
        noun = "candidate" if summary.accepted_candidate_count == 1 else "candidates"
        headline = (
            f"AI screening retained {summary.accepted_candidate_count} possible fracture {noun} "
            f"around {anatomy}."
        )
    elif summary.stage2c_decision in {"suspicious", "uncertain"}:
        headline = (
            f"No accepted detection box was retained, but the image-level safety layer marked the study as "
            f"{summary.stage2c_decision} around {anatomy}."
        )
    else:
        headline = (
            f"No high-confidence fracture candidate was retained around {anatomy}, with important limitations."
        )

    confidence_notes = []
    if primary_conf:
        confidence_notes.append(f"Primary candidate support score: {primary_conf}.")
    if stage2c_prob:
        confidence_notes.append(f"Stage 2C suspicious-signal support: {stage2c_prob}.")
    if anatomy_conf:
        confidence_notes.append(f"Anatomy classifier support for displayed label: {anatomy_conf}.")
    if not confidence_notes:
        confidence_notes.append("No calibrated clinical probability is available from this prototype.")

    safety_claims = _claim_by_id(allowed_claims, "project_safety_")
    pipeline_claims = _claim_by_id(allowed_claims, "pipeline_") + _claim_by_id(allowed_claims, "stage2c_")
    evidence_claims = [
        claim
        for claim in allowed_claims
        if str(claim.get("claim_id", "")).startswith(("guideline_", "pubmed_"))
    ]

    evidence_notes = [_evidence_note(claim, sources_by_id) for claim in evidence_claims[:max_evidence_notes]]
    source_audit_summary = (source_relevance_audit or {}).get("summary") or {}

    return {
        "summary_type": "deterministic_clinician_facing",
        "llm_used": False,
        "headline": headline,
        "finding_level": finding_level,
        "not_a_diagnosis": True,
        "human_verification_required": True,
        "case_context": {
            "image_id": summary.image_id,
            "anatomy_label": anatomy,
            "anatomy_output_type": summary.anatomy_output_type,
            "accepted_candidate_count": summary.accepted_candidate_count,
            "raw_candidate_count": summary.raw_candidate_count,
            "rejected_candidate_count": summary.rejected_candidate_count,
            "artifact_rejected_candidate_count": summary.artifact_rejected_candidate_count,
            "stage2c_decision": summary.stage2c_decision,
            "global_uncertainty_flags": summary.global_uncertainty_flags,
        },
        "confidence_notes": confidence_notes,
        "pipeline_interpretation": [
            _evidence_note(claim, sources_by_id) for claim in pipeline_claims[:3]
        ],
        "evidence_backed_context": evidence_notes,
        "clinician_actions_to_consider": clinical_support.get("clinician_review_considerations", [])[:4],
        "limitations": clinical_support.get("limitations", [])[:5],
        "source_audit_summary": {
            "raw_source_count": source_audit_summary.get("raw_source_count"),
            "kept_source_count": source_audit_summary.get("kept_source_count"),
            "excluded_source_count": source_audit_summary.get("excluded_source_count"),
            "kept_pubmed_count": source_audit_summary.get("kept_pubmed_count"),
            "excluded_pubmed_count": source_audit_summary.get("excluded_pubmed_count"),
        },
        "safety_statements": [
            _evidence_note(claim, sources_by_id) for claim in safety_claims[:3]
        ],
    }
