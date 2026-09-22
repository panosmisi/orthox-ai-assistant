from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary
from .writer_response_validator import build_validator_contract


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


def _state(summary: Stage3InputSummary) -> str:
    if summary.accepted_candidate_count > 0:
        return "candidate_retained"
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        return "image_level_warning_without_bbox"
    return "no_high_confidence_candidate_retained"


def _pipeline_facts(summary: Stage3InputSummary) -> List[Dict[str, Any]]:
    anatomy = _display_label(summary.anatomy_label or summary.anatomy_parent_label)
    facts = [
        {
            "fact_id": "case_state",
            "value": _state(summary),
            "allowed_use": "May describe the upstream pipeline state exactly as model output.",
        },
        {
            "fact_id": "displayed_anatomy",
            "value": anatomy,
            "allowed_use": "May use as anatomical context, not as perfect anatomical ground truth.",
        },
        {
            "fact_id": "accepted_candidate_count",
            "value": summary.accepted_candidate_count,
            "allowed_use": "May report count of retained candidates as AI screening output.",
        },
        {
            "fact_id": "raw_candidate_count",
            "value": summary.raw_candidate_count,
            "allowed_use": "May report only in technical/audit contexts.",
        },
        {
            "fact_id": "anatomy_output_type",
            "value": summary.anatomy_output_type,
            "allowed_use": "May explain parent fallback or fine-label uncertainty.",
        },
    ]
    if summary.primary_candidate_confidence is not None:
        facts.append(
            {
                "fact_id": "primary_candidate_support_score",
                "value": round(float(summary.primary_candidate_confidence) * 100.0, 3),
                "unit": "percent",
                "allowed_use": "May display as technical model support only, not clinical probability.",
            }
        )
    if summary.stage2c_mean_suspicious_probability is not None:
        facts.append(
            {
                "fact_id": "stage2c_caution_support_score",
                "value": round(float(summary.stage2c_mean_suspicious_probability) * 100.0, 3),
                "unit": "percent",
                "allowed_use": "May display as image-level caution support only.",
            }
        )
    if summary.anatomy_confidence is not None:
        facts.append(
            {
                "fact_id": "anatomy_support_score",
                "value": round(float(summary.anatomy_confidence) * 100.0, 3),
                "unit": "percent",
                "allowed_use": "May display as anatomy classifier support only.",
            }
        )
    return facts


def _allowed_claim_refs(allowed_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs = []
    for claim in allowed_claims:
        refs.append(
            {
                "claim_id": claim.get("claim_id"),
                "source_id": claim.get("source_id"),
                "evidence_level": claim.get("evidence_level"),
                "anatomy_relevance": claim.get("anatomy_relevance"),
                "allowed": claim.get("allowed") is True,
            }
        )
    return refs


def build_response_guardrails(
    summary: Stage3InputSummary,
    allowed_claims: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    review_plan: Dict[str, Any],
    evidence_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a future-LLM safety packet without running an LLM.

    The packet is designed for a later constrained writer. It is intentionally
    public-safe: it does not include raw PubMed abstracts or literal unsafe
    phrases that would trip the Stage 3 safety critic.
    """
    source_ids = [source.get("source_id") for source in sources if source.get("source_id")]
    claim_refs = _allowed_claim_refs(allowed_claims)
    invalid_claim_refs = [
        ref.get("claim_id")
        for ref in claim_refs
        if ref.get("allowed") is not True or ref.get("source_id") not in source_ids
    ]
    return {
        "version": "stage3_response_guardrails_v1",
        "purpose": "Safe packet for a future constrained LLM or template writer.",
        "llm_used": False,
        "llm_may_be_used_later": True,
        "current_writer_mode": "deterministic_template_only",
        "input_policy": {
            "may_use_pipeline_facts": True,
            "may_use_allowed_claims": True,
            "may_use_source_metadata": True,
            "may_use_original_image_pixels": False,
            "may_use_unapproved_medical_knowledge": False,
            "may_create_new_medical_claims": False,
        },
        "required_output_constraints": [
            "State that the output is not a standalone diagnosis.",
            "State that human clinical verification is required.",
            "Describe model scores as technical support, not clinical probabilities.",
            "Preserve uncertainty when no retained candidate exists.",
            "Use only allowed claims and pipeline facts.",
            "Do not give patient-specific treatment or clearance/reassurance instructions.",
        ],
        "blocked_content_categories": [
            "diagnostic certainty wording",
            "clearance/reassurance wording",
            "patient-specific treatment instructions",
            "surgical necessity statements",
            "clearance/reassurance or no-review instructions",
            "uncited medical claims",
            "claims based on raw PubMed abstracts in public output",
        ],
        "approved_inputs": {
            "pipeline_facts": _pipeline_facts(summary),
            "allowed_claim_refs": claim_refs,
            "source_ids": source_ids,
            "review_lane": (review_plan.get("case_lane") or {}).get("lane_id"),
            "review_priority": (review_plan.get("case_lane") or {}).get("priority"),
            "evidence_status": evidence_panel.get("evidence_status"),
        },
        "prohibited_inputs": [
            "raw PubMed abstracts",
            "uncurated web text",
            "free-form model speculation",
            "direct pixel reinterpretation by Stage 3",
            "manual assumptions not present in the case JSON",
        ],
        "output_schema_for_future_writer": {
            "clinician_summary": "short cautious paragraph using approved inputs only",
            "patient_note": "plain-language non-diagnostic explanation",
            "safety_footer": "required safety statements",
            "citations": "source ids for every evidence-backed statement",
        },
        "required_validator": build_validator_contract(),
        "validation": {
            "all_claim_refs_allowed_and_sourced": not invalid_claim_refs,
            "invalid_claim_ref_ids": invalid_claim_refs,
            "pubmed_abstracts_excluded": (evidence_panel.get("safety_checks") or {}).get(
                "pubmed_abstracts_not_exposed"
            )
            is True,
            "new_claim_creation_allowed": False,
            "human_verification_required": True,
        },
    }
