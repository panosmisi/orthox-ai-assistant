from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


LOCAL_CURATED_SOURCE = {
    "source_id": "stage3_curated_safety_context_v0",
    "source_type": "local_curated_project_rule",
    "title": "Stage 3 curated orthopedic safety-context rules v0",
    "citation": "Local deterministic rule set; not a clinical guideline.",
    "url": None,
    "evidence_level": "project_curated_context",
    "notes": "Used only for cautious context/red-flag phrasing until PubMed/guideline retrieval is enabled.",
}


def build_allowed_claims(summary: Stage3InputSummary) -> List[Dict[str, Any]]:
    claims = [
        {
            "claim_id": "project_safety_001",
            "claim_text": "This system is a research prototype and is not a standalone medical diagnosis.",
            "source_id": "stage3_curated_safety_context_v0",
            "evidence_level": "project_safety_statement",
            "anatomy_relevance": "general",
            "allowed": True,
        },
        {
            "claim_id": "project_safety_002",
            "claim_text": "Human clinical verification is required for all outputs.",
            "source_id": "stage3_curated_safety_context_v0",
            "evidence_level": "project_safety_statement",
            "anatomy_relevance": "general",
            "allowed": True,
        },
        {
            "claim_id": "project_safety_003",
            "claim_text": "Model support scores are not calibrated clinical probabilities.",
            "source_id": "stage3_curated_safety_context_v0",
            "evidence_level": "project_safety_statement",
            "anatomy_relevance": "general",
            "allowed": True,
        },
    ]
    if summary.accepted_candidate_count > 0:
        claims.append(
            {
                "claim_id": "pipeline_001",
                "claim_text": "The upstream detector/verifier retained at least one possible fracture candidate.",
                "source_id": "stage3_curated_safety_context_v0",
                "evidence_level": "pipeline_observation",
                "anatomy_relevance": "case_specific_pipeline_output",
                "allowed": True,
            }
        )
    elif summary.stage2c_decision in {"suspicious", "uncertain"}:
        claims.append(
            {
                "claim_id": "pipeline_002",
                "claim_text": "No fracture candidate was retained, but the Stage 2C image-level safety layer raised a warning.",
                "source_id": "stage3_curated_safety_context_v0",
                "evidence_level": "pipeline_observation",
                "anatomy_relevance": "case_specific_pipeline_output",
                "allowed": True,
            }
        )
    else:
        claims.append(
            {
                "claim_id": "pipeline_003",
                "claim_text": "No high-confidence fracture candidate was retained by the upstream pipeline, but subtle fracture cannot be excluded by this system.",
                "source_id": "stage3_curated_safety_context_v0",
                "evidence_level": "pipeline_observation",
                "anatomy_relevance": "case_specific_pipeline_output",
                "allowed": True,
            }
        )
    return claims


def what_pipeline_found(summary: Stage3InputSummary) -> str:
    anatomy = summary.anatomy_label or summary.anatomy_parent_label or "unspecified anatomy"
    if summary.accepted_candidate_count > 0:
        conf = ""
        if summary.primary_candidate_confidence is not None:
            conf = f" The primary detector candidate support score is approximately {summary.primary_candidate_confidence * 100:.1f}%."
        return (
            f"The locked upstream pipeline retained {summary.accepted_candidate_count} possible fracture candidate(s) "
            f"in an X-ray classified around {anatomy}.{conf} This is a screening signal, not a standalone diagnosis."
        )
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        prob = ""
        if summary.stage2c_mean_suspicious_probability is not None:
            prob = f" Stage 2C suspicious-signal support is approximately {summary.stage2c_mean_suspicious_probability * 100:.1f}%."
        return (
            f"The detector/verifier did not retain a fracture box, but Stage 2C produced an image-level "
            f"{summary.stage2c_decision} safety warning for an X-ray classified around {anatomy}.{prob} "
            f"This warning does not localize a fracture."
        )
    return (
        f"The upstream pipeline did not retain a high-confidence fracture candidate in an X-ray classified around {anatomy}. "
        "This does not rule out subtle, non-displaced, or radiographically occult injury."
    )


def build_clinical_support(
    summary: Stage3InputSummary,
    query_pack: Dict[str, Any],
    red_flags: Dict[str, Any],
) -> Dict[str, Any]:
    anatomy = summary.anatomy_label or summary.anatomy_parent_label or "the imaged region"
    context = [
        f"Use the Stage 2B anatomy result ({anatomy}) as context, not as a perfect anatomical ground truth.",
        "Interpret any retained candidate together with clinical history, physical examination, and image quality.",
    ]
    if summary.stage2c_decision in {"suspicious", "uncertain"} and summary.accepted_candidate_count == 0:
        context.append(
            "Because Stage 2C raised a clean-case warning without a retained box, subtle or occult fracture context may be relevant for clinician review."
        )

    differentials = [
        "fracture or cortical disruption in the reported anatomical region",
        "projectional overlap or artifact mimicking a fracture line",
        "normal anatomical variant or growth-plate appearance where relevant",
        "soft-tissue injury with no retained fracture candidate",
    ]

    considerations = [
        "Review the original radiograph rather than relying only on model overlays.",
        "Compare the AI candidate or warning with the site of maximal symptoms, mechanism of injury, and examination findings.",
        "If clinical concern remains despite negative or uncertain radiographs, follow local clinical pathways for further review or imaging.",
        "Use cited evidence/guidelines as background context only; they do not replace clinical judgement for this case.",
    ]

    limitations = [
        "This Stage 3 MVP uses deterministic evidence support and does not generate free-form treatment recommendations.",
        "It does not provide patient-specific diagnosis or treatment.",
        "It cannot rule out fracture when no candidate is retained.",
        "Anatomy classification may fall back to a parent label when fine-label confidence is low.",
        "Stage 2C attention heatmaps, when available, are explanatory attention maps and not fracture boundaries.",
    ]

    return {
        "what_pipeline_found": what_pipeline_found(summary),
        "diagnostic_context": context,
        "differential_considerations": differentials,
        "red_flags_to_check": red_flags,
        "clinician_review_considerations": considerations,
        "limitations": limitations,
        "human_verification_required": True,
    }


def write_template_stage3(
    summary: Stage3InputSummary,
    query_pack: Dict[str, Any],
    red_flags: Dict[str, Any],
    sources: List[Dict[str, Any]] | None = None,
    allowed_claims: List[Dict[str, Any]] | None = None,
    evidence_status: str = "local_curated_only",
    mode: str = "clinician_support",
) -> Dict[str, Any]:
    sources = sources or [LOCAL_CURATED_SOURCE]
    allowed_claims = allowed_claims or build_allowed_claims(summary)
    return {
        "ran": True,
        "version": "stage3_mvp_template_v7",
        "local_only": True,
        "paid_api_used": False,
        "writer": {
            "type": "deterministic_template",
            "model": None,
            "llm_used": False,
        },
        "mode": mode,
        "evidence_status": evidence_status,
        "input_summary": summary.to_dict(),
        "query_pack": query_pack,
        "sources": sources,
        "allowed_claims": allowed_claims,
        "clinical_support": build_clinical_support(summary, query_pack, red_flags),
        "critic": {
            "type": "rule_based",
            "verdict": "NOT_RUN_YET",
            "reasons": [],
        },
        "safety": {
            "not_standalone_diagnosis": True,
            "no_patient_specific_treatment": True,
            "all_medical_claims_cited": True,
            "forbidden_claims_detected": False,
            "safe_fallback_used": False,
        },
    }
