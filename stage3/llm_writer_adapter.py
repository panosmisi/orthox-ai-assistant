from __future__ import annotations

import json
from typing import Any, Dict, List

from .constrained_treatment_writer import build_constrained_treatment_writer_response
from .writer_response_validator import build_validator_contract, validate_writer_response


ADAPTER_VERSION = "stage3_llm_writer_adapter_v1"
PLANNED_MODEL_FAMILY = "medgemma_4b_it"


def _pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) * 100.0, 3)
    except (TypeError, ValueError):
        return None


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_source(source: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only source metadata that is safe for an LLM writer packet."""
    out = {
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "title": source.get("title"),
        "organization": source.get("organization"),
        "url": source.get("url"),
        "evidence_level": source.get("evidence_level"),
        "relevance": source.get("relevance") or source.get("source_relevance") or {},
        "pubmed_quality": source.get("pubmed_quality"),
    }
    if source.get("source_type") == "pubmed":
        out["pubmed"] = {
            "pmid": source.get("pmid"),
            "journal": source.get("journal"),
            "year": source.get("year"),
            "publication_types": source.get("publication_types") or [],
            "abstract_available": source.get("abstract_available") is True,
            "abstract_exposed": False,
        }
    return out


def _claim_for_packet(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("claim_text"),
        "source_id": claim.get("source_id"),
        "claim_type": claim.get("claim_type") or "controlled_context",
        "evidence_level": claim.get("evidence_level"),
        "anatomy_relevance": claim.get("anatomy_relevance"),
        "allowed_for_treatment_guidance": claim.get("allowed_for_treatment_guidance") is True,
        "allowed_for_diagnosis_context": claim.get("allowed_for_diagnosis_context") is True,
        "pubmed_source_role": claim.get("pubmed_source_role"),
        "pubmed_quality_tier": claim.get("pubmed_quality_tier"),
    }


def _allowed_claims_for_packet(stage3_block: Dict[str, Any], max_claims: int) -> List[Dict[str, Any]]:
    rag = stage3_block.get("rag_evidence_pack") or {}
    rag_claims = [
        claim
        for claim in _safe_list(rag.get("claims_for_llm"))
        if isinstance(claim, dict) and claim.get("allowed_for_llm") is True
    ]
    if rag_claims:
        return [_claim_for_packet(claim) for claim in rag_claims[:max_claims]]

    claims = [
        claim
        for claim in _safe_list(stage3_block.get("allowed_claims"))
        if isinstance(claim, dict) and claim.get("allowed") is True
    ]
    return [_claim_for_packet(claim) for claim in claims[:max_claims]]


def _sources_for_packet(stage3_block: Dict[str, Any], claim_cards: List[Dict[str, Any]], max_sources: int) -> List[Dict[str, Any]]:
    rag = stage3_block.get("rag_evidence_pack") or {}
    rag_sources = [s for s in _safe_list(rag.get("sources_for_llm")) if isinstance(s, dict)]
    source_ids_needed = {str(claim.get("source_id")) for claim in claim_cards if claim.get("source_id")}
    if rag_sources:
        sources = [_safe_source(source) for source in rag_sources if str(source.get("source_id")) in source_ids_needed]
    else:
        sources = [
            _safe_source(source)
            for source in _safe_list(stage3_block.get("sources"))
            if isinstance(source, dict) and str(source.get("source_id")) in source_ids_needed
        ]
    if not sources:
        sources = [_safe_source(source) for source in _safe_list(stage3_block.get("sources")) if isinstance(source, dict)]
    return sources[:max_sources]


def _confidence_packet(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    summary = stage3_block.get("input_summary") or {}
    app_payload = stage3_block.get("app_payload") or {}
    visual = stage3_block.get("visual_evidence") or {}
    primary_bbox = visual.get("primary_bbox") or {}
    stage2c_heatmap = visual.get("stage2c_heatmap") or {}
    final_candidate = {
        "fracture_candidate_status": summary.get("fracture_candidate_status"),
        "accepted_candidate_count": summary.get("accepted_candidate_count"),
        "raw_candidate_count": summary.get("raw_candidate_count"),
        "rejected_candidate_count": summary.get("rejected_candidate_count"),
        "artifact_rejected_candidate_count": summary.get("artifact_rejected_candidate_count"),
        "primary_candidate_support_score": summary.get("primary_candidate_confidence"),
        "primary_candidate_support_percent": _pct(summary.get("primary_candidate_confidence")),
        "primary_bbox_xyxy_pixels": summary.get("primary_candidate_bbox_xyxy"),
        "bbox_quality_score_proxy": primary_bbox.get("bbox_quality_score_proxy"),
        "bbox_quality_percent_proxy": primary_bbox.get("bbox_quality_percent_proxy"),
        "bbox_localization_note": primary_bbox.get("localization_note"),
    }
    stage2c = {
        "decision": summary.get("stage2c_decision"),
        "was_run": summary.get("stage2c_was_run"),
        "mean_suspicious_probability": summary.get("stage2c_mean_suspicious_probability"),
        "mean_suspicious_percent": _pct(summary.get("stage2c_mean_suspicious_probability")),
        "suspicious_votes": summary.get("stage2c_suspicious_votes"),
        "uncertain_votes": summary.get("stage2c_uncertain_votes"),
        "heatmap_available": summary.get("heatmap_available"),
        "heatmap_display_policy": stage2c_heatmap.get("display_policy"),
    }
    anatomy = {
        "selected_label": summary.get("anatomy_label"),
        "parent_label": summary.get("anatomy_parent_label"),
        "output_type": summary.get("anatomy_output_type"),
        "support_score": summary.get("anatomy_confidence"),
        "support_percent": _pct(summary.get("anatomy_confidence")),
        "top_probability_ranking": (app_payload.get("anatomy_panel") or {}).get("probability_ranking")
        or summary.get("anatomy_probability_ranking")
        or [],
    }
    return {
        "score_policy": {
            "display_all_scores": True,
            "scores_are_model_support_not_clinical_probabilities": True,
            "llm_must_not_convert_scores_to_diagnosis_probability": True,
        },
        "fracture_detection": final_candidate,
        "stage2c_image_level_safety": stage2c,
        "anatomy_classification": anatomy,
        "confidence_cards": _safe_list(app_payload.get("confidence_cards")),
    }


def build_llm_writer_input_packet(
    stage3_block: Dict[str, Any],
    max_claims: int = 12,
    max_sources: int = 8,
) -> Dict[str, Any]:
    """Build the only packet a future LLM writer should receive.

    The packet intentionally excludes raw PubMed abstracts and raw image pixels.
    It is designed for a writer model, not for a diagnostic model.
    """
    claim_cards = _allowed_claims_for_packet(stage3_block, max_claims=max_claims)
    sources = _sources_for_packet(stage3_block, claim_cards, max_sources=max_sources)
    app_payload = stage3_block.get("app_payload") or {}
    return {
        "version": f"{ADAPTER_VERSION}_input_packet",
        "intended_model_role": "controlled_medical_writer_not_diagnostician",
        "planned_model_family": PLANNED_MODEL_FAMILY,
        "llm_called": False,
        "case_identity": {
            "image_id": (stage3_block.get("input_summary") or {}).get("image_id"),
            "source_pipeline_version": (stage3_block.get("input_summary") or {}).get("source_pipeline_version"),
        },
        "pipeline_context": {
            "finding_state": ((app_payload.get("case_header") or {}).get("finding_state") or {}),
            "review_lane": ((stage3_block.get("review_plan") or {}).get("case_lane") or {}),
            "global_uncertainty_flags": (stage3_block.get("input_summary") or {}).get("global_uncertainty_flags") or [],
            "stage3_does_not_override_stage1_stage2": True,
        },
        "confidence_context": _confidence_packet(stage3_block),
        "care_guidance": stage3_block.get("care_guidance") or {},
        "treatment_guidance": stage3_block.get("treatment_guidance") or {},
        "approved_claims": claim_cards,
        "approved_sources": sources,
        "forbidden_outputs": [
            "standalone diagnosis",
            "confirmed fracture",
            "fracture ruled out",
            "all clear",
            "no doctor needed",
            "patient-specific treatment instruction",
            "surgery required",
            "apply cast",
            "discharge advice",
            "new uncited medical claim",
            "raw PubMed abstract text",
            "new interpretation of image pixels",
        ],
        "required_output_schema": {
            "writer_type": "llm_constrained_writer_candidate",
            "llm_used": True,
            "sections": {
                "clinician_summary": "short cautious clinician-facing paragraph",
                "patient_note": "plain-language note with limitations",
                "safety_footer": "required safety language",
            },
            "used_claim_ids": ["claim ids used from approved_claims only"],
            "used_source_ids": ["source ids used from approved_sources only"],
            "created_new_medical_claims": False,
            "used_raw_pubmed_abstracts": False,
            "reinterpreted_image_pixels": False,
        },
    }


def build_llm_writer_prompt(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Build deterministic prompt messages for a future local medical LLM."""
    system_message = (
        "You are a controlled medical writing assistant for an orthopedic X-ray decision-support prototype. "
        "You are not a radiologist, not a treating clinician, and not a diagnostic authority. "
        "Use only the provided JSON packet. Do not add medical facts, do not reinterpret image pixels, "
        "do not use raw PubMed abstracts, and do not convert AI model scores into clinical probabilities."
    )
    developer_message = (
        "Return only valid JSON matching required_output_schema. "
        "Every evidence-backed statement must be grounded in approved_claims and declared in used_claim_ids. "
        "Every used claim source must be declared in used_source_ids. "
        "Mention confidence scores as technical model support scores when relevant. "
        "Preserve uncertainty, require human verification, and avoid patient-specific treatment instructions."
    )
    return {
        "version": f"{ADAPTER_VERSION}_prompt_contract",
        "llm_called": False,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "developer", "content": developer_message},
            {
                "role": "user",
                "content_type": "application/json",
                "content": packet,
            },
        ],
        "generation_policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 700,
            "json_only": True,
            "no_chain_of_thought": True,
        },
    }


def validate_llm_writer_candidate(stage3_block: Dict[str, Any], candidate_response: Dict[str, Any]) -> Dict[str, Any]:
    result = validate_writer_response(stage3_block, candidate_response)
    errors = list(result.get("errors") or [])
    warnings = list(result.get("warnings") or [])

    if candidate_response.get("writer_type") != "llm_constrained_writer_candidate":
        errors.append(f"unexpected_llm_writer_type:{candidate_response.get('writer_type')}")
    if candidate_response.get("llm_used") is not True:
        errors.append("llm_used_not_true_for_llm_candidate")

    sections = candidate_response.get("sections") or {}
    if isinstance(sections, dict):
        joined = "\n".join(str(v) for v in sections.values()).lower()
        if "```" in joined:
            errors.append("markdown_code_fence_in_llm_response")
        if "as an ai" in joined or "i cannot" in joined:
            warnings.append("assistant_meta_language_detected")
        if "%" not in joined and "percent" not in joined:
            errors.append("no_confidence_score_mentioned")

    result["errors"] = errors
    result["warnings"] = warnings
    result["error_count"] = len(errors)
    result["warning_count"] = len(warnings)
    result["verdict"] = "PASS" if not errors else "FAIL"
    result["strict_llm_candidate_checks"] = {
        "expected_writer_type": "llm_constrained_writer_candidate",
        "llm_used_must_be_true": True,
        "markdown_code_fences_allowed": False,
        "confidence_scores_expected_in_text": True,
        "missing_confidence_score_is_hard_fail": True,
    }
    return result


def build_llm_writer_fallback(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    return build_constrained_treatment_writer_response(stage3_block)


def build_llm_writer_adapter_contract(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    packet = build_llm_writer_input_packet(stage3_block)
    prompt_contract = build_llm_writer_prompt(packet)
    fallback = build_llm_writer_fallback(stage3_block)
    return {
        "version": ADAPTER_VERSION,
        "purpose": "Safety contract and input packet for a future local medical LLM writer.",
        "status": "contract_only_no_llm_call",
        "planned_model_family": PLANNED_MODEL_FAMILY,
        "llm_called": False,
        "llm_output_display_allowed_without_validation": False,
        "input_packet": packet,
        "prompt_contract": prompt_contract,
        "required_validator": build_validator_contract(),
        "fallback_if_llm_missing_or_validation_fails": fallback,
        "safety_policy": {
            "raw_image_pixels_allowed": False,
            "raw_pubmed_abstracts_allowed": False,
            "unapproved_medical_knowledge_allowed": False,
            "new_medical_claim_creation_allowed": False,
            "must_display_confidence_scores_as_model_support": True,
            "must_state_human_verification_required": True,
            "must_state_not_a_standalone_diagnosis": True,
        },
        "adapter_audit": {
            "approved_claim_count": len(packet["approved_claims"]),
            "approved_source_count": len(packet["approved_sources"]),
            "confidence_card_count": len(packet["confidence_context"].get("confidence_cards") or []),
            "raw_pubmed_abstract_field_present": _raw_pubmed_abstract_field_present(packet),
            "fallback_validation_verdict": (fallback.get("validation") or {}).get("verdict"),
        },
    }


def _raw_pubmed_abstract_field_present(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "abstract" and isinstance(item, str) and item.strip():
                return True
            if _raw_pubmed_abstract_field_present(item):
                return True
    if isinstance(value, list):
        return any(_raw_pubmed_abstract_field_present(item) for item in value)
    return False
