from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


APP_CONTRACT_VERSION = "app_facing_xray_analysis_contract_v1"

ALLOWED_STATUS_IDS = {
    "candidate_retained",
    "image_level_warning_without_bbox",
    "no_high_confidence_candidate_retained",
}

FORBIDDEN_USER_FACING_PHRASES = [
    "no fracture",
    "fracture ruled out",
    "normal x-ray",
    "normal xray",
    "all clear",
    "diagnosis confirmed",
    "confirmed fracture",
]

LEGACY_STATUS_MAP = {
    "verified_candidate_present": "candidate_retained",
    "no_candidate_detected": "no_high_confidence_candidate_retained",
    "no_accepted_candidate_stage2c_suspicious": "image_level_warning_without_bbox",
    "no_accepted_candidate_stage2c_uncertain": "image_level_warning_without_bbox",
}


def _as_percent(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if value_f <= 1.0:
        return round(value_f * 100.0, 3)
    return round(value_f, 3)


def _clean_list(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return []
    return [item for item in items if item not in (None, "", [])]


def _stage2c_decision(source: Dict[str, Any]) -> str:
    decision = (source.get("stage2c_safety_check") or {}).get("decision")
    if decision:
        return str(decision)
    case = (((source.get("stage3_orthopedic_rag") or {}).get("final_display_payload") or {}).get("case") or {})
    if case.get("state_id") == "candidate_retained":
        return "not_run_existing_candidate_present"
    return "probably_clean"


def _app_state_id(raw_state: Any, source: Dict[str, Any]) -> str | None:
    state = str(raw_state) if raw_state is not None else None
    if state in ALLOWED_STATUS_IDS:
        return state
    if state in LEGACY_STATUS_MAP:
        return LEGACY_STATUS_MAP[state]
    final_status = (source.get("final_assessment") or {}).get("fracture_candidate_status")
    if final_status in LEGACY_STATUS_MAP:
        return LEGACY_STATUS_MAP[final_status]
    return state


def _candidate_region(candidate: Dict[str, Any], index: int) -> Dict[str, Any]:
    probability = candidate.get("candidate_probability_summary") or {}
    bbox_conf = candidate.get("bbox_confidence") or {}
    artifact = candidate.get("artifact_suppression") or {}
    return {
        "candidate_id": f"C{index}",
        "label": "fracture_candidate",
        "bbox": {
            "available": bool(candidate.get("xyxy_pixels")),
            "xyxy_pixels": candidate.get("xyxy_pixels"),
            "xywhn": candidate.get("xywhn"),
            "bbox_quality_percent_proxy": _as_percent(
                bbox_conf.get("bbox_quality_percent_proxy")
                or probability.get("final_fracture_support_percent")
                or candidate.get("confidence")
            ),
            "localization_note": bbox_conf.get("localization_note")
            or "Review cue only; not an exact fracture boundary.",
        },
        "stage1": {
            "detector_support_percent": _as_percent(
                probability.get("detector_fracture_candidate_percent") or candidate.get("confidence")
            ),
            "detector_sources": candidate.get("detector_sources")
            or candidate.get("source_detectors")
            or [candidate.get("candidate_source") or "locked_stage1_union"],
        },
        "stage2a_verifier": {
            "available": candidate.get("verifier_keep") is not None,
            "kept": bool(candidate.get("verifier_keep", candidate.get("final_keep", True))),
            "keep_votes": candidate.get("verifier_keep_votes"),
            "total_votes": candidate.get("verifier_total_votes"),
            "mean_support_percent": _as_percent(
                probability.get("verifier_fracture_probability_percent_mean")
                or candidate.get("verifier_prob_mean")
            ),
        },
        "artifact_suppression": {
            "available": bool(artifact),
            "suppressed": bool(artifact.get("suppressed")) if artifact else False,
            "rule": artifact.get("rule") if artifact else None,
            "flags": _clean_list(artifact.get("flags")) if artifact else [],
            "reasons": _clean_list(artifact.get("reasons")) if artifact else [],
        },
        "display_policy": {
            "review_cue_only": True,
            "not_diagnostic_boundary": True,
        },
    }


def _pubmed_sources(final_display: Dict[str, Any]) -> Dict[str, Any]:
    evidence = final_display.get("evidence") or {}
    sources = evidence.get("displayed_sources") or []
    source_cards = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        source_cards.append(
            {
                "id": item.get("source_id") or item.get("id"),
                "title": item.get("title"),
                "pmid": item.get("pmid"),
                "url": item.get("url"),
                "source_type": item.get("source_type"),
                "supports": item.get("used_for") or item.get("supports") or [],
                "limitations": item.get("limitations") or item.get("limitation") or [],
                "abstract_exposed": item.get("abstract_exposed") is True,
            }
        )
    retrieval_status = "used" if source_cards else "skipped"
    return {
        "retrieval_status": retrieval_status,
        "used_for": "management_context_only",
        "source_cards": source_cards,
        "source_summary": evidence.get("source_summary") or {},
    }


def build_app_facing_contract(final_pipeline_json: Dict[str, Any]) -> Dict[str, Any]:
    """Build the clean app-facing contract from a final integrated pipeline JSON.

    This adapter is presentation normalization only. It does not run inference,
    does not alter model decisions, and does not create medical claims.
    """
    source = final_pipeline_json
    stage3 = source.get("stage3_orthopedic_rag") or {}
    final_display = stage3.get("final_display_payload") or {}
    case = final_display.get("case") or {}
    visual = final_display.get("visual") or {}
    confidence = final_display.get("confidence") or {}
    anatomy = final_display.get("anatomy") or {}
    review = final_display.get("review") or {}
    management = final_display.get("management_guidance") or {}
    care = final_display.get("care_guidance") or {}
    llm = final_display.get("llm") or {}
    safety = final_display.get("safety") or stage3.get("safety") or {}
    exai = final_display.get("ai_pipeline_explanation") or {}
    stage2c = source.get("stage2c_safety_check") or {}
    stage2c_prob = stage2c.get("probability_summary") or {}
    retained_candidates = source.get("stage1_stage2a5_fracture_detections")
    if retained_candidates is None:
        retained_candidates = source.get("stage1_stage2a_fracture_detections") or []

    raw_state_id = case.get("state_id") or ((source.get("final_assessment") or {}).get("fracture_candidate_status"))
    state_id = _app_state_id(raw_state_id, source)
    analysis_mode = "locked_precomputed"
    if source.get("stage12_wrapper_metadata", {}).get("live_pixel_to_stage12_inference") is True:
        analysis_mode = "live_full_pipeline"

    return {
        "version": APP_CONTRACT_VERSION,
        "case": {
            "image_id": case.get("image_id") or source.get("image_id"),
            "image_path": case.get("image_path") or source.get("image_path"),
            "input_source": "fracatlas_validation",
            "analysis_mode": analysis_mode,
            "out_of_distribution": False,
            "academic_prototype": True,
        },
        "input": {
            "image_available": bool(case.get("image_path") or source.get("image_path")),
            "image_quality": {
                "available": False,
                "quality_label": None,
                "notes": [],
            },
            "warnings": [],
        },
        "status": {
            "state_id": state_id,
            "status_tone": case.get("status_tone"),
            "headline": case.get("headline"),
            "subheadline": case.get("subheadline"),
            "primary_action": case.get("primary_action"),
            "requires_human_review": True,
            "source_state_id": raw_state_id,
        },
        "visual": {
            "display_priority": visual.get("display_priority"),
            "show_primary_bbox": visual.get("show_primary_bbox") is True,
            "primary_bbox": visual.get("primary_bbox") or {"available": False},
            "heatmap_available": visual.get("heatmap_available") is True,
            "show_heatmap_by_default": visual.get("show_heatmap_by_default") is True,
            "heatmap_button_label": visual.get("heatmap_button_label"),
            "heatmap_warning": visual.get("heatmap_warning"),
        },
        "candidate_regions": [
            _candidate_region(candidate, idx)
            for idx, candidate in enumerate(retained_candidates or [], start=1)
            if isinstance(candidate, dict)
        ],
        "model_signals": {
            "score_policy": confidence.get("policy")
            or "Internal model support scores, not calibrated clinical probabilities.",
            "cards": confidence.get("cards") or [],
        },
        "anatomy": {
            "selected_label": anatomy.get("selected_label"),
            "selected_display_label": anatomy.get("selected_display_label"),
            "selected_support_percent": anatomy.get("selected_support_percent"),
            "parent_label": anatomy.get("parent_label"),
            "parent_display_label": anatomy.get("parent_display_label"),
            "output_type": anatomy.get("output_type"),
            "top_probabilities": anatomy.get("top_probabilities") or [],
            "display_badges": anatomy.get("display_badges") or [],
            "disclaimer": (
                "Anatomy scores describe body-region classification only. They do not indicate "
                "whether a fracture is present or absent."
            ),
        },
        "stage2c_caution": {
            "ran": stage2c.get("stage2c_was_run") is True,
            "decision": _stage2c_decision(source),
            "warning_visible": state_id == "image_level_warning_without_bbox",
            "support_percent": _as_percent(stage2c_prob.get("mean_suspicious_probability")),
            "bbox_created": False,
            "suppressed_existing_candidate": False,
            "heatmap_available": visual.get("heatmap_available") is True,
            "display_text": stage2c.get("screening_message"),
        },
        "management_context": {
            "enabled": management.get("enabled") is True,
            "label": management.get("label"),
            "is_treatment_plan": management.get("is_treatment_plan") is True,
            "guidance_strength": management.get("guidance_strength"),
            "why_this_section_is_shown": management.get("why_this_section_is_shown"),
            "management_direction": management.get("management_direction") or {},
            "case_management_suggestions": management.get("case_management_suggestions") or {},
            "anatomy_specific_management": management.get("anatomy_specific_management") or {},
            "clinical_inputs_required": _clean_list(management.get("clinical_inputs_required")),
            "red_flags": _clean_list(management.get("red_flags")),
            "ai_must_not_decide": _clean_list(management.get("ai_must_not_decide")),
            "limitations": _clean_list(management.get("limitations")),
            "care_guidance": care,
        },
        "pubmed_sources": _pubmed_sources(final_display),
        "explainability": {
            "enabled": exai.get("enabled") is True,
            "default_collapsed": exai.get("default_collapsed") is True,
            "uses_llm": exai.get("uses_llm") is True,
            "is_diagnostic_reasoning": exai.get("is_diagnostic_reasoning") is True,
            "panel_title": exai.get("panel_title"),
            "case_status_explanation": exai.get("case_status_explanation") or {},
            "stage_summaries": exai.get("stage_summaries") or [],
            "technical_details": exai.get("technical_details") or {},
        },
        "safety": {
            "not_standalone_diagnosis": safety.get("not_standalone_diagnosis", True) is True,
            "human_verification_required": safety.get("human_verification_required", True) is True,
            "scores_are_not_clinical_probabilities": safety.get("scores_are_not_clinical_probabilities", True) is True,
            "stage3_does_not_override_stage1_stage2": safety.get("stage3_does_not_override_stage1_stage2", True) is True,
            "forbidden_user_messages": deepcopy(FORBIDDEN_USER_FACING_PHRASES),
        },
        "technical_audit": {
            "pipeline_version": source.get("pipeline_version"),
            "contract_version": APP_CONTRACT_VERSION,
            "source_payload_version": final_display.get("version"),
            "stage3_version": stage3.get("version"),
            "runner_metadata": stage3.get("runner_metadata") or {},
            "validation": {
                "clinical_support_report": stage3.get("clinical_support_report_validation") or {},
                "ai_pipeline_explanation": stage3.get("ai_pipeline_explanation_validation") or {},
                "llm": llm.get("validation") or {},
            },
            "normalization_notes": [
                "Generated from locked final pipeline JSON.",
                "Presentation contract only; no new inference or medical claim generation.",
            ],
        },
    }


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"forbidden_user_messages", "technical_audit"}:
                continue
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)


def validate_app_facing_contract(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required = [
        "version",
        "case",
        "input",
        "status",
        "visual",
        "candidate_regions",
        "model_signals",
        "anatomy",
        "stage2c_caution",
        "management_context",
        "pubmed_sources",
        "explainability",
        "safety",
        "technical_audit",
    ]
    for key in required:
        if key not in record:
            errors.append(f"missing_top_level_block:{key}")
    if record.get("version") != APP_CONTRACT_VERSION:
        errors.append("wrong_app_contract_version")
    status = record.get("status") or {}
    if status.get("state_id") not in ALLOWED_STATUS_IDS:
        errors.append(f"invalid_status_id:{status.get('state_id')}")
    if status.get("requires_human_review") is not True:
        errors.append("human_review_not_required")
    safety = record.get("safety") or {}
    for flag in [
        "not_standalone_diagnosis",
        "human_verification_required",
        "scores_are_not_clinical_probabilities",
        "stage3_does_not_override_stage1_stage2",
    ]:
        if safety.get(flag) is not True:
            errors.append(f"safety_flag_not_true:{flag}")
    visual = record.get("visual") or {}
    if visual.get("show_heatmap_by_default") is True:
        errors.append("heatmap_shown_by_default")
    stage2c = record.get("stage2c_caution") or {}
    if stage2c.get("bbox_created") is not False:
        errors.append("stage2c_bbox_created_not_false")
    if stage2c.get("suppressed_existing_candidate") is not False:
        errors.append("stage2c_suppressed_existing_candidate_not_false")
    management = record.get("management_context") or {}
    if management.get("is_treatment_plan") is True:
        errors.append("management_context_marked_as_treatment_plan")
    exai = record.get("explainability") or {}
    if exai.get("uses_llm") is True:
        errors.append("explainability_uses_llm")
    if exai.get("is_diagnostic_reasoning") is True:
        errors.append("explainability_marked_diagnostic_reasoning")
    text = "\n".join(_walk_strings(record)).lower()
    for phrase in FORBIDDEN_USER_FACING_PHRASES:
        if phrase.lower() in text:
            errors.append(f"forbidden_user_facing_phrase:{phrase}")
    return sorted(set(errors))
