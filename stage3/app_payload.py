from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


def _pct_number(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 3)


def _support_level(value: float | None) -> str:
    if value is None:
        return "unavailable"
    value_f = float(value)
    if value_f < 0.40:
        return "low"
    if value_f < 0.75:
        return "moderate"
    return "high"


def _support_level_label(value: float | None) -> str:
    level = _support_level(value)
    return {
        "low": "Low model signal",
        "moderate": "Moderate model signal",
        "high": "High model signal",
        "unavailable": "Model signal unavailable",
    }[level]


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


def _finding_state(summary: Stage3InputSummary) -> Dict[str, Any]:
    anatomy = _display_label(summary.anatomy_label or summary.anatomy_parent_label)
    if summary.accepted_candidate_count > 0:
        return {
            "state_id": "candidate_retained",
            "severity": "review_required",
            "headline": f"Possible fracture candidate retained around {anatomy}.",
            "subheadline": "The marked region is a screening finding and requires clinician verification.",
            "primary_action": "Review original X-ray and marked region.",
            "why_assigned": (
                "This case received this status because a localized candidate was retained after "
                "the detector/verifier and artifact-suppression stages."
            ),
        }
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        return {
            "state_id": "image_level_warning_without_bbox",
            "severity": "review_required",
            "headline": f"Image-level caution signal around {anatomy}.",
            "subheadline": "No precise box was retained, but the safety layer recommends human review.",
            "primary_action": "Review original X-ray; optional heatmap may guide attention but is not a boundary.",
            "why_assigned": (
                "This case received this status because no localized candidate was retained, "
                "but the image-level caution layer raised a whole-image review signal."
            ),
        }
    return {
        "state_id": "no_high_confidence_candidate_retained",
        "severity": "review_required_with_limitations",
        "headline": f"No high-confidence marked region retained around {anatomy}.",
        "subheadline": "This does not exclude subtle or occult injury.",
        "primary_action": "Interpret with symptoms, examination findings, and clinician judgement.",
        "why_assigned": (
            "This case received this status because no high-confidence localized candidate was retained "
            "and no stronger image-level warning was available. This does not exclude fracture or other injury."
        ),
    }


def _candidate_card_note(state_id: str) -> str:
    if state_id == "candidate_retained":
        return (
            "This internal support score explains why a localized candidate was retained for clinician "
            "review. It is not a calibrated clinical probability and not a diagnosis."
        )
    if state_id == "image_level_warning_without_bbox":
        return "No localized candidate was retained, so no candidate support score is displayed."
    return "No high-confidence localized candidate was retained, so no candidate support score is displayed."


def _localization_card_note(state_id: str) -> str:
    if state_id == "candidate_retained":
        return (
            "A retained bounding box is available from the locked Stage 1/2 outputs. The box is a "
            "review cue, not a fracture boundary or management target. No independent calibrated "
            "localization probability is available."
        )
    if state_id == "image_level_warning_without_bbox":
        return (
            "No localized bounding box was retained. The image-level caution signal does not provide "
            "a fracture boundary or management target."
        )
    return (
        "No localized bounding box was retained. Review should focus on the complete radiographic "
        "study and clinical context."
    )


def _stage2c_card_note(state_id: str) -> str:
    if state_id == "image_level_warning_without_bbox":
        return (
            "This internal image-level support signal can increase review priority when no localized "
            "candidate is retained. It is not a calibrated clinical probability and does not localize a fracture."
        )
    if state_id == "no_high_confidence_candidate_retained":
        return (
            "This internal image-level support score is low and does not exclude fracture or other injury."
        )
    return (
        "Stage 2C is not used to override a retained localized candidate. It remains an internal "
        "whole-image safety signal only."
    )


def _anatomy_ranking(summary: Stage3InputSummary, limit: int = 8) -> List[Dict[str, Any]]:
    ranking = []
    for item in summary.anatomy_probability_ranking[:limit]:
        label = item.get("label") or item.get("class") or item.get("name")
        prob = item.get("probability")
        if prob is None:
            prob = item.get("confidence")
        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            prob_f = None
        ranking.append(
            {
                "label": label,
                "display_label": _display_label(str(label)) if label else None,
                "support_score": prob_f,
                "support_percent": _pct_number(prob_f),
            }
        )
    return ranking


def _first_items(items: List[Any], limit: int) -> List[Any]:
    return [item for item in items if item][:limit]


def _available_support_signals(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals = []
    for card in cards:
        if not card.get("available"):
            continue
        signals.append(
            {
                "id": card.get("id"),
                "label": card.get("label"),
                "percent": card.get("percent"),
                "support_level": card.get("support_level"),
                "support_level_label": card.get("support_level_label"),
                "score_kind": card.get("score_kind"),
                "calibration_note": card.get("calibration_note"),
                "display_note": card.get("display_note"),
            }
        )
    return signals


def _build_clinician_compact_card(
    finding: Dict[str, Any],
    visual_panel: Dict[str, Any],
    anatomy_panel: Dict[str, Any],
    confidence_cards: List[Dict[str, Any]],
    care_panel: Dict[str, Any],
    treatment_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the first-screen clinician card for the future app.

    This is intentionally a compact view of already-approved Stage 3 fields.
    It does not create new medical claims.
    """
    return {
        "version": "stage3_clinician_compact_card_v1",
        "purpose": "First-screen clinician summary for the prototype app.",
        "uses_existing_stage3_fields_only": True,
        "not_a_diagnosis": True,
        "no_patient_specific_treatment": True,
        "human_verification_required": True,
        "headline": finding.get("headline"),
        "key_safety_line": finding.get("subheadline"),
        "state_id": finding.get("state_id"),
        "severity": finding.get("severity"),
        "primary_action": finding.get("primary_action"),
        "visual_priority": visual_panel.get("display_priority"),
        "show_primary_bbox": visual_panel.get("show_primary_bbox"),
        "stage2c_heatmap_available": (visual_panel.get("stage2c_heatmap") or {}).get("available"),
        "stage2c_heatmap_default": visual_panel.get("show_stage2c_heatmap_by_default"),
        "anatomy": {
            "selected_display_label": anatomy_panel.get("selected_display_label"),
            "parent_display_label": anatomy_panel.get("parent_display_label"),
            "output_type": anatomy_panel.get("output_type"),
            "support_percent": anatomy_panel.get("selected_support_percent"),
        },
        "support_signals": _available_support_signals(confidence_cards),
        "top_review_focus": _first_items(care_panel.get("review_focus") or [], 2),
        "top_clinical_context_prompts": _first_items(care_panel.get("clinical_context_prompts") or [], 2),
        "top_radiograph_review_prompts": _first_items(care_panel.get("radiograph_review_prompts") or [], 2),
        "top_treatment_oriented_considerations": _first_items(
            treatment_panel.get("initial_management_considerations") or [], 2
        ),
        "key_uncertainty_notes": _first_items(care_panel.get("uncertainty_notes") or [], 1),
        "key_boundaries": _first_items(care_panel.get("do_not_infer") or [], 2),
        "display_rules": [
            "Show this compact card before the full report.",
            "Keep support scores visible as model support, not clinical probabilities.",
            "Keep human verification visible in the collapsed view.",
            "Do not hide the full report; use the compact card as a front page only.",
        ],
    }


def build_app_payload(
    summary: Stage3InputSummary,
    stage3_block: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact UI/report payload from the full deterministic Stage 3 block."""
    visual = stage3_block.get("visual_evidence") or {}
    clinician = stage3_block.get("clinician_summary") or {}
    patient = stage3_block.get("patient_friendly_note") or {}
    clinical_support = stage3_block.get("clinical_support") or {}
    review_plan = stage3_block.get("review_plan") or {}
    evidence_panel = stage3_block.get("evidence_panel") or {}
    response_guardrails = stage3_block.get("response_guardrails") or {}
    validated_writer = stage3_block.get("validated_writer_response") or {}
    writer_response = validated_writer.get("response") or {}
    care_guidance = stage3_block.get("care_guidance") or {}
    treatment_guidance = stage3_block.get("treatment_guidance") or {}
    rag_evidence_pack = stage3_block.get("rag_evidence_pack") or {}
    safety = stage3_block.get("safety") or {}

    finding = _finding_state(summary)
    state_id = finding.get("state_id")
    primary_bbox = visual.get("primary_bbox") or {}
    stage2c_heatmap = visual.get("stage2c_heatmap") or {}

    confidence_cards = [
        {
            "id": "fracture_candidate_support",
            "label": "Fracture candidate model signal",
            "available": summary.primary_candidate_confidence is not None,
            "score": summary.primary_candidate_confidence,
            "percent": _pct_number(summary.primary_candidate_confidence),
            "support_level": _support_level(summary.primary_candidate_confidence),
            "support_level_label": _support_level_label(summary.primary_candidate_confidence),
            "score_kind": "internal_model_support",
            "display_when": state_id if state_id == "candidate_retained" else "unavailable_no_candidate_retained",
            "display_context": state_id if state_id == "candidate_retained" else "unavailable_no_candidate_retained",
            "calibration_note": (
                "Internal detector/verifier support score, not a calibrated clinical probability "
                "of fracture and not a standalone diagnosis."
            ),
            "display_note": _candidate_card_note(str(state_id)),
        },
        {
            "id": "localization_evidence",
            "label": "Localization evidence",
            "available": bool(primary_bbox.get("available")),
            "score": None,
            "percent": None,
            "support_level": "unavailable",
            "support_level_label": "No calibrated localization probability",
            "score_kind": "bbox_available_no_calibrated_probability",
            "display_when": state_id if state_id == "candidate_retained" else "not_applicable_no_bbox",
            "display_context": state_id if state_id == "candidate_retained" else "not_applicable_no_bbox",
            "calibration_note": (
                "A retained bounding box is available from Stage 1/2. No independent calibrated "
                "probability is available that the box is perfectly localized."
            ),
            "display_note": _localization_card_note(str(state_id)),
        },
        {
            "id": "stage2c_caution_support",
            "label": "Image-level caution model signal",
            "available": summary.stage2c_mean_suspicious_probability is not None,
            "score": summary.stage2c_mean_suspicious_probability,
            "percent": _pct_number(summary.stage2c_mean_suspicious_probability),
            "support_level": _support_level(summary.stage2c_mean_suspicious_probability),
            "support_level_label": _support_level_label(summary.stage2c_mean_suspicious_probability),
            "score_kind": "internal_image_level_support",
            "display_when": "image_level_context" if state_id != "candidate_retained" else "internal_audit_only",
            "display_context": "image_level_context" if state_id != "candidate_retained" else "candidate_retained",
            "calibration_note": (
                "Internal image-level caution score; it does not localize a fracture and is not "
                "a calibrated clinical probability."
            ),
            "display_note": _stage2c_card_note(str(state_id)),
        },
        {
            "id": "anatomy_support",
            "label": "Anatomy label model signal",
            "available": summary.anatomy_confidence is not None,
            "score": summary.anatomy_confidence,
            "percent": _pct_number(summary.anatomy_confidence),
            "support_level": _support_level(summary.anatomy_confidence),
            "support_level_label": _support_level_label(summary.anatomy_confidence),
            "score_kind": "internal_anatomy_support",
            "display_when": "always",
            "display_context": "body_region_classification_only",
            "calibration_note": (
                "Anatomy classifier support only; it does not indicate presence or absence of "
                "fracture. If fine anatomy is uncertain, broader regional context is used."
            ),
            "display_note": "This score supports body-region classification only.",
        },
    ]

    anatomy_panel = {
        "selected_label": summary.anatomy_label,
        "selected_display_label": _display_label(summary.anatomy_label),
        "parent_label": summary.anatomy_parent_label,
        "parent_display_label": _display_label(summary.anatomy_parent_label),
        "output_type": summary.anatomy_output_type,
        "selected_support_score": summary.anatomy_confidence,
        "selected_support_percent": _pct_number(summary.anatomy_confidence),
        "probability_ranking": _anatomy_ranking(summary),
    }
    care_panel = {
        "version": care_guidance.get("version"),
        "dictionary_version": care_guidance.get("dictionary_version"),
        "anatomy_template": care_guidance.get("anatomy_template") or {},
        "wording_policy": care_guidance.get("wording_policy") or {},
        "care_tier": care_guidance.get("care_tier") or {},
        "review_focus": care_guidance.get("review_focus") or [],
        "clinical_context_prompts": care_guidance.get("clinical_context_prompts") or [],
        "radiograph_review_prompts": care_guidance.get("radiograph_review_prompts") or [],
        "review_checks": care_guidance.get("review_checks") or [],
        "pathway_considerations": care_guidance.get("pathway_considerations") or [],
        "uncertainty_notes": care_guidance.get("uncertainty_notes") or [],
        "when_to_escalate_review": care_guidance.get("when_to_escalate_review") or [],
        "do_not_infer": care_guidance.get("do_not_infer") or [],
        "supporting_source_ids": care_guidance.get("supporting_source_ids") or [],
        "template_source_ids": care_guidance.get("template_source_ids") or [],
        "not_a_diagnosis": care_guidance.get("not_a_diagnosis") is True,
        "no_patient_specific_treatment": care_guidance.get("no_patient_specific_treatment") is True,
        "human_verification_required": care_guidance.get("human_verification_required") is True,
    }
    treatment_panel = {
        "version": treatment_guidance.get("version"),
        "guidance_level": treatment_guidance.get("guidance_level"),
        "purpose": treatment_guidance.get("purpose"),
        "app_label": treatment_guidance.get("app_label"),
        "button_label": treatment_guidance.get("button_label"),
        "panel_title": treatment_guidance.get("panel_title"),
        "panel_badge": treatment_guidance.get("panel_badge"),
        "panel_warning": treatment_guidance.get("panel_warning"),
        "safety_disclaimer": treatment_guidance.get("safety_disclaimer"),
        "audience": treatment_guidance.get("audience"),
        "is_treatment_plan": treatment_guidance.get("is_treatment_plan") is True,
        "llm_used": treatment_guidance.get("llm_used") is True,
        "rag_used": treatment_guidance.get("rag_used") is True,
        "not_a_diagnosis": treatment_guidance.get("not_a_diagnosis") is True,
        "not_patient_specific_treatment": treatment_guidance.get("not_patient_specific_treatment") is True,
        "human_verification_required": treatment_guidance.get("human_verification_required") is True,
        "anatomy_context": treatment_guidance.get("anatomy_context") or {},
        "pipeline_context": treatment_guidance.get("pipeline_context") or {},
        "template": treatment_guidance.get("template") or {},
        "anatomy_specific_management_v3": treatment_guidance.get("anatomy_specific_management_v3") or {},
        "state_based_considerations": treatment_guidance.get("state_based_considerations") or [],
        "clinical_localizers": treatment_guidance.get("clinical_localizers") or [],
        "clinical_information_not_available_to_ai": (
            treatment_guidance.get("clinical_information_not_available_to_ai") or []
        ),
        "initial_management_considerations": treatment_guidance.get("initial_management_considerations") or [],
        "red_flags_do_not_miss": treatment_guidance.get("red_flags_do_not_miss") or [],
        "imaging_or_followup_considerations": treatment_guidance.get("imaging_or_followup_considerations") or [],
        "specialist_review_considerations": treatment_guidance.get("specialist_review_considerations") or [],
        "patient_safe_advice": treatment_guidance.get("patient_safe_advice") or [],
        "blocked_outputs": treatment_guidance.get("blocked_outputs") or [],
        "source_policy": treatment_guidance.get("source_policy") or {},
        "supporting_source_ids": treatment_guidance.get("supporting_source_ids") or [],
        "management_guidance": treatment_guidance.get("management_guidance") or {},
    }
    management_panel = treatment_guidance.get("management_guidance") or {}
    rag_panel = {
        "version": rag_evidence_pack.get("version"),
        "rag_available": rag_evidence_pack.get("rag_available") is True,
        "rag_mode": rag_evidence_pack.get("rag_mode"),
        "llm_used": rag_evidence_pack.get("llm_used") is True,
        "public_output_safe": rag_evidence_pack.get("public_output_safe") is True,
        "retrieval_plan": rag_evidence_pack.get("retrieval_plan") or {},
        "source_summary": rag_evidence_pack.get("source_summary") or {},
        "source_lanes": rag_evidence_pack.get("source_lanes") or {},
        "evidence_strength": rag_evidence_pack.get("evidence_strength") or {},
        "readiness_gate": rag_evidence_pack.get("readiness_gate") or {},
        "sources_for_llm": rag_evidence_pack.get("sources_for_llm") or [],
        "claims_for_llm": rag_evidence_pack.get("claims_for_llm") or [],
        "claim_summary": rag_evidence_pack.get("claim_summary") or {},
        "safety_policy": rag_evidence_pack.get("safety_policy") or {},
        "future_llm_contract": rag_evidence_pack.get("future_llm_contract") or {},
    }
    visual_panel = {
        "display_priority": (visual.get("overlays") or {}).get("display_priority"),
        "show_primary_bbox": (visual.get("overlays") or {}).get("show_primary_bbox"),
        "show_stage2c_heatmap_by_default": (visual.get("overlays") or {}).get(
            "show_stage2c_heatmap_by_default"
        ),
        "stage2c_heatmap_display_policy": (visual.get("overlays") or {}).get(
            "stage2c_heatmap_display_policy"
        ),
        "primary_bbox": primary_bbox,
        "stage2c_heatmap": stage2c_heatmap,
    }
    clinician_compact_card = _build_clinician_compact_card(
        finding=finding,
        visual_panel=visual_panel,
        anatomy_panel=anatomy_panel,
        confidence_cards=confidence_cards,
        care_panel=care_panel,
        treatment_panel=treatment_panel,
    )

    return {
        "version": "stage3_app_payload_v1",
        "intended_consumer": "prototype_app_or_report_ui",
        "case_header": {
            "image_id": summary.image_id,
            "image_path": summary.image_path,
            "finding_state": finding,
            "human_verification_required": True,
            "clinician_review_required": True,
            "not_a_diagnosis": True,
        },
        "visual_panel": visual_panel,
        "confidence_cards": confidence_cards,
        "anatomy_panel": anatomy_panel,
        "clinician_compact_card": clinician_compact_card,
        "clinician_panel": {
            "headline": clinician.get("headline"),
            "finding_level": clinician.get("finding_level"),
            "confidence_notes": clinician.get("confidence_notes") or [],
            "pipeline_interpretation": clinician.get("pipeline_interpretation") or [],
            "actions_to_consider": clinician.get("clinician_actions_to_consider") or [],
            "limitations": clinician.get("limitations") or [],
            "red_flags_to_check": clinical_support.get("red_flags_to_check") or {},
        },
        "review_plan_panel": {
            "version": review_plan.get("version"),
            "lane": review_plan.get("case_lane") or {},
            "model_signals": review_plan.get("model_signals") or [],
            "review_tasks": review_plan.get("review_tasks") or [],
            "uncertainty_notes": review_plan.get("uncertainty_notes") or [],
            "escalation_triggers_to_check": review_plan.get("escalation_triggers_to_check") or [],
            "output_boundaries": review_plan.get("output_boundaries") or [],
            "human_verification_required": review_plan.get("human_verification_required") is True,
            "not_a_diagnosis": review_plan.get("not_a_diagnosis") is True,
            "no_patient_specific_treatment": review_plan.get("no_patient_specific_treatment") is True,
        },
        "evidence_panel": {
            "version": evidence_panel.get("version"),
            "evidence_status": evidence_panel.get("evidence_status"),
            "source_summary": evidence_panel.get("source_summary") or {},
            "claim_summary": evidence_panel.get("claim_summary") or {},
            "displayed_sources": evidence_panel.get("displayed_sources") or [],
            "displayed_allowed_claims": evidence_panel.get("displayed_allowed_claims") or [],
            "safety_checks": evidence_panel.get("safety_checks") or {},
            "display_rules": evidence_panel.get("display_rules") or [],
        },
        "response_guardrails_panel": {
            "version": response_guardrails.get("version"),
            "llm_used": response_guardrails.get("llm_used") is True,
            "current_writer_mode": response_guardrails.get("current_writer_mode"),
            "required_output_constraints": response_guardrails.get("required_output_constraints") or [],
            "blocked_content_categories": response_guardrails.get("blocked_content_categories") or [],
            "approved_inputs": response_guardrails.get("approved_inputs") or {},
            "validation": response_guardrails.get("validation") or {},
        },
        "validated_writer_panel": {
            "version": validated_writer.get("version"),
            "writer_type": writer_response.get("writer_type"),
            "llm_used": writer_response.get("llm_used") is True,
            "sections": writer_response.get("sections") or {},
            "used_claim_ids": writer_response.get("used_claim_ids") or [],
            "used_source_ids": writer_response.get("used_source_ids") or [],
            "validation": validated_writer.get("validation") or {},
        },
        "care_guidance_panel": care_panel,
        "management_guidance_panel": management_panel,
        "treatment_guidance_panel": treatment_panel,
        "rag_evidence_panel": rag_panel,
        "patient_panel": {
            "enabled": bool(patient),
            "headline": patient.get("headline"),
            "plain_language_points": patient.get("plain_language_points") or [],
            "not_a_diagnosis": patient.get("not_a_diagnosis", True),
            "no_treatment_advice": patient.get("no_treatment_advice", True),
        },
        "safety_badges": {
            "not_standalone_diagnosis": safety.get("not_standalone_diagnosis") is True,
            "no_patient_specific_treatment": safety.get("no_patient_specific_treatment") is True,
            "human_verification_required": True,
            "scores_are_not_clinical_probabilities": True,
            "stage3_does_not_override_stage1_stage2": True,
        },
        "display_rules": [
            "Always show that this is AI screening support, not diagnosis.",
            "Always keep clinician review required visible.",
            "Do not display injury-exclusion or clearance or reassurance language when no candidate is retained.",
            "Show heatmaps only as optional explainability, not as fracture boundaries.",
            "Display model support scores as technical support, not clinical probabilities.",
        ],
    }
