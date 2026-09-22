from __future__ import annotations

import json
import re
from typing import Any, Dict, List


DEFAULT_FORBIDDEN_PHRASES = [
    "confirmed fracture",
    "definite fracture",
    "no fracture",
    "normal x-ray",
    "normal xray",
    "all clear",
    "diagnosis is",
    "apply cast",
    "treat with",
    "surgery required",
    "discharge",
    "no doctor needed",
    "you are fine",
    "definitely",
    "certainly",
]

USER_FACING_FORBIDDEN_PATTERNS = [
    ("all_clear", re.compile(r"\ball clear\b", re.IGNORECASE)),
    ("normal_xray", re.compile(r"\bnormal (x-ray|xray|radiograph|study)\b", re.IGNORECASE)),
    ("clean", re.compile(r"\bclean\b", re.IGNORECASE)),
    ("probably_clean", re.compile(r"\bprobably clean\b", re.IGNORECASE)),
    ("negative", re.compile(r"\bnegative\b", re.IGNORECASE)),
    ("confirmed_fracture", re.compile(r"\bconfirmed fracture\b", re.IGNORECASE)),
    ("definite_fracture", re.compile(r"\b(definite|definitive) fracture\b", re.IGNORECASE)),
    ("diagnosis_is", re.compile(r"\bdiagnosis is\b", re.IGNORECASE)),
    ("ruled_out", re.compile(r"\bruled out\b", re.IGNORECASE)),
]

USER_FACING_FORBIDDEN_TREATMENT_PATTERNS = [
    ("apply_cast", re.compile(r"\bapply (a )?cast\b", re.IGNORECASE)),
    ("apply_splint", re.compile(r"\bapply (a )?splint\b", re.IGNORECASE)),
    ("give_medication", re.compile(r"\bgive medication\b", re.IGNORECASE)),
    ("take_medication", re.compile(r"\btake medication\b", re.IGNORECASE)),
    ("allow_weight_bearing", re.compile(r"\ballow weight[- ]bearing\b", re.IGNORECASE)),
    ("non_weight_bearing_order", re.compile(r"\bnon[- ]weight[- ]bearing\b", re.IGNORECASE)),
    ("discharge", re.compile(r"\bdischarg(?:e|ed|ing)\b", re.IGNORECASE)),
    ("safe_to_discharge", re.compile(r"\bsafe to discharge\b", re.IGNORECASE)),
    ("no_followup_needed", re.compile(r"\bno follow[- ]up needed\b", re.IGNORECASE)),
    ("surgery_required", re.compile(r"\bsurgery required\b", re.IGNORECASE)),
    ("treat_as_fracture", re.compile(r"\btreat as (a )?fracture\b", re.IGNORECASE)),
    ("treat_as_sprain", re.compile(r"\btreat as (a )?sprain\b", re.IGNORECASE)),
    ("patient_should", re.compile(r"\bpatient should\b", re.IGNORECASE)),
]

PIPELINE_EXPLANATION_FORBIDDEN_PATTERNS = [
    ("confirmed_fracture", re.compile(r"\bconfirmed fracture\b", re.IGNORECASE)),
    ("fracture_diagnosed", re.compile(r"\bfracture (diagnosed|diagnosis)\b", re.IGNORECASE)),
    ("no_fracture", re.compile(r"\bno fracture\b", re.IGNORECASE)),
    ("fracture_excluded", re.compile(r"\bfracture (excluded|ruled out)\b", re.IGNORECASE)),
    ("normal_xray", re.compile(r"\bnormal (x-ray|xray|radiograph|study)\b", re.IGNORECASE)),
    ("clear_radiograph", re.compile(r"\b(clear|clean) (x-ray|xray|radiograph|study)\b", re.IGNORECASE)),
    ("negative_study", re.compile(r"\bnegative (x-ray|xray|radiograph|study|case)\b", re.IGNORECASE)),
    ("ai_thought", re.compile(r"\bthe ai (thought|believed|decided clinically)\b", re.IGNORECASE)),
    ("chain_of_thought", re.compile(r"\bchain[- ]of[- ]thought\b", re.IGNORECASE)),
    ("internal_reasoning", re.compile(r"\binternal reasoning\b", re.IGNORECASE)),
    ("diagnostic_reasoning", re.compile(r"\bdiagnostic reasoning\b", re.IGNORECASE)),
    ("treatment_target", re.compile(r"\btreatment target\b", re.IGNORECASE)),
]

SAFE_CONTAINING_PHRASES = [
    "no fracture candidate",
    "no high-confidence fracture candidate",
    "no retained fracture candidate",
    "did not retain a fracture",
    "does not rule out fracture",
]


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)


def find_forbidden_phrases(obj: Any, forbidden_phrases: List[str] | None = None) -> List[Dict[str, str]]:
    phrases = forbidden_phrases or DEFAULT_FORBIDDEN_PHRASES
    hits = []
    for text in _walk_strings(obj):
        lowered = text.lower()
        for phrase in phrases:
            phrase_l = phrase.lower()
            if phrase_l in lowered:
                if phrase_l == "no fracture" and any(safe in lowered for safe in SAFE_CONTAINING_PHRASES):
                    continue
                hits.append({"phrase": phrase, "text": text})
    return hits


def validate_stage3_block(stage3_block: Dict[str, Any], forbidden_phrases: List[str] | None = None) -> Dict[str, Any]:
    reasons = []
    forbidden_hits = find_forbidden_phrases(stage3_block, forbidden_phrases)
    if forbidden_hits:
        reasons.append("forbidden_phrases_detected")

    clinical = stage3_block.get("clinical_support") or {}
    safety = stage3_block.get("safety") or {}

    if clinical.get("human_verification_required") is not True:
        reasons.append("missing_human_verification_required")
    if safety.get("not_standalone_diagnosis") is not True:
        reasons.append("missing_not_standalone_diagnosis")
    if safety.get("no_patient_specific_treatment") is not True:
        reasons.append("missing_no_patient_specific_treatment")

    source_ids = {s.get("source_id") for s in stage3_block.get("sources", []) if isinstance(s, dict)}
    for claim in stage3_block.get("allowed_claims", []):
        if not isinstance(claim, dict):
            reasons.append("invalid_claim_object")
            continue
        if claim.get("allowed") is not True:
            reasons.append(f"claim_not_allowed:{claim.get('claim_id')}")
        source_id = claim.get("source_id")
        if not source_id:
            reasons.append(f"claim_missing_source:{claim.get('claim_id')}")
        elif source_id not in source_ids:
            reasons.append(f"claim_source_not_in_sources:{claim.get('claim_id')}")

    verdict = "PASS" if not reasons else "FAIL"
    return {
        "type": "rule_based",
        "verdict": verdict,
        "reasons": reasons,
        "forbidden_hits": forbidden_hits,
    }


def _walk_text_fields(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "case_status",
                "case_status_display",
                "output_type",
                "version",
                "item_id",
                "id",
                "source_id",
                "claim_id",
                "support_ids",
                "support_type",
                "source_type",
                "display_context",
                "evidence_level",
                "url",
                "pmid",
                "doi",
            }:
                continue
            yield from _walk_text_fields(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_text_fields(value)


def _walk_pipeline_visible_text_fields(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "case_status",
                "stage_id",
                "version",
                "technical_details",
                "evidence_trace",
                "raw_case_status",
                "raw_stage_ids",
                "support_scores",
            }:
                continue
            yield from _walk_pipeline_visible_text_fields(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_pipeline_visible_text_fields(value)


def validate_clinical_support_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the final user-facing Stage 3 clinical-support report contract."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(report, dict):
        return {
            "version": "stage3_clinical_support_report_validator_v1",
            "verdict": "FAIL",
            "errors": ["missing_clinical_support_report"],
            "warnings": warnings,
        }

    if report.get("version") != "stage3_clinical_support_report_v1":
        errors.append(f"unexpected_report_version:{report.get('version')}")
    if report.get("academic_prototype") is not True:
        errors.append("academic_prototype_flag_missing")
    if "not for clinical deployment" not in str(report.get("academic_prototype_disclaimer") or "").lower():
        errors.append("academic_prototype_disclaimer_missing")
    if not report.get("why_this_status_was_assigned"):
        errors.append("missing_why_status_was_assigned")

    text_fields = list(_walk_text_fields(report))
    for text in text_fields:
        for pattern_id, pattern in USER_FACING_FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"forbidden_user_facing_phrase:{pattern_id}")
        for pattern_id, pattern in USER_FACING_FORBIDDEN_TREATMENT_PATTERNS:
            if pattern.search(text):
                errors.append(f"forbidden_treatment_instruction:{pattern_id}")

    model = report.get("model_evidence") or {}
    if "not calibrated clinical probabilities" not in str(model.get("score_disclaimer") or "").lower():
        errors.append("missing_model_score_disclaimer")
    anatomy = report.get("anatomy") or {}
    if "body-region classification only" not in str(anatomy.get("anatomy_disclaimer") or "").lower():
        errors.append("missing_anatomy_disclaimer")

    case_status = report.get("case_status")
    status_text = " ".join(str(x) for x in [
        report.get("status_title"),
        report.get("status_explanation"),
        (report.get("clinical_support_summary") or {}).get("suggested_review_focus"),
    ]).lower()
    if case_status == "no_high_confidence_candidate_retained":
        if "does not exclude fracture or other injury" not in status_text:
            errors.append("no_candidate_missing_not_exclude_injury")
        management = report.get("management_context") or {}
        if management.get("guidance_level") != "level_1":
            errors.append(f"no_candidate_unexpected_guidance_level:{management.get('guidance_level')}")
        joined_model = json.dumps(report.get("model_evidence") or {}, ensure_ascii=False).lower()
        if "review should start at the marked region" in joined_model:
            errors.append("no_candidate_model_text_references_marked_region")
    if case_status == "image_level_warning_without_bbox":
        visual = report.get("visual_evidence") or {}
        heatmap_note = str(visual.get("heatmap_safety_note") or "").lower()
        if visual.get("heatmap_available") and "must not be interpreted as a fracture contour" not in heatmap_note:
            errors.append("stage2c_heatmap_missing_contour_warning")
        joined_model = json.dumps(report.get("model_evidence") or {}, ensure_ascii=False).lower()
        if "review should start at the marked region" in joined_model:
            errors.append("stage2c_model_text_references_marked_region")

    management = report.get("management_context") or {}
    if management.get("is_treatment_plan") is not False:
        errors.append("management_context_is_treatment_plan_not_false")
    if management.get("guidance_level") == "level_4":
        errors.append("forbidden_patient_specific_treatment_level")
    considerations = management.get("management_considerations")
    if not isinstance(considerations, list) or not considerations:
        errors.append("management_considerations_missing")
        considerations = []

    source_linked_evidence = report.get("source_linked_evidence") or {}
    pubmed_retrieval = source_linked_evidence.get("pubmed_retrieval") or {}
    evidence_mode = source_linked_evidence.get("evidence_mode") or {}
    if not isinstance(evidence_mode, dict) or not evidence_mode.get("label") or not evidence_mode.get("description"):
        errors.append("evidence_mode_missing")
    if not isinstance(pubmed_retrieval, dict) or not pubmed_retrieval:
        errors.append("pubmed_retrieval_status_missing")
    else:
        pubmed_statement = str(pubmed_retrieval.get("statement") or "").lower()
        if pubmed_retrieval.get("used_for_management_context") and int(pubmed_retrieval.get("source_count") or 0) <= 0:
            errors.append("pubmed_used_for_management_without_sources")
        if case_status == "candidate_retained" and pubmed_retrieval.get("trigger_reason") == "retained_candidate_present":
            if "management" not in pubmed_statement:
                errors.append("candidate_pubmed_statement_missing_management_scope")
            if "not diagnosis" not in pubmed_statement:
                errors.append("candidate_pubmed_statement_missing_not_diagnosis_boundary")
        if case_status == "no_high_confidence_candidate_retained":
            if pubmed_retrieval.get("status") != "skipped":
                errors.append(f"no_candidate_pubmed_not_skipped:{pubmed_retrieval.get('status')}")
            if pubmed_retrieval.get("used_for_management_context") is True:
                errors.append("no_candidate_pubmed_marked_used_for_management")
            if "pubmed-supported" in str(evidence_mode.get("label") or "").lower():
                errors.append("no_candidate_pubmed_supported_label_when_skipped")

    source_cards = (source_linked_evidence.get("source_cards") or [])
    source_ids = {card.get("id") for card in source_cards if isinstance(card, dict)}
    for idx, item in enumerate(considerations):
        if not isinstance(item, dict):
            errors.append(f"management_item_invalid:{idx}")
            continue
        support_ids = item.get("support_ids")
        if not isinstance(support_ids, list) or not support_ids:
            errors.append(f"management_item_missing_support_ids:{item.get('item_id')}")
        else:
            for support_id in support_ids:
                if support_id not in source_ids:
                    errors.append(f"management_item_support_id_not_in_source_cards:{support_id}")
        if not item.get("support_type"):
            errors.append(f"management_item_missing_support_type:{item.get('item_id')}")
        if not item.get("limitation"):
            errors.append(f"management_item_missing_limitation:{item.get('item_id')}")

    if not source_cards:
        errors.append("source_cards_missing")
    for idx, card in enumerate(source_cards):
        if not isinstance(card, dict):
            errors.append(f"source_card_invalid:{idx}")
            continue
        for field in ["id", "title", "source_type", "used_for", "limitations"]:
            if not card.get(field):
                errors.append(f"source_card_missing_{field}:{card.get('id')}")
        if isinstance(card.get("used_for"), list) and any("Guideline or professional-standard context" in str(item) for item in card.get("used_for")):
            errors.append(f"source_card_generic_used_for:{card.get('id')}")
        if card.get("source_type") == "pubmed" and not card.get("pmid"):
            errors.append(f"pubmed_source_card_missing_pmid:{card.get('id')}")

    if anatomy.get("fine_anatomy_uncertain") is True:
        uncertainty = " ".join(str(x) for x in [
            anatomy.get("uncertainty_note"),
            management.get("parent_fallback_note"),
        ]).lower()
        if "fine anatomical localization was uncertain" not in uncertainty:
            errors.append("parent_fallback_missing_uncertainty_text")
        if "do not select a site-specific management pathway" not in uncertainty:
            errors.append("parent_fallback_missing_pathway_warning")

    verdict = "PASS" if not errors else "FAIL"
    return {
        "version": "stage3_clinical_support_report_validator_v1",
        "verdict": verdict,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def validate_ai_pipeline_explanation(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the optional user-facing explanation of the AI pipeline path."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(explanation, dict):
        return {
            "version": "stage3_ai_pipeline_explanation_validator_v1",
            "verdict": "FAIL",
            "errors": ["missing_ai_pipeline_explanation"],
            "warnings": warnings,
        }

    if explanation.get("version") != "stage3_ai_pipeline_explanation_v2":
        errors.append(f"unexpected_explanation_version:{explanation.get('version')}")
    if explanation.get("enabled") is not True:
        errors.append("explanation_not_enabled")
    if explanation.get("uses_llm") is not False:
        errors.append("explanation_uses_llm_not_false")
    if explanation.get("is_diagnostic_reasoning") is not False:
        errors.append("explanation_diagnostic_reasoning_not_false")

    disclaimer = str(explanation.get("safety_disclaimer") or "").lower()
    if "not a diagnosis" not in disclaimer:
        errors.append("missing_not_a_diagnosis_disclaimer")
    if "does not replace clinician review" not in disclaimer:
        errors.append("missing_clinician_review_disclaimer")

    text_fields = list(_walk_text_fields(explanation))
    visible_text_fields = list(_walk_pipeline_visible_text_fields(explanation))
    joined = " ".join(text_fields).lower()
    visible_joined = " ".join(visible_text_fields).lower()
    if "not calibrated clinical probabilities" not in joined:
        errors.append("missing_score_calibration_disclaimer")
    if "body-region classification only" not in joined:
        errors.append("missing_anatomy_scope_disclaimer")

    for text in text_fields:
        lowered = text.lower()
        for pattern_id, pattern in PIPELINE_EXPLANATION_FORBIDDEN_PATTERNS:
            if pattern_id == "no_fracture" and any(safe in lowered for safe in SAFE_CONTAINING_PHRASES):
                continue
            if pattern_id == "treatment_target" and "not" in lowered:
                continue
            if pattern.search(text):
                errors.append(f"forbidden_explanation_phrase:{pattern_id}")

    stages = explanation.get("stage_summaries")
    if not isinstance(stages, list):
        errors.append("stage_summaries_missing")
        stages = []
    stage_ids = {stage.get("stage_id") for stage in stages if isinstance(stage, dict)}
    for required in {"stage1", "stage2a", "stage2a5", "stage2b", "stage2c", "stage3"}:
        if required not in stage_ids:
            errors.append(f"missing_stage_summary:{required}")
    for stage in stages:
        if not isinstance(stage, dict):
            errors.append("invalid_stage_summary_object")
            continue
        for field in [
            "stage_id",
            "stage_display_label",
            "stage_name",
            "purpose",
            "input_used",
            "output",
            "affected_result",
            "how_it_affected_result",
            "limitations",
        ]:
            if not stage.get(field):
                errors.append(f"stage_summary_missing_{field}:{stage.get('stage_id')}")
        effect = str(stage.get("how_it_affected_result") or "")
        if effect.startswith("Yes -") or effect.startswith("No -"):
            errors.append(f"stage_summary_mechanical_yes_no_prefix:{stage.get('stage_id')}")

    case = explanation.get("case_status_explanation") or {}
    state = case.get("case_status")
    raw_statuses = {
        "candidate_retained",
        "image_level_warning_without_bbox",
        "no_high_confidence_candidate_retained",
    }
    visible_status = str(case.get("human_status_label") or case.get("case_status_display") or "")
    if visible_status in raw_statuses:
        errors.append("primary_visible_status_is_raw_enum")
    raw_stage_ids = {"stage1", "stage2a", "stage2a5", "stage2b", "stage2c", "stage3"}
    for raw in raw_statuses | raw_stage_ids:
        if raw in visible_joined:
            errors.append(f"raw_enum_in_visible_explanation:{raw}")
    for field in [
        "human_status_label",
        "short_explanation",
        "why_this_result_was_shown",
        "bottom_line",
        "pipeline_flow",
        "pipeline_flow_text",
        "case_specific_limitation",
    ]:
        value = case.get(field)
        if value is None or value == "" or value == []:
            errors.append(f"case_status_missing_{field}")
    if not isinstance(case.get("pipeline_flow"), list) or len(case.get("pipeline_flow") or []) < 3:
        errors.append("pipeline_flow_too_short")

    evidence_used = explanation.get("evidence_used") or {}
    for field in [
        "localized_bbox",
        "heatmap",
        "image_level_caution",
        "anatomy_label",
        "clinical_symptoms",
        "mechanism_of_injury",
    ]:
        if not evidence_used.get(field):
            errors.append(f"evidence_used_missing_{field}")

    evidence = explanation.get("evidence_trace") or {}
    technical = explanation.get("technical_details") or {}
    if not isinstance(technical, dict) or technical.get("default_collapsed") is not True:
        errors.append("technical_details_missing_or_not_collapsed")
    if technical.get("raw_case_status") != state:
        errors.append("technical_details_raw_case_status_mismatch")
    if set(technical.get("raw_stage_ids") or []) != stage_ids:
        errors.append("technical_details_raw_stage_ids_mismatch")
    if not state:
        errors.append("missing_case_status")
    if state == "candidate_retained":
        if evidence.get("bbox_available") is not True:
            errors.append("candidate_status_without_bbox")
        if "review cue" not in joined and "screening cue" not in joined:
            errors.append("candidate_status_missing_review_cue_language")
    elif state == "image_level_warning_without_bbox":
        if evidence.get("bbox_available") is not False:
            errors.append("stage2c_status_unexpected_bbox")
        if "no precise bounding box" not in joined and "no localized bounding box" not in joined:
            errors.append("stage2c_missing_no_bbox_language")
        if evidence.get("heatmap_available") is True:
            if "paid attention" not in joined and "attention" not in joined:
                errors.append("stage2c_heatmap_missing_attention_language")
            if "does not mark a fracture location" not in joined:
                errors.append("stage2c_heatmap_missing_location_warning")
            if "fracture contour" not in joined or "bounding box" not in joined or "treatment target" not in joined:
                errors.append("stage2c_heatmap_missing_boundary_warning")
    elif state == "no_high_confidence_candidate_retained":
        if evidence.get("bbox_available") is not False:
            errors.append("no_candidate_status_unexpected_bbox")
        if "does not exclude fracture or other injury" not in joined:
            errors.append("no_candidate_missing_not_exclude_injury")
    else:
        warnings.append(f"unknown_case_status:{state}")

    if evidence.get("bbox_modified_by_stage3") is not False:
        errors.append("bbox_modified_by_stage3_not_false")
    if "stage 3 did not run new detection" not in joined:
        errors.append("missing_stage3_no_new_detection_boundary")
    if "stage 3 did not modify bounding boxes" not in joined:
        errors.append("missing_stage3_no_bbox_modification_boundary")
    if "did not decide treatment" not in joined:
        errors.append("missing_treatment_decision_boundary")
    if "clearance" not in joined:
        errors.append("missing_clearance_boundary")

    verdict = "PASS" if not errors else "FAIL"
    return {
        "version": "stage3_ai_pipeline_explanation_validator_v1",
        "verdict": verdict,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def ensure_json_serializable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False
