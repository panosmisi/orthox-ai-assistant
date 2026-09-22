from __future__ import annotations

from typing import Any, Dict, List


PIPELINE_EXPLANATION_VERSION = "stage3_ai_pipeline_explanation_v2"

SAFETY_DISCLAIMER = (
    "This explanation describes AI pipeline outputs. It is not a diagnosis "
    "and does not replace clinician review."
)

SCORE_DISCLAIMER = "Model scores are internal support signals, not calibrated clinical probabilities."

ANATOMY_DISCLAIMER = (
    "Anatomy output describes body-region classification only. It does not indicate "
    "whether a fracture is present or absent."
)

HUMAN_STATUS_LABELS = {
    "candidate_retained": "Localized AI review cue retained",
    "image_level_warning_without_bbox": "Image-level caution without localized box",
    "no_high_confidence_candidate_retained": "No high-confidence AI review cue retained",
}

STAGE_DISPLAY_LABELS = {
    "stage1": "Candidate detection",
    "stage2a": "Candidate verification",
    "stage2a5": "Artifact suppression",
    "stage2b": "Anatomy classification",
    "stage2c": "Image-level caution",
    "stage3": "Report assembly",
}


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "not available"


def _yes_no(value: bool | None, *, unavailable: str = "Not available") -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return unavailable


def _display_label(value: Any) -> str:
    if not value:
        return "unspecified region"
    return str(value)


def _finding_state(app_payload: Dict[str, Any]) -> Dict[str, Any]:
    return ((app_payload.get("case_header") or {}).get("finding_state") or {})


def _card(cards: List[Dict[str, Any]], card_id: str) -> Dict[str, Any]:
    for card in cards:
        if isinstance(card, dict) and card.get("id") == card_id:
            return card
    return {}


def _stage2c_decision(visual_panel: Dict[str, Any]) -> str | None:
    heatmap = visual_panel.get("stage2c_heatmap") or {}
    return heatmap.get("decision")


def _stage1_output(final_assessment: Dict[str, Any]) -> str:
    raw = int(final_assessment.get("raw_fracture_candidate_count") or 0)
    retained = int(final_assessment.get("fracture_detection_count") or 0)
    if raw > 0 and retained > 0:
        return f"{raw} visual candidate region(s) were proposed; {retained} remained available for final display."
    if raw > 0:
        return f"{raw} visual candidate region(s) were proposed, but none survived into the final localized output."
    return "No localized candidate region was available from the candidate-detection lane."


def _stage2a_output(final_assessment: Dict[str, Any]) -> str:
    retained = int(final_assessment.get("fracture_detection_count") or 0)
    verifier_rejected = int(final_assessment.get("verifier_rejected_fracture_candidate_count") or 0)
    if retained > 0:
        return f"The final retained candidate count after verification was {retained}."
    if verifier_rejected > 0:
        return f"The verification lane removed or weakened {verifier_rejected} candidate(s)."
    return "There was no retained localized candidate for final verification."


def _stage2a5_output(final_assessment: Dict[str, Any]) -> str:
    rejected = int(final_assessment.get("artifact_rejected_fracture_candidate_count") or 0)
    if rejected > 0:
        return f"Artifact suppression removed {rejected} candidate(s) from the final visual-output lane."
    return "Artifact suppression did not provide a separate removal event for this case."


def _stage2b_output(anatomy_panel: Dict[str, Any]) -> str:
    selected = anatomy_panel.get("selected_display_label") or anatomy_panel.get("parent_display_label")
    output_type = anatomy_panel.get("output_type")
    support = anatomy_panel.get("selected_support_percent")
    if selected:
        if output_type == "parent_fallback":
            return (
                f"The anatomy classifier selected broader regional context: {_display_label(selected)} "
                f"with internal support {_fmt_percent(support)} because fine localization was uncertain."
            )
        return f"The anatomy classifier selected {_display_label(selected)} with internal support {_fmt_percent(support)}."
    return "The anatomy classifier did not provide a usable display label."


def _stage2c_output(state_id: str | None, decision: str | None, heatmap_available: bool, support: Any) -> str:
    if state_id == "candidate_retained":
        return "Stage 2C was not needed for the final status because a localized candidate was already retained."
    if decision in {"suspicious", "uncertain"}:
        return (
            "The image-level caution layer raised a whole-image review signal "
            f"with internal support {_fmt_percent(support)}. Heatmap available: {_yes_no(heatmap_available)}."
        )
    return "No stronger image-level caution signal was retained for the final report."


def _human_status_label(state_id: str | None) -> str:
    return HUMAN_STATUS_LABELS.get(str(state_id or ""), "Limited AI screening output")


def _variant_index(seed: Any, family: str, count: int) -> int:
    if count <= 1:
        return 0
    key = f"{seed or ''}:{family}"
    return sum(ord(ch) for ch in key) % count


def _short_explanation(state_id: str | None, image_id: Any = None) -> str:
    if state_id == "candidate_retained":
        variants = [
            (
                "The system kept a localized AI review cue from the earlier detection pipeline. "
                "Stage 3 displayed this marked region for clinician review, but it is not a diagnosis."
            ),
            (
                "A localized review cue remained after the detection and filtering steps. Stage 3 displayed "
                "that marked region as an attention aid, not as proof of fracture or diagnosis."
            ),
            (
                "The upstream pipeline retained one localized visual cue. Stage 3 carried that cue into the "
                "report so it can focus review attention without deciding the clinical finding."
            ),
            (
                "A marked review cue remained available after the earlier AI checks. Stage 3 displayed it as "
                "a place to start review, while keeping the boundary that the cue is not a clinical conclusion."
            ),
            (
                "The detection pathway left a localized cue in the final output. Stage 3 showed that cue in "
                "the report to support review attention, not to state a diagnosis."
            ),
        ]
        return variants[_variant_index(image_id, "short_candidate", len(variants))]
    if state_id == "image_level_warning_without_bbox":
        variants = [
            (
                "The system did not keep a precise bounding box, but the image-level caution layer raised "
                "a whole-image review signal. Stage 3 therefore shows an image-level caution rather than a localized cue."
            ),
            (
                "No localized box survived into the final output. A separate image-level caution signal was "
                "retained, so Stage 3 reports a whole-image review warning instead of a marked region."
            ),
            (
                "The localized detection path did not produce a final box, but the image-level safety lane "
                "still raised caution. Stage 3 presents this as non-localized review attention."
            ),
            (
                "No precise local marker was kept for display. Because the whole-image caution lane remained "
                "active, Stage 3 reported a broader review warning instead of drawing a box."
            ),
            (
                "The pipeline did not retain a localized review cue, but it did retain a whole-image caution "
                "signal. The final report therefore asks for review attention without claiming a location."
            ),
        ]
        return variants[_variant_index(image_id, "short_stage2c", len(variants))]
    variants = [
        (
            "The system did not keep a high-confidence localized AI review cue and did not retain a stronger "
            "image-level warning. This is limited AI screening output and does not exclude fracture or other injury."
        ),
        (
            "No localized review cue or stronger image-level caution was retained for the final display. "
            "The result is limited AI screening output, not evidence that injury is absent."
        ),
        (
            "The pipeline did not retain a final localized cue and did not raise a stronger whole-image warning. "
            "This means the AI output is limited and still requires clinical review."
        ),
        (
            "The final display did not receive a retained localized cue or a stronger whole-image caution signal. "
            "This is a limited pipeline result, not a statement that injury is absent."
        ),
        (
            "The AI pipeline ended without a localized cue to show and without an additional image-level warning. "
            "That output remains limited and must be interpreted with human review."
        ),
    ]
    return variants[_variant_index(image_id, "short_no_candidate", len(variants))]


def _causal_reason(state_id: str | None, image_id: Any = None) -> str:
    if state_id == "candidate_retained":
        variants = [
            (
                "Stage 1 proposed a localized visual candidate. The later filtering stages did not remove it, "
                "so Stage 3 displayed it as a retained AI review cue. This cue should guide review attention, "
                "not determine diagnosis or treatment."
            ),
            (
                "A candidate region was proposed and remained available after the verification and artifact checks. "
                "Because it was still retained, Stage 3 displayed it as a localized review cue rather than a diagnosis."
            ),
            (
                "The detection lane produced a localized candidate that survived the downstream filters. Stage 3 "
                "therefore showed the marked cue while preserving the boundary that it cannot decide treatment or diagnosis."
            ),
            (
                "A localized candidate was created upstream and remained after the checks designed to reduce weak "
                "or artifact-like cues. Stage 3 therefore kept it visible as a review cue only."
            ),
            (
                "The candidate-detection path left a localized cue in the final structured output. Because no later "
                "stage removed it, Stage 3 displayed the cue while keeping it separate from diagnosis and treatment."
            ),
        ]
        return variants[_variant_index(image_id, "reason_candidate", len(variants))]
    if state_id == "image_level_warning_without_bbox":
        variants = [
            (
                "No localized bounding box was retained from the detection pipeline. However, Stage 2C raised "
                "an image-level caution signal, so Stage 3 reported a whole-image review warning instead of a localized cue."
            ),
            (
                "The localized candidate path ended without a retained box. Because Stage 2C still produced an "
                "image-level caution signal, Stage 3 used the whole-image warning lane for the final display."
            ),
            (
                "The earlier stages did not leave a precise box to display. Stage 2C supplied the remaining caution "
                "signal, so the final status became image-level review attention without localization."
            ),
            (
                "The localized candidate lane did not produce a box for the final report. Stage 2C still produced "
                "a caution signal, so Stage 3 used the non-localized warning status."
            ),
            (
                "There was no retained localized cue to draw. The available caution came from the image-level lane, "
                "which is why Stage 3 displayed a whole-image warning rather than a bounding box."
            ),
        ]
        return variants[_variant_index(image_id, "reason_stage2c", len(variants))]
    variants = [
        (
            "The detection and filtering pipeline did not retain a high-confidence localized cue, and no stronger "
            "image-level warning was available. Stage 3 therefore reported limited AI screening output. "
            "This does not exclude fracture or other injury."
        ),
        (
            "No localized cue survived the detection/filtering path, and Stage 2C did not add a stronger image-level "
            "warning. Stage 3 therefore displayed limited screening output rather than reassurance."
        ),
        (
            "The upstream stages did not provide a retained box or a stronger whole-image caution signal. Stage 3 "
            "reported that limited pipeline state while keeping the warning that injury is not excluded."
        ),
        (
            "The localized detection path and the image-level caution path both ended without a stronger retained "
            "cue for display. Stage 3 therefore used the limited screening status and preserved the warning that "
            "fracture or other injury is not excluded."
        ),
        (
            "No final box and no stronger whole-image caution signal were available to Stage 3. The report therefore "
            "shows limited AI screening output and keeps clinician review as necessary because injury is not excluded."
        ),
    ]
    return variants[_variant_index(image_id, "reason_no_candidate", len(variants))]


def _pipeline_flow(state_id: str | None) -> List[str]:
    if state_id == "candidate_retained":
        return [
            "Candidate proposed",
            "Candidate retained after filtering",
            "Anatomy context added",
            "Stage 3 displayed localized review cue",
        ]
    if state_id == "image_level_warning_without_bbox":
        return [
            "No localized candidate retained",
            "Image-level caution raised",
            "Anatomy context added",
            "Stage 3 displayed whole-image caution",
        ]
    return [
        "No high-confidence candidate retained",
        "No stronger image-level caution retained",
        "Anatomy context added",
        "Stage 3 displayed limited screening output",
    ]


def _case_specific_limitation(state_id: str | None, heatmap_available: bool) -> str:
    if state_id == "candidate_retained":
        return (
            "The retained bounding box is a review cue only. It is not a fracture boundary, diagnosis, "
            "or treatment target."
        )
    if state_id == "image_level_warning_without_bbox":
        if heatmap_available:
            return (
                "The heatmap may show where the model paid attention, but it does not mark a fracture "
                "location. It is not a fracture boundary and must not be interpreted as a fracture contour, "
                "bounding box, or treatment target."
            )
        return "The image-level signal does not localize a fracture and no precise bounding box is available."
    return (
        "The absence of a retained high-confidence AI cue does not exclude fracture or other injury, "
        "including subtle, non-displaced, occult, or clinically significant injury."
    )


def _stage1_affected(state_id: str | None, raw_count: int, retained_count: int, image_id: Any = None) -> str:
    if state_id == "candidate_retained" and retained_count > 0:
        variants = [
            "Stage 1 provided the localized candidate that later became the displayed review cue.",
            "Stage 1 supplied the localized visual candidate that stayed in the pipeline for later checks.",
            "The candidate-detection stage produced the localized cue that eventually reached the final display.",
        ]
        return variants[_variant_index(image_id, "stage1_candidate", len(variants))]
    if raw_count > 0:
        variants = [
            "Stage 1 proposed candidate regions, but none survived into the final localized output.",
            "Stage 1 found possible regions upstream, but later filtering left no localized cue for display.",
            "The detector produced early candidate regions, but no candidate remained in the final localized lane.",
        ]
        return variants[_variant_index(image_id, "stage1_filtered", len(variants))]
    variants = [
        "Stage 1 did not provide a candidate that survived into the final output.",
        "Stage 1 did not leave a localized visual cue for the later stages to keep.",
        "The candidate-detection lane produced no final localized cue for this case.",
    ]
    return variants[_variant_index(image_id, "stage1_none", len(variants))]


def _stage2a_affected(state_id: str | None, retained_count: int, image_id: Any = None) -> str:
    if state_id == "candidate_retained" and retained_count > 0:
        variants = [
            "The candidate remained available after verification, so it continued toward the final report.",
            "Verification did not remove the retained candidate, allowing it to stay in the localized review lane.",
            "The verifier kept the candidate available for the next display and safety checks.",
        ]
        return variants[_variant_index(image_id, "stage2a_candidate", len(variants))]
    variants = [
        "No retained candidate was available for final verification.",
        "The verifier had no final localized candidate to keep for display.",
        "Because no localized cue remained, Stage 2A did not add a retained candidate to the report path.",
    ]
    return variants[_variant_index(image_id, "stage2a_none", len(variants))]


def _stage2a5_affected(
    state_id: str | None,
    retained_count: int,
    artifact_rejected: int,
    image_id: Any = None,
) -> str:
    if artifact_rejected > 0:
        variants = [
            "Artifact suppression removed at least one candidate before the final display output.",
            "The artifact check filtered out one or more candidate regions before the final report.",
            "At least one candidate was removed by the artifact-suppression step before display.",
        ]
        return variants[_variant_index(image_id, "stage2a5_removed", len(variants))]
    if state_id == "candidate_retained" and retained_count > 0:
        variants = [
            "Artifact suppression did not remove the retained candidate.",
            "The artifact check allowed the retained candidate to remain visible as a review cue.",
            "No artifact rule removed the localized cue that continued to the final display.",
        ]
        return variants[_variant_index(image_id, "stage2a5_candidate", len(variants))]
    variants = [
        "No retained candidate was available for artifact suppression.",
        "The artifact step had no final localized candidate to filter.",
        "Because no retained cue was available, artifact suppression did not change the final visual output.",
    ]
    return variants[_variant_index(image_id, "stage2a5_none", len(variants))]


def _stage2b_affected(image_id: Any = None) -> str:
    variants = [
        (
            "The selected anatomy label shaped the wording and review context of the report. "
            "It did not determine injury status."
        ),
        (
            "The anatomy label provided body-region context for the report wording, without deciding "
            "whether injury is present."
        ),
        (
            "Stage 2B supplied anatomical context so the report could describe the region more clearly. "
            "It did not classify the image as injured or uninjured."
        ),
    ]
    return variants[_variant_index(image_id, "stage2b", len(variants))]


def _stage2c_affected(state_id: str | None, image_id: Any = None) -> str:
    if state_id == "candidate_retained":
        variants = [
            "Stage 2C was not needed for the final status because a localized candidate was already retained.",
            "The image-level caution lane did not drive the result because the localized cue lane already had a retained candidate.",
            "Stage 2C stayed secondary in this case; the final status came from the retained localized cue.",
        ]
        return variants[_variant_index(image_id, "stage2c_candidate", len(variants))]
    if state_id == "image_level_warning_without_bbox":
        variants = [
            "Stage 2C supplied the whole-image caution signal that determined the image-level warning status.",
            "The image-level caution lane became the active explanation path because no localized box was retained.",
            "Stage 2C provided the non-localized warning signal used for the final display state.",
        ]
        return variants[_variant_index(image_id, "stage2c_warning", len(variants))]
    variants = [
        "Stage 2C did not retain a stronger image-level warning for the final report.",
        "The image-level caution lane did not add a stronger warning to the final display.",
        "Stage 2C did not produce a retained whole-image caution signal for this case.",
    ]
    return variants[_variant_index(image_id, "stage2c_none", len(variants))]


def _stage3_affected(image_id: Any = None) -> str:
    variants = [
        (
            "Stage 3 assembled the locked outputs into the final report. It did not run new detection "
            "or modify any bounding box."
        ),
        (
            "Stage 3 converted the already locked pipeline outputs into report form. Stage 3 did not run "
            "new detection and Stage 3 did not modify bounding boxes."
        ),
        (
            "Stage 3 organized the previous stage outputs for display. Stage 3 did not run new detection "
            "and Stage 3 did not modify bounding boxes."
        ),
    ]
    return variants[_variant_index(image_id, "stage3", len(variants))]


def _stage_contribution_label(affected_result: str, contributed: bool) -> str:
    return affected_result


def _bottom_line(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return (
            "Bottom line: the system kept a localized review cue, but that cue is not a diagnosis, "
            "fracture boundary, or treatment target. A clinician must review the full study and clinical context."
        )
    if state_id == "image_level_warning_without_bbox":
        return (
            "Bottom line: the system raised a whole-image caution without a localized box. This increases "
            "review attention but does not identify a fracture location or treatment target."
        )
    return (
        "Bottom line: the system did not keep a strong localized cue. This does not mean the study is reassuring "
        "and does not exclude fracture or other injury."
    )


def _image_level_caution_available(state_id: str | None, decision: str | None) -> str:
    if state_id == "image_level_warning_without_bbox":
        return "Yes"
    if decision == "not_run_existing_candidate_present":
        return "Not applicable - localized candidate already retained"
    return "No"


def _heatmap_display(heatmap_available: bool, state_id: str | None) -> str:
    if heatmap_available and state_id == "image_level_warning_without_bbox":
        return "Optional visual support"
    if heatmap_available:
        return "Yes"
    return "No"


def _low_support_note(candidate_card: Dict[str, Any]) -> List[str]:
    if not candidate_card.get("available"):
        return []
    try:
        percent = float(candidate_card.get("percent"))
    except (TypeError, ValueError):
        return []
    if percent < 40.0:
        return [
            "The retained candidate has low internal model support and should be treated as a weak visual cue."
        ]
    return []


def _stage_summary(
    *,
    stage_id: str,
    stage_name: str,
    purpose: str,
    input_used: str,
    output: str,
    affected_result: str,
    contributed: bool,
    limitation: str,
) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "stage_display_label": STAGE_DISPLAY_LABELS.get(stage_id, stage_name),
        "stage_name": stage_name,
        "purpose": purpose,
        "input_used": input_used,
        "output": output,
        "affected_result": affected_result,
        "how_it_affected_result": _stage_contribution_label(affected_result, contributed),
        "contributed_to_report": bool(contributed),
        # Kept for older report/debug surfaces; V2 display should prefer affected_result.
        "user_explanation": affected_result,
        "limitations": [limitation],
    }


def build_ai_pipeline_explanation(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic explanation of the pipeline path for app/report transparency.

    The panel explains observable stage outputs and display decisions. It does
    not expose hidden reasoning and does not create new medical findings.
    """
    app_payload = stage3_block.get("app_payload") or {}
    source = stage3_block.get("input_summary") or {}
    final_assessment = source.get("final_assessment") or {}
    if not final_assessment:
        final_assessment = {
            "raw_fracture_candidate_count": source.get("raw_candidate_count"),
            "fracture_detection_count": source.get("accepted_candidate_count"),
            "rejected_fracture_candidate_count": source.get("rejected_candidate_count"),
            "artifact_rejected_fracture_candidate_count": source.get("artifact_rejected_candidate_count"),
            "verifier_rejected_fracture_candidate_count": source.get("verifier_rejected_candidate_count"),
        }

    case_header = app_payload.get("case_header") or {}
    image_id = case_header.get("image_id")
    finding = _finding_state(app_payload)
    state_id = finding.get("state_id")
    visual_panel = app_payload.get("visual_panel") or {}
    primary_bbox = visual_panel.get("primary_bbox") or {}
    anatomy_panel = app_payload.get("anatomy_panel") or {}
    confidence_cards = app_payload.get("confidence_cards") or []
    candidate_card = _card(confidence_cards, "fracture_candidate_support")
    stage2c_card = _card(confidence_cards, "stage2c_caution_support")
    heatmap = visual_panel.get("stage2c_heatmap") or {}
    stage2c_decision = _stage2c_decision(visual_panel)

    bbox_available = bool(primary_bbox.get("available"))
    heatmap_available = bool(heatmap.get("available"))
    raw_count = int(final_assessment.get("raw_fracture_candidate_count") or 0)
    retained_count = int(final_assessment.get("fracture_detection_count") or 0)
    artifact_rejected = int(final_assessment.get("artifact_rejected_fracture_candidate_count") or 0)
    selected_anatomy = anatomy_panel.get("selected_display_label") or anatomy_panel.get("parent_display_label")

    stage1_affected = _stage1_affected(state_id, raw_count, retained_count, image_id)
    stage2a_affected = _stage2a_affected(state_id, retained_count, image_id)
    stage2a5_affected = _stage2a5_affected(state_id, retained_count, artifact_rejected, image_id)
    stage2b_affected = _stage2b_affected(image_id)
    stage2c_affected = _stage2c_affected(state_id, image_id)
    stage3_affected = _stage3_affected(image_id)

    stage_summaries = [
        _stage_summary(
            stage_id="stage1",
            stage_name="Candidate detection",
            purpose="Searches the image for visual regions that may deserve later review.",
            input_used="Input X-ray image.",
            output=_stage1_output(final_assessment),
            affected_result=stage1_affected,
            contributed=raw_count > 0 or retained_count > 0,
            limitation="Candidate detection can miss subtle findings and can also propose false visual cues.",
        ),
        _stage_summary(
            stage_id="stage2a",
            stage_name="Candidate verification",
            purpose="Checks whether candidate regions have enough internal model support to stay under review.",
            input_used="Candidate regions from Stage 1, when available.",
            output=_stage2a_output(final_assessment),
            affected_result=stage2a_affected,
            contributed=state_id == "candidate_retained" and retained_count > 0,
            limitation="A verifier decision is not a diagnosis and cannot exclude injury.",
        ),
        _stage_summary(
            stage_id="stage2a5",
            stage_name="Artifact suppression",
            purpose="Reduces candidates that look more consistent with artifacts or non-anatomical structures.",
            input_used="Candidate regions retained after verification.",
            output=_stage2a5_output(final_assessment),
            affected_result=stage2a5_affected,
            contributed=state_id == "candidate_retained" or artifact_rejected > 0,
            limitation="Artifact checks are conservative and do not replace image review.",
        ),
        _stage_summary(
            stage_id="stage2b",
            stage_name="Anatomy classification",
            purpose="Classifies the broad body region shown in the image so the report can use appropriate context.",
            input_used="Whole X-ray image.",
            output=_stage2b_output(anatomy_panel),
            affected_result=stage2b_affected,
            contributed=True,
            limitation=ANATOMY_DISCLAIMER,
        ),
        _stage_summary(
            stage_id="stage2c",
            stage_name="Image-level caution",
            purpose="Checks whether the whole image should be flagged for review when no localized box is retained.",
            input_used="Whole X-ray image and current candidate-retention state.",
            output=_stage2c_output(
                state_id,
                stage2c_decision,
                heatmap_available,
                stage2c_card.get("percent") if stage2c_card.get("available") else None,
            ),
            affected_result=stage2c_affected,
            contributed=state_id == "image_level_warning_without_bbox",
            limitation="Image-level caution does not provide a fracture boundary.",
        ),
        _stage_summary(
            stage_id="stage3",
            stage_name="Report assembly",
            purpose="Converts locked pipeline outputs into a structured report and display payload.",
            input_used="Locked Stage 1/2 outputs and controlled Stage 3 display rules.",
            output=f"Final display status: {_human_status_label(state_id)}.",
            affected_result=stage3_affected,
            contributed=True,
            limitation="Stage 3 is a reporting layer, not a diagnostic model.",
        ),
    ]

    evidence_used = {
        "localized_bbox": _yes_no(bbox_available),
        "localized_bbox_available": bbox_available,
        "heatmap": _heatmap_display(heatmap_available, state_id),
        "heatmap_available": heatmap_available,
        "image_level_caution": _image_level_caution_available(state_id, stage2c_decision),
        "image_level_caution_available": state_id == "image_level_warning_without_bbox",
        "anatomy_label": _display_label(selected_anatomy) if selected_anatomy else "Not available",
        "anatomy_label_available": bool(selected_anatomy),
        "clinical_symptoms": "Not available to the AI unless explicitly provided in structured input",
        "mechanism_of_injury": "Not available to the AI unless explicitly provided in structured input",
    }

    evidence_trace = {
        "bbox_available": bbox_available,
        "bbox_display": evidence_used["localized_bbox"],
        "bbox_source": "locked_stage1_stage2_output" if bbox_available else None,
        "bbox_modified_by_stage3": False,
        "heatmap_available": heatmap_available,
        "heatmap_display": evidence_used["heatmap"],
        "heatmap_source": "stage2c_image_level_caution" if heatmap_available else None,
        "selected_anatomy": selected_anatomy,
        "anatomy_support": anatomy_panel.get("selected_support_percent"),
        "candidate_support": candidate_card.get("percent") if candidate_card.get("available") else None,
        "image_level_support": stage2c_card.get("percent") if stage2c_card.get("available") else None,
        "model_support_summary": (
            f"Candidate support: {_fmt_percent(candidate_card.get('percent'))}; "
            f"image-level support: {_fmt_percent(stage2c_card.get('percent'))}; "
            f"anatomy support: {_fmt_percent(anatomy_panel.get('selected_support_percent'))}."
        ),
        "artifact_check_summary": _stage2a5_output(final_assessment),
        "visual_evidence_note": (
            "A retained bounding box is available as a review cue."
            if bbox_available
            else "No localized bounding box is available in the final display output."
        ),
    }
    technical_details = {
        "default_collapsed": True,
        "raw_case_status": state_id,
        "raw_stage_ids": [stage["stage_id"] for stage in stage_summaries],
        "bbox_available": bbox_available,
        "heatmap_available": heatmap_available,
        "bbox_modified_by_stage3": False,
        "support_scores": {
            "candidate_support": candidate_card.get("percent") if candidate_card.get("available") else None,
            "image_level_support": stage2c_card.get("percent") if stage2c_card.get("available") else None,
            "anatomy_support": anatomy_panel.get("selected_support_percent"),
        },
    }

    status_limitations = [_case_specific_limitation(state_id, heatmap_available)]
    status_limitations.extend(_low_support_note(candidate_card))

    return {
        "version": PIPELINE_EXPLANATION_VERSION,
        "enabled": True,
        "label": "How the AI reached this result",
        "panel_title": "AI pipeline explanation",
        "default_collapsed": True,
        "audience": "clinician_patient_and_reviewer",
        "is_diagnostic_reasoning": False,
        "uses_llm": False,
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "pipeline_overview": (
            "The system processed this X-ray through a fixed multi-stage pipeline. Earlier stages generated "
            "or filtered visual and anatomy-related outputs. Stage 3 did not perform new detection; it "
            "converted the locked outputs into a structured report."
        ),
        "case_status_explanation": {
            "case_status": state_id,
            "case_status_display": _human_status_label(state_id),
            "human_status_label": _human_status_label(state_id),
            "short_explanation": _short_explanation(state_id, image_id),
            "short_reason": _causal_reason(state_id, image_id),
            "why_this_result_was_shown": _causal_reason(state_id, image_id),
            "bottom_line": _bottom_line(state_id),
            "pipeline_flow": _pipeline_flow(state_id),
            "pipeline_flow_text": " -> ".join(_pipeline_flow(state_id)),
            "case_specific_limitation": _case_specific_limitation(state_id, heatmap_available),
            "status_specific_limitations": status_limitations,
        },
        "evidence_used": evidence_used,
        "stage_summaries": stage_summaries,
        "evidence_trace": evidence_trace,
        "technical_details": technical_details,
        "what_ai_did_not_do": [
            "The AI did not make a diagnosis.",
            "The AI did not confirm a fracture.",
            "The AI did not rule out fracture or other injury.",
            "Stage 3 did not run new detection.",
            "Stage 3 did not modify bounding boxes.",
            "The system did not use clinical symptoms unless they were explicitly provided in structured input.",
            "The system did not decide treatment, immobilisation, weight-bearing, medication, clearance, follow-up, referral, procedure, or surgery.",
        ],
        "required_human_review": [
            "A qualified clinician should review the complete radiographic study.",
            "Any marked region or heatmap should be used only as an attention cue and must not be reviewed in isolation.",
            "Clinical examination, symptoms, mechanism of injury, and formal radiology review remain necessary.",
            SCORE_DISCLAIMER,
            "This panel explains pipeline behavior, not patient-specific medical reasoning.",
        ],
        "score_disclaimers": [
            SCORE_DISCLAIMER,
            ANATOMY_DISCLAIMER,
        ],
        "source_image_id": image_id,
        "technical_note": (
            "This panel explains deterministic pipeline outputs and display decisions. It does not expose "
            "hidden reasoning and does not add medical findings."
        ),
    }
