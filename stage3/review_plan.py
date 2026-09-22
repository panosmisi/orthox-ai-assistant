from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 3)


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


def _lane(summary: Stage3InputSummary) -> Dict[str, str]:
    anatomy = _display_label(summary.anatomy_label or summary.anatomy_parent_label)
    if summary.accepted_candidate_count > 0:
        return {
            "lane_id": "localized_candidate_review",
            "priority": "high_review_priority",
            "headline": f"Review retained localized candidate around {anatomy}.",
            "rationale": "The detector/verifier retained at least one candidate, so review should start from the marked region and then include the full radiograph.",
        }
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        return {
            "lane_id": "image_level_safety_review",
            "priority": "high_review_priority",
            "headline": f"Review image-level caution signal around {anatomy}.",
            "rationale": "No accepted box remains, but the image-level safety layer raised a caution signal, so the whole radiograph should be reviewed carefully.",
        }
    return {
        "lane_id": "no_high_confidence_candidate_review",
        "priority": "standard_review_with_limitations",
        "headline": f"No high-confidence candidate retained around {anatomy}.",
        "rationale": (
            "The system retained no high-confidence candidate. This is limited screening output "
            "and does not exclude subtle or occult injury."
        ),
    }


def _model_signals(summary: Stage3InputSummary) -> List[Dict[str, Any]]:
    return [
        {
            "signal_id": "fracture_candidate_support",
            "available": summary.primary_candidate_confidence is not None,
            "support_percent": _pct(summary.primary_candidate_confidence),
            "interpretation": "Technical detector/verifier support for a retained candidate; not a calibrated clinical probability.",
        },
        {
            "signal_id": "stage2c_image_level_caution",
            "available": summary.stage2c_mean_suspicious_probability is not None,
            "support_percent": _pct(summary.stage2c_mean_suspicious_probability),
            "interpretation": "Image-level caution support; it does not provide a fracture boundary.",
        },
        {
            "signal_id": "anatomy_label_support",
            "available": summary.anatomy_confidence is not None,
            "support_percent": _pct(summary.anatomy_confidence),
            "interpretation": "Technical support for the displayed anatomy label; parent fallback means fine localization was uncertain.",
        },
    ]


def _review_tasks(summary: Stage3InputSummary) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    if summary.accepted_candidate_count > 0:
        tasks.extend(
            [
                {
                    "task_id": "localized_region_review",
                    "task": "Inspect the retained bbox region on the original radiograph for cortical break, lucent line, displacement, or projectional artifact.",
                    "reason": "The upstream pipeline retained a localized candidate.",
                },
                {
                    "task_id": "full_image_context_review",
                    "task": "Review the full radiograph beyond the bbox for additional injury, missed second site, image-quality limitation, or misleading overlay context.",
                    "reason": "A correct bbox can still miss broader clinical context.",
                },
            ]
        )
    elif summary.stage2c_decision in {"suspicious", "uncertain"}:
        tasks.extend(
            [
                {
                    "task_id": "whole_image_subtle_injury_review",
                    "task": "Inspect the whole radiograph for subtle, non-displaced, or occult injury patterns because no reliable bbox is available.",
                    "reason": "Stage 2C raised a caution signal without a retained detector box.",
                },
                {
                    "task_id": "optional_heatmap_context",
                    "task": "If a heatmap is displayed, use it only as an attention cue and not as a fracture boundary.",
                    "reason": "Stage 2C heatmaps explain model attention but do not localize a fracture with bbox precision.",
                },
            ]
        )
    else:
        tasks.extend(
            [
                {
                    "task_id": "no_high_confidence_candidate_check",
                    "task": (
                        "Use the result as limited screening output only; correlate with symptoms, "
                        "focal tenderness, trauma mechanism, and image quality."
                    ),
                    "reason": "The system retained no high-confidence candidate but cannot rule out subtle injury.",
                },
                {
                    "task_id": "clinical_mismatch_check",
                    "task": "If clinical suspicion is high, do not let the AI output lower the priority of clinician review.",
                    "reason": "Low or absent model support is weaker than persistent clinical concern.",
                },
            ]
        )

    if summary.anatomy_output_type == "parent_fallback":
        tasks.append(
            {
                "task_id": "anatomy_uncertainty_check",
                "task": "Treat the anatomy label as broad region context rather than exact anatomical localization.",
                "reason": "The anatomy classifier fell back to a parent label.",
            }
        )
    return tasks


def _uncertainty_notes(summary: Stage3InputSummary) -> List[str]:
    notes = [
        "All support scores are model-support values, not calibrated clinical probabilities.",
        "Stage 3 does not reinterpret pixels and does not override Stage 1 or Stage 2.",
        "The output should be reviewed with the original radiograph, not only with overlays or summaries.",
    ]
    if summary.accepted_candidate_count == 0:
        notes.append("No retained candidate does not exclude subtle, non-displaced, or radiographically occult injury.")
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        notes.append("Stage 2C warning should increase attention, but it does not create a localizable fracture box.")
    if summary.anatomy_output_type == "parent_fallback":
        notes.append("Fine anatomy label was uncertain, so the parent anatomical region should be used for display and review.")
    return notes


def _escalation_triggers(red_flags: Dict[str, Any], limit: int = 8) -> List[Dict[str, str]]:
    triggers: List[Dict[str, str]] = []
    for category in ["contextual", "anatomy_specific", "general"]:
        for item in red_flags.get(category, []) or []:
            triggers.append(
                {
                    "category": category,
                    "trigger": str(item),
                    "meaning": "Clinical feature to actively check; not asserted by the AI.",
                }
            )
            if len(triggers) >= limit:
                return triggers
    return triggers


def build_review_plan(summary: Stage3InputSummary, red_flags: Dict[str, Any]) -> Dict[str, Any]:
    lane = _lane(summary)
    return {
        "version": "stage3_review_plan_v1",
        "purpose": "Structured clinician review plan derived from locked upstream outputs.",
        "llm_used": False,
        "not_a_diagnosis": True,
        "no_patient_specific_treatment": True,
        "human_verification_required": True,
        "stage3_does_not_override_stage1_stage2": True,
        "case_lane": lane,
        "model_signals": _model_signals(summary),
        "review_tasks": _review_tasks(summary),
        "uncertainty_notes": _uncertainty_notes(summary),
        "escalation_triggers_to_check": _escalation_triggers(red_flags),
        "output_boundaries": [
            "This plan is a review aid, not a diagnosis.",
            "It does not prescribe treatment.",
            "It does not replace clinician judgement, examination findings, or local clinical pathways.",
            "It must be interpreted together with the original radiograph.",
        ],
        "app_display": {
            "recommended_panel_title": "Clinician review plan",
            "default_visibility": "expanded_for_candidate_or_warning",
            "show_before_patient_note": True,
        },
    }
