from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


def _pct(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.1f}%"


def _display_label(label: str | None) -> str:
    if not label:
        return "the X-ray area"
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


def build_patient_friendly_note(
    summary: Stage3InputSummary,
    clinician_summary: Dict[str, Any],
    max_bullets: int = 4,
) -> Dict[str, Any]:
    """Build a short non-diagnostic patient-facing note.

    This text is deliberately conservative. It explains the AI screening output
    without giving diagnosis, reassurance, discharge advice, or treatment.
    """
    anatomy = _display_label(summary.anatomy_label or summary.anatomy_parent_label)
    candidate_score = _pct(summary.primary_candidate_confidence)
    stage2c_score = _pct(summary.stage2c_mean_suspicious_probability)

    if summary.accepted_candidate_count > 0:
        headline = f"The AI marked a possible area of concern around {anatomy}."
        explanation = (
            "This means the screening model found a region that should be reviewed by a clinician. "
            "It is not a diagnosis by itself."
        )
        score_text = (
            f"AI support score for the main marked region: {candidate_score}."
            if candidate_score
            else "The AI score is a technical support score, not a clinical probability."
        )
    elif summary.stage2c_decision in {"suspicious", "uncertain"}:
        headline = f"The AI did not keep a marked box, but it still raised a caution signal around {anatomy}."
        explanation = (
            "This means the image-level safety check saw a pattern that deserves human review, "
            "even though there is no precise marked region."
        )
        score_text = (
            f"AI support score for the caution signal: {stage2c_score}."
            if stage2c_score
            else "The AI score is a technical support score, not a clinical probability."
        )
    else:
        headline = f"The AI did not keep a high-confidence marked region around {anatomy}."
        explanation = (
            "This does not prove that the X-ray is clear. Subtle injuries can be difficult to see, "
            "so a clinician should interpret the image with symptoms and examination findings."
        )
        score_text = "No high-confidence AI region was retained for display."

    bullets: List[str] = [
        explanation,
        score_text,
        "AI outputs can miss subtle findings and can also mark normal structures or artifacts.",
        "A qualified clinician must make the final interpretation.",
    ]

    return {
        "note_type": "deterministic_patient_friendly",
        "llm_used": False,
        "headline": headline,
        "plain_language_points": bullets[:max_bullets],
        "not_a_diagnosis": True,
        "no_treatment_advice": True,
        "clinician_review_required": True,
        "linked_clinician_finding_level": clinician_summary.get("finding_level"),
    }
