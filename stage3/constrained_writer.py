from __future__ import annotations

from typing import Any, Dict, List

from .writer_response_validator import validate_writer_response


def _display_label(label: str | None) -> str:
    if not label:
        return "the imaged region"
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


def _claim_ids(stage3_block: Dict[str, Any]) -> List[str]:
    wanted_prefixes = ("project_safety_", "pipeline_", "stage2c_")
    ids: List[str] = []
    for claim in stage3_block.get("allowed_claims", []) or []:
        claim_id = str(claim.get("claim_id") or "")
        if claim_id.startswith(wanted_prefixes):
            ids.append(claim_id)
    if not ids:
        ids = [
            str(claim.get("claim_id"))
            for claim in stage3_block.get("allowed_claims", []) or []
            if claim.get("claim_id")
        ][:3]
    return ids


def _source_ids_for_claims(stage3_block: Dict[str, Any], claim_ids: List[str]) -> List[str]:
    source_ids: List[str] = []
    wanted = set(claim_ids)
    for claim in stage3_block.get("allowed_claims", []) or []:
        if claim.get("claim_id") in wanted and claim.get("source_id"):
            source_id = str(claim["source_id"])
            if source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def _case_context(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    summary = stage3_block.get("input_summary") or {}
    anatomy = _display_label(summary.get("anatomy_label") or summary.get("anatomy_parent_label"))
    accepted = int(summary.get("accepted_candidate_count") or 0)
    stage2c_decision = summary.get("stage2c_decision")
    if accepted > 0:
        state = "candidate_retained"
    elif stage2c_decision in {"suspicious", "uncertain"}:
        state = "image_level_warning_without_bbox"
    else:
        state = "no_high_confidence_candidate_retained"
    return {
        "anatomy": anatomy,
        "accepted_candidate_count": accepted,
        "state": state,
        "stage2c_decision": stage2c_decision,
    }


def _clinician_text(context: Dict[str, Any]) -> str:
    anatomy = context["anatomy"]
    state = context["state"]
    accepted = context["accepted_candidate_count"]
    prefix = (
        "This is not a standalone diagnosis. Human clinical verification is required. "
        "Model scores are not clinical probabilities. "
    )
    if state == "candidate_retained":
        noun = "candidate" if accepted == 1 else "candidates"
        return (
            prefix
            + f"The upstream pipeline retained {accepted} possible fracture {noun} around {anatomy}. "
            + "Review should begin with the marked region and continue on the original full radiograph."
        )
    if state == "image_level_warning_without_bbox":
        return (
            prefix
            + f"The detector did not retain a precise box, but the image-level safety layer raised a caution signal around {anatomy}. "
            + "Review should focus on the whole radiograph because the warning does not provide a fracture boundary."
        )
    return (
        prefix
        + f"The pipeline did not retain a high-confidence marked region around {anatomy}. "
        + "This does not exclude subtle or occult injury, so clinical context and image quality remain important."
    )


def _patient_text(context: Dict[str, Any]) -> str:
    anatomy = context["anatomy"]
    state = context["state"]
    prefix = (
        "This is not a standalone diagnosis. Human verification is required. "
        "The AI scores are not clinical probabilities. "
    )
    if state == "candidate_retained":
        return (
            prefix
            + f"The AI marked a possible area of concern around {anatomy}. "
            + "A qualified clinician should review the image and the marked region."
        )
    if state == "image_level_warning_without_bbox":
        return (
            prefix
            + f"The AI raised a caution signal around {anatomy}, but it did not keep a precise marked box. "
            + "A qualified clinician should review the full image."
        )
    return (
        prefix
        + f"The AI did not keep a high-confidence marked region around {anatomy}. "
        + "This does not exclude subtle injury, so symptoms and clinician review still matter."
    )


def build_constrained_writer_response(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    """Build and validate a deterministic response using the future-writer schema."""
    context = _case_context(stage3_block)
    claim_ids = _claim_ids(stage3_block)
    source_ids = _source_ids_for_claims(stage3_block, claim_ids)
    response = {
        "writer_type": "deterministic_constrained_writer_v1",
        "llm_used": False,
        "sections": {
            "clinician_summary": _clinician_text(context),
            "patient_note": _patient_text(context),
            "safety_footer": (
                "This is not a standalone diagnosis. Human verification is required. "
                "Model scores are not clinical probabilities."
            ),
        },
        "used_claim_ids": claim_ids,
        "used_source_ids": source_ids,
        "created_new_medical_claims": False,
        "used_raw_pubmed_abstracts": False,
        "reinterpreted_image_pixels": False,
    }
    return {
        "version": "stage3_validated_writer_response_v1",
        "response": response,
        "validation": validate_writer_response(stage3_block, response),
    }
