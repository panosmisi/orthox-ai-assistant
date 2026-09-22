from __future__ import annotations

from typing import Any, Dict, List

from .writer_response_validator import validate_writer_response


def _display_label(stage3_block: Dict[str, Any]) -> str:
    summary = stage3_block.get("input_summary") or {}
    label = summary.get("anatomy_label") or summary.get("anatomy_parent_label") or "the imaged region"
    display_map = {
        "knee_lower_leg": "knee/lower leg",
        "wrist_hand": "wrist/hand",
        "ankle_foot": "ankle/foot",
        "pelvis_hip_femur": "pelvis/hip/femur",
        "shoulder_upper_arm": "shoulder/upper arm",
        "leg": "lower limb",
        "hand": "hand/wrist region",
    }
    return display_map.get(str(label), str(label).replace("_", " "))


def _state_phrase(stage3_block: Dict[str, Any]) -> str:
    summary = stage3_block.get("input_summary") or {}
    accepted = int(summary.get("accepted_candidate_count") or 0)
    stage2c = summary.get("stage2c_decision")
    if accepted > 0:
        noun = "candidate" if accepted == 1 else "candidates"
        return f"the upstream pipeline retained {accepted} possible fracture {noun}"
    if stage2c in {"suspicious", "uncertain"}:
        return "the detector retained no localized box, but the image-level safety layer raised a caution signal"
    return "the pipeline retained no high-confidence localized fracture candidate"


def _claims_by_id(stage3_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(claim.get("claim_id")): claim
        for claim in stage3_block.get("allowed_claims", []) or []
        if isinstance(claim, dict) and claim.get("claim_id")
    }


def _source_ids_for_claims(stage3_block: Dict[str, Any], claim_ids: List[str]) -> List[str]:
    claims = _claims_by_id(stage3_block)
    out: List[str] = []
    for claim_id in claim_ids:
        source_id = claims.get(claim_id, {}).get("source_id")
        if source_id and str(source_id) not in out:
            out.append(str(source_id))
    return out


def _select_claim_ids(stage3_block: Dict[str, Any], max_treatment_claims: int = 3) -> List[str]:
    treatment_claims = []
    safety_claims = []
    pipeline_claims = []
    for claim in stage3_block.get("allowed_claims", []) or []:
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id or claim.get("allowed") is not True:
            continue
        if claim.get("claim_type") == "controlled_pubmed_treatment_orientation":
            treatment_claims.append(claim_id)
        elif claim_id.startswith("project_safety_"):
            safety_claims.append(claim_id)
        elif claim_id.startswith(("pipeline_", "stage2c_")):
            pipeline_claims.append(claim_id)

    selected = safety_claims[:3] + pipeline_claims[:1] + treatment_claims[:max_treatment_claims]
    if not selected:
        selected = [
            str(claim.get("claim_id"))
            for claim in stage3_block.get("allowed_claims", []) or []
            if claim.get("claim_id") and claim.get("allowed") is True
        ][:3]
    return selected


def _treatment_claim_count(stage3_block: Dict[str, Any], claim_ids: List[str]) -> int:
    claims = _claims_by_id(stage3_block)
    return sum(
        1
        for claim_id in claim_ids
        if claims.get(claim_id, {}).get("claim_type") == "controlled_pubmed_treatment_orientation"
    )


def _clinician_summary(stage3_block: Dict[str, Any], claim_ids: List[str]) -> str:
    anatomy = _display_label(stage3_block)
    state = _state_phrase(stage3_block)
    treatment_count = _treatment_claim_count(stage3_block, claim_ids)
    if treatment_count > 0:
        evidence_sentence = (
            f"{treatment_count} source-bound PubMed treatment-management context item(s) are available for {anatomy}. "
            "They may support clinician-led review of management considerations, but they do not provide patient-specific treatment instructions."
        )
    else:
        evidence_sentence = (
            "No PubMed treatment-management claim is available in the approved evidence pack for this case; "
            "use the deterministic review guidance and local clinical pathways."
        )
    return (
        "This is not a standalone diagnosis. Human clinical verification is required. "
        "Model scores are not clinical probabilities. "
        f"In this case, {state} around {anatomy}. "
        f"{evidence_sentence} The final care plan must be determined by clinician assessment, patient context, and local protocol."
    )


def _patient_note(stage3_block: Dict[str, Any], claim_ids: List[str]) -> str:
    anatomy = _display_label(stage3_block)
    treatment_count = _treatment_claim_count(stage3_block, claim_ids)
    context = (
        "The system found supporting medical-literature context for clinician review."
        if treatment_count > 0
        else "The system did not add treatment-management literature context for this case."
    )
    return (
        "This output is not a standalone diagnosis. Human verification is required. "
        "The AI scores are not clinical probabilities. "
        f"{context} A qualified clinician should interpret the {anatomy} image together with symptoms and examination findings."
    )


def build_constrained_treatment_writer_response(
    stage3_block: Dict[str, Any],
    max_treatment_claims: int = 3,
) -> Dict[str, Any]:
    """Build a validated treatment-oriented writer response without an LLM."""
    claim_ids = _select_claim_ids(stage3_block, max_treatment_claims=max_treatment_claims)
    source_ids = _source_ids_for_claims(stage3_block, claim_ids)
    response = {
        "writer_type": "deterministic_constrained_treatment_writer_v1",
        "llm_used": False,
        "sections": {
            "clinician_summary": _clinician_summary(stage3_block, claim_ids),
            "patient_note": _patient_note(stage3_block, claim_ids),
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
    treatment_claim_count = _treatment_claim_count(stage3_block, claim_ids)
    return {
        "version": "stage3_constrained_treatment_writer_experiment_v1",
        "purpose": "Validated deterministic writer experiment using approved treatment-oriented RAG claims.",
        "response": response,
        "treatment_claim_count": treatment_claim_count,
        "uses_treatment_claims": treatment_claim_count > 0,
        "validation": validate_writer_response(stage3_block, response),
    }
