from __future__ import annotations

from typing import Any, Dict, List, Set

from .safety_filter import find_forbidden_phrases


REQUIRED_RESPONSE_KEYS = {
    "writer_type",
    "llm_used",
    "sections",
    "used_claim_ids",
    "used_source_ids",
    "created_new_medical_claims",
    "used_raw_pubmed_abstracts",
    "reinterpreted_image_pixels",
}

REQUIRED_SECTION_KEYS = {
    "clinician_summary",
    "patient_note",
    "safety_footer",
}

REQUIRED_SAFE_MEANINGS = [
    "not a standalone",
    "human",
    "verification",
    "not clinical probabilities",
]


def _allowed_claim_ids(stage3_block: Dict[str, Any]) -> Set[str]:
    return {
        str(claim.get("claim_id"))
        for claim in stage3_block.get("allowed_claims", []) or []
        if isinstance(claim, dict) and claim.get("claim_id") and claim.get("allowed") is True
    }


def _source_ids(stage3_block: Dict[str, Any]) -> Set[str]:
    return {
        str(source.get("source_id"))
        for source in stage3_block.get("sources", []) or []
        if isinstance(source, dict) and source.get("source_id")
    }


def _claim_source_map(stage3_block: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for claim in stage3_block.get("allowed_claims", []) or []:
        if isinstance(claim, dict) and claim.get("claim_id") and claim.get("source_id"):
            mapping[str(claim["claim_id"])] = str(claim["source_id"])
    return mapping


def _text_blob(response: Dict[str, Any]) -> str:
    sections = response.get("sections") or {}
    values = []
    if isinstance(sections, dict):
        for value in sections.values():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
    return "\n".join(values)


def _missing_safe_meanings(text: str) -> List[str]:
    lowered = text.lower()
    return [meaning for meaning in REQUIRED_SAFE_MEANINGS if meaning not in lowered]


def validate_writer_response(stage3_block: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a future LLM/template response against Stage 3 guardrails.

    This does not call an LLM. It validates a candidate response object before it
    could be displayed to a clinician or patient.
    """
    errors: List[str] = []
    warnings: List[str] = []

    missing = sorted(REQUIRED_RESPONSE_KEYS - set(response.keys()))
    errors.extend(f"missing_response_key:{key}" for key in missing)

    sections = response.get("sections")
    if not isinstance(sections, dict):
        errors.append("sections_not_dict")
        sections = {}
    else:
        section_missing = sorted(REQUIRED_SECTION_KEYS - set(sections.keys()))
        errors.extend(f"missing_section:{key}" for key in section_missing)

    text = _text_blob(response)
    forbidden_hits = find_forbidden_phrases(response)
    if forbidden_hits:
        errors.append("forbidden_phrases_detected")
        errors.extend(f"forbidden_phrase:{hit.get('phrase')}" for hit in forbidden_hits)

    missing_safe = _missing_safe_meanings(text)
    errors.extend(f"missing_required_safe_meaning:{item}" for item in missing_safe)

    if response.get("created_new_medical_claims") is not False:
        errors.append("created_new_medical_claims_not_false")
    if response.get("used_raw_pubmed_abstracts") is not False:
        errors.append("used_raw_pubmed_abstracts_not_false")
    if response.get("reinterpreted_image_pixels") is not False:
        errors.append("reinterpreted_image_pixels_not_false")

    allowed_claim_ids = _allowed_claim_ids(stage3_block)
    source_ids = _source_ids(stage3_block)
    claim_source_map = _claim_source_map(stage3_block)

    used_claim_ids = response.get("used_claim_ids") or []
    used_source_ids = response.get("used_source_ids") or []
    if not isinstance(used_claim_ids, list):
        errors.append("used_claim_ids_not_list")
        used_claim_ids = []
    if not isinstance(used_source_ids, list):
        errors.append("used_source_ids_not_list")
        used_source_ids = []

    for claim_id in used_claim_ids:
        claim_id_s = str(claim_id)
        if claim_id_s not in allowed_claim_ids:
            errors.append(f"claim_id_not_allowed:{claim_id_s}")
        source_id = claim_source_map.get(claim_id_s)
        if source_id and source_id not in used_source_ids:
            errors.append(f"claim_source_not_declared:{claim_id_s}:{source_id}")

    for source_id in used_source_ids:
        source_id_s = str(source_id)
        if source_id_s not in source_ids:
            errors.append(f"source_id_not_available:{source_id_s}")

    if not used_claim_ids:
        warnings.append("no_claim_ids_used")
    if not used_source_ids:
        warnings.append("no_source_ids_used")

    guardrails = stage3_block.get("response_guardrails") or {}
    validation = guardrails.get("validation") or {}
    if validation.get("new_claim_creation_allowed") is not False:
        errors.append("stage3_guardrail_allows_new_claim_creation")
    if validation.get("all_claim_refs_allowed_and_sourced") is not True:
        errors.append("stage3_guardrail_claim_refs_invalid")

    return {
        "validator": "stage3_writer_response_validator_v1",
        "verdict": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checked_claim_count": len(used_claim_ids),
        "checked_source_count": len(used_source_ids),
        "forbidden_hit_count": len(forbidden_hits),
    }


def build_validator_contract() -> Dict[str, Any]:
    return {
        "validator": "stage3_writer_response_validator_v1",
        "purpose": "Validate a future LLM/template response before display.",
        "required_response_keys": sorted(REQUIRED_RESPONSE_KEYS),
        "required_section_keys": sorted(REQUIRED_SECTION_KEYS),
        "required_safe_meanings": REQUIRED_SAFE_MEANINGS,
        "hard_fail_conditions": [
            "forbidden phrase detected",
            "new medical claim creation flag is true",
            "raw PubMed abstract use flag is true",
            "image pixel reinterpretation flag is true",
            "claim id is not in allowed_claims",
            "source id is not in sources",
            "required safety meaning is missing",
        ],
    }
