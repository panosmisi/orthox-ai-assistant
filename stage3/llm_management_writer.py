from __future__ import annotations

import time
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set

from .local_llm_writer import (
    DEFAULT_MEDGEMMA_MODEL_ID,
    _build_generation_inputs,
    _load_hf_text_model,
    dependency_report,
    extract_json_object,
    model_path_status,
)
from .safety_filter import find_forbidden_phrases


LLM_MANAGEMENT_WRITER_VERSION = "stage3_llm_management_writer_v1"
LLM_MANAGEMENT_VALIDATOR_VERSION = "stage3_llm_management_writer_validator_v2"
EVIDENCE_CONTEXT_SECTION = "evidence_supported_context"
LEGACY_EVIDENCE_CONTEXT_SECTION = "pubmed_supported_context"

REQUIRED_MANAGEMENT_SECTIONS = [
    "management_orientation",
    "guidance_level",
    "why_this_section_is_shown",
    "case_specific_suggestions",
    EVIDENCE_CONTEXT_SECTION,
    "red_flags_or_escalation_triggers",
    "clinical_inputs_needed",
    "what_ai_must_not_decide",
    "safety_boundary",
]

LIST_MANAGEMENT_SECTIONS = {
    "case_specific_suggestions",
    EVIDENCE_CONTEXT_SECTION,
    "red_flags_or_escalation_triggers",
    "clinical_inputs_needed",
    "what_ai_must_not_decide",
}

MAX_MANAGEMENT_SECTION_ITEMS = {
    "case_specific_suggestions": 4,
    EVIDENCE_CONTEXT_SECTION: 4,
    "red_flags_or_escalation_triggers": 6,
    "clinical_inputs_needed": 8,
    "what_ai_must_not_decide": 8,
}

PROHIBITED_MANAGEMENT_ORDER_PHRASES = [
    "must immobilize",
    "must immobilise",
    "should immobilize",
    "should immobilise",
    "apply a cast",
    "put a cast",
    "place in a cast",
    "discharge the patient",
    "send the patient home",
    "weight bearing as tolerated",
    "non weight bearing",
    "non-weight-bearing",
    "prescribe",
    "administer",
    "requires surgery",
    "needs surgery",
    "operate",
]


def _extract_complete_json_object_at(text: str, start: int) -> Dict[str, Any]:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("no_complete_json_object_at_start")


def _extract_management_sections_object(raw_text: str) -> Dict[str, Any]:
    marker = '"management_orientation"'
    marker_idx = raw_text.find(marker)
    if marker_idx < 0:
        raise ValueError("no_management_orientation_section_found")
    start = raw_text.rfind("{", 0, marker_idx)
    if start < 0:
        raise ValueError("no_section_object_start_found")
    section = _extract_complete_json_object_at(raw_text, start)
    return {
        "writer_type": "llm_management_guidance_candidate",
        "llm_used": True,
        "sections": [section],
        "created_new_medical_claims": False,
        "used_raw_pubmed_abstracts": False,
        "reinterpreted_image_pixels": False,
        "issued_patient_specific_treatment_plan": False,
        "partial_outer_json_recovered": True,
    }


@dataclass
class LocalManagementLlmConfig:
    model_id_or_path: str = DEFAULT_MEDGEMMA_MODEL_ID
    local_files_only: bool = True
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 420
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    run_model: bool = False
    attn_implementation: str = "eager"
    require_torch_ge_2_6_for_medgemma: bool = True


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _source_ids(stage3_block: Dict[str, Any]) -> Set[str]:
    return {
        str(source.get("source_id"))
        for source in stage3_block.get("sources", []) or []
        if isinstance(source, dict) and source.get("source_id")
    }


def _source_lookup(stage3_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(source.get("source_id")): source
        for source in stage3_block.get("sources", []) or []
        if isinstance(source, dict) and source.get("source_id")
    }


def _claim_ids(stage3_block: Dict[str, Any]) -> Set[str]:
    return {
        str(claim.get("claim_id"))
        for claim in stage3_block.get("allowed_claims", []) or []
        if isinstance(claim, dict)
        and claim.get("allowed") is True
        and (claim.get("claim_lock") or {}).get("status") == "PASS"
        and claim.get("claim_id")
    }


def _claim_lookup(stage3_block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(claim.get("claim_id")): claim
        for claim in stage3_block.get("allowed_claims", []) or []
        if isinstance(claim, dict) and claim.get("claim_id")
    }


def _suggestion_ids(stage3_block: Dict[str, Any]) -> Set[str]:
    management = ((stage3_block.get("treatment_guidance") or {}).get("management_guidance") or {})
    suggestions = (management.get("case_management_suggestions") or {}).get("suggestions") or []
    return {
        str(item.get("suggestion_id"))
        for item in suggestions
        if isinstance(item, dict) and item.get("suggestion_id")
    }


def _management_panel(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    return ((stage3_block.get("app_payload") or {}).get("management_guidance_panel") or {})


def _display_label(stage3_block: Dict[str, Any]) -> str:
    panel = _management_panel(stage3_block)
    return str(panel.get("anatomy_area") or "the imaged region")


def _confidence_cards(stage3_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        card
        for card in _safe_list((stage3_block.get("app_payload") or {}).get("confidence_cards"))
        if isinstance(card, dict)
    ]


def _evidence_items_from_panel(stage3_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    panel = _management_panel(stage3_block)
    items = panel.get("evidence_rag_synthesis") or panel.get("pubmed_rag_synthesis") or []
    return [item for item in items if isinstance(item, dict)]


def _evidence_claim_ids_from_panel(stage3_block: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in _evidence_items_from_panel(stage3_block):
        if isinstance(item, dict) and item.get("claim_id"):
            out.append(str(item.get("claim_id")))
    return out


def _evidence_source_ids_from_panel(stage3_block: Dict[str, Any]) -> List[str]:
    panel = _management_panel(stage3_block)
    out: List[str] = []
    for item in _evidence_items_from_panel(stage3_block):
        if isinstance(item, dict) and item.get("source_id"):
            out.append(str(item.get("source_id")))
    return list(dict.fromkeys(out))


def _pubmed_claim_ids_from_panel(stage3_block: Dict[str, Any]) -> List[str]:
    panel = _management_panel(stage3_block)
    out: List[str] = []
    for item in panel.get("pubmed_rag_synthesis") or []:
        if isinstance(item, dict) and item.get("claim_id"):
            out.append(str(item.get("claim_id")))
    return out


def build_llm_management_writer_packet(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    """Build the only payload a management-oriented LLM writer may see.

    This packet excludes raw PubMed abstracts and image pixels. It is meant for
    wording management guidance, not for diagnosis or treatment planning.
    """
    panel = _management_panel(stage3_block)
    direction = panel.get("management_direction") or {}
    suggestions = panel.get("case_management_suggestions") or {}
    pubmed_items = panel.get("pubmed_rag_synthesis") or []
    evidence_items = panel.get("evidence_rag_synthesis") or pubmed_items
    scenario = panel.get("case_scenario") or {}
    anatomy_management = panel.get("anatomy_specific_management") or {}
    return {
        "version": f"{LLM_MANAGEMENT_WRITER_VERSION}_packet",
        "writer_role": "controlled_management_guidance_writer",
        "llm_may_create_new_medical_claims": False,
        "llm_may_use_raw_abstracts": False,
        "llm_may_reinterpret_image_pixels": False,
        "case": {
            "image_id": (stage3_block.get("input_summary") or {}).get("image_id"),
            "anatomy_area": panel.get("anatomy_area"),
            "case_status": panel.get("case_status"),
            "case_scenario": scenario.get("scenario_id"),
            "guidance_strength": panel.get("guidance_strength") or scenario.get("guidance_strength"),
            "why_this_section_is_shown": panel.get("why_this_section_is_shown") or scenario.get("why_this_section_is_shown"),
            "management_status": direction.get("status_id"),
        },
        "case_scenario": scenario,
        "confidence_cards": _confidence_cards(stage3_block),
        "management_direction": direction,
        "anatomy_specific_management": anatomy_management,
        "case_management_suggestions": suggestions,
        "evidence_rag_synthesis": evidence_items,
        "pubmed_rag_synthesis": pubmed_items,
        "clinical_inputs_required": panel.get("clinical_inputs_required") or [],
        "clinical_information_not_available_to_ai": panel.get("clinical_information_not_available_to_ai") or [],
        "red_flags": panel.get("red_flags") or [],
        "ai_must_not_decide": panel.get("ai_must_not_decide") or [],
        "sources": panel.get("sources") or [],
        "required_output_schema": {
            "writer_type": "llm_management_guidance_candidate",
            "llm_used": True,
            "sections": {
                "management_orientation": "short clinician-facing paragraph",
                "guidance_level": "one short phrase matching the provided guidance strength",
                "why_this_section_is_shown": "short explanation using the provided scenario only",
                "case_specific_suggestions": ["2-4 concise bullets copied or rephrased from case_management_suggestions"],
                EVIDENCE_CONTEXT_SECTION: ["0-4 concise bullets based only on evidence_rag_synthesis/pubmed_rag_synthesis"],
                "red_flags_or_escalation_triggers": ["short bullets copied or rephrased from red_flags"],
                "clinical_inputs_needed": ["short bullets copied from clinical_inputs_required"],
                "what_ai_must_not_decide": ["short bullets copied from ai_must_not_decide"],
                "safety_boundary": "short boundary statement",
            },
            "used_suggestion_ids": ["suggestion_id values used from case_management_suggestions only"],
            "used_claim_ids": ["claim_id values used from evidence_rag_synthesis/pubmed_rag_synthesis or approved claims only"],
            "used_source_ids": ["source_id values used from sources only"],
            "created_new_medical_claims": False,
            "used_raw_pubmed_abstracts": False,
            "reinterpreted_image_pixels": False,
            "issued_patient_specific_treatment_plan": False,
        },
    }


def build_llm_management_writer_prompt(packet: Dict[str, Any]) -> Dict[str, Any]:
    suggestions = (packet.get("case_management_suggestions") or {}).get("suggestions") or []
    pubmed_items = packet.get("pubmed_rag_synthesis") or []
    evidence_items = packet.get("evidence_rag_synthesis") or pubmed_items
    confidence_cards = packet.get("confidence_cards") or []
    case = packet.get("case") or {}
    scenario = packet.get("case_scenario") or {}
    anatomy_management = packet.get("anatomy_specific_management") or {}
    compact = {
        "case": {
            "image_id": case.get("image_id"),
            "anatomy_area": case.get("anatomy_area"),
            "case_status": case.get("case_status"),
            "scenario_id": case.get("case_scenario") or scenario.get("scenario_id"),
            "guidance_strength": case.get("guidance_strength") or scenario.get("guidance_strength"),
            "why_shown": case.get("why_this_section_is_shown") or scenario.get("why_this_section_is_shown"),
        },
        "confidence_cards": confidence_cards[:4],
        "allowed_anatomy_management_context": {
            "template_id": anatomy_management.get("template_id"),
            "review_focus": anatomy_management.get("review_focus"),
            "case_adjustment": anatomy_management.get("case_adjustment"),
            "anatomy_adjustment": anatomy_management.get("anatomy_adjustment"),
            "clinical_localizers": (anatomy_management.get("clinical_localizers") or [])[:4],
            "management_considerations": (anatomy_management.get("management_considerations") or [])[:3],
            "followup_or_imaging_context": (anatomy_management.get("followup_or_imaging_context") or [])[:2],
        },
        "allowed_suggestions": [
            {
                "suggestion_id": item.get("suggestion_id"),
                "text": item.get("text"),
            }
            for item in suggestions[:2]
            if isinstance(item, dict)
        ],
        "allowed_evidence_context": [
            {
                "claim_id": item.get("claim_id"),
                "source_id": item.get("source_id"),
                "pmid": item.get("pmid"),
                "text": item.get("clinician_facing_summary") or item.get("text"),
                "limitations": item.get("limitations") or [],
                "evidence_lane": item.get("evidence_lane") or "pubmed",
            }
            for item in evidence_items[:1]
            if isinstance(item, dict)
        ],
        "clinical_inputs_required": (packet.get("clinical_inputs_required") or [])[:5],
        "clinical_information_not_available_to_ai": (
            packet.get("clinical_information_not_available_to_ai") or []
        )[:5],
        "red_flags": (packet.get("red_flags") or [])[:4],
        "ai_must_not_decide": (packet.get("ai_must_not_decide") or [])[:5],
    }
    return {
        "version": f"{LLM_MANAGEMENT_WRITER_VERSION}_prompt",
        "llm_called": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a controlled medical writing assistant for a fracture X-ray prototype. "
                    "You write clinician-facing management orientation, not diagnosis and not a treatment plan."
                ),
            },
            {
                "role": "developer",
                "content": (
                    "Return one complete JSON object only. Start with {\"writer_type\". Do not repeat the input. "
                    "Do not wrap output in markdown or code fences. "
                    "Use only allowed_suggestions and allowed_evidence_context. Do not add medical facts. Do not prescribe. "
                    "Keep management_orientation under 35 words. Return exactly two case_specific_suggestions by copying "
                    "allowed_suggestions[].text exactly when allowed_suggestions are present, and copy their suggestion_id values. "
                    "Return at most one evidence_supported_context bullet from allowed_evidence_context. "
                    "Use case_scenario and guidance_strength to choose conservative wording. "
                    "Copy short items from clinical_inputs_required, red_flags, and ai_must_not_decide when producing those sections. "
                    "Use clinical_information_not_available_to_ai to avoid implying knowledge of symptoms, mechanism, examination, "
                    "mobility, wound status, or formal radiology review. "
                    "Do not decide immobilisation, weight-bearing, medication, procedure, follow-up interval, "
                    "clearance/reassurance, or surgery. Every evidence statement must use a supplied claim/source id."
                ),
            },
            {
                "role": "user",
                "content_type": "application/json",
                "content": {
                    "instruction": (
                        "Use FACTS only. Do not echo FACTS. Return JSON only with keys: writer_type, llm_used, sections, "
                        "used_suggestion_ids, used_claim_ids, used_source_ids, created_new_medical_claims, "
                        "used_raw_pubmed_abstracts, reinterpreted_image_pixels, issued_patient_specific_treatment_plan. "
                        "sections must include: management_orientation, guidance_level, why_this_section_is_shown, "
                        "case_specific_suggestions, evidence_supported_context, red_flags_or_escalation_triggers, "
                        "clinical_inputs_needed, what_ai_must_not_decide, safety_boundary. "
                        "All boolean safety flags must be false. safety_boundary must say: This is not a treatment plan. "
                        "Human clinical verification is required."
                    ),
                    "FACTS": compact,
                },
            },
        ],
        "generation_policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "json_only": True,
            "max_output_tokens": 650,
        },
    }


def build_deterministic_management_writer_fallback(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    panel = _management_panel(stage3_block)
    direction = panel.get("management_direction") or {}
    suggestions = (panel.get("case_management_suggestions") or {}).get("suggestions") or []
    evidence_items = _evidence_items_from_panel(stage3_block)
    suggestion_bullets = [
        str(item.get("text"))
        for item in suggestions[:3]
        if isinstance(item, dict) and item.get("text")
    ]
    evidence_bullets = [
        str(item.get("clinician_facing_summary") or item.get("text"))
        for item in evidence_items[:3]
        if isinstance(item, dict) and (item.get("clinician_facing_summary") or item.get("text"))
    ]
    response = {
        "writer_type": "deterministic_management_guidance_fallback_v1",
        "llm_used": False,
        "sections": {
            "management_orientation": direction.get("summary")
            or "This output can orient clinician review but is not a treatment plan.",
            "guidance_level": str((panel.get("case_scenario") or {}).get("guidance_strength") or "clinician_context"),
            "why_this_section_is_shown": str(
                panel.get("why_this_section_is_shown")
                or (panel.get("case_scenario") or {}).get("why_this_section_is_shown")
                or "This section is shown to provide clinician-facing management context."
            ),
            "case_specific_suggestions": suggestion_bullets,
            EVIDENCE_CONTEXT_SECTION: evidence_bullets,
            "red_flags_or_escalation_triggers": (panel.get("red_flags") or [])[:4],
            "clinical_inputs_needed": (panel.get("clinical_inputs_required") or [])[:5],
            "what_ai_must_not_decide": (panel.get("ai_must_not_decide") or [])[:5],
            "safety_boundary": (
                "This is not a treatment plan. Human clinical verification is required. "
                + str(
                    direction.get("decision_boundary")
                    or "Model scores are not clinical probabilities."
                )
            ),
        },
        "used_suggestion_ids": [
            str(item.get("suggestion_id"))
            for item in suggestions[:3]
            if isinstance(item, dict) and item.get("suggestion_id")
        ],
        "used_claim_ids": _evidence_claim_ids_from_panel(stage3_block)[:3],
        "used_source_ids": _evidence_source_ids_from_panel(stage3_block)[:3],
        "created_new_medical_claims": False,
        "used_raw_pubmed_abstracts": False,
        "reinterpreted_image_pixels": False,
        "issued_patient_specific_treatment_plan": False,
    }
    return {
        "version": f"{LLM_MANAGEMENT_WRITER_VERSION}_fallback",
        "response": response,
        "validation": validate_llm_management_writer_candidate(
            stage3_block,
            {**response, "writer_type": "llm_management_guidance_candidate", "llm_used": True},
            require_llm_used=True,
        ),
    }


def _section_text(response: Dict[str, Any]) -> str:
    sections = response.get("sections") or {}
    values: List[str] = []
    if isinstance(sections, dict):
        for value in sections.values():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
    return "\n".join(values)


def _list_section(sections: Dict[str, Any], key: str) -> List[Any]:
    value = sections.get(key)
    return value if isinstance(value, list) else []


def _evidence_context_section(sections: Dict[str, Any]) -> List[Any]:
    value = sections.get(EVIDENCE_CONTEXT_SECTION)
    if isinstance(value, list):
        return value
    legacy = sections.get(LEGACY_EVIDENCE_CONTEXT_SECTION)
    return legacy if isinstance(legacy, list) else []


def _safety_boundary_errors(text: str) -> List[str]:
    lowered = text.lower()
    errors: List[str] = []
    if "not a treatment plan" not in lowered:
        errors.append("missing_boundary_meaning:not a treatment plan")
    if "human" not in lowered:
        errors.append("missing_boundary_meaning:human")
    if "verification" not in lowered:
        errors.append("missing_boundary_meaning:verification")
    return errors


def _prohibited_order_hits(text: str) -> List[str]:
    lowered = text.lower()
    return [phrase for phrase in PROHIBITED_MANAGEMENT_ORDER_PHRASES if phrase in lowered]


def _source_id_for_claim(claim: Dict[str, Any]) -> str | None:
    source_id = claim.get("source_id")
    if source_id:
        return str(source_id)
    source_ids = claim.get("source_ids")
    if isinstance(source_ids, list) and source_ids:
        return str(source_ids[0])
    return None


def validate_llm_management_writer_candidate(
    stage3_block: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    require_llm_used: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if candidate.get("writer_type") != "llm_management_guidance_candidate":
        errors.append(f"unexpected_writer_type:{candidate.get('writer_type')}")
    if require_llm_used and candidate.get("llm_used") is not True:
        errors.append("llm_used_not_true")

    sections = candidate.get("sections")
    if not isinstance(sections, dict):
        errors.append("sections_not_dict")
        sections = {}
    for key in REQUIRED_MANAGEMENT_SECTIONS:
        if key == EVIDENCE_CONTEXT_SECTION:
            if key not in sections and LEGACY_EVIDENCE_CONTEXT_SECTION not in sections:
                errors.append(f"missing_section:{key}")
            continue
        if key not in sections:
            errors.append(f"missing_section:{key}")
    for key in LIST_MANAGEMENT_SECTIONS:
        value = _evidence_context_section(sections) if key == EVIDENCE_CONTEXT_SECTION else sections.get(key)
        if not isinstance(value, list):
            errors.append(f"{key}_not_list")
        elif len(value or []) > MAX_MANAGEMENT_SECTION_ITEMS[key]:
            errors.append(f"{key}_too_many_items:{len(value or [])}>{MAX_MANAGEMENT_SECTION_ITEMS[key]}")
    for key in ["management_orientation", "guidance_level", "why_this_section_is_shown", "safety_boundary"]:
        if key in sections and not isinstance(sections.get(key), str):
            errors.append(f"{key}_not_string")

    text = _section_text(candidate)
    forbidden_hits = find_forbidden_phrases(candidate)
    if forbidden_hits:
        errors.append("forbidden_phrases_detected")
        errors.extend(f"forbidden_phrase:{hit.get('phrase')}" for hit in forbidden_hits)
    order_hits = _prohibited_order_hits(text)
    if order_hits:
        errors.append("prohibited_management_order_detected")
        errors.extend(f"prohibited_management_order:{hit}" for hit in order_hits)

    errors.extend(_safety_boundary_errors(str(sections.get("safety_boundary") or "")))

    if candidate.get("created_new_medical_claims") is not False:
        errors.append("created_new_medical_claims_not_false")
    if candidate.get("used_raw_pubmed_abstracts") is not False:
        errors.append("used_raw_pubmed_abstracts_not_false")
    if candidate.get("reinterpreted_image_pixels") is not False:
        errors.append("reinterpreted_image_pixels_not_false")
    if candidate.get("issued_patient_specific_treatment_plan") is not False:
        errors.append("issued_patient_specific_treatment_plan_not_false")

    allowed_suggestion_ids = _suggestion_ids(stage3_block)
    allowed_claim_ids = _claim_ids(stage3_block)
    available_source_ids = _source_ids(stage3_block)
    claims_by_id = _claim_lookup(stage3_block)
    sources_by_id = _source_lookup(stage3_block)

    used_suggestion_ids = candidate.get("used_suggestion_ids") or []
    used_claim_ids = candidate.get("used_claim_ids") or []
    used_source_ids = candidate.get("used_source_ids") or []
    if not isinstance(used_suggestion_ids, list):
        errors.append("used_suggestion_ids_not_list")
        used_suggestion_ids = []
    if not isinstance(used_claim_ids, list):
        errors.append("used_claim_ids_not_list")
        used_claim_ids = []
    if not isinstance(used_source_ids, list):
        errors.append("used_source_ids_not_list")
        used_source_ids = []

    for suggestion_id in used_suggestion_ids:
        if str(suggestion_id) not in allowed_suggestion_ids:
            errors.append(f"suggestion_id_not_allowed:{suggestion_id}")
    for claim_id in used_claim_ids:
        claim_key = str(claim_id)
        claim = claims_by_id.get(claim_key)
        if claim_key not in allowed_claim_ids:
            errors.append(f"claim_id_not_allowed:{claim_id}")
            if claim and (claim.get("claim_lock") or {}).get("status") != "PASS":
                errors.append(f"claim_lock_not_pass:{claim_id}")
            continue
        source_id = _source_id_for_claim(claim or {})
        if source_id and source_id not in {str(item) for item in used_source_ids}:
            errors.append(f"claim_source_not_cited:{claim_id}:{source_id}")
        if source_id and source_id not in available_source_ids:
            errors.append(f"claim_source_not_available:{claim_id}:{source_id}")
    for source_id in used_source_ids:
        source_key = str(source_id)
        if source_key not in available_source_ids:
            errors.append(f"source_id_not_available:{source_id}")
        else:
            _ = sources_by_id.get(source_key)

    case_suggestions = _list_section(sections, "case_specific_suggestions")
    evidence_context = _evidence_context_section(sections)
    used_claim_id_set = {str(item) for item in used_claim_ids}
    used_source_id_set = {str(item) for item in used_source_ids}
    if case_suggestions and not used_suggestion_ids:
        errors.append("case_suggestions_without_suggestion_ids")
    if evidence_context and not used_claim_ids:
        errors.append("evidence_context_without_claim_ids")
    if evidence_context and not used_source_ids:
        errors.append("evidence_context_without_source_ids")
    if used_claim_id_set and not used_source_id_set:
        errors.append("claim_ids_without_source_ids")

    if not used_suggestion_ids:
        warnings.append("no_suggestion_ids_used")
    if _evidence_claim_ids_from_panel(stage3_block) and not used_claim_ids:
        warnings.append("evidence_available_but_no_claim_ids_used")

    return {
        "validator": LLM_MANAGEMENT_VALIDATOR_VERSION,
        "verdict": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checked_suggestion_count": len(used_suggestion_ids),
        "checked_claim_count": len(used_claim_ids),
        "checked_source_count": len(used_source_ids),
        "forbidden_hit_count": len(forbidden_hits),
        "prohibited_order_hit_count": len(order_hits),
    }


def build_llm_management_writer_contract(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    packet = build_llm_management_writer_packet(stage3_block)
    fallback = build_deterministic_management_writer_fallback(stage3_block)
    return {
        "version": LLM_MANAGEMENT_WRITER_VERSION,
        "purpose": "Controlled PubMed/RAG management-guidance writer contract for a future local LLM.",
        "status": "contract_only_no_llm_call",
        "llm_called": False,
        "production_enabled": False,
        "input_packet": packet,
        "prompt_contract": build_llm_management_writer_prompt(packet),
        "validator": {
            "validator": LLM_MANAGEMENT_VALIDATOR_VERSION,
            "hard_fail_conditions": [
                "forbidden phrase",
                "missing required management section",
                "section length exceeds controlled limit",
                "unapproved suggestion id",
                "unapproved claim id",
                "claim without PASS claim-lock",
                "PubMed context without approved claim/source ids",
                "claim source not cited",
                "unavailable source id",
                "prohibited management order",
                "raw PubMed abstract use",
                "new medical claim creation",
                "patient-specific treatment plan",
                "image reinterpretation",
            ],
        },
        "fallback_if_llm_missing_or_validation_fails": fallback,
        "display_response": fallback["response"],
        "validation": fallback["validation"],
        "safety_policy": {
            "uses_only_case_management_suggestions": True,
            "uses_only_approved_pubmed_claims": True,
            "raw_pubmed_abstracts_allowed": False,
            "new_medical_claim_creation_allowed": False,
            "patient_specific_treatment_plan_allowed": False,
            "human_verification_required": True,
        },
        "adapter_audit": {
            "suggestion_count": len((packet.get("case_management_suggestions") or {}).get("suggestions") or []),
            "pubmed_synthesis_count": len(packet.get("pubmed_rag_synthesis") or []),
            "source_count": len(packet.get("sources") or []),
            "fallback_validation_verdict": (fallback.get("validation") or {}).get("verdict"),
        },
    }


def _candidate_from_model_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    sections = parsed.get("sections") if isinstance(parsed.get("sections"), dict) else {}
    if not sections and isinstance(parsed.get("sections"), list) and parsed.get("sections"):
        first = parsed["sections"][0]
        if isinstance(first, dict):
            sections = first
    if not sections:
        sections = {
            "management_orientation": parsed.get("management_orientation"),
            "case_specific_suggestions": parsed.get("case_specific_suggestions"),
            EVIDENCE_CONTEXT_SECTION: parsed.get(EVIDENCE_CONTEXT_SECTION)
            or parsed.get(LEGACY_EVIDENCE_CONTEXT_SECTION),
            "safety_boundary": parsed.get("safety_boundary"),
        }
    used_suggestion_ids = list(_safe_list(parsed.get("used_suggestion_ids")))
    used_claim_ids = list(_safe_list(parsed.get("used_claim_ids")))
    used_source_ids = list(_safe_list(parsed.get("used_source_ids")))

    def _text_list(value: Any, *, kind: str) -> List[str]:
        out: List[str] = []
        for item in _safe_list(value):
            if isinstance(item, dict):
                text = item.get("text") or item.get("summary") or item.get("value")
                if text:
                    out.append(str(text))
                if kind == "suggestion" and item.get("suggestion_id"):
                    used_suggestion_ids.append(str(item.get("suggestion_id")))
                if kind == "pubmed":
                    if item.get("claim_id"):
                        used_claim_ids.append(str(item.get("claim_id")))
                    if item.get("source_id"):
                        used_source_ids.append(str(item.get("source_id")))
            else:
                out.append(str(item))
        return out

    return {
        "writer_type": "llm_management_guidance_candidate",
        "llm_used": True,
        "sections": {
            "management_orientation": str(sections.get("management_orientation") or ""),
            "guidance_level": str(sections.get("guidance_level") or ""),
            "why_this_section_is_shown": str(sections.get("why_this_section_is_shown") or ""),
            "case_specific_suggestions": _text_list(sections.get("case_specific_suggestions"), kind="suggestion"),
            EVIDENCE_CONTEXT_SECTION: _text_list(
                sections.get(EVIDENCE_CONTEXT_SECTION) or sections.get(LEGACY_EVIDENCE_CONTEXT_SECTION),
                kind="pubmed",
            ),
            "red_flags_or_escalation_triggers": _text_list(sections.get("red_flags_or_escalation_triggers"), kind="generic"),
            "clinical_inputs_needed": _text_list(sections.get("clinical_inputs_needed"), kind="generic"),
            "what_ai_must_not_decide": _text_list(sections.get("what_ai_must_not_decide"), kind="generic"),
            "safety_boundary": str(sections.get("safety_boundary") or ""),
        },
        "used_suggestion_ids": list(dict.fromkeys(str(item) for item in used_suggestion_ids if item)),
        "used_claim_ids": list(dict.fromkeys(str(item) for item in used_claim_ids if item)),
        "used_source_ids": list(dict.fromkeys(str(item) for item in used_source_ids if item)),
        "created_new_medical_claims": parsed.get("created_new_medical_claims") is True,
        "used_raw_pubmed_abstracts": parsed.get("used_raw_pubmed_abstracts") is True,
        "reinterpreted_image_pixels": parsed.get("reinterpreted_image_pixels") is True,
        "issued_patient_specific_treatment_plan": parsed.get("issued_patient_specific_treatment_plan") is True,
    }


def _repair_candidate_references(stage3_block: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing IDs only when output text matches approved input text."""
    panel = _management_panel(stage3_block)
    suggestions = (panel.get("case_management_suggestions") or {}).get("suggestions") or []
    evidence_items = _evidence_items_from_panel(stage3_block)
    sections = candidate.get("sections") or {}

    used_suggestion_ids = [str(item) for item in _safe_list(candidate.get("used_suggestion_ids")) if item]
    used_claim_ids = [str(item) for item in _safe_list(candidate.get("used_claim_ids")) if item]
    used_source_ids = [str(item) for item in _safe_list(candidate.get("used_source_ids")) if item]

    suggestion_texts = [str(item) for item in _safe_list(sections.get("case_specific_suggestions"))]
    pubmed_texts = [str(item) for item in _evidence_context_section(sections)]
    repaired_suggestion_texts: List[str] = []
    repaired_pubmed_texts: List[str] = []

    suggestion_by_id = {
        str(item.get("suggestion_id")): item
        for item in suggestions
        if isinstance(item, dict) and item.get("suggestion_id")
    }
    pubmed_by_claim_id = {
        str(item.get("claim_id")): item
        for item in evidence_items
        if isinstance(item, dict) and item.get("claim_id")
    }

    for out_text in suggestion_texts:
        if out_text in suggestion_by_id:
            suggestion = suggestion_by_id[out_text]
            source_text = str(suggestion.get("text") or out_text)
            repaired_suggestion_texts.append(source_text)
            if out_text not in used_suggestion_ids:
                used_suggestion_ids.append(out_text)
            for source_id in suggestion.get("source_ids") or []:
                if source_id and str(source_id) not in used_source_ids:
                    used_source_ids.append(str(source_id))
            continue

        repaired_suggestion_texts.append(out_text)
        out_norm = " ".join(out_text.lower().split())
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            source_text = str(suggestion.get("text") or "")
            source_norm = " ".join(source_text.lower().split())
            if source_norm and (out_norm == source_norm or out_norm in source_norm or source_norm in out_norm):
                suggestion_id = suggestion.get("suggestion_id")
                if suggestion_id and str(suggestion_id) not in used_suggestion_ids:
                    used_suggestion_ids.append(str(suggestion_id))
                for source_id in suggestion.get("source_ids") or []:
                    if source_id and str(source_id) not in used_source_ids:
                        used_source_ids.append(str(source_id))

    for out_text in pubmed_texts:
        if out_text in pubmed_by_claim_id:
            item = pubmed_by_claim_id[out_text]
            source_text = str(item.get("clinician_facing_summary") or item.get("text") or out_text)
            repaired_pubmed_texts.append(source_text)
            claim_id = item.get("claim_id")
            source_id = item.get("source_id")
            if claim_id and str(claim_id) not in used_claim_ids:
                used_claim_ids.append(str(claim_id))
            if source_id and str(source_id) not in used_source_ids:
                used_source_ids.append(str(source_id))
            continue

        repaired_pubmed_texts.append(out_text)
        out_norm = " ".join(out_text.lower().split())
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            source_text = str(item.get("clinician_facing_summary") or item.get("text") or "")
            source_norm = " ".join(source_text.lower().split())
            if source_norm and (out_norm == source_norm or out_norm in source_norm or source_norm in out_norm):
                claim_id = item.get("claim_id")
                source_id = item.get("source_id")
                if claim_id and str(claim_id) not in used_claim_ids:
                    used_claim_ids.append(str(claim_id))
                if source_id and str(source_id) not in used_source_ids:
                    used_source_ids.append(str(source_id))

    repaired_sections = dict(sections)
    repaired_sections["case_specific_suggestions"] = repaired_suggestion_texts
    repaired_sections[EVIDENCE_CONTEXT_SECTION] = repaired_pubmed_texts
    repaired_sections.pop(LEGACY_EVIDENCE_CONTEXT_SECTION, None)

    repaired = dict(candidate)
    repaired["sections"] = repaired_sections
    repaired["used_suggestion_ids"] = used_suggestion_ids
    repaired["used_claim_ids"] = used_claim_ids
    repaired["used_source_ids"] = used_source_ids
    repaired["reference_repair"] = {
        "applied": True,
        "policy": "IDs are inferred only from exact or containment matches against approved suggestion/PubMed text.",
        "used_suggestion_id_count": len(used_suggestion_ids),
        "used_claim_id_count": len(used_claim_ids),
        "used_source_id_count": len(used_source_ids),
    }
    return repaired


def _move_inputs_to_model_device(inputs: Dict[str, Any], model: Any) -> Dict[str, Any]:
    if not hasattr(model, "device"):
        return inputs
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(model.device) if hasattr(value, "to") else value
    return out


def run_local_llm_management_writer(
    stage3_block: Dict[str, Any],
    config: LocalManagementLlmConfig | None = None,
) -> Dict[str, Any]:
    """Run the local management LLM writer behind strict validation."""
    cfg = config or LocalManagementLlmConfig()
    started = time.time()
    contract = build_llm_management_writer_contract(stage3_block)
    deps = dependency_report()
    status = model_path_status(cfg.model_id_or_path)

    base = {
        **contract,
        "status": contract.get("status"),
        "llm_called": False,
        "production_enabled": False,
        "config": asdict(cfg),
        "dependency_report": deps,
        "model_path_status": status,
    }

    if not cfg.run_model:
        return {
            **base,
            "status": "SKIPPED_MODEL_RUN_DISABLED",
            "display_response": contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "validation": contract["fallback_if_llm_missing_or_validation_fails"]["validation"],
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if not deps.get("torch") or not deps.get("transformers"):
        return {
            **base,
            "status": "SKIPPED_DEPENDENCY_MISSING",
            "display_response": contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "validation": contract["fallback_if_llm_missing_or_validation_fails"]["validation"],
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if cfg.local_files_only and not status.get("config_json_exists"):
        return {
            **base,
            "status": "SKIPPED_MODEL_UNAVAILABLE_LOCAL_CACHE",
            "display_response": contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "validation": contract["fallback_if_llm_missing_or_validation_fails"]["validation"],
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if (
        cfg.require_torch_ge_2_6_for_medgemma
        and "medgemma" in cfg.model_id_or_path.lower()
        and deps.get("torch_ge_2_6") is not True
    ):
        return {
            **base,
            "status": "SKIPPED_TORCH_VERSION_UNSUPPORTED",
            "required_torch_version": ">=2.6",
            "current_torch_version": deps.get("torch_version"),
            "display_response": contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "validation": contract["fallback_if_llm_missing_or_validation_fails"]["validation"],
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    raw_text = None
    generation_reached = False
    try:
        import torch

        loaded = _load_hf_text_model(
            type(
                "CompatConfig",
                (),
                {
                    "model_id_or_path": cfg.model_id_or_path,
                    "local_files_only": cfg.local_files_only,
                    "device": cfg.device,
                    "torch_dtype": cfg.torch_dtype,
                    "trust_remote_code": cfg.trust_remote_code,
                    "attn_implementation": cfg.attn_implementation,
                },
            )()
        )
        model = loaded["model"]
        tokenizer = loaded.get("tokenizer")
        processor = loaded.get("processor")
        inputs = _build_generation_inputs(contract["prompt_contract"], loaded)
        inputs = _move_inputs_to_model_device(inputs, model)

        with torch.no_grad():
            tokenizer_like = tokenizer or getattr(processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer_like, "pad_token_id", None)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                top_p=cfg.top_p,
                pad_token_id=(
                    getattr(tokenizer, "eos_token_id", None)
                    or getattr(processor, "eos_token_id", None)
                    or getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                ),
                bad_words_ids=[[pad_token_id]] if pad_token_id is not None else None,
                suppress_tokens=[pad_token_id] if pad_token_id is not None else None,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids[0]
        decoder = tokenizer or processor
        if decoder is None or not hasattr(decoder, "decode"):
            raise RuntimeError("no_decoder_available")
        raw_text = decoder.decode(generated_ids, skip_special_tokens=True)
        generation_reached = True
        try:
            parsed = extract_json_object(raw_text)
            parser_recovery = None
        except Exception:
            parsed = _extract_management_sections_object(raw_text)
            parser_recovery = {
                "applied": True,
                "policy": (
                    "Recovered the first complete management sections object only; ignored incomplete extra tail. "
                    "Recovered content is still validated against approved suggestion, claim, and source IDs."
                ),
            }
        candidate = _repair_candidate_references(stage3_block, _candidate_from_model_output(parsed))
        validation = validate_llm_management_writer_candidate(stage3_block, candidate)
        accepted = validation.get("verdict") == "PASS"
        out = {
            **base,
            "status": "PASS" if accepted else "FAIL_VALIDATION_FALLBACK_USED",
            "llm_called": True,
            "raw_model_text": raw_text,
            "candidate_response": candidate,
            "validation": validation,
            "display_response": candidate if accepted else contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "fallback_used": not accepted,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if parser_recovery:
            out["parser_recovery"] = parser_recovery
        return out
    except Exception as exc:
        out = {
            **base,
            "status": "ERROR_FALLBACK_USED",
            "llm_called": generation_reached,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "display_response": contract["fallback_if_llm_missing_or_validation_fails"]["response"],
            "validation": contract["fallback_if_llm_missing_or_validation_fails"]["validation"],
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if raw_text is not None:
            out["raw_model_text_on_error"] = raw_text
            out["raw_model_text_on_error_preview"] = raw_text[:1000]
        return out
