from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


def _source_type_counts(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for source in sources:
        source_type = str(source.get("source_type") or "missing")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _rag_mode(sources: List[Dict[str, Any]]) -> str:
    source_types = {source.get("source_type") for source in sources}
    has_pubmed = "pubmed" in source_types
    has_guideline = "guideline_metadata" in source_types
    if has_pubmed and has_guideline:
        return "guideline_registry_plus_pubmed"
    if has_pubmed:
        return "pubmed_only"
    if has_guideline:
        return "guideline_registry_only"
    return "local_curated_only"


def _safe_source(source: Dict[str, Any]) -> Dict[str, Any]:
    quality = source.get("pubmed_quality") or {}
    relevance = source.get("source_relevance") or {}
    return {
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "title": source.get("title"),
        "organization": source.get("organization"),
        "url": source.get("url"),
        "evidence_level": source.get("evidence_level"),
        "relevance": relevance,
        "evidence_lane": _source_lane(source),
        "quality_tier": quality.get("quality_tier") or _guideline_quality_tier(source),
        "source_role": quality.get("source_role") or _non_pubmed_source_role(source),
        "allowed_for_llm_context": _source_allowed_for_llm(source),
        "allowed_for_treatment_guidance": _source_allowed_for_treatment(source),
        "allowed_for_diagnosis_context": _source_allowed_for_diagnosis(source),
        "supports": _source_supports(source),
        "limitations": _source_limitations(source),
        "pubmed": {
            "pmid": source.get("pmid"),
            "journal": source.get("journal"),
            "year": source.get("year"),
            "publication_types": source.get("publication_types") or [],
            "abstract_available": source.get("abstract_available") is True,
            "abstract_exposed": False,
        }
        if source.get("source_type") == "pubmed"
        else None,
        "pubmed_quality": source.get("pubmed_quality") if source.get("source_type") == "pubmed" else None,
    }


def _source_lane(source: Dict[str, Any]) -> str:
    source_type = source.get("source_type")
    if source_type == "pubmed":
        return "pubmed_live_or_cached_literature"
    if source_type == "guideline_metadata":
        return "curated_guideline_registry"
    if source_type == "local_curated_project_rule":
        return "local_project_safety_rule"
    return "other_or_unknown_source"


def _guideline_quality_tier(source: Dict[str, Any]) -> str:
    if source.get("source_type") == "guideline_metadata":
        return "curated_guideline"
    if source.get("source_type") == "local_curated_project_rule":
        return "local_safety_rule"
    return "unclassified"


def _non_pubmed_source_role(source: Dict[str, Any]) -> str:
    if source.get("source_type") == "guideline_metadata":
        return "guideline_management_context"
    if source.get("source_type") == "local_curated_project_rule":
        return "project_safety_context"
    return "unclassified_context"


def _source_allowed_for_llm(source: Dict[str, Any]) -> bool:
    if source.get("source_type") == "pubmed":
        return (source.get("pubmed_quality") or {}).get("allowed_for_llm_context") is True
    return source.get("source_type") in {"guideline_metadata", "local_curated_project_rule"}


def _source_allowed_for_treatment(source: Dict[str, Any]) -> bool:
    if source.get("source_type") == "pubmed":
        return (source.get("pubmed_quality") or {}).get("allowed_for_treatment_guidance") is True
    return source.get("source_type") == "guideline_metadata"


def _source_allowed_for_diagnosis(source: Dict[str, Any]) -> bool:
    if source.get("source_type") == "pubmed":
        return (source.get("pubmed_quality") or {}).get("allowed_for_diagnosis_context") is True
    return False


def _source_supports(source: Dict[str, Any]) -> List[str]:
    if source.get("source_type") == "pubmed":
        quality = source.get("pubmed_quality") or {}
        role = quality.get("source_role") or "PubMed background context"
        tier = quality.get("quality_tier") or "unclassified"
        return [
            f"PubMed source retained as {role} with quality tier {tier}.",
            "May support only approved claim-level context after quality/relevance filtering.",
        ]
    if source.get("source_type") == "guideline_metadata":
        topics = ", ".join(str(item) for item in source.get("topic_scope") or [])
        return [
            f"Curated guideline metadata retained for clinician-facing context{': ' + topics if topics else ''}.",
            "May support general management context but not patient-specific decisions.",
        ]
    if source.get("source_type") == "local_curated_project_rule":
        return ["Local project safety rule retained to constrain wording and prevent overclaiming."]
    return ["Source retained as background context only."]


def _source_limitations(source: Dict[str, Any]) -> List[str]:
    limitations = [
        "Does not provide patient-specific diagnosis or treatment instructions.",
        "Must be interpreted with symptoms, examination, full imaging review, and local protocol.",
    ]
    if source.get("source_type") == "pubmed":
        limitations.append("Raw abstract is not exposed to the public report or unconstrained LLM generation.")
        if (source.get("pubmed_quality") or {}).get("allowed_for_treatment_guidance") is not True:
            limitations.append("Not allowed for treatment-management guidance unless claim-level filtering approves it.")
    elif source.get("source_type") == "guideline_metadata":
        limitations.append("Guideline metadata is summarized context, not a complete local clinical pathway.")
    return limitations


def _source_lanes(sources: List[Dict[str, Any]], source_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    lanes: Dict[str, Dict[str, Any]] = {}
    card_by_id = {str(card.get("source_id")): card for card in source_cards if card.get("source_id")}
    for source in sources:
        lane_id = _source_lane(source)
        lane = lanes.setdefault(
            lane_id,
            {
                "lane_id": lane_id,
                "source_count": 0,
                "allowed_for_llm_count": 0,
                "allowed_for_treatment_guidance_count": 0,
                "source_ids": [],
                "quality_tiers": {},
                "roles": {},
            },
        )
        lane["source_count"] += 1
        source_id = str(source.get("source_id")) if source.get("source_id") else None
        if source_id:
            lane["source_ids"].append(source_id)
        if _source_allowed_for_llm(source):
            lane["allowed_for_llm_count"] += 1
        if _source_allowed_for_treatment(source):
            lane["allowed_for_treatment_guidance_count"] += 1
        card = card_by_id.get(str(source.get("source_id"))) or _safe_source(source)
        tier = str(card.get("quality_tier") or "unclassified")
        role = str(card.get("source_role") or "unclassified_context")
        lane["quality_tiers"][tier] = lane["quality_tiers"].get(tier, 0) + 1
        lane["roles"][role] = lane["roles"].get(role, 0) + 1
    return {
        "version": "stage3_rag_source_lanes_v1",
        "lane_count": len(lanes),
        "lanes": list(lanes.values()),
    }


def _evidence_strength(mode: str, source_cards: List[Dict[str, Any]], allowed_for_llm: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_curated_guideline = any(card.get("evidence_lane") == "curated_guideline_registry" for card in source_cards)
    has_treatment_pubmed = any(
        card.get("evidence_lane") == "pubmed_live_or_cached_literature"
        and card.get("allowed_for_treatment_guidance") is True
        for card in source_cards
    )
    has_pubmed_claims = any(claim.get("allowed_for_treatment_guidance") is True for claim in allowed_for_llm)
    if has_curated_guideline and has_treatment_pubmed and has_pubmed_claims:
        level = "guideline_plus_pubmed_claims"
        label = "Guideline-backed with PubMed claim support"
    elif has_curated_guideline:
        level = "guideline_backed"
        label = "Guideline-backed context"
    elif has_treatment_pubmed and has_pubmed_claims:
        level = "pubmed_supported_limited"
        label = "PubMed-supported but limited context"
    elif source_cards:
        level = "anatomy_general_or_background_only"
        label = "Background context only"
    else:
        level = "insufficient_evidence_context"
        label = "Insufficient evidence context"
    return {
        "version": "stage3_rag_evidence_strength_v1",
        "level": level,
        "label": label,
        "rag_mode": mode,
        "has_curated_guideline": has_curated_guideline,
        "has_treatment_pubmed_source": has_treatment_pubmed,
        "has_pubmed_treatment_claim": has_pubmed_claims,
        "interpretation": (
            "This describes support for management-context wording only. It is not confidence in diagnosis, "
            "fracture presence, or patient-specific treatment."
        ),
    }


def _readiness_gate(
    source_cards: List[Dict[str, Any]],
    allowed_for_llm: List[Dict[str, Any]],
    evidence_strength: Dict[str, Any],
) -> Dict[str, Any]:
    failures: List[str] = []
    if not source_cards:
        failures.append("no_sources_available")
    if any(card.get("pubmed", {}).get("abstract_exposed") is True for card in source_cards if isinstance(card.get("pubmed"), dict)):
        failures.append("raw_pubmed_abstract_exposed")
    if not all(claim.get("allowed_for_llm") is True for claim in allowed_for_llm):
        failures.append("claim_not_allowed_for_llm_present")
    if evidence_strength.get("level") == "insufficient_evidence_context":
        failures.append("insufficient_evidence_context")
    return {
        "version": "stage3_rag_readiness_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "llm_pack_ready": not failures,
        "failure_reasons": failures,
        "required_properties": [
            "sources are classified into evidence lanes",
            "raw PubMed abstracts are not exposed",
            "LLM may use only approved claim cards",
            "evidence strength is explicitly labeled",
        ],
    }


def _claim_for_llm(claim: Dict[str, Any], sources_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source_id = claim.get("source_id")
    source = sources_by_id.get(str(source_id))
    claim_lock = claim.get("claim_lock") or {}
    allowed = claim.get("allowed") is True and source is not None and claim_lock.get("status") == "PASS"
    blocked_reason = None if allowed else "claim_not_allowed_or_missing_source"
    if claim.get("allowed") is True and claim_lock.get("status") != "PASS":
        blocked_reason = "claim_lock_not_pass"
    quality = (source or {}).get("pubmed_quality") or {}
    if allowed and quality and quality.get("allowed_for_llm_context") is not True:
        allowed = False
        blocked_reason = "pubmed_source_quality_gate_blocked_llm_context"
    return {
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("claim_text"),
        "source_id": source_id,
        "evidence_level": claim.get("evidence_level"),
        "anatomy_relevance": claim.get("anatomy_relevance"),
        "claim_type": claim.get("claim_type") or "controlled_context",
        "claim_role": claim.get("claim_role"),
        "claim_lock": claim_lock,
        "supports": claim.get("supports") or [],
        "limitations": claim.get("limitations") or [],
        "llm_visibility": claim.get("llm_visibility") or {},
        "allowed_for_llm": allowed,
        "blocked_reason": blocked_reason,
        "pubmed_source_role": claim.get("pubmed_source_role"),
        "pubmed_quality_tier": claim.get("pubmed_quality_tier"),
        "allowed_for_treatment_guidance": claim.get("allowed_for_treatment_guidance") is True,
        "allowed_for_diagnosis_context": claim.get("allowed_for_diagnosis_context") is True,
    }


def _retriever_status_map(retrieval_statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for status in retrieval_statuses:
        retriever = status.get("retriever")
        if retriever:
            out[str(retriever)] = status
    return out


def build_rag_evidence_pack(
    summary: Stage3InputSummary,
    query_pack: Dict[str, Any],
    evidence_status: str,
    sources: List[Dict[str, Any]],
    allowed_claims: List[Dict[str, Any]],
    source_relevance_audit: Dict[str, Any],
    retrieval_statuses: List[Dict[str, Any]],
    max_sources_for_llm: int = 8,
    max_claims_for_llm: int = 12,
) -> Dict[str, Any]:
    """Build the controlled RAG pack for later LLM use.

    This does not call an LLM. It creates the only evidence payload that a future
    LLM writer should be allowed to consume.
    """
    sources_by_id = {str(source.get("source_id")): source for source in sources if source.get("source_id")}
    claim_cards = [_claim_for_llm(claim, sources_by_id) for claim in allowed_claims]
    allowed_for_llm = [claim for claim in claim_cards if claim.get("allowed_for_llm")]
    retriever_statuses = _retriever_status_map(retrieval_statuses)
    mode = _rag_mode(sources)
    source_cards = [_safe_source(source) for source in sources[:max_sources_for_llm]]
    source_lanes = _source_lanes(sources, source_cards)
    evidence_strength = _evidence_strength(mode, source_cards, allowed_for_llm)
    readiness_gate = _readiness_gate(source_cards, allowed_for_llm, evidence_strength)
    return {
        "version": "stage3_rag_evidence_pack_v1",
        "evidence_pack_schema_version": "stage3_rag_evidence_pack_schema_v2",
        "purpose": "Controlled source-and-claim pack for future RAG/LLM response generation.",
        "rag_available": mode != "local_curated_only",
        "rag_mode": mode,
        "llm_used": False,
        "public_output_safe": True,
        "anatomy_context": {
            "label": summary.anatomy_label,
            "parent_label": summary.anatomy_parent_label,
            "output_type": summary.anatomy_output_type,
        },
        "pipeline_context": {
            "accepted_candidate_count": summary.accepted_candidate_count,
            "stage2c_decision": summary.stage2c_decision,
            "evidence_status": evidence_status,
        },
        "retrieval_plan": {
            "query_strategy": query_pack.get("query_strategy") or {},
            "query_specs": query_pack.get("query_specs") or [],
            "queries": query_pack.get("queries") or [],
            "stage2c_context_added": query_pack.get("stage2c_context_added") is True,
            "retrievers": {
                "guideline_registry": retriever_statuses.get("guideline_registry")
                or {"retriever": "guideline_registry", "status": "not_run"},
                "pubmed": retriever_statuses.get("pubmed") or {"retriever": "pubmed", "status": "disabled"},
            },
        },
        "source_summary": {
            "ranked_source_count": len(sources),
            "source_type_counts": _source_type_counts(sources),
            "source_relevance_summary": source_relevance_audit.get("summary") or {},
            "pubmed_quality_gate": (source_relevance_audit.get("summary") or {}).get("pubmed_quality_gate") or {},
            "source_lanes": source_lanes,
            "evidence_strength": evidence_strength,
            "readiness_gate": readiness_gate,
        },
        "source_lanes": source_lanes,
        "evidence_strength": evidence_strength,
        "readiness_gate": readiness_gate,
        "sources_for_llm": source_cards,
        "claims_for_llm": allowed_for_llm[:max_claims_for_llm],
        "claim_summary": {
            "total_claim_count": len(claim_cards),
            "allowed_for_llm_count": len(allowed_for_llm),
            "blocked_claim_count": len(claim_cards) - len(allowed_for_llm),
            "treatment_oriented_claim_count": sum(
                1 for claim in allowed_for_llm if claim.get("allowed_for_treatment_guidance") is True
            ),
            "diagnosis_context_claim_count": sum(
                1 for claim in allowed_for_llm if claim.get("allowed_for_diagnosis_context") is True
            ),
        },
        "safety_policy": {
            "no_raw_pubmed_abstracts_in_public_output": True,
            "llm_may_only_use_claims_for_llm": True,
            "llm_may_not_create_new_medical_claims": True,
            "llm_may_not_use_sources_marked_abstract_exposed": True,
            "guideline_metadata_is_context_not_patient_specific_treatment": True,
            "pubmed_is_supporting_literature_not_standalone_treatment_protocol": True,
            "pubmed_treatment_guidance_requires_quality_gate_allowance": True,
            "evidence_strength_must_be_displayed_as_context_not_model_confidence": True,
            "source_lanes_must_be_preserved_for_audit": True,
        },
        "future_llm_contract": {
            "allowed_inputs": [
                "pipeline_context",
                "sources_for_llm",
                "claims_for_llm",
                "treatment_guidance",
                "care_guidance",
            ],
            "required_validator": "stage3_writer_response_validator_v1",
            "fallback_if_validation_fails": "deterministic_template",
            "requires_readiness_gate_pass": True,
            "evidence_pack_schema_version": "stage3_rag_evidence_pack_schema_v2",
        },
    }
