from __future__ import annotations

from typing import Any, Dict, List


def _count_by(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _audit_lookup(source_relevance_audit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("source_id")): item
        for item in source_relevance_audit.get("items", []) or []
        if isinstance(item, dict) and item.get("source_id")
    }


def _source_card(source: Dict[str, Any], audit_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    audit = audit_by_id.get(str(source.get("source_id")), {})
    return {
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "title": source.get("title"),
        "organization": source.get("organization"),
        "url": source.get("url"),
        "evidence_level": source.get("evidence_level"),
        "relevance_label": audit.get("relevance_label"),
        "relevance_score": audit.get("relevance_score"),
        "kept": audit.get("kept", True),
        "keep_reasons": audit.get("keep_reasons", []),
        "exclusion_reasons": audit.get("exclusion_reasons", []),
        "abstract_exposed": "abstract" in source,
        "pubmed_quality": source.get("pubmed_quality") if source.get("source_type") == "pubmed" else None,
    }


def _claim_card(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("claim_text"),
        "source_id": claim.get("source_id"),
        "evidence_level": claim.get("evidence_level"),
        "anatomy_relevance": claim.get("anatomy_relevance"),
        "allowed": claim.get("allowed") is True,
        "claim_role": claim.get("claim_role"),
        "claim_lock": claim.get("claim_lock") or {},
        "supports": claim.get("supports") or [],
        "limitations": claim.get("limitations") or [],
        "allowed_for_llm_context": claim.get("allowed_for_llm_context") is True,
        "pubmed_source_role": claim.get("pubmed_source_role"),
        "pubmed_quality_tier": claim.get("pubmed_quality_tier"),
        "allowed_for_treatment_guidance": claim.get("allowed_for_treatment_guidance") is True,
        "allowed_for_diagnosis_context": claim.get("allowed_for_diagnosis_context") is True,
    }


def build_evidence_panel(
    evidence_status: str,
    sources: List[Dict[str, Any]],
    allowed_claims: List[Dict[str, Any]],
    source_relevance_audit: Dict[str, Any],
    retrieval_statuses: List[Dict[str, Any]],
    max_sources: int = 8,
    max_claims: int = 10,
) -> Dict[str, Any]:
    """Build a compact transparency panel for app/report display.

    This panel does not create new medical claims. It summarizes already selected
    sources, already allowed claims, and the source relevance audit.
    """
    audit_by_id = _audit_lookup(source_relevance_audit)
    source_cards = [_source_card(source, audit_by_id) for source in sources[:max_sources]]
    claim_cards = [_claim_card(claim) for claim in allowed_claims[:max_claims]]
    source_ids = {source.get("source_id") for source in sources if source.get("source_id")}
    invalid_claims = [
        claim.get("claim_id")
        for claim in claim_cards
        if not claim.get("allowed") or not claim.get("source_id") or claim.get("source_id") not in source_ids
    ]
    abstracts_exposed = [card.get("source_id") for card in source_cards if card.get("abstract_exposed")]

    return {
        "version": "stage3_evidence_panel_v1",
        "purpose": "Evidence and claim transparency for deterministic Stage 3 support.",
        "llm_used": False,
        "creates_new_medical_claims": False,
        "evidence_status": evidence_status,
        "retrieval_statuses": retrieval_statuses,
        "source_summary": {
            "displayed_source_count": len(source_cards),
            "total_ranked_source_count": len(sources),
            "source_type_counts": _count_by(sources, "source_type"),
            "relevance_summary": (source_relevance_audit.get("summary") or {}),
            "pubmed_quality_gate": (source_relevance_audit.get("summary") or {}).get("pubmed_quality_gate") or {},
        },
        "claim_summary": {
            "displayed_claim_count": len(claim_cards),
            "total_allowed_claim_count": len(allowed_claims),
            "claim_evidence_level_counts": _count_by(allowed_claims, "evidence_level"),
            "invalid_displayed_claim_ids": invalid_claims,
            "claim_lock_status_counts": _count_by(
                [claim.get("claim_lock") or {} for claim in allowed_claims],
                "status",
            ),
        },
        "displayed_sources": source_cards,
        "displayed_allowed_claims": claim_cards,
        "safety_checks": {
            "all_displayed_claims_allowed": not invalid_claims,
            "all_displayed_claims_have_sources": not invalid_claims,
            "pubmed_abstracts_not_exposed": not abstracts_exposed,
            "abstract_exposed_source_ids": abstracts_exposed,
            "panel_is_transparency_only": True,
            "all_displayed_claims_have_claim_lock": all(
                bool(claim.get("claim_lock")) for claim in claim_cards
            ),
            "all_displayed_claim_locks_pass": all(
                (claim.get("claim_lock") or {}).get("status") == "PASS"
                for claim in claim_cards
            ),
        },
        "display_rules": [
            "Show sources and allowed claims as provenance, not as patient-specific diagnosis.",
            "Do not expose PubMed abstracts in public output.",
            "Do not treat guideline metadata as direct treatment advice for the specific patient.",
            "Keep source relevance audit visible in developer/audit views.",
        ],
    }
