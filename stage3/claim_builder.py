from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary
from .pubmed_claim_extractor import extract_pubmed_abstract_claims
from .pubmed_treatment_claim_extractor import extract_pubmed_treatment_orientation_claims
from .template_writer import build_allowed_claims


CLAIM_LOCK_VERSION = "stage3_source_claim_extraction_lock_v1"

RAW_AUDIT_PHRASES = (
    "source-matching audit",
    "matched terms",
    "identified management concepts",
    "identified imaging concepts",
)

BLOCKED_CLAIM_PHRASES = (
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
)

SAFE_CONTAINING_PHRASES = (
    "no fracture candidate",
    "no high-confidence fracture candidate",
    "no retained fracture candidate",
    "did not retain a fracture",
    "does not rule out fracture",
)


def _claim(claim_id: str, text: str, source: Dict[str, Any], anatomy_relevance: str) -> Dict[str, Any]:
    claim = {
        "claim_id": claim_id,
        "claim_text": text,
        "source_id": source.get("source_id"),
        "evidence_level": source.get("evidence_level"),
        "anatomy_relevance": anatomy_relevance,
        "allowed": True,
    }
    if source.get("pubmed_quality"):
        quality = source.get("pubmed_quality") or {}
        claim["pubmed_source_role"] = quality.get("source_role")
        claim["pubmed_quality_tier"] = quality.get("quality_tier")
        claim["allowed_for_treatment_guidance"] = quality.get("allowed_for_treatment_guidance") is True
        claim["allowed_for_diagnosis_context"] = quality.get("allowed_for_diagnosis_context") is True
    return claim


def _source_kind(source: Dict[str, Any] | None) -> str:
    if not source:
        return "missing_source"
    source_type = source.get("source_type")
    if source_type == "pubmed":
        return "pubmed"
    if source_type == "guideline_metadata":
        return "curated_guideline"
    if source_type == "local_curated_project_rule":
        return "local_safety_rule"
    return "pipeline_or_project_context"


def _claim_role(claim: Dict[str, Any], source: Dict[str, Any] | None) -> str:
    claim_type = str(claim.get("claim_type") or "")
    claim_id = str(claim.get("claim_id") or "")
    if claim_type == "controlled_pubmed_treatment_orientation":
        return "pubmed_treatment_management_context"
    if claim_type == "controlled_pubmed_abstract_context":
        return "pubmed_imaging_or_occult_context"
    if claim_id.startswith("guideline_") or _source_kind(source) == "curated_guideline":
        return "curated_guideline_context"
    if claim_id.startswith("stage2c_"):
        return "pipeline_safety_context"
    if claim_id.startswith("project_safety_"):
        return "local_project_safety_context"
    return "controlled_context"


def _sanitize_claim_text(text: str) -> str:
    clean = " ".join(str(text or "").split())
    # Keep public claim text clean: raw matching/audit details belong in metadata, not in LLM-facing claims.
    sentences = [part.strip() for part in clean.split(".") if part.strip()]
    kept = [
        sentence
        for sentence in sentences
        if not any(phrase in sentence.lower() for phrase in RAW_AUDIT_PHRASES)
    ]
    if kept:
        clean = ". ".join(kept).strip()
        if not clean.endswith("."):
            clean += "."
    for orphan in [" The.", " This.", " It."]:
        clean = clean.replace(orphan, ".")
    return clean


def _claim_supports(claim: Dict[str, Any], role: str, source: Dict[str, Any] | None) -> List[str]:
    if role == "pubmed_treatment_management_context":
        return [
            "clinician-facing treatment-management background",
            "non-prescriptive management context only",
            "must be cited through approved PubMed claim/source ids",
        ]
    if role == "pubmed_imaging_or_occult_context":
        return [
            "clinician-facing imaging, occult, or follow-up context",
            "caution against premature reassurance when clinical concern persists",
            "must be cited through approved PubMed claim/source ids",
        ]
    if role == "curated_guideline_context":
        return [
            "curated guideline/standard context for clinician review",
            "general fracture assessment or management background",
        ]
    if role == "pipeline_safety_context":
        return [
            "pipeline state explanation",
            "safe wording when no localized bbox is available",
        ]
    return ["controlled context for safe report wording"]


def _claim_limitations(claim: Dict[str, Any], role: str) -> List[str]:
    limitations = [
        "does not diagnose or exclude fracture",
        "does not provide patient-specific treatment instructions",
        "requires symptoms, examination, full imaging review, and local protocol",
    ]
    if role.startswith("pubmed_"):
        limitations.append("derived from source metadata/approved claim extraction, not raw public abstract display")
    if role == "pubmed_treatment_management_context":
        limitations.append("does not decide immobilisation, weight-bearing, medication, procedure, follow-up, clearance/reassurance, or surgery")
    return limitations


def _lock_claim(
    claim: Dict[str, Any],
    sources_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    locked = dict(claim)
    source = sources_by_id.get(str(claim.get("source_id")))
    role = _claim_role(claim, source)
    sanitized_text = _sanitize_claim_text(str(claim.get("claim_text") or ""))
    lowered_text = sanitized_text.lower()
    blocked_hits = []
    for phrase in BLOCKED_CLAIM_PHRASES:
        phrase_l = phrase.lower()
        if phrase_l not in lowered_text:
            continue
        if phrase_l == "no fracture" and any(safe in lowered_text for safe in SAFE_CONTAINING_PHRASES):
            continue
        blocked_hits.append(phrase)
    raw_audit_hits = [
        phrase for phrase in RAW_AUDIT_PHRASES if phrase in sanitized_text.lower()
    ]
    quality = (source or {}).get("pubmed_quality") or {}
    allowed = claim.get("allowed") is True and source is not None and bool(sanitized_text) and not blocked_hits and not raw_audit_hits
    if source and source.get("source_type") == "pubmed":
        allowed = (
            allowed
            and quality.get("allowed_for_llm_context") is True
            and quality.get("quality_tier") in {"strong", "moderate", "background"}
        )
        if role == "pubmed_treatment_management_context":
            allowed = allowed and quality.get("allowed_for_treatment_guidance") is True
    locked["claim_text"] = sanitized_text
    locked["allowed"] = allowed
    locked["claim_lock"] = {
        "version": CLAIM_LOCK_VERSION,
        "status": "PASS" if allowed else "FAIL",
        "source_kind": _source_kind(source),
        "claim_role": role,
        "llm_visible_text_is_sanitized": True,
        "raw_source_text_exposed": False,
        "raw_audit_text_exposed": False,
        "blocked_phrase_hits": blocked_hits,
        "raw_audit_phrase_hits": raw_audit_hits,
        "source_exists": source is not None,
        "requires_source_id": True,
        "requires_claim_id": True,
        "allowed_for_llm_context": allowed,
    }
    locked["claim_role"] = role
    locked["supports"] = _claim_supports(claim, role, source)
    locked["limitations"] = _claim_limitations(claim, role)
    locked["allowed_for_llm_context"] = allowed
    locked["llm_visibility"] = {
        "visible_to_llm": allowed,
        "visible_text_field": "claim_text",
        "raw_pubmed_abstract_visible": False,
        "raw_matching_audit_visible": False,
    }
    return locked


def _lock_claims(claims: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources_by_id = {str(source.get("source_id")): source for source in sources if source.get("source_id")}
    locked = [_lock_claim(claim, sources_by_id) for claim in claims if claim.get("source_id")]
    return [claim for claim in locked if claim.get("source_id")]


def build_evidence_claims(
    summary: Stage3InputSummary,
    sources: List[Dict[str, Any]],
    pubmed_abstract_claims_enabled: bool = False,
    pubmed_treatment_claims_enabled: bool = False,
    max_pubmed_abstract_claims: int = 6,
    max_pubmed_treatment_claims: int = 4,
) -> List[Dict[str, Any]]:
    claims = build_allowed_claims(summary)
    anatomy = summary.anatomy_label or summary.anatomy_parent_label or "musculoskeletal"

    guideline_sources = [s for s in sources if s.get("source_type") == "guideline_metadata"]
    pubmed_sources = [s for s in sources if s.get("source_type") == "pubmed"]

    idx = 1
    for source in guideline_sources[:3]:
        claims.append(
            _claim(
                f"guideline_context_{idx:03d}",
                f"A curated guideline or orthopaedic standard source is available for general fracture assessment context relevant to {anatomy}.",
                source,
                "medium" if summary.anatomy_output_type == "parent_fallback" else "high",
            )
        )
        idx += 1

    abstract_claim_count = 0
    treatment_claim_count = 0
    for source in pubmed_sources[:5]:
        quality = source.get("pubmed_quality") or {}
        if quality and quality.get("allowed_for_llm_context") is not True:
            continue
        if (
            pubmed_treatment_claims_enabled
            and treatment_claim_count < max_pubmed_treatment_claims
            and quality.get("allowed_for_treatment_guidance") is True
        ):
            extracted_treatment = extract_pubmed_treatment_orientation_claims(
                summary,
                source,
                claim_id_start=idx,
                max_claims_per_source=min(1, max_pubmed_treatment_claims - treatment_claim_count),
            )
            claims.extend(extracted_treatment)
            idx += len(extracted_treatment)
            treatment_claim_count += len(extracted_treatment)
            if extracted_treatment:
                continue
        if pubmed_abstract_claims_enabled and abstract_claim_count < max_pubmed_abstract_claims:
            extracted = extract_pubmed_abstract_claims(
                summary,
                source,
                claim_id_start=idx,
                max_claims_per_source=min(2, max_pubmed_abstract_claims - abstract_claim_count),
            )
            for claim in extracted:
                claim["pubmed_source_role"] = quality.get("source_role")
                claim["pubmed_quality_tier"] = quality.get("quality_tier")
                claim["allowed_for_treatment_guidance"] = quality.get("allowed_for_treatment_guidance") is True
                claim["allowed_for_diagnosis_context"] = quality.get("allowed_for_diagnosis_context") is True
            claims.extend(extracted)
            idx += len(extracted)
            abstract_claim_count += len(extracted)
            if extracted:
                continue

        title = source.get("title") or "PubMed source"
        claims.append(
            _claim(
                f"pubmed_context_{idx:03d}",
                f"A PubMed-indexed source relevant to the query set was retrieved: {title}",
                source,
                "medium" if summary.anatomy_output_type == "parent_fallback" else "high",
            )
        )
        idx += 1

    if summary.stage2c_decision in {"suspicious", "uncertain"} and summary.accepted_candidate_count == 0:
        # This is a cautious pipeline-context claim, not a literature claim.
        source = {"source_id": "stage3_curated_safety_context_v0", "evidence_level": "pipeline_observation"}
        claims.append(
            _claim(
                "stage2c_occult_context_001",
                "When the detector/verifier retains no box but Stage 2C raises a warning, the output should be framed as image-level concern without localization.",
                source,
                "case_specific_pipeline_output",
            )
        )

    # Keep source-linked claims, but require the final claim-lock metadata for every downstream consumer.
    return _lock_claims(claims, sources)
