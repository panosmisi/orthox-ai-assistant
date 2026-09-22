from __future__ import annotations

import re
from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


MANAGEMENT_TERMS = (
    "management",
    "treatment",
    "therapy",
    "operative",
    "nonoperative",
    "non-operative",
    "surgical",
    "surgery",
    "fixation",
    "immobilization",
    "immobilisation",
    "casting",
    "splint",
    "follow-up",
    "follow up",
    "rehabilitation",
    "outcomes",
)

REVIEW_TERMS = (
    "review",
    "systematic review",
    "meta-analysis",
    "guideline",
    "recommendations",
    "consensus",
)

OUTCOME_TERMS = (
    "outcome",
    "outcomes",
    "complication",
    "complications",
    "nonunion",
    "malunion",
    "instability",
    "return",
)

FOLLOWUP_TERMS = (
    "follow-up",
    "follow up",
    "repeat",
    "delayed",
    "occult",
    "missed",
    "negative initial",
)

PEDIATRIC_TERMS = (
    "pediatric",
    "paediatric",
    "child",
    "children",
    "adolescent",
    "growth plate",
    "physis",
)

FORBIDDEN_CLAIM_PHRASES = (
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


def _blob(source: Dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in [
            source.get("title"),
            source.get("abstract"),
            " ".join(source.get("publication_types") or []),
        ]
        if part
    ).lower()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _anatomy_phrase(summary: Stage3InputSummary) -> str:
    label = summary.anatomy_label or summary.anatomy_parent_label
    return label.replace("_", " ") if label else "musculoskeletal region"


def _source_allowed(source: Dict[str, Any]) -> bool:
    if source.get("source_type") != "pubmed":
        return False
    quality = source.get("pubmed_quality") or {}
    return (
        quality.get("allowed_for_treatment_guidance") is True
        and quality.get("allowed_for_llm_context") is True
        and quality.get("quality_tier") in {"strong", "moderate"}
    )


def _safe_claim_text(text: str) -> bool:
    lowered = text.lower()
    return not any(phrase in lowered for phrase in FORBIDDEN_CLAIM_PHRASES)


def _term_phrase(source: Dict[str, Any]) -> str:
    quality = source.get("pubmed_quality") or {}
    matched = quality.get("matched_terms") or {}
    management_terms = matched.get("treatment_management") or []
    imaging_terms = matched.get("diagnosis_imaging") or []
    outcome_terms = [term for term in management_terms if term in {"outcomes", "rehabilitation", "follow-up", "follow up"}]
    parts = []
    if management_terms:
        parts.append("management concepts such as " + ", ".join(management_terms[:4]))
    if imaging_terms:
        parts.append("imaging concepts such as " + ", ".join(imaging_terms[:4]))
    if outcome_terms:
        parts.append("outcome/follow-up concepts such as " + ", ".join(outcome_terms[:3]))
    return "; ".join(parts)


def _claim(
    claim_id: str,
    text: str,
    source: Dict[str, Any],
    summary: Stage3InputSummary,
    concept_tags: List[str],
) -> Dict[str, Any] | None:
    if not _safe_claim_text(text):
        return None
    quality = source.get("pubmed_quality") or {}
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "source_id": source.get("source_id"),
        "evidence_level": source.get("evidence_level"),
        "anatomy_relevance": "medium" if summary.anatomy_output_type == "parent_fallback" else "high",
        "allowed": True,
        "claim_type": "controlled_pubmed_treatment_orientation",
        "concept_tags": concept_tags,
        "pubmed_source_role": quality.get("source_role"),
        "pubmed_quality_tier": quality.get("quality_tier"),
        "allowed_for_treatment_guidance": True,
        "allowed_for_diagnosis_context": quality.get("allowed_for_diagnosis_context") is True,
        "treatment_claim_policy": {
            "non_prescriptive": True,
            "clinician_review_only": True,
            "no_patient_specific_treatment": True,
            "no_medication_or_dosage": True,
            "no_surgery_necessity": True,
            "no_cast_or_splint_instruction": True,
            "no_discharge_or_clearance_advice": True,
        },
    }


def extract_pubmed_treatment_orientation_claims(
    summary: Stage3InputSummary,
    source: Dict[str, Any],
    claim_id_start: int = 1,
    max_claims_per_source: int = 1,
) -> List[Dict[str, Any]]:
    """Extract cautious treatment-oriented claims from approved PubMed sources.

    The emitted claims are not patient-specific treatment. They are deliberately
    generic, clinician-facing review considerations that a future constrained
    writer may cite after validation.
    """
    if not _source_allowed(source):
        return []

    text = _blob(source)
    if not _has_any(text, MANAGEMENT_TERMS):
        return []

    anatomy = _anatomy_phrase(summary)
    claims: List[Dict[str, Any]] = []

    def append(suffix: str, claim_text: str, tags: List[str]) -> None:
        claim = _claim(
            f"pubmed_treatment_{claim_id_start + len(claims):03d}_{suffix}",
            claim_text,
            source,
            summary,
            tags,
        )
        if claim:
            claims.append(claim)

    if _has_any(text, REVIEW_TERMS) and _has_any(text, MANAGEMENT_TERMS):
        term_context = _term_phrase(source)
        detail = f" The source-matching audit identified {term_context}." if term_context else ""
        append(
            "management_review_context",
            (
                f"The cited PubMed source provides treatment-management background for {anatomy}; "
                f"it may support clinician-led review of management considerations.{detail} It does not provide "
                "patient-specific treatment instructions."
            ),
            ["management_review_context", "non_prescriptive_treatment_orientation"],
        )

    if _has_any(text, OUTCOME_TERMS):
        term_context = _term_phrase(source)
        detail = f" The source-matching audit identified {term_context}." if term_context else ""
        append(
            "outcome_context",
            (
                f"The cited PubMed source discusses outcomes or complications relevant to {anatomy}; "
                f"this may support cautious clinician review of risk context without determining the patient's care plan.{detail}"
            ),
            ["outcomes_or_complications_context", "clinician_review_only"],
        )

    if _has_any(text, FOLLOWUP_TERMS):
        term_context = _term_phrase(source)
        detail = f" The source-matching audit identified {term_context}." if term_context else ""
        append(
            "followup_context",
            (
                f"The cited PubMed source includes follow-up, delayed, occult, or missed-fracture context relevant to "
                f"{anatomy}; this supports avoiding premature reassurance when clinical concern persists.{detail}"
            ),
            ["followup_or_occult_context", "avoid_premature_reassurance"],
        )

    if _has_any(text, PEDIATRIC_TERMS):
        append(
            "pediatric_management_context",
            (
                "The cited PubMed source includes pediatric or growth-plate context; this may support extra caution "
                "during clinician review when pediatric anatomy is relevant."
            ),
            ["pediatric_or_growth_plate_context", "clinician_review_only"],
        )

    if not claims:
        title = re.sub(r"\s+", " ", str(source.get("title") or "PubMed source")).strip()
        append(
            "general_management_context",
            (
                f"The cited PubMed source is relevant to treatment-management background for {anatomy}: {title}. "
                "It should be used only as general clinician-facing context."
            ),
            ["general_management_context", "non_prescriptive_treatment_orientation"],
        )

    return claims[:max_claims_per_source]
