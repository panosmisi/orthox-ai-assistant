from __future__ import annotations

import re
from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


FRACTURE_TERMS = ("fracture", "fractures")
RADIOGRAPH_TERMS = ("radiograph", "radiographs", "radiographic", "x-ray", "xray", "plain film")
OCCULT_TERMS = ("occult", "missed", "inconclusive", "negative initial", "initial radiographs")
ADVANCED_IMAGING_TERMS = ("mri", "magnetic resonance", "ct", "computed tomography", "ultrasound", "sonography")
DIAGNOSTIC_PERFORMANCE_TERMS = (
    "diagnostic accuracy",
    "sensitivity",
    "specificity",
    "positive predictive",
    "negative predictive",
)
PEDIATRIC_TERMS = ("pediatric", "paediatric", "child", "children", "adolescent", "growth plate", "physis")


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


def _source_is_relevant(source: Dict[str, Any]) -> bool:
    text = _blob(source)
    if not source.get("abstract"):
        return False
    return _has_any(text, FRACTURE_TERMS) and (
        _has_any(text, RADIOGRAPH_TERMS) or _has_any(text, ADVANCED_IMAGING_TERMS)
    )


def _anatomy_phrase(summary: Stage3InputSummary) -> str:
    label = summary.anatomy_label or summary.anatomy_parent_label
    if not label:
        return "musculoskeletal"
    return label.replace("_", " ")


def extract_pubmed_abstract_claims(
    summary: Stage3InputSummary,
    source: Dict[str, Any],
    claim_id_start: int = 1,
    max_claims_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """Build tightly controlled PubMed abstract claims.

    The extractor intentionally avoids treatment, diagnosis, and patient-specific
    advice. It only emits generic evidence-context statements when the abstract
    contains a small set of recognized concepts.
    """
    if source.get("source_type") != "pubmed" or not _source_is_relevant(source):
        return []

    text = _blob(source)
    anatomy = _anatomy_phrase(summary)
    claims: List[Dict[str, Any]] = []

    def append_claim(suffix: str, claim_text: str, concept_tags: List[str]) -> None:
        claims.append(
            {
                "claim_id": f"pubmed_abstract_{claim_id_start + len(claims):03d}_{suffix}",
                "claim_text": claim_text,
                "source_id": source.get("source_id"),
                "evidence_level": source.get("evidence_level"),
                "anatomy_relevance": "medium" if summary.anatomy_output_type == "parent_fallback" else "high",
                "allowed": True,
                "claim_type": "controlled_pubmed_abstract_context",
                "concept_tags": concept_tags,
            }
        )

    if _has_any(text, OCCULT_TERMS) and (
        _has_any(text, RADIOGRAPH_TERMS) or _has_any(text, ADVANCED_IMAGING_TERMS)
    ):
        append_claim(
            "occult_context",
            (
                f"The cited PubMed abstract discusses occult or initially difficult-to-detect fracture assessment "
                f"in imaging contexts relevant to {anatomy}; this supports cautious human review when model output is "
                "negative, uncertain, or discordant with clinical concern."
            ),
            ["occult_or_missed_fracture", "radiograph_limitations"],
        )

    if _has_any(text, DIAGNOSTIC_PERFORMANCE_TERMS):
        append_claim(
            "diagnostic_performance_context",
            (
                f"The cited PubMed abstract evaluates diagnostic imaging performance for suspected fracture; it should "
                f"be treated as background evidence for {anatomy} review, not as a case-specific conclusion."
            ),
            ["diagnostic_imaging_performance"],
        )

    if _has_any(text, PEDIATRIC_TERMS):
        append_claim(
            "pediatric_context",
            (
                "The cited PubMed abstract includes pediatric or growth-plate context, which can make radiographic "
                "fracture interpretation more difficult and reinforces the need for clinician verification."
            ),
            ["pediatric_or_growth_plate_context"],
        )

    if not claims and re.search(r"\b(fracture|fractures)\b", text):
        append_claim(
            "general_radiographic_context",
            (
                f"The cited PubMed abstract is relevant to radiographic or diagnostic imaging assessment of fracture "
                f"around {anatomy}; it is included as background evidence only."
            ),
            ["general_fracture_imaging_context"],
        )

    return claims[:max_claims_per_source]
