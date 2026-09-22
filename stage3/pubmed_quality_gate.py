from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple


HIGH_VALUE_EVIDENCE_LEVELS = {
    "guideline_or_standard",
    "systematic_review_or_meta_analysis",
    "review_article",
    "diagnostic_accuracy_or_cohort",
}

LOW_VALUE_EVIDENCE_LEVELS = {"case_report_or_small_series"}

TREATMENT_TERMS = {
    "management",
    "treatment",
    "therapy",
    "therapeutic",
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
}

DIAGNOSIS_IMAGING_TERMS = {
    "diagnosis",
    "diagnostic",
    "radiograph",
    "radiographs",
    "radiographic",
    "radiography",
    "x-ray",
    "xray",
    "plain film",
    "ct",
    "computed tomography",
    "mri",
    "magnetic resonance",
    "ultrasound",
    "sonography",
    "sensitivity",
    "specificity",
    "accuracy",
    "occult",
    "missed",
}

AI_METHOD_TERMS = {
    "artificial intelligence",
    "deep learning",
    "machine learning",
    "neural network",
    "convolutional",
    "algorithm",
    "computer-aided",
    "computer aided",
}

REGISTRY_TERMS = {
    "registry",
    "database",
    "epidemiology",
    "incidence",
    "prevalence",
    "national patient register",
}

OUT_OF_SCOPE_TERMS = {
    "veterinary",
    "dog",
    "dogs",
    "cat",
    "cats",
    "mandibular",
    "dental",
    "tooth",
    "teeth",
    "vertebral",
    "skull",
}


def _blob(source: Dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in [
            source.get("title"),
            source.get("abstract"),
            source.get("journal"),
            " ".join(source.get("publication_types") or []),
            source.get("query"),
        ]
        if part
    ).lower()


def _hits(text: str, terms: set[str]) -> List[str]:
    hits = []
    for term in terms:
        if " " in term or "-" in term:
            if term in text:
                hits.append(term)
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            hits.append(term)
    return sorted(hits)


def evaluate_pubmed_source_quality(source: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a PubMed source by safe downstream use.

    This gate is intentionally conservative. It does not decide whether a paper
    is clinically true. It decides what role the paper may play in this system:
    diagnosis/imaging context, treatment-management background, AI-methods
    background, or no LLM context.
    """
    if source.get("source_type") != "pubmed":
        return {
            "applies": False,
            "quality_tier": "not_pubmed",
            "source_role": "not_pubmed",
            "allowed_for_llm_context": True,
            "allowed_for_treatment_guidance": False,
            "allowed_for_diagnosis_context": False,
            "exclusion_reasons": [],
            "matched_terms": {},
        }

    text = _blob(source)
    evidence_level = source.get("evidence_level") or "article"
    treatment_hits = _hits(text, TREATMENT_TERMS)
    imaging_hits = _hits(text, DIAGNOSIS_IMAGING_TERMS)
    ai_hits = _hits(text, AI_METHOD_TERMS)
    registry_hits = _hits(text, REGISTRY_TERMS)
    out_of_scope_hits = _hits(text, OUT_OF_SCOPE_TERMS)

    source_role = "background_fracture_context"
    if registry_hits:
        source_role = "registry_or_epidemiology_context"
    if imaging_hits:
        source_role = "diagnosis_imaging_context"
    if treatment_hits:
        source_role = "treatment_management_context"
    if ai_hits:
        source_role = "ai_or_algorithm_methodology"

    exclusion_reasons: List[str] = []
    if out_of_scope_hits:
        exclusion_reasons.append("out_of_scope_terms:" + ",".join(out_of_scope_hits[:5]))
    if evidence_level in LOW_VALUE_EVIDENCE_LEVELS:
        exclusion_reasons.append("low_directness_case_report_or_small_series")

    quality_score = 0
    if evidence_level == "guideline_or_standard":
        quality_score += 45
    elif evidence_level == "systematic_review_or_meta_analysis":
        quality_score += 40
    elif evidence_level == "review_article":
        quality_score += 28
    elif evidence_level == "diagnostic_accuracy_or_cohort":
        quality_score += 25
    elif evidence_level == "article":
        quality_score += 12
    elif evidence_level in LOW_VALUE_EVIDENCE_LEVELS:
        quality_score += 4

    if treatment_hits:
        quality_score += 20
    if imaging_hits:
        quality_score += 18
    if registry_hits:
        quality_score += 8
    if ai_hits:
        quality_score -= 8
    if out_of_scope_hits:
        quality_score -= 40

    if out_of_scope_hits:
        quality_tier = "excluded"
    elif quality_score >= 65:
        quality_tier = "strong"
    elif quality_score >= 45:
        quality_tier = "moderate"
    elif quality_score >= 25:
        quality_tier = "background"
    else:
        quality_tier = "weak"

    allowed_for_llm_context = quality_tier in {"strong", "moderate", "background"}
    allowed_for_diagnosis_context = bool(imaging_hits) and allowed_for_llm_context
    allowed_for_treatment_guidance = (
        bool(treatment_hits)
        and evidence_level in HIGH_VALUE_EVIDENCE_LEVELS
        and quality_tier in {"strong", "moderate"}
        and not ai_hits
        and not out_of_scope_hits
    )

    if source_role == "ai_or_algorithm_methodology":
        allowed_for_treatment_guidance = False
    if quality_tier == "weak":
        allowed_for_llm_context = False
        exclusion_reasons.append("weak_pubmed_quality_tier")

    if source_role == "background_fracture_context":
        exclusion_reasons.append("no_specific_imaging_or_management_role")

    return {
        "applies": True,
        "version": "pubmed_source_quality_gate_v1",
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "source_role": source_role,
        "allowed_for_llm_context": allowed_for_llm_context,
        "allowed_for_treatment_guidance": allowed_for_treatment_guidance,
        "allowed_for_diagnosis_context": allowed_for_diagnosis_context,
        "evidence_level": evidence_level,
        "exclusion_reasons": sorted(set(exclusion_reasons)),
        "matched_terms": {
            "treatment_management": treatment_hits[:8],
            "diagnosis_imaging": imaging_hits[:8],
            "ai_methodology": ai_hits[:8],
            "registry_epidemiology": registry_hits[:8],
            "out_of_scope": out_of_scope_hits[:8],
        },
    }


def apply_pubmed_quality_gate(sources: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    role_counts = Counter()
    tier_counts = Counter()
    allowed_llm = 0
    allowed_treatment = 0
    allowed_diagnosis = 0

    for source in sources:
        if source.get("source_type") != "pubmed":
            enriched.append(source)
            continue
        quality = evaluate_pubmed_source_quality(source)
        updated = dict(source)
        updated["pubmed_quality"] = quality
        enriched.append(updated)
        role_counts[quality.get("source_role")] += 1
        tier_counts[quality.get("quality_tier")] += 1
        if quality.get("allowed_for_llm_context"):
            allowed_llm += 1
        if quality.get("allowed_for_treatment_guidance"):
            allowed_treatment += 1
        if quality.get("allowed_for_diagnosis_context"):
            allowed_diagnosis += 1

    return enriched, {
        "version": "pubmed_source_quality_gate_v1",
        "pubmed_source_count": sum(1 for source in sources if source.get("source_type") == "pubmed"),
        "quality_tier_counts": dict(tier_counts),
        "source_role_counts": dict(role_counts),
        "allowed_for_llm_context_count": allowed_llm,
        "allowed_for_treatment_guidance_count": allowed_treatment,
        "allowed_for_diagnosis_context_count": allowed_diagnosis,
    }
