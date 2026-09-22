from __future__ import annotations

from typing import Any, Dict, List


EVIDENCE_LEVEL_SCORE = {
    "guideline_or_standard": 100,
    "systematic_review_or_meta_analysis": 90,
    "review_article": 70,
    "diagnostic_accuracy_or_cohort": 60,
    "article": 40,
    "case_report_or_small_series": 20,
    "project_curated_context": 10,
    "project_safety_statement": 10,
    "pipeline_observation": 10,
}


def _text_blob(source: Dict[str, Any]) -> str:
    parts = [
        source.get("title"),
        source.get("abstract"),
        source.get("organization"),
        " ".join(source.get("topic_scope") or []),
        " ".join(source.get("anatomy_scope") or []),
        source.get("query"),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def rank_sources(
    sources: List[Dict[str, Any]],
    anatomy_label: str | None,
    parent_label: str | None,
    query_terms: List[str],
    max_sources: int = 8,
) -> List[Dict[str, Any]]:
    ranked = []
    anatomy_terms = {t for t in [anatomy_label, parent_label] if t}
    for source in sources:
        blob = _text_blob(source)
        score = EVIDENCE_LEVEL_SCORE.get(source.get("evidence_level"), 30)
        for term in anatomy_terms:
            for token in term.replace("_", " ").split():
                if len(token) >= 3 and token.lower() in blob:
                    score += 12
        for query in query_terms:
            query_score = 0
            for token in query.lower().replace("-", " ").split():
                if len(token) >= 5 and token in blob:
                    query_score += 2
            score += min(query_score, 12)
        if "fracture" in blob:
            score += 10
        if "radiograph" in blob or "x-ray" in blob or "xray" in blob:
            score += 8
        pubmed_quality = source.get("pubmed_quality") or {}
        if pubmed_quality:
            tier = pubmed_quality.get("quality_tier")
            role = pubmed_quality.get("source_role")
            if tier == "strong":
                score += 14
            elif tier == "moderate":
                score += 8
            elif tier == "weak":
                score -= 15
            elif tier == "excluded":
                score -= 40
            if role == "treatment_management_context":
                score += 8
            elif role == "diagnosis_imaging_context":
                score += 5
            elif role == "ai_or_algorithm_methodology":
                score -= 6
        enriched = dict(source)
        enriched["rank_score"] = score
        ranked.append(enriched)
    ranked.sort(key=lambda s: s.get("rank_score", 0), reverse=True)
    return ranked[:max_sources]
