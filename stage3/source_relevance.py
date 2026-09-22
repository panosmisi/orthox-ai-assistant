from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .input_adapter import Stage3InputSummary


FRACTURE_TERMS = {"fracture", "fractures", "trauma", "injury", "injuries"}
IMAGING_TERMS = {
    "radiograph",
    "radiographs",
    "radiographic",
    "radiography",
    "x-ray",
    "xray",
    "plain film",
    "mri",
    "ct",
    "computed tomography",
    "ultrasound",
    "sonography",
}
DIAGNOSTIC_TERMS = {"diagnosis", "diagnostic", "accuracy", "sensitivity", "specificity", "occult", "missed"}
EXCLUSION_TERMS = {
    "veterinary",
    "dog",
    "dogs",
    "cat",
    "cats",
    "mandibular",
    "dental",
    "tooth",
    "teeth",
    "facial",
    "skull",
    "vertebral",
}

ANATOMY_SYNONYMS = {
    "wrist_hand": {"wrist", "hand", "scaphoid", "metacarpal", "phalangeal", "finger", "distal radius"},
    "hand": {"hand", "metacarpal", "phalangeal", "finger", "thumb"},
    "elbow": {"elbow", "radial head", "supracondylar", "olecranon", "humerus"},
    "forearm": {"forearm", "radius", "ulna", "monteggia", "galeazzi"},
    "shoulder_upper_arm": {"shoulder", "humerus", "proximal humerus", "clavicle", "upper arm"},
    "shoulder": {"shoulder", "proximal humerus", "clavicle"},
    "ankle_foot": {"ankle", "foot", "malleolar", "malleolus", "metatarsal", "navicular", "calcaneus"},
    "knee_lower_leg": {"knee", "tibia", "fibula", "patella", "tibial plateau", "lower leg"},
    "leg": {"leg", "tibia", "fibula", "knee", "ankle", "lower leg"},
    "pelvis_hip_femur": {"pelvis", "pelvic", "hip", "femur", "femoral", "femoral neck"},
    "hip": {"hip", "femur", "femoral", "femoral neck", "pelvis"},
}


def _blob(source: Dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in [
            source.get("title"),
            source.get("abstract"),
            source.get("journal"),
            " ".join(source.get("publication_types") or []),
            source.get("organization"),
            " ".join(source.get("topic_scope") or []),
            " ".join(source.get("anatomy_scope") or []),
        ]
        if part
    ).lower()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())


def _contains_any(text: str, terms: set[str]) -> Tuple[bool, List[str]]:
    hits = []
    for term in terms:
        if " " in term or "-" in term:
            if term in text:
                hits.append(term)
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            hits.append(term)
    hits = sorted(hits)
    return bool(hits), hits


def _query_hits(text: str, queries: List[str]) -> List[str]:
    hits = []
    for query in queries:
        q_tokens = [t for t in _tokens(query) if len(t) >= 5]
        if not q_tokens:
            continue
        matched = sum(1 for token in q_tokens if token in text)
        if matched >= min(2, len(q_tokens)):
            hits.append(query)
    return hits


def _anatomy_hits(summary: Stage3InputSummary, text: str) -> List[str]:
    labels = [summary.anatomy_label, summary.anatomy_parent_label]
    terms = set()
    for label in labels:
        if not label:
            continue
        terms.update(ANATOMY_SYNONYMS.get(label, set()))
        terms.update(t for t in label.replace("_", " ").split() if len(t) >= 3)
    return sorted(term for term in terms if term in text)


def evaluate_source_relevance(
    source: Dict[str, Any],
    summary: Stage3InputSummary,
    queries: List[str],
    keep_low_relevance: bool = False,
) -> Dict[str, Any]:
    source_type = source.get("source_type")
    source_id = source.get("source_id")
    audit = {
        "source_id": source_id,
        "source_type": source_type,
        "kept": True,
        "relevance_label": "high",
        "relevance_score": 0,
        "matched_anatomy_terms": [],
        "matched_query_terms": [],
        "matched_concepts": [],
        "exclusion_reasons": [],
        "keep_reasons": [],
    }

    if source_type == "local_curated_project_rule":
        audit["relevance_score"] = 100
        audit["keep_reasons"].append("local_project_safety_context")
        return audit

    if source_type == "guideline_metadata":
        audit["relevance_score"] = 85
        audit["keep_reasons"].append("curated_guideline_metadata")
        if source.get("match_reasons"):
            audit["keep_reasons"].extend(source.get("match_reasons") or [])
        return audit

    text = _blob(source)
    has_fracture, fracture_hits = _contains_any(text, FRACTURE_TERMS)
    has_imaging, imaging_hits = _contains_any(text, IMAGING_TERMS)
    has_diagnostic, diagnostic_hits = _contains_any(text, DIAGNOSTIC_TERMS)
    _, exclusion_hits = _contains_any(text, EXCLUSION_TERMS)
    anatomy_hits = _anatomy_hits(summary, text)
    query_hits = _query_hits(text, queries)

    audit["matched_anatomy_terms"] = anatomy_hits
    audit["matched_query_terms"] = query_hits
    audit["matched_concepts"] = sorted(set(fracture_hits + imaging_hits + diagnostic_hits))

    score = 0
    if has_fracture:
        score += 30
        audit["keep_reasons"].append("fracture_or_injury_context")
    else:
        audit["exclusion_reasons"].append("missing_fracture_or_injury_context")

    if has_imaging:
        score += 25
        audit["keep_reasons"].append("imaging_context")
    else:
        audit["exclusion_reasons"].append("missing_imaging_context")

    if anatomy_hits:
        score += 25
        audit["keep_reasons"].append("anatomy_overlap")

    if query_hits:
        score += min(20, 8 + 4 * len(query_hits))
        audit["keep_reasons"].append("query_overlap")

    if has_diagnostic:
        score += 10
        audit["keep_reasons"].append("diagnostic_context")

    if exclusion_hits:
        score -= 35
        audit["exclusion_reasons"].append("outside_scope_terms:" + ",".join(exclusion_hits[:5]))

    evidence_level = source.get("evidence_level")
    if evidence_level in {"systematic_review_or_meta_analysis", "guideline_or_standard"}:
        score += 10
        audit["keep_reasons"].append("higher_evidence_level")
    elif evidence_level == "case_report_or_small_series":
        score -= 8
        audit["exclusion_reasons"].append("low_evidence_level")

    audit["relevance_score"] = score
    if score >= 75:
        audit["relevance_label"] = "high"
    elif score >= 55:
        audit["relevance_label"] = "medium"
    elif score >= 40:
        audit["relevance_label"] = "low"
    else:
        audit["relevance_label"] = "excluded"

    if not has_fracture or not has_imaging:
        audit["kept"] = False
    elif audit["relevance_label"] == "excluded":
        audit["kept"] = False
    elif audit["relevance_label"] == "low" and not keep_low_relevance:
        audit["kept"] = False
        audit["exclusion_reasons"].append("low_relevance_below_policy_threshold")
    else:
        audit["kept"] = True

    return audit


def apply_source_relevance_filter(
    sources: List[Dict[str, Any]],
    summary: Stage3InputSummary,
    queries: List[str],
    keep_low_relevance: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept = []
    audits = []
    for source in sources:
        audit = evaluate_source_relevance(source, summary, queries, keep_low_relevance=keep_low_relevance)
        audits.append(audit)
        if audit["kept"]:
            enriched = dict(source)
            enriched["source_relevance"] = {
                "label": audit["relevance_label"],
                "score": audit["relevance_score"],
                "keep_reasons": audit["keep_reasons"],
            }
            kept.append(enriched)
    return kept, audits
