from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_REGISTRY = Path(__file__).with_name("guideline_registry.yaml")


def load_guideline_registry(path: str | Path = DEFAULT_REGISTRY) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("guidelines") or [])


def _matches_anatomy(entry: Dict[str, Any], anatomy_label: str | None, parent_label: str | None) -> bool:
    scopes = set(entry.get("anatomy_scope") or [])
    if "unknown" in scopes:
        return True
    if anatomy_label and anatomy_label in scopes:
        return True
    # Avoid overly broad parent matches for narrow anatomy-specific guidelines.
    if parent_label and parent_label in scopes and len(scopes) >= 4:
        return True
    return False


def _topic_score(entry: Dict[str, Any], query_terms: List[str]) -> int:
    topics = " ".join(entry.get("topic_scope") or []).lower()
    title = str(entry.get("title") or "").lower()
    score = 0
    for query in query_terms:
        for token in query.lower().replace("-", " ").split():
            if len(token) < 4:
                continue
            if token in topics or token in title:
                score += 1
    return score


def retrieve_guidelines(
    anatomy_label: str | None,
    parent_label: str | None,
    query_terms: List[str],
    registry_path: str | Path = DEFAULT_REGISTRY,
    max_results: int = 4,
) -> List[Dict[str, Any]]:
    entries = load_guideline_registry(registry_path)
    ranked = []
    for entry in entries:
        if not _matches_anatomy(entry, anatomy_label, parent_label):
            continue
        score = 100
        score += _topic_score(entry, query_terms)
        if entry.get("evidence_level") == "guideline_or_standard":
            score += 50
        ranked.append((score, entry))
    ranked.sort(key=lambda item: item[0], reverse=True)

    sources = []
    for score, entry in ranked[:max_results]:
        sources.append(
            {
                "source_id": entry.get("source_id"),
                "source_type": "guideline_metadata",
                "organization": entry.get("organization"),
                "title": entry.get("title"),
                "url": entry.get("url"),
                "year": entry.get("year"),
                "evidence_level": entry.get("evidence_level"),
                "anatomy_scope": entry.get("anatomy_scope"),
                "topic_scope": entry.get("topic_scope"),
                "relevance_score": score,
                "notes": entry.get("notes"),
            }
        )
    return sources
