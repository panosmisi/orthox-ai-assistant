from __future__ import annotations

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "runs" / "cache" / "stage3_pubmed"


class PubMedClient:
    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        timeout_seconds: int = 15,
        rate_limit_rps: float = 3.0,
        api_key_env: str = "NCBI_API_KEY",
        offline_mode: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.rate_limit_rps = rate_limit_rps
        self.api_key = os.environ.get(api_key_env)
        self.offline_mode = offline_mode
        self._last_request_time = 0.0

    def _cache_path(self, name: str, params: Dict[str, Any]) -> Path:
        raw = json.dumps(params, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{name}_{digest}.json"

    def _request(self, endpoint: str, params: Dict[str, Any], cache_name: str) -> Dict[str, Any]:
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        params.setdefault("tool", "ProjectDiplomatikiStage3")
        params.setdefault("email", "example@example.com")

        cache_path = self._cache_path(cache_name, {"endpoint": endpoint, "params": params})
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.offline_mode:
            return {"ok": False, "from_cache": False, "offline": True, "error": "offline_cache_miss"}

        min_interval = 1.0 / max(self.rate_limit_rps, 0.1)
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            self._last_request_time = time.time()
            response.raise_for_status()
            payload = {
                "ok": True,
                "from_cache": False,
                "offline": False,
                "url": response.url,
                "text": response.text,
            }
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        except Exception as exc:
            return {"ok": False, "from_cache": False, "offline": False, "error": repr(exc)}

    def esearch(self, query: str, retmax: int = 5) -> Dict[str, Any]:
        return self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": "relevance",
            },
            "esearch",
        )

    def efetch(self, pmids: List[str]) -> Dict[str, Any]:
        return self._request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
            "efetch",
        )

    def search_and_fetch(self, query: str, retmax: int = 5) -> Dict[str, Any]:
        search = self.esearch(query, retmax=retmax)
        if not search.get("ok"):
            return {
                "query": query,
                "ok": False,
                "status": "search_failed",
                "error": search.get("error"),
                "offline": search.get("offline", False),
                "sources": [],
            }
        try:
            payload = json.loads(search["text"])
            pmids = payload.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            return {"query": query, "ok": False, "status": "search_parse_failed", "error": repr(exc), "sources": []}
        if not pmids:
            return {"query": query, "ok": True, "status": "no_results", "sources": []}

        fetch = self.efetch(pmids)
        if not fetch.get("ok"):
            return {
                "query": query,
                "ok": False,
                "status": "fetch_failed",
                "error": fetch.get("error"),
                "offline": fetch.get("offline", False),
                "sources": [],
            }
        sources = parse_pubmed_xml(fetch["text"], query=query)
        return {"query": query, "ok": True, "status": "available", "sources": sources}


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def parse_pubmed_xml(xml_text: str, query: str = "") -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    sources = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract_parts = [_text(node) for node in article.findall(".//Abstract/AbstractText")]
        abstract = "\n".join(part for part in abstract_parts if part)
        journal = _text(article.find(".//Journal/Title"))
        year = _text(article.find(".//PubDate/Year"))
        publication_types = [_text(node) for node in article.findall(".//PublicationType")]
        if not pmid or not title:
            continue
        sources.append(
            {
                "source_id": f"pmid_{pmid}",
                "source_type": "pubmed",
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "publication_types": publication_types,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "query": query,
                "evidence_level": infer_evidence_level(title, abstract, publication_types),
            }
        )
    return sources


def infer_evidence_level(title: str, abstract: str, publication_types: List[str]) -> str:
    title_text = (title or "").lower()
    abstract_text = (abstract or "").lower()
    publication_text = " ".join(publication_types or []).lower()
    combined_text = " ".join([title_text, abstract_text, publication_text])

    # Be conservative: an abstract mentioning a guideline does not make the
    # article itself a guideline. Prefer PubMed publication types or explicit
    # guideline/consensus wording in the title.
    if (
        "practice guideline" in publication_text
        or "guideline" in publication_text
        or "consensus development conference" in publication_text
        or "guideline" in title_text
        or "consensus statement" in title_text
        or "recommendations" in title_text
    ):
        return "guideline_or_standard"
    if "meta-analysis" in publication_text or "meta-analysis" in title_text or "systematic review" in title_text:
        return "systematic_review_or_meta_analysis"
    if "review" in publication_text or "review" in title_text:
        return "review_article"
    if (
        "cohort" in combined_text
        or "diagnostic accuracy" in combined_text
        or "sensitivity" in combined_text
        or "specificity" in combined_text
    ):
        return "diagnostic_accuracy_or_cohort"
    if "case reports" in publication_text or "case report" in title_text or "case series" in title_text:
        return "case_report_or_small_series"
    return "article"


def retrieve_pubmed_for_queries(
    queries: List[str],
    max_results_per_query: int = 3,
    total_max_sources: int = 8,
    offline_mode: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    timeout_seconds: int = 15,
    rate_limit_rps: float = 3.0,
) -> Dict[str, Any]:
    client = PubMedClient(
        cache_dir=cache_dir,
        offline_mode=offline_mode,
        timeout_seconds=timeout_seconds,
        rate_limit_rps=rate_limit_rps,
    )
    all_sources = []
    query_statuses = []
    seen = set()
    for query in queries:
        result = client.search_and_fetch(query, retmax=max_results_per_query)
        query_statuses.append({k: result.get(k) for k in ["query", "ok", "status", "error", "offline"]})
        for source in result.get("sources", []):
            sid = source.get("source_id")
            if sid in seen:
                continue
            all_sources.append(source)
            seen.add(sid)
            if len(all_sources) >= total_max_sources:
                break
        if len(all_sources) >= total_max_sources:
            break
    if all_sources:
        status = "available"
    elif any(q.get("offline") for q in query_statuses):
        status = "unavailable"
    else:
        status = "unavailable"
    return {
        "evidence_status": status,
        "sources": all_sources,
        "query_statuses": query_statuses,
    }
