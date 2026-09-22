from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .input_adapter import extract_stage3_input
from .app_payload import build_app_payload
from .care_guidance import build_care_guidance
from .claim_builder import build_evidence_claims
from .constrained_writer import build_constrained_writer_response
from .evidence_panel import build_evidence_panel
from .clinician_summary import build_clinician_summary
from .evidence_ranker import rank_sources
from .final_display import build_final_display_payload
from .patient_note import build_patient_friendly_note
from .guideline_client import retrieve_guidelines
from .local_llm_writer import LocalLlmWriterConfig, run_local_llm_writer
from .llm_management_writer import (
    LocalManagementLlmConfig,
    build_llm_management_writer_contract,
    run_local_llm_management_writer,
)
from .pubmed_client import retrieve_pubmed_for_queries
from .pubmed_quality_gate import apply_pubmed_quality_gate
from .query_builder import build_query_pack
from .rag_evidence_pack import build_rag_evidence_pack
from .red_flag_engine import build_red_flags
from .review_plan import build_review_plan
from .response_guardrails import build_response_guardrails
from .safety_filter import (
    validate_ai_pipeline_explanation,
    validate_clinical_support_report,
    validate_stage3_block,
)
from .source_relevance import apply_source_relevance_filter
from .template_writer import LOCAL_CURATED_SOURCE, write_template_stage3
from .treatment_guidance import build_treatment_guidance
from .visual_evidence import build_visual_evidence


DEFAULT_CONFIG = Path(__file__).with_name("stage3_config.yaml")

PUBMED_MANAGEMENT_QUERY_ROLES = {
    "management_context",
    "candidate_management_context",
    "red_flag_escalation_context",
    "red_flag_alignment_context",
    "adjacent_joint_context",
    "clinical_localizer_context",
}


def public_source_record(source: Dict[str, Any]) -> Dict[str, Any]:
    """Return source metadata safe for final JSON output.

    Full PubMed abstracts are used internally for controlled claim extraction,
    but the final pipeline output should expose concise citations/metadata only.
    """
    out = dict(source)
    if out.get("source_type") == "pubmed":
        abstract = out.pop("abstract", "")
        out["abstract_available"] = bool(abstract)
        out["abstract_character_count"] = len(abstract or "")
    return out


def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _should_run_pubmed(summary, evidence_cfg: Dict[str, Any]) -> tuple[bool, str]:
    if evidence_cfg.get("pubmed_enabled", False) is not True:
        return False, "disabled_by_config"
    policy = str(evidence_cfg.get("pubmed_trigger_policy") or "always")
    if policy == "retained_candidate_only":
        if summary.accepted_candidate_count > 0:
            return True, "retained_candidate_present"
        return False, "skipped_no_retained_candidate"
    if policy == "retained_candidate_or_stage2c_warning":
        if summary.accepted_candidate_count > 0:
            return True, "retained_candidate_present"
        if summary.stage2c_decision in {"suspicious", "uncertain"}:
            return True, "stage2c_warning_present"
        return False, "skipped_no_candidate_or_stage2c_warning"
    return True, "always"


def _pubmed_management_queries(query_pack: Dict[str, Any], evidence_cfg: Dict[str, Any]) -> list[str]:
    if evidence_cfg.get("pubmed_management_queries_only", True) is not True:
        return list(query_pack.get("queries") or [])
    specs = query_pack.get("query_specs") or []
    queries = [
        str(spec.get("query"))
        for spec in specs
        if isinstance(spec, dict)
        and spec.get("query")
        and str(spec.get("query_role") or "") in PUBMED_MANAGEMENT_QUERY_ROLES
    ]
    if not queries:
        queries = list(query_pack.get("queries") or [])
    seen = set()
    out = []
    for query in queries:
        key = query.lower()
        if key not in seen:
            out.append(query)
            seen.add(key)
    return out


def run_stage3_on_json(pipeline_json: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config or load_config()
    out = copy.deepcopy(pipeline_json)
    summary = extract_stage3_input(pipeline_json)
    query_pack = build_query_pack(summary, max_queries=int(cfg.get("query_builder", {}).get("max_queries", 8)))
    red_flags = build_red_flags(summary)
    mode = cfg.get("stage3", {}).get("default_mode", "clinician_support")
    evidence_cfg = cfg.get("evidence", {})

    sources = [LOCAL_CURATED_SOURCE]
    retrieval_statuses = []

    if evidence_cfg.get("guideline_registry_enabled", True):
        guideline_sources = retrieve_guidelines(
            summary.anatomy_label,
            summary.anatomy_parent_label,
            query_pack.get("queries", []),
            max_results=int(evidence_cfg.get("max_guideline_sources", 4)),
        )
        sources.extend(guideline_sources)
        retrieval_statuses.append(
            {
                "retriever": "guideline_registry",
                "status": "available" if guideline_sources else "no_results",
                "source_count": len(guideline_sources),
            }
        )

    pubmed_should_run, pubmed_trigger_reason = _should_run_pubmed(summary, evidence_cfg)
    if pubmed_should_run:
        pubmed_queries = _pubmed_management_queries(query_pack, evidence_cfg)
        pubmed = retrieve_pubmed_for_queries(
            pubmed_queries,
            max_results_per_query=int(evidence_cfg.get("max_pubmed_results_per_query", 3)),
            total_max_sources=int(evidence_cfg.get("total_max_pubmed_sources", 8)),
            offline_mode=bool(evidence_cfg.get("pubmed_offline_mode", False)),
            cache_dir=evidence_cfg.get("pubmed_cache_dir", "runs/cache/stage3_pubmed"),
            timeout_seconds=int(evidence_cfg.get("pubmed_timeout_seconds", 15)),
            rate_limit_rps=float(evidence_cfg.get("pubmed_rate_limit_rps", 3.0)),
        )
        sources.extend(pubmed.get("sources", []))
        retrieval_statuses.append(
            {
                "retriever": "pubmed",
                "status": pubmed.get("evidence_status"),
                "trigger_reason": pubmed_trigger_reason,
                "source_count": len(pubmed.get("sources", [])),
                "queries_used": pubmed_queries,
                "query_statuses": pubmed.get("query_statuses", []),
            }
        )
    else:
        retrieval_statuses.append(
            {
                "retriever": "pubmed",
                "status": "skipped",
                "trigger_reason": pubmed_trigger_reason,
                "source_count": 0,
                "queries_used": [],
                "query_statuses": [],
            }
        )

    if evidence_cfg.get("source_relevance_filter_enabled", True):
        filtered_sources, source_audits = apply_source_relevance_filter(
            sources,
            summary,
            query_pack.get("queries", []),
            keep_low_relevance=bool(evidence_cfg.get("keep_low_relevance_sources", False)),
        )
    else:
        filtered_sources = sources
        source_audits = [
            {
                "source_id": source.get("source_id"),
                "source_type": source.get("source_type"),
                "kept": True,
                "relevance_label": "not_audited",
                "relevance_score": None,
                "matched_anatomy_terms": [],
                "matched_query_terms": [],
                "matched_concepts": [],
                "exclusion_reasons": [],
                "keep_reasons": ["source_relevance_filter_disabled"],
            }
            for source in sources
        ]

    pubmed_quality_summary: Dict[str, Any] = {
        "version": "pubmed_source_quality_gate_v1",
        "pubmed_source_count": 0,
        "quality_tier_counts": {},
        "source_role_counts": {},
        "allowed_for_llm_context_count": 0,
        "allowed_for_treatment_guidance_count": 0,
        "allowed_for_diagnosis_context_count": 0,
    }
    if evidence_cfg.get("pubmed_quality_gate_enabled", True):
        filtered_sources, pubmed_quality_summary = apply_pubmed_quality_gate(filtered_sources)

    ranked_sources = rank_sources(
        filtered_sources,
        summary.anatomy_label,
        summary.anatomy_parent_label,
        query_pack.get("queries", []),
        max_sources=int(evidence_cfg.get("total_max_ranked_sources", 10)),
    )
    source_ids = {s.get("source_id") for s in ranked_sources}
    if LOCAL_CURATED_SOURCE["source_id"] not in source_ids:
        ranked_sources.append(LOCAL_CURATED_SOURCE)
    allowed_claims = build_evidence_claims(
        summary,
        ranked_sources,
        pubmed_abstract_claims_enabled=bool(evidence_cfg.get("pubmed_abstract_claims_enabled", False)),
        pubmed_treatment_claims_enabled=bool(evidence_cfg.get("pubmed_treatment_claims_enabled", False)),
        max_pubmed_abstract_claims=int(evidence_cfg.get("max_pubmed_abstract_claims", 6)),
        max_pubmed_treatment_claims=int(evidence_cfg.get("max_pubmed_treatment_claims", 4)),
    )

    has_pubmed = any(s.get("source_type") == "pubmed" for s in ranked_sources)
    has_guideline = any(s.get("source_type") == "guideline_metadata" for s in ranked_sources)
    if has_pubmed and has_guideline:
        evidence_status = "available"
    elif has_guideline:
        evidence_status = "guideline_metadata_only"
    elif has_pubmed:
        evidence_status = "pubmed_only"
    else:
        evidence_status = evidence_cfg.get("evidence_status_without_retrieval", "local_curated_only")

    public_ranked_sources = [public_source_record(source) for source in ranked_sources]

    stage3_block = write_template_stage3(
        summary,
        query_pack,
        red_flags,
        sources=public_ranked_sources,
        allowed_claims=allowed_claims,
        evidence_status=evidence_status,
        mode=mode,
    )
    stage3_block["visual_evidence"] = build_visual_evidence(summary, pipeline_json)
    stage3_block["review_plan"] = build_review_plan(summary, red_flags)
    stage3_block["retrieval_statuses"] = retrieval_statuses
    stage3_block["source_relevance_audit"] = {
        "policy": {
            "keep_low_relevance_sources": bool(evidence_cfg.get("keep_low_relevance_sources", False)),
            "source_relevance_filter_enabled": bool(evidence_cfg.get("source_relevance_filter_enabled", True)),
            "pubmed_trigger_policy": str(evidence_cfg.get("pubmed_trigger_policy") or "always"),
            "pubmed_trigger_reason": pubmed_trigger_reason,
            "pubmed_management_queries_only": bool(evidence_cfg.get("pubmed_management_queries_only", True)),
            "pubmed_requires_fracture_or_injury_context": True,
            "pubmed_requires_imaging_context": True,
        },
        "summary": {
            "raw_source_count": len(sources),
            "kept_source_count": len(filtered_sources),
            "excluded_source_count": len(sources) - len(filtered_sources),
            "kept_pubmed_count": sum(
                1 for s in filtered_sources if s.get("source_type") == "pubmed"
            ),
            "excluded_pubmed_count": sum(
                1
                for a in source_audits
                if a.get("source_type") == "pubmed" and not a.get("kept")
            ),
            "pubmed_quality_gate": pubmed_quality_summary,
        },
        "items": source_audits,
    }
    stage3_block["evidence_panel"] = build_evidence_panel(
        evidence_status=evidence_status,
        sources=public_ranked_sources,
        allowed_claims=allowed_claims,
        source_relevance_audit=stage3_block["source_relevance_audit"],
        retrieval_statuses=retrieval_statuses,
    )
    stage3_block["response_guardrails"] = build_response_guardrails(
        summary=summary,
        allowed_claims=allowed_claims,
        sources=public_ranked_sources,
        review_plan=stage3_block["review_plan"],
        evidence_panel=stage3_block["evidence_panel"],
    )
    stage3_block["validated_writer_response"] = build_constrained_writer_response(stage3_block)
    stage3_block["care_guidance"] = build_care_guidance(summary, public_ranked_sources)
    stage3_block["treatment_guidance"] = build_treatment_guidance(
        summary,
        red_flags,
        public_ranked_sources,
        allowed_claims=allowed_claims,
    )
    stage3_block["rag_evidence_pack"] = build_rag_evidence_pack(
        summary=summary,
        query_pack=query_pack,
        evidence_status=evidence_status,
        sources=public_ranked_sources,
        allowed_claims=allowed_claims,
        source_relevance_audit=stage3_block["source_relevance_audit"],
        retrieval_statuses=retrieval_statuses,
        max_sources_for_llm=int(evidence_cfg.get("rag_pack_max_sources_for_llm", 8)),
        max_claims_for_llm=int(evidence_cfg.get("rag_pack_max_claims_for_llm", 12)),
    )
    stage3_block["treatment_guidance"]["rag_used"] = bool(stage3_block["rag_evidence_pack"].get("rag_available"))
    stage3_block["clinician_summary"] = build_clinician_summary(
        summary,
        stage3_block.get("clinical_support", {}),
        allowed_claims,
        public_ranked_sources,
        source_relevance_audit=stage3_block["source_relevance_audit"],
        max_evidence_notes=int(evidence_cfg.get("max_clinician_summary_evidence_notes", 5)),
    )
    if cfg.get("stage3", {}).get("patient_friendly_note_enabled", True):
        stage3_block["patient_friendly_note"] = build_patient_friendly_note(
            summary,
            stage3_block["clinician_summary"],
            max_bullets=int(cfg.get("stage3", {}).get("max_patient_note_bullets", 4)),
        )
    stage3_block["app_payload"] = build_app_payload(summary, stage3_block)
    llm_management_cfg = cfg.get("llm_management_writer", {})
    if llm_management_cfg.get("enabled", False):
        stage3_block["llm_management_writer"] = run_local_llm_management_writer(
            stage3_block,
            LocalManagementLlmConfig(
                model_id_or_path=str(llm_management_cfg.get("model_id_or_path", "google/medgemma-4b-it")),
                local_files_only=bool(llm_management_cfg.get("local_files_only", True)),
                device=str(llm_management_cfg.get("device", "auto")),
                torch_dtype=str(llm_management_cfg.get("torch_dtype", "auto")),
                max_new_tokens=int(llm_management_cfg.get("max_new_tokens", 420)),
                temperature=float(llm_management_cfg.get("temperature", 0.0)),
                top_p=float(llm_management_cfg.get("top_p", 1.0)),
                trust_remote_code=bool(llm_management_cfg.get("trust_remote_code", False)),
                run_model=bool(llm_management_cfg.get("run_model", False)),
                attn_implementation=str(llm_management_cfg.get("attn_implementation", "eager")),
                require_torch_ge_2_6_for_medgemma=bool(
                    llm_management_cfg.get("require_torch_ge_2_6_for_medgemma", True)
                ),
            ),
        )
    else:
        stage3_block["llm_management_writer"] = build_llm_management_writer_contract(stage3_block)
    stage3_block["app_payload"]["llm_management_writer_panel"] = {
        "version": stage3_block["llm_management_writer"].get("version"),
        "status": stage3_block["llm_management_writer"].get("status"),
        "llm_called": stage3_block["llm_management_writer"].get("llm_called") is True,
        "production_enabled": stage3_block["llm_management_writer"].get("production_enabled") is True,
        "display_response": stage3_block["llm_management_writer"].get("display_response") or {},
        "validation": stage3_block["llm_management_writer"].get("validation") or {},
        "safety_policy": stage3_block["llm_management_writer"].get("safety_policy") or {},
        "adapter_audit": stage3_block["llm_management_writer"].get("adapter_audit") or {},
    }
    stage3_block["final_display_payload"] = build_final_display_payload(stage3_block)
    stage3_block["clinical_support_report_validation"] = validate_clinical_support_report(
        (stage3_block.get("final_display_payload") or {}).get("clinical_support_report") or {}
    )
    stage3_block["ai_pipeline_explanation_validation"] = validate_ai_pipeline_explanation(
        (stage3_block.get("final_display_payload") or {}).get("ai_pipeline_explanation") or {}
    )
    stage3_block["evidence_controls"] = {
        "guideline_registry_enabled": bool(evidence_cfg.get("guideline_registry_enabled", True)),
        "pubmed_enabled": bool(evidence_cfg.get("pubmed_enabled", False)),
        "pubmed_trigger_policy": str(evidence_cfg.get("pubmed_trigger_policy") or "always"),
        "pubmed_trigger_reason": pubmed_trigger_reason,
        "pubmed_management_queries_only": bool(evidence_cfg.get("pubmed_management_queries_only", True)),
        "pubmed_abstract_claims_enabled": bool(evidence_cfg.get("pubmed_abstract_claims_enabled", False)),
        "pubmed_treatment_claims_enabled": bool(evidence_cfg.get("pubmed_treatment_claims_enabled", False)),
        "pubmed_offline_mode": bool(evidence_cfg.get("pubmed_offline_mode", False)),
        "pubmed_quality_gate_enabled": bool(evidence_cfg.get("pubmed_quality_gate_enabled", True)),
        "source_relevance_filter_enabled": bool(evidence_cfg.get("source_relevance_filter_enabled", True)),
    }
    critic = validate_stage3_block(stage3_block, cfg.get("safety", {}).get("forbidden_phrases"))
    stage3_block["critic"] = {
        "type": critic["type"],
        "verdict": critic["verdict"],
        "reasons": critic["reasons"],
    }
    stage3_block["safety"]["forbidden_claims_detected"] = bool(critic["forbidden_hits"])
    if critic["verdict"] != "PASS":
        stage3_block["safety"]["safe_fallback_used"] = True
        stage3_block["clinical_support"]["limitations"].append(
            "Stage 3 safety filter flagged the generated support text; use the output only as audit evidence."
        )

    llm_writer_cfg = cfg.get("llm_writer", {})
    if llm_writer_cfg.get("enabled", False):
        stage3_block["local_llm_writer"] = run_local_llm_writer(
            stage3_block,
            LocalLlmWriterConfig(
                model_id_or_path=str(llm_writer_cfg.get("model_id_or_path", "google/medgemma-4b-it")),
                local_files_only=bool(llm_writer_cfg.get("local_files_only", True)),
                device=str(llm_writer_cfg.get("device", "auto")),
                torch_dtype=str(llm_writer_cfg.get("torch_dtype", "auto")),
                max_new_tokens=int(llm_writer_cfg.get("max_new_tokens", 700)),
                temperature=float(llm_writer_cfg.get("temperature", 0.0)),
                top_p=float(llm_writer_cfg.get("top_p", 1.0)),
                trust_remote_code=bool(llm_writer_cfg.get("trust_remote_code", False)),
                run_model=bool(llm_writer_cfg.get("run_model", False)),
                attn_implementation=str(llm_writer_cfg.get("attn_implementation", "eager")),
                prompt_mode=str(llm_writer_cfg.get("prompt_mode", "sections")),
                compact_max_claims=int(llm_writer_cfg.get("compact_max_claims", 6)),
                compact_max_sources=int(llm_writer_cfg.get("compact_max_sources", 6)),
            ),
        )
    else:
        stage3_block["local_llm_writer"] = {
            "version": "stage3_local_llm_writer_v1",
            "status": "DISABLED_BY_CONFIG",
            "production_enabled": False,
            "llm_called": False,
            "planned_model_family": "medgemma_4b_it",
            "note": "Production output remains deterministic unless llm_writer.enabled and llm_writer.run_model are explicitly enabled in a research config.",
        }

    out["stage3_orthopedic_rag"] = stage3_block
    return out


def run_file(input_path: Path, output_path: Path, config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    with input_path.open("r", encoding="utf-8") as f:
        pipeline_json = json.load(f)
    out = run_stage3_on_json(pipeline_json, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Stage 3 orthopedic RAG MVP block to locked pipeline JSON.")
    parser.add_argument("--input", required=True, help="Locked pipeline JSON file or directory")
    parser.add_argument("--output", required=True, help="Output JSON file or directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--enable-pubmed", action="store_true", help="Enable live/cached PubMed retrieval for this run.")
    parser.add_argument("--offline-pubmed", action="store_true", help="Use PubMed cache only; never make live requests.")
    parser.add_argument(
        "--enable-pubmed-abstract-claims",
        action="store_true",
        help="Allow conservative, template-controlled claims derived from PubMed abstracts.",
    )
    parser.add_argument(
        "--enable-pubmed-treatment-claims",
        action="store_true",
        help="Allow conservative, non-prescriptive treatment-management claims from PubMed sources that pass quality gates.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    config = load_config(config_path)
    if args.enable_pubmed:
        config.setdefault("evidence", {})["pubmed_enabled"] = True
    if args.offline_pubmed:
        config.setdefault("evidence", {})["pubmed_offline_mode"] = True
    if args.enable_pubmed_abstract_claims:
        config.setdefault("evidence", {})["pubmed_abstract_claims_enabled"] = True
    if args.enable_pubmed_treatment_claims:
        config.setdefault("evidence", {})["pubmed_treatment_claims_enabled"] = True

    def run_file_with_config(src: Path, dst: Path) -> None:
        with src.open("r", encoding="utf-8") as f:
            pipeline_json = json.load(f)
        out = run_stage3_on_json(pipeline_json, config)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for src in sorted(input_path.glob("*.json")):
            if src.name == "summary.json":
                continue
            run_file_with_config(src, output_path / src.name)
    else:
        run_file_with_config(input_path, output_path)


if __name__ == "__main__":
    main()
