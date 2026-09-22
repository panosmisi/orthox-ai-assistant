from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _pubmed_trigger_label(trigger: Any) -> str:
    return {
        "retained_candidate_present": "Retained localized AI review cue present",
        "skipped_no_retained_candidate": "No localized candidate required PubMed management-context retrieval",
    }.get(str(trigger or ""), "Evidence-trigger criteria not met or unavailable")


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _md_escape(value: Any) -> str:
    return _safe(value).replace("|", "\\|").replace("\n", " ")


def _image_markdown(image_path: str | None) -> str:
    if not image_path:
        return "_Original image path unavailable._"
    # Markdown renderers in the Codex app handle local Windows paths best with forward slashes.
    normalized = Path(image_path).as_posix()
    return f"![Original X-ray]({normalized})"


def _bullet_list(items: Iterable[Any]) -> str:
    values = [str(item) for item in items if item]
    if not values:
        return "- N/A"
    return "\n".join(f"- {item}" for item in values)


def _first_items(items: Iterable[Any], limit: int) -> List[Any]:
    return [item for item in items if item][:limit]


def _confidence_table(cards: List[Dict[str, Any]]) -> str:
    rows = [
        "| Signal | Available | Model signal level | Internal support score | Display context | Note |",
        "|---|---:|---|---:|---|---|",
    ]
    for card in cards:
        support_score = _pct(card.get("percent")) if card.get("percent") is not None else "not available"
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(card.get("label")),
                    _md_escape(card.get("available")),
                    _md_escape(card.get("support_level_label") or card.get("support_level")),
                    support_score,
                    _md_escape(card.get("display_context") or card.get("display_when")),
                    _md_escape(card.get("calibration_note") or card.get("display_note")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _anatomy_table(ranking: List[Dict[str, Any]]) -> str:
    rows = [
        "| Label | Display label | Support |",
        "|---|---|---:|",
    ]
    for item in ranking[:8]:
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(item.get("label")),
                    _md_escape(item.get("display_label")),
                    _pct(item.get("support_percent")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A |")
    return "\n".join(rows)


def _pipeline_status_table(payload: Dict[str, Any]) -> str:
    header = payload.get("case_header") or {}
    finding = header.get("finding_state") or {}
    visual = payload.get("visual_panel") or {}
    anatomy = payload.get("anatomy_panel") or {}
    return "\n".join(
        [
            "| Field | Value |",
            "|---|---|",
            f"| Finding state | {_md_escape(finding.get('state_id'))} |",
            f"| Severity | {_md_escape(finding.get('severity'))} |",
            f"| Visual priority | {_md_escape(visual.get('display_priority'))} |",
            f"| Show bbox | {_md_escape(visual.get('show_primary_bbox'))} |",
            f"| Heatmap default | {_md_escape(visual.get('show_stage2c_heatmap_by_default'))} |",
            f"| Anatomy label | {_md_escape(anatomy.get('selected_display_label'))} |",
            f"| Anatomy output type | {_md_escape(anatomy.get('output_type'))} |",
            f"| Human verification required | {_md_escape(header.get('human_verification_required'))} |",
            f"| Not a diagnosis | {_md_escape(header.get('not_a_diagnosis'))} |",
        ]
    )


def _compact_support_table(signals: List[Dict[str, Any]]) -> str:
    rows = [
        "| Signal | Support | Note |",
        "|---|---:|---|",
    ]
    for signal in signals:
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(signal.get("label")),
                    _pct(signal.get("percent")),
                    _md_escape(signal.get("calibration_note")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A |")
    return "\n".join(rows)


def _clinician_compact_card_section(card: Dict[str, Any]) -> List[str]:
    anatomy = card.get("anatomy") or {}
    return [
        "## Clinician Compact Card",
        "",
        f"Headline: **{_safe(card.get('headline'))}**",
        "",
        f"State: `{_safe(card.get('state_id'))}`",
        "",
        f"Primary action: **{_safe(card.get('primary_action'))}**",
        "",
        f"Visual priority: `{_safe(card.get('visual_priority'))}`",
        "",
        f"Anatomy: **{_safe(anatomy.get('selected_display_label') or anatomy.get('parent_display_label'))}**",
        "",
        f"Anatomy output type: `{_safe(anatomy.get('output_type'))}`",
        "",
        "Support signals:",
        "",
        _compact_support_table(card.get("support_signals") or []),
        "",
        "Top review focus:",
        "",
        _bullet_list(_first_items(card.get("top_review_focus") or [], 2)),
        "",
        "Top clinical context prompts:",
        "",
        _bullet_list(_first_items(card.get("top_clinical_context_prompts") or [], 2)),
        "",
        "Top radiograph review prompts:",
        "",
        _bullet_list(_first_items(card.get("top_radiograph_review_prompts") or [], 2)),
        "",
        "Key uncertainty notes:",
        "",
        _bullet_list(_first_items(card.get("key_uncertainty_notes") or [], 1)),
        "",
        "Key boundaries:",
        "",
        _bullet_list(_first_items(card.get("key_boundaries") or [], 2)),
        "",
    ]


def _treatment_guidance_section(treatment: Dict[str, Any]) -> List[str]:
    management = treatment.get("management_guidance") or {}
    direction = management.get("management_direction") or {}
    case_suggestions = management.get("case_management_suggestions") or {}
    anatomy_management = management.get("anatomy_specific_management") or treatment.get("anatomy_specific_management_v3") or {}
    traceability = management.get("guidance_traceability") or {}
    context_lines = []
    for item in _first_items(management.get("management_context") or [], 3):
        if not isinstance(item, dict):
            continue
        sources = ", ".join(str(source_id) for source_id in item.get("source_ids") or [])
        context_lines.append(
            f"- {_safe(item.get('text'))} "
            f"Sources: `{_safe(sources or 'none')}`. "
            f"Boundary: {_safe(item.get('prohibited_interpretation'))}"
        )
    pubmed_lines = []
    for item in _first_items(management.get("pubmed_rag_synthesis") or [], 4):
        if not isinstance(item, dict):
            continue
        citation = item.get("pmid") or item.get("source_id") or "source unavailable"
        pubmed_lines.append(
            f"- {_safe(item.get('text'))} "
            f"Source: `{_safe(citation)}`. "
            f"Role: `{_safe(item.get('pubmed_source_role'))}`. "
            f"Boundary: {_safe(item.get('prohibited_interpretation'))}"
        )
    suggestion_lines = []
    for item in _first_items(case_suggestions.get("suggestions") or [], 4):
        if not isinstance(item, dict):
            continue
        sources = ", ".join(str(source_id) for source_id in item.get("source_ids") or [])
        suggestion_lines.append(
            f"- **{_safe(item.get('category'))}**: {_safe(item.get('text'))} "
            f"Rationale: {_safe(item.get('rationale'))} "
            f"Sources: `{_safe(sources or 'none')}`. "
            f"Boundary: {_safe(item.get('prohibited_interpretation'))}"
        )
    source_lines = []
    for source in _first_items(management.get("sources") or [], 4):
        if not isinstance(source, dict):
            continue
        supports = "; ".join(str(item) for item in _first_items(source.get("supports") or [], 1))
        source_lines.append(
            f"- **{_safe(source.get('id'))}**: {_safe(source.get('title'))} "
            f"({_safe(source.get('publisher_or_journal'))}, {_safe(source.get('year'))}). "
            f"Supports: {_safe(supports)}"
        )
    trace_lines = []
    for item in _first_items(traceability.get("items") or [], 6):
        if not isinstance(item, dict):
            continue
        sources = ", ".join(str(source_id) for source_id in item.get("source_ids") or [])
        trace_lines.append(
            f"- `{_safe(item.get('item_type'))}` linked to `{_safe(sources or 'none')}`: "
            f"{_safe(item.get('text'))}"
        )
    return [
        "## Evidence-Linked Management Context",
        "",
        f"Button label: **{_safe(management.get('button_label') or treatment.get('button_label'))}**",
        "",
        f"Panel badge: **{_safe(management.get('panel_badge') or treatment.get('panel_badge'))}**",
        "",
        f"Warning: **{_safe(management.get('panel_warning') or treatment.get('panel_warning'))}**",
        "",
        _safe(management.get("safety_disclaimer") or treatment.get("safety_disclaimer")),
        "",
        "Management direction:",
        "",
        f"Status: **{_safe(direction.get('status_id'))}**",
        "",
        _safe(direction.get("summary")),
        "",
        "Direction bullets:",
        "",
        _bullet_list(_first_items(direction.get("direction_bullets") or [], 3)),
        "",
        f"Decision boundary: {_safe(direction.get('decision_boundary'))}",
        "",
        "Anatomy-specific management context:",
        "",
        f"Template: `{_safe(anatomy_management.get('template_id'))}`",
        "",
        f"Review focus: {_safe(anatomy_management.get('review_focus'))}",
        "",
        "Clinical localizers:",
        "",
        _bullet_list(_first_items(anatomy_management.get("clinical_localizers") or [], 4)),
        "",
        "Anatomy-specific considerations:",
        "",
        _bullet_list(_first_items(anatomy_management.get("management_considerations") or [], 3)),
        "",
        "Case-oriented management suggestions:",
        "",
        "\n".join(suggestion_lines) if suggestion_lines else "- No case-oriented suggestions available.",
        "",
        "Management context:",
        "",
        "\n".join(context_lines) if context_lines else "- N/A",
        "",
        "PubMed RAG synthesis:",
        "",
        "\n".join(pubmed_lines) if pubmed_lines else "- No approved PubMed-derived claims were available for this case.",
        "",
        "Before management decisions, clinician needs:",
        "",
        _bullet_list(_first_items(management.get("clinical_inputs_required") or [], 6)),
        "",
        "Clinical information not available to the AI:",
        "",
        _bullet_list(_first_items(management.get("clinical_information_not_available_to_ai") or [], 7)),
        "",
        "Red flags requiring escalation:",
        "",
        _bullet_list(_first_items(management.get("red_flags") or [], 4)),
        "",
        "AI must not decide:",
        "",
        _bullet_list(_first_items(management.get("ai_must_not_decide") or [], 6)),
        "",
        "Case-status cautions:",
        "",
        _bullet_list(_first_items(management.get("status_specific_cautions") or [], 3)),
        "",
        "Sources used for this panel:",
        "",
        "\n".join(source_lines) if source_lines else "- No source cards available.",
        "",
        "Management item source traceability:",
        "",
        f"All items source-linked: **{_safe(traceability.get('all_items_source_linked'))}**",
        "",
        "\n".join(trace_lines) if trace_lines else "- No traceability items available.",
        "",
        "Limitations:",
        "",
        _bullet_list(_first_items(management.get("limitations") or [], 4)),
        "",
    ]


def _llm_management_writer_section(panel: Dict[str, Any]) -> List[str]:
    response = panel.get("display_response") or {}
    sections = response.get("sections") or {}
    validation = panel.get("validation") or {}
    audit = panel.get("adapter_audit") or {}
    return [
        "## LLM/RAG Management Writer",
        "",
        f"Status: **{_safe(panel.get('status'))}**",
        "",
        f"LLM called: **{_safe(panel.get('llm_called'))}**",
        "",
        f"Validation verdict: **{_safe(validation.get('verdict'))}**",
        "",
        f"Suggestion count in packet: **{_safe(audit.get('suggestion_count'))}**",
        "",
        f"PubMed synthesis count in packet: **{_safe(audit.get('pubmed_synthesis_count'))}**",
        "",
        "Management orientation:",
        "",
        _safe(sections.get("management_orientation")),
        "",
        "Case-specific suggestions:",
        "",
        _bullet_list(_first_items(sections.get("case_specific_suggestions") or [], 4)),
        "",
        "Evidence-supported context:",
        "",
        _bullet_list(
            _first_items(
                sections.get("evidence_supported_context")
                or sections.get("pubmed_supported_context")
                or [],
                4,
            )
        ),
        "",
        "Safety boundary:",
        "",
        _safe(sections.get("safety_boundary")),
        "",
    ]


def _rag_claims_table(claims: List[Dict[str, Any]]) -> str:
    rows = [
        "| Claim id | Source | Lock | Role | Allowed for LLM | Treatment allowed | Claim |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for claim in claims[:8]:
        lock = claim.get("claim_lock") or {}
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(claim.get("claim_id")),
                    _md_escape(claim.get("source_id")),
                    _md_escape(lock.get("status")),
                    _md_escape(claim.get("claim_role") or claim.get("pubmed_source_role")),
                    _md_escape(claim.get("allowed_for_llm")),
                    _md_escape(claim.get("allowed_for_treatment_guidance")),
                    _md_escape(claim.get("claim_text")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def _rag_source_lanes_table(source_lanes: Dict[str, Any]) -> str:
    rows = [
        "| Lane | Sources | LLM-allowed | Treatment-allowed | Quality tiers | Roles |",
        "|---|---:|---:|---:|---|---|",
    ]
    for lane in source_lanes.get("lanes") or []:
        if not isinstance(lane, dict):
            continue
        tiers = ", ".join(f"{k}:{v}" for k, v in (lane.get("quality_tiers") or {}).items())
        roles = ", ".join(f"{k}:{v}" for k, v in (lane.get("roles") or {}).items())
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(lane.get("lane_id")),
                    _safe(lane.get("source_count")),
                    _safe(lane.get("allowed_for_llm_count")),
                    _safe(lane.get("allowed_for_treatment_guidance_count")),
                    _md_escape(tiers),
                    _md_escape(roles),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def _rag_query_specs_table(query_specs: List[Dict[str, Any]]) -> str:
    rows = [
        "| Query | Role | Anatomy focus | Priority |",
        "|---|---|---|---:|",
    ]
    for item in query_specs[:8]:
        if not isinstance(item, dict):
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(item.get("query")),
                    _md_escape(item.get("query_role")),
                    _md_escape(item.get("anatomy_focus")),
                    _safe(item.get("priority")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def _rag_evidence_pack_section(rag: Dict[str, Any]) -> List[str]:
    retrieval = rag.get("retrieval_plan") or {}
    retrievers = retrieval.get("retrievers") or {}
    safety = rag.get("safety_policy") or {}
    contract = rag.get("future_llm_contract") or {}
    source_summary = rag.get("source_summary") or {}
    pubmed_quality = source_summary.get("pubmed_quality_gate") or {}
    source_lanes = rag.get("source_lanes") or source_summary.get("source_lanes") or {}
    evidence_strength = rag.get("evidence_strength") or source_summary.get("evidence_strength") or {}
    readiness_gate = rag.get("readiness_gate") or source_summary.get("readiness_gate") or {}
    return [
        "## RAG Evidence Pack",
        "",
        f"Evidence pack schema: **{_safe(rag.get('evidence_pack_schema_version'))}**",
        "",
        f"RAG mode: **{_safe(rag.get('rag_mode'))}**",
        "",
        f"Evidence strength: **{_safe(evidence_strength.get('label'))}**",
        "",
        f"Evidence strength level: `{_safe(evidence_strength.get('level'))}`",
        "",
        f"Evidence strength boundary: {_safe(evidence_strength.get('interpretation'))}",
        "",
        f"Readiness gate: **{_safe(readiness_gate.get('status'))}**",
        "",
        f"LLM pack ready: **{_safe(readiness_gate.get('llm_pack_ready'))}**",
        "",
        f"RAG available: **{_safe(rag.get('rag_available'))}**",
        "",
        f"LLM used: **{_safe(rag.get('llm_used'))}**",
        "",
        f"Guideline retriever status: **{_safe((retrievers.get('guideline_registry') or {}).get('status'))}**",
        "",
        f"PubMed retriever status: **{_safe((retrievers.get('pubmed') or {}).get('status'))}**",
        "",
        f"PubMed quality gate: **{_safe(pubmed_quality.get('version'))}**",
        "",
        f"PubMed treatment-management sources allowed: **{_safe(pubmed_quality.get('allowed_for_treatment_guidance_count'))}**",
        "",
        f"PubMed diagnosis/imaging sources allowed: **{_safe(pubmed_quality.get('allowed_for_diagnosis_context_count'))}**",
        "",
        "Source lanes:",
        "",
        _rag_source_lanes_table(source_lanes),
        "",
        "Query set:",
        "",
        _bullet_list(retrieval.get("queries") or []),
        "",
        "Structured query strategy:",
        "",
        f"Strategy: `{_safe((retrieval.get('query_strategy') or {}).get('version'))}`",
        "",
        f"Goal: {_safe((retrieval.get('query_strategy') or {}).get('goal'))}",
        "",
        _rag_query_specs_table(retrieval.get("query_specs") or []),
        "",
        "Claim pack for future LLM:",
        "",
        _rag_claims_table(rag.get("claims_for_llm") or []),
        "",
        "RAG safety policy:",
        "",
        _bullet_list(
            [
                "No raw PubMed abstracts in public output"
                if safety.get("no_raw_pubmed_abstracts_in_public_output")
                else None,
                "LLM may only use claims_for_llm"
                if safety.get("llm_may_only_use_claims_for_llm")
                else None,
                "LLM may not create new medical claims"
                if safety.get("llm_may_not_create_new_medical_claims")
                else None,
                "Guideline metadata is context, not patient-specific treatment"
                if safety.get("guideline_metadata_is_context_not_patient_specific_treatment")
                else None,
                "PubMed is supporting literature, not standalone treatment protocol"
                if safety.get("pubmed_is_supporting_literature_not_standalone_treatment_protocol")
                else None,
            ]
        ),
        "",
        f"Future LLM required validator: **{_safe(contract.get('required_validator'))}**",
        "",
        f"Fallback if validation fails: **{_safe(contract.get('fallback_if_validation_fails'))}**",
        "",
    ]


def _review_plan_section(review_plan: Dict[str, Any]) -> List[str]:
    lane = review_plan.get("lane") or {}
    tasks = review_plan.get("review_tasks") or []
    uncertainty = review_plan.get("uncertainty_notes") or []
    triggers = review_plan.get("escalation_triggers_to_check") or []
    boundaries = review_plan.get("output_boundaries") or []
    task_lines = [
        f"- **{_safe(task.get('task_id'))}**: {_safe(task.get('task'))} Reason: {_safe(task.get('reason'))}"
        for task in tasks
        if isinstance(task, dict)
    ]
    trigger_lines = [
        f"- **{_safe(trigger.get('category'))}**: {_safe(trigger.get('trigger'))}"
        for trigger in triggers
        if isinstance(trigger, dict)
    ]
    return [
        "## Clinician Review Plan",
        "",
        f"Lane: **{_safe(lane.get('lane_id'))}**",
        "",
        f"Priority: **{_safe(lane.get('priority'))}**",
        "",
        f"Plan headline: **{_safe(lane.get('headline'))}**",
        "",
        _safe(lane.get("rationale")),
        "",
        "Review tasks:",
        "",
        "\n".join(task_lines) if task_lines else "- N/A",
        "",
        "Uncertainty notes:",
        "",
        _bullet_list(uncertainty),
        "",
        "Escalation triggers to check:",
        "",
        "\n".join(trigger_lines) if trigger_lines else "- N/A",
        "",
        "Output boundaries:",
        "",
        _bullet_list(boundaries),
        "",
    ]


def _sources_table(sources: List[Dict[str, Any]]) -> str:
    rows = [
        "| Source | Type | Relevance | Evidence level | PubMed role | Quality tier | Treatment allowed | URL |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    for source in sources[:8]:
        quality = source.get("pubmed_quality") or {}
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(source.get("title") or source.get("source_id")),
                    _md_escape(source.get("source_type")),
                    _md_escape(source.get("relevance_label")),
                    _md_escape(source.get("evidence_level")),
                    _md_escape(quality.get("source_role")),
                    _md_escape(quality.get("quality_tier")),
                    _md_escape(quality.get("allowed_for_treatment_guidance")),
                    _md_escape(source.get("url")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def _claims_table(claims: List[Dict[str, Any]]) -> str:
    rows = [
        "| Claim id | Allowed | Source | PubMed role | Treatment allowed | Claim |",
        "|---|---:|---|---|---:|---|",
    ]
    for claim in claims[:8]:
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_escape(claim.get("claim_id")),
                    _md_escape(claim.get("allowed")),
                    _md_escape(claim.get("source_id")),
                    _md_escape(claim.get("pubmed_source_role")),
                    _md_escape(claim.get("allowed_for_treatment_guidance")),
                    _md_escape(claim.get("claim_text")),
                ]
            )
            + " |"
        )
    if len(rows) == 2:
        rows.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(rows)


def _evidence_section(evidence: Dict[str, Any]) -> List[str]:
    safety = evidence.get("safety_checks") or {}
    source_summary = evidence.get("source_summary") or {}
    claim_summary = evidence.get("claim_summary") or {}
    return [
        "## Evidence Transparency",
        "",
        f"Evidence status: **{_safe(evidence.get('evidence_status'))}**",
        "",
        f"Displayed sources: **{_safe(source_summary.get('displayed_source_count'))}** of **{_safe(source_summary.get('total_ranked_source_count'))}**",
        "",
        f"Displayed allowed claims: **{_safe(claim_summary.get('displayed_claim_count'))}** of **{_safe(claim_summary.get('total_allowed_claim_count'))}**",
        "",
        f"PubMed treatment-management sources allowed: **{_safe((source_summary.get('pubmed_quality_gate') or {}).get('allowed_for_treatment_guidance_count'))}**",
        "",
        "Source table:",
        "",
        _sources_table(evidence.get("displayed_sources") or []),
        "",
        "Allowed claim table:",
        "",
        _claims_table(evidence.get("displayed_allowed_claims") or []),
        "",
        "Evidence safety checks:",
        "",
        _bullet_list(
            [
                "All displayed claims are allowed" if safety.get("all_displayed_claims_allowed") else None,
                "All displayed claims have sources" if safety.get("all_displayed_claims_have_sources") else None,
                "PubMed abstracts are not exposed" if safety.get("pubmed_abstracts_not_exposed") else None,
                "Panel is transparency only" if safety.get("panel_is_transparency_only") else None,
            ]
        ),
        "",
        "Evidence display rules:",
        "",
        _bullet_list(evidence.get("display_rules") or []),
        "",
    ]


def _response_guardrails_section(guardrails: Dict[str, Any]) -> List[str]:
    validation = guardrails.get("validation") or {}
    approved = guardrails.get("approved_inputs") or {}
    return [
        "## Response Guardrails",
        "",
        f"Current writer mode: **{_safe(guardrails.get('current_writer_mode'))}**",
        "",
        f"LLM used: **{_safe(guardrails.get('llm_used'))}**",
        "",
        f"Review lane available to writer: **{_safe(approved.get('review_lane'))}**",
        "",
        f"Evidence status available to writer: **{_safe(approved.get('evidence_status'))}**",
        "",
        "Required output constraints:",
        "",
        _bullet_list(guardrails.get("required_output_constraints") or []),
        "",
        "Blocked content categories:",
        "",
        _bullet_list(guardrails.get("blocked_content_categories") or []),
        "",
        "Guardrail validation:",
        "",
        _bullet_list(
            [
                "All claim references are allowed and sourced"
                if validation.get("all_claim_refs_allowed_and_sourced")
                else None,
                "PubMed abstracts are excluded"
                if validation.get("pubmed_abstracts_excluded")
                else None,
                "New claim creation is not allowed"
                if validation.get("new_claim_creation_allowed") is False
                else None,
                "Human verification required"
                if validation.get("human_verification_required")
                else None,
            ]
        ),
        "",
    ]


def _validated_writer_section(writer_panel: Dict[str, Any]) -> List[str]:
    sections = writer_panel.get("sections") or {}
    validation = writer_panel.get("validation") or {}
    return [
        "## Validated Writer Response",
        "",
        f"Writer type: **{_safe(writer_panel.get('writer_type'))}**",
        "",
        f"LLM used: **{_safe(writer_panel.get('llm_used'))}**",
        "",
        f"Validation verdict: **{_safe(validation.get('verdict'))}**",
        "",
        "Clinician-facing draft:",
        "",
        _safe(sections.get("clinician_summary")),
        "",
        "Patient-friendly draft:",
        "",
        _safe(sections.get("patient_note")),
        "",
        "Safety footer:",
        "",
        _safe(sections.get("safety_footer")),
        "",
    ]


def _care_guidance_section(care: Dict[str, Any]) -> List[str]:
    tier = care.get("care_tier") or {}
    template = care.get("anatomy_template") or {}
    checks = care.get("review_checks") or []
    considerations = care.get("pathway_considerations") or []
    check_lines = [
        f"- **{_safe(item.get('check_id'))}**: {_safe(item.get('check'))} Reason: {_safe(item.get('why'))}"
        for item in checks
        if isinstance(item, dict)
    ][:4]
    consideration_lines = [
        f"- **{_safe(item.get('consideration_id'))}**: {_safe(item.get('text'))}"
        for item in considerations
        if isinstance(item, dict)
    ][:3]
    return [
        "## Care Guidance",
        "",
        f"Care tier: **{_safe(tier.get('tier_id'))}**",
        "",
        f"Urgency: **{_safe(tier.get('urgency'))}**",
        "",
        f"Meaning: {_safe(tier.get('meaning'))}",
        "",
        f"Dictionary: `{_safe(care.get('dictionary_version'))}`",
        "",
        f"Anatomy template: `{_safe(template.get('template_id'))}`",
        "",
        "Review focus:",
        "",
        _bullet_list(_first_items(care.get("review_focus") or [], 3)),
        "",
        "Clinical context prompts:",
        "",
        _bullet_list(_first_items(care.get("clinical_context_prompts") or [], 2)),
        "",
        "Radiograph review prompts:",
        "",
        _bullet_list(_first_items(care.get("radiograph_review_prompts") or [], 2)),
        "",
        "Review checks:",
        "",
        "\n".join(check_lines) if check_lines else "- N/A",
        "",
        "Pathway considerations:",
        "",
        "\n".join(consideration_lines) if consideration_lines else "- N/A",
        "",
        "Uncertainty notes:",
        "",
        _bullet_list(_first_items(care.get("uncertainty_notes") or [], 2)),
        "",
        "When to escalate review:",
        "",
        _bullet_list(_first_items(care.get("when_to_escalate_review") or [], 4)),
        "",
        "Do not infer:",
        "",
        _bullet_list(_first_items(care.get("do_not_infer") or [], 3)),
        "",
    ]


def _clinical_support_report_section(report: Dict[str, Any]) -> List[str]:
    summary = report.get("clinical_support_summary") or {}
    visual = report.get("visual_evidence") or {}
    model = report.get("model_evidence") or {}
    anatomy = report.get("anatomy") or {}
    review = report.get("clinical_review_guidance") or {}
    management = report.get("management_context") or {}
    evidence = report.get("source_linked_evidence") or {}
    pubmed_retrieval = evidence.get("pubmed_retrieval") or {}
    evidence_mode = evidence.get("evidence_mode") or {}
    pipeline = report.get("pipeline_explanation") or {}
    management_lines = []
    for item in management.get("management_considerations") or []:
        if not isinstance(item, dict):
            management_lines.append(f"- {_safe(item)}")
            continue
        support_ids = ", ".join(str(sid) for sid in item.get("support_ids") or [])
        management_lines.append(
            f"- {_safe(item.get('text'))}\n"
            f"  Support: `{_safe(support_ids or 'source unavailable')}` "
            f"({_safe(item.get('support_type'))}). "
            f"Limitation: {_safe(item.get('limitation'))}"
        )
    source_card_lines = []
    for source in evidence.get("source_cards") or []:
        if not isinstance(source, dict):
            continue
        used_for = "; ".join(str(item) for item in (source.get("used_for") or [])[:2])
        limitations = "; ".join(str(item) for item in (source.get("limitations") or [])[:2])
        citation_parts = [
            source.get("publisher_or_journal"),
            source.get("year"),
            f"PMID {source.get('pmid')}" if source.get("pmid") else None,
        ]
        citation = ", ".join(str(part) for part in citation_parts if part)
        source_card_lines.append(
            f"- **{_safe(source.get('title'))}**\n"
            f"  Type: `{_safe(source.get('source_type'))}`. Evidence level: `{_safe(source.get('evidence_level'))}`. "
            f"{_safe(citation, default='')}\n"
            f"  Used for: {_safe(used_for)}\n"
            f"  Limitation: {_safe(limitations)}"
        )
    return [
        "## Stage 3 Clinical-Support Report",
        "",
        f"> {_safe(report.get('academic_prototype_disclaimer'))}",
        "",
        "### Case status",
        "",
        f"**{_safe(report.get('status_title'))}**",
        "",
        _safe(report.get("status_explanation")),
        "",
        f"Why this status was assigned: {_safe(report.get('why_this_status_was_assigned'))}",
        "",
        "### AI clinical-support summary",
        "",
        f"Suggested review focus: **{_safe(summary.get('suggested_review_focus'))}**",
        "",
        _safe(summary.get("interpretation")),
        "",
        "Required human review:",
        "",
        _bullet_list(summary.get("required_human_review") or []),
        "",
        "### Visual evidence",
        "",
        _safe(visual.get("visual_evidence_note")),
        "",
        _safe(visual.get("heatmap_safety_note"), default=""),
        "",
        f"- Bounding box displayed: `{_safe(visual.get('bbox_displayed'))}`",
        f"- Heatmap available: `{_safe(visual.get('heatmap_available'))}`",
        "",
        "### Model evidence",
        "",
        f"- Candidate internal support level: `{_safe(model.get('candidate_support_level'))}`",
        f"- Candidate internal support score: `{_safe(model.get('candidate_support_score'))}`",
        f"- Image-level internal support level: `{_safe(model.get('image_level_support_level'))}`",
        "",
        _safe(model.get("candidate_display_note"), default=""),
        "",
        _safe(model.get("localization_display_note"), default=""),
        "",
        _safe(model.get("image_level_display_note"), default=""),
        "",
        _safe(model.get("score_disclaimer")),
        "",
        "### Anatomy context",
        "",
        f"- Selected region: **{_safe(anatomy.get('selected_display_label') or anatomy.get('parent_display_label'))}**",
        f"- Fine anatomy uncertain: `{_safe(anatomy.get('fine_anatomy_uncertain'))}`",
        "",
        _safe(anatomy.get("uncertainty_note"), default=""),
        "",
        _safe(anatomy.get("anatomy_disclaimer")),
        "",
        "### Clinical review guidance",
        "",
        "Review focus:",
        "",
        _bullet_list(review.get("review_focus") or []),
        "",
        "Clinical inputs required before management decisions:",
        "",
        _bullet_list(review.get("clinical_inputs_required") or []),
        "",
        "Red flags / escalation triggers for clinician review:",
        "",
        _bullet_list(review.get("red_flags") or []),
        "",
        "### Evidence-linked management context for clinician review",
        "",
        _safe(management.get("subtitle")),
        "",
        f"Guidance level: **{_safe(management.get('guidance_level_label'))}**",
        "",
        f"Why this section is shown: {_safe(management.get('why_shown'))}",
        "",
        _safe(management.get("parent_fallback_note"), default=""),
        "",
        "Management considerations:",
        "",
        "\n".join(management_lines) if management_lines else "- N/A",
        "",
        "AI must not decide:",
        "",
        _bullet_list(management.get("ai_must_not_decide") or []),
        "",
        "Safety boundary:",
        "",
        _safe(management.get("safety_boundary")),
        "",
        "### Source-linked evidence",
        "",
        f"Evidence status: **{_safe(evidence.get('evidence_status'))}**",
        "",
        f"Evidence mode: **{_safe(evidence_mode.get('label'))}**",
        "",
        _safe(evidence_mode.get("description"), default=""),
        "",
        "PubMed/RAG retrieval:",
        "",
        f"- Status: `{_safe(pubmed_retrieval.get('status'))}`",
        f"- Trigger: {_safe(_pubmed_trigger_label(pubmed_retrieval.get('trigger_reason')))}",
        f"- Source count: `{_safe(pubmed_retrieval.get('source_count'))}`",
        f"- Scope: `{_safe(pubmed_retrieval.get('scope'))}`",
        "",
        _safe(pubmed_retrieval.get("statement")),
        "",
        "Displayed sources:",
        "",
        "\n".join(source_card_lines) if source_card_lines else "- N/A",
        "",
        "### How the AI reached this result",
        "",
        _safe(pipeline.get("plain_language_summary")),
        "",
        _bullet_list(pipeline.get("stage_summaries") or []),
        "",
    ]


def _pipeline_explanation_section(explanation: Dict[str, Any]) -> List[str]:
    if not isinstance(explanation, dict) or explanation.get("enabled") is not True:
        return []
    case_status = explanation.get("case_status_explanation") or {}
    evidence = explanation.get("evidence_trace") or {}
    evidence_used = explanation.get("evidence_used") or {}
    technical = explanation.get("technical_details") or {}
    stage_lines: List[str] = []
    for stage in explanation.get("stage_summaries") or []:
        if not isinstance(stage, dict):
            continue
        stage_lines.extend(
            [
                f"### {_safe(stage.get('stage_display_label') or stage.get('stage_name'))}",
                "",
                f"- Purpose: {_safe(stage.get('purpose'))}",
                f"- Input used: {_safe(stage.get('input_used'))}",
                f"- Output for this image: {_safe(stage.get('output'))}",
                f"- How it affected the result: {_safe(stage.get('how_it_affected_result') or stage.get('affected_result'))}",
                "",
                "Limitations:",
                "",
                _bullet_list(stage.get("limitations") or []),
                "",
            ]
        )
    return [
        "<details>",
        f"<summary>{_safe(explanation.get('label'))}</summary>",
        "",
        f"## {_safe(explanation.get('panel_title'))}",
        "",
        f"> {_safe(explanation.get('safety_disclaimer'))}",
        "",
        _safe(explanation.get("pipeline_overview")),
        "",
        "### Short explanation",
        "",
        _safe(case_status.get("short_explanation")),
        "",
        "### Why this result was shown",
        "",
        f"- Display status: **{_safe(case_status.get('human_status_label') or case_status.get('case_status_display'))}**",
        f"- Reason: {_safe(case_status.get('why_this_result_was_shown') or case_status.get('short_reason'))}",
        "",
        "Pipeline flow:",
        "",
        _safe(case_status.get("pipeline_flow_text")),
        "",
        "Status-specific limitations:",
        "",
        _bullet_list(case_status.get("status_specific_limitations") or []),
        "",
        "### Bottom line",
        "",
        _safe(case_status.get("bottom_line")),
        "",
        "### Evidence used in this explanation",
        "",
        f"- Localized bbox: `{_safe(evidence_used.get('localized_bbox'))}`",
        f"- Heatmap: `{_safe(evidence_used.get('heatmap'))}`",
        f"- Image-level caution: `{_safe(evidence_used.get('image_level_caution'))}`",
        f"- Anatomy label: **{_safe(evidence_used.get('anatomy_label'))}**",
        f"- Clinical symptoms: {_safe(evidence_used.get('clinical_symptoms'))}",
        f"- Mechanism of injury: {_safe(evidence_used.get('mechanism_of_injury'))}",
        "",
        "### Stage-by-stage explanation",
        "",
        *stage_lines,
        "### What the AI did not do",
        "",
        _bullet_list(explanation.get("what_ai_did_not_do") or []),
        "",
        "### Required human review",
        "",
        _bullet_list(explanation.get("required_human_review") or []),
        "",
        "### Score disclaimers",
        "",
        _bullet_list(explanation.get("score_disclaimers") or []),
        "",
        f"Technical note: {_safe(explanation.get('technical_note'))}",
        "",
        "<details>",
        "<summary>Technical details</summary>",
        "",
        f"- Raw case status: `{_safe(technical.get('raw_case_status') or case_status.get('case_status'))}`",
        f"- Raw stage ids: `{_safe(', '.join(technical.get('raw_stage_ids') or []))}`",
        f"- Localized bounding box available: `{_safe(evidence.get('bbox_display') or evidence.get('bbox_available'))}`",
        f"- Bounding box source: `{_safe(evidence.get('bbox_source'))}`",
        f"- Bounding box modified by Stage 3: `{_safe(evidence.get('bbox_modified_by_stage3'))}`",
        f"- Heatmap available: `{_safe(evidence.get('heatmap_display') or evidence.get('heatmap_available'))}`",
        f"- Heatmap source: `{_safe(evidence.get('heatmap_source'))}`",
        f"- Selected anatomy context: **{_safe(evidence.get('selected_anatomy'))}**",
        f"- Model-support summary: {_safe(evidence.get('model_support_summary'))}",
        f"- Artifact check: {_safe(evidence.get('artifact_check_summary'))}",
        f"- Visual evidence note: {_safe(evidence.get('visual_evidence_note'))}",
        "",
        "</details>",
        "",
        "</details>",
        "",
    ]


def render_case_report_markdown(final_json: Dict[str, Any]) -> str:
    """Render a deterministic human-readable case report from Stage 3 app payload."""
    stage3 = final_json.get("stage3_orthopedic_rag") or {}
    payload = stage3.get("app_payload") or {}
    header = payload.get("case_header") or {}
    finding = header.get("finding_state") or {}
    visual = payload.get("visual_panel") or {}
    clinician = payload.get("clinician_panel") or {}
    compact_card = payload.get("clinician_compact_card") or {}
    review_plan = payload.get("review_plan_panel") or {}
    evidence = payload.get("evidence_panel") or {}
    response_guardrails = payload.get("response_guardrails_panel") or {}
    validated_writer = payload.get("validated_writer_panel") or {}
    care_guidance = payload.get("care_guidance_panel") or {}
    treatment_guidance = payload.get("treatment_guidance_panel") or {}
    llm_management_writer = payload.get("llm_management_writer_panel") or {}
    rag_evidence = payload.get("rag_evidence_panel") or {}
    patient = payload.get("patient_panel") or {}
    safety = payload.get("safety_badges") or {}
    final_display = stage3.get("final_display_payload") or {}
    clinical_report = final_display.get("clinical_support_report") or {}
    ai_pipeline_explanation = final_display.get("ai_pipeline_explanation") or {}

    image_id = header.get("image_id") or final_json.get("image_id") or "unknown_image"
    lines = [
        f"# AI X-ray Review Report: {image_id}",
        "",
        "> Research prototype output. This is not a standalone medical diagnosis.",
        "",
        *_clinical_support_report_section(clinical_report),
        *_pipeline_explanation_section(ai_pipeline_explanation),
        "<details>",
        "<summary>Technical audit details</summary>",
        "",
        "## Case Status",
        "",
        f"**{_safe(finding.get('headline'))}**",
        "",
        _safe(finding.get("subheadline")),
        "",
        f"Recommended review action: **{_safe(finding.get('primary_action'))}**",
        "",
        _pipeline_status_table(payload),
        "",
        *_clinician_compact_card_section(compact_card),
        "## Visual Evidence",
        "",
        _image_markdown(header.get("image_path")),
        "",
        f"- Display priority: `{_safe(visual.get('display_priority'))}`",
        f"- Primary bbox available: `{_safe((visual.get('primary_bbox') or {}).get('available'))}`",
        f"- Stage 2C heatmap available: `{_safe((visual.get('stage2c_heatmap') or {}).get('available'))}`",
        f"- Heatmap display policy: `{_safe(visual.get('stage2c_heatmap_display_policy'))}`",
        "",
        "Important: any heatmap is an explainability view, not a fracture boundary.",
        "",
        "## Model Support Scores",
        "",
        _confidence_table(payload.get("confidence_cards") or []),
        "",
        "Scores are technical model support values, not calibrated clinical probabilities.",
        "",
        "## Anatomy",
        "",
        _anatomy_table((payload.get("anatomy_panel") or {}).get("probability_ranking") or []),
        "",
        "## Clinician Summary",
        "",
        f"Headline: **{_safe(clinician.get('headline'))}**",
        "",
        "Confidence notes:",
        "",
        _bullet_list(clinician.get("confidence_notes") or []),
        "",
        "Actions to consider:",
        "",
        _bullet_list(clinician.get("actions_to_consider") or []),
        "",
        *_review_plan_section(review_plan),
        *_evidence_section(evidence),
        *_response_guardrails_section(response_guardrails),
        *_validated_writer_section(validated_writer),
        *_care_guidance_section(care_guidance),
        *_treatment_guidance_section(treatment_guidance),
        *_llm_management_writer_section(llm_management_writer),
        *_rag_evidence_pack_section(rag_evidence),
        "Limitations:",
        "",
        _bullet_list(clinician.get("limitations") or []),
        "",
        "## Patient-Friendly Note",
        "",
        f"**{_safe(patient.get('headline'))}**",
        "",
        _bullet_list(patient.get("plain_language_points") or []),
        "",
        "## Safety Badges",
        "",
        _bullet_list(
            [
                "Not standalone diagnosis" if safety.get("not_standalone_diagnosis") else None,
                "No patient-specific treatment advice" if safety.get("no_patient_specific_treatment") else None,
                "Human verification required" if safety.get("human_verification_required") else None,
                "Scores are not clinical probabilities" if safety.get("scores_are_not_clinical_probabilities") else None,
                "Stage 3 does not override Stage 1/2" if safety.get("stage3_does_not_override_stage1_stage2") else None,
            ]
        ),
        "",
        "## Display Rules",
        "",
        _bullet_list(payload.get("display_rules") or []),
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.json") if p.name != "summary.json")


def render_path(input_path: Path, output_path: Path) -> Dict[str, Any]:
    data = load_json(input_path)
    report = render_case_report_markdown(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "image_id": data.get("image_id"),
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 3 final JSON into deterministic Markdown case reports.")
    parser.add_argument("--input", required=True, help="Stage 3 JSON file or directory.")
    parser.add_argument("--output", required=True, help="Markdown output file or directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    inputs = find_inputs(input_root)
    single = len(inputs) == 1 and output_root.suffix.lower() == ".md"
    rows = []
    for src in inputs:
        dst = output_root if single else output_root / f"{src.stem}.md"
        rows.append(render_path(src, dst))
    summary = {
        "renderer": "stage3_report_renderer_v1",
        "input": str(input_root),
        "output": str(output_root),
        "processed_count": len(rows),
        "status": "PASS",
        "items": rows,
    }
    if not single:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
