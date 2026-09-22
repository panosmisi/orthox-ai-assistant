from __future__ import annotations

import re
from typing import Any, Dict, List

from .pipeline_explanation import build_ai_pipeline_explanation


BROAD_ANATOMY_DISPLAY_LABELS = {
    "lower limb",
    "hand/wrist region",
    "unspecified anatomy",
    "imaged region",
}

ACADEMIC_PROTOTYPE_DISCLAIMER = (
    "Academic prototype - not for clinical deployment. This demo is intended for "
    "research and evaluation of AI-assisted clinical-support reporting. It is not "
    "a diagnostic system and must not be used to make patient-care decisions."
)

SCORE_DISCLAIMER = "These are internal model support scores, not calibrated clinical probabilities."

ANATOMY_DISCLAIMER = (
    "Anatomy classifier output supports body-region classification only. It does not "
    "indicate whether a fracture is present or absent."
)

CANONICAL_AI_MUST_NOT_DECIDE = [
    "diagnosis",
    "fracture exclusion",
    "immobilisation",
    "support/protection",
    "weight-bearing or mobility status",
    "medication",
    "clearance or reassurance decisions",
    "return to work/sport/activity",
    "need for further imaging",
    "referral or specialist review",
    "follow-up interval",
    "procedure",
    "surgery",
]

STATUS_DISPLAY_LABELS = {
    "candidate_retained": "Localized AI review cue retained",
    "image_level_warning_without_bbox": "Image-level caution without localized bounding box",
    "no_high_confidence_candidate_retained": "No high-confidence AI review cue retained",
}


def _fmt_percent(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return None


def _clean_list(items: Any, limit: int = 5) -> List[Any]:
    if not isinstance(items, list):
        return []
    return [item for item in items if item][:limit]


def _sanitize_user_facing_text(text: str) -> str:
    replacements = {
        "discharge/clearance": "clearance or reassurance",
        "discharge / clearance": "clearance or reassurance",
        "discharge": "clearance",
        "negative or uncertain radiographs": "limited or uncertain radiographs",
        "negative radiographs": "limited radiographs",
        "release-from-care decisions": "clearance or reassurance decisions",
        "release from care": "clearance or reassurance",
        "clearance/reassurance decisions": "clearance or reassurance decisions",
        "clearance/reassurance": "clearance or reassurance",
        "severe or increasing pain/swelling": "severe, escalating, or disproportionate pain/swelling",
        "treatment target": "management target",
    }
    out = str(text)
    for old, new in replacements.items():
        out = re.sub(re.escape(old), new, out, flags=re.IGNORECASE)
    out = re.sub(
        r"\bvisible deformity\b(?!\s+or\s+dislocation\s+concern)",
        "visible deformity or dislocation concern",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _sanitize_user_facing(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize_user_facing_text(obj)
    if isinstance(obj, list):
        return [_sanitize_user_facing(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _sanitize_user_facing(value) for key, value in obj.items()}
    return obj


def _available_cards(cards: Any) -> List[Dict[str, Any]]:
    if not isinstance(cards, list):
        return []
    out = []
    for card in cards:
        if not isinstance(card, dict) or not card.get("available"):
            continue
        out.append(
            {
                "id": card.get("id"),
                "label": card.get("label"),
                "support_percent": card.get("percent"),
                "support_text": _fmt_percent(card.get("percent")),
                "support_level": card.get("support_level"),
                "support_level_label": card.get("support_level_label"),
                "score_kind": card.get("score_kind"),
                "display_context": card.get("display_context") or card.get("display_when"),
                "calibration_note": card.get("calibration_note"),
                "display_note": card.get("display_note"),
            }
        )
    return out


def _all_cards(cards: Any) -> List[Dict[str, Any]]:
    if not isinstance(cards, list):
        return []
    out = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        out.append(
            {
                "id": card.get("id"),
                "label": card.get("label"),
                "available": bool(card.get("available")),
                "support_percent": card.get("percent"),
                "support_text": _fmt_percent(card.get("percent")),
                "support_level": card.get("support_level"),
                "support_level_label": card.get("support_level_label"),
                "score_kind": card.get("score_kind"),
                "display_context": card.get("display_context") or card.get("display_when"),
                "calibration_note": card.get("calibration_note"),
                "display_note": card.get("display_note"),
            }
        )
    return out


def _anatomy_probabilities(anatomy_panel: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking = anatomy_panel.get("probability_ranking") or []
    out = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "label": item.get("label"),
                "display_label": item.get("display_label"),
                "support_percent": item.get("support_percent"),
                "support_text": _fmt_percent(item.get("support_percent")),
            }
        )
    return out


def _stable_variant_index(seed: str | None, family: str, count: int) -> int:
    if count <= 1:
        return 0
    key = f"{seed or ''}:{family}"
    return sum(ord(ch) for ch in key) % count


def _support_value(anatomy_panel: Dict[str, Any]) -> float | None:
    try:
        return float(anatomy_panel.get("selected_support_percent"))
    except (TypeError, ValueError):
        return None


def _anatomy_display_phrase(anatomy_panel: Dict[str, Any]) -> str:
    label = anatomy_panel.get("selected_display_label") or "the imaged region"
    support = _support_value(anatomy_panel)
    if label == "lower limb":
        base = "a broad lower-limb region"
    elif label == "hand/wrist region":
        base = "the hand/wrist region"
    elif label == "unspecified anatomy":
        base = "the imaged region"
    else:
        base = str(label)

    if support is None:
        return f"{base} with unavailable anatomy support"
    if support < 60.0:
        return f"{base} with low anatomy-label support"
    if support < 75.0:
        return f"{base} with borderline anatomy-label support"
    return base


def _candidate_count_from_text(text: str) -> str:
    match = re.search(r"retained\s+(\d+)\s+possible fracture candidate", text, re.IGNORECASE)
    if not match:
        return "one or more possible fracture candidates"
    count = int(match.group(1))
    if count == 1:
        return "1 possible fracture candidate"
    return f"{count} possible fracture candidates"


def _anatomy_display_badges(anatomy_panel: Dict[str, Any]) -> List[Dict[str, Any]]:
    badges: List[Dict[str, Any]] = []
    label = anatomy_panel.get("selected_display_label")
    support = anatomy_panel.get("selected_support_percent")
    try:
        support_f = float(support)
    except (TypeError, ValueError):
        support_f = None

    if label in BROAD_ANATOMY_DISPLAY_LABELS:
        badges.append(
            {
                "id": "broad_anatomy_label",
                "label": "Broad anatomical region",
                "severity": "info",
                "message": "The anatomy classifier selected a broad region rather than a precise site.",
            }
        )
    if support_f is None:
        badges.append(
            {
                "id": "missing_anatomy_support",
                "label": "Anatomy support unavailable",
                "severity": "caution",
                "message": "No anatomy support score was available for this case.",
            }
        )
    elif support_f < 60.0:
        badges.append(
            {
                "id": "low_anatomy_support",
                "label": "Low anatomy confidence",
                "severity": "caution",
                "message": "The anatomy label has low model support and should be interpreted cautiously.",
            }
        )
    elif support_f < 75.0:
        badges.append(
            {
                "id": "borderline_anatomy_support",
                "label": "Borderline anatomy confidence",
                "severity": "info",
                "message": "The anatomy label is usable but not high-confidence.",
            }
        )
    return badges


def _status_tone(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return "possible_fracture_candidate"
    if state_id == "image_level_warning_without_bbox":
        return "caution_without_localized_box"
    return "no_high_confidence_candidate_with_review_required"


def _visual_instruction(visual_panel: Dict[str, Any]) -> Dict[str, Any]:
    heatmap = visual_panel.get("stage2c_heatmap") or {}
    primary_bbox = visual_panel.get("primary_bbox") or {}
    display_priority = visual_panel.get("display_priority")
    return {
        "display_priority": display_priority,
        "show_primary_bbox": bool(visual_panel.get("show_primary_bbox")),
        "primary_bbox": primary_bbox if primary_bbox.get("available") else {},
        "heatmap_available": bool(heatmap.get("available")),
        "show_heatmap_by_default": bool(visual_panel.get("show_stage2c_heatmap_by_default")),
        "heatmap_button_label": "View caution heatmap" if heatmap.get("available") else None,
        "heatmap_warning": (
            "Heatmap is an explainability aid, not a fracture boundary."
            if heatmap.get("available")
            else None
        ),
    }


def _state_title(state_id: str | None, anatomy_label: str | None) -> str:
    anatomy = anatomy_label or "the imaged region"
    if state_id == "candidate_retained":
        return f"Localized AI review cue retained around {anatomy}"
    if state_id == "image_level_warning_without_bbox":
        return "Image-level caution signal without localized bounding box"
    return "No high-confidence AI review cue retained"


def _state_explanation(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return (
            "A localized candidate was retained from the locked Stage 1/2 outputs. "
            "This is a screening cue for clinician review, not a diagnosis."
        )
    if state_id == "image_level_warning_without_bbox":
        return (
            "No localized bounding box was retained. The image-level caution layer "
            "raised a whole-image warning. This does not identify a fracture boundary."
        )
    return (
        "No high-confidence localized AI candidate was retained. This does not exclude "
        "fracture or other injury."
    )


def _why_status_assigned(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return (
            "This status was assigned because a localized candidate was retained after "
            "the detector/verifier and artifact-suppression stages."
        )
    if state_id == "image_level_warning_without_bbox":
        return (
            "This status was assigned because no localized candidate was retained, but "
            "the image-level caution layer raised a whole-image review signal."
        )
    return (
        "This status was assigned because no high-confidence localized candidate was retained "
        "and no stronger image-level warning was available. This does not exclude fracture or other injury."
    )


def _suggested_review_focus(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return (
            "Review the marked region, the complete radiographic study, relevant adjacent "
            "anatomy, symptoms, examination findings, and formal radiology review."
        )
    if state_id == "image_level_warning_without_bbox":
        return (
            "Review the complete radiographic study and clinical context. Any heatmap is "
            "optional visual support and must not be interpreted as a fracture contour, "
            "management target, or bounding box."
        )
    return (
        "Review the complete radiographic study with symptoms, mechanism, examination "
        "findings, and formal radiology review. Do not use absence of a retained AI "
        "candidate to exclude injury."
    )


def _guidance_level(
    state_id: str | None,
    anatomy_panel: Dict[str, Any],
    management_panel: Dict[str, Any],
    evidence_panel: Dict[str, Any],
) -> Dict[str, Any]:
    has_sources = bool(evidence_panel.get("displayed_sources"))
    parent_fallback = anatomy_panel.get("output_type") == "parent_fallback"
    if state_id == "candidate_retained" and has_sources and not parent_fallback:
        return {
            "level": "level_3",
            "label": "Evidence-linked clinician context - not patient-specific treatment.",
            "patient_specific_treatment_plan_allowed": False,
        }
    if state_id == "candidate_retained":
        return {
            "level": "level_2",
            "label": "Anatomy-aware clinician review context - clinical localization required.",
            "patient_specific_treatment_plan_allowed": False,
        }
    if state_id == "image_level_warning_without_bbox":
        return {
            "level": "level_1",
            "label": "Limited AI contribution - image-level clinical review context.",
            "patient_specific_treatment_plan_allowed": False,
        }
    return {
        "level": "level_1",
        "label": "Limited AI contribution - clinical context required.",
        "patient_specific_treatment_plan_allowed": False,
    }


def _evidence_mode(pubmed_retrieval: Dict[str, Any], state_id: str | None) -> Dict[str, Any]:
    if pubmed_retrieval.get("used_for_management_context"):
        label = "PubMed-supported context"
        description = (
            "PubMed/RAG support was used for clinician-facing management context because "
            "a retained localized AI review cue met the evidence-trigger criteria."
        )
    elif state_id == "no_high_confidence_candidate_retained":
        label = "Limited clinical review context; PubMed not triggered"
        description = (
            "PubMed/RAG support was enabled, but retrieval was not used because no localized "
            "candidate required fracture-management literature context."
        )
    else:
        label = "Curated guideline metadata + local safety rules"
        description = (
            "PubMed/RAG support was enabled, but retrieval is used only when evidence-trigger "
            "criteria are met. This case uses curated guideline metadata and local safety rules."
        )
    return {
        "label": label,
        "description": description,
        "pubmed_enabled_but_triggered_only_when_needed": True,
    }


def _visual_evidence_report(state_id: str | None, visual_panel: Dict[str, Any]) -> Dict[str, Any]:
    heatmap = visual_panel.get("stage2c_heatmap") or {}
    primary_bbox = visual_panel.get("primary_bbox") or {}
    bbox_available = bool(primary_bbox.get("available"))
    if bbox_available:
        note = (
            "Visual evidence: retained bounding box from Stage 1/2. The bounding box is "
            "a review cue, not a fracture boundary or management target."
        )
    else:
        note = "Visual evidence: no localized bounding box retained."
    heatmap_note = None
    if heatmap.get("available"):
        heatmap_note = (
            "Heatmap, if displayed, is optional visual support and must not be interpreted "
            "as a fracture contour, management target, or bounding box."
        )
    return {
        "bbox_available": bbox_available,
        "bbox_displayed": bool(visual_panel.get("show_primary_bbox")),
        "bbox_modified_by_stage3": False,
        "heatmap_available": bool(heatmap.get("available")),
        "heatmap_display_default": bool(visual_panel.get("show_stage2c_heatmap_by_default")),
        "visual_priority": visual_panel.get("display_priority"),
        "visual_evidence_note": note,
        "heatmap_safety_note": heatmap_note,
    }


def _model_evidence_report(app_payload: Dict[str, Any]) -> Dict[str, Any]:
    cards = app_payload.get("confidence_cards") or []

    def find_card(card_id: str) -> Dict[str, Any]:
        for card in cards:
            if isinstance(card, dict) and card.get("id") == card_id:
                return card
        return {}

    candidate = find_card("fracture_candidate_support")
    localization = find_card("localization_evidence")
    stage2c = find_card("stage2c_caution_support")
    return {
        "candidate_support_score": candidate.get("percent"),
        "candidate_support_level": candidate.get("support_level") or "unavailable",
        "candidate_support_label": candidate.get("support_level_label"),
        "candidate_display_note": candidate.get("display_note"),
        "localization_display_note": localization.get("display_note"),
        "localization_display_context": localization.get("display_context") or localization.get("display_when"),
        "image_level_support_score": stage2c.get("percent"),
        "image_level_support_level": stage2c.get("support_level") or "unavailable",
        "image_level_display_note": stage2c.get("display_note"),
        "image_level_display_context": stage2c.get("display_context") or stage2c.get("display_when"),
        "score_disclaimer": SCORE_DISCLAIMER,
    }


def _anatomy_report(anatomy_panel: Dict[str, Any]) -> Dict[str, Any]:
    output_type = anatomy_panel.get("output_type")
    fine_uncertain = output_type == "parent_fallback" or (
        anatomy_panel.get("selected_display_label") in BROAD_ANATOMY_DISPLAY_LABELS
    )
    uncertainty_note = None
    if fine_uncertain:
        uncertainty_note = (
            "Fine anatomical localization was uncertain; the report uses a broader "
            "anatomical region until clinical review localizes the concern."
        )
    return {
        "selected_label": anatomy_panel.get("selected_label"),
        "selected_display_label": anatomy_panel.get("selected_display_label"),
        "selected_label_confidence": anatomy_panel.get("selected_support_percent"),
        "parent_label": anatomy_panel.get("parent_label"),
        "parent_display_label": anatomy_panel.get("parent_display_label"),
        "output_type": output_type,
        "fine_anatomy_uncertain": fine_uncertain,
        "uncertainty_note": uncertainty_note,
        "top_probabilities": _anatomy_probabilities(anatomy_panel),
        "anatomy_disclaimer": ANATOMY_DISCLAIMER,
    }


def _clinical_inputs_required(management_panel: Dict[str, Any]) -> List[str]:
    configured = management_panel.get("clinical_inputs_required") or []
    if configured:
        return _clean_list(configured, 12)
    return [
        "mechanism of injury",
        "exact pain location",
        "focal bony tenderness",
        "ability to use or bear weight when relevant",
        "range of motion when relevant",
        "neurovascular examination",
        "skin condition / open wound status",
        "pain severity and progression",
        "swelling progression",
        "complete radiographic series / views",
        "formal radiology review",
        "prior imaging or comparison study when available",
    ]


def _support_type_from_ids(source_ids: List[str]) -> str:
    joined = " ".join(source_ids).lower()
    if "pubmed" in joined or "pmid" in joined:
        return "pubmed_review"
    if "guideline" in joined:
        return "guideline"
    if "boast" in joined or "boa" in joined:
        return "professional_standard"
    if "safety" in joined or "stage3" in joined:
        return "local_safety_rule"
    return "source_linked_context"


def _management_considerations(management_panel: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = (
        management_panel.get("case_management_suggestions", {}).get("suggestions")
        or management_panel.get("management_context")
        or []
    )
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items[:6], start=1):
        if not isinstance(item, dict):
            text = str(item)
            source_ids = ["stage3_curated_safety_context_v0"]
            limitation = "Does not provide patient-specific treatment advice."
            category = "clinician_review_context"
        else:
            text = item.get("text") or item.get("summary") or item.get("rationale") or ""
            source_ids = list(item.get("source_ids") or item.get("support_ids") or [])
            if not source_ids:
                source_id = item.get("source_id") or item.get("safety_rule_id")
                source_ids = [source_id] if source_id else ["stage3_curated_safety_context_v0"]
            limitation = item.get("prohibited_interpretation") or item.get("limitation") or (
                "Does not provide patient-specific treatment advice."
            )
            category = item.get("category") or item.get("item_type") or "clinician_review_context"
        out.append(
            {
                "item_id": f"management_consideration_{idx:02d}",
                "category": category,
                "text": text,
                "support_ids": source_ids,
                "support_type": _support_type_from_ids(source_ids),
                "limitation": limitation,
            }
        )
    return out


def _source_cards(evidence_panel: Dict[str, Any], fine_anatomy_uncertain: bool = False) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for source in (evidence_panel.get("displayed_sources") or [])[:8]:
        if not isinstance(source, dict):
            continue
        quality = source.get("pubmed_quality") or {}
        source_id = source.get("source_id") or source.get("id")
        source_type = source.get("source_type")
        pmid = source.get("pmid")
        if not pmid and source_type == "pubmed" and isinstance(source_id, str) and source_id.startswith("pmid_"):
            pmid = source_id.replace("pmid_", "", 1)
        publisher = (
            source.get("publisher_or_journal")
            or source.get("journal")
            or source.get("organization")
            or source.get("publisher")
        )
        evidence_level = source.get("evidence_level") or quality.get("quality_tier")
        supports = _source_used_for(source)
        if not supports:
            if source_type == "pubmed":
                supports = ["PubMed management-background evidence for clinician review."]
            elif source_type == "guideline_metadata":
                supports = ["Curated guideline metadata for clinician review context and local pathway awareness."]
            else:
                supports = ["Local safety and reporting-boundary context."]
        limitations = _source_limitations(source)
        if not limitations:
            limitations = [
                "Does not interpret this radiograph and does not provide patient-specific treatment advice."
            ]
        if fine_anatomy_uncertain:
            limitations = list(limitations) + [
                "Fine anatomical localization is uncertain; site-specific source relevance depends on clinician localization."
            ]
        cards.append(
            {
                "id": source_id,
                "title": source.get("title") or source_id,
                "source_type": source_type,
                "evidence_level": evidence_level,
                "publisher_or_journal": publisher,
                "year": source.get("year"),
                "pmid": pmid,
                "doi": source.get("doi"),
                "url": source.get("url"),
                "used_for": supports[:3],
                "limitations": limitations[:3],
            }
        )
    return cards


def _pubmed_retrieval_statement(
    stage3_block: Dict[str, Any] | None,
    state_id: str | None,
) -> Dict[str, Any]:
    stage3_block = stage3_block or {}
    retrievals = stage3_block.get("retrieval_statuses") or []
    pubmed = next(
        (
            item
            for item in retrievals
            if isinstance(item, dict) and item.get("retriever") == "pubmed"
        ),
        {},
    )
    controls = stage3_block.get("evidence_controls") or {}
    policy = (stage3_block.get("source_relevance_audit") or {}).get("policy") or {}
    trigger_reason = (
        pubmed.get("trigger_reason")
        or controls.get("pubmed_trigger_reason")
        or policy.get("pubmed_trigger_reason")
    )
    trigger_policy = (
        controls.get("pubmed_trigger_policy")
        or policy.get("pubmed_trigger_policy")
        or "unknown"
    )
    status = pubmed.get("status") or "unavailable"
    source_count = int(pubmed.get("source_count") or 0)
    queries_used = [
        str(query)
        for query in (pubmed.get("queries_used") or [])
        if str(query).strip()
    ][:5]
    used = (
        state_id == "candidate_retained"
        and trigger_reason == "retained_candidate_present"
        and status == "available"
        and source_count > 0
    )
    if used:
        statement = (
            "PubMed RAG was run for management-context evidence because a retained "
            "localized AI review cue was present. PubMed sources are used only for "
            "clinician-facing management context, not diagnosis, fracture confirmation, "
            "or patient-specific treatment."
        )
    elif trigger_reason == "retained_candidate_present":
        statement = (
            "PubMed retrieval was attempted for management context because a retained "
            "localized AI review cue was present, but PubMed evidence was unavailable "
            "or insufficient. The report therefore falls back to curated guideline "
            "metadata and local safety rules."
        )
    elif status == "skipped":
        statement = (
            "PubMed RAG was not run for this case because no retained localized AI "
            "review cue required fracture-management literature context. The report "
            "uses limited clinical review context, curated guideline metadata, and "
            "local safety rules."
        )
    else:
        statement = (
            "PubMed RAG was not available for this case. The report uses curated "
            "guideline metadata and local safety rules."
        )
    return {
        "retriever": "pubmed",
        "trigger_policy": trigger_policy,
        "trigger_reason": trigger_reason,
        "status": status,
        "source_count": source_count,
        "queries_used": queries_used,
        "used_for_management_context": used,
        "scope": "management_context_only_not_diagnosis",
        "statement": statement,
    }


def _source_used_for(source: Dict[str, Any]) -> List[str]:
    source_id = str(source.get("source_id") or source.get("id") or "").lower()
    title = str(source.get("title") or "").lower()
    source_type = source.get("source_type")
    if "nice_ng38" in source_id or "non-complex" in title:
        return [
            "General non-complex fracture assessment, emergency-department pathway context, pain assessment, documentation, and clinician-led management context."
        ]
    if "nice_ng37" in source_id or "complex" in title:
        return [
            "Complex fracture, open injury, neurovascular concern, escalation, and trauma pathway context."
        ]
    if "boast_ankle" in source_id or "ankle fracture" in title:
        return [
            "Ankle fracture pathway context when clinician review localizes concern to the ankle or malleolar region."
        ]
    if "ottawa_knee" in source_id or "ottawa knee" in title:
        return [
            "Knee injury imaging-rule context for clinician review of mechanism, focal tenderness, weight-bearing status, and need for formal imaging-pathway assessment."
        ]
    if "ottawa_ankle" in source_id or "ottawa ankle" in title or "ottawa foot" in title:
        return [
            "Ankle/foot injury imaging-rule context for clinician review of malleolar, navicular, fifth-metatarsal tenderness, weight-bearing status, and local imaging-pathway assessment."
        ]
    if "boast" in source_id:
        return ["Orthopaedic trauma standards and referral/pathway context."]
    if "acr" in source_id or "hand" in title or "wrist" in title:
        return [
            "Acute hand/wrist trauma imaging and review context; supports correlation of suspected injury location with appropriate views and clinician assessment."
        ]
    if source_type == "local_curated_project_rule" or "safety" in source_id:
        return [
            "Stage 3 reporting boundaries, non-diagnostic language, score disclaimers, and AI-not-decide constraints."
        ]
    if source_type == "pubmed":
        return [
            "PubMed management-background evidence for clinician review; used only as source-linked context, not diagnosis or patient-specific treatment."
        ]
    return list(source.get("supports") or [])


def _source_limitations(source: Dict[str, Any]) -> List[str]:
    source_id = str(source.get("source_id") or source.get("id") or "").lower()
    title = str(source.get("title") or "").lower()
    source_type = source.get("source_type")
    if "boast_ankle" in source_id or "ankle fracture" in title:
        return ["Relevant as site-specific context only when symptoms and imaging localize to the ankle."]
    if "ottawa_knee" in source_id or "ottawa knee" in title:
        return ["Supports clinician-led imaging-rule context only; does not interpret this radiograph or decide management."]
    if "ottawa_ankle" in source_id or "ottawa ankle" in title or "ottawa foot" in title:
        return ["Supports clinician-led imaging-rule context only; does not interpret this radiograph or decide management."]
    if "boast" in source_id:
        return ["General professional-standard context only; not a patient-specific management decision."]
    if "acr" in source_id or "hand" in title or "wrist" in title:
        return ["Does not automatically determine further imaging or treatment for this patient."]
    if source_type == "local_curated_project_rule" or "safety" in source_id:
        return ["Project safety rule, not external clinical evidence."]
    return list(source.get("limitations") or [])


def _clinical_support_report(
    app_payload: Dict[str, Any],
    text: Dict[str, str],
    stage3_block: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    case_header = app_payload.get("case_header") or {}
    finding = case_header.get("finding_state") or {}
    state_id = finding.get("state_id")
    visual_panel = app_payload.get("visual_panel") or {}
    anatomy_panel = app_payload.get("anatomy_panel") or {}
    evidence_panel = app_payload.get("evidence_panel") or {}
    management_panel = app_payload.get("management_guidance_panel") or {}
    care_panel = app_payload.get("care_guidance_panel") or {}

    anatomy = _anatomy_report(anatomy_panel)
    guidance_level = _guidance_level(state_id, anatomy_panel, management_panel, evidence_panel)
    status_title = _state_title(state_id, anatomy.get("selected_display_label") or anatomy.get("parent_display_label"))

    report = {
        "version": "stage3_clinical_support_report_v1",
        "report_title": "Stage 3 Clinical-Support Report",
        "academic_prototype": True,
        "academic_prototype_disclaimer": ACADEMIC_PROTOTYPE_DISCLAIMER,
        "case_id": case_header.get("image_id"),
        "case_status": state_id,
        "case_status_display": STATUS_DISPLAY_LABELS.get(state_id, "AI review status"),
        "status_title": status_title,
        "status_explanation": _state_explanation(state_id),
        "why_this_status_was_assigned": _why_status_assigned(state_id),
        "clinical_support_summary": {
            "title": "AI clinical-support summary",
            "suggested_review_focus": _suggested_review_focus(state_id),
            "interpretation": text.get("clinician_summary"),
            "required_human_review": [
                "Human clinician review is mandatory.",
                "Formal radiology review remains outside the scope of this AI report.",
                "Stage 3 does not alter Stage 1/2 outputs, bounding boxes, or scores.",
            ],
        },
        "visual_evidence": _visual_evidence_report(state_id, visual_panel),
        "model_evidence": _model_evidence_report(app_payload),
        "anatomy": anatomy,
        "clinical_review_guidance": {
            "review_focus": _clean_list(care_panel.get("review_focus"), 6),
            "clinical_inputs_required": _clinical_inputs_required(management_panel),
            "red_flags": _clean_list(management_panel.get("red_flags") or care_panel.get("when_to_escalate_review"), 8),
            "limitations": _clean_list(management_panel.get("limitations"), 6),
        },
        "management_context": {
            "title": "Evidence-linked management context for clinician review",
            "subtitle": "Not a treatment plan. Not patient-specific medical advice.",
            "is_treatment_plan": False,
            "guidance_level": guidance_level["level"],
            "guidance_level_label": guidance_level["label"],
            "why_shown": management_panel.get("why_this_section_is_shown") or _state_explanation(state_id),
            "parent_fallback_note": (
                "Fine anatomical localization was uncertain. Do not select a site-specific management "
                "pathway until symptoms, full radiographic review, and clinician assessment localize "
                "the concern."
                if anatomy["fine_anatomy_uncertain"]
                else None
            ),
            "management_considerations": _management_considerations(management_panel),
            "ai_must_not_decide": CANONICAL_AI_MUST_NOT_DECIDE,
            "safety_boundary": (
                "This is not a diagnosis or treatment plan. Human clinical verification is required. "
                "The AI must not decide diagnosis, fracture exclusion, immobilisation, support/protection, "
                "weight-bearing or mobility status, medication, clearance or reassurance decisions, return "
                "to work/sport/activity, need for further imaging, referral or specialist review, follow-up "
                "interval, procedure, or surgery."
            ),
        },
        "source_linked_evidence": {
            "evidence_status": evidence_panel.get("evidence_status"),
            "pubmed_retrieval": _pubmed_retrieval_statement(stage3_block, state_id),
            "evidence_mode": _evidence_mode(_pubmed_retrieval_statement(stage3_block, state_id), state_id),
            "source_cards": _source_cards(evidence_panel, fine_anatomy_uncertain=anatomy["fine_anatomy_uncertain"]),
            "displayed_sources": _clean_list(evidence_panel.get("displayed_sources"), 8),
            "displayed_allowed_claims": _clean_list(evidence_panel.get("displayed_allowed_claims"), 8),
        },
        "compact_clinical_support_view": {
            "display_by_default": True,
            "case_status": STATUS_DISPLAY_LABELS.get(state_id, "AI review status"),
            "why_this_status_was_assigned": _why_status_assigned(state_id),
            "summary": text.get("clinician_summary"),
            "visual_evidence": _visual_evidence_report(state_id, visual_panel).get("visual_evidence_note"),
            "model_evidence": _all_cards(app_payload.get("confidence_cards")),
            "anatomy_context": anatomy,
            "clinical_inputs_required": _clinical_inputs_required(management_panel),
            "red_flags": _clean_list(management_panel.get("red_flags") or care_panel.get("when_to_escalate_review"), 8),
            "evidence_mode": _evidence_mode(_pubmed_retrieval_statement(stage3_block, state_id), state_id),
            "source_linked_evidence_summary": {
                "source_count": len(_source_cards(evidence_panel, fine_anatomy_uncertain=anatomy["fine_anatomy_uncertain"])),
                "pubmed_source_count": _pubmed_retrieval_statement(stage3_block, state_id).get("source_count"),
            },
            "safety_boundary": (
                "This is not a diagnosis or treatment plan. Human clinical verification is required. "
                "The AI must not decide diagnosis, fracture exclusion, immobilisation, support/protection, "
                "weight-bearing or mobility status, medication, clearance or reassurance decisions, return "
                "to work/sport/activity, need for further imaging, referral or specialist review, follow-up "
                "interval, procedure, or surgery."
            ),
        },
        "pipeline_explanation": {
            "enabled": True,
            "plain_language_summary": (
                "Stage 3 assembled the locked detector, verifier, artifact, anatomy, and caution-layer "
                "outputs into a clinician-support report. It did not run new detection or change any "
                "bounding box."
            ),
            "stage_summaries": [
                "Stage 1 - Candidate detection",
                "Stage 2A - Candidate verification",
                "Stage 2A.5 - Artifact suppression",
                "Stage 2B - Anatomy classification",
                "Stage 2C - Image-level caution",
                "Stage 3 - Clinical-support report assembly",
            ],
            "technical_audit_available": True,
        },
        "technical_audit": {
            "display_by_default": False,
            "contains_internal_pipeline_terms": True,
        },
    }
    return _sanitize_user_facing(report)


def _display_text_from_writer(app_payload: Dict[str, Any]) -> Dict[str, str]:
    case_header = app_payload.get("case_header") or {}
    finding = case_header.get("finding_state") or {}
    anatomy_panel = app_payload.get("anatomy_panel") or {}
    image_id = case_header.get("image_id")
    state_id = finding.get("state_id")
    anatomy_phrase = _anatomy_display_phrase(anatomy_panel)
    writer = app_payload.get("validated_writer_panel") or {}
    sections = writer.get("sections") or {}
    raw_clinician = ""
    raw_patient = ""
    raw_footer = ""
    if sections:
        raw_clinician = _remove_leading_safety_boilerplate(sections.get("clinician_summary") or "")
        raw_patient = _remove_leading_safety_boilerplate(sections.get("patient_note") or "")
        raw_footer = sections.get("safety_footer") or ""

    card = app_payload.get("clinician_compact_card") or {}
    patient = app_payload.get("patient_panel") or {}
    patient_points = patient.get("plain_language_points") or []
    if not raw_clinician:
        raw_clinician = " ".join(
            part
            for part in [
                card.get("headline"),
                card.get("primary_action"),
            ]
            if part
        )
    if not raw_patient:
        raw_patient = " ".join(str(p) for p in patient_points[:2])
    if not raw_footer:
        raw_footer = (
            "This is not a standalone diagnosis. Human verification is required. "
            "Model scores are technical support signals, not clinical probabilities."
        )

    return _polished_display_sections(
        image_id=image_id,
        state_id=state_id,
        anatomy_phrase=anatomy_phrase,
        raw_clinician=raw_clinician,
        raw_patient=raw_patient,
        raw_footer=raw_footer,
    )


def _polished_display_sections(
    image_id: str | None,
    state_id: str | None,
    anatomy_phrase: str,
    raw_clinician: str,
    raw_patient: str,
    raw_footer: str,
) -> Dict[str, str]:
    if state_id == "candidate_retained":
        count_text = _candidate_count_from_text(raw_clinician)
        clinician_variants = [
            f"The screening pipeline retained {count_text} around {anatomy_phrase}. Review should start with the marked region and then continue on the original full radiograph.",
            f"A localized fracture candidate was retained around {anatomy_phrase}. The marked area should be checked first, but the full X-ray still needs clinician review.",
            f"The model kept a marked region of concern around {anatomy_phrase}. Treat it as a screening signal that requires visual confirmation on the complete radiograph.",
            f"Stage 1/2 retained {count_text} near {anatomy_phrase}. The box is a review target, not a diagnostic boundary.",
        ]
        patient_variants = [
            f"The AI marked a possible area of concern around {anatomy_phrase}. A qualified clinician should review both the marked area and the full image.",
            f"The system highlighted an area around {anatomy_phrase} for clinician review. This does not confirm a fracture by itself.",
            f"A marked region was kept by the AI around {anatomy_phrase}. A clinician needs to compare it with the full X-ray and symptoms.",
            f"The image includes an AI-marked area near {anatomy_phrase}. It should be checked by a qualified clinician before any conclusion is made.",
        ]
        family = "candidate"
    elif state_id == "image_level_warning_without_bbox":
        clinician_variants = [
            f"The detector did not retain a precise box, but the image-level safety layer raised a caution signal around {anatomy_phrase}. Review should focus on the whole radiograph because the warning does not provide a fracture boundary.",
            f"No localized box was kept, but the image-level check still flagged {anatomy_phrase} for review. Use the original X-ray, and optionally the heatmap, as a broad attention guide.",
            f"The system raised a caution signal around {anatomy_phrase} without producing a retained bounding box. This should trigger full-image review rather than box-based interpretation.",
            f"An image-level warning was produced for {anatomy_phrase}, but no precise candidate box survived the pipeline. The heatmap may guide attention, but it is not a localization result.",
        ]
        patient_variants = [
            f"The AI raised a caution signal around {anatomy_phrase}, but it did not keep a precise marked box. A qualified clinician should review the full image.",
            f"The system did not mark one exact spot, but it still recommends clinician review around {anatomy_phrase}.",
            f"The AI found a broad caution signal near {anatomy_phrase}. This is not a diagnosis and should be reviewed on the full X-ray.",
            f"No exact box was kept, but the image-level check suggests that {anatomy_phrase} deserves careful human review.",
        ]
        family = "stage2c"
    else:
        clinician_variants = [
            f"The pipeline did not retain a high-confidence marked region around {anatomy_phrase}. This does not exclude subtle or occult injury, so clinical context and image quality remain important.",
            f"No high-confidence fracture candidate was kept around {anatomy_phrase}. The result should be treated as limited screening output, not as proof that injury is absent.",
            f"The AI did not keep a localized region of concern around {anatomy_phrase}. Subtle findings can still be missed, so symptoms and clinician judgement remain essential.",
            f"No retained box is available for {anatomy_phrase}. If symptoms or examination findings remain concerning, the X-ray still needs careful human review.",
        ]
        patient_variants = [
            f"The AI did not keep a high-confidence marked region around {anatomy_phrase}. This does not exclude subtle injury, so symptoms and clinician review still matter.",
            f"The system did not highlight a strong area of concern around {anatomy_phrase}. This should not be treated as a final clinical conclusion.",
            f"No strong AI-marked region was kept near {anatomy_phrase}. A clinician should still interpret the image together with symptoms.",
            f"The AI did not retain a strong marked area around {anatomy_phrase}. Human review remains important if pain, swelling, trauma, or clinical concern exists.",
        ]
        family = "no_high_confidence_candidate"

    idx = _stable_variant_index(image_id, family, len(clinician_variants))
    return {
        "clinician_summary": clinician_variants[idx],
        "patient_note": patient_variants[idx],
        "safety_footer": raw_footer
        or "This is not a standalone diagnosis. Human verification is required. Model scores are not clinical probabilities.",
    }


def _fallback_display_text_from_writer(app_payload: Dict[str, Any]) -> Dict[str, str]:
    card = app_payload.get("clinician_compact_card") or {}
    patient = app_payload.get("patient_panel") or {}
    patient_points = patient.get("plain_language_points") or []
    return {
        "clinician_summary": " ".join(
            part
            for part in [
                card.get("headline"),
                card.get("primary_action"),
            ]
            if part
        ),
        "patient_note": " ".join(str(p) for p in patient_points[:2]),
        "safety_footer": (
            "This is not a standalone diagnosis. Human verification is required. "
            "Model scores are technical support signals, not clinical probabilities."
        ),
    }


def _remove_leading_safety_boilerplate(text: str) -> str:
    text = " ".join(str(text or "").split())
    prefixes = [
        "This is not a standalone diagnosis. Human clinical verification is required. Model scores are not clinical probabilities.",
        "This is not a standalone diagnosis. Human verification is required. The AI scores are not clinical probabilities.",
        "This output is not a standalone diagnosis. Human verification is required. The AI scores are not clinical probabilities.",
        "This AI output is not a standalone diagnosis.",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text or "Human clinical verification is required."


def build_final_display_payload(stage3_block: Dict[str, Any]) -> Dict[str, Any]:
    """Build the final app-facing display object.

    This layer is deliberately presentation-only. It uses already validated
    Stage 3 outputs and does not create new medical claims.
    """
    app_payload = stage3_block.get("app_payload") or {}
    case_header = app_payload.get("case_header") or {}
    finding = case_header.get("finding_state") or {}
    visual_panel = app_payload.get("visual_panel") or {}
    anatomy_panel = app_payload.get("anatomy_panel") or {}
    evidence_panel = app_payload.get("evidence_panel") or {}
    treatment_panel = app_payload.get("treatment_guidance_panel") or {}
    management_panel = app_payload.get("management_guidance_panel") or {}
    llm_management_panel = app_payload.get("llm_management_writer_panel") or {}
    care_panel = app_payload.get("care_guidance_panel") or {}
    guardrails = app_payload.get("response_guardrails_panel") or {}
    validated_writer = app_payload.get("validated_writer_panel") or {}
    safety_badges = app_payload.get("safety_badges") or {}

    state_id = finding.get("state_id")
    text = _display_text_from_writer(app_payload)
    anatomy_badges = _anatomy_display_badges(anatomy_panel)
    clinical_support_report = _clinical_support_report(app_payload, text, stage3_block)
    ai_pipeline_explanation = build_ai_pipeline_explanation(stage3_block)

    return {
        "version": "stage3_final_display_payload_v1",
        "purpose": "Stable final display contract for the future prototype app.",
        "presentation_only": True,
        "creates_new_medical_claims": False,
        "clinical_support_report": clinical_support_report,
        "ai_pipeline_explanation": ai_pipeline_explanation,
        "case": {
            "image_id": case_header.get("image_id"),
            "image_path": case_header.get("image_path"),
            "status_tone": _status_tone(state_id),
            "state_id": state_id,
            "headline": finding.get("headline"),
            "subheadline": finding.get("subheadline"),
            "primary_action": finding.get("primary_action"),
        },
        "visual": _visual_instruction(visual_panel),
        "text": text,
        "confidence": {
            "policy": (
                "Scores are internal model support signals. They are not calibrated clinical "
                "probabilities, do not diagnose fracture, and must not be used to exclude injury."
            ),
            "cards": _available_cards(app_payload.get("confidence_cards")),
        },
        "anatomy": {
            "selected_label": anatomy_panel.get("selected_label"),
            "selected_display_label": anatomy_panel.get("selected_display_label"),
            "selected_support_percent": anatomy_panel.get("selected_support_percent"),
            "selected_support_text": _fmt_percent(anatomy_panel.get("selected_support_percent")),
            "parent_label": anatomy_panel.get("parent_label"),
            "parent_display_label": anatomy_panel.get("parent_display_label"),
            "output_type": anatomy_panel.get("output_type"),
            "top_probabilities": _anatomy_probabilities(anatomy_panel),
            "display_badges": anatomy_badges,
        },
        "display_badges": anatomy_badges,
        "review": {
            "top_focus": _clean_list(care_panel.get("review_focus"), 2),
            "top_radiograph_checks": _clean_list(care_panel.get("radiograph_review_prompts"), 2),
            "top_context_questions": _clean_list(care_panel.get("clinical_context_prompts"), 2),
            "uncertainty_notes": _clean_list(care_panel.get("uncertainty_notes"), 1),
            "do_not_infer": _clean_list(care_panel.get("do_not_infer"), 2),
        },
        "management_guidance": {
            "enabled": bool(management_panel.get("enabled")),
            "label": management_panel.get("label") or treatment_panel.get("app_label"),
            "button_label": management_panel.get("button_label") or treatment_panel.get("button_label"),
            "panel_title": management_panel.get("panel_title") or treatment_panel.get("panel_title"),
            "panel_badge": management_panel.get("panel_badge") or treatment_panel.get("panel_badge"),
            "panel_warning": management_panel.get("panel_warning") or treatment_panel.get("panel_warning"),
            "audience": management_panel.get("audience") or treatment_panel.get("audience"),
            "is_treatment_plan": management_panel.get("is_treatment_plan") is True,
            "safety_disclaimer": management_panel.get("safety_disclaimer") or treatment_panel.get("safety_disclaimer"),
            "case_scenario": management_panel.get("case_scenario") or treatment_panel.get("management_scenario") or {},
            "guidance_strength": management_panel.get("guidance_strength") or treatment_panel.get("management_guidance_strength"),
            "why_this_section_is_shown": management_panel.get("why_this_section_is_shown"),
            "management_direction": management_panel.get("management_direction") or {},
            "case_management_suggestions": management_panel.get("case_management_suggestions") or {},
            "anatomy_specific_management": management_panel.get("anatomy_specific_management")
            or treatment_panel.get("anatomy_specific_management_v3")
            or {},
            "management_context": _clean_list(management_panel.get("management_context"), 3),
            "guidance_traceability": management_panel.get("guidance_traceability") or {},
            "evidence_rag_synthesis": _clean_list(management_panel.get("evidence_rag_synthesis"), 4),
            "evidence_rag_policy": management_panel.get("evidence_rag_policy") or {},
            "pubmed_rag_synthesis": _clean_list(management_panel.get("pubmed_rag_synthesis"), 4),
            "pubmed_rag_policy": management_panel.get("pubmed_rag_policy") or {},
            "clinical_inputs_required": _clean_list(management_panel.get("clinical_inputs_required"), 5),
            "clinical_information_not_available_to_ai": _clean_list(
                management_panel.get("clinical_information_not_available_to_ai")
                or treatment_panel.get("clinical_information_not_available_to_ai"),
                6,
            ),
            "red_flags": _clean_list(management_panel.get("red_flags"), 4),
            "ai_must_not_decide": _clean_list(management_panel.get("ai_must_not_decide"), 5),
            "status_specific_cautions": _clean_list(management_panel.get("status_specific_cautions"), 3),
            "sources": _clean_list(management_panel.get("sources"), 4),
            "limitations": _clean_list(management_panel.get("limitations"), 4),
        },
        "llm_management_guidance": {
            "version": llm_management_panel.get("version"),
            "status": llm_management_panel.get("status"),
            "llm_called": llm_management_panel.get("llm_called") is True,
            "production_enabled": llm_management_panel.get("production_enabled") is True,
            "display_response": llm_management_panel.get("display_response") or {},
            "validation": llm_management_panel.get("validation") or {},
            "safety_policy": llm_management_panel.get("safety_policy") or {},
            "adapter_audit": llm_management_panel.get("adapter_audit") or {},
        },
        "care_guidance": {
            "guidance_level": treatment_panel.get("guidance_level"),
            "anatomy_specific_management": treatment_panel.get("anatomy_specific_management_v3") or {},
            "clinical_localizers": _clean_list(treatment_panel.get("clinical_localizers"), 4),
            "clinical_information_not_available_to_ai": _clean_list(
                treatment_panel.get("clinical_information_not_available_to_ai"), 6
            ),
            "initial_management_considerations": _clean_list(
                treatment_panel.get("initial_management_considerations"), 2
            ),
            "red_flags_do_not_miss": _clean_list(treatment_panel.get("red_flags_do_not_miss"), 3),
            "imaging_or_followup_considerations": _clean_list(
                treatment_panel.get("imaging_or_followup_considerations"), 2
            ),
            "blocked_outputs": _clean_list(treatment_panel.get("blocked_outputs"), 3),
            "compatibility_note": "Use management_guidance for the app-facing panel.",
        },
        "evidence": {
            "status": evidence_panel.get("evidence_status"),
            "source_summary": evidence_panel.get("source_summary") or {},
            "claim_summary": evidence_panel.get("claim_summary") or {},
            "displayed_sources": _clean_list(evidence_panel.get("displayed_sources"), 5),
            "displayed_allowed_claims": _clean_list(evidence_panel.get("displayed_allowed_claims"), 6),
        },
        "llm": {
            "writer_type": validated_writer.get("writer_type"),
            "llm_used": bool(validated_writer.get("llm_used")),
            "validation": validated_writer.get("validation") or {},
            "guardrail_mode": guardrails.get("current_writer_mode"),
        },
        "safety": {
            "not_standalone_diagnosis": safety_badges.get("not_standalone_diagnosis") is True,
            "human_verification_required": safety_badges.get("human_verification_required") is True,
            "scores_are_not_clinical_probabilities": safety_badges.get(
                "scores_are_not_clinical_probabilities"
            )
            is True,
            "stage3_does_not_override_stage1_stage2": safety_badges.get(
                "stage3_does_not_override_stage1_stage2"
            )
            is True,
        },
        "display_rules": [
            "Show the strongest case-specific review points first.",
            "Keep support percentages visible with the score-policy note.",
            "Show heatmap only on user request.",
        ],
    }
