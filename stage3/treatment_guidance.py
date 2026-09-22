from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


TREATMENT_GUIDANCE_VERSION = "stage3_treatment_guidance_v1"
MANAGEMENT_GUIDANCE_VERSION = "stage3_management_guidance_v1"
MANAGEMENT_DIRECTION_VERSION = "stage3_management_direction_v1"
CASE_MANAGEMENT_SUGGESTIONS_VERSION = "stage3_case_management_suggestions_v1"
ANATOMY_MANAGEMENT_TEMPLATE_VERSION = "stage3_anatomy_management_templates_v3"
MANAGEMENT_GUIDANCE_LABEL = "Evidence-linked management context"
MANAGEMENT_SAFETY_DISCLAIMER = (
    "This section is not a diagnosis or treatment plan. It summarizes source-informed "
    "management context for clinician review. Decisions about immobilisation, "
    "weight-bearing, medication, clearance or reassurance, follow-up, imaging, referral, "
    "procedure, or surgery must be made by a qualified clinician according to local "
    "protocol and the full clinical context."
)


def _display_label(label: str | None) -> str:
    if not label:
        return "the imaged region"
    display_map = {
        "knee_lower_leg": "knee/lower leg",
        "wrist_hand": "wrist/hand",
        "ankle_foot": "ankle/foot",
        "pelvis_hip_femur": "pelvis/hip/femur",
        "shoulder_upper_arm": "shoulder/upper arm",
        "leg": "lower limb",
        "hand": "hand/wrist region",
        "hip": "pelvis/hip/femur",
    }
    return display_map.get(label, label.replace("_", " "))


def _state(summary: Stage3InputSummary) -> str:
    if summary.accepted_candidate_count > 0:
        return "localized_candidate"
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        return "image_level_caution_without_localization"
    return "no_high_confidence_candidate_retained"


def _support_band(score: float | None) -> str:
    if score is None:
        return "support_unavailable"
    if score >= 0.75:
        return "high_support"
    if score >= 0.40:
        return "moderate_support"
    return "low_support"


def _anatomy_certainty(summary: Stage3InputSummary) -> str:
    if summary.anatomy_output_type == "parent_fallback":
        return "broad_parent_label"
    if summary.anatomy_confidence is None:
        return "anatomy_support_unavailable"
    if summary.anatomy_confidence >= 0.80:
        return "high_anatomy_support"
    if summary.anatomy_confidence >= 0.50:
        return "moderate_anatomy_support"
    return "low_anatomy_support"


def _case_scenario(summary: Stage3InputSummary, label: str | None) -> Dict[str, Any]:
    state = _state(summary)
    candidate_band = _support_band(summary.primary_candidate_confidence)
    anatomy_certainty = _anatomy_certainty(summary)
    stage2c_band = _support_band(summary.stage2c_mean_suspicious_probability)

    if state == "localized_candidate":
        if candidate_band == "high_support":
            scenario_id = "localized_high_support_candidate"
            guidance_strength = "strong_context"
            title = "Localized candidate with strong model support"
        elif candidate_band == "moderate_support":
            scenario_id = "localized_moderate_support_candidate"
            guidance_strength = "moderate_context"
            title = "Localized candidate with moderate model support"
        else:
            scenario_id = "localized_low_support_candidate"
            guidance_strength = "weak_context"
            title = "Low-support localized candidate retained for review"
        why_shown = (
            "Stage 1/2 retained a localized candidate. The guidance can orient clinician review "
            "only if the marked region matches symptoms, examination, and the full radiographic study."
        )
    elif state == "image_level_caution_without_localization":
        scenario_id = "image_level_suspicion_without_bbox"
        guidance_strength = "caution_only"
        title = "Image-level caution without localized bbox"
        why_shown = (
            "No localized candidate was retained, but Stage 2C raised an image-level caution signal. "
            "This increases review priority but does not provide a management target."
        )
    else:
        scenario_id = "no_high_confidence_candidate_retained"
        guidance_strength = "limited_context"
        title = "No high-confidence candidate retained"
        why_shown = (
            "The detector/verifier did not retain a high-confidence localized candidate. "
            "This limits AI contribution and must not be interpreted as fracture exclusion."
        )

    flags: List[str] = []
    if candidate_band == "low_support" and state == "localized_candidate":
        flags.append("low_model_support")
    if anatomy_certainty in {"broad_parent_label", "low_anatomy_support"}:
        flags.append("anatomy_uncertain")
    if summary.accepted_candidate_count > 1:
        flags.append("multiple_candidates")
    if state == "image_level_caution_without_localization":
        flags.append("no_bbox_for_stage2c_warning")

    return {
        "version": "stage3_management_case_scenario_v1",
        "scenario_id": scenario_id,
        "title": title,
        "state": state,
        "anatomy_area": _display_label(label),
        "candidate_support_band": candidate_band,
        "stage2c_support_band": stage2c_band,
        "anatomy_certainty": anatomy_certainty,
        "guidance_strength": guidance_strength,
        "why_this_section_is_shown": why_shown,
        "flags": flags,
        "not_a_treatment_plan": True,
        "clinician_only": True,
    }


def _source_ids_for_treatment_context(sources: List[Dict[str, Any]], label: str | None) -> List[str]:
    preferred = []
    general = []
    broad_labels = {"leg", "hand", "hip"}
    for source in sources:
        source_id = str(source.get("source_id") or "")
        if not source_id:
            continue
        scope = source.get("anatomy_scope") or []
        topics = " ".join(source.get("topic_scope") or []).lower()
        if label and label in scope:
            preferred.append(source_id)
        elif "unknown" in scope or "fracture" in topics or "trauma" in topics:
            general.append(source_id)

    source_map = {str(source.get("source_id")): source for source in sources if source.get("source_id")}

    def priority(source_id: str) -> tuple[int, str]:
        source = source_map.get(source_id) or {}
        scope = source.get("anatomy_scope") or []
        topics = " ".join(source.get("topic_scope") or []).lower()
        if label in broad_labels and len(scope) < 4:
            return (2, source_id)
        if "fracture assessment" in topics or "complex fracture" in topics:
            return (0, source_id)
        if "trauma standards" in topics or "orthopaedic standards" in topics:
            return (1, source_id)
        return (1, source_id)

    return sorted(list(dict.fromkeys(preferred + general)), key=priority)[:6]


def _evidence_type(source: Dict[str, Any]) -> str:
    if source.get("source_type") == "guideline_metadata":
        return "guideline"
    quality = source.get("pubmed_quality") or {}
    tier = str(quality.get("quality_tier") or "").lower()
    publication_types = " ".join(source.get("publication_types") or []).lower()
    if "systematic" in tier or "meta" in tier or "systematic" in publication_types or "meta-analysis" in publication_types:
        return "systematic_review"
    if "review" in tier or "review" in publication_types:
        return "review"
    if source.get("source_type") == "pubmed":
        return "primary_study"
    return "limited"


def _source_support_text(source: Dict[str, Any]) -> str:
    title = str(source.get("title") or "Untitled source")
    organization = source.get("organization") or source.get("journal") or "source"
    topics = [str(topic) for topic in (source.get("topic_scope") or [])[:4] if topic]
    notes = str(source.get("notes") or "").strip()
    notes_lower = notes.lower()
    if notes and not any(
        marker in notes_lower
        for marker in [
            "curated metadata only",
            "exact url may require",
            "stage 3 does not scrape",
            "individual boast documents can be added later",
        ]
    ):
        return notes
    if topics:
        return f"{organization}: '{title}' supports context around {', '.join(topics)}."
    return f"{organization}: '{title}' is retained as contextual evidence for clinician review."


def _source_cards(sources: List[Dict[str, Any]], source_ids: List[str]) -> List[Dict[str, Any]]:
    source_map = {str(source.get("source_id")): source for source in sources if source.get("source_id")}
    cards = []
    for source_id in source_ids:
        source = source_map.get(str(source_id))
        if not source:
            continue
        cards.append(
            {
                "id": source.get("source_id"),
                "title": source.get("title"),
                "authors": source.get("authors") or [],
                "publisher_or_journal": source.get("organization") or source.get("journal"),
                "year": source.get("year"),
                "pmid": source.get("pmid"),
                "doi": source.get("doi"),
                "url": source.get("url"),
                "publication_type": source.get("publication_types") or source.get("source_type"),
                "evidence_level": source.get("evidence_level") or _evidence_type(source),
                "supports": [_source_support_text(source)],
                "limitations": [
                    "Used as clinician-facing context, not as patient-specific treatment advice.",
                    "Local protocol and qualified clinician judgment override this prototype.",
                ],
            }
        )
    return cards


def _management_context_items(
    considerations: List[str],
    source_ids: List[str],
    *,
    evidence_type: str = "guideline_context",
) -> List[Dict[str, Any]]:
    items = []
    for idx, text in enumerate(considerations[:4], start=1):
        items.append(
            {
                "item_id": f"management_context_{idx:02d}",
                "text": text,
                "source_ids": source_ids[:3],
                "evidence_type": evidence_type if source_ids else "limited_internal_context",
                "safety_level": "clinician_only",
                "prohibited_interpretation": (
                    "Do not interpret this as a direct treatment instruction or patient-specific plan."
                ),
            }
        )
    return items


def _source_lookup(sources: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(source.get("source_id")): source for source in sources if source.get("source_id")}


def _pubmed_rag_synthesis_items(
    allowed_claims: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    *,
    max_items: int = 4,
) -> List[Dict[str, Any]]:
    """Convert approved PubMed claims into clinician-facing RAG evidence items.

    Raw abstracts are never exposed here. The input claims have already been
    extracted by controlled templates and checked against their source IDs.
    """
    sources_by_id = _source_lookup(sources)
    ranked_claims = sorted(
        [
            claim
            for claim in allowed_claims
            if claim.get("allowed") is True
            and str(claim.get("claim_id") or "").startswith("pubmed_")
            and (sources_by_id.get(str(claim.get("source_id"))) or {}).get("source_type") == "pubmed"
        ],
        key=lambda claim: (
            0 if claim.get("allowed_for_treatment_guidance") is True else 1,
            0 if str(claim.get("claim_type") or "").startswith("controlled_pubmed_treatment") else 1,
            str(claim.get("claim_id") or ""),
        ),
    )

    items = []
    for idx, claim in enumerate(ranked_claims[:max_items], start=1):
        source = sources_by_id.get(str(claim.get("source_id"))) or {}
        quality = source.get("pubmed_quality") or {}
        title = source.get("title") or "PubMed source"
        role = claim.get("pubmed_source_role") or quality.get("source_role") or "background_context"
        anatomy_text = str(claim.get("anatomy_relevance") or "relevant anatomy")
        if claim.get("allowed_for_treatment_guidance") is True:
            summary_text = (
                "This PubMed source is retained as clinician-facing management-background evidence for review. "
                "It can support discussion of local management considerations, but it does not choose "
                "conservative care, operative care, immobilisation, weight-bearing, medication, or follow-up "
                "for this patient."
            )
        elif claim.get("allowed_for_diagnosis_context") is True:
            summary_text = (
                "This PubMed source is retained as diagnostic-imaging background. It can support clinician "
                "awareness of imaging limitations, but it does not decide whether this patient has a fracture."
            )
        else:
            summary_text = (
                "This PubMed source is retained as limited background context only. It should not be used for "
                "patient-specific management decisions."
            )
        items.append(
            {
                "item_id": f"pubmed_rag_synthesis_{idx:02d}",
                "claim_id": claim.get("claim_id"),
                "text": claim.get("claim_text"),
                "clinician_facing_summary": summary_text,
                "supports": [
                    "clinician-facing management or imaging context",
                    f"source role: {role}",
                    f"anatomy relevance: {anatomy_text}",
                ],
                "limitations": [
                    "Does not provide patient-specific treatment instructions.",
                    "Does not decide diagnosis, immobilisation, weight-bearing, medication, follow-up, procedure, or surgery.",
                    "Must be interpreted with symptoms, examination, full imaging review, and local protocol.",
                ],
                "source_id": claim.get("source_id"),
                "pmid": source.get("pmid"),
                "title": title,
                "journal": source.get("journal"),
                "year": source.get("year"),
                "url": source.get("url"),
                "claim_type": claim.get("claim_type"),
                "evidence_level": claim.get("evidence_level") or source.get("evidence_level"),
                "pubmed_source_role": claim.get("pubmed_source_role") or quality.get("source_role"),
                "pubmed_quality_tier": claim.get("pubmed_quality_tier") or quality.get("quality_tier"),
                "allowed_for_treatment_guidance": claim.get("allowed_for_treatment_guidance") is True,
                "allowed_for_diagnosis_context": claim.get("allowed_for_diagnosis_context") is True,
                "safety_level": "clinician_only_controlled_pubmed_claim",
                "abstract_exposed": False,
                "prohibited_interpretation": (
                    "Do not treat this PubMed-derived claim as a diagnosis, direct treatment order, "
                    "or patient-specific management plan."
                ),
            }
        )
    return items


def _evidence_rag_synthesis_items(
    allowed_claims: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    *,
    max_items: int = 4,
    label: str | None = None,
    case_state: str | None = None,
) -> List[Dict[str, Any]]:
    """Build public evidence-context bullets from locked claims.

    PubMed claims are preferred when available. If live/cached PubMed retrieval is
    disabled, curated guideline claims may still provide source-linked context so
    the clinician-facing panel does not become empty.
    """
    pubmed_items = _pubmed_rag_synthesis_items(allowed_claims, sources, max_items=max_items)
    if pubmed_items:
        for item in pubmed_items:
            item["evidence_lane"] = "pubmed"
            item["display_label"] = "PubMed-supported context"
        return pubmed_items

    sources_by_id = _source_lookup(sources)
    ranked_source_ids = _source_ids_for_treatment_context(sources, label)
    source_rank = {source_id: idx for idx, source_id in enumerate(ranked_source_ids)}
    guideline_claims = [
        claim
        for claim in allowed_claims
        if claim.get("allowed") is True
        and (claim.get("claim_lock") or {}).get("status") == "PASS"
        and str(claim.get("claim_id") or "").startswith("guideline_")
        and (sources_by_id.get(str(claim.get("source_id"))) or {}).get("source_type") == "guideline_metadata"
    ]
    guideline_claims = sorted(
        guideline_claims,
        key=lambda claim: (
            source_rank.get(str(claim.get("source_id")), 999),
            str(claim.get("claim_id") or ""),
        ),
    )
    items: List[Dict[str, Any]] = []
    for idx, claim in enumerate(guideline_claims[:max_items], start=1):
        source = sources_by_id.get(str(claim.get("source_id"))) or {}
        title = source.get("title") or "Curated guideline source"
        organization = source.get("organization") or "curated guideline source"
        topics = [str(topic) for topic in (source.get("topic_scope") or [])[:3] if topic]
        topic_text = f" around {', '.join(topics)}" if topics else ""
        if case_state == "no_high_confidence_candidate_retained":
            summary_text = (
                f"{organization} guidance metadata is retained only as background for clinician review "
                f"if symptoms or mechanism remain concerning{topic_text}. Because no high-confidence "
                "localized candidate was retained, this is not fracture-specific management guidance and "
                "does not indicate treatment."
            )
        elif case_state == "image_level_caution_without_localization":
            summary_text = (
                f"{organization} guidance metadata is retained as review-priority context{topic_text}. "
                "Because no localized bbox was retained, it may orient careful review but does not provide "
                "a management target or site-specific management pathway."
            )
        else:
            summary_text = (
                f"{organization} guidance metadata is retained as clinician-facing fracture-management "
                f"context{topic_text}. It can orient review, but it does not choose immobilisation, "
                "weight-bearing, medication, follow-up, clearance or reassurance decisions, procedure, or surgery for this patient."
            )
        items.append(
            {
                "item_id": f"evidence_rag_synthesis_{idx:02d}",
                "claim_id": claim.get("claim_id"),
                "text": claim.get("claim_text"),
                "clinician_facing_summary": summary_text,
                "supports": [
                    "clinician-facing guideline context",
                    "source-linked management orientation",
                    "non-prescriptive review support",
                ],
                "limitations": [
                    "Curated guideline metadata is not a complete patient-specific protocol.",
                    "Does not decide diagnosis, immobilisation, weight-bearing, medication, follow-up, procedure, or surgery.",
                    "Must be interpreted with symptoms, examination, full imaging review, and local protocol.",
                ],
                "source_id": claim.get("source_id"),
                "pmid": None,
                "title": title,
                "journal": source.get("journal"),
                "year": source.get("year"),
                "url": source.get("url"),
                "claim_type": claim.get("claim_type"),
                "evidence_level": claim.get("evidence_level") or source.get("evidence_level"),
                "evidence_lane": "curated_guideline",
                "display_label": "Guideline-supported context",
                "allowed_for_treatment_guidance": False,
                "allowed_for_diagnosis_context": False,
                "safety_level": "clinician_only_controlled_guideline_claim",
                "abstract_exposed": False,
                "prohibited_interpretation": (
                    "Do not treat this guideline-derived context as a diagnosis, direct treatment order, "
                    "or patient-specific management plan."
                ),
            }
        )
    return items


def _clinical_inputs_required(label: str | None) -> List[str]:
    inputs = [
        "mechanism of injury",
        "exact pain location",
        "focal bony tenderness",
        "pain severity and progression",
        "swelling progression",
        "neurovascular examination",
        "skin condition / open wound status",
        "complete radiographic series / views",
        "formal radiology review",
        "prior imaging or comparison study when available",
        "previous injury or surgery when relevant",
        "age, frailty, comorbidity, and bone-health context when relevant",
    ]
    if label in {"ankle_foot", "knee_lower_leg", "leg", "pelvis_hip_femur", "hip"}:
        inputs.insert(3, "ability to use or bear weight when clinically relevant")
        inputs.insert(4, "range of motion when relevant")
    else:
        inputs.insert(3, "ability to use the limb when clinically relevant")
        inputs.insert(4, "range of motion when relevant")
    return inputs


def _ai_must_not_decide(label: str | None) -> List[str]:
    return [
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


def _clinical_information_not_available(summary: Stage3InputSummary, label: str | None) -> List[str]:
    missing = [
        "mechanism of injury and time since injury",
        "patient age, relevant comorbidities, medication, and bone-health risk factors",
        "exact pain location, focal tenderness, swelling pattern, and functional limitation",
        "open wound / skin condition and neurovascular examination",
        "full radiographic series quality, projections, and formal radiology interpretation",
    ]
    if summary.accepted_candidate_count > 0:
        missing.append("whether the marked bbox matches the patient's focal symptoms")
    elif summary.stage2c_decision in {"suspicious", "uncertain"}:
        missing.append("the exact symptomatic site, because Stage 2C does not provide a localized bbox")
    else:
        missing.append("whether clinical suspicion remains despite no retained high-confidence candidate")
    if label in {"ankle_foot", "knee_lower_leg", "leg", "pelvis_hip_femur", "hip"}:
        missing.append("ability to bear weight or mobilize in the real clinical encounter")
    else:
        missing.append("ability to use the limb/joint and the degree of motion limitation")
    return missing


def _status_specific_cautions(summary: Stage3InputSummary) -> List[str]:
    state = _state(summary)
    if state == "localized_candidate":
        return [
            "The retained bounding box is a screening cue, not a management target.",
            "Clinician review should start at the marked region and continue across the complete radiographic study.",
            "Do not base management decisions only on the AI score or bounding box.",
        ]
    if state == "image_level_caution_without_localization":
        return [
            "No localized fracture boundary is available.",
            "The image-level signal may justify whole-image review but does not identify a management target.",
            "Any heatmap is optional visual support and must not be interpreted as a fracture contour, treatment area, or bounding box.",
        ]
    return [
        "No high-confidence localized AI finding was retained.",
        "This does not exclude fracture or other injury.",
        "If clinical suspicion remains, clinician-led review, additional imaging, or follow-up may still be required according to local protocol.",
        "Do not use this result to reassure, clear from care, or rule out injury.",
    ]


def _management_direction(
    summary: Stage3InputSummary,
    label: str | None,
    pubmed_synthesis: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """State-aware management direction for the app.

    This is deliberately treatment-oriented but non-prescriptive: it tells the
    clinician what kind of management pathway context the AI output can support,
    and when it should not produce retained-candidate management guidance.
    """
    state = _state(summary)
    anatomy = _display_label(label)
    pubmed_available = bool(pubmed_synthesis)
    source_phrase = "PubMed/guideline context" if pubmed_available else "guideline/local-protocol context"

    if state == "localized_candidate":
        status_id = "fracture_specific_management_context_available"
        summary_text = (
            f"A localized fracture candidate is present around {anatomy}. The system may support "
            "clinician-led fracture-pathway thinking, but only after the marked region is checked "
            "against symptoms, examination, and the full radiographic study."
        )
        direction_bullets = [
            "Start from the retained bounding box, then review the complete image and clinical presentation.",
            (
                "If the marked region agrees with focal symptoms and examination, use local fracture pathway "
                "review to decide immobilisation/support, further imaging, follow-up, or specialist review."
            ),
            (
                f"Use the {source_phrase} below as background for management discussion, not as a direct order."
            ),
        ]
    elif state == "image_level_caution_without_localization":
        status_id = "review_priority_context_without_treatment_target"
        summary_text = (
            f"The image-level safety layer raised concern around {anatomy}, but no localized management target "
            "or bounding box is available. This can increase review priority, not define management."
        )
        direction_bullets = [
            "Review the whole image and the symptomatic region before selecting any site-specific pathway.",
            "Do not use a heatmap or image-level warning to decide immobilisation, mobility, medication, or procedure.",
            (
                "If symptoms remain concerning, clinician-led follow-up imaging or specialist review may be considered "
                "according to local protocol."
            ),
        ]
    else:
        status_id = "no_retained_candidate_management_guidance_from_ai"
        summary_text = (
            f"No high-confidence fracture candidate was retained around {anatomy}. The system therefore does not "
            "generate retained-candidate management guidance for this image. This must not be interpreted as injury "
            "exclusion or as clearance or reassurance if symptoms, examination, or mechanism remain concerning."
        )
        direction_bullets = [
            "No retained-candidate AI management pathway is suggested from the detections.",
            "Treat this as limited screening support only; clinician judgement and local protocol determine whether further action is needed.",
            (
                "If focal pain, swelling, deformity, inability to use/bear weight, or high-risk mechanism persists, "
                "clinician-led review, further imaging, or follow-up may still be appropriate."
            ),
        ]

    return {
        "version": MANAGEMENT_DIRECTION_VERSION,
        "status_id": status_id,
        "anatomy_area": anatomy,
        "treatment_oriented": True,
        "is_treatment_plan": False,
        "clinician_only": True,
        "human_verification_required": True,
        "pubmed_context_available": pubmed_available,
        "summary": summary_text,
        "direction_bullets": direction_bullets,
        "decision_boundary": (
            "This section may orient clinician review, but it must not decide diagnosis, immobilisation, "
            "weight-bearing, medication, procedure, follow-up interval, clearance or reassurance decisions, or surgery."
        ),
    }


def _source_ids_for_suggestion(
    source_ids: List[str],
    pubmed_synthesis: List[Dict[str, Any]],
) -> List[str]:
    pubmed_ids = [str(item.get("source_id")) for item in pubmed_synthesis if item.get("source_id")]
    return list(dict.fromkeys(pubmed_ids + source_ids))[:3]


def _case_management_suggestion(
    suggestion_id: str,
    category: str,
    text: str,
    source_ids: List[str],
    rationale: str,
    *,
    priority: str = "review",
) -> Dict[str, Any]:
    return {
        "suggestion_id": suggestion_id,
        "category": category,
        "priority": priority,
        "text": text,
        "rationale": rationale,
        "source_ids": source_ids,
        "clinician_only": True,
        "is_direct_order": False,
        "requires_human_confirmation": True,
        "prohibited_interpretation": (
            "Do not interpret this as a patient-specific treatment instruction; it is a clinician-facing "
            "management consideration that depends on examination, symptoms, full imaging review, and local protocol."
        ),
    }


def _source_trace_for_text(text: str, source_ids: List[str], *, item_type: str) -> Dict[str, Any]:
    return {
        "item_type": item_type,
        "text": text,
        "source_ids": source_ids,
        "source_required": True,
        "source_status": "linked" if source_ids else "missing",
        "supports": [
            "clinician-facing management context only",
            "not a diagnosis",
            "not a patient-specific treatment instruction",
        ],
        "limitations": [
            "requires symptoms, examination, full imaging review, and local protocol",
            "must not decide immobilisation, weight-bearing, medication, procedure, follow-up, clearance or reassurance decisions, or surgery",
        ],
    }


def _anatomy_specific_suggestions(
    label: str | None,
    state: str,
    sources_for_suggestion: List[str],
) -> List[Dict[str, Any]]:
    if state != "localized_candidate":
        return []
    anatomy_text = _display_label(label)
    anatomy_map = {
        "wrist_hand": (
            "occult_wrist_hand_review",
            "occult-injury pathway context",
            "If the symptomatic area involves the scaphoid, distal radius/ulna, carpus, or hand alignment, consider local pathway review for protective support, additional views, or follow-up imaging as clinician-led decisions.",
            "Wrist/hand injuries can be subtle and management depends strongly on focal tenderness, alignment, and clinical suspicion.",
        ),
        "hand": (
            "hand_alignment_review",
            "alignment and function context",
            "If finger, metacarpal, or hand alignment concern matches the marked region, consider local pathway review for protective support and specialist input when rotation, angulation, or functional deficit is present.",
            "Hand management is highly dependent on alignment, rotation, function, and open/neurovascular findings.",
        ),
        "elbow": (
            "elbow_motion_occult_review",
            "elbow motion and occult-injury context",
            "If focal elbow tenderness or reduced motion matches the candidate region, consider local pathway review for support, additional views, or follow-up assessment rather than relying on the AI score alone.",
            "Elbow injury review depends on motion, focal tenderness, projection quality, and associated forearm/wrist symptoms.",
        ),
        "forearm": (
            "forearm_adjacent_joint_review",
            "forearm and adjacent-joint context",
            "If radius/ulna injury is clinically suspected, consider clinician-led review of both adjacent joints and local pathway decisions around support, further imaging, or specialist input.",
            "Forearm trauma can involve linked wrist or elbow injury patterns, so management context should not narrow too early.",
        ),
        "shoulder_upper_arm": (
            "shoulder_upper_arm_function_review",
            "shoulder/upper-arm function context",
            "If the candidate region matches shoulder or upper-arm symptoms, consider local pathway review around comfort support, functional limitation, dislocation concern, and need for specialist review.",
            "Shoulder/upper-arm management depends on location, function, deformity, dislocation concern, and neurovascular status.",
        ),
        "shoulder": (
            "shoulder_parent_function_review",
            "shoulder function context",
            "If the candidate region matches shoulder symptoms, consider local pathway review around comfort support, functional limitation, dislocation concern, and need for specialist review.",
            "Shoulder management depends on function, deformity, dislocation concern, and neurovascular status.",
        ),
        "ankle_foot": (
            "ankle_foot_focal_tenderness_review",
            "ankle/foot pathway context",
            "If malleolar, navicular, base-of-fifth-metatarsal, or focal foot tenderness matches the candidate region, consider local pathway review around support, imaging adequacy, follow-up, and specialist input when red flags exist.",
            "Ankle/foot management context depends on focal tenderness, ability to bear weight, alignment, and neurovascular/open-injury status.",
        ),
        "knee_lower_leg": (
            "knee_lower_leg_weight_bearing_review",
            "knee/lower-leg pathway context",
            "If the candidate region matches knee or lower-leg symptoms, consider local pathway review around support, imaging adequacy, follow-up, and specialist input when deformity, severe swelling, or compartment concern exists.",
            "Knee/lower-leg management depends on weight-bearing ability, focal tenderness, mechanism, swelling, and neurovascular status.",
        ),
        "leg": (
            "lower_limb_region_narrowing_review",
            "lower-limb pathway selection context",
            "First narrow the symptomatic region to hip/pelvis, knee/lower leg, ankle/foot, or long-bone shaft before selecting a local management pathway.",
            "A parent lower-limb label is useful for broad review but is not precise enough for site-specific pathway selection.",
        ),
        "pelvis_hip_femur": (
            "hip_pelvis_mobility_review",
            "hip/pelvis/femur pathway context",
            "If hip, pelvis, or femur symptoms match the candidate region, consider clinician-led escalation when pain, inability to mobilize, high-energy mechanism, or occult injury concern persists.",
            "Hip/pelvis/femur management context is strongly affected by mobility, mechanism, occult injury concern, and neurovascular status.",
        ),
        "hip": (
            "hip_parent_mobility_review",
            "hip pathway context",
            "If hip symptoms match the candidate region, consider clinician-led escalation when pain, inability to mobilize, high-energy mechanism, or occult injury concern persists.",
            "Hip management context is strongly affected by mobility, mechanism, occult injury concern, and neurovascular status.",
        ),
    }
    item = anatomy_map.get(label)
    if not item:
        return [
            _case_management_suggestion(
                "generic_region_pathway_review",
                "site-specific pathway context",
                f"If the marked region matches symptoms around {anatomy_text}, use clinician-led local pathway review before choosing support, imaging, follow-up, or specialist input.",
                sources_for_suggestion,
                "The AI output can orient review, but exact pathway selection depends on clinical localization.",
            )
        ]
    suggestion_id, category, text, rationale = item
    return [
        _case_management_suggestion(
            suggestion_id,
            category,
            text,
            sources_for_suggestion,
            rationale,
        )
    ]


def _case_management_suggestions(
    summary: Stage3InputSummary,
    label: str | None,
    source_ids: List[str],
    pubmed_synthesis: List[Dict[str, Any]],
) -> Dict[str, Any]:
    state = _state(summary)
    anatomy = _display_label(label)
    sources_for_suggestion = _source_ids_for_suggestion(source_ids, pubmed_synthesis)
    candidate_score = summary.primary_candidate_confidence
    stage2c_score = summary.stage2c_mean_suspicious_probability

    suggestions: List[Dict[str, Any]] = []
    if state == "localized_candidate":
        confidence_phrase = (
            "higher model support"
            if candidate_score is not None and candidate_score >= 0.75
            else "moderate or uncertain model support"
        )
        suggestions.append(
            _case_management_suggestion(
                "candidate_concordance_review",
                "candidate-to-symptom concordance",
                (
                    f"Check whether the retained candidate around {anatomy} matches focal pain, tenderness, "
                    f"mechanism, and examination before using it to orient management discussion."
                ),
                sources_for_suggestion,
                f"The detector/verifier retained a localized candidate with {confidence_phrase}; concordance with the patient context is still required.",
                priority="high",
            )
        )
        suggestions.extend(_anatomy_specific_suggestions(label, state, sources_for_suggestion))
        suggestions.append(
            _case_management_suggestion(
                "comfort_support_pathway_review",
                "comfort and temporary support context",
                (
                    "While clinician verification is pending, consider local-protocol comfort measures and temporary "
                    "protection/support of the symptomatic region when clinically appropriate."
                ),
                sources_for_suggestion,
                "This is a conservative bridge consideration, not a decision about final treatment, mobility, or procedure.",
            )
        )
    elif state == "image_level_caution_without_localization":
        caution_phrase = (
            "strong image-level caution signal"
            if stage2c_score is not None and stage2c_score >= 0.7
            else "image-level caution signal"
        )
        suggestions.extend(
            [
                _case_management_suggestion(
                    "whole_image_before_pathway",
                    "whole-image review before pathway selection",
                    (
                        f"The {caution_phrase} can justify careful whole-image and symptom-focused review, but it "
                        "should not trigger site-specific support, imaging, or specialist pathway decisions by itself."
                    ),
                    sources_for_suggestion,
                    "Stage 2C does not provide a localized candidate or management target.",
                    priority="high",
                ),
                _case_management_suggestion(
                    "symptom_localizes_pathway",
                    "symptom-localized pathway context",
                    (
                        "If clinical symptoms localize to a specific region despite no retained box, use the relevant "
                        "local anatomical pathway and consider further review according to clinical concern."
                    ),
                    sources_for_suggestion,
                    "The image-level warning is useful only when paired with examination and image coverage.",
                ),
            ]
        )
    else:
        suggestions.extend(
            [
                _case_management_suggestion(
                    "no_retained_candidate_limited_screening_context",
                    "limited screening context",
                    (
                        "In no-high-confidence-candidate cases, the AI output should be treated only as limited screening "
                        "support. Decisions about reassurance, clearance, further imaging, follow-up, or referral must "
                        "remain clinician-led and based on symptoms, examination, mechanism, complete radiographic review, "
                        "formal radiology review, and local protocol."
                    ),
                    sources_for_suggestion,
                    "Absence of a retained candidate is not injury exclusion; it only limits what the AI can contribute.",
                ),
                _case_management_suggestion(
                    "persistent_symptoms_override_ai_absence",
                    "persistent-symptom safety context",
                    (
                        "If focal symptoms, swelling, deformity, inability to use/bear weight, or high-risk mechanism "
                        "persist, clinician-led follow-up, further imaging, or specialist review may still be appropriate."
                    ),
                    sources_for_suggestion,
                    "Clinical concern can outweigh low AI support, especially for subtle or occult injuries.",
                    priority="high",
                ),
            ]
        )

    return {
        "version": CASE_MANAGEMENT_SUGGESTIONS_VERSION,
        "state": state,
        "anatomy_area": anatomy,
        "suggestion_count": len(suggestions),
        "uses_pubmed_context": bool(pubmed_synthesis),
        "suggestions": suggestions[:4],
        "display_policy": {
            "show_after_management_direction": True,
            "clinician_only": True,
            "not_patient_specific_treatment": True,
            "stable_structure_with_case_specific_content": True,
            "hide_when_empty": False,
        },
    }


def _guidance_traceability(
    management_context: List[Dict[str, Any]],
    case_suggestions: Dict[str, Any],
    pubmed_synthesis: List[Dict[str, Any]],
    source_ids: List[str],
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for item in management_context:
        if not isinstance(item, dict):
            continue
        items.append(
            _source_trace_for_text(
                str(item.get("text") or ""),
                [str(source_id) for source_id in item.get("source_ids") or []],
                item_type="management_context",
            )
        )
    for item in (case_suggestions.get("suggestions") or []):
        if not isinstance(item, dict):
            continue
        items.append(
            _source_trace_for_text(
                str(item.get("text") or ""),
                [str(source_id) for source_id in item.get("source_ids") or source_ids],
                item_type="case_management_suggestion",
            )
        )
    for item in pubmed_synthesis:
        if not isinstance(item, dict):
            continue
        items.append(
            _source_trace_for_text(
                str(item.get("clinician_facing_summary") or item.get("text") or ""),
                [str(item.get("source_id"))] if item.get("source_id") else [],
                item_type="pubmed_rag_synthesis",
            )
        )
    missing = [item for item in items if item.get("source_status") != "linked"]
    return {
        "version": "stage3_guidance_traceability_v1",
        "all_public_management_items_require_sources": True,
        "all_items_source_linked": not missing,
        "item_count": len(items),
        "missing_source_item_count": len(missing),
        "items": items[:12],
    }


def _anatomy_template(label: str | None) -> Dict[str, Any]:
    templates: Dict[str, Dict[str, Any]] = {
        "wrist_hand": {
            "template_id": "wrist_hand_management_v3",
            "review_focus": "distal radius/ulna, carpus, scaphoid region, metacarpals, and finger alignment",
            "clinical_localizers": [
                "exact point of maximal tenderness: distal radius/ulna, anatomical snuffbox, metacarpal, phalanx, or carpal region",
                "visible or clinical malrotation, angulation, shortening, or loss of normal hand cascade",
                "ability to actively move fingers and wrist compared with the painful region",
                "neurovascular status of the hand and digits",
            ],
            "management_considerations": [
                "If the visual cue and symptoms localize to the distal radius/ulna or carpus, frame the case around local wrist fracture-pathway review rather than a generic upper-limb pathway.",
                "If snuffbox or scaphoid-region tenderness is part of the presentation, keep occult scaphoid injury in the review plan even when the retained candidate is weak or not perfectly localized.",
                "If metacarpal or finger alignment is clinically abnormal, prioritize rotation, angulation, open-injury, and functional assessment before any pathway decision.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the hand",
                "open wound over a suspected injury",
                "scaphoid-region tenderness with persistent symptoms",
                "malrotation or alignment concern in fingers/metacarpals",
            ],
            "followup_considerations": [
                "If symptoms remain focal despite uncertain initial views, local protocol may support additional views, delayed repeat imaging, or specialist review.",
                "Specialist review context becomes stronger when alignment, neurovascular, open-injury, or persistent occult-injury concern is present.",
            ],
        },
        "hand": {
            "template_id": "hand_parent_management_v3",
            "inherits": "wrist_hand",
        },
        "elbow": {
            "template_id": "elbow_management_v3",
            "review_focus": "elbow joint line, radial head/neck, olecranon, distal humerus, and associated forearm/wrist symptoms",
            "clinical_localizers": [
                "point tenderness over radial head/neck, olecranon, distal humerus, or joint line",
                "range of motion, especially extension, flexion, pronation, and supination limitation",
                "visible effusion or fat-pad concern if present on the radiographic series",
                "hand neurovascular status and associated forearm/wrist pain after trauma",
            ],
            "management_considerations": [
                "If focal elbow tenderness and motion loss match the candidate, orient review toward elbow injury pathway context rather than treating it as a generic arm finding.",
                "If radiographs are uncertain but clinical restriction is marked, keep occult elbow injury or associated radial-head injury in the clinician review plan.",
                "If symptoms extend into the forearm or wrist, avoid narrowing the assessment to the elbow alone before adjacent-region injury is considered.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the arm or hand",
                "open injury concern",
                "marked loss of elbow motion with focal tenderness",
                "associated wrist or forearm symptoms after trauma",
            ],
            "followup_considerations": [
                "Persistent focal elbow symptoms with uncertain imaging may justify clinician-led additional views, repeat assessment, or follow-up imaging under local protocol.",
                "Specialist review context becomes stronger with deformity, neurovascular concern, open injury, instability concern, or severe motion loss.",
            ],
        },
        "forearm": {
            "template_id": "forearm_management_v3",
            "review_focus": "radius/ulna shaft, distal and proximal radioulnar relationships, and both adjacent joints",
            "clinical_localizers": [
                "shaft tenderness over radius or ulna and whether deformity is present",
                "wrist and elbow pain, alignment, and range of motion",
                "swelling severity and pain out of proportion to examination",
                "neurovascular status of the hand",
            ],
            "management_considerations": [
                "If radius/ulna shaft concern matches the candidate, review both wrist and elbow before choosing any single-site pathway.",
                "If deformity, linked-joint pain, or severe swelling is present, treat the AI output as a prompt for broader forearm trauma review rather than an isolated bbox finding.",
                "If the AI confidence is low but clinical tenderness is focal, do not let a weak bbox down-rank clinician concern.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the hand",
                "deformity or severe swelling",
                "compartment-syndrome concern",
                "associated wrist or elbow injury concern",
            ],
            "followup_considerations": [
                "Specialist review context becomes stronger with deformity, neurovascular concern, open injury, severe swelling, or linked wrist/elbow concern.",
                "Additional imaging or follow-up context is stronger when clinical symptoms exceed AI localization confidence.",
            ],
        },
        "shoulder_upper_arm": {
            "template_id": "shoulder_upper_arm_management_v3",
            "review_focus": "shoulder alignment, proximal humerus, clavicle-area overlap, humeral shaft, and distal neurovascular status",
            "clinical_localizers": [
                "whether pain localizes to shoulder joint, proximal humerus, clavicle-area overlap, or shaft",
                "visible deformity, dislocation concern, or loss of normal contour",
                "ability to actively move the shoulder and elbow when clinically appropriate",
                "distal neurovascular status of the arm and hand",
            ],
            "management_considerations": [
                "Before selecting a pathway, separate shoulder-joint/dislocation concern from proximal humerus, clavicle-area overlap, or humeral-shaft concern.",
                "If deformity, dislocation concern, or marked functional limitation is present, prioritize clinician-led escalation context over routine review wording.",
                "If the candidate is weak but symptoms are strongly localized to shoulder/upper arm, keep projection adequacy and additional-view review in scope.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the arm or hand",
                "fracture-dislocation concern",
                "open injury concern",
                "visible deformity or severe functional limitation",
            ],
            "followup_considerations": [
                "Specialist review context becomes stronger with dislocation concern, neurovascular concern, deformity, severe functional limitation, or high-energy mechanism.",
                "If projection or coverage is limited, clinician-led additional imaging review may be more relevant than relying on model support alone.",
            ],
        },
        "shoulder": {
            "template_id": "shoulder_parent_management_v3",
            "inherits": "shoulder_upper_arm",
        },
        "ankle_foot": {
            "template_id": "ankle_foot_management_v3",
            "review_focus": "malleoli, ankle mortise, talus/navicular region, base of fifth metatarsal, midfoot, and forefoot",
            "clinical_localizers": [
                "ability to bear weight or mobilize when clinically relevant",
                "focal bony tenderness over malleoli, navicular, base of fifth metatarsal, midfoot, or metatarsals",
                "ankle mortise/alignment concern, deformity, or marked swelling",
                "neurovascular status and skin/open-wound condition of the foot",
            ],
            "management_considerations": [
                "If symptoms localize to malleolar, navicular, base-of-fifth-metatarsal, midfoot, or metatarsal regions, use that localization to choose the relevant ankle/foot review pathway.",
                "Weight-bearing ability is clinical context only; the AI must not convert it into a mobility instruction.",
                "If swelling, deformity, mortise/alignment concern, or open injury is present, escalation context should take priority over confidence-score discussion.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the foot",
                "open injury concern",
                "visible deformity",
                "malleolar, navicular, or base-of-fifth-metatarsal focal tenderness with persistent symptoms",
                "marked midfoot pain or alignment concern after trauma",
            ],
            "followup_considerations": [
                "Persistent focal tenderness with uncertain imaging may justify local ankle/foot imaging-pathway review, additional views, or follow-up according to clinician judgement.",
                "Specialist review context becomes stronger with neurovascular concern, open injury, deformity, unstable injury concern, midfoot alignment concern, or high-energy mechanism.",
            ],
        },
        "knee_lower_leg": {
            "template_id": "knee_lower_leg_management_v3",
            "review_focus": "patella, tibial plateau, proximal fibula, tibia/fibula shaft, knee alignment, and compartment-risk features",
            "clinical_localizers": [
                "ability to bear weight or mobilize when clinically relevant",
                "focal tenderness over patella, tibial plateau, proximal fibula, tibial/fibular shaft, or joint line",
                "knee range of motion, effusion, deformity, and swelling severity",
                "distal neurovascular status and pain/swelling out of proportion to examination",
            ],
            "management_considerations": [
                "If symptoms localize to patella, tibial plateau, proximal fibula, or shaft, choose that pathway context rather than treating the image as a generic lower-limb case.",
                "Weight-bearing and knee motion are clinical context only; the AI must not convert them into mobility or clearance or reassurance instructions.",
                "If severe swelling, deformity, or disproportionate pain is present, escalation and compartment-risk review should outrank a weak or low-support model signal.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the lower limb",
                "open injury concern",
                "visible deformity or severe swelling",
                "tibial plateau, patellar, proximal fibula, or compartment-syndrome concern",
            ],
            "followup_considerations": [
                "Persistent focal symptoms with uncertain imaging may justify local knee/lower-leg imaging-pathway review or follow-up according to clinician judgement.",
                "Specialist review context becomes stronger with neurovascular concern, open injury, deformity, suspected unstable injury, high-energy mechanism, or compartment concern.",
            ],
        },
        "leg": {
            "template_id": "leg_parent_management_v3",
            "review_focus": "broad lower-limb localization before choosing hip, knee/lower-leg, ankle/foot, or shaft pathway",
            "clinical_localizers": [
                "exact symptomatic subregion: hip/groin, thigh, knee, shin/calf, ankle, or foot",
                "ability to bear weight or mobilize when clinically relevant",
                "deformity, severe swelling, open wound, or focal bony tenderness",
                "distal neurovascular status",
            ],
            "management_considerations": [
                "A parent lower-limb label is not precise enough for a site-specific pathway; first narrow the clinical and radiographic region.",
                "If symptoms localize clearly, use the relevant hip/pelvis, knee/lower-leg, ankle/foot, or shaft pathway context rather than this broad label.",
                "The AI must not turn lower-limb model support into a weight-bearing, mobility, or clearance or reassurance instruction.",
            ],
            "do_not_miss": [
                "neurovascular symptoms in the lower limb",
                "open injury concern",
                "visible deformity or severe swelling",
                "compartment-syndrome concern",
            ],
            "followup_considerations": [
                "Clinician-led region narrowing should precede any specific management pathway.",
                "Specialist review context becomes stronger with neurovascular concern, deformity, open injury, high-energy mechanism, or compartment concern.",
            ],
        },
        "pelvis_hip_femur": {
            "template_id": "pelvis_hip_femur_management_v3",
            "review_focus": "hip joint, femoral neck/intertrochanteric region, pelvis, femoral shaft, mobility, and occult injury concern",
            "clinical_localizers": [
                "exact pain location: groin, lateral hip, pelvis, thigh, or femoral shaft",
                "ability to mobilize or bear weight when clinically relevant",
                "mechanism of injury, age/risk profile, deformity, and limb position",
                "distal neurovascular status",
            ],
            "management_considerations": [
                "If symptoms localize to hip/groin or proximal femur, occult hip/femoral-neck concern may remain relevant even when model localization is weak.",
                "Mobility is clinical context only; the AI must not convert it into a weight-bearing or clearance or reassurance instruction.",
                "High-energy mechanism, inability to mobilize, deformity, or severe focal pain should shift the wording toward escalation context rather than routine follow-up context.",
            ],
            "do_not_miss": [
                "inability to mobilize or bear weight when clinically relevant",
                "severe hip/groin pain after trauma",
                "neurovascular symptoms in the lower limb",
                "high-energy trauma or occult hip fracture concern",
            ],
            "followup_considerations": [
                "Specialist review context becomes stronger with inability to mobilize, high-energy mechanism, deformity, neurovascular concern, or occult hip concern.",
                "Persistent hip/groin symptoms with uncertain radiographs may justify clinician-led follow-up imaging pathway review.",
            ],
        },
        "hip": {
            "template_id": "hip_parent_management_v3",
            "inherits": "pelvis_hip_femur",
        },
        "unknown": {
            "template_id": "generic_management_v3",
            "review_focus": "clinical localization before any site-specific pathway",
            "clinical_localizers": [
                "exact symptomatic region and mechanism of injury",
                "visible deformity, open wound, swelling severity, and focal tenderness",
                "ability to use or mobilize the involved region when clinically relevant",
                "neurovascular status",
            ],
            "management_considerations": [
                "Match the symptomatic region to the radiographic field before choosing any site-specific pathway.",
                "Use the AI output as screening support only; it is not precise enough by itself to define management.",
                "If symptoms are focal but anatomy classification is uncertain, clinician-led localization should override the broad AI label.",
            ],
            "do_not_miss": [
                "open injury concern",
                "neurovascular symptoms",
                "visible deformity",
                "high-energy trauma mechanism",
            ],
            "followup_considerations": [
                "Consider additional clinician review if symptoms are persistent, worsening, or not explained by the AI output.",
                "Consider specialist review if red flags or unstable injury concern exists.",
            ],
        },
    }
    key = label or "unknown"
    template = dict(templates.get(key) or templates["unknown"])
    inherited = template.get("inherits")
    if inherited:
        base = dict(templates[inherited])
        base["template_id"] = template["template_id"]
        base["inherited_from"] = inherited
        return base
    return template


def _anatomy_management_v3(
    summary: Stage3InputSummary,
    label: str | None,
    template: Dict[str, Any],
) -> Dict[str, Any]:
    state = _state(summary)
    scenario = _case_scenario(summary, label)
    candidate_band = scenario["candidate_support_band"]
    anatomy_certainty = scenario["anatomy_certainty"]

    if state == "localized_candidate":
        if candidate_band == "low_support":
            case_adjustment = (
                "The retained bbox is a weak visual cue. Use it to focus review, but do not let it outweigh "
                "clinical localization, full-image review, or formal radiology interpretation."
            )
        elif candidate_band == "moderate_support":
            case_adjustment = (
                "The retained bbox has moderate model support. It can help prioritize the marked region, "
                "but management context still depends on symptom concordance and complete image review."
            )
        else:
            case_adjustment = (
                "The retained bbox has stronger model support. Start review at the marked region, then "
                "confirm against symptoms, examination, alignment, and the full radiographic series."
            )
    elif state == "image_level_caution_without_localization":
        case_adjustment = (
            "No bbox was retained. The image-level warning can raise review priority, but it cannot define "
            "a fracture site, management target, or anatomy-specific pathway without clinician localization."
        )
    else:
        case_adjustment = (
            "No high-confidence candidate was retained. This limits AI management context, but it must not "
            "be treated as injury exclusion or clearance or reassurance when symptoms, mechanism, or examination remain concerning."
        )

    if anatomy_certainty in {"broad_parent_label", "low_anatomy_support"}:
        anatomy_adjustment = (
            "Anatomy support is broad or uncertain; keep pathway selection at parent-region level until "
            "the clinician localizes the symptomatic site."
        )
    else:
        anatomy_adjustment = (
            "Anatomy support is adequate for region-specific context, but not for diagnosis or treatment decisions."
        )

    return {
        "version": ANATOMY_MANAGEMENT_TEMPLATE_VERSION,
        "template_id": template.get("template_id"),
        "display_label": _display_label(label),
        "inherited_from": template.get("inherited_from"),
        "review_focus": template.get("review_focus"),
        "case_adjustment": case_adjustment,
        "anatomy_adjustment": anatomy_adjustment,
        "clinical_localizers": list(template.get("clinical_localizers") or []),
        "management_considerations": list(template.get("management_considerations") or []),
        "followup_or_imaging_context": list(template.get("followup_considerations") or []),
        "red_flags_or_escalation_triggers": list(template.get("do_not_miss") or []),
        "non_prescriptive_boundary": [
            "This section may guide clinician review, not decide treatment.",
            "Do not infer immobilisation, weight-bearing, medication, procedure, clearance or reassurance decisions, follow-up interval, or surgery from this template.",
        ],
        "wording_policy": {
            "stable_schema": True,
            "case_specific_wording": True,
            "avoid_copy_paste_generic_text": True,
            "no_patient_specific_orders": True,
        },
    }


def _state_considerations(summary: Stage3InputSummary) -> List[str]:
    state = _state(summary)
    if state == "localized_candidate":
        return [
            "A retained bbox can prioritize clinical review, but treatment decisions still require clinician verification.",
            "If the marked region matches symptoms and examination, consider local fracture pathway review under clinician control.",
        ]
    if state == "image_level_caution_without_localization":
        return [
            "The image-level warning can raise review priority, but it must not be used to choose site-specific treatment.",
            "Do not use the heatmap as a fracture boundary or as the basis for immobilization, weight-bearing, or procedure decisions.",
        ]
    return [
        "No retained high-confidence candidate does not exclude subtle injury.",
        "If clinical concern persists, treatment-oriented decisions should follow symptoms, examination, and local clinical pathways rather than AI absence.",
    ]


def _blocked_outputs(label: str | None) -> List[str]:
    blocked = [
        "Do not state that a fracture is confirmed or excluded.",
        "Do not tell the patient that no clinician review is needed.",
        "Do not prescribe medication, dosage, or timing.",
        "Do not order casting, splinting, immobilization, procedures, or surgery.",
        "Do not provide return-to-sport, return-to-work, or clearance or reassurance decisions.",
    ]
    if label in {"ankle_foot", "knee_lower_leg", "leg", "pelvis_hip_femur", "hip"}:
        blocked.append("Do not provide weight-bearing or mobility instructions.")
    else:
        blocked.append("Do not provide activity restriction, return-to-use, or limb-use instructions.")
    return blocked


def build_treatment_guidance(
    summary: Stage3InputSummary,
    red_flags: Dict[str, Any],
    sources: List[Dict[str, Any]],
    allowed_claims: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    label = summary.anatomy_label or summary.anatomy_parent_label
    template = _anatomy_template(label)
    scenario = _case_scenario(summary, label)
    source_ids = _source_ids_for_treatment_context(sources, label)
    pubmed_synthesis = _pubmed_rag_synthesis_items(allowed_claims or [], sources)
    evidence_max_items = 1 if scenario["state"] == "no_high_confidence_candidate_retained" else 3
    evidence_synthesis = _evidence_rag_synthesis_items(
        allowed_claims or [],
        sources,
        label=label,
        case_state=scenario["state"],
        max_items=evidence_max_items,
    )
    management_direction = _management_direction(summary, label, pubmed_synthesis)
    case_suggestions = _case_management_suggestions(summary, label, source_ids, pubmed_synthesis)
    anatomy_management = _anatomy_management_v3(summary, label, template)
    pubmed_source_ids = [str(item.get("source_id")) for item in pubmed_synthesis if item.get("source_id")]
    evidence_source_ids = [str(item.get("source_id")) for item in evidence_synthesis if item.get("source_id")]
    panel_source_ids = list(dict.fromkeys(source_ids + pubmed_source_ids + evidence_source_ids))
    state_considerations = _state_considerations(summary)
    management_considerations = anatomy_management.get("management_considerations") or []
    red_flags_combined = (
        list(red_flags.get("general") or [])
        + list(red_flags.get("anatomy_specific") or [])
        + list(anatomy_management.get("red_flags_or_escalation_triggers") or [])
    )
    followup_considerations = anatomy_management.get("followup_or_imaging_context") or []
    management_context = _management_context_items(
        [
            anatomy_management["case_adjustment"],
            anatomy_management["anatomy_adjustment"],
        ]
        + state_considerations
        + list(anatomy_management.get("clinical_localizers") or [])
        + management_considerations
        + followup_considerations,
        source_ids,
    )
    clinical_information_not_available = _clinical_information_not_available(summary, label)
    guidance_traceability = _guidance_traceability(
        management_context,
        case_suggestions,
        pubmed_synthesis,
        source_ids,
    )
    return {
        "version": TREATMENT_GUIDANCE_VERSION,
        "guidance_level": "clinician_treatment_orientation",
        "management_scenario": scenario,
        "management_guidance_strength": scenario["guidance_strength"],
        "purpose": "Evidence-linked management context for clinician review, not a treatment plan.",
        "app_label": MANAGEMENT_GUIDANCE_LABEL,
        "button_label": MANAGEMENT_GUIDANCE_LABEL,
        "panel_title": "Evidence-linked management context for clinician review",
        "panel_badge": "Clinician-only context",
        "panel_warning": "Not a treatment plan",
        "safety_disclaimer": MANAGEMENT_SAFETY_DISCLAIMER,
        "llm_used": False,
        "rag_used": False,
        "not_a_diagnosis": True,
        "not_patient_specific_treatment": True,
        "is_treatment_plan": False,
        "audience": "clinician_only",
        "human_verification_required": True,
        "anatomy_context": {
            "label": summary.anatomy_label,
            "parent_label": summary.anatomy_parent_label,
            "display_label": _display_label(label),
            "output_type": summary.anatomy_output_type,
        },
        "pipeline_context": {
            "state": _state(summary),
            "scenario_id": scenario["scenario_id"],
            "guidance_strength": scenario["guidance_strength"],
            "accepted_candidate_count": summary.accepted_candidate_count,
            "stage2c_decision": summary.stage2c_decision,
            "primary_candidate_confidence": summary.primary_candidate_confidence,
            "anatomy_confidence": summary.anatomy_confidence,
        },
        "template": {
            "version": ANATOMY_MANAGEMENT_TEMPLATE_VERSION,
            "template_id": template.get("template_id"),
            "display_label": _display_label(label),
            "inherited_from": template.get("inherited_from"),
            "review_focus": template.get("review_focus"),
        },
        "anatomy_specific_management_v3": anatomy_management,
        "state_based_considerations": state_considerations,
        "clinical_localizers": anatomy_management.get("clinical_localizers") or [],
        "clinical_information_not_available_to_ai": clinical_information_not_available,
        "initial_management_considerations": management_considerations,
        "red_flags_do_not_miss": red_flags_combined,
        "imaging_or_followup_considerations": followup_considerations,
        "specialist_review_considerations": [
            item for item in followup_considerations if "specialist" in item.lower()
        ],
        "management_guidance": {
            "version": MANAGEMENT_GUIDANCE_VERSION,
            "enabled": True,
            "label": MANAGEMENT_GUIDANCE_LABEL,
            "button_label": MANAGEMENT_GUIDANCE_LABEL,
            "panel_title": "Evidence-linked management context for clinician review",
            "panel_badge": "Clinician-only context",
            "panel_warning": "Not a treatment plan",
            "audience": "clinician_only",
            "is_treatment_plan": False,
            "safety_disclaimer": MANAGEMENT_SAFETY_DISCLAIMER,
            "anatomy_area": _display_label(label),
            "case_status": _state(summary),
            "case_scenario": scenario,
            "guidance_strength": scenario["guidance_strength"],
            "why_this_section_is_shown": scenario["why_this_section_is_shown"],
            "management_direction": management_direction,
            "case_management_suggestions": case_suggestions,
            "anatomy_specific_management": anatomy_management,
            "management_context": management_context,
            "guidance_traceability": guidance_traceability,
            "evidence_rag_synthesis": evidence_synthesis,
            "pubmed_rag_synthesis": pubmed_synthesis,
            "evidence_rag_policy": {
                "prefers_pubmed_claims_when_available": True,
                "uses_curated_guideline_claims_when_pubmed_unavailable": True,
                "raw_abstracts_exposed": False,
                "uses_only_allowed_locked_claims": True,
                "no_new_medical_claims_created_in_display": True,
            },
            "pubmed_rag_policy": {
                "enabled_when_pubmed_claims_available": True,
                "raw_abstracts_exposed": False,
                "uses_only_allowed_claims": True,
                "treatment_claims_prioritized": True,
                "no_new_medical_claims_created_in_display": True,
            },
            "clinical_inputs_required": _clinical_inputs_required(label),
            "clinical_information_not_available_to_ai": clinical_information_not_available,
            "red_flags": red_flags_combined[:8],
            "ai_must_not_decide": _ai_must_not_decide(label),
            "status_specific_cautions": _status_specific_cautions(summary),
            "sources": _source_cards(sources, panel_source_ids),
            "source_traceability": {
                "source_linked_management_item_count": sum(
                    1 for item in management_context if item.get("source_ids")
                ),
                "evidence_rag_synthesis_item_count": len(evidence_synthesis),
                "pubmed_rag_synthesis_item_count": len(pubmed_synthesis),
                "pubmed_treatment_claim_count": sum(
                    1 for item in pubmed_synthesis if item.get("allowed_for_treatment_guidance") is True
                ),
                "source_count": len(_source_cards(sources, panel_source_ids)),
                "curated_guidelines_preferred": True,
                "pubmed_live_required": False,
                "pubmed_can_be_cached_or_live_research_mode": True,
            },
            "limitations": [
                "This is not patient-specific medical advice.",
                "This is not a treatment plan.",
                "PubMed or guideline metadata may provide context rather than complete clinical protocols.",
                "Local clinical protocol and qualified clinician judgment override this tool.",
            ],
        },
        "patient_safe_advice": [
            "This prototype cannot decide treatment; a clinician should interpret the image together with symptoms and examination.",
            "Worsening pain, deformity, open injury concern, or neurovascular symptoms should prompt urgent clinician review.",
            "If symptoms persist despite low AI support, the result should not be treated as injury exclusion or clearance or reassurance.",
        ],
        "blocked_outputs": _blocked_outputs(label),
        "source_policy": {
            "source_ids_are_context_only": True,
            "guideline_or_rag_claims_must_be_approved_before_llm_use": True,
            "pubmed_literature_is_supporting_context_not_standalone_treatment_protocol": True,
            "future_llm_writer_may_rephrase_only_allowed_fields": True,
        },
        "supporting_source_ids": source_ids,
    }
