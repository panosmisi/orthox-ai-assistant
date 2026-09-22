from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List


UNIFIED_CONTRACT_VERSION = "unified_xray_analysis_contract_v1"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_text(value: Any) -> str | None:
    value_f = _as_float(value)
    if value_f is None:
        return None
    return f"{value_f:.1f}%"


def _score_to_percent(value: Any) -> float | None:
    value_f = _as_float(value)
    if value_f is None:
        return None
    if value_f <= 1.0:
        return round(value_f * 100.0, 3)
    return round(value_f, 3)


def _clean_list(items: Any, limit: int | None = None) -> List[Any]:
    if not isinstance(items, list):
        return []
    clean = [item for item in items if item is not None and item != ""]
    return clean[:limit] if limit else clean


def _candidate_count_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"retained\s+(\d+)\s+possible fracture candidate", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if "candidate" in text.lower():
        return 1
    return None


def _support_card(cards: Iterable[Dict[str, Any]], card_id: str) -> Dict[str, Any] | None:
    for card in cards:
        if card.get("id") == card_id:
            return card
    return None


def _not_run(reason: str) -> Dict[str, Any]:
    return {
        "ran": False,
        "status": "not_run",
        "not_run_reason": reason,
    }


def _not_available(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "not_available_reason": reason,
    }


def _candidate_xyxy(candidate: Dict[str, Any]) -> List[float] | None:
    bbox = candidate.get("bbox") or {}
    xyxy = bbox.get("xyxy_pixels")
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_intersection(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: List[float], b: List[float]) -> float:
    inter = _box_intersection(a, b)
    denom = _box_area(a) + _box_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def _box_containment(a: List[float], b: List[float]) -> float:
    inter = _box_intersection(a, b)
    smaller = min(_box_area(a), _box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def _grouping_relation(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a conservative grouping relation, or None if candidates stay separate.

    This is intentionally overlap-first. It does not merge candidates merely
    because they are nearby, since nearby boxes can represent different small
    fractures.
    """
    box_a = _candidate_xyxy(a)
    box_b = _candidate_xyxy(b)
    if not box_a or not box_b:
        return None
    iou = _box_iou(box_a, box_b)
    containment = _box_containment(box_a, box_b)
    reasons: List[str] = []
    if iou >= 0.35:
        reasons.append("strong_bbox_overlap_iou_gte_0p35")
    if containment >= 0.70:
        reasons.append("one_candidate_largely_contained_in_the_other_gte_70pct")
    if not reasons:
        return None
    return {
        "candidate_a": a.get("candidate_id"),
        "candidate_b": b.get("candidate_id"),
        "iou": round(iou, 6),
        "containment_min_area": round(containment, 6),
        "reasons": reasons,
    }


def _candidate_support_for_grouping(candidate: Dict[str, Any]) -> float:
    stage1 = candidate.get("stage1") or {}
    verifier = candidate.get("stage2a_verifier") or {}
    values = [
        _as_float(stage1.get("detector_support_percent")),
        _as_float(verifier.get("mean_support_percent")),
    ]
    votes = verifier.get("keep_votes")
    total = verifier.get("total_votes")
    try:
        if total:
            values.append(float(votes or 0) / float(total) * 100.0)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return max([v for v in values if v is not None] or [0.0])


def _union_bbox(candidates: List[Dict[str, Any]]) -> List[float] | None:
    boxes = [_candidate_xyxy(candidate) for candidate in candidates]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def build_conservative_finding_groups(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build non-destructive, conservative finding groups.

    Every candidate remains in `findings.candidates`. Groups only organize
    candidates for display/audit when bbox geometry strongly supports that
    they are probably describing the same broader finding.
    """
    n = len(candidates)
    relations: List[Dict[str, Any]] = []
    parent = list(range(n))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            relation = _grouping_relation(candidates[i], candidates[j])
            if relation:
                relations.append(relation)
                union(i, j)

    components: Dict[int, List[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    groups: List[Dict[str, Any]] = []
    ungrouped: List[str] = []
    group_index = 1
    for indices in components.values():
        if len(indices) < 2:
            candidate_id = candidates[indices[0]].get("candidate_id")
            if candidate_id:
                ungrouped.append(candidate_id)
            continue

        group_candidates = [candidates[idx] for idx in indices]
        group_candidate_ids = [str(candidate.get("candidate_id")) for candidate in group_candidates]
        group_relations = [
            relation
            for relation in relations
            if relation.get("candidate_a") in group_candidate_ids and relation.get("candidate_b") in group_candidate_ids
        ]
        representative = max(group_candidates, key=_candidate_support_for_grouping)
        max_iou = max((float(r.get("iou") or 0.0) for r in group_relations), default=0.0)
        max_containment = max((float(r.get("containment_min_area") or 0.0) for r in group_relations), default=0.0)
        reasons = sorted({reason for relation in group_relations for reason in relation.get("reasons", [])})
        groups.append(
            {
                "group_id": f"F{group_index}",
                "group_type": "conservative_overlap_group",
                "candidate_ids": group_candidate_ids,
                "representative_candidate_id": representative.get("candidate_id"),
                "group_bbox_xyxy_pixels": _union_bbox(group_candidates),
                "grouping_confidence": "high" if max_iou >= 0.35 or max_containment >= 0.80 else "moderate",
                "max_pairwise_iou": round(max_iou, 6),
                "max_pairwise_containment_min_area": round(max_containment, 6),
                "reasons": reasons,
                "relations": group_relations,
                "safety_notes": [
                    "Non-destructive grouping: all original candidate boxes remain available and must be reviewed individually.",
                    "Grouping is display/audit organization only, not a medical claim that candidates are the same fracture.",
                    "Nearby candidates without strong overlap are deliberately left ungrouped to avoid hiding separate small fractures.",
                ],
            }
        )
        group_index += 1

    return {
        "available": True,
        "status": "conservative_non_destructive_grouping_v1",
        "policy": {
            "strong_iou_threshold": 0.35,
            "containment_min_area_threshold": 0.70,
            "close_without_overlap_grouping": False,
            "destructive_merge": False,
            "drops_candidates": False,
        },
        "groups": groups,
        "ungrouped_candidate_ids": sorted(ungrouped),
        "group_count": len(groups),
        "grouped_candidate_count": sum(len(group["candidate_ids"]) for group in groups),
        "candidate_count": n,
        "global_safety_note": (
            "Finding groups are conservative display organization. They do not delete, suppress, or clinically merge candidate boxes."
        ),
    }


def attach_conservative_finding_groups(record: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(record)
    findings = out.setdefault("findings", {})
    candidates = findings.get("candidates") or []
    findings["finding_grouping"] = build_conservative_finding_groups(candidates)
    return out


def _bbox_from_visual(visual: Dict[str, Any]) -> Dict[str, Any]:
    bbox = visual.get("primary_bbox") or {}
    if not visual.get("show_primary_bbox") or not bbox.get("available"):
        return {
            "available": False,
            "xyxy_pixels": None,
            "bbox_source": None,
            "detector_support_percent": None,
            "bbox_quality_percent_proxy": None,
            "localization_evidence_type": None,
            "independent_localization_probability_available": False,
            "localization_note": None,
        }
    return {
        "available": True,
        "xyxy_pixels": bbox.get("xyxy_pixels"),
        "bbox_source": bbox.get("bbox_source"),
        "detector_support_percent": _score_to_percent(bbox.get("detector_support_percent")),
        "bbox_quality_percent_proxy": _score_to_percent(bbox.get("bbox_quality_percent_proxy")),
        "localization_evidence_type": bbox.get("localization_evidence_type"),
        "independent_localization_probability_available": bool(
            bbox.get("independent_localization_probability_available")
        ),
        "localization_note": bbox.get("localization_note"),
        "review_note": bbox.get("review_note"),
    }


def _candidate_from_external(candidate: Dict[str, Any], index: int) -> Dict[str, Any]:
    detector_conf = _score_to_percent(candidate.get("confidence"))
    verifier_mean = _score_to_percent(candidate.get("verifier_prob_mean"))
    return {
        "candidate_id": f"C{index}",
        "label": candidate.get("label") or "fracture",
        "bbox": {
            "available": bool(candidate.get("xyxy_pixels")),
            "xyxy_pixels": candidate.get("xyxy_pixels"),
            "xywhn": candidate.get("xywhn"),
            "bbox_source": "live_stage1_soft_merged_candidate",
            "detector_support_percent": detector_conf,
            "bbox_quality_percent_proxy": detector_conf,
            "localization_note": (
                "YOLO candidate box from live external-image adapter. "
                "This is a review target, not a diagnostic boundary."
            ),
        },
        "stage1": {
            "detector_support_percent": detector_conf,
            "detector_sources": candidate.get("detector_sources") or [candidate.get("detector_source")],
            "source_candidate_count": candidate.get("source_candidate_count"),
            "merge_method": candidate.get("merge_method"),
        },
        "stage2a_verifier": {
            "available": True,
            "kept": bool(candidate.get("verifier_keep")),
            "keep_votes": candidate.get("verifier_keep_votes"),
            "total_votes": candidate.get("verifier_total_votes"),
            "min_votes": candidate.get("verifier_min_votes"),
            "mean_support_percent": verifier_mean,
            "seed_probabilities_percent": [
                _score_to_percent(prob) for prob in candidate.get("verifier_probs") or []
            ],
            "policy": candidate.get("verifier_policy"),
        },
        "artifact_suppression": _not_available("stage2a5_artifact_suppression_not_run_in_external_live_demo"),
    }


def _final_state_from_counts(accepted_count: int, stage2c_decision: str | None = None) -> Dict[str, Any]:
    if accepted_count > 0:
        return {
            "state_id": "candidate_retained",
            "status_tone": "possible_fracture_candidate",
            "headline": "Possible fracture candidate retained.",
            "requires_human_review": True,
        }
    if stage2c_decision in {"suspicious", "uncertain"}:
        return {
            "state_id": "image_level_warning_without_bbox",
            "status_tone": "caution_without_localized_box",
            "headline": "Image-level caution signal without retained box.",
            "requires_human_review": True,
        }
    return {
        "state_id": "no_high_confidence_candidate_retained",
        "status_tone": "no_high_confidence_candidate_with_review_required",
        "headline": "No high-confidence marked region retained.",
        "requires_human_review": True,
    }


def _base_contract(
    *,
    image_id: str | None,
    image_path: str | None,
    input_source: str,
    analysis_mode: str,
    source_payload_type: str,
) -> Dict[str, Any]:
    return {
        "version": UNIFIED_CONTRACT_VERSION,
        "purpose": "Single stable analysis schema for FracAtlas validation images and external user-supplied X-rays.",
        "creates_new_medical_claims": False,
        "case": {
            "image_id": image_id,
            "image_path": image_path,
            "input_source": input_source,
            "analysis_mode": analysis_mode,
            "source_payload_type": source_payload_type,
            "out_of_distribution": input_source != "fracatlas_validation",
            "contract_completeness": "partial",
        },
        "input": {
            "image_available": bool(image_path),
            "image_quality": {
                "available": False,
                "quality_label": None,
                "notes": [],
            },
            "warnings": [],
        },
        "stages": {
            "stage1_detection": _not_run("not_populated"),
            "stage2a_verifier": _not_run("not_populated"),
            "stage2a5_artifact_suppression": _not_run("not_populated"),
            "stage2b_anatomy": _not_run("not_populated"),
            "stage2c_image_level_safety": _not_run("not_populated"),
            "stage3_care_guidance": _not_run("not_populated"),
        },
        "findings": {
            "finding_grouping": {
                "available": False,
                "status": "not_implemented_in_contract_v1",
                "note": "Box-level candidates are listed; finding-level grouping is a planned next functional step.",
            },
            "candidates": [],
        },
        "final_assessment": _final_state_from_counts(0),
        "display_text": {
            "clinician_summary": None,
            "patient_note": None,
            "safety_footer": (
                "This is not a standalone diagnosis. Human verification is required. "
                "Model scores are technical support signals, not clinical probabilities."
            ),
        },
        "review": {
            "top_focus": [],
            "top_radiograph_checks": [],
            "top_context_questions": [],
            "uncertainty_notes": [],
            "do_not_infer": [],
        },
        "care_guidance": {
            "guidance_level": None,
            "initial_management_considerations": [],
            "red_flags_do_not_miss": [],
            "imaging_or_followup_considerations": [],
            "blocked_outputs": [],
        },
        "evidence": {
            "status": None,
            "source_summary": {},
            "displayed_sources": [],
            "displayed_allowed_claims": [],
        },
        "llm": {
            "writer_type": None,
            "llm_used": False,
            "guardrail_mode": None,
            "validation": {},
        },
        "safety": {
            "not_standalone_diagnosis": True,
            "human_verification_required": True,
            "scores_are_not_clinical_probabilities": True,
            "stage3_does_not_override_stage1_stage2": True,
        },
        "artifacts": {},
        "contract_audit": {
            "schema_keys_present": True,
            "missing_source_fields": [],
            "normalization_notes": [],
        },
    }


def build_unified_from_final_display(
    payload: Dict[str, Any],
    *,
    input_source: str = "fracatlas_validation",
    analysis_mode: str = "locked_full_pipeline",
) -> Dict[str, Any]:
    """Normalize an existing Stage 3 final-display payload into the unified analysis contract."""
    case = payload.get("case") or {}
    visual = payload.get("visual") or {}
    confidence = payload.get("confidence") or {}
    anatomy = payload.get("anatomy") or {}
    review = payload.get("review") or {}
    care = payload.get("care_guidance") or {}
    evidence = payload.get("evidence") or {}
    llm = payload.get("llm") or {}
    safety = payload.get("safety") or {}
    text = payload.get("text") or {}
    cards = confidence.get("cards") or []
    state_id = case.get("state_id")
    candidate_count = _candidate_count_from_text(text.get("clinician_summary"))
    if candidate_count is None:
        candidate_count = 1 if state_id == "candidate_retained" else 0

    out = _base_contract(
        image_id=case.get("image_id"),
        image_path=case.get("image_path"),
        input_source=input_source,
        analysis_mode=analysis_mode,
        source_payload_type=payload.get("version") or "stage3_final_display_payload",
    )
    out["case"]["contract_completeness"] = "full_locked_display_contract"
    out["case"]["status_tone"] = case.get("status_tone")

    fracture_card = _support_card(cards, "fracture_candidate_support")
    localization_card = _support_card(cards, "localization_evidence")
    anatomy_card = _support_card(cards, "anatomy_support")
    stage2c_card = _support_card(cards, "stage2c_caution_support")

    out["stages"]["stage1_detection"] = {
        "ran": True,
        "status": "available_in_locked_pipeline",
        "raw_candidate_count": None,
        "merged_candidate_count": None,
        "retained_candidate_count": candidate_count,
        "top_detector_support_percent": (fracture_card or {}).get("support_percent"),
        "policy": "locked_stage1_union_policy_as_encoded_in_source_payload",
        "missing_detail_reason": "final_display_payload_does_not_store_raw_stage1_candidate_table",
    }
    out["stages"]["stage2a_verifier"] = {
        "ran": True,
        "status": "available_in_locked_pipeline",
        "accepted_candidate_count": candidate_count,
        "rejected_candidate_count": None,
        "policy": "locked_verifier_policy_as_encoded_in_source_payload",
        "candidate_votes_available": False,
        "missing_detail_reason": "final_display_payload_does_not_store_verifier_seed_votes",
    }
    out["stages"]["stage2a5_artifact_suppression"] = {
        "ran": True,
        "status": "available_in_locked_pipeline",
        "artifact_rejected_candidate_count": None,
        "policy": "locked_production_strict_artifact_policy_as_encoded_in_source_payload",
        "missing_detail_reason": "final_display_payload_does_not_store_artifact_audit_table",
    }
    out["stages"]["stage2b_anatomy"] = {
        "ran": True,
        "status": "available",
        "selected_label": anatomy.get("selected_label"),
        "selected_display_label": anatomy.get("selected_display_label"),
        "selected_support_percent": anatomy.get("selected_support_percent"),
        "parent_label": anatomy.get("parent_label"),
        "parent_display_label": anatomy.get("parent_display_label"),
        "output_type": anatomy.get("output_type"),
        "top_probabilities": anatomy.get("top_probabilities") or [],
        "support_card_percent": (anatomy_card or {}).get("support_percent"),
    }
    heatmap = visual.get("stage2c_heatmap") or {}
    out["stages"]["stage2c_image_level_safety"] = {
        "ran": state_id == "image_level_warning_without_bbox",
        "status": "available" if state_id == "image_level_warning_without_bbox" else "not_triggered_or_not_needed",
        "decision": "suspicious_or_uncertain" if state_id == "image_level_warning_without_bbox" else None,
        "suspicious_support_percent": (stage2c_card or {}).get("support_percent"),
        "heatmap": {
            "available": bool(visual.get("heatmap_available")),
            "show_by_default": bool(visual.get("show_heatmap_by_default")),
            "warning": visual.get("heatmap_warning"),
            "raw": heatmap,
        },
        "not_run_reason": (
            None
            if state_id == "image_level_warning_without_bbox"
            else "retained_candidate_or_no_high_confidence_candidate_path"
        ),
    }
    out["stages"]["stage3_care_guidance"] = {
        "ran": True,
        "status": "available",
        "guidance_level": care.get("guidance_level"),
        "evidence_status": evidence.get("status"),
        "llm_used": bool(llm.get("llm_used")),
    }

    bbox = _bbox_from_visual(visual)
    if candidate_count > 0:
        out["findings"]["candidates"].append(
            {
                "candidate_id": "C1",
                "label": "fracture",
                "bbox": bbox,
                "stage1": {
                    "detector_support_percent": bbox.get("detector_support_percent")
                    or (fracture_card or {}).get("support_percent"),
                    "detector_sources": [],
                    "source_candidate_count": None,
                    "merge_method": None,
                },
                "stage2a_verifier": {
                    "available": True,
                    "kept": True,
                    "keep_votes": None,
                    "total_votes": None,
                    "min_votes": None,
                    "mean_support_percent": None,
                    "seed_probabilities_percent": [],
                    "policy": "locked_verifier_policy_as_encoded_in_source_payload",
                },
                "artifact_suppression": {
                    "available": True,
                    "rejected": False,
                    "policy": "locked_production_strict_artifact_policy_as_encoded_in_source_payload",
                },
            }
        )
        if candidate_count > 1:
            out["contract_audit"]["normalization_notes"].append(
                "final_display_payload_exposes_primary_bbox_only; additional retained boxes are not individually reconstructed"
            )

    out["final_assessment"] = {
        "state_id": state_id,
        "status_tone": case.get("status_tone"),
        "headline": case.get("headline"),
        "subheadline": case.get("subheadline"),
        "primary_action": case.get("primary_action"),
        "requires_human_review": True,
    }
    out["display_text"] = {
        "clinician_summary": text.get("clinician_summary"),
        "patient_note": text.get("patient_note"),
        "safety_footer": text.get("safety_footer"),
    }
    out["review"] = {
        "top_focus": _clean_list(review.get("top_focus")),
        "top_radiograph_checks": _clean_list(review.get("top_radiograph_checks")),
        "top_context_questions": _clean_list(review.get("top_context_questions")),
        "uncertainty_notes": _clean_list(review.get("uncertainty_notes")),
        "do_not_infer": _clean_list(review.get("do_not_infer")),
    }
    out["care_guidance"] = {
        "guidance_level": care.get("guidance_level"),
        "initial_management_considerations": _clean_list(care.get("initial_management_considerations")),
        "red_flags_do_not_miss": _clean_list(care.get("red_flags_do_not_miss")),
        "imaging_or_followup_considerations": _clean_list(care.get("imaging_or_followup_considerations")),
        "blocked_outputs": _clean_list(care.get("blocked_outputs")),
    }
    out["evidence"] = {
        "status": evidence.get("status"),
        "source_summary": evidence.get("source_summary") or {},
        "displayed_sources": _clean_list(evidence.get("displayed_sources")),
        "displayed_allowed_claims": _clean_list(evidence.get("displayed_allowed_claims")),
    }
    out["llm"] = {
        "writer_type": llm.get("writer_type"),
        "llm_used": bool(llm.get("llm_used")),
        "guardrail_mode": llm.get("guardrail_mode"),
        "validation": llm.get("validation") or {},
    }
    out["safety"].update({key: bool(value) for key, value in safety.items()})
    return attach_conservative_finding_groups(out)


def build_unified_from_external_stage2a(
    stage2a: Dict[str, Any],
    *,
    input_source: str = "external_user_supplied_image",
    analysis_mode: str = "external_live_stage1_stage2a_partial",
) -> Dict[str, Any]:
    """Build the same unified contract from a live external Stage 2A verifier JSON."""
    image_id = stage2a.get("image_id")
    image_path = stage2a.get("image_path")
    accepted = stage2a.get("accepted_candidates") or []
    rejected = stage2a.get("rejected_candidates") or []
    accepted_sorted = sorted(
        accepted,
        key=lambda c: (
            _as_float(c.get("confidence")) or 0.0,
            int(c.get("verifier_keep_votes") or 0),
            _as_float(c.get("verifier_prob_mean")) or 0.0,
        ),
        reverse=True,
    )
    rejected_sorted = sorted(
        rejected,
        key=lambda c: (
            _as_float(c.get("confidence")) or 0.0,
            int(c.get("verifier_keep_votes") or 0),
            _as_float(c.get("verifier_prob_mean")) or 0.0,
        ),
        reverse=True,
    )
    candidates = [
        _candidate_from_external(candidate, index)
        for index, candidate in enumerate(accepted_sorted + rejected_sorted, start=1)
    ]
    top_detector = max((_score_to_percent(c.get("confidence")) or 0.0 for c in accepted_sorted), default=None)
    mean_verifier_values = [
        _score_to_percent(c.get("verifier_prob_mean"))
        for c in accepted_sorted
        if _score_to_percent(c.get("verifier_prob_mean")) is not None
    ]
    mean_verifier = (
        round(sum(mean_verifier_values) / len(mean_verifier_values), 3) if mean_verifier_values else None
    )

    out = _base_contract(
        image_id=image_id,
        image_path=image_path,
        input_source=input_source,
        analysis_mode=analysis_mode,
        source_payload_type=stage2a.get("stage2a_live_version") or "stage2a_live_json",
    )
    out["case"]["contract_completeness"] = "partial_external_live_contract"
    out["input"]["warnings"].append(
        "External image is outside the locked FracAtlas validation distribution; scores are technical support signals only."
    )

    out["stages"]["stage1_detection"] = {
        "ran": True,
        "status": "available_live_external_adapter",
        "raw_candidate_count": stage2a.get("raw_candidate_count"),
        "merged_candidate_count": stage2a.get("raw_candidate_count"),
        "retained_candidate_count": stage2a.get("accepted_candidate_count"),
        "top_detector_support_percent": top_detector,
        "policy": stage2a.get("input_stage1_live_version"),
        "missing_detail_reason": None,
    }
    out["stages"]["stage2a_verifier"] = {
        "ran": True,
        "status": "available_live_external_adapter",
        "accepted_candidate_count": stage2a.get("accepted_candidate_count"),
        "rejected_candidate_count": stage2a.get("rejected_candidate_count"),
        "policy": (stage2a.get("policy") or {}).get("ensemble"),
        "min_votes": (stage2a.get("policy") or {}).get("min_votes"),
        "candidate_votes_available": True,
        "mean_support_percent_across_accepted_candidates": mean_verifier,
    }
    out["stages"]["stage2a5_artifact_suppression"] = _not_run(
        "stage2a5_artifact_suppression_not_yet_wired_for_external_live_adapter"
    )
    out["stages"]["stage2b_anatomy"] = _not_run(
        "stage2b_live_anatomy_adapter_not_yet_wired_for_external_image_contract"
    )
    out["stages"]["stage2c_image_level_safety"] = {
        "ran": False,
        "status": "not_needed_or_not_wired",
        "decision": None,
        "suspicious_support_percent": None,
        "heatmap": {
            "available": False,
            "show_by_default": False,
            "warning": None,
            "raw": {},
        },
        "not_run_reason": "retained_candidates_present; stage2c_external_live_adapter_not_yet_wired",
    }
    out["stages"]["stage3_care_guidance"] = {
        "ran": True,
        "status": "conservative_external_demo_guidance",
        "guidance_level": "clinician_review_orientation",
        "evidence_status": "not_retrieved_for_external_demo",
        "llm_used": False,
    }
    out["findings"]["candidates"] = candidates
    out["final_assessment"] = _final_state_from_counts(len(accepted_sorted))
    out["final_assessment"].update(
        {
            "headline": (
                "External X-ray: fracture-candidate signal retained by Stage 1 and Stage 2A."
                if accepted_sorted
                else "External X-ray: no verified candidate retained by the partial live adapter."
            ),
            "subheadline": (
                "This is a partial external live analysis; full locked Stage 2B/2C contract is not yet wired."
            ),
            "primary_action": "Review original X-ray and all retained candidate regions.",
        }
    )
    out["display_text"] = {
        "clinician_summary": (
            f"The external-image adapter retained {len(accepted_sorted)} possible fracture candidate"
            f"{'' if len(accepted_sorted) == 1 else 's'} after verifier review. "
            "This is a screening signal requiring clinician verification on the full radiograph."
        ),
        "patient_note": (
            "The AI marked one or more areas for clinician review. This does not confirm a fracture by itself, "
            "but it means the image should not be treated as automatically clear."
        ),
        "safety_footer": (
            "This is not a standalone diagnosis. Human verification is required. "
            "External-image scores are not calibrated clinical probabilities."
        ),
    }
    out["review"] = {
        "top_focus": [
            "Review each retained candidate box on the original full radiograph.",
            "Check whether multiple boxes represent one large injury pattern or separate findings.",
            "Correlate with symptoms, mechanism of injury, swelling, deformity, and examination findings.",
        ],
        "top_radiograph_checks": [
            "Inspect cortical disruption, displacement, angulation, and adjacent joint involvement.",
            "Check whether the image coverage is sufficient for the suspected injury region.",
        ],
        "top_context_questions": [
            "Is there deformity, open wound, neurovascular symptom, severe swelling, or high-energy trauma?",
            "Does the pain/tenderness match the marked region?",
        ],
        "uncertainty_notes": [
            "External-image distribution has not been benchmarked like FracAtlas validation.",
            "Multiple boxes may over-represent one fracture event until finding-level grouping is implemented.",
        ],
        "do_not_infer": [
            "Do not treat the AI output as a final diagnosis.",
            "Do not infer fracture type, stability, treatment, or surgical need from this prototype output alone.",
            "Do not treat absence of an unmarked region as proof of no injury.",
        ],
    }
    out["care_guidance"] = {
        "guidance_level": "clinician_review_orientation",
        "initial_management_considerations": [
            "Consider urgent clinician review when the radiograph shows a strong retained fracture-candidate signal.",
            "Until reviewed, avoid using the AI score to make weight-bearing, immobilization, or discharge decisions.",
            "Use local trauma/orthopedic pathways for analgesia, immobilization, imaging, and referral decisions.",
        ],
        "red_flags_do_not_miss": [
            "open injury concern",
            "visible deformity",
            "neurovascular symptoms such as numbness, weakness, pallor, or reduced pulses",
            "severe or increasing pain/swelling",
        ],
        "imaging_or_followup_considerations": [
            "Consider additional views or clinician-directed imaging if the fracture extent or joint involvement is unclear.",
            "Consider specialist review when displacement, instability, open injury, neurovascular concern, or severe deformity is suspected.",
        ],
        "blocked_outputs": [
            "Do not state that a fracture is confirmed or excluded.",
            "Do not prescribe medication, dosage, casting, reduction, surgery, or follow-up timing.",
            "Do not provide return-to-activity or discharge decisions.",
        ],
    }
    out["evidence"] = {
        "status": "not_retrieved_for_external_demo",
        "source_summary": {},
        "displayed_sources": [],
        "displayed_allowed_claims": [],
    }
    out["llm"] = {
        "writer_type": "deterministic_template",
        "llm_used": False,
        "guardrail_mode": "external_demo_no_llm",
        "validation": {},
    }
    out["contract_audit"]["normalization_notes"].extend(
        [
            "external_stage2a_source_normalized_to_unified_contract",
            "stage2b_stage2c_stage2a5_marked_not_run_instead_of_filled_with_manual_claims",
        ]
    )
    return attach_conservative_finding_groups(out)


def _candidate_from_pipeline(candidate: Dict[str, Any], index: int) -> Dict[str, Any]:
    summary = candidate.get("candidate_probability_summary") or {}
    bbox_conf = candidate.get("bbox_confidence") or {}
    artifact = candidate.get("artifact_suppression") or {}
    return {
        "candidate_id": f"C{index}",
        "label": candidate.get("label") or "fracture",
        "bbox": {
            "available": bool(candidate.get("xyxy_pixels")),
            "xyxy_pixels": candidate.get("xyxy_pixels"),
            "xywhn": candidate.get("xywhn"),
            "bbox_source": candidate.get("candidate_source"),
            "detector_support_percent": _score_to_percent(summary.get("detector_fracture_candidate_percent")),
            "bbox_quality_percent_proxy": _score_to_percent(bbox_conf.get("bbox_quality_percent_proxy")),
            "localization_note": bbox_conf.get("localization_note"),
        },
        "stage1": {
            "detector_support_percent": _score_to_percent(summary.get("detector_fracture_candidate_percent")),
            "detector_sources": candidate.get("detector_sources") or [],
            "source_candidate_count": candidate.get("source_candidate_count"),
            "merge_method": candidate.get("merge_method"),
        },
        "stage2a_verifier": {
            "available": True,
            "kept": bool(candidate.get("verifier_keep")),
            "keep_votes": candidate.get("verifier_keep_votes"),
            "total_votes": candidate.get("verifier_total_votes"),
            "min_votes": candidate.get("verifier_min_votes"),
            "mean_support_percent": _score_to_percent(summary.get("verifier_fracture_probability_percent_mean")),
            "seed_probabilities_percent": [
                _score_to_percent(prob) for prob in (candidate.get("verifier_probs") or [])
            ],
            "policy": candidate.get("verifier_policy"),
        },
        "artifact_suppression": {
            "available": isinstance(artifact, dict),
            "rejected": bool(artifact.get("suppressed")),
            "policy": artifact.get("rule"),
            "reasons": artifact.get("reasons") or [],
            "flags": artifact.get("flags") or [],
            "features": artifact.get("features") or {},
        },
    }


def build_unified_from_live_full_pipeline(
    pipeline_json: Dict[str, Any],
    final_display_payload: Dict[str, Any] | None = None,
    *,
    input_source: str = "external_user_supplied_image",
    analysis_mode: str = "live_full_pipeline",
) -> Dict[str, Any]:
    """Normalize a true live Stage 1->3 pipeline JSON into the same unified contract."""
    final = pipeline_json.get("final_assessment") or {}
    stage2b = pipeline_json.get("stage2b_anatomy") or {}
    anatomy_policy = stage2b.get("policy") or {}
    fine = stage2b.get("fine_classifier") or {}
    stage2c = pipeline_json.get("stage2c_safety_check") or {}
    stage2c_prob = stage2c.get("probability_summary") or {}
    stage2a5 = pipeline_json.get("stage2a5_artifact_suppression") or {}
    display = final_display_payload or {}

    out = _base_contract(
        image_id=pipeline_json.get("image_id"),
        image_path=pipeline_json.get("image_path"),
        input_source=input_source,
        analysis_mode=analysis_mode,
        source_payload_type=pipeline_json.get("pipeline_version") or "live_full_pipeline_json",
    )
    out["case"]["contract_completeness"] = "live_full_pipeline_contract"
    out["case"]["out_of_distribution"] = input_source != "fracatlas_validation"
    if out["case"]["out_of_distribution"]:
        out["input"]["warnings"].append(
            "External image is outside the locked FracAtlas validation distribution; scores are technical support signals only."
        )

    out["stages"]["stage1_detection"] = {
        "ran": True,
        "status": "available_live_adapter",
        "raw_candidate_count": final.get("raw_fracture_candidate_count"),
        "merged_candidate_count": pipeline_json.get("stage1_stage2a_raw_fracture_candidate_count"),
        "retained_candidate_count": final.get("verified_fracture_candidate_count"),
        "top_detector_support_percent": _score_to_percent(
            ((final.get("primary_fracture_detection") or {}).get("candidate_probability_summary") or {}).get(
                "detector_fracture_candidate_percent"
            )
        ),
        "policy": "run4_0p25_run5_0p15_softmerge_iou_0p40",
        "missing_detail_reason": None,
    }
    out["stages"]["stage2a_verifier"] = {
        "ran": True,
        "status": "available_live_adapter",
        "accepted_candidate_count": final.get("verified_fracture_candidate_count"),
        "rejected_candidate_count": final.get("verifier_rejected_fracture_candidate_count"),
        "policy": "runtime_safe_6seed_keep_if_votes_gte_2",
        "candidate_votes_available": True,
    }
    out["stages"]["stage2a5_artifact_suppression"] = {
        "ran": True,
        "status": "available_live_adapter",
        "artifact_rejected_candidate_count": stage2a5.get("artifact_rejected_count"),
        "kept_count_after_artifact_suppression": stage2a5.get("kept_count_after_artifact_suppression"),
        "policy": stage2a5.get("rule"),
    }
    out["stages"]["stage2b_anatomy"] = {
        "ran": True,
        "status": "available_live_adapter",
        "selected_label": anatomy_policy.get("label"),
        "selected_display_label": anatomy_policy.get("label"),
        "selected_support_percent": _score_to_percent(anatomy_policy.get("confidence")),
        "parent_label": anatomy_policy.get("parent_label"),
        "parent_display_label": anatomy_policy.get("parent_label"),
        "output_type": anatomy_policy.get("output_type"),
        "top_probabilities": fine.get("probability_ranking") or [],
        "support_card_percent": _score_to_percent(anatomy_policy.get("confidence")),
    }
    out["stages"]["stage2c_image_level_safety"] = {
        "evaluated": True,
        "ran": bool(stage2c.get("stage2c_was_run")),
        "status": "available_live_adapter" if stage2c else "not_available",
        "decision": stage2c.get("decision"),
        "suspicious_support_percent": _score_to_percent(stage2c_prob.get("mean_suspicious_probability")),
        "heatmap": {
            "available": False,
            "show_by_default": False,
            "warning": None,
            "raw": {},
        },
        "not_run_reason": None if stage2c.get("stage2c_was_run") else stage2c.get("screening_message"),
    }
    out["stages"]["stage3_care_guidance"] = {
        "ran": bool(display),
        "status": "available" if display else "not_available",
        "guidance_level": (display.get("care_guidance") or {}).get("guidance_level"),
        "evidence_status": (display.get("evidence") or {}).get("status"),
        "llm_used": bool((display.get("llm") or {}).get("llm_used")),
    }

    out["findings"]["candidates"] = [
        _candidate_from_pipeline(candidate, index)
        for index, candidate in enumerate(pipeline_json.get("stage1_stage2a_fracture_detections") or [], start=1)
    ]
    out["findings"]["artifact_rejected_candidates"] = [
        _candidate_from_pipeline(candidate, index)
        for index, candidate in enumerate(pipeline_json.get("stage1_stage2a_rejected_fracture_candidates") or [], start=1)
        if (candidate.get("artifact_suppression") or {}).get("suppressed")
    ]

    state_map = {
        "verified_candidate_present": "candidate_retained",
        "no_accepted_candidate_stage2c_suspicious": "image_level_warning_without_bbox",
        "no_accepted_candidate_stage2c_uncertain": "image_level_warning_without_bbox",
        "no_candidate_detected": "no_high_confidence_candidate_retained",
        "no_accepted_candidate_stage2c_probably_clean": "no_high_confidence_candidate_retained",
    }
    out["final_assessment"] = {
        "state_id": state_map.get(final.get("fracture_candidate_status"), final.get("fracture_candidate_status")),
        "status_tone": _status_tone_from_state(state_map.get(final.get("fracture_candidate_status"))),
        "headline": (display.get("case") or {}).get("headline") or final.get("screening_statement"),
        "subheadline": (display.get("case") or {}).get("subheadline"),
        "primary_action": (display.get("case") or {}).get("primary_action"),
        "requires_human_review": True,
    }
    if display:
        out["display_text"] = display.get("text") or out["display_text"]
        out["review"] = display.get("review") or out["review"]
        out["care_guidance"] = display.get("care_guidance") or out["care_guidance"]
        out["evidence"] = display.get("evidence") or out["evidence"]
        out["llm"] = display.get("llm") or out["llm"]
        out["safety"].update(display.get("safety") or {})
    out["contract_audit"]["normalization_notes"].append("live_full_pipeline_normalized_to_unified_contract")
    return attach_conservative_finding_groups(out)


def _status_tone_from_state(state_id: str | None) -> str:
    if state_id == "candidate_retained":
        return "possible_fracture_candidate"
    if state_id == "image_level_warning_without_bbox":
        return "caution_without_localized_box"
    return "no_high_confidence_candidate_with_review_required"


REQUIRED_TOP_LEVEL_KEYS = [
    "version",
    "purpose",
    "case",
    "input",
    "stages",
    "findings",
    "final_assessment",
    "display_text",
    "review",
    "care_guidance",
    "evidence",
    "llm",
    "safety",
    "artifacts",
    "contract_audit",
]

REQUIRED_STAGE_KEYS = [
    "stage1_detection",
    "stage2a_verifier",
    "stage2a5_artifact_suppression",
    "stage2b_anatomy",
    "stage2c_image_level_safety",
    "stage3_care_guidance",
]


def validate_unified_contract(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in record:
            errors.append(f"missing_top_level_key:{key}")
    stages = record.get("stages") or {}
    for key in REQUIRED_STAGE_KEYS:
        if key not in stages:
            errors.append(f"missing_stage_key:{key}")
        elif "ran" not in stages[key]:
            errors.append(f"stage_missing_ran_flag:{key}")
    if record.get("version") != UNIFIED_CONTRACT_VERSION:
        errors.append("wrong_contract_version")
    display_text = record.get("display_text") or {}
    if not display_text.get("safety_footer"):
        errors.append("missing_safety_footer")
    if (record.get("final_assessment") or {}).get("requires_human_review") is not True:
        errors.append("final_assessment_must_require_human_review")
    return errors


def with_artifacts(record: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(record)
    out.setdefault("artifacts", {}).update(artifacts)
    return out
