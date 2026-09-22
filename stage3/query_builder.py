from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


QUERY_STRATEGY_VERSION = "stage3_pubmed_query_strategy_v2"


def _q(text: str, role: str, focus: str, priority: int = 50) -> Dict[str, Any]:
    return {
        "query": text,
        "query_role": role,
        "anatomy_focus": focus,
        "priority": priority,
        "allowed_downstream_use": "source_retrieval_only",
        "not_a_medical_claim": True,
    }


ANATOMY_QUERY_MAP = {
    "wrist_hand": [
        _q("distal radius fracture management review radiograph", "management_context", "distal_radius_ulna", 90),
        _q("scaphoid fracture occult radiograph follow-up review", "occult_followup_context", "scaphoid_carpus", 92),
        _q("wrist fracture radiograph clinical management review", "management_context", "wrist_carpus", 82),
        _q("metacarpal fracture management review radiograph", "management_context", "metacarpal_hand", 72),
        _q("phalangeal fracture radiograph management review", "management_context", "phalanges_hand", 65),
        _q("hand fracture malrotation angulation management review", "red_flag_alignment_context", "hand_alignment", 78),
    ],
    "hand": [
        _q("hand fracture malrotation angulation management review", "red_flag_alignment_context", "hand_alignment", 88),
        _q("metacarpal fracture management review radiograph", "management_context", "metacarpal_hand", 86),
        _q("phalangeal fracture radiograph management review", "management_context", "phalanges_hand", 78),
        _q("wrist fracture radiograph clinical management review", "management_context", "wrist_carpus", 58),
    ],
    "elbow": [
        _q("radial head fracture management review radiograph", "management_context", "radial_head", 92),
        _q("occult elbow fracture fat pad sign radiograph follow-up", "occult_followup_context", "elbow_occult", 90),
        _q("olecranon fracture management review radiograph", "management_context", "olecranon", 74),
        _q("distal humerus supracondylar fracture radiograph management", "management_context", "distal_humerus", 70),
        _q("elbow fracture range of motion clinical review radiograph", "clinical_localizer_context", "elbow_motion", 68),
    ],
    "forearm": [
        _q("radius ulna shaft fracture management review radiograph", "management_context", "radius_ulna_shaft", 90),
        _q("forearm fracture associated elbow wrist injury review", "adjacent_joint_context", "forearm_adjacent_joints", 85),
        _q("Monteggia Galeazzi fracture radiograph management review", "red_flag_alignment_context", "forearm_linked_injury", 82),
        _q("forearm fracture compartment syndrome clinical review", "red_flag_escalation_context", "forearm_swelling", 68),
    ],
    "shoulder_upper_arm": [
        _q("proximal humerus fracture management review radiograph", "management_context", "proximal_humerus", 90),
        _q("shoulder fracture dislocation management review radiograph", "red_flag_alignment_context", "shoulder_dislocation", 88),
        _q("humeral shaft fracture management review radiograph", "management_context", "humeral_shaft", 78),
        _q("clavicle fracture management review radiograph", "management_context", "clavicle_overlap", 64),
    ],
    "shoulder": [
        _q("proximal humerus fracture management review radiograph", "management_context", "proximal_humerus", 90),
        _q("shoulder fracture dislocation management review radiograph", "red_flag_alignment_context", "shoulder_dislocation", 88),
        _q("clavicle fracture management review radiograph", "management_context", "clavicle_overlap", 64),
    ],
    "ankle_foot": [
        _q("ankle fracture management guideline malleolar radiograph", "management_context", "ankle_malleoli", 92),
        _q("Ottawa ankle rules radiography fracture review", "imaging_decision_context", "ankle_decision_rule", 88),
        _q("fifth metatarsal fracture management review radiograph", "management_context", "base_fifth_metatarsal", 84),
        _q("navicular fracture radiograph management review", "management_context", "navicular_midfoot", 78),
        _q("midfoot injury fracture radiograph management review", "red_flag_alignment_context", "midfoot_alignment", 74),
        _q("occult ankle fracture radiograph follow-up review", "occult_followup_context", "ankle_occult", 70),
    ],
    "knee_lower_leg": [
        _q("tibial plateau fracture management review radiograph", "management_context", "tibial_plateau", 92),
        _q("patella fracture management review radiograph", "management_context", "patella", 84),
        _q("proximal fibula fracture ankle injury radiograph review", "adjacent_joint_context", "proximal_fibula", 80),
        _q("tibia fibula shaft fracture management review radiograph", "management_context", "tibia_fibula_shaft", 76),
        _q("lower leg fracture compartment syndrome clinical review", "red_flag_escalation_context", "compartment_risk", 72),
    ],
    "leg": [
        _q("tibia fibula shaft fracture management review radiograph", "management_context", "tibia_fibula_shaft", 86),
        _q("lower leg fracture compartment syndrome clinical review", "red_flag_escalation_context", "compartment_risk", 82),
        _q("tibial plateau fracture management review radiograph", "management_context", "tibial_plateau", 64),
        _q("ankle fracture management guideline malleolar radiograph", "management_context", "ankle_malleoli", 58),
    ],
    "pelvis_hip_femur": [
        _q("occult hip fracture radiograph MRI follow-up review", "occult_followup_context", "occult_hip", 94),
        _q("femoral neck fracture management review radiograph", "management_context", "femoral_neck", 90),
        _q("proximal femur fracture management review radiograph", "management_context", "proximal_femur", 84),
        _q("pelvic fracture radiograph management review", "management_context", "pelvis", 76),
        _q("hip fracture inability to bear weight radiograph review", "clinical_localizer_context", "hip_mobility", 72),
    ],
    "hip": [
        _q("occult hip fracture radiograph MRI follow-up review", "occult_followup_context", "occult_hip", 94),
        _q("femoral neck fracture management review radiograph", "management_context", "femoral_neck", 90),
        _q("hip fracture inability to bear weight radiograph review", "clinical_localizer_context", "hip_mobility", 72),
    ],
}

UNKNOWN_QUERIES = [
    _q("musculoskeletal fracture radiograph management review", "management_context", "unknown_musculoskeletal", 55),
    _q("missed fracture radiograph follow-up review", "occult_followup_context", "unknown_occult", 52),
    _q("occult fracture radiography clinical review", "occult_followup_context", "unknown_occult", 50),
    _q("fracture radiograph diagnostic accuracy review", "imaging_review_context", "unknown_imaging", 45),
]

OCCULT_TERMS = [
    _q("occult fracture radiograph follow-up review", "occult_followup_context", "stage2c_occult", 76),
    _q("subtle fracture radiograph missed fracture review", "occult_followup_context", "stage2c_subtle", 72),
    _q("missed fracture radiograph clinical follow-up", "occult_followup_context", "stage2c_missed", 70),
    _q("radiographically occult fracture imaging review", "imaging_review_context", "stage2c_occult", 68),
]


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def _dedupe_query_specs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in sorted(items, key=lambda spec: int(spec.get("priority", 0)), reverse=True):
        query = str(item.get("query") or "").strip()
        key = query.lower()
        if query and key not in seen:
            out.append(item)
            seen.add(key)
    return out


def build_query_pack(summary: Stage3InputSummary, max_queries: int = 8) -> Dict[str, Any]:
    label = summary.anatomy_label or summary.anatomy_parent_label
    parent = summary.anatomy_parent_label
    base = ANATOMY_QUERY_MAP.get(label or "") or ANATOMY_QUERY_MAP.get(parent or "") or UNKNOWN_QUERIES

    specificity = "high"
    if not label or base == UNKNOWN_QUERIES:
        specificity = "low"
    elif summary.anatomy_output_type == "parent_fallback":
        specificity = "medium"

    query_specs = [dict(item) for item in base]
    if summary.stage2c_decision in {"suspicious", "uncertain"} and summary.accepted_candidate_count == 0:
        query_specs.extend(dict(item) for item in OCCULT_TERMS)

    if summary.fracture_candidate_status == "verified_candidate_present":
        query_specs.append(
            _q(
                f"{(label or parent or 'musculoskeletal').replace('_', ' ')} fracture radiograph management review",
                "candidate_management_context",
                label or parent or "unknown_musculoskeletal",
                86,
            )
        )

    query_specs = _dedupe_query_specs(query_specs)[:max_queries]
    queries = _dedupe([str(item.get("query")) for item in query_specs])[:max_queries]
    role_counts: Dict[str, int] = {}
    for item in query_specs:
        role = str(item.get("query_role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "version": QUERY_STRATEGY_VERSION,
        "anatomy_label": label,
        "parent_label": parent,
        "query_specificity": specificity,
        "stage2c_context_added": summary.stage2c_decision in {"suspicious", "uncertain"},
        "query_strategy": {
            "version": QUERY_STRATEGY_VERSION,
            "goal": "Retrieve management, follow-up, occult-injury, red-flag, and imaging-review context; not treatment orders.",
            "anatomy_specific": base != UNKNOWN_QUERIES,
            "role_counts": role_counts,
            "max_queries": max_queries,
            "keeps_legacy_query_list": True,
            "not_a_medical_claim": True,
        },
        "query_specs": query_specs,
        "queries": queries,
    }
