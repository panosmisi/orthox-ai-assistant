from __future__ import annotations

from typing import Dict, List

from .input_adapter import Stage3InputSummary


GENERAL_RED_FLAGS = [
    "open injury concern",
    "visible deformity or dislocation concern",
    "neurovascular symptoms such as numbness, weakness, pallor, or reduced pulses",
    "severe, escalating, or disproportionate pain/swelling",
    "high-energy trauma mechanism",
    "persistent focal pain despite limited or uncertain radiographs",
]

ANATOMY_RED_FLAGS = {
    "wrist_hand": [
        "anatomical snuffbox tenderness or scaphoid concern",
        "metacarpal or phalangeal malrotation/alignment concern",
        "pediatric growth plate tenderness if pediatric context is present",
    ],
    "hand": [
        "metacarpal or phalangeal malrotation/alignment concern",
        "open wound over a suspected fracture",
        "pediatric growth plate tenderness if pediatric context is present",
    ],
    "elbow": [
        "fat pad sign or occult elbow fracture concern",
        "radial head tenderness with limited rotation",
        "pediatric supracondylar fracture concern if pediatric context is present",
    ],
    "forearm": [
        "wrist or elbow associated injury concern",
        "deformity or compartment-syndrome concern",
        "neurovascular symptoms in the hand",
    ],
    "shoulder_upper_arm": [
        "fracture-dislocation concern",
        "clavicle or proximal humerus deformity",
        "neurovascular symptoms in the arm/hand",
    ],
    "shoulder": [
        "fracture-dislocation concern",
        "clavicle or proximal humerus deformity",
        "neurovascular symptoms in the arm/hand",
    ],
    "ankle_foot": [
        "inability to bear weight",
        "malleolar, navicular, or base of fifth metatarsal focal tenderness",
        "neurovascular symptoms in the foot",
    ],
    "knee_lower_leg": [
        "inability to bear weight",
        "tibial plateau or patellar injury concern",
        "proximal fibula tenderness when ankle injury is suspected",
    ],
    "leg": [
        "inability to bear weight",
        "deformity or compartment-syndrome concern",
        "proximal fibula tenderness when ankle injury is suspected",
    ],
    "pelvis_hip_femur": [
        "inability to mobilize or bear weight",
        "severe hip/groin pain after trauma",
        "occult hip fracture concern despite limited or uncertain radiographs",
    ],
    "hip": [
        "inability to mobilize or bear weight",
        "severe hip/groin pain after trauma",
        "occult hip fracture concern despite limited or uncertain radiographs",
    ],
}


def build_red_flags(summary: Stage3InputSummary) -> Dict[str, List[str]]:
    label = summary.anatomy_label or summary.anatomy_parent_label or "unknown"
    anatomy_flags = ANATOMY_RED_FLAGS.get(label, [])
    if not anatomy_flags and summary.anatomy_parent_label:
        anatomy_flags = ANATOMY_RED_FLAGS.get(summary.anatomy_parent_label, [])

    contextual = []
    if summary.stage2c_decision in {"suspicious", "uncertain"} and summary.accepted_candidate_count == 0:
        contextual.append("subtle or occult fracture concern because Stage 2C raised an image-level warning without a retained box")
    if summary.anatomy_output_type == "parent_fallback":
        contextual.append("anatomy label uncertainty; review should not rely on fine anatomical localization alone")

    return {
        "phrasing_rule": "These are red flags a clinician may need to check; they are not asserted patient findings.",
        "general": GENERAL_RED_FLAGS,
        "anatomy_specific": anatomy_flags,
        "contextual": contextual,
    }
