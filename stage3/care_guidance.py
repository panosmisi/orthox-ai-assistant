from __future__ import annotations

from typing import Any, Dict, List

from .input_adapter import Stage3InputSummary


CARE_GUIDANCE_DICTIONARY_VERSION = "stage3_care_guidance_dictionary_v2_1"


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
    }
    return display_map.get(label, label.replace("_", " "))


def _state(summary: Stage3InputSummary) -> str:
    if summary.accepted_candidate_count > 0:
        return "candidate_retained"
    if summary.stage2c_decision in {"suspicious", "uncertain"}:
        return "image_level_warning_without_bbox"
    return "no_high_confidence_candidate_retained"


def _stable_index(summary: Stage3InputSummary, family: str, count: int) -> int:
    if count <= 1:
        return 0
    seed = "|".join(
        [
            str(summary.image_id or ""),
            str(summary.anatomy_label or ""),
            str(summary.anatomy_parent_label or ""),
            str(summary.anatomy_output_type or ""),
            str(summary.accepted_candidate_count),
            str(summary.stage2c_decision or ""),
            family,
        ]
    )
    return sum(ord(ch) for ch in seed) % count


def _rotate(items: List[str], offset: int) -> List[str]:
    if not items:
        return []
    offset = offset % len(items)
    return items[offset:] + items[:offset]


def _dedupe_strings(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        key = " ".join(str(item).lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(str(item))
    return out


def _status_review_focus(summary: Stage3InputSummary) -> List[str]:
    state = _state(summary)
    if state == "candidate_retained":
        variants = [
            "Start with the retained AI-marked region, then deliberately inspect the complete radiographic study.",
            "Use the bbox as the first review target, but do not stop the review at the box edge.",
            "Treat the retained region as a triage cue and compare it with the whole image and symptoms.",
        ]
    elif state == "image_level_warning_without_bbox":
        variants = [
            "No localized box is available, so the review should be whole-image rather than box-led.",
            "Use the image-level warning as a broad attention cue, not as a site diagnosis.",
            "If a heatmap is shown, treat it as model attention only and continue full-image review.",
        ]
    else:
        variants = [
            "No high-confidence candidate was retained; this should not be used as proof that injury is absent.",
            "Because no strong AI-marked region remains, clinical concern and image quality should drive the review.",
            "Treat the output as limited screening support only, especially if symptoms are focal or persistent.",
        ]
    return [variants[_stable_index(summary, "status_review_focus", len(variants))]]


def _anatomy_micro_focus(label: str | None, summary: Stage3InputSummary) -> List[str]:
    pools: Dict[str, List[str]] = {
        "wrist_hand": [
            "For wrist-dominant symptoms, explicitly include distal radius, distal ulna, carpal alignment, and the scaphoid region in the check.",
            "For hand-dominant symptoms, compare metacarpal/phalangeal alignment with focal tenderness and visible joint congruity.",
            "When the field spans wrist and hand, decide clinically whether the pain is radial wrist, central carpus, metacarpal, or digit-dominant.",
            "If pediatric wrist anatomy is possible, do not let growth-plate appearance alone drive an AI-based conclusion.",
        ],
        "hand": [
            "Use the hand/wrist parent label as regional context; narrow the site clinically before applying a fine wrist or hand checklist.",
            "Check whether the symptomatic area is wrist, metacarpal, or digit-dominant before relying on a single regional label.",
        ],
        "elbow": [
            "For elbow-dominant pain, include radial head/neck, olecranon, distal humerus, and joint alignment in the first pass.",
            "If motion is limited but the line is subtle, indirect elbow signs and clinical examination should carry extra weight.",
            "Do not ignore forearm or wrist symptoms when the mechanism could link adjacent regions.",
        ],
        "forearm": [
            "For forearm-dominant symptoms, inspect both radius and ulna along the visible shaft and include both adjacent joints when shown.",
            "If swelling or deformity is broad, avoid narrowing the review to only the highest AI-signal point.",
            "Check whether the field of view includes elbow and wrist adequately for the clinical question.",
        ],
        "shoulder_upper_arm": [
            "Separate shoulder, clavicle/AC-region, proximal humerus, and humeral shaft concern before narrowing the review.",
            "For shoulder-dominant symptoms, alignment and projection quality matter as much as the marked point.",
            "For upper-arm-dominant symptoms, confirm that the humeral shaft is sufficiently covered before relying on the regional label.",
        ],
        "shoulder": [
            "Use the shoulder parent label as broad context and verify whether symptoms are clavicle, AC joint, glenohumeral, or humeral-dominant.",
        ],
        "ankle_foot": [
            "For ankle-dominant symptoms, include medial/lateral malleoli, ankle mortise alignment, distal tibia/fibula, and talar dome where visible.",
            "For midfoot symptoms, explicitly correlate navicular and base-of-fifth-metatarsal tenderness with image coverage.",
            "If the marked area is proximal to the ankle joint, include distal tibial/fibular cortices rather than forcing a foot-only review.",
            "Weight-bearing ability is useful clinical context but must not be inferred from the model score.",
        ],
        "knee_lower_leg": [
            "For knee-dominant symptoms, include patella, tibial plateau, proximal tibia/fibula, and knee alignment.",
            "For lower-leg symptoms, inspect tibial and fibular shafts and avoid treating a broad lower-limb view as knee-only.",
            "If the image is long-leg or broad coverage, identify whether the clinical pain is knee, tibia/fibula shaft, ankle, or hip-adjacent.",
            "Severe or disproportionate lower-leg pain/swelling should remain a clinical red flag regardless of AI localization.",
        ],
        "leg": [
            "Because this is a parent lower-limb context, first determine whether the clinical problem is hip/pelvis, knee, lower leg, ankle, or foot.",
            "Broad lower-limb coverage should be handled as regional context rather than precise anatomy classification.",
        ],
        "pelvis_hip_femur": [
            "For hip/groin symptoms, include femoral neck, femoral head, intertrochanteric region, acetabulum, and pelvic ring where visible.",
            "For thigh-dominant symptoms, confirm femoral shaft coverage before interpreting the combined pelvis/hip/femur label.",
            "Inability to mobilize or bear weight should remain clinically important even when AI localization is weak.",
        ],
        "hip": [
            "Use the hip parent label as broad context and verify whether symptoms are pelvic, groin/hip, proximal femur, or thigh-dominant.",
        ],
        "unknown": [
            "First confirm the symptomatic region and image coverage before applying any site-specific checklist.",
            "Use the model output as review support only when it matches the clinical region of concern.",
        ],
    }
    key = label or "unknown"
    variants = pools.get(key) or pools["unknown"]
    return [variants[_stable_index(summary, "anatomy_micro_focus", len(variants))]]


def _case_aware_strings(
    *,
    summary: Stage3InputSummary,
    template: Dict[str, Any],
    field: str,
    label: str | None,
) -> List[str]:
    base = [str(item) for item in (template.get(field) or []) if item]
    base = _rotate(base, _stable_index(summary, f"{field}_rotation", len(base)))
    if field == "review_focus":
        return _dedupe_strings(_status_review_focus(summary) + _anatomy_micro_focus(label, summary) + base)
    return _dedupe_strings(base)


ANATOMY_GUIDANCE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "wrist_hand": {
        "template_id": "wrist_hand_review_v2",
        "display_label": "wrist/hand",
        "aliases": ["hand", "wrist"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_acr_acute_hand_wrist_trauma",
        ],
        "review_focus": [
            "Distal radius/ulna region when symptoms are near the wrist.",
            "Scaphoid region when clinical tenderness is relevant.",
            "Metacarpal, phalangeal, and joint alignment when symptoms are in the hand or fingers.",
        ],
        "clinical_context_prompts": [
            "Ask whether pain is focal over the wrist, thumb side of the wrist, hand, or fingers.",
            "Correlate swelling, deformity, rotation, and loss of function with the imaged region.",
            "Consider whether the initial radiographic views fully cover the symptomatic region.",
        ],
        "radiograph_review_prompts": [
            "Inspect cortical continuity around distal radius and ulna.",
            "Review carpal alignment and the scaphoid region if symptoms match.",
            "Check metacarpal and phalangeal alignment, rotation clues, and joint congruity.",
        ],
        "site_specific_checks": [
            {
                "check_id": "wrist_hand_scaphoid_region",
                "check": "If symptoms match, check for scaphoid-region tenderness and review whether further clinician-led assessment is needed.",
                "why": "Some wrist injuries may be subtle on initial radiographs.",
            },
            {
                "check_id": "wrist_hand_alignment",
                "check": "Check finger/hand alignment, rotation, and focal tenderness against the image findings.",
                "why": "Functional alignment and joint congruity matter even when AI support is low.",
            },
            {
                "check_id": "wrist_hand_coverage",
                "check": "Confirm that the radiographic field includes the clinically symptomatic wrist, hand, or finger region.",
                "why": "A model result outside the symptomatic area should not reduce clinical concern.",
            },
        ],
        "uncertainty_notes": [
            "Wrist/hand injuries can be subtle on early or limited views.",
            "A parent-level hand/wrist label should be treated as regional context, not exact localization.",
        ],
    },
    "hand": {
        "template_id": "hand_parent_review_v2",
        "display_label": "hand/wrist region",
        "aliases": ["wrist_hand"],
        "inherit_from": "wrist_hand",
    },
    "elbow": {
        "template_id": "elbow_review_v2",
        "display_label": "elbow",
        "aliases": ["forearm", "hand"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Radial head/neck region.",
            "Olecranon and distal humerus region.",
            "Joint alignment and indirect signs that may require clinician attention.",
        ],
        "clinical_context_prompts": [
            "Correlate focal elbow pain, reduced range of motion, swelling, and mechanism of injury.",
            "Check whether symptoms extend to forearm or wrist because associated injury can change review focus.",
        ],
        "radiograph_review_prompts": [
            "Inspect radial head/neck, olecranon, distal humerus, and joint alignment.",
            "Look for indirect signs around the elbow when no clear line is obvious.",
            "Confirm that the image includes the symptomatic elbow region adequately.",
        ],
        "site_specific_checks": [
            {
                "check_id": "elbow_range_of_motion_context",
                "check": "Correlate the image with elbow range of motion and focal tenderness.",
                "why": "Function and focal examination can remain important when image findings are subtle.",
            },
            {
                "check_id": "elbow_associated_forearm_wrist",
                "check": "If symptoms or mechanism suggest it, review forearm and wrist context rather than treating the elbow label as isolated.",
                "why": "Linked elbow/forearm/wrist injury patterns can be missed by a single regional label.",
            },
        ],
        "uncertainty_notes": [
            "Elbow and forearm labels can overlap in projection and clinical coverage.",
            "Image-level warnings should not be converted into a precise elbow localization.",
        ],
    },
    "forearm": {
        "template_id": "forearm_review_v2",
        "display_label": "forearm",
        "aliases": ["elbow", "wrist_hand", "hand"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Radius and ulna shafts.",
            "Elbow and wrist alignment when the mechanism suggests a linked injury.",
        ],
        "clinical_context_prompts": [
            "Correlate focal forearm tenderness with wrist and elbow symptoms.",
            "Check for deformity, swelling, and neurovascular symptoms when clinically relevant.",
        ],
        "radiograph_review_prompts": [
            "Inspect both radius and ulna along their visible length.",
            "Check that elbow and wrist joints are adequately represented if the clinical question includes them.",
        ],
        "site_specific_checks": [
            {
                "check_id": "forearm_both_bones_context",
                "check": "Review radius and ulna together and correlate with focal tenderness.",
                "why": "Forearm injury review should not focus on only one bone when symptoms are broader.",
            },
            {
                "check_id": "forearm_joint_ends",
                "check": "Consider elbow and wrist context if pain, mechanism, or image coverage suggests associated injury.",
                "why": "Forearm findings can be linked with joint-level injury patterns.",
            },
        ],
        "uncertainty_notes": [
            "Forearm, wrist, and elbow labels can overlap depending on radiographic coverage.",
        ],
    },
    "shoulder_upper_arm": {
        "template_id": "shoulder_upper_arm_review_v2",
        "display_label": "shoulder/upper arm",
        "aliases": ["shoulder"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Proximal humerus, humeral shaft, clavicle/acromioclavicular region, and shoulder alignment when visible.",
            "Image coverage: shoulder-only, humerus-only, and mixed shoulder/upper-arm views are not equivalent.",
        ],
        "clinical_context_prompts": [
            "Correlate pain location with shoulder, clavicle, upper arm, and range of motion.",
            "Check whether the image field includes the clinically symptomatic area.",
        ],
        "radiograph_review_prompts": [
            "Inspect proximal humerus, visible humeral shaft, clavicle region when included, and shoulder alignment.",
            "Check for projection limitations if the candidate is weak or the anatomy label is parent-level.",
        ],
        "site_specific_checks": [
            {
                "check_id": "shoulder_upper_arm_site_match",
                "check": "Confirm whether the clinical symptom is shoulder, clavicle-area, or upper-arm dominant.",
                "why": "The model label is regional and should not be treated as exact site localization.",
            },
            {
                "check_id": "shoulder_alignment_context",
                "check": "Review shoulder alignment and visible proximal humerus region when the image coverage supports it.",
                "why": "Shoulder/upper-arm review depends heavily on projection and field of view.",
            },
        ],
        "uncertainty_notes": [
            "Shoulder and upper-arm views may include different anatomy; the label is a context label, not a full report.",
        ],
    },
    "shoulder": {
        "template_id": "shoulder_parent_review_v2",
        "display_label": "shoulder/upper arm",
        "aliases": ["shoulder_upper_arm"],
        "inherit_from": "shoulder_upper_arm",
    },
    "ankle_foot": {
        "template_id": "ankle_foot_review_v2",
        "display_label": "ankle/foot",
        "aliases": ["leg"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_boa_boast_ankle_fractures",
            "guideline_ottawa_ankle_rules",
        ],
        "review_focus": [
            "Medial and lateral malleolar region.",
            "Midfoot points such as navicular and base of fifth metatarsal when symptoms match.",
            "Weight-bearing ability as clinical context, not as an AI-derived decision.",
        ],
        "clinical_context_prompts": [
            "Ask whether the patient can bear weight, if clinically relevant.",
            "Correlate focal tenderness over malleoli, navicular, and base of fifth metatarsal.",
            "Check whether symptoms are ankle-dominant, midfoot-dominant, or more proximal.",
        ],
        "radiograph_review_prompts": [
            "Inspect malleolar cortices and ankle mortise region when visible.",
            "Inspect navicular and base of fifth metatarsal region when the image includes them.",
            "Check image coverage if symptoms are outside the visible ankle/foot field.",
        ],
        "site_specific_checks": [
            {
                "check_id": "ankle_foot_weight_bearing",
                "check": "Check ability to bear weight and the site of maximal tenderness.",
                "why": "Ankle/foot assessment often depends on weight-bearing ability and focal bony tenderness.",
            },
            {
                "check_id": "ankle_foot_focal_points",
                "check": "Specifically consider malleolar, navicular, and base-of-fifth-metatarsal tenderness.",
                "why": "These are clinically important ankle/foot review points.",
            },
            {
                "check_id": "ankle_foot_symptom_region_match",
                "check": "Confirm whether symptoms are ankle, midfoot, forefoot, or lower-leg dominant.",
                "why": "A broad ankle/foot label should not suppress review of adjacent symptomatic regions.",
            },
        ],
        "uncertainty_notes": [
            "Ankle/foot decision-rule context belongs to clinician review, not automatic AI clearance.",
            "A low AI score does not exclude subtle injury when focal tenderness persists.",
        ],
    },
    "knee_lower_leg": {
        "template_id": "knee_lower_leg_review_v2",
        "display_label": "knee/lower leg",
        "aliases": ["leg"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_ottawa_knee_rules",
        ],
        "review_focus": [
            "Patella, tibial plateau/proximal tibia, proximal fibula, tibial/fibular shaft depending on field of view.",
            "Ability to bear weight and knee motion as clinical context when relevant.",
            "Adjacent ankle/hip symptoms if the image or mechanism is broad.",
        ],
        "clinical_context_prompts": [
            "Correlate focal tenderness around patella, fibular head, tibial plateau, tibial shaft, or ankle.",
            "Check ability to bear weight and knee flexion context when clinically relevant.",
            "Check whether the image is knee-focused, lower-leg-focused, or long-leg coverage.",
        ],
        "radiograph_review_prompts": [
            "Inspect patella, tibial plateau, proximal fibula, visible tibia/fibula shafts, and joint alignment.",
            "If the view is long-leg or broad, avoid forcing a knee-only interpretation.",
        ],
        "site_specific_checks": [
            {
                "check_id": "knee_leg_weight_bearing",
                "check": "Check ability to bear weight and focal tenderness around knee, tibia, fibula, and ankle as clinically relevant.",
                "why": "Lower-limb injury review should not rely only on a single AI region.",
            },
            {
                "check_id": "knee_leg_associated_site",
                "check": "Consider whether symptoms suggest an associated injury away from the most obvious image region.",
                "why": "Linked knee/ankle/lower-leg injuries can be clinically important.",
            },
            {
                "check_id": "knee_leg_view_scope",
                "check": "Confirm whether the image is a knee, lower-leg, or long-leg view before interpreting the anatomy label.",
                "why": "Broad lower-limb views can make fine anatomical labels less reliable.",
            },
        ],
        "uncertainty_notes": [
            "Knee and lower-leg labels can overlap when the radiograph covers a broad region.",
            "Long-leg images should be treated as regional review prompts rather than precise localization.",
        ],
    },
    "leg": {
        "template_id": "leg_parent_review_v2",
        "display_label": "lower limb",
        "aliases": ["ankle_foot", "knee_lower_leg"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Use the lower-limb parent label as broad context only.",
            "Differentiate knee/lower-leg, ankle/foot, and hip/pelvis symptom regions during clinical review.",
        ],
        "clinical_context_prompts": [
            "Identify the exact symptomatic region before relying on any fine anatomical checklist.",
            "Correlate weight-bearing ability, focal tenderness, swelling, deformity, and mechanism of injury.",
        ],
        "radiograph_review_prompts": [
            "Review image coverage first: knee, lower leg, ankle/foot, hip/pelvis, or long-leg.",
            "Avoid over-interpreting a parent-level lower-limb label as a precise site.",
        ],
        "site_specific_checks": [
            {
                "check_id": "leg_parent_exact_region_needed",
                "check": "Confirm the clinically symptomatic lower-limb region before applying a fine-site checklist.",
                "why": "Parent-level lower-limb output is intentionally broad.",
            },
            {
                "check_id": "leg_weight_bearing_context",
                "check": "Check ability to bear weight and focal tenderness when clinically relevant.",
                "why": "Lower-limb clinical context remains important even when AI localization is weak.",
            },
        ],
        "uncertainty_notes": [
            "Parent fallback means fine anatomy was not reliable enough for exact site wording.",
        ],
    },
    "pelvis_hip_femur": {
        "template_id": "pelvis_hip_femur_review_v2",
        "display_label": "pelvis/hip/femur",
        "aliases": ["hip"],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Pelvic ring, acetabular/hip region, proximal femur, and visible femoral shaft depending on field of view.",
            "Weight-bearing limitation and high-energy mechanism as clinical context.",
        ],
        "clinical_context_prompts": [
            "Correlate groin, hip, pelvic, or thigh pain with the imaged field.",
            "Check ability to mobilize or bear weight as clinical context when relevant.",
            "Consider whether symptoms and image coverage match; pelvis and femur views are not interchangeable.",
        ],
        "radiograph_review_prompts": [
            "Inspect pelvic ring symmetry, hip joint region, proximal femur, and visible femoral shaft when included.",
            "Check for projection and coverage limitations before relying on model confidence.",
        ],
        "site_specific_checks": [
            {
                "check_id": "pelvis_hip_site_match",
                "check": "Confirm whether symptoms are pelvic, hip/groin, or thigh-dominant.",
                "why": "The combined label is broad and should be narrowed clinically.",
            },
            {
                "check_id": "pelvis_hip_weight_bearing_context",
                "check": "Check ability to bear weight or mobilize when clinically relevant.",
                "why": "Functional limitation can remain important even with weak AI support.",
            },
        ],
        "uncertainty_notes": [
            "Pelvis/hip/femur is a broad class; Stage 3 should not convert it into a precise site.",
        ],
    },
    "hip": {
        "template_id": "hip_parent_review_v2",
        "display_label": "pelvis/hip/femur",
        "aliases": ["pelvis_hip_femur"],
        "inherit_from": "pelvis_hip_femur",
    },
    "unknown": {
        "template_id": "generic_regional_review_v2",
        "display_label": "the imaged region",
        "aliases": [],
        "source_basis": [
            "guideline_nice_ng38_non_complex_fractures",
            "guideline_nice_ng37_complex_fractures",
        ],
        "review_focus": [
            "Confirm the clinically symptomatic region and match it to the radiographic field.",
            "Use model output as screening support only.",
        ],
        "clinical_context_prompts": [
            "Correlate pain, tenderness, swelling, deformity, mechanism, and function with the imaged field.",
            "Check whether additional clinical review is needed if symptoms do not match the AI output.",
        ],
        "radiograph_review_prompts": [
            "Review image quality, projection, and coverage before interpreting model confidence.",
            "Inspect the full radiograph rather than only the most visually obvious area.",
        ],
        "site_specific_checks": [
            {
                "check_id": "generic_region_match",
                "check": "Confirm that the model context matches the clinically symptomatic region.",
                "why": "Uncertain or broad anatomy labels cannot provide precise site guidance.",
            }
        ],
        "uncertainty_notes": [
            "The anatomy context is broad or uncertain.",
        ],
    },
}


def _resolve_template(label: str | None) -> Dict[str, Any]:
    key = label or "unknown"
    template = ANATOMY_GUIDANCE_TEMPLATES.get(key) or ANATOMY_GUIDANCE_TEMPLATES["unknown"]
    inherited = template.get("inherit_from")
    if inherited:
        base = dict(ANATOMY_GUIDANCE_TEMPLATES[inherited])
        base["template_id"] = template["template_id"]
        base["display_label"] = template.get("display_label") or base.get("display_label")
        base["aliases"] = template.get("aliases") or base.get("aliases") or []
        base["inherited_from"] = inherited
        return base
    return dict(template)


def _source_ids_for_anatomy(summary: Stage3InputSummary, sources: List[Dict[str, Any]]) -> List[str]:
    label = summary.anatomy_label or summary.anatomy_parent_label or ""
    source_ids = []
    for source in sources:
        source_id = str(source.get("source_id") or "")
        scope = source.get("anatomy_scope") or []
        if label in scope or "unknown" in scope:
            source_ids.append(source_id)
    if not source_ids:
        source_ids = [str(source.get("source_id")) for source in sources if source.get("source_id")]
    return source_ids[:5]


def _care_tier(summary: Stage3InputSummary) -> Dict[str, str]:
    state = _state(summary)
    if state == "candidate_retained":
        return {
            "tier_id": "localized_candidate_review",
            "urgency": "clinician_review_required",
            "meaning": "A localized AI candidate exists and should be reviewed on the original radiograph.",
        }
    if state == "image_level_warning_without_bbox":
        return {
            "tier_id": "image_level_caution_review",
            "urgency": "clinician_review_required",
            "meaning": "No reliable bbox remains, but an image-level warning recommends careful whole-image review.",
        }
    return {
        "tier_id": "no_high_confidence_candidate_retained",
        "urgency": "review_with_clinical_context",
        "meaning": "No high-confidence AI candidate remains; this does not exclude subtle or occult injury and cannot be used as clearance.",
    }


def _universal_checks() -> List[Dict[str, str]]:
    return [
        {
            "check_id": "symptom_correlation",
            "check": "Correlate the AI output with focal pain, tenderness, swelling, mechanism of injury, and ability to use the limb.",
            "why": "The model output is weaker than persistent clinical concern.",
        },
        {
            "check_id": "image_quality_check",
            "check": "Review image quality, projection, coverage of the symptomatic region, and whether additional views are clinically needed.",
            "why": "Missed or subtle injuries can depend on projection and coverage.",
        },
        {
            "check_id": "neurovascular_check",
            "check": "Check and document neurovascular status when clinically relevant.",
            "why": "Neurovascular concern changes clinical priority regardless of AI output.",
        },
    ]


def _anatomy_checks(summary: Stage3InputSummary) -> List[Dict[str, str]]:
    label = summary.anatomy_label or summary.anatomy_parent_label
    template = _resolve_template(label)
    checks = template.get("site_specific_checks") or []
    return [dict(item) for item in checks if isinstance(item, dict)]


def _pathway_considerations(summary: Stage3InputSummary) -> List[Dict[str, Any]]:
    state = _state(summary)
    items: List[Dict[str, Any]] = []
    if state == "candidate_retained":
        items.append(
            {
                "consideration_id": "candidate_review_pathway",
                "text": "Use the retained bbox as a review starting point, then inspect the full radiograph and clinical context.",
                "source_role": "pipeline_observation",
            }
        )
    elif state == "image_level_warning_without_bbox":
        items.append(
            {
                "consideration_id": "caution_without_bbox_pathway",
                "text": "Because the caution is image-level, do not present it as a precise injury location; use it to increase review attention.",
                "source_role": "pipeline_observation",
            }
        )
    else:
        items.append(
            {
                "consideration_id": "no_high_confidence_candidate_pathway",
                "text": (
                    "Use the result as limited screening support only; persistent clinical concern "
                    "should drive clinician-led review."
                ),
                "source_role": "pipeline_observation",
            }
        )

    label = summary.anatomy_label or summary.anatomy_parent_label
    if label == "ankle_foot":
        items.append(
            {
                "consideration_id": "ankle_foot_rule_context",
                "text": "For suspected ankle/foot injuries, clinical decision-rule context and local imaging pathways should be considered by the clinician.",
                "source_role": "guideline_context",
            }
        )
    elif label == "knee_lower_leg":
        items.append(
            {
                "consideration_id": "knee_rule_context",
                "text": "For suspected knee/lower-leg injuries, clinical decision-rule context and local imaging pathways should be considered by the clinician.",
                "source_role": "guideline_context",
            }
        )
    elif label in {"wrist_hand", "hand"}:
        items.append(
            {
                "consideration_id": "wrist_hand_occult_context",
                "text": "For wrist/hand symptoms, subtle injuries may require clinician-led follow-up review even when model support is low.",
                "source_role": "guideline_context",
            }
        )
    return items


def _is_lower_limb_context(label: str | None) -> bool:
    return label in {"ankle_foot", "knee_lower_leg", "leg", "pelvis_hip_femur", "hip"}


def _generic_escalation_triggers(label: str | None) -> List[str]:
    triggers = [
        "worsening pain, swelling, deformity, or open-injury concern",
        "neurovascular symptoms such as numbness, weakness, pallor, or reduced pulses",
        "persistent focal bony tenderness despite low AI support",
        "high-energy trauma or clinician concern that does not match the AI output",
    ]
    if _is_lower_limb_context(label):
        triggers.insert(2, "inability to bear weight when clinically relevant")
    else:
        triggers.insert(2, "loss of function or inability to use the limb when clinically relevant")
    return triggers


def _output_boundaries(label: str | None) -> List[str]:
    boundaries = [
        "Do not infer absence of injury from no retained high-confidence candidate.",
        "Do not infer patient-specific management from AI output alone.",
        "Do not use this output as a substitute for local clinical pathways.",
        "Do not convert heatmap attention into a fracture boundary.",
    ]
    if _is_lower_limb_context(label):
        boundaries.insert(2, "Do not assign mobility or weight-bearing instructions from this prototype.")
    else:
        boundaries.insert(2, "Do not assign activity restriction, return-to-use, or immobilization instructions from this prototype.")
    return boundaries


def build_care_guidance(
    summary: Stage3InputSummary,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build conservative anatomy-aware care/review guidance.

    This is intentionally not patient-specific treatment advice.
    """
    label = summary.anatomy_label or summary.anatomy_parent_label
    template = _resolve_template(label)
    source_ids = _source_ids_for_anatomy(summary, sources)
    review_focus = _case_aware_strings(
        summary=summary,
        template=template,
        field="review_focus",
        label=label,
    )
    clinical_context_prompts = _case_aware_strings(
        summary=summary,
        template=template,
        field="clinical_context_prompts",
        label=label,
    )
    radiograph_review_prompts = _case_aware_strings(
        summary=summary,
        template=template,
        field="radiograph_review_prompts",
        label=label,
    )
    return {
        "version": "stage3_care_guidance_v2",
        "dictionary_version": CARE_GUIDANCE_DICTIONARY_VERSION,
        "purpose": "Structured clinical review guidance, not patient-specific treatment.",
        "llm_used": False,
        "not_a_diagnosis": True,
        "no_patient_specific_treatment": True,
        "human_verification_required": True,
        "anatomy_context": {
            "label": summary.anatomy_label,
            "parent_label": summary.anatomy_parent_label,
            "display_label": _display_label(summary.anatomy_label or summary.anatomy_parent_label),
            "output_type": summary.anatomy_output_type,
        },
        "anatomy_template": {
            "template_id": template.get("template_id"),
            "display_label": template.get("display_label"),
            "aliases": template.get("aliases") or [],
            "inherited_from": template.get("inherited_from"),
            "source_basis": template.get("source_basis") or [],
        },
        "wording_policy": {
            "stable_structure": True,
            "deterministic_case_variation": True,
            "variation_seed_fields": [
                "image_id",
                "anatomy_label",
                "anatomy_parent_label",
                "anatomy_output_type",
                "accepted_candidate_count",
                "stage2c_decision",
            ],
            "same_input_same_wording": True,
            "random_generation_used": False,
        },
        "care_tier": _care_tier(summary),
        "review_focus": review_focus,
        "clinical_context_prompts": clinical_context_prompts,
        "radiograph_review_prompts": radiograph_review_prompts,
        "review_checks": _universal_checks() + _anatomy_checks(summary),
        "pathway_considerations": _pathway_considerations(summary),
        "uncertainty_notes": template.get("uncertainty_notes") or [],
        "when_to_escalate_review": _generic_escalation_triggers(label),
        "do_not_infer": _output_boundaries(label),
        "supporting_source_ids": source_ids,
        "template_source_ids": template.get("source_basis") or [],
        "citation_policy": {
            "source_ids_are_context_only": True,
            "guidelines_do_not_replace_clinician_judgement": True,
            "no_direct_treatment_order_generated": True,
        },
    }
