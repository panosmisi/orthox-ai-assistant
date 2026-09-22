from __future__ import annotations

from typing import Any, Dict

from .input_adapter import Stage3InputSummary


def _primary_candidate(pipeline_json: Dict[str, Any]) -> Dict[str, Any]:
    final = pipeline_json.get("final_assessment") or {}
    primary = final.get("primary_fracture_detection")
    return primary if isinstance(primary, dict) else {}


def build_visual_evidence(summary: Stage3InputSummary, pipeline_json: Dict[str, Any]) -> Dict[str, Any]:
    """Describe visual material that Stage 3 may display without re-reading the image diagnostically.

    Stage 3 is intentionally not a second vision model. This block gives the UI/report layer
    enough metadata to show the original radiograph, retained bbox, and optional Stage 2C
    heatmap while keeping the medical interpretation tied to validated upstream outputs.
    """
    primary = _primary_candidate(pipeline_json)
    probability_summary = primary.get("candidate_probability_summary") or {}
    bbox_confidence = primary.get("bbox_confidence") or {}

    has_candidate_box = summary.accepted_candidate_count > 0 and bool(summary.primary_candidate_bbox_xyxy)
    has_stage2c_warning = summary.stage2c_decision in {"suspicious", "uncertain"} and summary.accepted_candidate_count == 0

    overlays = {
        "show_primary_bbox": bool(has_candidate_box),
        "show_stage2c_heatmap_by_default": False,
        "stage2c_heatmap_available": bool(summary.heatmap_available),
        "stage2c_heatmap_display_policy": (
            "optional_on_request_with_explainability_warning"
            if summary.heatmap_available
            else "not_available"
        ),
        "display_priority": (
            "primary_bbox"
            if has_candidate_box
            else "stage2c_optional_heatmap"
            if has_stage2c_warning and summary.heatmap_available
            else "original_image_only"
        ),
    }

    return {
        "version": "stage3_visual_evidence_v1",
        "purpose": "UI/report visual context only; not an independent Stage 3 image diagnosis.",
        "image": {
            "image_id": summary.image_id,
            "image_path": summary.image_path,
            "display_original_radiograph": bool(summary.image_path),
        },
        "primary_bbox": {
            "available": bool(has_candidate_box),
            "xyxy_pixels": summary.primary_candidate_bbox_xyxy if has_candidate_box else None,
            "bbox_source": "upstream_stage1_stage2_primary_candidate" if has_candidate_box else None,
            "detector_support_score": summary.primary_candidate_confidence,
            "detector_support_percent": (
                round(summary.primary_candidate_confidence * 100.0, 3)
                if summary.primary_candidate_confidence is not None
                else None
            ),
            "final_fracture_support_score": probability_summary.get("final_fracture_support_score"),
            "final_fracture_support_percent": probability_summary.get("final_fracture_support_percent"),
            "bbox_quality_score_proxy": bbox_confidence.get("bbox_quality_score_proxy"),
            "bbox_quality_percent_proxy": bbox_confidence.get("bbox_quality_percent_proxy"),
            "independent_localization_probability_available": False,
            "localization_evidence_type": (
                "retained_upstream_bbox_without_calibrated_probability" if has_candidate_box else None
            ),
            "localization_note": bbox_confidence.get(
                "localization_note",
                (
                    "A retained bounding box is available from Stage 1/2, but YOLO does not output "
                    "a separate calibrated probability that the box is perfectly localized."
                ),
            ),
            "review_note": (
                "Review should start at the marked region and then continue across the full "
                "radiograph/study."
                if has_candidate_box
                else None
            ),
        },
        "stage2c_heatmap": {
            "available": bool(summary.heatmap_available),
            "decision": summary.stage2c_decision,
            "mean_suspicious_probability": summary.stage2c_mean_suspicious_probability,
            "mean_suspicious_percent": (
                round(summary.stage2c_mean_suspicious_probability * 100.0, 3)
                if summary.stage2c_mean_suspicious_probability is not None
                else None
            ),
            "important_semantics": {
                "not_a_fracture_boundary": True,
                "does_not_create_a_bbox": True,
                "shows_model_attention_not_diagnosis": True,
                "recommended_ui_behavior": "show only if clinician/user opens explainability view",
            },
        },
        "overlays": overlays,
        "stage3_visual_policy": {
            "stage3_reads_structured_outputs": True,
            "stage3_does_not_reinterpret_pixels": True,
            "stage3_does_not_override_stage1_stage2": True,
            "human_visual_review_required": True,
            "score_calibration_note": (
                "Displayed scores are model support scores, not calibrated clinical probabilities."
            ),
        },
    }
