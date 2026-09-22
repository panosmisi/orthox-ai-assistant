from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

from live_stage1_detector import run_stage1_live
from live_stage2a_verifier import verify_stage1_record
from live_stage2b_anatomy import classify_anatomy
from live_stage2c_safety import run_stage2c_safety
from stage3.runtime import load_config, run_stage3_on_json
from stage3.unified_analysis_contract import build_unified_from_live_full_pipeline, validate_unified_contract


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "live_full_pipeline_adapter_v1"
PIPELINE_VERSION = (
    "live_stage1_run4_run5_softmerge_orientation_rescue_stage2a_verifier_v2_"
    "stage2a5_strict_artifact_stage2b_anatomy_v4_stage2c_safety_v1"
)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clamp_box(box: Iterable[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def edge_density(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    gray_u8 = gray_crop.astype(np.uint8)
    lo, hi = np.percentile(gray_u8, [10, 90])
    spread = max(20.0, float(hi - lo))
    edges = cv2.Canny(gray_u8, threshold1=max(10, int(lo)), threshold2=max(30, int(lo + spread)))
    return float((edges > 0).mean())


def artifact_features(image: Image.Image, candidate: Dict[str, Any]) -> Dict[str, float]:
    width, height = image.size
    x1, y1, x2, y2 = clamp_box(candidate["xyxy_pixels"], width, height)
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    arr = np.asarray(crop, dtype=np.uint8)
    area = ((x2 - x1) / max(width, 1)) * ((y2 - y1) / max(height, 1))
    aspect = ((x2 - x1) / max(width, 1)) / max(((y2 - y1) / max(height, 1)), 1e-6)
    margins = [
        x1 / max(width, 1),
        y1 / max(height, 1),
        (width - x2) / max(width, 1),
        (height - y2) / max(height, 1),
    ]
    return {
        "crop_mean": round(float(arr.mean()) if arr.size else 0.0, 6),
        "crop_std": round(float(arr.std()) if arr.size else 0.0, 6),
        "dark_ratio": round(float((arr < 25).mean()) if arr.size else 0.0, 6),
        "bright_ratio": round(float((arr > 240).mean()) if arr.size else 0.0, 6),
        "edge_density": round(edge_density(arr), 6),
        "margin_min": round(float(min(margins)), 6),
        "area": round(float(area), 6),
        "aspect": round(float(aspect), 6),
    }


def strict_artifact_decision(features: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
    flags: List[str] = []
    reasons: List[str] = []
    area = features["area"]
    margin_min = features["margin_min"]
    edge = features["edge_density"]
    std = features["crop_std"]
    dark = features["dark_ratio"]
    bright = features["bright_ratio"]
    aspect = features["aspect"]

    if area < 0.003:
        flags.append("ultra_tiny_candidate")
    if margin_min < 0.025:
        flags.append("near_image_border")
    if std < 10.0:
        flags.append("low_texture_crop")
    if edge > 0.22:
        flags.append("high_edge_density")
    if dark > 0.45:
        flags.append("mostly_dark_crop")
    if bright > 0.45:
        flags.append("mostly_bright_crop")
    if aspect > 6.0 or aspect < 0.12:
        flags.append("extreme_aspect_ratio")

    suppress = False
    if margin_min < 0.025 and area < 0.018 and (dark > 0.45 or bright > 0.45) and (std < 18.0 or edge > 0.20):
        suppress = True
        reasons.append("near_border_high_contrast_marker_like_crop")
    elif margin_min < 0.015 and area < 0.006 and (aspect > 5.0 or aspect < 0.16) and edge > 0.16:
        suppress = True
        reasons.append("tiny_border_marker_like_geometry")

    return suppress, reasons, flags


def apply_stage2a5_artifact_suppression(
    image_path: Path,
    accepted_candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in accepted_candidates:
        enriched = deepcopy(candidate)
        features = artifact_features(image, enriched)
        suppressed, reasons, flags = strict_artifact_decision(features)
        enriched["artifact_suppression"] = {
            "stage": "Stage 2A.5",
            "mode": "live_runtime_feature_artifact_suppression",
            "rule": "strict",
            "suppressed": suppressed,
            "reasons": reasons,
            "flags": flags,
            "features": features,
            "runtime_note": (
                "Production strict artifact filtering is conservative and image-feature based. "
                "It does not use ground-truth text/artifact annotations at runtime."
            ),
        }
        if suppressed:
            enriched["final_keep"] = False
            enriched["final_rejection_reason"] = "stage2a5_artifact_suppression"
            rejected.append(enriched)
        else:
            enriched["final_keep"] = True
            enriched["final_rejection_reason"] = None
            kept.append(enriched)
    summary = {
        "stage": "Stage 2A.5",
        "mode": "live_runtime_feature_artifact_suppression",
        "rule": "strict",
        "input_detection_count": len(accepted_candidates),
        "artifact_rejected_count": len(rejected),
        "kept_count_after_artifact_suppression": len(kept),
        "note": "Conservative runtime feature-based artifact suppression; intended to remove obvious text/marker artifacts without suppressing likely true positives.",
    }
    return kept, rejected, summary


def candidate_support(candidate: Dict[str, Any]) -> Dict[str, Any]:
    det = float(candidate.get("confidence") or 0.0)
    verifier = candidate.get("verifier_prob_mean")
    vote_fraction = None
    if candidate.get("verifier_total_votes"):
        vote_fraction = float(candidate.get("verifier_keep_votes") or 0) / float(candidate["verifier_total_votes"])
    verifier_f = float(verifier) if verifier is not None else None
    final_score = max([v for v in [det, verifier_f, vote_fraction] if v is not None] or [det])
    return {
        "detector_fracture_candidate_confidence": det,
        "detector_fracture_candidate_percent": det * 100.0,
        "verifier_fracture_probability_mean": verifier_f,
        "verifier_fracture_probability_percent_mean": verifier_f * 100.0 if verifier_f is not None else None,
        "verifier_vote_fraction": vote_fraction,
        "verifier_vote_percent": vote_fraction * 100.0 if vote_fraction is not None else None,
        "final_fracture_support_score": final_score,
        "final_fracture_support_percent": final_score * 100.0,
        "score_calibration_note": (
            "These are model support scores, not calibrated clinical probabilities. "
            "They should be displayed as decision-support confidence, not diagnostic certainty."
        ),
    }


def bbox_confidence(candidate: Dict[str, Any]) -> Dict[str, Any]:
    xywhn = candidate.get("xywhn") or [0.0, 0.0, 0.0, 0.0]
    support = candidate_support(candidate)
    score = support["final_fracture_support_score"]
    return {
        "bbox_confidence_source": "YOLO detection confidence plus verifier agreement",
        "bbox_quality_score_proxy": score,
        "bbox_quality_percent_proxy": score * 100.0,
        "localization_note": (
            "YOLO does not output a separate probability that the box is perfectly localized. "
            "This field is a conservative proxy from detector confidence, verifier support, and artifact checks."
        ),
        "normalized_area": float(xywhn[2]) * float(xywhn[3]) if len(xywhn) >= 4 else None,
        "normalized_width": float(xywhn[2]) if len(xywhn) >= 4 else None,
        "normalized_height": float(xywhn[3]) if len(xywhn) >= 4 else None,
    }


def enrich_candidate(candidate: Dict[str, Any], source_line: int) -> Dict[str, Any]:
    out = deepcopy(candidate)
    out["source_line"] = source_line
    out["candidate_source"] = "live_stage1_stage2a_adapter"
    out["candidate_probability_summary"] = candidate_support(out)
    out["bbox_confidence"] = bbox_confidence(out)
    out.setdefault("final_keep", bool(out.get("verifier_keep")))
    out.setdefault("final_rejection_reason", None if out.get("final_keep") else "stage2a_verifier_rejected")
    return out


def build_final_assessment(
    image_id: str,
    kept: List[Dict[str, Any]],
    verifier_rejected: List[Dict[str, Any]],
    artifact_rejected: List[Dict[str, Any]],
    stage2a_raw_count: int,
    stage1_raw_candidate_count: int,
    stage2a5_summary: Dict[str, Any],
    anatomy: Dict[str, Any],
    stage2c: Dict[str, Any],
) -> Dict[str, Any]:
    if kept:
        status = "verified_candidate_present"
        statement = "At least one fracture candidate was retained after verifier and artifact review. This is a screening signal, not an autonomous diagnosis."
        suspected = True
    elif stage2c.get("decision") == "suspicious":
        status = "no_accepted_candidate_stage2c_suspicious"
        statement = stage2c.get("screening_message")
        suspected = False
    elif stage2c.get("decision") == "uncertain":
        status = "no_accepted_candidate_stage2c_uncertain"
        statement = stage2c.get("screening_message")
        suspected = False
    else:
        status = "no_candidate_detected"
        statement = "No high-confidence fracture candidate was retained. This must not be interpreted as ruling out fracture."
        suspected = False

    primary = None
    if kept:
        primary = sorted(
            kept,
            key=lambda c: float((c.get("candidate_probability_summary") or {}).get("final_fracture_support_score") or 0.0),
            reverse=True,
        )[0]

    flags = [
        "live_stage1_stage2a",
        "stage2a5_artifact_suppression_enabled",
        "stage2b_anatomy_live",
    ]
    if kept:
        flags.append("verified_fracture_candidate_present")
        flags.append("stage2c_safety_check_not_run")
    else:
        flags.append("stage2c_safety_check_run")
        flags.append(f"stage2c_{stage2c.get('decision')}")
    if anatomy.get("policy", {}).get("output_type") == "parent_fallback":
        flags.append("stage2b_parent_fallback")

    return {
        "pipeline_version": PIPELINE_VERSION,
        "stage_order": [
            "Stage 1: Run4@0.25 + Run5@0.15 fracture candidate detection with soft-merge IoU 0.40 and orientation rescue when the original upload yields no candidate",
            "Stage 2A: runtime-safe 6-seed verifier v2, keep if votes >= 2",
            "Stage 2A.5: conservative strict artifact suppression after verifier",
            "Stage 2B: whole-image anatomy classification with fine-label fallback policy",
            "Stage 2C: image-level clean-case safety warning, run only when no accepted fracture candidate remains",
            "Stage 3: deterministic care guidance and display payload",
        ],
        "fracture_candidate_status": status,
        "screening_statement": statement,
        "fracture_suspected": suspected,
        "fracture_suspected_semantics": "`true` means at least one candidate was retained by Stage 2A/2A.5. `false` never means fracture is ruled out.",
        "raw_fracture_candidate_count": stage1_raw_candidate_count,
        "merged_fracture_candidate_count": stage2a_raw_count,
        "verified_fracture_candidate_count": len(kept),
        "rejected_fracture_candidate_count": len(verifier_rejected) + len(artifact_rejected),
        "verifier_rejected_fracture_candidate_count": len(verifier_rejected),
        "artifact_rejected_fracture_candidate_count": len(artifact_rejected),
        "fracture_detection_count": len(kept),
        "primary_fracture_detection": primary,
        "stage2a5_artifact_suppression": stage2a5_summary,
        "stage2c_safety_decision": stage2c.get("decision"),
        "anatomy_label": anatomy.get("policy", {}).get("label"),
        "anatomy_parent_label": anatomy.get("policy", {}).get("parent_label"),
        "anatomy_output_type": anatomy.get("policy", {}).get("output_type"),
        "anatomy_confidence": anatomy.get("policy", {}).get("confidence"),
        "human_verification_required": True,
        "clinician_review_required": True,
        "reliability_flags": flags,
        "medical_scope_note": "Research prototype output. It may support review, but it is not a standalone medical diagnosis and must not be used to exclude fracture without clinician assessment.",
    }


def run_live_full_pipeline_one(
    image_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "",
    config_path: Path = REPO_ROOT / "stage3" / "stage3_config.production.yaml",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_records = run_stage1_live(source=image_path, device=device)
    if len(stage1_records) != 1:
        raise RuntimeError(f"Expected one Stage 1 record for single image, got {len(stage1_records)}")
    stage1 = stage1_records[0]
    write_json(output_dir / "stage1_live" / f"{image_path.stem}.stage1_live.json", stage1)

    stage2a = verify_stage1_record(stage1, min_votes=2, device=device)
    write_json(output_dir / "stage2a_live" / f"{image_path.stem}.stage2a_live.json", stage2a)

    accepted = [enrich_candidate(c, idx) for idx, c in enumerate(stage2a.get("accepted_candidates") or [], start=1)]
    verifier_rejected = [
        enrich_candidate(c, idx)
        for idx, c in enumerate(stage2a.get("rejected_candidates") or [], start=len(accepted) + 1)
    ]
    kept, artifact_rejected, stage2a5_summary = apply_stage2a5_artifact_suppression(image_path, accepted)
    anatomy = classify_anatomy(image_path, device=device)
    stage2c = run_stage2c_safety(image_path, accepted_candidate_count=len(kept), device=device)

    pipeline_json = {
        "pipeline_version": PIPELINE_VERSION,
        "image_path": str(image_path),
        "image_id": image_path.name,
        "stage1_stage2a_detection_mode": "live_pixel_to_model",
        "stage1_stage2a_raw_fracture_candidates": accepted + verifier_rejected,
        "stage1_stage2a_raw_fracture_candidate_count": stage2a.get("raw_candidate_count"),
        "stage1_stage2a_fracture_detections": kept,
        "stage1_stage2a_fracture_detection_count": len(kept),
        "stage1_stage2a5_fracture_detections": kept,
        "stage1_stage2a5_fracture_detection_count": len(kept),
        "stage1_stage2a_rejected_fracture_candidates": verifier_rejected + artifact_rejected,
        "stage1_stage2a_rejected_fracture_candidate_count": len(verifier_rejected) + len(artifact_rejected),
        "stage1_stage2a_runtime_audit": {
            "stage1_live_json": str(output_dir / "stage1_live" / f"{image_path.stem}.stage1_live.json"),
            "stage2a_live_json": str(output_dir / "stage2a_live" / f"{image_path.stem}.stage2a_live.json"),
            "stage1_raw_candidate_count": stage1.get("raw_candidate_count"),
            "stage1_merged_candidate_count": stage1.get("merged_candidate_count"),
            "stage1_orientation_rescue": stage1.get("orientation_rescue"),
            "stage2a_accepted_candidate_count": stage2a.get("accepted_candidate_count"),
            "stage2a_rejected_candidate_count": stage2a.get("rejected_candidate_count"),
        },
        "stage2a5_artifact_suppression": stage2a5_summary,
        "stage2c_safety_check": stage2c,
        "stage2b_anatomy": anatomy,
        "notes": [
            "Absence of a retained fracture candidate must be reported as no candidate detected or candidate rejected, never as no fracture.",
            "Stage 2C, when enabled, is an image-level safety warning layer and never creates or suppresses bounding boxes.",
            "External image scores are technical support signals, not calibrated clinical probabilities.",
        ],
    }
    pipeline_json["final_assessment"] = build_final_assessment(
        image_id=image_path.name,
        kept=kept,
        verifier_rejected=verifier_rejected,
        artifact_rejected=artifact_rejected,
        stage2a_raw_count=int(stage2a.get("raw_candidate_count") or 0),
        stage1_raw_candidate_count=int(stage1.get("raw_candidate_count") or 0),
        stage2a5_summary=stage2a5_summary,
        anatomy=anatomy,
        stage2c=stage2c,
    )
    pipeline_path = output_dir / "pipeline_json" / f"{image_path.stem}.live_full_pipeline.json"
    write_json(pipeline_path, pipeline_json)

    config = load_config(config_path)
    stage3_output = run_stage3_on_json(pipeline_json, config)
    final_display = stage3_output["stage3_orthopedic_rag"]["final_display_payload"]
    stage3_path = output_dir / "stage3_json" / f"{image_path.stem}.stage3.json"
    final_display_path = output_dir / "final_display_payloads" / f"{image_path.stem}.json"
    write_json(stage3_path, stage3_output)
    write_json(final_display_path, final_display)

    unified = build_unified_from_live_full_pipeline(
        pipeline_json,
        final_display,
        input_source="external_user_supplied_image",
        analysis_mode="live_full_pipeline",
    )
    unified["artifacts"].update(
        {
            "stage1_live_json": str(output_dir / "stage1_live" / f"{image_path.stem}.stage1_live.json"),
            "stage2a_live_json": str(output_dir / "stage2a_live" / f"{image_path.stem}.stage2a_live.json"),
            "pipeline_json": str(pipeline_path),
            "stage3_json": str(stage3_path),
            "final_display_payload": str(final_display_path),
        }
    )
    unified_errors = validate_unified_contract(unified)
    unified_path = output_dir / "unified_contracts" / f"{image_path.stem}.unified.json"
    write_json(unified_path, unified)
    finding_grouping = (unified.get("findings") or {}).get("finding_grouping") or {}

    summary = {
        "status": "PASS" if not unified_errors else "FAIL",
        "image_id": image_path.name,
        "pipeline_version": PIPELINE_VERSION,
        "stage1_raw_candidate_count": stage1.get("raw_candidate_count"),
        "stage1_merged_candidate_count": stage1.get("merged_candidate_count"),
        "stage2a_accepted_candidate_count": stage2a.get("accepted_candidate_count"),
        "stage2a_rejected_candidate_count": stage2a.get("rejected_candidate_count"),
        "stage2a5_artifact_rejected_count": len(artifact_rejected),
        "final_candidate_count": len(kept),
        "finding_group_count": finding_grouping.get("group_count"),
        "grouped_candidate_count": finding_grouping.get("grouped_candidate_count"),
        "ungrouped_candidate_ids": finding_grouping.get("ungrouped_candidate_ids"),
        "stage2b_policy": anatomy.get("policy"),
        "stage2c_decision": stage2c.get("decision"),
        "stage2c_was_run": stage2c.get("stage2c_was_run"),
        "stage3_critic": stage3_output["stage3_orthopedic_rag"].get("critic", {}).get("verdict"),
        "unified_contract_errors": unified_errors,
        "outputs": {
            "pipeline_json": str(pipeline_path),
            "stage3_json": str(stage3_path),
            "final_display_payload": str(final_display_path),
            "unified_contract": str(unified_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live external image through Stage 1 -> Stage 3 and unified contract.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="")
    parser.add_argument("--config", default=str(REPO_ROOT / "stage3" / "stage3_config.production.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_live_full_pipeline_one(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        device=args.device,
        config_path=Path(args.config),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
