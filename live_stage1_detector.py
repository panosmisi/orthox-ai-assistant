from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image, ImageOps

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device, smart_inference_mode


REPO_ROOT = Path(__file__).resolve().parent
RUN4_WEIGHTS = REPO_ROOT / "runs" / "train" / "yolov9-c_fracatlas_bg10_recall" / "weights" / "best.pt"
RUN5_WEIGHTS = REPO_ROOT / "runs" / "train" / "yolov9-c_fracatlas_tiny_os_run5" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "live_stage1_detector_v1"
PIPELINE_STAGE1_VERSION = "stage1_run4_0p25_run5_0p15_orientation_rescue_live_adapter_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ORIENTATION_VARIANTS = (
    ("original", None),
    ("rot90_ccw", Image.Transpose.ROTATE_90),
    ("rot90_cw", Image.Transpose.ROTATE_270),
    ("rot180", Image.Transpose.ROTATE_180),
)


def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def clamp_xyxy(box: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def map_box_to_original(
    box: Iterable[float],
    orientation_variant: str,
    original_shape: Tuple[int, int],
) -> List[float]:
    orig_h, orig_w = original_shape
    x1, y1, x2, y2 = [float(v) for v in box]
    if orientation_variant == "original":
        mapped = [x1, y1, x2, y2]
    elif orientation_variant == "rot90_ccw":
        mapped = [orig_w - y2, x1, orig_w - y1, x2]
    elif orientation_variant == "rot90_cw":
        mapped = [y1, orig_h - x2, y2, orig_h - x1]
    elif orientation_variant == "rot180":
        mapped = [orig_w - x2, orig_h - y2, orig_w - x1, orig_h - y1]
    else:
        raise ValueError(f"Unsupported orientation variant: {orientation_variant}")
    return clamp_xyxy(mapped, width=orig_w, height=orig_h)


def find_image_inputs(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    return [
        path
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def create_orientation_variant_batch(source: Path, work_dir: Path) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    batch_dir = work_dir / "orientation_variants"
    batch_dir.mkdir(parents=True, exist_ok=True)
    variant_map: Dict[str, Dict[str, Any]] = {}
    for index, image_path in enumerate(find_image_inputs(source)):
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        for orientation_name, transpose_op in ORIENTATION_VARIANTS:
            variant_image = image if transpose_op is None else image.transpose(transpose_op)
            variant_name = f"{index:04d}__{image_path.stem}__{orientation_name}.png"
            variant_path = batch_dir / variant_name
            variant_image.save(variant_path)
            variant_map[variant_name] = {
                "original_image_id": image_path.name,
                "original_image_path": str(image_path),
                "original_shape": {"height": height, "width": width},
                "orientation_variant": orientation_name,
                "orientation_adjusted": orientation_name != "original",
                "variant_image_path": str(variant_path),
                "variant_shape": {"height": variant_image.height, "width": variant_image.width},
            }
    return batch_dir, variant_map


def transform_candidate_to_original(
    candidate: Dict[str, Any],
    variant_info: Dict[str, Any],
) -> Dict[str, Any]:
    original_shape = (
        int(variant_info["original_shape"]["height"]),
        int(variant_info["original_shape"]["width"]),
    )
    mapped_xyxy = map_box_to_original(
        candidate["xyxy_pixels"],
        variant_info["orientation_variant"],
        original_shape,
    )
    transformed = candidate_from_det(
        mapped_xyxy,
        float(candidate["confidence"]),
        int(candidate.get("class_id", 0)),
        original_shape,
        candidate["detector_source"],
    )
    transformed["orientation_variant"] = variant_info["orientation_variant"]
    transformed["orientation_adjusted"] = bool(variant_info["orientation_adjusted"])
    transformed["variant_xyxy_pixels"] = candidate["xyxy_pixels"]
    transformed["variant_image_shape"] = variant_info["variant_shape"]
    transformed["variant_image_path"] = variant_info["variant_image_path"]
    return transformed


def orientation_rescue_candidate_allowed(candidate: Dict[str, Any]) -> bool:
    confidence = float(candidate.get("confidence") or 0.0)
    detector_sources = set(candidate.get("detector_sources") or [])
    if confidence >= 0.75:
        return True
    return confidence >= 0.45 and len(detector_sources) >= 2


def weighted_merge_group(group: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> Dict[str, Any]:
    weight_sum = sum(max(float(c["confidence"]), 1e-6) for c in group)
    merged_xyxy = []
    for idx in range(4):
        merged_xyxy.append(sum(float(c["xyxy_pixels"][idx]) * max(float(c["confidence"]), 1e-6) for c in group) / weight_sum)
    confidence = max(float(c["confidence"]) for c in group)
    h, w = image_shape
    x1, y1, x2, y2 = merged_xyxy
    xywhn = [
        ((x1 + x2) / 2.0) / w,
        ((y1 + y2) / 2.0) / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    ]
    sources = sorted({c["detector_source"] for c in group})
    return {
        "class_id": 0,
        "label": "fracture",
        "confidence": confidence,
        "xyxy_pixels": [round(v, 6) for v in merged_xyxy],
        "xywhn": [round(v, 8) for v in xywhn],
        "detector_sources": sources,
        "source_candidate_count": len(group),
        "merge_method": "confidence_weighted_soft_merge",
        "orientation_variants": sorted({c.get("orientation_variant", "original") for c in group}),
        "orientation_adjusted": any(bool(c.get("orientation_adjusted")) for c in group),
        "merged_from": [
            {
                "detector_source": c["detector_source"],
                "confidence": c["confidence"],
                "xyxy_pixels": c["xyxy_pixels"],
                "orientation_variant": c.get("orientation_variant", "original"),
                "orientation_adjusted": bool(c.get("orientation_adjusted")),
                "variant_xyxy_pixels": c.get("variant_xyxy_pixels"),
            }
            for c in group
        ],
    }


def soft_merge_candidates(
    candidates: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_threshold: float = 0.40,
) -> List[Dict[str, Any]]:
    remaining = sorted(candidates, key=lambda c: float(c["confidence"]), reverse=True)
    merged = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep = []
        for candidate in remaining:
            if int(candidate.get("class_id", 0)) == int(seed.get("class_id", 0)) and box_iou_xyxy(
                seed["xyxy_pixels"], candidate["xyxy_pixels"]
            ) >= iou_threshold:
                group.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        merged.append(weighted_merge_group(group, image_shape))
    return sorted(merged, key=lambda c: float(c["confidence"]), reverse=True)


def candidate_from_det(
    xyxy: Iterable[float],
    conf: float,
    cls: int,
    image_shape: Tuple[int, int],
    detector_source: str,
) -> Dict[str, Any]:
    h, w = image_shape
    xyxy_list = [float(v) for v in xyxy]
    xyxy_tensor = torch.tensor(xyxy_list).view(1, 4)
    xywhn = (xyxy2xywh(xyxy_tensor) / torch.tensor([w, h, w, h])).view(-1).tolist()
    return {
        "class_id": int(cls),
        "label": "fracture" if int(cls) == 0 else str(cls),
        "confidence": round(float(conf), 6),
        "xyxy_pixels": [round(v, 6) for v in xyxy_list],
        "xywhn": [round(float(v), 8) for v in xywhn],
        "detector_source": detector_source,
    }


@smart_inference_mode()
def run_detector_batch(
    weights: Path,
    source: Path,
    detector_source: str,
    conf_thres: float,
    device: str = "",
    imgsz: int = 640,
    iou_thres: float = 0.45,
    max_det: int = 300,
    half: bool = False,
) -> Dict[str, Dict[str, Any]]:
    selected_device = select_device(device)
    model = DetectMultiBackend(str(weights), device=selected_device, fp16=half)
    stride, pt = model.stride, model.pt
    checked_imgsz = check_img_size((imgsz, imgsz), s=stride)
    dataset = LoadImages(str(source), img_size=checked_imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *checked_imgsz))

    outputs: Dict[str, Dict[str, Any]] = {}
    for path, im, im0s, _vid_cap, _s in dataset:
        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]

        pred = model(im_tensor, augment=False, visualize=False)
        pred = pred[0][1]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False, max_det=max_det)

        image_path = Path(path)
        im0 = im0s.copy()
        h, w = im0.shape[:2]
        candidates = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    candidates.append(candidate_from_det(xyxy, float(conf), int(cls), (h, w), detector_source))
        outputs[image_path.name] = {
            "image_path": str(image_path),
            "image_id": image_path.name,
            "image_shape": {"height": h, "width": w},
            "detector_source": detector_source,
            "weights": str(weights),
            "confidence_threshold": conf_thres,
            "iou_threshold": iou_thres,
            "candidate_count": len(candidates),
            "candidates": sorted(candidates, key=lambda c: float(c["confidence"]), reverse=True),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def run_stage1_live(
    source: Path,
    run4_weights: Path = RUN4_WEIGHTS,
    run5_weights: Path = RUN5_WEIGHTS,
    run4_conf: float = 0.25,
    run5_conf: float = 0.15,
    device: str = "",
    imgsz: int = 640,
    nms_iou: float = 0.45,
    merge_iou: float = 0.40,
    half: bool = False,
    orientation_rescue: bool = True,
) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="orthox_stage1_orientation_") as tmp:
        detector_source = source
        variant_map: Dict[str, Dict[str, Any]] = {}
        if orientation_rescue:
            detector_source, variant_map = create_orientation_variant_batch(source, Path(tmp))

        run4 = run_detector_batch(
            run4_weights,
            detector_source,
            "run4_bg10_conf0.25",
            run4_conf,
            device,
            imgsz,
            nms_iou,
            half=half,
        )
        run5 = run_detector_batch(
            run5_weights,
            detector_source,
            "run5_tiny_os_conf0.15",
            run5_conf,
            device,
            imgsz,
            nms_iou,
            half=half,
        )

        if not orientation_rescue:
            image_ids = sorted(set(run4) | set(run5))
            records = []
            for image_id in image_ids:
                r4 = run4.get(image_id)
                r5 = run5.get(image_id)
                image_shape = (
                    (r4 or r5)["image_shape"]["height"],
                    (r4 or r5)["image_shape"]["width"],
                )
                raw_candidates = []
                if r4:
                    raw_candidates.extend(r4["candidates"])
                if r5:
                    raw_candidates.extend(r5["candidates"])
                merged = soft_merge_candidates(raw_candidates, image_shape=image_shape, iou_threshold=merge_iou)
                records.append(
                    {
                        "stage1_live_version": PIPELINE_STAGE1_VERSION,
                        "image_id": image_id,
                        "image_path": (r4 or r5)["image_path"],
                        "image_shape": (r4 or r5)["image_shape"],
                        "stage": "Stage 1 live detector adapter",
                        "important_scope": {
                            "runs_yolo_from_pixels": True,
                            "verifier_live": False,
                            "anatomy_live": False,
                            "stage2c_live": False,
                            "not_final_pipeline_output": True,
                        },
                        "policies": {
                            "run4_confidence": run4_conf,
                            "run5_confidence": run5_conf,
                            "per_detector_nms_iou": nms_iou,
                            "soft_merge_iou": merge_iou,
                            "image_size": imgsz,
                            "orientation_rescue_enabled": False,
                        },
                        "run4": r4,
                        "run5": r5,
                        "raw_candidate_count": len(raw_candidates),
                        "merged_candidate_count": len(merged),
                        "merged_candidates": merged,
                        "medical_scope_note": (
                            "Stage 1 live output is only raw fracture candidate detection. "
                            "It is not verifier-filtered, not anatomy-aware, and not a diagnosis."
                        ),
                    }
                )
            return records

        grouped: Dict[str, Dict[str, Any]] = {}
        for info in variant_map.values():
            image_id = info["original_image_id"]
            grouped.setdefault(
                image_id,
                {
                    "image_id": image_id,
                    "image_path": info["original_image_path"],
                    "image_shape": info["original_shape"],
                    "run4_original": [],
                    "run5_original": [],
                    "run4_rescue": [],
                    "run5_rescue": [],
                    "variant_counts": {},
                },
            )

        def collect_detector_outputs(outputs: Dict[str, Dict[str, Any]], detector_key: str) -> None:
            for variant_name, output in outputs.items():
                info = variant_map[variant_name]
                image_id = info["original_image_id"]
                target = grouped[image_id]
                transformed = [
                    transform_candidate_to_original(candidate, info)
                    for candidate in output.get("candidates", [])
                ]
                target["variant_counts"][f"{detector_key}:{info['orientation_variant']}"] = len(transformed)
                bucket = f"{detector_key}_{'rescue' if info['orientation_adjusted'] else 'original'}"
                target[bucket].extend(transformed)

        collect_detector_outputs(run4, "run4")
        collect_detector_outputs(run5, "run5")

        records = []
        for image_id in sorted(grouped):
            item = grouped[image_id]
            image_shape = (item["image_shape"]["height"], item["image_shape"]["width"])
            original_raw = item["run4_original"] + item["run5_original"]
            rescue_raw = item["run4_rescue"] + item["run5_rescue"]
            original_merged = soft_merge_candidates(original_raw, image_shape=image_shape, iou_threshold=merge_iou)
            rescue_merged_all = soft_merge_candidates(rescue_raw, image_shape=image_shape, iou_threshold=merge_iou)
            rescue_eligible = [
                candidate
                for candidate in rescue_merged_all
                if orientation_rescue_candidate_allowed(candidate)
            ]

            rescue_applied = False
            rescue_reason = "not_needed_original_candidates_present"
            selected_raw = original_raw
            merged = original_merged
            if not original_merged:
                if rescue_eligible:
                    rescue_applied = True
                    rescue_reason = "original_orientation_empty_rotated_candidate_retained"
                    selected_raw = rescue_raw
                    merged = rescue_eligible
                    for candidate in merged:
                        candidate["orientation_rescue_applied"] = True
                        candidate["orientation_rescue_reason"] = rescue_reason
                else:
                    rescue_reason = "original_orientation_empty_no_strong_rotated_candidate"
                    selected_raw = []
                    merged = []

            run4_candidates = item["run4_rescue" if rescue_applied else "run4_original"]
            run5_candidates = item["run5_rescue" if rescue_applied else "run5_original"]
            orientation_audit = {
                "enabled": True,
                "variants_tested": [name for name, _ in ORIENTATION_VARIANTS],
                "applied": rescue_applied,
                "decision": rescue_reason,
                "selection_policy": (
                    "Rotated detections are used only when the original orientation has no merged candidate. "
                    "A rotated candidate must have confidence >=0.75, or confidence >=0.45 with both Run4 and Run5 support."
                ),
                "original_raw_candidate_count": len(original_raw),
                "original_merged_candidate_count": len(original_merged),
                "rotated_raw_candidate_count": len(rescue_raw),
                "rotated_merged_candidate_count": len(rescue_merged_all),
                "rotated_eligible_candidate_count": len(rescue_eligible),
                "variant_candidate_counts": item["variant_counts"],
            }

            def detector_record(detector_key: str, candidates: List[Dict[str, Any]], conf_thres: float, weights: Path) -> Dict[str, Any]:
                return {
                    "image_path": item["image_path"],
                    "image_id": image_id,
                    "image_shape": item["image_shape"],
                    "detector_source": "run4_bg10_conf0.25" if detector_key == "run4" else "run5_tiny_os_conf0.15",
                    "weights": str(weights),
                    "confidence_threshold": conf_thres,
                    "iou_threshold": nms_iou,
                    "candidate_count": len(candidates),
                    "candidates": sorted(candidates, key=lambda c: float(c["confidence"]), reverse=True),
                    "orientation_rescue_selected": rescue_applied,
                }

            records.append(
                {
                    "stage1_live_version": PIPELINE_STAGE1_VERSION,
                    "image_id": image_id,
                    "image_path": item["image_path"],
                    "image_shape": item["image_shape"],
                    "stage": "Stage 1 live detector adapter",
                    "important_scope": {
                        "runs_yolo_from_pixels": True,
                        "orientation_rescue_live": True,
                        "verifier_live": False,
                        "anatomy_live": False,
                        "stage2c_live": False,
                        "not_final_pipeline_output": True,
                    },
                    "policies": {
                        "run4_confidence": run4_conf,
                        "run5_confidence": run5_conf,
                        "per_detector_nms_iou": nms_iou,
                        "soft_merge_iou": merge_iou,
                        "image_size": imgsz,
                        "orientation_rescue_enabled": True,
                        "orientation_rescue_use_only_when_original_empty": True,
                        "orientation_rescue_high_conf_threshold": 0.75,
                        "orientation_rescue_dual_detector_threshold": 0.45,
                    },
                    "run4": detector_record("run4", run4_candidates, run4_conf, run4_weights),
                    "run5": detector_record("run5", run5_candidates, run5_conf, run5_weights),
                    "orientation_rescue": orientation_audit,
                    "raw_candidate_count": len(selected_raw),
                    "merged_candidate_count": len(merged),
                    "merged_candidates": sorted(merged, key=lambda c: float(c["confidence"]), reverse=True),
                    "medical_scope_note": (
                        "Stage 1 live output is only raw fracture candidate detection. "
                        "Orientation rescue may use rotated internal copies, but returned boxes are mapped back "
                        "onto the original uploaded image. It is not verifier-filtered, not anatomy-aware, and not a diagnosis."
                    ),
                }
            )
        return records


def write_outputs(records: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        out_path = output_dir / f"{Path(record['image_id']).stem}.stage1_live.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rows.append(
            {
                "image_id": record["image_id"],
                "output": str(out_path),
                "run4_candidates": record["run4"]["candidate_count"] if record.get("run4") else 0,
                "run5_candidates": record["run5"]["candidate_count"] if record.get("run5") else 0,
                "raw_candidate_count": record["raw_candidate_count"],
                "merged_candidate_count": record["merged_candidate_count"],
                "top_confidence": record["merged_candidates"][0]["confidence"] if record["merged_candidates"] else None,
            }
        )
    summary = {
        "stage1_live_version": PIPELINE_STAGE1_VERSION,
        "processed_count": len(records),
        "total_merged_candidates": sum(r["merged_candidate_count"] for r in records),
        "outputs": rows,
        "status": "PASS",
        "scope": "stage1_only_raw_live_detector_adapter",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked Stage 1 Run4+Run5 detector adapter from image pixels.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="", help="'' auto, '0' CUDA, or 'cpu'.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--run4-conf", type=float, default=0.25)
    parser.add_argument("--run5-conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--merge-iou", type=float, default=0.40)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-orientation-rescue", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = run_stage1_live(
        source=Path(args.source),
        run4_conf=args.run4_conf,
        run5_conf=args.run5_conf,
        device=args.device,
        imgsz=args.imgsz,
        nms_iou=args.nms_iou,
        merge_iou=args.merge_iou,
        half=args.half,
        orientation_rescue=not args.no_orientation_rescue,
    )
    summary = write_outputs(records, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
