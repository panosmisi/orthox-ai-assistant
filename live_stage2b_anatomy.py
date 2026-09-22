from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "stage2b_student_runs"
    / "fracatlas_finetune_from_mura_lera_effb0_v1"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "live_stage2b_anatomy_v1"

FINE_TO_PARENT = {
    "ankle_foot": "leg",
    "knee_lower_leg": "leg",
    "pelvis_hip_femur": "hip",
    "shoulder_upper_arm": "shoulder",
    "elbow": "hand",
    "forearm": "hand",
    "wrist_hand": "hand",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rank(probs: Dict[str, float]) -> List[Dict[str, Any]]:
    return [
        {"label": label, "probability": prob, "percent": prob * 100.0}
        for label, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True)
    ]


def _top2(probs: Dict[str, float]) -> Tuple[str, float, str | None, float | None]:
    ranking = _rank(probs)
    top1 = ranking[0]
    top2 = ranking[1] if len(ranking) > 1 else None
    return (
        str(top1["label"]),
        float(top1["probability"]),
        str(top2["label"]) if top2 else None,
        float(top2["probability"]) if top2 else None,
    )


def parent_probs_from_fine(fine_probs: Dict[str, float]) -> Dict[str, float]:
    parent: Dict[str, float] = {}
    for fine, prob in fine_probs.items():
        parent_label = FINE_TO_PARENT.get(fine, "unknown")
        parent[parent_label] = parent.get(parent_label, 0.0) + float(prob)
    total = sum(parent.values())
    if total > 0:
        parent = {k: v / total for k, v in parent.items()}
    return parent


def load_model(checkpoint: Path, device: torch.device) -> Tuple[nn.Module, List[str], int]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    classes = list(ckpt["classes"])
    img_size = int((ckpt.get("args") or {}).get("img_size", 224))
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, classes, img_size


def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def classify_anatomy(
    image_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: str = "",
    fine_threshold: float = 0.80,
) -> Dict[str, Any]:
    selected_device = torch.device(
        f"cuda:{device}" if device and device != "cpu" and torch.cuda.is_available()
        else "cuda:0" if not device and torch.cuda.is_available()
        else "cpu"
    )
    model, classes, img_size = load_model(checkpoint, selected_device)
    tf = make_transform(img_size)
    image = Image.open(image_path).convert("RGB")
    tensor = tf(image).unsqueeze(0).to(selected_device)
    logits = model(tensor)
    probs_t = torch.softmax(logits, dim=1).detach().cpu().view(-1)
    fine_probs = {label: float(probs_t[idx].item()) for idx, label in enumerate(classes)}
    parent_probs = parent_probs_from_fine(fine_probs)

    fine_top1, fine_conf, fine_top2, fine_top2_conf = _top2(fine_probs)
    parent_top1, parent_conf, parent_top2, parent_top2_conf = _top2(parent_probs)
    output_type = "fine" if fine_conf >= fine_threshold else "parent_fallback"
    selected_label = fine_top1 if output_type == "fine" else parent_top1
    selected_conf = fine_conf if output_type == "fine" else parent_conf
    reason = "fine_confidence_above_threshold" if output_type == "fine" else "fine_confidence_below_threshold_parent_fallback"

    result = {
        "policy": {
            "output_type": output_type,
            "label": selected_label,
            "parent_label": FINE_TO_PARENT.get(fine_top1, parent_top1) if output_type == "fine" else parent_top1,
            "confidence": selected_conf,
            "threshold": fine_threshold,
            "reason": reason,
        },
        "parent_classifier": {
            "top1": parent_top1,
            "top1_conf": parent_conf,
            "top2": parent_top2,
            "top2_conf": parent_top2_conf,
            "margin": parent_conf - (parent_top2_conf or 0.0),
            "all_probs": parent_probs,
            "probability_ranking": _rank(parent_probs),
            "note": "Runtime parent probabilities are aggregated from the fine 7-class classifier.",
        },
        "fine_classifier": {
            "top1": fine_top1,
            "top1_conf": fine_conf,
            "top2": fine_top2,
            "top2_conf": fine_top2_conf,
            "margin": fine_conf - (fine_top2_conf or 0.0),
            "all_probs": fine_probs,
            "probability_ranking": _rank(fine_probs),
        },
        "runtime_metadata": {
            "stage": "Stage 2B whole-image anatomy classifier",
            "version": "live_stage2b_effb0_fine7_v1",
            "checkpoint": str(checkpoint),
            "image_path": str(image_path),
            "img_size": img_size,
            "device": str(selected_device),
        },
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live Stage 2B anatomy classifier on one X-ray.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="")
    parser.add_argument("--fine-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Path(args.image)
    result = classify_anatomy(
        image_path=image,
        checkpoint=Path(args.checkpoint),
        device=args.device,
        fine_threshold=args.fine_threshold,
    )
    out_dir = Path(args.output_dir)
    out_path = out_dir / f"{image.stem}.stage2b_anatomy.json"
    write_json(out_path, result)
    print(json.dumps({"status": "PASS", "output": str(out_path), "policy": result["policy"]}, indent=2))


if __name__ == "__main__":
    main()
