from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_GLOB = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "stage1b_feasibility_v1"
    / "training_runs"
    / "stage1b_resnet18_fracatlas_seed*"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "live_stage2c_safety_v1"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_paths(pattern: Path | str = DEFAULT_CHECKPOINT_GLOB) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(str(pattern)))]


def build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if ckpt.get("arch") != "resnet18":
        raise ValueError(f"Unsupported Stage 2C checkpoint arch: {ckpt.get('arch')}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def make_transform(img_size: int = 384) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def run_stage2c_safety(
    image_path: Path,
    accepted_candidate_count: int,
    device: str = "",
    checkpoint_glob: Path | str = DEFAULT_CHECKPOINT_GLOB,
    suspicious_threshold: float = 0.70,
    uncertain_low: float = 0.60,
    vote_count: int = 2,
) -> Dict[str, Any]:
    base = {
        "stage": "Stage 2C Clean-Case Safety Warning Layer",
        "version": "stage1b_resnet18_fracatlas_3seed_screening_first_v1",
        "policy": {
            "policy_name": "screening_first",
            "mode": "mean_or_votes",
            "suspicious_threshold": suspicious_threshold,
            "uncertain_low": uncertain_low,
            "vote_count": vote_count,
            "trigger": "run_only_when_final_detector_verifier_has_no_accepted_candidate",
        },
        "image_id": image_path.name,
        "image_path": str(image_path),
        "important_semantics": {
            "not_a_detector": True,
            "does_not_create_bounding_boxes": True,
            "does_not_suppress_existing_candidates": True,
            "not_a_diagnosis": True,
        },
        "human_verification_required": True,
        "clinician_review_required": True,
        "legacy_internal_name": "stage1b",
        "stage2c_evaluated": True,
        "accepted_candidate_count_before_stage2c": accepted_candidate_count,
    }
    if accepted_candidate_count > 0:
        return {
            **base,
            "decision": "not_run_existing_candidate_present",
            "screening_message": (
                "Stage 2C did not run because at least one localized fracture candidate was already retained. "
                "Stage 2C is an image-level safety layer only for clean/no-box cases."
            ),
            "probability_summary": {
                "mean_suspicious_probability": None,
                "min_suspicious_probability": None,
                "max_suspicious_probability": None,
                "probability_range": None,
                "suspicious_votes": None,
                "uncertain_votes": None,
                "seed_count": 0,
            },
            "per_seed": [],
            "stage2c_was_run": False,
        }

    paths = checkpoint_paths(checkpoint_glob)
    if not paths:
        raise FileNotFoundError(f"No Stage 2C checkpoints found for pattern: {checkpoint_glob}")
    selected_device = torch.device(
        f"cuda:{device}" if device and device != "cpu" and torch.cuda.is_available()
        else "cuda:0" if not device and torch.cuda.is_available()
        else "cpu"
    )
    image = Image.open(image_path).convert("RGB")
    tensor = make_transform(384)(image).unsqueeze(0).to(selected_device)
    per_seed = []
    probs = []
    for path in paths:
        model = build_model(path, selected_device)
        logit = model(tensor)
        prob = float(torch.sigmoid(logit.view(-1)[0]).item())
        ckpt = torch.load(path, map_location="cpu")
        seed = ckpt.get("seed", path.parent.name)
        probs.append(prob)
        per_seed.append({"seed": f"seed{seed}", "checkpoint": str(path), "suspicious_probability": prob})
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_prob = sum(probs) / len(probs)
    suspicious_votes = sum(1 for p in probs if p >= suspicious_threshold)
    uncertain_votes = sum(1 for p in probs if p >= uncertain_low)
    if mean_prob >= suspicious_threshold or suspicious_votes >= vote_count:
        decision = "suspicious"
        message = "No high-confidence fracture candidate was accepted, but the image-level safety check flagged this X-ray as suspicious. Human verification required."
    elif mean_prob >= uncertain_low or uncertain_votes >= vote_count:
        decision = "uncertain"
        message = "No high-confidence fracture candidate was accepted, but the image-level safety check is uncertain. Human verification remains important."
    else:
        decision = "probably_clean"
        message = "No high-confidence fracture candidate was accepted and the image-level safety check did not raise a strong warning. This does not rule out subtle injury."

    return {
        **base,
        "decision": decision,
        "screening_message": message,
        "probability_summary": {
            "mean_suspicious_probability": mean_prob,
            "min_suspicious_probability": min(probs),
            "max_suspicious_probability": max(probs),
            "probability_range": max(probs) - min(probs),
            "suspicious_votes": suspicious_votes,
            "uncertain_votes": uncertain_votes,
            "seed_count": len(probs),
        },
        "per_seed": per_seed,
        "stage2c_was_run": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live Stage 2C image-level safety layer.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--accepted-candidate-count", type=int, required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Path(args.image)
    result = run_stage2c_safety(image, args.accepted_candidate_count, device=args.device)
    out_path = Path(args.output_dir) / f"{image.stem}.stage2c_safety.json"
    write_json(out_path, result)
    print(json.dumps({"status": "PASS", "output": str(out_path), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
