from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "live_stage2a_verifier_v1"
STAGE2A_VERSION = "stage2a_live_resnet18_multiview_6seed_v2"
DEFAULT_CHECKPOINTS = [
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "exp01_resnet18_multiview" / "best.pt",
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "exp02_resnet18_multiview_seed1" / "best.pt",
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "stability_seed2" / "best.pt",
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "stability_seed3" / "best.pt",
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "stability_seed4" / "best.pt",
    REPO_ROOT / "runs" / "analysis" / "verifier_training" / "stability_seed5" / "best.pt",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultiViewVerifier(nn.Module):
    def __init__(self, metadata_dim: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512 * 3 + metadata_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, local: torch.Tensor, context: torch.Tensor, full: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                self.backbone(local),
                self.backbone(context),
                self.backbone(full),
                metadata,
            ],
            dim=1,
        )
        return self.head(features).squeeze(1)


@dataclass
class LoadedVerifier:
    seed_name: str
    path: Path
    model: MultiViewVerifier
    threshold: float
    metadata_columns: List[str]
    metadata_mean: torch.Tensor
    metadata_std: torch.Tensor


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_verifier_checkpoint(path: Path, device: torch.device) -> LoadedVerifier:
    ckpt = torch.load(path, map_location=device)
    columns = list(ckpt["metadata_columns"])
    model = MultiViewVerifier(metadata_dim=len(columns)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    seed = str((ckpt.get("args") or {}).get("seed", path.parent.name))
    threshold = float((ckpt.get("selected") or {}).get("threshold", 0.5))
    mean = torch.tensor(ckpt["metadata_mean"], dtype=torch.float32, device=device).view(1, -1)
    std = torch.tensor(ckpt["metadata_std"], dtype=torch.float32, device=device).view(1, -1)
    std = torch.clamp(std, min=1e-6)
    return LoadedVerifier(
        seed_name=f"seed{seed}",
        path=path,
        model=model,
        threshold=threshold,
        metadata_columns=columns,
        metadata_mean=mean,
        metadata_std=std,
    )


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


def expand_box(box: Iterable[float], factor: float, width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side_w = bw * factor
    side_h = bh * factor
    return clamp_box([cx - side_w / 2.0, cy - side_h / 2.0, cx + side_w / 2.0, cy + side_h / 2.0], width, height)


def crop_resize(image: Image.Image, box: Iterable[float], size: int) -> Image.Image:
    width, height = image.size
    crop_box = clamp_box(box, width, height)
    crop = image.crop(tuple(crop_box))
    return crop.resize((size, size), Image.BILINEAR).convert("RGB")


def build_views(image: Image.Image, candidate: Dict[str, Any]) -> Tuple[Image.Image, Image.Image, Image.Image]:
    width, height = image.size
    xyxy = candidate["xyxy_pixels"]
    local_box = expand_box(xyxy, factor=2.0, width=width, height=height)
    context_box = expand_box(xyxy, factor=5.0, width=width, height=height)
    local = crop_resize(image, local_box, 224)
    context = crop_resize(image, context_box, 224)

    full = image.convert("RGB").copy()
    draw = ImageDraw.Draw(full)
    x1, y1, x2, y2 = clamp_box(xyxy, width, height)
    line_width = max(2, int(round(min(width, height) * 0.006)))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
    full = full.resize((384, 384), Image.BILINEAR)
    return local, context, full


def candidate_source_is_run5(candidate: Dict[str, Any]) -> float:
    sources = candidate.get("detector_sources") or [candidate.get("detector_source", "")]
    return 1.0 if any("run5" in str(source).lower() for source in sources) else 0.0


def metadata_values(candidate: Dict[str, Any], columns: List[str]) -> List[float]:
    xywhn = candidate.get("xywhn") or [0.0, 0.0, 0.0, 0.0]
    w = float(xywhn[2])
    h = float(xywhn[3])
    values = {
        "candidate_conf": float(candidate.get("confidence", 0.0)),
        "candidate_x": float(xywhn[0]),
        "candidate_y": float(xywhn[1]),
        "candidate_w": w,
        "candidate_h": h,
        "candidate_area": w * h,
        "candidate_aspect": w / max(h, 1e-6),
        "candidate_source_is_run5": candidate_source_is_run5(candidate),
    }
    return [values[column] for column in columns]


def transform_views(local: Image.Image, context: Image.Image, full: Image.Image, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tf224 = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tf384 = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return (
        tf224(local).unsqueeze(0).to(device),
        tf224(context).unsqueeze(0).to(device),
        tf384(full).unsqueeze(0).to(device),
    )


def verify_candidate(
    image: Image.Image,
    candidate: Dict[str, Any],
    verifiers: List[LoadedVerifier],
    min_votes: int = 2,
) -> Dict[str, Any]:
    device = next(verifiers[0].model.parameters()).device
    local, context, full = build_views(image, candidate)
    local_t, context_t, full_t = transform_views(local, context, full, device)

    seed_results = []
    probs = []
    votes = 0
    with torch.no_grad():
        for verifier in verifiers:
            raw_metadata = torch.tensor([metadata_values(candidate, verifier.metadata_columns)], dtype=torch.float32, device=device)
            metadata = (raw_metadata - verifier.metadata_mean) / verifier.metadata_std
            logit = verifier.model(local_t, context_t, full_t, metadata)
            prob = float(torch.sigmoid(logit).item())
            keep = prob >= verifier.threshold
            votes += int(keep)
            probs.append(prob)
            seed_results.append(
                {
                    "seed": verifier.seed_name,
                    "checkpoint": str(verifier.path),
                    "probability": prob,
                    "threshold": verifier.threshold,
                    "keep_vote": keep,
                }
            )

    keep_final = votes >= min_votes
    prob_mean = sum(probs) / len(probs) if probs else None
    result = dict(candidate)
    result.update(
        {
            "verifier_keep": keep_final,
            "verifier_keep_votes": votes,
            "verifier_total_votes": len(verifiers),
            "verifier_min_votes": min_votes,
            "verifier_prob_mean": prob_mean,
            "verifier_probs": probs,
            "verifier_seed_results": seed_results,
            "verifier_policy": "runtime_safe_6seed_keep_if_votes_gte_2",
            "verifier_crop_note": (
                "Runtime local/context/full-marked views are reconstructed from the candidate box. "
                "Checkpoint loading and saved-crop preprocessing were parity-checked; generated live crops still require audit."
            ),
        }
    )
    return result


def verify_stage1_record(
    stage1_record: Dict[str, Any],
    checkpoints: List[Path] = DEFAULT_CHECKPOINTS,
    min_votes: int = 2,
    device: str = "",
) -> Dict[str, Any]:
    selected_device = torch.device(f"cuda:{device}" if device and device != "cpu" and torch.cuda.is_available() else "cuda:0" if not device and torch.cuda.is_available() else "cpu")
    verifiers = [load_verifier_checkpoint(path, selected_device) for path in checkpoints]
    return verify_stage1_record_with_loaded_verifiers(stage1_record, verifiers, min_votes=min_votes)


def verify_stage1_record_with_loaded_verifiers(
    stage1_record: Dict[str, Any],
    verifiers: List[LoadedVerifier],
    min_votes: int = 2,
) -> Dict[str, Any]:
    image = Image.open(stage1_record["image_path"]).convert("RGB")
    raw_candidates = stage1_record.get("merged_candidates") or []
    verified = [verify_candidate(image, candidate, verifiers, min_votes=min_votes) for candidate in raw_candidates]
    accepted = [c for c in verified if c["verifier_keep"]]
    rejected = [c for c in verified if not c["verifier_keep"]]
    accepted_sorted = sorted(accepted, key=lambda c: (c.get("verifier_keep_votes", 0), c.get("verifier_prob_mean") or 0.0, c.get("confidence", 0.0)), reverse=True)
    rejected_sorted = sorted(rejected, key=lambda c: (c.get("verifier_keep_votes", 0), c.get("verifier_prob_mean") or 0.0, c.get("confidence", 0.0)), reverse=True)
    return {
        "stage2a_live_version": STAGE2A_VERSION,
        "image_id": stage1_record.get("image_id"),
        "image_path": stage1_record.get("image_path"),
        "stage": "Stage 2A live verifier adapter",
        "input_stage1_live_version": stage1_record.get("stage1_live_version"),
        "policy": {
            "ensemble": "6_seed_resnet18_multiview",
            "min_votes": min_votes,
            "candidate_views": ["local_crop_224", "context_crop_224", "full_marked_384"],
            "metadata_columns": verifiers[0].metadata_columns if verifiers else [],
        },
        "important_scope": {
            "verifier_live": True,
            "artifact_suppression_live": False,
            "anatomy_live": False,
            "stage2c_live": False,
            "not_final_pipeline_output": True,
        },
        "raw_candidate_count": len(raw_candidates),
        "accepted_candidate_count": len(accepted_sorted),
        "rejected_candidate_count": len(rejected_sorted),
        "accepted_candidates": accepted_sorted,
        "rejected_candidates": rejected_sorted,
        "medical_scope_note": (
            "Stage 2A live output filters raw Stage 1 candidates. It is not artifact-filtered, not anatomy-aware, "
            "and not a standalone diagnosis."
        ),
    }


def write_stage2a_output(record: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(record['image_id']).stem}.stage2a_live.json"
    write_json(out_path, record)
    summary = {
        "stage2a_live_version": STAGE2A_VERSION,
        "processed_count": 1,
        "image_id": record["image_id"],
        "raw_candidate_count": record["raw_candidate_count"],
        "accepted_candidate_count": record["accepted_candidate_count"],
        "rejected_candidate_count": record["rejected_candidate_count"],
        "output": str(out_path),
        "status": "PASS",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2A live verifier on a Stage 1 live JSON record.")
    parser.add_argument("--stage1-json", required=True, help="Path to *.stage1_live.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="", help="'' auto, '0' CUDA, or 'cpu'.")
    parser.add_argument("--min-votes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1 = load_json(Path(args.stage1_json))
    record = verify_stage1_record(stage1, min_votes=args.min_votes, device=args.device)
    summary = write_stage2a_output(record, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
