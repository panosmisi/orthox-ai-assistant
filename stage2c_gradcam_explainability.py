import argparse
import csv
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "stage1b_feasibility_v1"
    / "stage2c_visual_warning_audit"
    / "stage2c_visual_warning_audit.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "stage1b_feasibility_v1"
    / "stage2c_gradcam_explainability_v2"
)
DEFAULT_CHECKPOINT_GLOB = (
    REPO_ROOT
    / "runs"
    / "analysis"
    / "stage1b_feasibility_v1"
    / "training_runs"
    / "stage1b_resnet18_fracatlas_seed*"
    / "best.pt"
)
DEFAULT_LABEL_DIR = PROJECT_ROOT / "dataset_yolo" / "val" / "labels"


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


def make_transform(img_size: int = 384):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class GradCamRunner:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        target_layer = self.model.layer4[-1]
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def run(self, tensor: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor.to(self.device))
        score = logits[:, 0].sum()
        score.backward()
        probability = torch.sigmoid(logits[:, 0]).detach().cpu().item()

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = normalize01(cam)
        return cam, probability


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def read_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_yolo_boxes(label_path: Path, image_w: int, image_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = int((xc - bw / 2.0) * image_w)
        y1 = int((yc - bh / 2.0) * image_h)
        x2 = int((xc + bw / 2.0) * image_w)
        y2 = int((yc + bh / 2.0) * image_h)
        boxes.append((max(0, x1), max(0, y1), min(image_w - 1, x2), min(image_h - 1, y2)))
    return boxes


def foreground_mask(gray: np.ndarray):
    """Heuristic X-ray foreground mask. It is for display QA, not a clinical segmentation."""
    gray_u8 = gray.astype(np.uint8)
    h, w = gray_u8.shape
    border = max(4, int(min(h, w) * 0.06))
    border_pixels = np.concatenate(
        [
            gray_u8[:border, :].ravel(),
            gray_u8[-border:, :].ravel(),
            gray_u8[:, :border].ravel(),
            gray_u8[:, -border:].ravel(),
        ]
    )
    bg = float(np.median(border_pixels))
    p2, p98 = np.percentile(gray_u8, [2, 98])
    dyn = max(12.0, float(p98 - p2))
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)

    diff_mask = np.abs(blurred.astype(np.float32) - bg) > max(10.0, 0.10 * dyn)
    if bg < 100:
        intensity_mask = blurred > (p2 + 0.12 * dyn)
    elif bg > 170:
        intensity_mask = blurred < (p98 - 0.12 * dyn)
    else:
        intensity_mask = np.abs(blurred.astype(np.float32) - bg) > max(8.0, 0.08 * dyn)
    mask = (diff_mask | intensity_mask).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_area = max(80, int(h * w * 0.002))
    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == idx] = 1
    if cleaned.sum() < min_area:
        cleaned = mask

    cleaned = cv2.dilate(cleaned, np.ones((7, 7), np.uint8), iterations=1)
    return cleaned.astype(bool)


def cam_metrics(cam: np.ndarray, fg_mask: np.ndarray, gt_boxes):
    cam = normalize01(cam)
    total = float(cam.sum()) + 1e-8
    fg_energy = float(cam[fg_mask].sum() / total) if fg_mask.any() else 0.0

    threshold = float(np.quantile(cam, 0.80))
    top = cam >= threshold
    top_total = int(top.sum())
    top_fg = float((top & fg_mask).sum() / max(1, top_total))

    h, w = cam.shape
    border = max(2, int(min(h, w) * 0.05))
    border_mask = np.zeros_like(top, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    top_border = float((top & border_mask).sum() / max(1, top_total))

    gt_mask = np.zeros_like(top, dtype=bool)
    for x1, y1, x2, y2 in gt_boxes:
        gt_mask[y1 : y2 + 1, x1 : x2 + 1] = True
    top_gt = float((top & gt_mask).sum() / max(1, top_total)) if gt_boxes else None
    gt_mean = float(cam[gt_mask].mean()) if gt_mask.any() else None

    if top_fg >= 0.70 and top_border <= 0.20:
        quality = "plausible_attention"
        display_recommendation = "show_on_request"
    elif top_fg >= 0.50 and top_border <= 0.35:
        quality = "mixed_attention"
        display_recommendation = "show_on_request_with_caution"
    else:
        quality = "weak_or_artifact_risk"
        display_recommendation = "hide_by_default_or_show_with_strong_warning"

    return {
        "foreground_energy_fraction": fg_energy,
        "top20_foreground_fraction": top_fg,
        "top20_border_fraction": top_border,
        "top20_gt_fraction": top_gt,
        "gt_cam_mean": gt_mean,
        "attention_quality": quality,
        "display_recommendation": display_recommendation,
    }


def overlay_heatmap(rgb: Image.Image, cam: np.ndarray, alpha: float = 0.42):
    w, h = rgb.size
    cam_u8 = (normalize01(cam) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat_img = Image.fromarray(heat).resize((w, h), Image.BILINEAR)
    return Image.blend(rgb.convert("RGB"), heat_img, alpha)


def mask_preview(mask: np.ndarray, size):
    mask_u8 = (mask.astype(np.uint8) * 255)
    return Image.fromarray(mask_u8, mode="L").convert("RGB").resize(size, Image.NEAREST)


def draw_boxes(img: Image.Image, boxes, color=(0, 255, 0), width=4):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle(box, outline=color, width=width)
    return out


def add_title(img: Image.Image, title: str, height: int = 36):
    out = Image.new("RGB", (img.width, img.height + height), (20, 20, 20))
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 9), title[:150], fill=(240, 240, 240), font=font)
    return out


def make_panel(image_path: Path, raw_cam, masked_cam, fg_mask, boxes, row, metrics, out_path: Path):
    rgb = Image.open(image_path).convert("RGB")
    w, h = rgb.size
    raw_overlay = overlay_heatmap(rgb, raw_cam)
    masked_overlay = overlay_heatmap(rgb, masked_cam)
    fg_img = mask_preview(fg_mask, (w, h))

    base = draw_boxes(rgb, boxes)
    raw_overlay = draw_boxes(raw_overlay, boxes)
    masked_overlay = draw_boxes(masked_overlay, boxes)

    columns = [
        add_title(base, "Original + GT boxes"),
        add_title(raw_overlay, "Raw Grad-CAM"),
        add_title(masked_overlay, "Foreground-masked Grad-CAM"),
        add_title(fg_img, "Foreground mask"),
    ]
    panel = Image.new("RGB", (sum(c.width for c in columns), max(c.height for c in columns)), (230, 230, 230))
    x = 0
    for col in columns:
        panel.paste(col, (x, 0))
        x += col.width

    header = (
        f"{row.get('image_id')} | {row.get('decision')} | gt={row.get('gt_status')} | "
        f"p={float(row.get('mean_prob', 0.0)):.3f} | quality={metrics['attention_quality']}"
    )
    panel = add_title(panel, header, height=42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92)


def make_contact_sheet(image_paths, out_path: Path, thumb_w: int = 620, cols: int = 1):
    if not image_paths:
        return
    thumbs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        scale = thumb_w / im.width
        thumbs.append(im.resize((thumb_w, max(1, int(im.height * scale))), Image.LANCZOS))
    rows = math.ceil(len(thumbs) / cols)
    row_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (thumb_w * cols, row_h * rows), (235, 235, 235))
    for idx, im in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * row_h
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def select_rows(rows):
    selected = []
    for row in rows:
        decision = row.get("decision")
        gt_status = row.get("gt_status")
        if decision in {"suspicious", "uncertain"}:
            selected.append(row)
        elif decision == "probably_clean" and gt_status == "positive":
            selected.append(row)
    controls = [r for r in rows if r.get("decision") == "probably_clean" and r.get("gt_status") == "negative"]
    selected.extend(controls[:12])
    return selected


def run(args):
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    contact_dir = out_dir / "contact_sheets"
    heatmap_dir = out_dir / "heatmaps_npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoints = sorted(Path(p) for p in glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found with glob: {args.checkpoint_glob}")

    models_loaded = [build_model(p, device) for p in checkpoints]
    runners = [GradCamRunner(m, device) for m in models_loaded]
    tfm = make_transform(args.img_size)

    rows = select_rows(read_rows(Path(args.input_csv)))
    if args.limit:
        rows = rows[: args.limit]

    result_rows = []
    category_to_panels = {}
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        pil = Image.open(image_path).convert("RGB")
        tensor = tfm(pil).unsqueeze(0)

        cams = []
        probs = []
        for runner in runners:
            cam_small, prob = runner.run(tensor)
            cam = cv2.resize(cam_small, pil.size, interpolation=cv2.INTER_CUBIC)
            cams.append(normalize01(cam))
            probs.append(prob)
        raw_cam = normalize01(np.mean(cams, axis=0))

        gray = np.array(pil.convert("L"))
        fg = foreground_mask(gray)
        masked_cam = normalize01(raw_cam * (fg.astype(np.float32) * 0.90 + 0.10))

        boxes = read_yolo_boxes(Path(args.label_dir) / f"{image_path.stem}.txt", pil.width, pil.height)
        metrics = cam_metrics(raw_cam, fg, boxes)
        masked_metrics = cam_metrics(masked_cam, fg, boxes)

        category = f"{row.get('decision')}_{row.get('gt_status')}"
        panel_path = panels_dir / category / f"{image_path.stem}_gradcam_v2_panel.jpg"
        make_panel(image_path, raw_cam, masked_cam, fg, boxes, row, masked_metrics, panel_path)
        npy_path = heatmap_dir / category / f"{image_path.stem}_masked_cam.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, masked_cam)

        category_to_panels.setdefault(category, []).append(panel_path)
        result_rows.append(
            {
                "image_id": row.get("image_id"),
                "decision": row.get("decision"),
                "gt_status": row.get("gt_status"),
                "mean_prob_from_pipeline": row.get("mean_prob"),
                "gradcam_mean_prob": float(np.mean(probs)),
                "gradcam_min_prob": float(np.min(probs)),
                "gradcam_max_prob": float(np.max(probs)),
                "gt_box_count": row.get("gt_box_count"),
                "raw_attention_quality": metrics["attention_quality"],
                "masked_attention_quality": masked_metrics["attention_quality"],
                "masked_display_recommendation": masked_metrics["display_recommendation"],
                "raw_top20_foreground_fraction": metrics["top20_foreground_fraction"],
                "masked_top20_foreground_fraction": masked_metrics["top20_foreground_fraction"],
                "raw_top20_border_fraction": metrics["top20_border_fraction"],
                "masked_top20_border_fraction": masked_metrics["top20_border_fraction"],
                "raw_top20_gt_fraction": metrics["top20_gt_fraction"],
                "masked_top20_gt_fraction": masked_metrics["top20_gt_fraction"],
                "raw_gt_cam_mean": metrics["gt_cam_mean"],
                "masked_gt_cam_mean": masked_metrics["gt_cam_mean"],
                "panel_path": str(panel_path),
                "masked_heatmap_npy_path": str(npy_path),
            }
        )

    for runner in runners:
        runner.close()

    csv_path = out_dir / "stage2c_gradcam_explainability_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(result_rows[0].keys()) if result_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    for category, paths in category_to_panels.items():
        make_contact_sheet(paths, contact_dir / f"{category}.jpg", cols=1)
    make_contact_sheet([p for paths in category_to_panels.values() for p in paths], contact_dir / "ALL_selected.jpg", cols=1)

    quality_counts = {}
    display_counts = {}
    for r in result_rows:
        quality_counts[r["masked_attention_quality"]] = quality_counts.get(r["masked_attention_quality"], 0) + 1
        display_counts[r["masked_display_recommendation"]] = display_counts.get(r["masked_display_recommendation"], 0) + 1
    warning_rows = [r for r in result_rows if r["decision"] in {"suspicious", "uncertain"}]
    warning_quality_counts = {}
    warning_display_counts = {}
    for r in warning_rows:
        warning_quality_counts[r["masked_attention_quality"]] = warning_quality_counts.get(r["masked_attention_quality"], 0) + 1
        warning_display_counts[r["masked_display_recommendation"]] = warning_display_counts.get(r["masked_display_recommendation"], 0) + 1

    positives = [r for r in result_rows if r["gt_status"] == "positive"]
    avg_raw_gt = np.mean([float(r["raw_top20_gt_fraction"]) for r in positives if r["raw_top20_gt_fraction"] not in (None, "", "None")]) if positives else None
    avg_masked_gt = np.mean([float(r["masked_top20_gt_fraction"]) for r in positives if r["masked_top20_gt_fraction"] not in (None, "", "None")]) if positives else None

    summary = {
        "out_dir": str(out_dir),
        "input_csv": str(args.input_csv),
        "selected_image_count": len(result_rows),
        "device": str(device),
        "checkpoint_count": len(checkpoints),
        "quality_counts_all_selected": quality_counts,
        "quality_counts_stage2c_warnings_only": warning_quality_counts,
        "display_recommendation_counts_all_selected": display_counts,
        "display_recommendation_counts_stage2c_warnings_only": warning_display_counts,
        "avg_raw_top20_gt_fraction_positive": None if avg_raw_gt is None else float(avg_raw_gt),
        "avg_masked_top20_gt_fraction_positive": None if avg_masked_gt is None else float(avg_masked_gt),
        "interpretation": {
            "heatmap_name": "Stage 2C attention heatmap",
            "not_a_fracture_bbox": True,
            "not_clinical_localization": True,
            "masked_heatmap_reduces_background_artifact_risk": True,
        },
        "recommended_official_pipeline_policy": {
            "generate_when": "Stage 2C runs and decision is suspicious or uncertain",
            "do_not_generate_when": "accepted Stage 1/2A candidate already exists or Stage 2C decision is probably_clean",
            "ui_behavior": "hidden by default; user can click View attention heatmap",
            "json_key": "stage2c_explainability",
            "required_fields": [
                "available",
                "heatmap_name",
                "attention_quality",
                "display_recommendation",
                "heatmap_path",
                "panel_path",
                "disclaimer",
            ],
            "mandatory_disclaimer": "The heatmap shows image regions that influenced the Stage 2C safety classifier. It is not a fracture boundary and should not replace clinical review.",
        },
    }
    (out_dir / "stage2c_gradcam_explainability_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Stage 2C Grad-CAM Explainability v2",
        "",
        "This run hardens the original Grad-CAM pilot with a heuristic foreground mask and attention-quality flags.",
        "",
        "Strict interpretation: this is an attention explanation for the Stage 2C image-level safety classifier. It is not a fracture box.",
        "",
        "## Results",
        f"- Selected images: {len(result_rows)}",
        f"- Checkpoints: {len(checkpoints)}",
        f"- Quality counts, all selected: {quality_counts}",
        f"- Quality counts, Stage 2C warnings only: {warning_quality_counts}",
        f"- Display recommendations, all selected: {display_counts}",
        f"- Display recommendations, Stage 2C warnings only: {warning_display_counts}",
        f"- Avg raw top-20% CAM fraction inside GT boxes, positives: {summary['avg_raw_top20_gt_fraction_positive']}",
        f"- Avg masked top-20% CAM fraction inside GT boxes, positives: {summary['avg_masked_top20_gt_fraction_positive']}",
        "",
        "## Adoptable Contract",
        "- Show the heatmap only on user request, e.g. `View attention heatmap`.",
        "- Display it only as `Stage 2C attention heatmap`.",
        "- Generate it only when Stage 2C actually runs and returns `suspicious` or `uncertain`.",
        "- Do not generate it when an accepted Stage 1/2A candidate already exists; the YOLO box remains the primary localization output.",
        "- Do not show it by default for `probably_clean` images.",
        "- Always show the attention quality flag.",
        "- If quality is `weak_or_artifact_risk`, the UI must warn that the heatmap explanation is weak.",
        "- Never call this `fracture localization` or `fracture heatmap`.",
        "",
        "## Strict Technical Interpretation",
        "- Foreground masking improves visual cleanliness and reduces background/border clutter.",
        "- It does not solve localization: the average top-20% CAM fraction inside GT boxes remains very low.",
        "- Therefore the feature is useful for explainability, not for finding the fracture.",
        "- The correct product behavior is a secondary `View attention heatmap` action, not a default diagnostic overlay.",
        "",
        "## Outputs",
        f"- CSV: `{csv_path}`",
        f"- Contact sheets: `{contact_dir}`",
        f"- Panels: `{panels_dir}`",
    ]
    (out_dir / "STAGE2C_GRADCAM_EXPLAINABILITY_V2.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2C foreground-masked Grad-CAM explainability audit.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--checkpoint-glob", default=str(DEFAULT_CHECKPOINT_GLOB))
    parser.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR))
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
