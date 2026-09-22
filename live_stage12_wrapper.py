from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from validate_locked_pipeline_outputs import validate_record


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCKED_STAGE12_DIR = (
    REPO_ROOT / "runs" / "analysis" / "stage1b_feasibility_v1" / "stage2c_full_val_enabled"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "runs" / "analysis" / "live_stage12_wrapper_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LiveStage12NotImplemented(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    paths = []
    for path in sorted(input_path.iterdir()):
        if path.name == "summary.json":
            continue
        if path.suffix.lower() == ".json" or path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def locked_json_for_image(image_path: Path, locked_stage12_dir: Path) -> Path:
    candidates = [
        locked_stage12_dir / f"{image_path.stem}.json",
        locked_stage12_dir / image_path.with_suffix(".json").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No locked Stage 1/2 JSON found for image {image_path}. "
        f"Expected one of: {', '.join(str(c) for c in candidates)}"
    )


def resolve_precomputed_input(input_path: Path, locked_stage12_dir: Path) -> Tuple[Path, str]:
    if input_path.suffix.lower() == ".json":
        return input_path, "locked_stage12_json_input"
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        return locked_json_for_image(input_path, locked_stage12_dir), "image_lookup_precomputed_stage12"
    raise ValueError(f"Unsupported input type: {input_path}")


def output_path_for_input(input_path: Path, output_root: Path, single_output: bool) -> Path:
    if single_output and output_root.suffix.lower() == ".json":
        return output_root
    if input_path.suffix.lower() == ".json":
        return output_root / input_path.name
    return output_root / f"{input_path.stem}.json"


def artifact_entry(path: Path, role: str, required_for: str) -> Dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "required_for": required_for,
    }


def build_artifact_manifest() -> Dict[str, Any]:
    run4 = REPO_ROOT / "runs" / "train" / "yolov9-c_fracatlas_bg10_recall"
    run5 = REPO_ROOT / "runs" / "train" / "yolov9-c_fracatlas_tiny_os_run5"
    verifier_root = REPO_ROOT / "runs" / "analysis" / "verifier_training"
    stage2b = (
        REPO_ROOT
        / "runs"
        / "analysis"
        / "stage2b_student_runs"
        / "fracatlas_finetune_from_mura_lera_effb0_v1"
    )
    stage2c = REPO_ROOT / "runs" / "analysis" / "stage1b_feasibility_v1" / "training_runs"
    lock_dir = REPO_ROOT / "runs" / "analysis" / "pipeline_lock_v1"

    artifacts = [
        artifact_entry(lock_dir / "pipeline_lock_v1_manifest.json", "locked pipeline manifest", "all modes"),
        artifact_entry(lock_dir / "PIPELINE_LOCK_V1.md", "human-readable pipeline lock report", "audit"),
        artifact_entry(DEFAULT_LOCKED_STAGE12_DIR, "validated locked Stage 1/2 JSON directory", "precomputed mode"),
        artifact_entry(run4 / "weights" / "best.pt", "Stage 1 Run4 YOLOv9-c detector weight", "future live mode"),
        artifact_entry(run4 / "opt.yaml", "Stage 1 Run4 train/inference context", "audit/live reconstruction"),
        artifact_entry(run5 / "weights" / "best.pt", "Stage 1 Run5 YOLOv9-c detector weight", "future live mode"),
        artifact_entry(run5 / "opt.yaml", "Stage 1 Run5 train/inference context", "audit/live reconstruction"),
        artifact_entry(
            REPO_ROOT / "runs" / "analysis" / "verifier_official_ensemble_report" / "official_ensemble_report.json",
            "older verifier official ensemble report",
            "audit only",
        ),
        artifact_entry(verifier_root / "exp01_resnet18_multiview" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(verifier_root / "exp02_resnet18_multiview_seed1" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(verifier_root / "stability_seed2" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(verifier_root / "stability_seed3" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(verifier_root / "stability_seed4" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(verifier_root / "stability_seed5" / "best.pt", "Verifier seed/model candidate", "future live mode"),
        artifact_entry(stage2b / "best.pt", "Stage 2B EfficientNet-B0 anatomy classifier", "future live mode"),
        artifact_entry(stage2b / "config.json", "Stage 2B class map/config", "future live mode"),
        artifact_entry(stage2b / "final_report.json", "Stage 2B validation/test report", "audit"),
        artifact_entry(stage2c / "stage1b_resnet18_fracatlas_seed0" / "best.pt", "Stage 2C ResNet18 seed0", "future live mode"),
        artifact_entry(stage2c / "stage1b_resnet18_fracatlas_seed1" / "best.pt", "Stage 2C ResNet18 seed1", "future live mode"),
        artifact_entry(stage2c / "stage1b_resnet18_fracatlas_seed2" / "best.pt", "Stage 2C ResNet18 seed2", "future live mode"),
        artifact_entry(REPO_ROOT / "stage2c_gradcam_explainability.py", "optional Stage 2C Grad-CAM heatmap", "optional UI"),
        artifact_entry(REPO_ROOT / "validate_locked_pipeline_outputs.py", "Stage 1/2 contract validator", "all modes"),
        artifact_entry(REPO_ROOT / "live_stage1_detector.py", "Stage 1 live Run4+Run5 detector adapter", "future live mode"),
        artifact_entry(REPO_ROOT / "live_stage2a_verifier.py", "Stage 2A live verifier adapter", "future live mode"),
    ]

    missing_required_precomputed = [
        item for item in artifacts if item["required_for"] in {"all modes", "precomputed mode"} and not item["exists"]
    ]
    future_live_missing = [
        item for item in artifacts if item["required_for"] == "future live mode" and not item["exists"]
    ]
    return {
        "wrapper_name": "live_stage12_wrapper_v1",
        "status": "precomputed_locked_mode_ready",
        "honest_scope": {
            "precomputed_locked_stage12_mode": "implemented",
            "stage1_live_detector_adapter": "implemented",
            "stage2a_live_verifier_adapter": "implemented_smoke_tested_not_locked",
            "live_pixel_to_stage12_mode": "not_implemented_yet",
            "reason": (
                "The locked JSON contract is validated and app-ready. Stage 1 Run4/Run5 live detection is now "
                "implemented separately and Stage 2A verifier has a smoke-tested live adapter, but full true live "
                "mode still needs artifact, anatomy, Stage 2C, and final schema adapters wired and validated."
            ),
        },
        "locked_pipeline": load_json(lock_dir / "pipeline_lock_v1_manifest.json")
        if (lock_dir / "pipeline_lock_v1_manifest.json").exists()
        else None,
        "artifact_status": {
            "all_precomputed_required_artifacts_present": len(missing_required_precomputed) == 0,
            "missing_required_precomputed": missing_required_precomputed,
            "future_live_missing_artifacts": future_live_missing,
        },
        "artifacts": artifacts,
        "live_adapter_plan": [
            "Stage 1 adapter: run Run4@0.25 and Run5@0.15 from image pixels, then apply the locked soft-merge policy.",
            "Stage 2A adapter: implemented as smoke-tested 6-seed ResNet18 multiview verifier; needs broader validation before lock.",
            "Stage 2A.5 adapter: apply production strict artifact suppression to retained candidate crops.",
            "Stage 2B adapter: run whole-image anatomy classifier and apply fine>=0.80 else parent fallback.",
            "Stage 2C adapter: run only when no accepted candidate remains; never create a bounding box.",
            "Contract adapter: emit exactly the same JSON schema as the locked Stage 1/2 outputs.",
        ],
        "known_policy_note": (
            "The locked JSON outputs are the source of truth for production behavior. Older verifier reports may "
            "contain different seed/vote policies from exploratory runs; they are retained for audit, not for "
            "overriding the locked contract."
        ),
    }


def run_precomputed_one(
    input_path: Path,
    output_path: Path,
    locked_stage12_dir: Path = DEFAULT_LOCKED_STAGE12_DIR,
    validate: bool = True,
) -> Dict[str, Any]:
    upstream_json, input_mode = resolve_precomputed_input(input_path, locked_stage12_dir)
    data = load_json(upstream_json)
    data["stage12_wrapper_metadata"] = {
        "wrapper": "live_stage12_wrapper_v1",
        "mode": "precomputed_locked_stage12",
        "input_path": str(input_path),
        "input_mode": input_mode,
        "locked_stage12_json": str(upstream_json),
        "live_pixel_to_stage12_inference": False,
        "note": (
            "This output comes from the validated locked Stage 1/2 JSON contract. "
            "It is suitable for app integration testing, but it is not fresh pixel-to-model inference."
        ),
    }
    write_json(output_path, data)

    errors: List[str] = []
    warnings: List[str] = []
    if validate:
        errors, warnings = validate_record(output_path, data)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "upstream_json": str(upstream_json),
        "input_mode": input_mode,
        "image_id": data.get("image_id"),
        "stage2c_decision": (data.get("stage2c_safety_check") or {}).get("decision"),
        "candidate_status": (data.get("final_assessment") or {}).get("fracture_candidate_status"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": ";".join(errors),
        "warnings": ";".join(warnings),
    }


def run_live_one(*_: Any, **__: Any) -> Dict[str, Any]:
    raise LiveStage12NotImplemented(
        "True live Stage 1/2 pixel-to-JSON inference is not implemented in v1. "
        "Use --mode precomputed for validated locked outputs, or implement the live adapters listed in the manifest."
    )


def write_summary(report_dir: Path, rows: List[Dict[str, Any]], manifest: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "live_stage12_artifact_manifest.json"
    write_json(manifest_path, manifest)

    csv_path = report_dir / "live_stage12_wrapper_v1_results.csv"
    fieldnames = [
        "input",
        "output",
        "upstream_json",
        "input_mode",
        "image_id",
        "stage2c_decision",
        "candidate_status",
        "error_count",
        "warning_count",
        "errors",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "wrapper": "live_stage12_wrapper_v1",
        "mode": args.mode,
        "input": str(args.input),
        "output": str(args.output),
        "locked_stage12_dir": str(args.locked_stage12_dir),
        "processed_count": len(rows),
        "total_errors": sum(int(row["error_count"]) for row in rows),
        "total_warnings": sum(int(row["warning_count"]) for row in rows),
        "status": "PASS" if all(int(row["error_count"]) == 0 for row in rows) else "FAIL",
        "results_csv": str(csv_path),
        "artifact_manifest": str(manifest_path),
        "precomputed_locked_stage12_mode": "implemented",
        "live_pixel_to_stage12_mode": "not_implemented_yet",
    }
    write_json(report_dir / "live_stage12_wrapper_v1_summary.json", summary)
    return summary


def write_human_report(report_dir: Path, summary: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    lock = manifest.get("locked_pipeline") or {}
    metrics = lock.get("official_metrics", {})
    lines = [
        "# Live Stage 1+2 Wrapper v1",
        "",
        "## Status",
        "",
        "- Precomputed locked Stage 1+2 mode: implemented and validated.",
        "- Stage 1 pixel-to-candidates adapter: implemented and smoke-tested.",
        "- Stage 2A live verifier adapter: implemented and smoke-tested, not locked yet.",
        "- True live pixel-to-Stage1+2 inference: not implemented yet.",
        "- This is intentional: the wrapper must not pretend that cached validation JSON is fresh model inference.",
        "",
        "## What This Step Adds",
        "",
        "This wrapper gives the future application a stable Stage 1+2 contract today. It can accept a locked JSON or an image name that exists in the locked validation set, copy the validated Stage 1+2 output, attach wrapper metadata, and validate the output schema.",
        "",
        "It also creates an artifact manifest so we know exactly which weights, reports, and policies are available before wiring true live inference.",
        "",
        "## Locked Pipeline Source Of Truth",
        "",
        f"- Pipeline version: `{lock.get('pipeline_version')}`",
        f"- Stage order: {len(lock.get('stage_order', []))} locked stages.",
        "- Stage 1: Run4@0.25 + Run5@0.15 with soft merge.",
        "- Stage 2A: locked verifier behavior as encoded in the validated JSON outputs.",
        "- Stage 2A.5: production strict artifact suppression.",
        "- Stage 2B: anatomy classifier, fine threshold 0.80 with parent fallback.",
        "- Stage 2C: clean-case safety warning only when no accepted candidate remains.",
        "",
        "## Metrics Kept From The Lock",
        "",
    ]
    frac = metrics.get("fracatlas_val", {})
    for name, values in frac.items():
        lines.append(
            f"- FracAtlas `{name}`: TP {values.get('tp')}, FP {values.get('fp')}, FN {values.get('fn')}, "
            f"P {values.get('precision')}, R {values.get('recall')}, F1 {values.get('f1')}."
        )
    graz = metrics.get("grazpedwri_clean_full", {})
    for name, values in graz.items():
        lines.append(
            f"- GRAZ `{name}`: TP {values.get('tp')}, FP {values.get('fp')}, FN {values.get('fn')}, "
            f"P {values.get('precision')}, R {values.get('recall')}, F1 {values.get('f1')}."
        )
    lines.extend(
        [
            "",
            "## Current Run",
            "",
            f"- Processed files: {summary.get('processed_count')}",
            f"- Contract status: {summary.get('status')}",
            f"- Validation errors: {summary.get('total_errors')}",
            f"- Validation warnings: {summary.get('total_warnings')}",
            f"- Results CSV: `{summary.get('results_csv')}`",
            f"- Artifact manifest: `{summary.get('artifact_manifest')}`",
            "",
            "## Critical Engineering Decision",
            "",
            "The Stage 1 and Stage 2A live adapters now exist. The next implementation step is validation, not another model: run Stage1->Stage2A on a representative FracAtlas subset, compare accepted/rejected candidates against locked outputs where possible, then wire Stage 2A.5 artifact suppression.",
        ]
    )
    report_path = report_dir / "LIVE_STAGE12_WRAPPER_V1_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1+2 wrapper for locked/precomputed and future live modes.")
    parser.add_argument("--input", required=True, help="Input locked JSON/image file or directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output JSON file or directory.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Report directory.")
    parser.add_argument("--locked-stage12-dir", default=str(DEFAULT_LOCKED_STAGE12_DIR))
    parser.add_argument("--mode", choices=["precomputed", "live"], default="precomputed")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    locked_stage12_dir = Path(args.locked_stage12_dir)
    report_dir = Path(args.report_dir)
    manifest = build_artifact_manifest()

    if args.mode == "live":
        write_json(report_dir / "live_stage12_artifact_manifest.json", manifest)
        raise LiveStage12NotImplemented(
            "Live mode is intentionally blocked in v1. The artifact manifest was written; implement adapters next."
        )

    inputs = find_inputs(input_root)
    rows = []
    single_output = len(inputs) == 1
    for input_path in inputs:
        output_path = output_path_for_input(input_path, output_root, single_output)
        rows.append(
            run_precomputed_one(
                input_path=input_path,
                output_path=output_path,
                locked_stage12_dir=locked_stage12_dir,
                validate=not args.no_validate,
            )
        )
    summary = write_summary(report_dir, rows, manifest, args)
    report_path = write_human_report(report_dir, summary, manifest)
    summary["human_report"] = str(report_path)
    write_json(report_dir / "live_stage12_wrapper_v1_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
