from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from stage3.runtime import load_config, run_stage3_on_json
from validate_stage3_contract import validate_record


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCKED_STAGE12_DIR = (
    REPO_ROOT / "runs" / "analysis" / "stage1b_feasibility_v1" / "stage2c_full_val_enabled"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "final_pipeline_runner_v1" / "outputs"
DEFAULT_REPORT_DIR = REPO_ROOT / "runs" / "analysis" / "final_pipeline_runner_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
        locked_stage12_dir / image_path.name.replace(image_path.suffix, ".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No locked Stage 1/2 JSON found for image {image_path}. "
        f"Expected one of: {', '.join(str(c) for c in candidates)}"
    )


def resolve_upstream_json(input_path: Path, locked_stage12_dir: Path) -> Tuple[Path, str]:
    if input_path.suffix.lower() == ".json":
        return input_path, "locked_json_input"
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        return locked_json_for_image(input_path, locked_stage12_dir), "image_lookup_precomputed_stage12"
    raise ValueError(f"Unsupported input type: {input_path}")


def build_stage3_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    evidence = config.setdefault("evidence", {})
    if args.enable_pubmed:
        evidence["pubmed_enabled"] = True
    if args.offline_pubmed:
        evidence["pubmed_offline_mode"] = True
    if args.enable_pubmed_abstract_claims:
        evidence["pubmed_abstract_claims_enabled"] = True
    if args.disable_guidelines:
        evidence["guideline_registry_enabled"] = False
    stage3 = config.setdefault("stage3", {})
    if args.disable_patient_note:
        stage3["patient_friendly_note_enabled"] = False
    return config


def output_path_for_input(input_path: Path, upstream_json_path: Path, output_root: Path, single_output: bool) -> Path:
    if single_output and output_root.suffix.lower() == ".json":
        return output_root
    if input_path.suffix.lower() == ".json":
        return output_root / input_path.name
    return output_root / f"{input_path.stem}.json"


def run_one(
    input_path: Path,
    output_path: Path,
    locked_stage12_dir: Path,
    config: Dict[str, Any],
    validate: bool = True,
) -> Dict[str, Any]:
    upstream_json_path, input_mode = resolve_upstream_json(input_path, locked_stage12_dir)
    upstream = load_json(upstream_json_path)
    final_output = run_stage3_on_json(upstream, config)
    final_output.setdefault("stage3_orthopedic_rag", {})
    final_output["stage3_orthopedic_rag"]["runner_metadata"] = {
        "runner": "final_pipeline_runner_v1",
        "input_path": str(input_path),
        "input_mode": input_mode,
        "locked_stage12_json": str(upstream_json_path),
        "stage3_added": True,
        "live_stage1_stage2_inference": False,
        "notes": (
            "This runner uses locked/precomputed Stage 1+2 JSON as upstream input. "
            "Live YOLO/verifier/anatomy inference is intentionally not implemented in v1."
        ),
    }
    write_json(output_path, final_output)

    errors: List[str] = []
    warnings: List[str] = []
    if validate:
        errors, warnings = validate_record(output_path, final_output, locked_stage12_dir)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "upstream_json": str(upstream_json_path),
        "input_mode": input_mode,
        "image_id": final_output.get("image_id"),
        "stage3_critic": final_output.get("stage3_orthopedic_rag", {}).get("critic", {}).get("verdict"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": ";".join(errors),
        "warnings": ";".join(warnings),
    }


def write_summary(
    report_dir: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "final_pipeline_runner_v1_results.csv"
    fieldnames = [
        "input",
        "output",
        "upstream_json",
        "input_mode",
        "image_id",
        "stage3_critic",
        "error_count",
        "warning_count",
        "errors",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evidence_cfg = config.get("evidence", {})
    summary = {
        "runner": "final_pipeline_runner_v1",
        "input": str(args.input),
        "output": str(args.output),
        "locked_stage12_dir": str(args.locked_stage12_dir),
        "processed_count": len(rows),
        "total_errors": sum(int(row["error_count"]) for row in rows),
        "total_warnings": sum(int(row["warning_count"]) for row in rows),
        "status": "PASS" if all(int(row["error_count"]) == 0 for row in rows) else "FAIL",
        "results_csv": str(csv_path),
        "config": str(args.config),
        "pubmed_enabled": bool(evidence_cfg.get("pubmed_enabled", False)),
        "pubmed_abstract_claims_enabled": bool(evidence_cfg.get("pubmed_abstract_claims_enabled", False)),
        "offline_pubmed": bool(evidence_cfg.get("pubmed_offline_mode", False)),
        "guideline_registry_enabled": bool(evidence_cfg.get("guideline_registry_enabled", True)),
        "source_relevance_filter_enabled": bool(evidence_cfg.get("source_relevance_filter_enabled", True)),
        "live_stage1_stage2_inference": False,
    }
    summary_path = report_dir / "final_pipeline_runner_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final integrated pipeline runner v1. "
            "Uses locked/precomputed Stage 1+2 JSON, appends Stage 3, and validates the final contract."
        )
    )
    parser.add_argument("--input", required=True, help="Input locked JSON/image file or directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output JSON file or directory.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Report directory.")
    parser.add_argument("--locked-stage12-dir", default=str(DEFAULT_LOCKED_STAGE12_DIR))
    parser.add_argument("--config", default=str(REPO_ROOT / "stage3" / "stage3_config.yaml"))
    parser.add_argument("--no-validate", action="store_true", help="Skip final Stage 3 contract validation.")
    parser.add_argument("--enable-pubmed", action="store_true", help="Enable live/cached PubMed retrieval.")
    parser.add_argument("--offline-pubmed", action="store_true", help="Use PubMed cache only.")
    parser.add_argument("--enable-pubmed-abstract-claims", action="store_true")
    parser.add_argument("--disable-guidelines", action="store_true")
    parser.add_argument("--disable-patient-note", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    locked_stage12_dir = Path(args.locked_stage12_dir)
    config = build_stage3_config(args)

    inputs = find_inputs(input_root)
    single_output = len(inputs) == 1
    rows = []
    for input_path in inputs:
        dst = output_path_for_input(input_path, Path(""), output_root, single_output)
        rows.append(
            run_one(
                input_path=input_path,
                output_path=dst,
                locked_stage12_dir=locked_stage12_dir,
                config=config,
                validate=not args.no_validate,
            )
        )

    summary = write_summary(Path(args.report_dir), rows, args, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
