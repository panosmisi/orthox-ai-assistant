# OrthoX

OrthoX is the source-code package of the diploma thesis application for AI-assisted X-ray review support in orthopedics.

The repository contains only the core implementation files that represent the final system. Datasets, trained model weights, experiment runs, temporary files, and old development artifacts are intentionally excluded.

## Core Structure

- `web_app/`
  - FastAPI application, HTML templates, static assets, upload flow, review flow, result presentation, and problem description mode.

- `live_full_pipeline_adapter.py`
  - Connects the live application flow with the staged analysis pipeline.

- `live_stage1_detector.py`
  - Stage 1 fracture-candidate detection.

- `live_stage2a_verifier.py`
  - Stage 2A verification logic.

- `live_stage2b_anatomy.py`
  - Stage 2B anatomical support logic.

- `live_stage2c_safety.py`
  - Stage 2C safety and artifact-suppression logic.

- `stage3/`
  - RAG evidence handling, clinical-context construction, explainability text, safety guardrails, and final report/display logic.

- `stage2c_gradcam_explainability.py`
  - Grad-CAM explainability support.

- `models/` and `utils/`
  - Supporting YOLO runtime modules required by the detector code.

## Excluded Material

The following are not included in this clean repository package:

- datasets,
- trained model weights and checkpoints,
- experiment runs,
- generated outputs,
- local virtual environments,
- logs,
- old drafts and failed attempts.

This keeps the repository small, reviewable, and appropriate for academic evaluation.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the web application:

```powershell
python -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000
```

Full live inference requires the trained model weights and datasets to be placed locally. These large files are intentionally excluded from the public GitHub-ready folder.

Optional environment variables:

- `NCBI_API_KEY`
  - Optional PubMed/NCBI API key used only for higher request limits.

- `PROBLEM_DESCRIPTION_LLM_ENABLED`
  - Set to `1` only when local VLM/LLM inference is available.
