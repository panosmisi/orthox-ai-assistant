# OrthoX Web App

This folder contains the FastAPI web interface of OrthoX.

## Main File

- `main.py`
  - Defines the application routes.
  - Handles X-ray upload and preflight checks.
  - Connects the user interface with the live analysis pipeline.
  - Presents the result, visual evidence, clinical context, explainability, evidence, and problem description flow.

## Assets

- `templates/`
  - Jinja2 templates for the user interface.

- `static/`
  - CSS, JavaScript, and image assets used by the interface.

## Run

From the repository root:

```powershell
python -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000
```

Model weights and datasets are not included in this repository. They must be placed locally before full inference can run.
