from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "web_app"
CURATED_MANIFEST = REPO_ROOT / "docs" / "app_design" / "curated_demo_cases_v1_manifest.json"
GOLDEN_CONTRACT_DIR = REPO_ROOT / "runs" / "analysis" / "app_prep_pack_v1_golden" / "app_contract_json"
GOLDEN_MARKDOWN_DIR = REPO_ROOT / "runs" / "analysis" / "app_prep_pack_v1_golden" / "markdown_reports"
UPLOAD_PREVIEW_DIR = REPO_ROOT / "runs" / "analysis" / "web_app_uploads_preflight"
UPLOAD_LIVE_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "web_app_live_upload_analysis"
PROBLEM_LLM_JOB_DIR = REPO_ROOT / "tmp" / "problem_llm_jobs"
LIVE_ANALYSIS_DEVICE = os.environ.get("XRAY_REVIEW_LIVE_DEVICE", "0")
PROBLEM_LLM_ENABLED = os.environ.get("PROBLEM_DESCRIPTION_LLM_ENABLED", "0").lower() not in {"0", "false", "no"}
PROBLEM_LLM_MODEL_ID = os.environ.get("PROBLEM_DESCRIPTION_LLM_MODEL", "google/medgemma-4b-it")
PROBLEM_LLM_MAX_NEW_TOKENS = int(os.environ.get("PROBLEM_DESCRIPTION_LLM_MAX_NEW_TOKENS", "260"))
PROBLEM_LLM_MAX_SECONDS = float(os.environ.get("PROBLEM_DESCRIPTION_LLM_MAX_SECONDS", "360"))
PROBLEM_LLM_ISOLATED = os.environ.get("PROBLEM_DESCRIPTION_LLM_ISOLATED", "1").lower() not in {"0", "false", "no"}

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MIN_SHORT_SIDE = 360
RECOMMENDED_SHORT_SIDE = 768
MAX_LONG_SIDE = 8000
MAX_ASPECT_RATIO = 5.0
WARN_ASPECT_RATIO = 3.0
MAX_MEAN_SATURATION = 45.0
WARN_MEAN_SATURATION = 25.0
MAX_COLORFULNESS = 30.0
WARN_COLORFULNESS = 18.0
VERY_DARK_MEAN_BRIGHTNESS = 15.0
VERY_BRIGHT_MEAN_BRIGHTNESS = 245.0
MIN_VISIBLE_CONTRAST_STD = 8.0
MIN_VISIBLE_DYNAMIC_RANGE = 35.0
LIMITED_QUALITY_CONTRAST_STD = 18.0
LIMITED_QUALITY_DYNAMIC_RANGE = 60.0
LIMITED_QUALITY_SHARPNESS = 20.0


app = FastAPI(title="AI-assisted X-ray review support", version="0.1.0")
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=APP_ROOT / "templates")


def display_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def curated_manifest() -> Dict[str, Any]:
    if not CURATED_MANIFEST.exists():
        return {"cases": []}
    return read_json(CURATED_MANIFEST)


def curated_cases() -> List[Dict[str, Any]]:
    cases = curated_manifest().get("cases") or []
    return sorted(cases, key=lambda item: item.get("priority", 999))


def case_by_demo_id(demo_id: str) -> Dict[str, Any]:
    for case in curated_cases():
        if case["demo_id"].lower() == demo_id.lower() or case["image_id"].lower() == demo_id.lower():
            return case
    raise HTTPException(status_code=404, detail="Demo case not found")


def load_contract_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return read_json(REPO_ROOT / case["app_contract_json"])


def safe_image_path(contract: Dict[str, Any]) -> Path:
    image_path = Path((contract.get("case") or {}).get("image_path") or "")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return image_path


def safe_uploaded_preview_path(preview_id: str) -> Path:
    if not preview_id.isalnum() or len(preview_id) > 64:
        raise HTTPException(status_code=404, detail="Preview not found")
    path = UPLOAD_PREVIEW_DIR / f"{preview_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return path


def uploaded_preflight_metadata_path(preview_id: str) -> Path:
    if not preview_id.isalnum() or len(preview_id) > 64:
        raise HTTPException(status_code=404, detail="Preview not found")
    return UPLOAD_PREVIEW_DIR / f"{preview_id}.preflight.json"


def read_uploaded_preflight_metadata(preview_id: str) -> Dict[str, Any]:
    path = uploaded_preflight_metadata_path(preview_id)
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def image_dimensions(image_path: Path) -> Dict[str, int]:
    with Image.open(image_path) as img:
        return {"width": img.width, "height": img.height}


def bbox_percent(contract: Dict[str, Any], dimensions: Dict[str, int]) -> Dict[str, float] | None:
    bbox = ((contract.get("visual") or {}).get("primary_bbox") or {})
    xyxy = bbox.get("xyxy_pixels")
    if not bbox.get("available") or not xyxy or not dimensions["width"] or not dimensions["height"]:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "left": 100.0 * x1 / dimensions["width"],
        "top": 100.0 * y1 / dimensions["height"],
        "width": 100.0 * width / dimensions["width"],
        "height": 100.0 * height / dimensions["height"],
    }


def candidate_bbox_percent(candidate: Dict[str, Any], dimensions: Dict[str, int], index: int) -> Dict[str, Any] | None:
    bbox = candidate.get("bbox") or {}
    xyxy = bbox.get("xyxy_pixels")
    if not bbox.get("available") or not xyxy or not dimensions["width"] or not dimensions["height"]:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
    except (TypeError, ValueError):
        return None
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0 or height <= 0:
        return None
    return {
        "candidate_id": candidate.get("candidate_id") or f"C{index}",
        "index": index,
        "left": 100.0 * x1 / dimensions["width"],
        "top": 100.0 * y1 / dimensions["height"],
        "width": 100.0 * width / dimensions["width"],
        "height": 100.0 * height / dimensions["height"],
    }


def candidate_bboxes_percent(candidates: List[Dict[str, Any]], dimensions: Dict[str, int]) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        box = candidate_bbox_percent(candidate, dimensions, index)
        if box:
            boxes.append(box)
    return boxes


def status_class(state_id: str) -> str:
    return {
        "candidate_retained": "status-candidate",
        "image_level_warning_without_bbox": "status-warning",
        "no_high_confidence_candidate_retained": "status-limited",
    }.get(state_id, "status-neutral")


def list_value(payload: Dict[str, Any], key: str, limit: int | None = None) -> List[Any]:
    values = payload.get(key)
    if not isinstance(values, list):
        return []
    return values[:limit] if limit else values


def upload_gate_requirements() -> List[str]:
    return [
        "Exported PNG or JPEG radiograph image; native DICOM studies are not supported in this V1 demo.",
        f"File size up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        f"Minimum short side {MIN_SHORT_SIDE}px; {RECOMMENDED_SHORT_SIDE}px or higher is preferred.",
        "Readable radiograph with adequate brightness, contrast, and sharpness.",
        "Radiograph-like grayscale appearance; ordinary color photographs are rejected.",
        "Limited-quality but radiograph-like images may continue with a strong caution instead of being treated as normal inputs.",
        "No extreme cropping, blank screenshots, or heavy compression; sideways radiographs may be orientation-checked, but upright exports are preferred.",
        "The tool must not treat a failed preflight image as analyzable.",
    ]


def upload_result(
    status: str,
    headline: str,
    message: str,
    metrics: Dict[str, Any] | None = None,
    blocking: List[str] | None = None,
    warnings: List[str] | None = None,
    preview_id: str | None = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "headline": headline,
        "message": message,
        "metrics": metrics or {},
        "blocking": blocking or [],
        "warnings": warnings or [],
        "preview_id": preview_id,
        "requirements": upload_gate_requirements(),
    }


def normalize_limited_quality_radiograph(rgb: Image.Image) -> Image.Image:
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    low, high = np.percentile(enhanced, [0.5, 99.5])
    if high > low:
        enhanced = np.clip((enhanced.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced).convert("RGB")


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


NEGATION_CUES = [
    "no",
    "not",
    "without",
    "deny",
    "denies",
    "denied",
    "do not",
    "does not",
    "don't",
    "doesn't",
    "never",
    "negative for",
    "absence of",
    "free of",
]


def keyword_is_negated(text: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    pattern = keyword_pattern(keyword)
    for match in pattern.finditer(text):
        if keyword_match_is_negated(text, match.start(), match.end()):
            return True
    return False


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip().lower())
    if re.fullmatch(r"[a-z0-9][a-z0-9\s\-/]*[a-z0-9]", keyword.strip().lower()) or re.fullmatch(r"[a-z0-9]", keyword.strip().lower()):
        return re.compile(r"(?<![a-z0-9_])" + escaped + r"(?![a-z0-9_])")
    return re.compile(escaped)


def keyword_match_is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 160) : start]
    after = text[end : min(len(text), end + 90)]
    before_same_clause = re.split(r"[.;:\n]", before)[-1].strip()
    after_same_clause = re.split(r"[.;:\n]", after)[0].strip()
    for cue in NEGATION_CUES:
        if re.search(r"\b" + re.escape(cue) + r"\b", before_same_clause):
            return True
    reverse_negation_patterns = [
        r"\b(?:do not|does not|don't|doesn't|did not|didn't|is not|isn't|are not|aren't)\b.{0,45}\b(?:hurt|ache|pain|painful|sore|tender|swollen|injured|affected|involved)\b",
    ]
    return any(re.search(pattern, after_same_clause) for pattern in reverse_negation_patterns)


def keyword_has_positive_match(text: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    pattern = keyword_pattern(keyword)
    return any(not keyword_match_is_negated(text, match.start(), match.end()) for match in pattern.finditer(text))


def keyword_has_negated_match(text: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    pattern = keyword_pattern(keyword)
    return any(keyword_match_is_negated(text, match.start(), match.end()) for match in pattern.finditer(text))


def contains_positive_any(text: str, keywords: List[str]) -> bool:
    for keyword in keywords:
        if keyword_has_positive_match(text, keyword):
            return True
    return False


def positive_keyword_matches(text: str, keywords: List[str]) -> List[str]:
    matches = []
    for keyword in keywords:
        if keyword_has_positive_match(text, keyword):
            matches.append(keyword)
    return matches[:4]


ANATOMY_KEYWORDS = {
    "wrist": ["wrist", "carpus", "καρπ", "πηχεοκαρπ"],
    "hand": ["hand", "palm", "metacarp", "χερι", "χέρι", "παλαμ"],
    "finger": ["finger", "thumb", "digit", "δαχτυλ", "αντιχειρ"],
    "forearm": ["forearm", "radius", "ulna", "αντιβραχ", "πήχ", "πηχ"],
    "elbow": ["elbow", "αγκων", "αγκών"],
    "shoulder": ["shoulder", "clavicle", "collarbone", "ωμ", "κλειδ"],
    "ankle": ["ankle", "malleolus", "αστραγ", "σφυρ"],
    "foot": ["foot", "toe", "metatars", "ποδι", "πόδι", "πελμ"],
    "knee": ["knee", "patella", "γονατ", "γόνατ", "επιγονατ"],
    "hip": ["hip", "pelvis", "ισχι", "ισχί", "λεκάν", "λεκαν"],
    "leg": ["leg", "tibia", "fibula", "shin", "lower leg", "καλαμ", "κνημ", "περόνη"],
    "ribs": ["rib", "chest wall", "πλευρ", "θωρακ"],
    "back": ["back", "spine", "lumbar", "neck", "πλατη", "πλάτη", "σπονδυλ", "αυχεν"],
}

MECHANISM_KEYWORDS = {
    "fall": ["fall", "fell", "slipped", "trip", "πτώση", "πεσα", "έπεσα", "επέσα"],
    "sports injury": ["basketball", "football", "soccer", "running", "gym", "sport", "αθλη", "μπασκετ", "ποδοσφ"],
    "twisting injury": ["twist", "twisted", "twisting", "rolled", "rolling", "sprain", "sprained", "στραμπ", "γυρισ", "στρίψ"],
    "direct blow": ["hit", "blow", "impact", "collision", "χτύπ", "χτυπ", "τρακαρ"],
    "high-energy trauma": ["car", "motorbike", "traffic", "crash", "height", "τροχαι", "μηχαν", "αυτοκιν"],
    "crush injury": ["crush", "trapped", "caught", "συνθλι", "πλακω"],
}

MECHANISM_KEYWORDS["direct blow"].extend(["bump", "bumped", "knock", "knocked", "struck", "against a"])

SYMPTOM_KEYWORDS = {
    "pain": ["pain", "ache", "hurts", "πονο", "πόνο", "πονά", "πονα"],
    "swelling": ["swelling", "swollen", "edema", "πρηξ", "πρησ"],
    "bruising": ["bruise", "bruising", "black and blue", "μελαν", "μωλωπ"],
    "reduced movement": ["cannot move", "limited movement", "stiff", "range of motion", "δεν μπορώ να κουν", "δυσκαμψ"],
    "unable to use limb": [
        "cannot grip",
        "cannot walk",
        "can't walk",
        "cannot bear",
        "can't bear",
        "bear weight",
        "cannot put weight",
        "can't put weight",
        "put weight on",
        "weight on the foot",
        "weight on my foot",
        "δεν μπορώ να πατή",
        "δεν μπορω να πατη",
        "δεν μπορώ να πιά",
        "δεν μπορω να πια",
    ],
    "numbness": ["numb", "tingling", "pins", "μουδια", "μυρμηγκ"],
    "weakness": ["weak", "weakness", "αδυναμ"],
    "visible deformity": ["deformity", "deformed", "crooked", "bent", "out of place", "παραμορφ", "στραβ"],
    "open wound": ["open wound", "bleeding", "bone visible", "cut", "αιμορ", "ανοιχτ", "πληγ"],
    "fever/systemic symptoms": ["fever", "chills", "infection", "πυρετ", "ρίγ", "ριγ"],
}

TIME_KEYWORDS = ["today", "yesterday", "hour", "hours", "day", "days", "week", "weeks", "σημερα", "σήμερα", "χθες", "ωρα", "ώρα", "μερα", "μέρα"]

SYMPTOM_KEYWORDS["pain"].extend(["sore", "soreness", "tender", "tenderness"])
SYMPTOM_KEYWORDS["numbness"].extend(["color change", "coldness", "reduced sensation"])

SENSORY_RED_FLAG_KEYWORDS = ["numb", "tingling", "pins", "reduced sensation", "μουδια", "μυρμηγκ"]
VASCULAR_RED_FLAG_KEYWORDS = ["color change", "coldness", "cold", "blue", "pale", "no pulse", "κρυο", "μπλε", "χλωμ"]

RED_FLAG_RULES = [
    {"label": "Open injury concern", "keywords": SYMPTOM_KEYWORDS["open wound"], "message": "Open wound, bleeding, or visible bone language was mentioned."},
    {"label": "Altered sensation language", "keywords": SENSORY_RED_FLAG_KEYWORDS, "message": "Sensory language was mentioned. This is a safety trigger for human review, but it does not confirm circulation compromise from text alone."},
    {"label": "Circulation concern", "keywords": VASCULAR_RED_FLAG_KEYWORDS, "message": "Color, coldness, or pulse-related language was mentioned."},
    {"label": "Deformity / dislocation concern", "keywords": SYMPTOM_KEYWORDS["visible deformity"], "message": "Visible deformity or abnormal alignment language was mentioned."},
    {"label": "High-energy mechanism", "keywords": MECHANISM_KEYWORDS["high-energy trauma"] + MECHANISM_KEYWORDS["crush injury"], "message": "The described mechanism may be higher-energy than a simple minor injury."},
    {"label": "Systemic symptom concern", "keywords": SYMPTOM_KEYWORDS["fever/systemic symptoms"], "message": "Fever, chills, or infection-related language was mentioned."},
]

IRRELEVANT_HINTS = [
    "weather",
    "recipe",
    "stock",
    "football score",
    "movie",
    "politics",
    "homework",
    "essay",
    "joke",
    "code",
    "laptop",
    "keyboard",
    "screen",
    "computer",
    "phone",
    "printer",
    "wifi",
    "software",
]


def matched_labels(text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
    return [label for label, keywords in keyword_map.items() if contains_positive_any(text, keywords)]


def keyword_contexts(text: str, keyword: str, radius: int = 52) -> List[str]:
    contexts = []
    pattern = keyword_pattern(keyword)
    for match in pattern.finditer(text):
        contexts.append(text[max(0, match.start() - radius) : min(len(text), match.end() + radius)])
    return contexts


def keyword_source_phrase(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind("\n", 0, start))
    right_candidates = [idx for idx in [text.find(".", end), text.find(";", end), text.find("\n", end)] if idx != -1]
    right = min(right_candidates) if right_candidates else min(len(text), end + 100)
    phrase = text[left + 1 : right].strip()
    return phrase[:180]


def keyword_source_spans(text: str, concept: str, keywords: List[str], status: str, limit: int = 2) -> List[Dict[str, Any]]:
    spans = []
    for keyword in keywords:
        pattern = keyword_pattern(keyword)
        for match in pattern.finditer(text):
            negated = keyword_match_is_negated(text, match.start(), match.end())
            if (status == "denied" and not negated) or (status == "present" and negated):
                continue
            spans.append(
                {
                    "concept": concept,
                    "status": status,
                    "matched_keyword": keyword,
                    "source_text": keyword_source_phrase(text, match.start(), match.end()),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            if len(spans) >= limit:
                return spans
    return spans


def build_extraction_evidence(
    text: str,
    anatomy_context: Dict[str, Any],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    evidence = []
    for concept in anatomy_context.get("present") or []:
        evidence.extend(keyword_source_spans(text, f"anatomy:{concept}", ANATOMY_KEYWORDS.get(concept, []), "present", limit=1))
    for concept in anatomy_context.get("adjacent_or_functional") or []:
        spans = keyword_source_spans(text, f"anatomy:{concept}", ANATOMY_KEYWORDS.get(concept, []), "present", limit=1)
        for item in spans:
            item["status"] = "context_only"
        evidence.extend(spans)
    for concept in anatomy_context.get("denied") or []:
        evidence.extend(keyword_source_spans(text, f"anatomy:{concept}", ANATOMY_KEYWORDS.get(concept, []), "denied", limit=1))
    for concept in mechanisms:
        evidence.extend(keyword_source_spans(text, f"mechanism:{concept}", MECHANISM_KEYWORDS.get(concept, []), "present", limit=1))
    for concept in symptoms:
        evidence.extend(keyword_source_spans(text, f"symptom:{concept}", SYMPTOM_KEYWORDS.get(concept, []), "present", limit=1))
    for concept in denied_symptoms:
        evidence.extend(keyword_source_spans(text, f"symptom:{concept}", SYMPTOM_KEYWORDS.get(concept, []), "denied", limit=1))
    for flag in red_flags:
        triggers = flag.get("triggers") or []
        if triggers:
            evidence.append(
                {
                    "concept": f"red_flag:{flag.get('label')}",
                    "status": "present",
                    "matched_keyword": triggers[0],
                    "source_text": flag.get("why_triggered", ""),
                }
            )
    return evidence[:18]


def anatomy_context_is_functional_or_adjacent(text: str, keyword: str) -> bool:
    contexts = keyword_contexts(text, keyword)
    if not contexts:
        return False
    has_injury_context = False
    has_functional_context = False
    injury_patterns = [
        r"\bpain\b",
        r"\bpainful\b",
        r"\bhurts?\b",
        r"\bsore\b",
        r"\bsoreness\b",
        r"\bswollen\b",
        r"\bswelling\b",
        r"\btender",
        r"\blocali[sz]ed\b",
        r"\bpress\b",
        r"\bmiddle\b",
        r"\bmid\b",
        r"\baround\b",
        r"\bhit\b",
        r"\bimpact\b",
        r"\bblow\b",
        r"\bstruck\b",
    ]
    functional_patterns = [
        r"\bcan\s+move\b",
        r"\bmove\s+my\b",
        r"\bmove\s+the\b",
        r"\bnormally\b",
        r"\bnormal\b",
        r"\bnot\s+the\b",
        r"\bnot\s+my\b",
        r"\bno\s+pain\s+(?:in|at|around)\b",
    ]
    for context in contexts:
        if re.search(r"\bcan\s+move\s+(?:my|the)\b", context):
            has_functional_context = True
            continue
        if re.search(r"\bcan\s+move\b", context) and re.search(r"\bnormally\b", context):
            has_functional_context = True
            continue
        if keyword == "thumb" and re.search(r"\bthumb\s+side\s+(?:of\s+)?(?:the\s+)?wrist\b", context):
            has_functional_context = True
            continue
        if keyword == "hand" and re.search(r"\boutstretched\s+hand\b", context):
            has_functional_context = True
            continue
        if any(re.search(pattern, context) for pattern in injury_patterns):
            has_injury_context = True
            continue
        if any(re.search(pattern, context) for pattern in functional_patterns):
            has_functional_context = True
    return has_functional_context and not has_injury_context


def anatomy_primary_score(text: str, keywords: List[str]) -> int:
    score = 0
    injury_focus = [
        r"\bpain\b",
        r"\bpainful\b",
        r"\bhurts?\b",
        r"\bsore\b",
        r"\bswollen\b",
        r"\bswelling\b",
        r"\btender",
        r"\bpress\b",
        r"\bmiddle\b",
        r"\baround\b",
        r"\bmostly\b",
    ]
    mechanism_focus = [r"\bhit\b", r"\bimpact\b", r"\bblow\b", r"\btwist", r"\brolled\b", r"\bfell\b", r"\bfall\b"]
    downweight = [r"\boutstretched\b", r"\bcan\s+move\b", r"\bnormally\b", r"\bnot\s+the\b", r"\bnot\s+my\b"]
    for keyword in keywords:
        if not keyword_pattern(keyword).search(text):
            continue
        score += 1
        for context in keyword_contexts(text, keyword, radius=58):
            if any(re.search(pattern, context) for pattern in injury_focus):
                score += 4
            if any(re.search(pattern, context) for pattern in mechanism_focus):
                score += 1
            if any(re.search(pattern, context) for pattern in downweight):
                score -= 3
    return score


def matched_labels_with_negation(text: str, keyword_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    present = []
    denied = []
    for label, keywords in keyword_map.items():
        has_positive = contains_positive_any(text, keywords)
        has_denied = any(keyword_has_negated_match(text, keyword) for keyword in keywords)
        if has_positive:
            present.append(label)
        elif has_denied:
            denied.append(label)
    return {"present": present, "denied": denied}


def classify_anatomy_context(text: str, keyword_map: Dict[str, List[str]]) -> Dict[str, List[str] | str | None]:
    present = []
    adjacent_or_functional = []
    denied = []
    scores = {}
    for label, keywords in keyword_map.items():
        positive_keywords = [keyword for keyword in keywords if keyword_has_positive_match(text, keyword)]
        denied_keywords = [keyword for keyword in keywords if keyword_has_negated_match(text, keyword)]
        if positive_keywords:
            if all(anatomy_context_is_functional_or_adjacent(text, keyword) for keyword in positive_keywords):
                adjacent_or_functional.append(label)
            else:
                present.append(label)
                scores[label] = anatomy_primary_score(text, positive_keywords)
        elif denied_keywords:
            denied.append(label)
    primary = max(present, key=lambda label: scores.get(label, 0)) if present else (adjacent_or_functional[0] if adjacent_or_functional else None)
    if primary in present:
        present = [primary] + [label for label in present if label != primary]
    return {
        "primary": primary,
        "present": present,
        "adjacent_or_functional": adjacent_or_functional,
        "denied": denied,
    }


def matched_symptom_labels(text: str) -> Dict[str, List[str]]:
    present = []
    denied = []
    for label, keywords in SYMPTOM_KEYWORDS.items():
        positive = contains_positive_any(text, keywords)
        negated = any(keyword_has_negated_match(text, keyword) for keyword in keywords)
        if positive:
            present.append(label)
        elif negated:
            denied.append(label)
    return {"present": present, "denied": denied}


GENERAL_AREA_TERMS = [
    "arm",
    "upper limb",
    "leg",
    "lower limb",
    "hand",
    "foot",
    "chest",
    "back",
    "neck",
]

SPECIFIC_ANATOMY_LABELS = {
    "wrist",
    "finger",
    "forearm",
    "elbow",
    "shoulder",
    "ankle",
    "knee",
    "hip",
    "ribs",
    "back",
}

FUNCTION_STATUS_TERMS = [
    "cannot grip",
    "can grip",
    "cannot walk",
    "can walk",
    "can't walk",
    "cannot bear",
    "can't bear",
    "bear weight",
    "cannot put weight",
    "can't put weight",
    "put weight on",
    "weight on the foot",
    "weight on my foot",
    "move my",
    "can move",
    "cannot move",
    "range of motion",
    "use the limb",
    "use my",
]


def build_input_quality(
    text: str,
    anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    has_time: bool,
) -> Dict[str, Any]:
    checks = [
        {
            "id": "general_area",
            "label": "General body area",
            "met": bool(anatomy or contains_any(text, GENERAL_AREA_TERMS)),
            "hint": "Mention the broad area, for example arm, leg, hand, foot, chest, or back.",
        },
        {
            "id": "specific_location",
            "label": "Specific location",
            "met": bool(set(anatomy) & SPECIFIC_ANATOMY_LABELS),
            "hint": "Add a more exact site, for example wrist, ankle, elbow, knee, shoulder, or ribs.",
        },
        {
            "id": "mechanism",
            "label": "How it happened",
            "met": bool(mechanisms),
            "hint": "Describe the mechanism, for example fall, twist, impact, sport, traffic injury, or crush.",
        },
        {
            "id": "timing",
            "label": "When it happened",
            "met": has_time,
            "hint": "State when symptoms started or when the injury happened.",
        },
        {
            "id": "pain_swelling_symptoms",
            "label": "Main symptoms",
            "met": bool(set(symptoms) & {"pain", "swelling", "bruising", "reduced movement", "weakness"}),
            "hint": "Mention pain, swelling, bruising, movement limitation, or weakness if present.",
        },
        {
            "id": "function_status",
            "label": "Current function",
            "met": "unable to use limb" in symptoms or contains_any(text, FUNCTION_STATUS_TERMS),
            "hint": "Say whether the patient can grip, walk, bear weight, move the joint, or use the limb.",
        },
        {
            "id": "neurovascular_status",
            "label": "Sensation / circulation status",
            "met": "numbness" in symptoms or "numbness" in denied_symptoms or contains_positive_any(text, ["cold", "blue", "pale", "no pulse"]),
            "hint": "State whether there is numbness, tingling, color change, coldness, or reduced sensation.",
        },
        {
            "id": "deformity_wound_status",
            "label": "Deformity / wound status",
            "met": bool(set(symptoms + denied_symptoms) & {"visible deformity", "open wound"}),
            "hint": "State whether there is visible deformity, abnormal alignment, bleeding, cut, or open wound.",
        },
    ]
    met = [item for item in checks if item["met"]]
    missing = [item for item in checks if not item["met"]]
    total = len(checks)
    score = len(met)
    percent = round(score / total * 100)
    if percent >= 75:
        level = "Good intake detail"
        message = "The description has enough detail for a useful cautious review."
    elif percent >= 50:
        level = "Partial intake detail"
        message = "The description is usable, but important clinical context is still missing."
    else:
        level = "Limited intake detail"
        message = "The description is too limited; add the missing details before using the output."
    return {
        "score": score,
        "total": total,
        "percent": percent,
        "level": level,
        "message": message,
        "met": [{"label": item["label"]} for item in met],
        "missing": [{"label": item["label"], "hint": item["hint"]} for item in missing],
        "checks": checks,
    }


def extraction_confidence(
    anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    has_time: bool,
) -> List[Dict[str, str]]:
    items = []
    items.append(
        {
            "field": "Anatomical region",
            "confidence": "High" if anatomy and set(anatomy) & SPECIFIC_ANATOMY_LABELS else ("Medium" if anatomy else "Low"),
            "basis": "Specific anatomy term detected." if anatomy and set(anatomy) & SPECIFIC_ANATOMY_LABELS else ("Only broad anatomy language detected." if anatomy else "No reliable anatomy term detected."),
        }
    )
    items.append(
        {
            "field": "Mechanism",
            "confidence": "High" if mechanisms else "Low",
            "basis": "Mechanism keyword detected." if mechanisms else "Mechanism not described clearly.",
        }
    )
    items.append(
        {
            "field": "Symptoms / function",
            "confidence": "High" if len(symptoms) >= 2 else ("Medium" if symptoms else "Low"),
            "basis": "Multiple symptom/function signals detected." if len(symptoms) >= 2 else ("One symptom/function signal detected." if symptoms else "Symptoms are not described clearly."),
        }
    )
    items.append(
        {
            "field": "Denied red-flag symptoms",
            "confidence": "High" if denied_symptoms else "Low",
            "basis": "Negated symptom language detected." if denied_symptoms else "No explicit denied red-flag symptoms detected.",
        }
    )
    items.append(
        {
            "field": "Timing",
            "confidence": "Medium" if has_time else "Low",
            "basis": "Timing language detected." if has_time else "Timing not stated.",
        }
    )
    return items


def evidence_completeness(anatomy: List[str], mechanisms: List[str], symptoms: List[str], red_flags: List[Dict[str, str]], has_time: bool) -> Dict[str, Any]:
    score = 0
    available = []
    missing = []
    if anatomy:
        score += 2
        available.append("Anatomical region is described.")
    else:
        missing.append("Anatomical region is missing or unclear.")
    if mechanisms:
        score += 2
        available.append("Mechanism of injury is described.")
    else:
        missing.append("Mechanism of injury is missing.")
    if len(symptoms) >= 2:
        score += 2
        available.append("Multiple symptoms or functional clues are available.")
    elif symptoms:
        score += 1
        available.append("Some symptom information is available.")
        missing.append("Additional symptoms/function details would reduce uncertainty.")
    else:
        missing.append("Symptoms are not described clearly.")
    if has_time:
        score += 1
        available.append("Timing is described.")
    else:
        missing.append("Timing since onset/injury is missing.")
    if red_flags:
        available.append("Red-flag language was detected and should override routine interpretation.")
    missing.extend(
        [
            "Physical examination findings are not available in text-only mode.",
            "Focal tenderness and range-of-motion findings are not available unless described.",
            "Imaging or formal radiology review is not attached to this text-only mode.",
        ]
    )
    if score >= 6 and not red_flags:
        level = "Moderate"
    elif score >= 4:
        level = "Limited-to-moderate"
    else:
        level = "Limited"
    return {"level": level, "available": available, "missing": missing, "reasons": available + missing}


def suggested_questions(
    text: str,
    anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    has_time: bool,
    input_quality: Dict[str, Any],
    red_flags: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    questions = []

    def add(question: str, missing_field: str, reason: str, priority: str = "useful") -> None:
        if not any(item["missing_field"] == missing_field for item in questions):
            questions.append(
                {
                    "question": question,
                    "missing_field": missing_field,
                    "reason": reason,
                    "priority": priority,
                }
            )

    if not anatomy:
        add("Where exactly is the pain or injury located?", "specific_location", "The anatomical site is not specific enough.", "critical")
    if not mechanisms:
        add("How did the injury happen?", "mechanism", "The mechanism changes the review context.", "critical")
    if not has_time:
        add("When did the pain or injury start?", "timing", "Timing helps separate acute injury from persistent or chronic symptoms.")
    if "numbness" not in symptoms and "numbness" not in denied_symptoms:
        add(
            "Is there numbness, tingling, color change, coldness, or reduced sensation?",
            "neurovascular_status",
            "Sensory or circulation-related symptoms can change the safety priority if present.",
            "critical",
        )
    if "visible deformity" not in symptoms and "visible deformity" not in denied_symptoms:
        add(
            "Is there visible deformity or abnormal alignment?",
            "deformity_status",
            "Visible deformity can change the urgency of human review.",
            "critical",
        )
    if "open wound" not in symptoms and "open wound" not in denied_symptoms:
        add(
            "Is there bleeding, a cut, or an open wound near the painful area?",
            "wound_status",
            "Open injury language changes the safety priority.",
            "critical",
        )
    function_stated = "unable to use limb" in symptoms or contains_any(text, FUNCTION_STATUS_TERMS)
    if not function_stated:
        add(
            "Can the patient use the limb normally, bear weight, or grip objects?",
            "function_status",
            "Function helps prioritize the review and missing-information list.",
        )
    if "swelling" not in symptoms and "swelling" not in denied_symptoms and "bruising" not in denied_symptoms:
        add("Is there swelling or bruising?", "swelling_bruising", "Swelling or bruising helps characterize the injury pattern.", "optional")

    if red_flags:
        priority_order = {"critical": 0, "useful": 1, "optional": 2}
        questions = sorted(questions, key=lambda item: priority_order.get(item["priority"], 1))
    return questions[:5]


def consideration_templates(primary_anatomy: str | None) -> List[Dict[str, str]]:
    templates = {
        "wrist": [("Scaphoid-region injury", "FOOSH/wrist pain patterns can require focused clinical examination."), ("Distal radius injury", "Wrist pain after a fall may involve the distal radius region."), ("Ligament sprain / soft-tissue injury", "Pain and swelling can also reflect non-fracture soft-tissue injury.")],
        "forearm": [("Forearm contusion / soft-tissue injury", "Localized soreness and mild swelling after a direct blow can fit a soft-tissue injury pattern."), ("Radius/ulna-region injury", "Focal mid-forearm pain after impact should still be reviewed for possible long-bone injury."), ("Adjacent wrist/elbow functional screen", "Normal wrist and elbow movement is useful context, but it does not fully exclude forearm injury.")],
        "ankle": [("Ankle-region traumatic injury pattern", "Pain, swelling, inability to bear weight, deformity, or malleolar tenderness can require urgent clinician review."), ("Ankle sprain / ligament injury", "Twisting mechanisms commonly cause ligament injury."), ("Foot or midfoot injury", "Pain extending into the foot may need separate localization.")],
        "knee": [("Patellar or periarticular injury", "Direct trauma, swelling, or reduced motion can require focused review."), ("Ligament or meniscal injury", "Twisting mechanisms and functional limitation may fit internal knee injury."), ("Contusion / soft-tissue injury", "Pain after impact can be non-fracture but still clinically relevant.")],
        "elbow": [("Radial head / elbow-region injury", "Fall mechanisms with elbow pain may need careful range-of-motion assessment."), ("Elbow dislocation or alignment concern", "Deformity or severe motion restriction raises concern."), ("Soft-tissue injury", "Pain and swelling may be soft-tissue related.")],
        "shoulder": [("Clavicle or proximal humerus-region injury", "Fall or direct impact can involve shoulder-girdle bones."), ("Dislocation/alignment concern", "Visible deformity or inability to move the shoulder is important."), ("Soft-tissue injury", "Pain after impact may reflect soft-tissue injury.")],
        "hip": [("Hip or pelvic-region injury", "Hip pain after trauma, especially impaired weight-bearing, requires careful review."), ("Soft-tissue contusion", "Pain after impact may be soft-tissue related."), ("Referred pain / adjacent-region issue", "Poor localization can make hip versus pelvis/leg unclear.")],
        "leg": [("Tibia/fibula-region injury", "Lower-leg pain after trauma may involve long-bone injury."), ("Soft-tissue injury", "Impact or twisting can cause non-fracture injury."), ("Adjacent joint involvement", "Knee or ankle symptoms may change the clinical focus.")],
        "hand": [("Metacarpal or carpal-region injury", "Hand pain, swelling, or grip limitation may require focused review."), ("Finger/thumb injury", "Digit-specific symptoms should be localized."), ("Soft-tissue injury", "Pain and swelling can be ligament/tendon related.")],
    }
    selected = templates.get(primary_anatomy or "")
    if not selected:
        selected = [("Bone injury to review", "The description includes musculoskeletal injury language but localization is incomplete."), ("Soft-tissue injury", "Pain, swelling, or function loss can occur without fracture."), ("Adjacent-region involvement", "Poor localization means nearby joints or bones may also matter.")]
    return [{"condition": condition, "why_relevant": why} for condition, why in selected]


def build_clinical_considerations(primary_anatomy: str | None, anatomy: List[str], mechanisms: List[str], symptoms: List[str], red_flags: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    primary = primary_anatomy or (anatomy[0] if anatomy else None)
    considerations = []
    for item in consideration_templates(primary):
        support = []
        priority_score = 0
        if primary:
            support.append(f"Reported/estimated region: {primary}.")
            priority_score += 1
        if mechanisms:
            support.append(f"Mechanism mentioned: {', '.join(mechanisms[:2])}.")
            priority_score += 1
        if symptoms:
            support.append(f"Symptoms/function clues: {', '.join(symptoms[:3])}.")
            priority_score += min(2, len(symptoms))
        if red_flags:
            priority_score += 2
        considerations.append(
            {
                "condition": item["condition"],
                "relevance": "Review consideration",
                "priority_score": priority_score,
                "why_relevant": item["why_relevant"],
                "supporting_findings": support or ["The text contains general musculoskeletal injury language."],
                "reasoning": (
                    "This item is listed because the described body region, mechanism, and symptom/function signals can fit this review pattern. "
                    "It should be checked against examination and imaging."
                    if priority_score >= 2
                    else "This is a broad review prompt because the text lacks enough localization or symptom detail."
                ),
                "uncertainty": ["No physical examination findings are available.", "No imaging is attached to this text-only workflow.", "The system cannot confirm focal tenderness or exact injury pattern from text alone."],
                "would_change_assessment": ["Exact tenderness location.", "Sensation and circulation examination.", "Ability to bear weight/use the limb.", "Clinician review of any available imaging."],
            }
        )
    ranked = sorted(considerations, key=lambda item: item["priority_score"], reverse=True)
    for idx, item in enumerate(ranked):
        if red_flags and idx == 0:
            item["relevance"] = "Urgent condition requiring clinician assessment"
        elif idx == 0:
            item["relevance"] = "Primary consideration"
        elif idx == 1:
            item["relevance"] = "Relevant alternative"
        else:
            item["relevance"] = "Lower-priority alternative"
    return ranked


def human_list(items: List[str], limit: int = 3) -> str:
    selected = [item for item in items[:limit] if item]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return ", ".join(selected[:-1]) + f" and {selected[-1]}"


DISPLAY_TERMS = {
    "unable to use limb": "inability to use the limb or bear weight",
    "visible deformity": "visible deformity",
    "open wound": "open wound or bleeding",
    "numbness": "altered sensation",
    "reduced movement": "reduced movement",
    "sports injury": "sports-related injury",
    "direct blow": "direct impact",
    "Open injury concern": "open wound/bleeding concern",
    "Altered sensation language": "altered sensation language",
    "Circulation concern": "circulation concern",
    "Deformity / dislocation concern": "visible deformity/alignment concern",
    "High-energy mechanism": "higher-energy mechanism",
    "Systemic symptom concern": "fever or infection-related concern",
}


def display_term(term: str) -> str:
    return DISPLAY_TERMS.get(term, term)


def display_list(items: List[str], limit: int = 3) -> str:
    return human_list([display_term(item) for item in items], limit)


def clinical_support_strength(
    primary_anatomy: str | None,
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
    input_quality: Dict[str, Any],
) -> Dict[str, Any]:
    score = 0
    reasons = []
    if primary_anatomy:
        score += 2
        reasons.append(f"The painful area is localized to the {primary_anatomy} region.")
    if mechanisms:
        score += 2
        reasons.append(f"The mechanism is clinically relevant: {display_list(mechanisms, 2)}.")
    if symptoms:
        score += min(2, len(symptoms))
        reasons.append(f"The description includes injury findings such as {display_list(symptoms, 3)}.")
    if denied_symptoms:
        score += 1
        reasons.append(f"Some escalation features are explicitly denied: {display_list(denied_symptoms, 3)}.")
    if red_flags:
        score += 2
        reasons.append(f"Escalation trigger text is present: {display_list([flag.get('label', '') for flag in red_flags], 2)}.")
    if input_quality.get("percent", 0) >= 75:
        score += 1
        reasons.append("The written description contains enough structured detail for a focused text review.")

    if red_flags and score >= 6:
        level = "Escalation trigger support"
    elif score >= 7:
        level = "Good"
    elif score >= 4:
        level = "Moderate"
    else:
        level = "Limited"

    limitation = "Based only on written words; this is not clinical certainty, diagnosis, or a calibrated risk score."
    return {"level": level, "reasons": reasons[:4], "limitation": limitation}


def clinical_level_rationale(action_level: str, primary_anatomy: str | None, mechanisms: List[str], symptoms: List[str], red_flags: List[Dict[str, str]]) -> str:
    area = primary_anatomy or "the described region"
    if action_level == "Level 4":
        trigger_text = display_list([flag.get("label", "") for flag in red_flags if flag.get("label")], 3) or "warning language"
        return (
            f"This level is driven by {trigger_text} in the written description. "
            "It does not confirm fracture, dislocation, or vascular injury; it means the text should not be cleared by an automated tool alone."
        )
    if action_level == "Level 3":
        return f"This level is based on {area} trauma with functional concern, where waiting at home would be too passive."
    if action_level == "Level 2":
        return f"This level is based on localized {area} symptoms after injury, especially if tenderness, swelling, or reduced function persists."
    if mechanisms or symptoms:
        return f"This level stays lower because the {area} symptoms are described without urgent warning signs."
    return "This level remains cautious because the description is still limited."


def clinical_supporting_phrases(
    primary_anatomy: str | None,
    adjacent_anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
) -> List[str]:
    support: List[str] = []
    red_flag_labels = [item.get("label", "") for item in red_flags if item.get("label")]
    if red_flag_labels:
        support.append(f"Warning signs mentioned: {display_list(red_flag_labels, 3)}.")
    anatomy_specific = {
        "forearm": "Direct forearm impact can be soft-tissue related, but focal bony tenderness would shift review toward radius/ulna injury.",
        "wrist": "Wrist pain after a fall or impact keeps scaphoid and distal-radius review on the table until localization is clearer.",
        "ankle": "Ankle trauma is prioritized by weight-bearing ability, deformity, malleolar tenderness, and sensation/circulation status.",
        "foot": "Foot trauma needs localization because midfoot, metatarsal, and soft-tissue patterns can look similar in text.",
        "knee": "Knee trauma is clarified by swelling, locking/giving-way, range of motion, and ability to bear weight.",
        "elbow": "Elbow injury review depends heavily on extension and forearm rotation, not pain wording alone.",
        "shoulder": "Shoulder-region trauma needs separation between clavicle/proximal-humerus injury, alignment concern, and soft-tissue pain.",
        "hand": "Hand injury review depends on focal metacarpal/finger tenderness, grip function, swelling, and neurovascular status.",
        "leg": "Lower-leg trauma needs careful distinction between soft-tissue impact and focal tibia/fibula tenderness.",
    }
    if primary_anatomy in anatomy_specific:
        support.append(anatomy_specific[primary_anatomy])
    elif primary_anatomy:
        support.append(f"The main painful area is localized to the {primary_anatomy} region.")
    elif mechanisms:
        support.append("The injury mechanism is relevant, but the exact painful area is not fully localized.")
    if symptoms and len(symptoms) >= 2:
        support.append(f"The combination of {display_list(symptoms, 3)} raises more concern than isolated pain alone.")
    elif symptoms:
        support.append("Only one main symptom is described, so exact tenderness location still matters.")
    if denied_symptoms:
        support.append(f"Denied features such as {display_list(denied_symptoms, 3)} lower some immediate concern, but do not rule out injury.")
    if adjacent_anatomy:
        support.append(f"Nearby areas ({human_list(adjacent_anatomy, 3)}) are noted as context, not as the main site.")
    return support or ["The text provides limited but relevant musculoskeletal injury context."]


def alternative_reason(condition: str, primary_anatomy: str | None, mechanisms: List[str], symptoms: List[str]) -> str:
    lower = condition.lower()
    if "distal radius" in lower:
        return "Falls onto an outstretched hand or wrist impact can involve the distal-radius region, and the description cannot localize tenderness precisely."
    if "scaphoid" in lower:
        return "Thumb-side wrist pain after a fall can require focused scaphoid review, especially when imaging is not available here."
    if "ligament" in lower or "soft-tissue" in lower or "contusion" in lower:
        return "Pain, swelling, bruising, or reduced movement can come from soft-tissue injury as well as bone injury."
    if "radius" in lower or "ulna" in lower or "tibia" in lower or "fibula" in lower:
        return "Long-bone region injury remains relevant when pain is focal after direct impact or twisting trauma."
    if "dislocation" in lower or "alignment" in lower:
        return "Alignment concerns stay on the review list when deformity, severe limitation, or high-risk wording is present."
    if "adjacent" in lower or "foot" in lower or "midfoot" in lower:
        return "Adjacent-region symptoms can shift the clinical focus and should be checked during examination."
    if primary_anatomy:
        return f"This remains possible because {primary_anatomy}-region text cannot separate bone, joint, and soft-tissue findings without examination."
    return "This remains possible because the written description is not specific enough to separate injury patterns."


def watch_for_items(primary_anatomy: str | None, symptoms: List[str], red_flags: List[Dict[str, str]], action_level: str) -> List[str]:
    if action_level == "Level 4":
        labels = {flag.get("label", "") for flag in red_flags}
        items: List[str] = []
        anatomy_priority = {
            "elbow": "loss of full elbow extension, painful forearm rotation, increasing swelling, or focal bony tenderness",
            "wrist": "snuffbox/thumb-side tenderness, worsening grip, increasing swelling, or finger sensory change",
            "forearm": "worsening pain with rotation, increasing tightness/swelling, or focal radius/ulna tenderness",
            "ankle": "inability to bear weight, increasing deformity, malleolar tenderness, or foot sensory/color change",
            "foot": "midfoot tenderness, spreading bruising, inability to bear weight, or toe sensory/color change",
            "knee": "rapid swelling, locking/giving-way, inability to fully bend/straighten, or weight-bearing loss",
            "shoulder": "visible deformity, inability to move the shoulder, or new arm sensory/strength change",
            "hand": "worsening grip, finger motion loss, focal metacarpal tenderness, or finger sensory/color change",
            "leg": "increasing lower-leg tightness/swelling, deformity, focal tibia/fibula tenderness, or foot sensory/color change",
        }
        if primary_anatomy in anatomy_priority:
            items.append(anatomy_priority[primary_anatomy])
        if "Altered sensation language" in labels:
            items.append("whether sensory symptoms become constant, spread, or are associated with weakness, coldness, color change, or worsening function")
        if "Circulation concern" in labels:
            items.append("skin color, warmth, capillary refill, pulse, or any cold/pale/blue appearance")
        if "Open injury concern" in labels:
            items.append("any open wound, bleeding, contamination, or visible bone")
        if "Deformity / dislocation concern" in labels:
            items.append("abnormal alignment, visible deformity, or inability to use the limb normally")
        items.append("because warning language is already present, do not wait for progression before human clinical review")
        return items[:5]
    anatomy_watch = {
        "forearm": [
            "increasing forearm swelling or tightness",
            "worsening pain with rotation of the forearm",
            "focal tenderness directly over the radius or ulna",
            "new numbness, weakness, color change, or coldness in the hand",
        ],
        "wrist": [
            "thumb-side wrist or snuffbox tenderness",
            "worsening grip strength or increasing wrist swelling",
            "finger numbness, color change, coldness, or progressive weakness",
            "visible deformity or pain that remains focal over bone",
        ],
        "hand": [
            "worsening grip limitation or finger motion loss",
            "increasing hand swelling or focal metacarpal tenderness",
            "new finger numbness, color change, coldness, or deformity",
        ],
        "ankle": [
            "inability to bear weight or worsening limp",
            "increasing ankle deformity, swelling, or focal malleolar tenderness",
            "coldness, numbness, color change, or increasing foot weakness",
        ],
        "foot": [
            "worsening weight-bearing pain",
            "midfoot tenderness, spreading swelling, or bruising",
            "coldness, numbness, color change, or increasing toe weakness",
        ],
        "knee": [
            "increasing swelling around the knee",
            "locking, giving way, or inability to fully bend/straighten the knee",
            "new numbness, coldness, color change, or worsening inability to bear weight",
        ],
        "elbow": [
            "inability to fully rotate the forearm or extend the elbow",
            "increasing elbow swelling or focal bony tenderness",
            "new hand numbness, weakness, color change, or coldness",
        ],
        "shoulder": [
            "increasing deformity or inability to move the shoulder",
            "new arm numbness, weakness, color change, or coldness",
            "pain that remains focal around the clavicle or proximal humerus",
        ],
        "leg": [
            "increasing lower-leg swelling, tightness, or deformity",
            "worsening pain with weight-bearing",
            "new foot numbness, weakness, color change, or coldness",
        ],
    }
    items = anatomy_watch.get(primary_anatomy or "", [
        "increasing pain or swelling",
        "reduced ability to move or use the affected area",
        "new numbness, tingling, weakness, color change, or coldness",
        "visible deformity, open wound, or bleeding",
    ])
    if "open wound" not in symptoms:
        items.append("any new cut, bleeding, or open wound near the painful area")
    return items[:5]


def action_summary_for(action_level: str, primary_anatomy: str | None, symptoms: List[str], red_flags: List[Dict[str, str]]) -> str:
    area = primary_anatomy or "the affected area"
    if action_level == "Level 4":
        labels = [flag.get("label", "") for flag in red_flags if flag.get("label")]
        red_flag_text = display_list(labels, 3)
        if labels == ["Altered sensation language"]:
            return (
                "In-person review should not be delayed because the text reports altered sensation language. "
                "This is a safety trigger from the written description, not proof of circulation compromise or a confirmed nerve injury."
            )
        if red_flag_text:
            return f"Urgent in-person assessment is appropriate because the description includes {red_flag_text}. Do not use this response to delay emergency or trauma review."
        return "Urgent in-person assessment is appropriate because the description contains high-priority warning language."
    if action_level == "Level 3":
        return f"Prompt clinical assessment is appropriate for the {area} because the description suggests more than a minor self-limited injury. Examination, and imaging if clinically needed, should guide the next decision."
    if action_level == "Level 2":
        anatomy_focus = {
            "wrist": "snuffbox/thumb-side tenderness, distal-radius tenderness, persistent swelling, or painful grip",
            "forearm": "focal radius/ulna tenderness, increasing swelling, or pain with rotation",
            "ankle": "malleolar tenderness, persistent swelling, or difficulty bearing weight",
            "foot": "midfoot/base-of-fifth tenderness, spreading bruising, or difficulty bearing weight",
            "knee": "focal bony tenderness, swelling, locking/giving-way, or limited motion",
            "elbow": "limited extension, pain with forearm rotation, or focal bony tenderness",
            "shoulder": "focal clavicle/proximal-humerus tenderness or persistent movement limitation",
            "hand": "focal metacarpal/finger tenderness, grip limitation, or increasing swelling",
            "leg": "focal tibia/fibula tenderness, increasing swelling, or painful weight-bearing",
        }
        focus = anatomy_focus.get(primary_anatomy or "", "persistent focal tenderness, swelling, or reduced function")
        return f"Routine clinician review is reasonable if any of the following are present or persist: {focus}. The text does not show an emergency pattern, but it cannot rule out structural injury."
    anatomy_focus = {
        "forearm": "forearm soreness stays mild, function remains normal, and no focal bony tenderness or swelling develops",
        "wrist": "wrist pain is mild, improving, and not focal over the snuffbox or distal radius",
        "ankle": "weight-bearing remains comfortable and swelling does not progress",
        "foot": "weight-bearing remains comfortable and pain is not focal over the midfoot or metatarsals",
        "knee": "movement remains comfortable and swelling or instability does not develop",
        "elbow": "extension and forearm rotation remain comfortable",
        "shoulder": "movement remains comfortable and there is no focal clavicle/proximal-humerus tenderness",
    }
    focus = anatomy_focus.get(primary_anatomy or "", "symptoms remain mild, improving, and no warning features develop")
    return f"Self-monitoring is reasonable only while {focus}. Seek clinician review if symptoms persist, become focal, or function worsens."


def refinement_questions(primary_anatomy: str | None, action_level: str) -> List[Dict[str, str]]:
    if action_level not in {"Level 1", "Level 2"}:
        return []
    questions = {
        "wrist": "Is the pain strongest in the anatomical snuffbox/thumb-side wrist or directly over the distal radius?",
        "forearm": "Is the tenderness directly over the bone or mainly in the surrounding soft tissue?",
        "ankle": "Can the patient bear weight now, and is the pain focal over either malleolus?",
        "foot": "Is the pain focal in the midfoot/base of the fifth metatarsal or more diffuse?",
        "knee": "Is there focal bony tenderness, locking/giving-way, or mainly soft-tissue soreness?",
        "elbow": "Can the patient fully extend the elbow and rotate the forearm?",
        "shoulder": "Is the pain focal over the clavicle/proximal humerus or mainly muscular around the shoulder?",
        "leg": "Is the pain focal over the tibia/fibula or more diffuse in the soft tissue?",
        "hand": "Is the pain focal over a metacarpal/finger bone or mainly in the soft tissue?",
    }
    question = questions.get(primary_anatomy or "", "Is the pain focal over bone or more diffuse in the surrounding soft tissue?")
    return [{
        "question": question,
        "missing_field": "focused_localization_refinement",
        "reason": "This would make the review more specific without changing the current safety level.",
        "priority": "useful refinement",
    }]


def urgent_non_delaying_questions(primary_anatomy: str | None, red_flags: List[Dict[str, str]]) -> List[Dict[str, str]]:
    labels = {item.get("label", "") for item in red_flags}
    if "Altered sensation language" in labels:
        anatomy_questions = {
            "elbow": "Is the tingling limited to the ring/little fingers, and is hand color, warmth, and strength normal?",
            "wrist": "Are the fingers warm and normally colored, and is the altered sensation constant or intermittent?",
            "hand": "Which fingers have altered sensation, and is finger color/warmth normal?",
            "forearm": "Is altered sensation spreading into the hand, and is grip strength normal?",
            "ankle": "Is foot/toe sensation normal now, and is the foot warm and normally colored?",
            "foot": "Are the toes warm and normally colored, and is altered sensation constant or intermittent?",
        }
        question = anatomy_questions.get(primary_anatomy or "", "Is the altered sensation constant or intermittent, and is color, warmth, and strength normal?")
        return [{
            "question": question,
            "missing_field": "non_delaying_red_flag_detail",
            "reason": "Useful for the clinician, but it should not delay the recommended in-person review.",
            "priority": "non-delaying clarification",
        }]
    if "Circulation concern" in labels:
        return [{
            "question": "Is the affected area cold, pale/blue, or associated with reduced pulse or worsening weakness?",
            "missing_field": "non_delaying_circulation_detail",
            "reason": "This helps characterize the warning sign, but urgent review should not wait for a text answer.",
            "priority": "non-delaying clarification",
        }]
    return []


def follow_up_display_config(follow_up: List[Dict[str, str]], action_level: str) -> Dict[str, str]:
    priorities = {item.get("priority", "") for item in follow_up}
    if "critical" in priorities:
        return {
            "title": "Key missing details",
            "empty_message": "No essential follow-up question before the recommended action.",
        }
    if "non-delaying clarification" in priorities:
        return {
            "title": "Non-delaying clarification",
            "empty_message": "No clarification should delay the recommended in-person review.",
        }
    if follow_up:
        return {
            "title": "Most useful clarification",
            "empty_message": "No essential follow-up question before the recommended action.",
        }
    if action_level == "Level 4":
        return {
            "title": "Non-delaying clarification",
            "empty_message": "No follow-up question should delay urgent in-person assessment.",
        }
    return {
        "title": "Useful missing details",
        "empty_message": "No essential follow-up question before the recommended action.",
    }


def build_clinical_direction(
    primary_anatomy: str | None,
    adjacent_anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
    considerations: List[Dict[str, Any]],
    input_quality: Dict[str, Any],
) -> Dict[str, Any]:
    most_compatible = considerations[0]["condition"] if considerations else "Musculoskeletal injury pattern for review"
    relevant_alternative = (
        considerations[1]["condition"]
        if len(considerations) > 1
        else f"{primary_anatomy or 'regional'} bone or soft-tissue injury that cannot be separated from text alone"
    )
    lower_priority = considerations[2]["condition"] if len(considerations) > 2 else "Adjacent-region involvement"

    support = clinical_supporting_phrases(primary_anatomy, adjacent_anatomy, mechanisms, symptoms, denied_symptoms, red_flags)
    support_strength = clinical_support_strength(primary_anatomy, mechanisms, symptoms, denied_symptoms, red_flags, input_quality)

    red_flag_labels = [item.get("label", "") for item in red_flags]
    severe_features = bool(red_flags)
    functional_loss = "unable to use limb" in symptoms
    swelling_or_pain = bool(set(symptoms) & {"pain", "swelling", "bruising", "reduced movement"})
    high_energy = "high-energy trauma" in mechanisms or any("High-energy" in label for label in red_flag_labels)
    open_deformity_neuro = bool(set(symptoms) & {"open wound", "visible deformity", "numbness", "weakness"})

    if severe_features and (open_deformity_neuro or len(red_flags) >= 2):
        action = {
            "level": "Level 4",
            "title": "Urgent in-person assessment",
            "tone": "urgent",
            "summary": "",
        }
    elif severe_features or functional_loss or high_energy:
        action = {
            "level": "Level 3",
            "title": "Prompt clinical assessment",
            "tone": "prompt",
            "summary": "",
        }
    elif swelling_or_pain and ("swelling" in symptoms or "reduced movement" in symptoms or input_quality.get("percent", 0) < 75):
        action = {
            "level": "Level 2",
            "title": "Routine clinical assessment if symptoms persist or local tenderness is focal",
            "tone": "routine",
            "summary": "",
        }
    else:
        action = {
            "level": "Level 1",
            "title": "Self-monitoring context with clear warning signs",
            "tone": "monitor",
            "summary": "",
        }

    if action["level"] == "Level 4":
        if considerations:
            most_compatible = f"{considerations[0]['condition']} with warning-language requiring human review"
        elif primary_anatomy:
            most_compatible = f"{primary_anatomy}-region injury pattern with warning-language requiring human review"
        else:
            most_compatible = "Musculoskeletal injury pattern with warning-language requiring human review"
        relevant_alternative = considerations[1]["condition"] if len(considerations) > 1 else "Fracture/dislocation-region injury concern"
        lower_priority = considerations[2]["condition"] if len(considerations) > 2 else "Associated soft-tissue or adjacent-region injury"
    watch_for = watch_for_items(primary_anatomy, symptoms, red_flags, action["level"])
    level_rationale = clinical_level_rationale(action["level"], primary_anatomy, mechanisms, symptoms, red_flags)
    action["summary"] = action_summary_for(action["level"], primary_anatomy, symptoms, red_flags)

    area_text = primary_anatomy or "the described region"
    mechanism_text = f" after {display_list(mechanisms, 2)}" if mechanisms else ""
    symptom_text = f" with {display_list(symptoms, 3)}" if symptoms else ""
    if action["level"] == "Level 4":
        plain_language = (
            f"This reads as a {area_text} injury scenario{mechanism_text}{symptom_text} where the text includes warning-language. "
            "The priority is human review of the described warning feature, while the exact injury type still cannot be decided from text alone."
        )
    elif action["level"] == "Level 3":
        plain_language = (
            f"This description raises more than routine concern for the {area_text}{mechanism_text}{symptom_text}. "
            "The safest interpretation is prompt clinician assessment rather than home monitoring."
        )
    elif action["level"] == "Level 2":
        plain_language = (
            f"The wording points to a focused {area_text} injury pattern{mechanism_text}{symptom_text}. "
            "It is not an emergency pattern from the submitted text, but persistent focal symptoms would justify clinical review."
        )
    else:
        plain_language = (
            f"The description is most consistent with a lower-acuity {area_text} injury pattern{mechanism_text}{symptom_text}. "
            "This depends on symptoms remaining mild and no warning features developing."
        )
    uncertainty = [
        "The written description cannot check focal tenderness or alignment.",
        "No imaging or formal radiology review is attached here.",
        "Symptoms can change after the initial description.",
    ]
    return {
        "headline": "Urgent review priority (text-only, non-diagnostic)" if action["level"] == "Level 4" else "Text-only clinical review focus",
        "most_compatible": most_compatible,
        "plain_language": plain_language,
        "alternatives": [
            {"label": "Relevant alternative", "text": relevant_alternative, "reason": alternative_reason(relevant_alternative, primary_anatomy, mechanisms, symptoms)},
            {"label": "Lower-priority alternative", "text": lower_priority, "reason": alternative_reason(lower_priority, primary_anatomy, mechanisms, symptoms)},
        ],
        "action": action,
        "level_rationale": level_rationale,
        "support_strength": support_strength,
        "supporting_factors": support,
        "uncertainty": uncertainty,
        "watch_for": watch_for,
        "closing": (
            "This is review support, not a treatment plan. Because warning-language is present, do not use this output to delay in-person clinical review."
            if action["level"] == "Level 4"
            else "This is review support, not a treatment plan. Seek professional medical assessment if symptoms are worsening, persistent, focal over bone, or include warning signs."
        ),
    }


def clinical_reasoning_summary(
    anatomy: List[str],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
    evidence_pack: Dict[str, Any],
) -> Dict[str, Any]:
    body = anatomy[0] if anatomy else "an unclear musculoskeletal region"
    mechanism_text = ", ".join(mechanisms[:2]) if mechanisms else "an unclear mechanism"
    symptom_text = ", ".join(symptoms[:3]) if symptoms else "limited symptom detail"
    red_flag_text = (
        f"{len(red_flags)} escalation trigger(s) were found: {', '.join(flag['label'] for flag in red_flags)}."
        if red_flags
        else "No explicit escalation trigger was detected in the submitted text."
    )
    denied_text = (
        f"The text also explicitly denies: {', '.join(denied_symptoms)}."
        if denied_symptoms
        else "No important negative symptoms were stated clearly."
    )
    rag_text = (
        "The intake is specific enough for PubMed/RAG context."
        if (evidence_pack.get("readiness") or {}).get("ready")
        else "PubMed/RAG context is deferred until more detail is added."
    )
    return {
        "summary": (
            "The submitted text was converted into structured review context. "
            f"{red_flag_text} {denied_text} {rag_text}"
        ),
        "reasoning_steps": [
            "The system first checked whether the text described a musculoskeletal problem.",
            "It then extracted body region, mechanism, symptoms, denied symptoms, timing, and red-flag language.",
            "Review priorities were ranked from the extracted body region, mechanism, symptoms, and escalation triggers.",
            "PubMed/RAG was used only if the intake was sufficiently structured.",
        ],
    }


def clinical_consistency_gate(
    primary_anatomy: str | None,
    anatomy_context: Dict[str, Any],
    mechanisms: List[str],
    symptoms: List[str],
    denied_symptoms: List[str],
    red_flags: List[Dict[str, str]],
    clinical_direction: Dict[str, Any],
) -> Dict[str, Any]:
    issues = []
    denied_anatomy = set(anatomy_context.get("denied") or [])
    present_anatomy = set(anatomy_context.get("present") or [])
    action_level = ((clinical_direction.get("action") or {}).get("level") or "")
    if primary_anatomy and primary_anatomy in denied_anatomy:
        issues.append("Denied or painless anatomy was selected as the primary injury site.")
    if primary_anatomy and present_anatomy and primary_anatomy not in present_anatomy:
        issues.append("Primary anatomy does not match the extracted painful/injured site.")
    denied_red_flag_map = {
        "Open injury concern": "open wound",
        "Altered sensation language": "numbness",
        "Circulation concern": "numbness",
        "Deformity / dislocation concern": "visible deformity",
        "Systemic symptom concern": "fever/systemic symptoms",
    }
    for flag in red_flags:
        denied_concept = denied_red_flag_map.get(flag.get("label", ""))
        if denied_concept and denied_concept in denied_symptoms:
            issues.append(f"Denied finding was treated as red flag: {denied_concept}.")
    high_risk_red_flags = {
        "Open injury concern",
        "Altered sensation language",
        "Circulation concern",
        "Deformity / dislocation concern",
        "High-energy mechanism",
    }
    high_risk_count = sum(1 for flag in red_flags if flag.get("label") in high_risk_red_flags)
    if high_risk_count >= 2 and action_level != "Level 4":
        issues.append("Multiple high-risk red flags were detected but action level is not Level 4.")
    if not red_flags and action_level == "Level 4":
        issues.append("Level 4 action was selected without a positive red-flag trigger.")
    ready = not issues
    return {
        "status": "pass" if ready else "fail",
        "ready_for_rendering": ready,
        "issues": issues,
        "message": (
            "Structured extraction passed consistency checks."
            if ready
            else "Structured extraction contains unresolved inconsistency. A provisional clinical direction was withheld pending corrected input or human review."
        ),
    }


def withheld_clinical_direction(consistency: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "headline": "Clinical direction withheld",
        "most_compatible": "Structured extraction inconsistency",
        "plain_language": consistency.get("message") or "The structured extraction contains unresolved inconsistency.",
        "alternatives": [],
        "action": {
            "level": "Review withheld",
            "title": "Correct input or obtain human review",
            "tone": "protected",
            "summary": "The app did not generate a provisional clinical direction because extracted fields were internally inconsistent.",
        },
        "level_rationale": "The system withheld this section because the extracted clinical fields did not pass consistency checks.",
        "support_strength": {
            "level": "Withheld",
            "reasons": consistency.get("issues", []),
            "limitation": "No support strength is reported when consistency checks fail.",
        },
        "supporting_factors": consistency.get("issues", []),
        "uncertainty": ["The input should be corrected or reviewed by a human before using this text-only support mode."],
        "watch_for": [],
        "closing": "No clinical direction, PubMed/RAG synthesis, or LLM wording should override a failed consistency check.",
    }


def problem_pubmed_query(primary_anatomy: str | None, mechanisms: List[str], red_flags: List[Dict[str, str]]) -> str:
    anatomy_term = primary_anatomy or "musculoskeletal"
    anatomy_query_terms = {
        "wrist": "(wrist OR scaphoid OR distal radius)",
        "hand": "(hand OR metacarpal OR carpal)",
        "forearm": "(forearm OR radius OR ulna)",
        "elbow": "(elbow OR radial head)",
        "ankle": "(ankle OR malleolus)",
        "foot": "(foot OR metatarsal OR midfoot)",
        "knee": "(knee OR patella)",
        "shoulder": "(shoulder OR clavicle OR proximal humerus)",
        "hip": "(hip OR pelvis)",
        "leg": "(tibia OR fibula OR lower leg)",
    }
    anatomy_clause = anatomy_query_terms.get(anatomy_term, f"({anatomy_term})")
    injury_terms = "fracture OR injury OR trauma"
    if "twisting injury" in mechanisms:
        injury_terms = "sprain OR fracture OR injury"
    if "fall" in mechanisms or "sports injury" in mechanisms:
        injury_terms = f"({injury_terms}) AND (acute OR fall OR trauma OR radiograph OR imaging)"
    if red_flags:
        red_flag_terms = {
            "wrist": "(fracture OR scaphoid OR distal radius OR dislocation OR radiograph OR imaging OR emergency OR urgent)",
            "hand": "(fracture OR metacarpal OR dislocation OR radiograph OR imaging OR emergency OR urgent)",
            "forearm": "(fracture OR radius OR ulna OR compartment OR radiograph OR imaging OR emergency OR urgent)",
            "elbow": "(fracture OR radial head OR elbow dislocation OR radiograph OR imaging OR emergency OR urgent)",
            "ankle": "(fracture OR dislocation OR deformity OR malleolar OR radiograph OR imaging OR emergency OR urgent)",
            "foot": "(fracture OR midfoot OR metatarsal OR Lisfranc OR radiograph OR imaging OR emergency OR urgent)",
            "knee": "(fracture OR patella OR dislocation OR radiograph OR imaging OR emergency OR urgent)",
            "shoulder": "(fracture OR dislocation OR clavicle OR proximal humerus OR radiograph OR imaging OR emergency OR urgent)",
            "leg": "(fracture OR tibia OR fibula OR compartment OR radiograph OR imaging OR emergency OR urgent)",
        }
        injury_terms = f"({injury_terms}) AND {red_flag_terms.get(anatomy_term, '(fracture OR dislocation OR radiograph OR imaging OR emergency OR urgent)')}"
    return (
        f"{anatomy_clause} AND ({injury_terms}) AND "
        "(management OR diagnosis OR imaging OR guideline) AND "
        "(review[Publication Type] OR systematic review[Publication Type] OR guideline[Publication Type] OR meta-analysis[Publication Type]) "
        "NOT chronic NOT Kienbock NOT hypermobility NOT osteoporosis NOT postmenopausal NOT \"stress fracture\""
    )


def pubmed_json(endpoint: str, params: Dict[str, Any], timeout: float = 4.0) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}?{query}",
        headers={"User-Agent": "ProjectDiplomatiki-XrayReview/0.1 academic prototype"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evidence_level_from_pubtypes(pubtypes: List[str]) -> str:
    joined = " ".join(pubtypes).lower()
    if "guideline" in joined:
        return "Guideline"
    if "meta-analysis" in joined:
        return "Meta-analysis"
    if "systematic review" in joined:
        return "Systematic review"
    if "review" in joined:
        return "Review article"
    return "PubMed source"


def pubmed_support_statement(anatomy_term: str, mechanisms: List[str], red_flags: List[Dict[str, str]]) -> str:
    if red_flags:
        return (
            f"Topic-overlap evidence for {anatomy_term} injury review with escalation-related search terms; "
            "it supports background clinician context, not a patient-matched urgency decision."
        )
    if "twisting injury" in mechanisms:
        return (
            f"Background evidence for {anatomy_term} twisting injury, sprain/fracture differential, "
            "and imaging or management review."
        )
    if "fall" in mechanisms or "sports injury" in mechanisms:
        return (
            f"Background evidence for {anatomy_term} injury after fall or sport-related trauma, "
            "including imaging or management review."
        )
    return f"Background evidence for {anatomy_term} musculoskeletal injury review and clinician context."


def pubmed_selection_summary(anatomy_term: str, mechanisms: List[str], red_flags: List[Dict[str, str]]) -> str:
    mechanism_text = ", ".join(mechanisms[:2]) if mechanisms else "unspecified mechanism"
    if red_flags:
        return (
            f"Retrieved because the intake mentioned {anatomy_term}, {mechanism_text}, "
            "and warning-language; the query prioritized anatomy-specific reviews/guidelines around imaging, diagnosis, or management."
        )
    return (
        f"Retrieved because the intake mentioned {anatomy_term} with {mechanism_text}; "
        "the query prioritized reviews/guidelines around imaging, diagnosis, or management."
    )


def pubmed_source_match(title: str, pubtypes: List[str], anatomy_term: str, mechanisms: List[str], red_flags: List[Dict[str, str]]) -> Dict[str, Any]:
    title_l = title.lower()
    pubtype_l = " ".join(pubtypes).lower()
    score = 0
    reasons = []
    anatomy_aliases = {
        "wrist": ["wrist", "scaphoid", "distal radius"],
        "hand": ["hand", "metacarpal", "carpal", "finger"],
        "forearm": ["forearm", "radius", "ulna"],
        "elbow": ["elbow", "radial head"],
        "ankle": ["ankle", "malleolar", "malleolus"],
        "foot": ["foot", "midfoot", "metatarsal", "lisfranc"],
        "knee": ["knee", "patella"],
        "shoulder": ["shoulder", "clavicle", "proximal humerus"],
        "hip": ["hip", "pelvis"],
        "leg": ["tibia", "fibula", "lower leg"],
    }
    aliases = anatomy_aliases.get(anatomy_term, [anatomy_term.lower()] if anatomy_term else [])
    anatomy_matched = bool(anatomy_term and anatomy_term != "musculoskeletal" and any(term in title_l for term in aliases))
    if anatomy_matched:
        score += 2
        reasons.append("anatomy/topic appears in title")
    if any(term in title_l for term in ["fracture", "injury", "trauma", "sprain"]):
        score += 1
        reasons.append("injury/fracture language appears in title")
    if any(term in title_l for term in ["management", "diagnosis", "imaging", "guideline", "treatment"]):
        score += 1
        reasons.append("management/diagnosis/imaging language appears in title")
    if any(kind in pubtype_l for kind in ["guideline", "systematic review", "meta-analysis", "review"]):
        score += 1
        reasons.append("higher-level publication type")
    red_flag_title_terms = ["emergency", "urgent", "neurovascular", "open", "dislocation", "fracture"]
    if anatomy_term == "ankle":
        red_flag_title_terms.append("malleolar")
    if anatomy_term == "elbow":
        red_flag_title_terms.append("radial head")
    if red_flags and any(term in title_l for term in red_flag_title_terms):
        score += 1
        reasons.append("warning-related topic appears in title")
    if "twisting injury" in mechanisms and any(term in title_l for term in ["sprain", "ligament", "instability"]):
        score += 0 if red_flags else 1
        if not red_flags:
            reasons.append("twisting/sprain mechanism match")
    if ("fall" in mechanisms or "sports injury" in mechanisms) and any(term in title_l for term in ["acute", "trauma", "fracture", "scaphoid", "distal radius"]):
        score += 1
        reasons.append("acute trauma context match")
    if any(term in title_l for term in ["chronic", "kienböck", "kienbock", "hypermobility", "occupational therapy"]):
        score -= 2
        reasons.append("penalized: chronic or broad rehabilitation topic")
    if red_flags and "sprain" in title_l and not any(term in title_l for term in ["fracture", "dislocation", "malleolar", "radial head"]):
        score -= 2
        reasons.append("penalized: sprain-only focus in red-flag case")
    if anatomy_term != "ankle" and any(term in title_l for term in ["malleolar", "malleolus", "ankle"]):
        score -= 3
        reasons.append("penalized: ankle-specific source for non-ankle case")
    if anatomy_term and anatomy_term != "musculoskeletal" and not anatomy_matched:
        score -= 1
        reasons.append("limited anatomy overlap in title")
    if any(term in title_l for term in ["pediatric", "children", "child"]) and "child" not in title_l:
        score -= 1
        reasons.append("penalized: population may not match the submitted case")
    if score >= 5 and anatomy_matched:
        label = "Focused topic overlap"
    elif score >= 3 and anatomy_matched:
        label = "Moderate topic overlap"
    elif score < 0:
        label = "Excluded as insufficiently relevant"
    else:
        label = "Broad background source"
    return {"score": score, "label": label, "reasons": reasons[:3]}


def fetch_pubmed_problem_sources(query: str, anatomy_term: str, mechanisms: List[str], red_flags: List[Dict[str, str]], limit: int = 6) -> List[Dict[str, str]]:
    search = pubmed_json(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": limit,
            "sort": "relevance",
        },
    )
    ids = ((search.get("esearchresult") or {}).get("idlist") or [])[:limit]
    if not ids:
        return []
    summary = pubmed_json(
        "esummary.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        },
    )
    result = summary.get("result") or {}
    cards = []
    for pmid in ids:
        item = result.get(pmid) or {}
        title = item.get("title") or f"PubMed source {pmid}"
        pubtypes = item.get("pubtype") or []
        pubdate = item.get("pubdate") or ""
        year = pubdate[:4] if pubdate else "Year unavailable"
        journal = item.get("fulljournalname") or item.get("source") or "Journal unavailable"
        match = pubmed_source_match(title, pubtypes, anatomy_term, mechanisms, red_flags)
        cards.append(
            {
                "title": title.rstrip("."),
                "level": evidence_level_from_pubtypes(pubtypes),
                "summary": pubmed_selection_summary(anatomy_term, mechanisms, red_flags),
                "limitation": "Source selection is automated; clinician must verify applicability to the patient.",
                "pmid": pmid,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "supports": pubmed_support_statement(anatomy_term, mechanisms, red_flags),
                "match_score": match["score"],
                "match_label": match["label"],
                "match_reasons": match["reasons"],
            }
        )
    return sorted(cards, key=lambda card: card.get("match_score", 0), reverse=True)


def rag_readiness(primary_anatomy: str | None, mechanisms: List[str], symptoms: List[str], red_flags: List[Dict[str, str]], input_quality: Dict[str, Any]) -> Dict[str, Any]:
    blockers = []
    if input_quality.get("score", 0) < 5:
        blockers.append("At least 5 of 8 requested intake details are needed for PubMed/RAG context.")
    if not primary_anatomy:
        blockers.append("A specific anatomical region is missing.")
    if not mechanisms and not red_flags:
        blockers.append("Mechanism of injury or red-flag context is missing.")
    if not symptoms and not red_flags:
        blockers.append("Symptoms or functional clues are missing.")
    ready = not blockers
    return {
        "ready": ready,
        "level": "RAG-ready" if ready else "RAG deferred",
        "blockers": blockers,
        "message": (
            "The input is structured enough for PubMed/RAG evidence context."
            if ready
            else "The intake can still be structured, but PubMed/RAG evidence context is deferred until the missing details are added."
        ),
    }


def evidence_synthesis(cards: List[Dict[str, Any]], readiness: Dict[str, Any]) -> Dict[str, Any]:
    if not readiness.get("ready"):
        return {
            "headline": "Evidence synthesis not generated",
            "bullets": [
                "The written intake is not specific enough for a useful PubMed/RAG synthesis.",
                "Structured intake can continue, but literature context should wait until the missing fields are added.",
            ],
            "claims": [],
        }
    source_cards = [card for card in cards if card.get("pmid")]
    strong_cards = [card for card in source_cards if str(card.get("match_label", "")).startswith("Focused")]
    moderate_cards = [card for card in source_cards if str(card.get("match_label", "")).startswith("Moderate")]
    broad_cards = [card for card in source_cards if str(card.get("match_label", "")).startswith("Broad")]
    strong = len(strong_cards)
    moderate = len(moderate_cards)
    broad = len(broad_cards)
    if not source_cards:
        return {
            "headline": "No PubMed sources retrieved",
            "bullets": ["The query did not return source cards, so no evidence synthesis was generated."],
            "claims": [],
        }
    claim_cards = (strong_cards + moderate_cards)[:3]
    claims = []
    for card in claim_cards:
        claim_text = card.get("supports") or "Source provides background clinician-review context."
        source_id = f"PMID:{card.get('pmid')}"
        existing = next((claim for claim in claims if claim["claim"] == claim_text), None)
        if existing:
            existing["source_ids"].append(source_id)
            if str(card.get("match_label", "")).startswith("Strong"):
                existing["case_match"] = "strong"
            continue
        claims.append(
            {
                "claim": claim_text,
                "source_ids": [source_id],
                "evidence_level": card.get("level") or "PubMed source",
                "case_match": "focused topic-overlap" if str(card.get("match_label", "")).startswith("Focused") else "moderate topic-overlap",
                "limitation": card.get("limitation") or "Clinician must verify applicability.",
            }
        )
    bullets = [
        f"{len(source_cards)} PubMed source card(s) were retrieved for background clinician context.",
        f"Most relevant source profile: {strong} focused topic-overlap, {moderate} moderate topic-overlap, {broad} broad background.",
        "The sources provide background clinician context only; they do not determine diagnosis, imaging need, immobilisation, medication, surgery, or discharge.",
    ]
    if strong == 0:
        bullets.append("No strong source match was found, so the evidence context should be treated as broad background rather than case-specific guidance.")
    return {"headline": "Evidence synthesis for clinician review", "bullets": bullets, "claims": claims}


def problem_evidence_cards(primary_anatomy: str | None, mechanisms: List[str], symptoms: List[str], red_flags: List[Dict[str, str]], input_quality: Dict[str, Any]) -> Dict[str, Any]:
    anatomy_phrase = primary_anatomy or "musculoskeletal"
    query = problem_pubmed_query(primary_anatomy, mechanisms, red_flags)
    readiness = rag_readiness(primary_anatomy, mechanisms, symptoms, red_flags, input_quality)
    if not readiness["ready"]:
        boundary_card = {
            "title": "Clinical context boundary",
            "level": "Safety rule",
            "summary": "The text can be organized as intake, but the evidence query was not run because the prompt is not specific enough.",
            "limitation": "Add the missing intake details before using PubMed/RAG context.",
            "pmid": "",
            "journal": "Internal safety policy",
            "year": "V1",
            "url": "",
            "supports": "Boundary for low-detail Problem Description Mode outputs.",
            "match_score": 0,
            "match_label": "RAG not run",
            "match_reasons": readiness["blockers"],
        }
        return {
            "status": "PubMed/RAG deferred: insufficient structured context",
            "query": "",
            "readiness": readiness,
            "synthesis": evidence_synthesis([], readiness),
            "cards": [boundary_card],
            "visible_cards": [boundary_card],
            "background_cards": [],
        }
    try:
        cards = fetch_pubmed_problem_sources(query, anatomy_phrase, mechanisms, red_flags)
        status = "PubMed retrieval completed" if cards else "PubMed returned no sources"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        cards = []
        status = f"PubMed retrieval unavailable: {exc.__class__.__name__}"

    cards.append(
        {
            "title": "Clinical context boundary",
            "level": "Safety rule",
            "summary": "Text symptoms can organize clinical concern but cannot diagnose fracture or exclude injury.",
            "limitation": "Physical examination, imaging, and clinician judgement remain required.",
            "pmid": "",
            "journal": "Internal safety policy",
            "year": "V1",
            "url": "",
            "supports": "Boundary for all Problem Description Mode outputs.",
            "match_score": 0,
            "match_label": "Safety boundary",
            "match_reasons": ["internal output boundary"],
        }
    )
    visible_cards = [
        card
        for card in cards
        if str(card.get("match_label", "")).startswith(("Focused", "Moderate")) or card.get("level") == "Safety rule"
    ]
    background_cards = [
        card
        for card in cards
        if str(card.get("match_label", "")).startswith(("Broad", "Excluded"))
    ]
    return {
        "status": status,
        "query": query,
        "readiness": readiness,
        "synthesis": evidence_synthesis(cards, readiness),
        "cards": cards,
        "visible_cards": visible_cards,
        "background_cards": background_cards,
    }


def deterministic_problem_llm_fallback(
    extracted_cards: List[Dict[str, str]],
    timeline: Dict[str, str],
    red_flags: List[Dict[str, str]],
    evidence_pack: Dict[str, Any],
    clinical_direction: Dict[str, Any],
) -> Dict[str, Any]:
    source_pool = evidence_pack.get("visible_cards") or evidence_pack.get("cards", [])
    source_pmids = [card.get("pmid") for card in source_pool if card.get("pmid")][:3]
    rag_ready = (evidence_pack.get("readiness") or {}).get("ready", False)
    red_flag_text = (
        "Red-flag language is present, so escalation review should take priority over routine interpretation."
        if red_flags
        else "No explicit red-flag language was detected in the submitted text, but absence in text does not exclude risk."
    )
    evidence_summary = (
        [
            "The PubMed/RAG layer retrieved source cards for general clinician context only; it does not determine diagnosis or management.",
            "Management decisions require examination findings, imaging review, local protocol, and clinician judgement.",
        ]
        if rag_ready
        else [
            "PubMed/RAG evidence retrieval was deferred because the input lacks enough structured clinical detail.",
            "Add missing anatomy, mechanism, symptom, timing, or function details before treating the evidence context as useful.",
        ]
    )
    return {
        "status": "deterministic_fallback",
        "llm_used": False,
        "clinician_context": (
            f"The submitted text is most compatible with {clinical_direction.get('most_compatible', 'a musculoskeletal injury pattern for review')}. "
            f"{(clinical_direction.get('action') or {}).get('summary', red_flag_text)}"
        ),
        "evidence_summary": evidence_summary,
        "safety_boundaries": [
            "Text-based clinician review context only.",
            "Not a treatment plan or patient-specific instruction.",
            "Professional assessment is needed for decisions.",
        ],
        "source_pmids": source_pmids,
        "limitations": [
            "No physical examination findings are available.",
            "No imaging is attached to this text-only workflow.",
            "Automated PubMed source selection must be verified by a clinician.",
        ],
    }


def problem_llm_prompt_packet(
    extracted_cards: List[Dict[str, str]],
    timeline: Dict[str, str],
    red_flags: List[Dict[str, str]],
    follow_up_questions: List[str],
    clinical_considerations: List[Dict[str, Any]],
    evidence_pack: Dict[str, Any],
    clinical_reasoning: Dict[str, Any],
    clinical_direction: Dict[str, Any],
) -> Dict[str, Any]:
    intake_map = {
        str(card.get("label")): str(card.get("value"))
        for card in extracted_cards
        if card.get("label") and card.get("value")
    }
    red_flag_compact = [
        {
            "label": item.get("label"),
            "trigger": item.get("trigger"),
        }
        for item in red_flags[:5]
    ]
    source_cards = []
    for card in (evidence_pack.get("visible_cards") or evidence_pack.get("cards", [])):
        source_cards.append(
            {
                "title": card.get("title"),
                "pmid": card.get("pmid"),
                "year": card.get("year"),
                "supports": card.get("supports"),
                "limitation": card.get("limitation"),
                "match": card.get("match_label"),
            }
        )
    schema_contract = {
        "clinician_context": "one cautious paragraph, maximum 45 words",
        "evidence_summary": ["2 short bullets, each grounded in source_cards"],
        "safety_boundaries": ["2 short boundaries; the first item must be exactly: Text-based clinician review context only."],
        "source_pmids": ["PMIDs copied only from source_cards"],
        "limitations": ["2 short limitations"],
    }
    return {
        "task": "Return one valid JSON object for clinician review.",
        "output_contract": {
            "format": "valid JSON only; no markdown fences; no prose before or after JSON",
            "exact_top_level_keys": list(schema_contract.keys()),
            "do_not_rename_keys": True,
            "forbidden_key_examples": ["clinical_context", "assessment", "treatment_plan"],
            "schema": schema_contract,
            "length_limits": {
                "clinician_context_words": 45,
                "evidence_summary_items": 2,
                "evidence_summary_words_each": 16,
                "safety_boundaries_items": 2,
                "limitations_items": 2,
            },
        },
        "scope": (
            "Use this writer only after the deterministic intake gate has accepted a musculoskeletal "
            "injury or symptom description. Do not broaden the task beyond that accepted scope."
        ),
        "desired_output_style": "Short, clinician-facing, evidence-linked, conservative, no filler.",
        "hard_rules": [
            "Do not diagnose.",
            "Do not claim fracture is present or absent.",
            "Do not use the phrase 'rule out'; write that text alone cannot determine whether fracture or other injury is present.",
            "Do not infer a fracture type.",
            "Do not infer laterality unless explicitly stated.",
            "Do not convert denied symptoms into present symptoms.",
            "Do not treat source titles as patient-specific evidence.",
            "Do not prescribe medication, immobilisation, surgery, discharge, or follow-up.",
            "Do not provide emergency instructions beyond recommending urgent clinician review when red flags are present.",
            "Do not answer non-musculoskeletal, technical, legal, financial, or general chat requests.",
            "Use only the structured intake and source cards provided.",
            "Return JSON only with the exact output_contract keys.",
            "Every source_pmids value must be copied from source_cards.",
            "Keep the answer short enough to finish the JSON object.",
            "Do not change the deterministic clinical_direction or action level.",
        ],
        "structured_intake": {
            "primary_injury_site": intake_map.get("Primary injury site", "not specified"),
            "other_anatomy_mentioned": intake_map.get("Other anatomy mentioned", "none stated"),
            "mechanism": timeline.get("mechanism", "not specified"),
            "symptoms_or_function": timeline.get("symptoms", "not specified"),
            "timing": timeline.get("timing", "not specified"),
            "denied_symptoms": intake_map.get("Symptoms specifically denied", "none stated"),
            "red_flags": red_flag_compact,
            "reasoning_summary": clinical_reasoning.get("summary"),
        },
        "clinical_direction": {
            "most_compatible": clinical_direction.get("most_compatible"),
            "action_level": (clinical_direction.get("action") or {}).get("level"),
            "action_title": (clinical_direction.get("action") or {}).get("title"),
            "recommended_next_step": (clinical_direction.get("action") or {}).get("summary"),
            "watch_for": clinical_direction.get("watch_for", [])[:5],
            "uncertainty": clinical_direction.get("uncertainty", [])[:3],
        },
        "pubmed_rag": {
            "retrieval_status": evidence_pack.get("status"),
            "query": evidence_pack.get("query"),
            "source_cards": source_cards[:2],
        },
        "required_json_schema": schema_contract,
    }


def normalise_problem_llm_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(candidate)
    if "clinician_context" not in normalized and "clinical_context" in normalized:
        normalized["clinician_context"] = normalized.pop("clinical_context")

    repairs = []

    if isinstance(normalized.get("clinician_context"), list):
        normalized["clinician_context"] = " ".join(str(item) for item in normalized["clinician_context"] if item)
        repairs.append("joined_clinician_context_list")
    for list_key in ["evidence_summary", "safety_boundaries", "limitations"]:
        value = normalized.get(list_key)
        if isinstance(value, str) and value.strip():
            normalized[list_key] = [value.strip()]
            repairs.append(f"coerced_{list_key}_string_to_list")
    source_pmids = normalized.get("source_pmids")
    if isinstance(source_pmids, str):
        normalized["source_pmids"] = [item.strip() for item in re.split(r"[,;]", source_pmids) if item.strip()]
        repairs.append("coerced_source_pmids_string_to_list")
    elif isinstance(source_pmids, list):
        normalized["source_pmids"] = [str(item).strip() for item in source_pmids if str(item).strip()]

    replacements = [
        (
            re.compile(r"scaphoid fractures are a possibility[^.]*\.", re.IGNORECASE),
            "Scaphoid-region injury is a clinical consideration for review, but it cannot be confirmed from text alone.",
            "replaced_diagnostic_possibility_wording",
        ),
        (
            re.compile(r"\bfractures are a possibility\b", re.IGNORECASE),
            "bone injury remains a review consideration",
            "replaced_generic_fracture_possibility_wording",
        ),
        (
            re.compile(r"\bfracture is possible\b", re.IGNORECASE),
            "bone injury remains a review consideration",
            "replaced_fracture_possible_wording",
        ),
        (
            re.compile(r"further imaging is warranted[^.]*\.", re.IGNORECASE),
            "Clinician-led assessment may determine whether imaging is appropriate.",
            "replaced_imaging_directive_wording",
        ),
        (
            re.compile(r"further imaging is needed to rule out fracture\.", re.IGNORECASE),
            "Text alone cannot determine whether fracture or other injury is present.",
            "replaced_rule_out_wording",
        ),
        (
            re.compile(r"rule out (a )?fracture", re.IGNORECASE),
            "evaluate whether fracture or other injury is present",
            "replaced_rule_out_phrase",
        ),
        (
            re.compile(r"rule out serious injury", re.IGNORECASE),
            "evaluate whether serious injury is present",
            "replaced_rule_out_phrase",
        ),
        (
            re.compile(r"radiographic imaging is essential for diagnosing wrist fractures\.", re.IGNORECASE),
            "Radiographic imaging is discussed in the wrist-fracture assessment literature.",
            "replaced_imaging_certainty_wording",
        ),
        (
            re.compile(r"\bimaging is warranted\b", re.IGNORECASE),
            "clinician-led assessment may determine whether imaging is appropriate",
            "replaced_imaging_warranted_phrase",
        ),
        (
            re.compile(r"\bimaging is needed\b", re.IGNORECASE),
            "clinician-led assessment may determine whether imaging is appropriate",
            "replaced_imaging_needed_phrase",
        ),
        (
            re.compile(r"management of scaphoid fractures may involve immobilization and/or surgical intervention\.", re.IGNORECASE),
            "The retrieved sources discuss management pathways, but this text-only module does not choose one.",
            "replaced_treatment_pathway_wording",
        ),
    ]

    def repair_text(value: Any) -> Any:
        if isinstance(value, str):
            repaired = value
            for pattern, replacement, repair_id in replacements:
                repaired, count = pattern.subn(replacement, repaired)
                if count:
                    repairs.append(repair_id)
            return repaired
        if isinstance(value, list):
            return [repair_text(item) for item in value]
        if isinstance(value, dict):
            return {key: repair_text(item) for key, item in value.items()}
        return value

    normalized = repair_text(normalized)

    boundaries = normalized.get("safety_boundaries")
    if not isinstance(boundaries, list):
        boundaries = []
    boundary_text = " ".join(str(item) for item in boundaries).lower()
    if "text-based" not in boundary_text and "clinician review context" not in boundary_text:
        boundaries.insert(0, "Text-based clinician review context only.")
        repairs.append("added_text_based_context_boundary")
    if "human" not in boundary_text and "clinician" not in boundary_text:
        boundaries.append("Human clinician review is required.")
        repairs.append("added_human_review_boundary")
    normalized["safety_boundaries"] = boundaries[:4]
    if repairs:
        normalized["guardrail_repairs"] = repairs
    return normalized


def problem_llm_forbidden_terms() -> List[str]:
    return [
        "no evidence of fracture",
        "rule out a fracture",
        "rule out fracture",
        "rule out serious injury",
        "confirmed fracture",
        "fracture ruled out",
        "fracture is ruled out",
        "you have a fracture",
        "you do not have a fracture",
        "no fracture",
        "not fractured",
        "fracture is present",
        "fracture is absent",
        "fractures are a possibility",
        "fracture type",
        "diagnosis is",
        "likely diagnosis",
        "imaging is warranted",
        "imaging is needed",
        "definitely",
        "certainly",
        "clearly shows",
        "normal x-ray",
        "negative x-ray",
        "must take",
        "take ibuprofen",
        "take paracetamol",
        "apply a cast",
        "put a cast",
        "needs surgery",
        "surgical intervention",
        "surgical fixation is required",
        "safe to ignore",
        "no doctor needed",
        "all clear",
        "return to play",
        "weight bear as tolerated",
    ]


def validate_problem_llm_output(candidate: Dict[str, Any], allowed_pmids: List[str]) -> Dict[str, Any]:
    errors = []
    for key in ["clinician_context", "evidence_summary", "safety_boundaries", "source_pmids", "limitations"]:
        if key not in candidate:
            errors.append(f"missing_{key}")
    for key in ["evidence_summary", "safety_boundaries", "source_pmids", "limitations"]:
        if key in candidate and not isinstance(candidate.get(key), list):
            errors.append(f"{key}_not_list")
    text_blob = json.dumps(candidate, ensure_ascii=False).lower()
    for term in problem_llm_forbidden_terms():
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text_blob):
            errors.append(f"forbidden_term:{term}")
    used_pmids = [str(pmid) for pmid in candidate.get("source_pmids", []) if pmid]
    for pmid in used_pmids:
        if pmid not in allowed_pmids:
            errors.append(f"unknown_pmid:{pmid}")
    if "text-based" not in text_blob and "clinician review context" not in text_blob:
        errors.append("missing_text_based_context_boundary")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "allowed_pmids": allowed_pmids,
        "used_pmids": used_pmids,
    }


def problem_llm_display_metadata(result: Dict[str, Any]) -> Dict[str, str]:
    status = str(result.get("status") or "")
    validation = (result.get("validation") or {}).get("verdict") or "not run"
    if status == "PASS" and result.get("llm_called") is True:
        return {
            "label": "Validated LLM + PubMed summary",
            "note": "A local LLM generated this section from structured intake and PubMed source cards, then passed validation.",
            "technical": f"{status}; validator {validation}",
        }
    if status == "SKIPPED_LLM_DISABLED":
        return {
            "label": "Controlled RAG summary",
            "note": "The structured intake and PubMed/RAG packet were summarized by a constrained safety writer.",
            "technical": f"{status}; validator {validation}",
        }
    if status.startswith("SKIPPED"):
        return {
            "label": "Controlled RAG summary",
            "note": "The optional LLM writer was not used, so the app displayed a validated deterministic summary from the same structured RAG packet.",
            "technical": f"{status}; validator {validation}",
        }
    if status.startswith("FAIL"):
        if "TIMEOUT" in status:
            return {
                "label": "Safe fallback used after slow LLM",
                "note": "The local LLM took too long, so the app displayed the validated deterministic summary instead of waiting indefinitely.",
                "technical": f"{status}; validator {validation}",
            }
        return {
            "label": "Safe fallback used",
            "note": "The LLM writer did not produce acceptable output, so the app rejected it and displayed the deterministic safety summary.",
            "technical": f"{status}; validator {validation}",
        }
    return {
        "label": "Controlled RAG summary",
        "note": "The app displayed a constrained summary from structured intake and PubMed source cards.",
        "technical": f"{status or 'unknown'}; validator {validation}",
    }


def _run_problem_llm_writer_inline(
    extracted_cards: List[Dict[str, str]],
    timeline: Dict[str, str],
    red_flags: List[Dict[str, str]],
    follow_up_questions: List[str],
    clinical_considerations: List[Dict[str, Any]],
    evidence_pack: Dict[str, Any],
    clinical_reasoning: Dict[str, Any],
    clinical_direction: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = deterministic_problem_llm_fallback(extracted_cards, timeline, red_flags, evidence_pack, clinical_direction)
    llm_source_cards = evidence_pack.get("visible_cards") or evidence_pack.get("cards", [])
    allowed_pmids = [str(card.get("pmid")) for card in llm_source_cards if card.get("pmid")]
    if not PROBLEM_LLM_ENABLED:
        result = {
            "status": "SKIPPED_LLM_DISABLED",
            "llm_called": False,
            "llm_attempted": False,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": []},
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result
    if not allowed_pmids:
        result = {
            "status": "SKIPPED_NO_PUBMED_SOURCES",
            "llm_called": False,
            "llm_attempted": False,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": []},
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result

    started = datetime.now()
    try:
        from stage3.local_llm_writer import (
            LocalLlmWriterConfig,
            _build_generation_inputs,
            _load_hf_text_model,
            dependency_report,
            extract_json_object,
            model_path_status,
        )

        deps = dependency_report()
        model_status = model_path_status(PROBLEM_LLM_MODEL_ID)
        if not deps.get("torch") or not deps.get("transformers") or not model_status.get("config_json_exists"):
            result = {
                "status": "SKIPPED_MODEL_UNAVAILABLE",
                "llm_called": False,
                "llm_attempted": False,
                "dependency_report": deps,
                "model_path_status": model_status,
                "display": fallback,
                "validation": {"verdict": "PASS", "errors": []},
            }
            result["display_metadata"] = problem_llm_display_metadata(result)
            return result
        cfg = LocalLlmWriterConfig(
            model_id_or_path=PROBLEM_LLM_MODEL_ID,
            local_files_only=True,
            max_new_tokens=PROBLEM_LLM_MAX_NEW_TOKENS,
            temperature=0.0,
            top_p=1.0,
            run_model=True,
            prompt_mode="compact",
            attn_implementation="eager",
        )
        packet = problem_llm_prompt_packet(
            extracted_cards,
            timeline,
            red_flags,
            follow_up_questions,
            clinical_considerations,
            evidence_pack,
            clinical_reasoning,
            clinical_direction,
        )
        prompt_contract = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a controlled medical writing assistant for an academic musculoskeletal clinical-support prototype. "
                        "Do not make final clinical decisions, exclude fracture, prescribe, or invent clinical facts. "
                        "Your role is to turn accepted structured intake, deterministic clinical direction, and PubMed source cards into cautious clinician-review context. "
                        "Return one valid JSON object only, with no markdown fences and no commentary."
                    ),
                },
                {
                    "role": "developer",
                    "content": (
                        "Use only the provided JSON packet. Do not create new medical claims. "
                        "Preserve negations and missing information exactly. "
                        "If a symptom is denied, do not list it as present or as a red flag. "
                        "Do not provide treatment instructions. Copy PMIDs only from source_cards. "
                        "Do not use the phrase 'rule out'. "
                        "Do not change the provided clinical_direction action level. "
                        "Use exactly these top-level keys: clinician_context, evidence_summary, safety_boundaries, source_pmids, limitations. "
                        "Do not rename clinician_context to clinical_context."
                    ),
                },
                {"role": "user", "content": packet},
            ]
        }
        loaded = _load_hf_text_model(cfg)
        model = loaded["model"]
        processor = loaded.get("processor")
        tokenizer = loaded.get("tokenizer")
        inputs = _build_generation_inputs(prompt_contract, loaded)
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        import torch

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                max_time=PROBLEM_LLM_MAX_SECONDS,
                do_sample=False,
                pad_token_id=(
                    getattr(tokenizer, "eos_token_id", None)
                    or getattr(processor, "eos_token_id", None)
                    or getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                ),
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids[0]
        decoder = tokenizer or processor
        raw_text = decoder.decode(generated_ids, skip_special_tokens=True)
        try:
            parsed = extract_json_object(raw_text)
        except ValueError as exc:
            elapsed = round((datetime.now() - started).total_seconds(), 3)
            timed_out = elapsed >= max(PROBLEM_LLM_MAX_SECONDS - 2, 1)
            result = {
                "status": "FAIL_LLM_TIMEOUT_FALLBACK_USED" if timed_out else "FAIL_JSON_PARSE_FALLBACK_USED",
                "llm_called": True,
                "llm_attempted": True,
                "raw_preview": raw_text[:700],
                "display": fallback,
                "validation": {"verdict": "FAIL", "errors": [f"json_parse:{exc}"]},
                "elapsed_seconds": elapsed,
                "timeout_seconds": PROBLEM_LLM_MAX_SECONDS,
            }
            result["display_metadata"] = problem_llm_display_metadata(result)
            return result
        parsed["status"] = parsed.get("status") or "llm_generated_candidate"
        parsed["llm_used"] = True
        parsed = normalise_problem_llm_candidate(parsed)
        validation = validate_problem_llm_output(parsed, allowed_pmids)
        elapsed = round((datetime.now() - started).total_seconds(), 3)
        if validation["verdict"] != "PASS":
            result = {
                "status": "FAIL_VALIDATION_FALLBACK_USED",
                "llm_called": True,
                "llm_attempted": True,
                "raw_preview": raw_text[:500],
                "candidate": parsed,
                "display": fallback,
                "validation": validation,
                "elapsed_seconds": elapsed,
                "timeout_seconds": PROBLEM_LLM_MAX_SECONDS,
            }
            result["display_metadata"] = problem_llm_display_metadata(result)
            return result
        result = {
            "status": "PASS",
            "llm_called": True,
            "llm_attempted": True,
            "display": parsed,
            "validation": validation,
            "elapsed_seconds": elapsed,
            "timeout_seconds": PROBLEM_LLM_MAX_SECONDS,
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result
    except Exception as exc:
        result = {
            "status": f"FAIL_LLM_EXCEPTION_{exc.__class__.__name__}",
            "llm_called": False,
            "llm_attempted": True,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": []},
            "error_message": str(exc)[:240],
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result


def run_problem_llm_writer(
    extracted_cards: List[Dict[str, str]],
    timeline: Dict[str, str],
    red_flags: List[Dict[str, str]],
    follow_up_questions: List[str],
    clinical_considerations: List[Dict[str, Any]],
    evidence_pack: Dict[str, Any],
    clinical_reasoning: Dict[str, Any],
    clinical_direction: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = deterministic_problem_llm_fallback(extracted_cards, timeline, red_flags, evidence_pack, clinical_direction)
    llm_source_cards = evidence_pack.get("visible_cards") or evidence_pack.get("cards", [])
    allowed_pmids = [str(card.get("pmid")) for card in llm_source_cards if card.get("pmid")]
    if not PROBLEM_LLM_ISOLATED or not PROBLEM_LLM_ENABLED or not allowed_pmids:
        return _run_problem_llm_writer_inline(
            extracted_cards,
            timeline,
            red_flags,
            follow_up_questions,
            clinical_considerations,
            evidence_pack,
            clinical_reasoning,
            clinical_direction,
        )

    PROBLEM_LLM_JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex
    input_path = PROBLEM_LLM_JOB_DIR / f"{job_id}.input.json"
    output_path = PROBLEM_LLM_JOB_DIR / f"{job_id}.output.json"
    err_path = PROBLEM_LLM_JOB_DIR / f"{job_id}.stderr.log"
    payload = {
        "extracted_cards": extracted_cards,
        "timeline": timeline,
        "red_flags": red_flags,
        "follow_up_questions": follow_up_questions,
        "clinical_considerations": clinical_considerations,
        "evidence_pack": evidence_pack,
        "clinical_reasoning": clinical_reasoning,
        "clinical_direction": clinical_direction,
    }
    input_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    env = os.environ.copy()
    env["PROBLEM_DESCRIPTION_LLM_ENABLED"] = "1"
    env["PROBLEM_DESCRIPTION_LLM_MODEL"] = PROBLEM_LLM_MODEL_ID
    env["PROBLEM_DESCRIPTION_LLM_MAX_NEW_TOKENS"] = str(PROBLEM_LLM_MAX_NEW_TOKENS)
    env["PROBLEM_DESCRIPTION_LLM_MAX_SECONDS"] = str(PROBLEM_LLM_MAX_SECONDS)
    env["PROBLEM_DESCRIPTION_LLM_ISOLATED"] = "0"
    timeout_seconds = max(PROBLEM_LLM_MAX_SECONDS + 120, 240)
    started = datetime.now()
    try:
        with err_path.open("w", encoding="utf-8") as err:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "web_app.problem_llm_worker",
                    str(input_path),
                    str(output_path),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=err,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired:
        elapsed = round((datetime.now() - started).total_seconds(), 3)
        result = {
            "status": "FAIL_LLM_PROCESS_TIMEOUT_FALLBACK_USED",
            "llm_called": True,
            "llm_attempted": True,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": ["isolated_llm_timeout"]},
            "elapsed_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result
    except Exception as exc:
        result = {
            "status": f"FAIL_LLM_PROCESS_EXCEPTION_{exc.__class__.__name__}",
            "llm_called": False,
            "llm_attempted": True,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": []},
            "error_message": str(exc)[:240],
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result

    if completed.returncode != 0 or not output_path.exists():
        error_preview = err_path.read_text(encoding="utf-8", errors="replace")[-900:] if err_path.exists() else ""
        result = {
            "status": "FAIL_LLM_PROCESS_EXIT_FALLBACK_USED",
            "llm_called": False,
            "llm_attempted": True,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": ["isolated_llm_process_failed"]},
            "error_message": error_preview[:240],
            "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result

    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result = {
            "status": f"FAIL_LLM_PROCESS_OUTPUT_{exc.__class__.__name__}",
            "llm_called": False,
            "llm_attempted": True,
            "display": fallback,
            "validation": {"verdict": "PASS", "errors": ["isolated_llm_output_unreadable"]},
            "error_message": str(exc)[:240],
        }
        result["display_metadata"] = problem_llm_display_metadata(result)
        return result
    result.setdefault("isolated_worker", True)
    result.setdefault("process_elapsed_seconds", round((datetime.now() - started).total_seconds(), 3))
    return result


def analyze_problem_description(description: str) -> Dict[str, Any]:
    raw = (description or "").strip()
    text = raw.lower()
    clinical_anchor_terms = ["pain", "hurt", "injury", "fall", "swelling", "xray", "x-ray", "trauma", "πονο", "πόνο", "χτύπ", "χτυπ", "πτώση", "έπεσα", "επέσα", "πρηξ"]
    clinical_anchor_terms = [term for term in clinical_anchor_terms if term != "fall"]
    clinical_anchor_terms.extend(["radiograph", "fracture", "bone", "joint"])
    anatomy_context = classify_anatomy_context(text, ANATOMY_KEYWORDS)
    anatomy = list(anatomy_context["present"] or [])
    adjacent_anatomy = list(anatomy_context["adjacent_or_functional"] or [])
    denied_anatomy = list(anatomy_context["denied"] or [])
    mechanisms = matched_labels(text, MECHANISM_KEYWORDS)
    symptom_matches = matched_symptom_labels(text)
    symptoms = symptom_matches["present"]
    denied_symptoms = symptom_matches["denied"]
    has_time = contains_any(text, TIME_KEYWORDS)
    input_quality = build_input_quality(text, anatomy, mechanisms, symptoms, denied_symptoms, has_time)

    if len(raw) < 25:
        return {
            "status": "rejected",
            "rejection_kind": "insufficient_detail",
            "headline": "More clinical detail is needed",
            "message": "No clinical analysis was performed because the submitted text is too short to review safely.",
            "example": "Example: I fell while playing basketball, my wrist is swollen, and I cannot grip objects.",
            "input_quality": input_quality,
        }
    has_nonclinical_context = contains_any(text, IRRELEVANT_HINTS)
    has_clinical_anchor = bool(anatomy or symptoms or contains_positive_any(text, clinical_anchor_terms))
    enough_core_clinical_detail = bool(anatomy and symptoms and input_quality.get("score", 0) >= 5)
    weak_fracture_only_query = ("fracture" in text or "καταγμα" in text) and not (anatomy or mechanisms or symptoms)
    if has_nonclinical_context and (weak_fracture_only_query or not enough_core_clinical_detail):
        return {
            "status": "rejected",
            "rejection_kind": "out_of_scope",
            "headline": "This description cannot be analyzed",
            "message": "No clinical analysis was performed because the text does not describe a patient musculoskeletal injury or X-ray review task.",
            "example": "Please try again with a patient description that includes body area, specific location, mechanism, symptoms, timing, and current function.",
            "input_quality": input_quality,
        }
    if weak_fracture_only_query or not has_clinical_anchor or not enough_core_clinical_detail:
        return {
            "status": "rejected",
            "rejection_kind": "insufficient_clinical_context",
            "headline": "More musculoskeletal context is needed",
            "message": "No clinical analysis was performed because the description does not include enough usable patient-specific detail.",
            "example": "Include the body area, specific location, mechanism, symptoms, timing, and what the patient can or cannot do now.",
            "input_quality": input_quality,
        }

    red_flags = []
    for rule in RED_FLAG_RULES:
        triggers = positive_keyword_matches(text, rule["keywords"])
        if triggers:
            red_flags.append(
                {
                    "label": rule["label"],
                    "message": rule["message"],
                    "triggers": triggers,
                    "why_triggered": f"Mentioned phrase(s): {', '.join(triggers)}.",
                }
            )
    extraction_evidence = build_extraction_evidence(text, anatomy_context, mechanisms, symptoms, denied_symptoms, red_flags)
    escalation_message = ""
    if len(red_flags) >= 2:
        escalation_message = (
            "Several warning signs are described. Urgent human clinical assessment should come before any background evidence discussion."
        )
    completeness = evidence_completeness(anatomy or adjacent_anatomy, mechanisms, symptoms, red_flags, has_time)
    primary_anatomy = anatomy_context["primary"]
    other_anatomy = [label for label in anatomy if label != primary_anatomy] + [label for label in adjacent_anatomy if label != primary_anatomy]
    evidence_pack = problem_evidence_cards(primary_anatomy, mechanisms, symptoms, red_flags, input_quality)
    extracted_cards = [
        {"label": "Primary injury site", "value": primary_anatomy if primary_anatomy else "Not clearly stated"},
        {"label": "Other anatomy mentioned", "value": ", ".join(other_anatomy) if other_anatomy else "None clearly stated"},
        {"label": "Anatomy explicitly denied", "value": ", ".join(denied_anatomy) if denied_anatomy else "None clearly stated"},
        {"label": "Mechanism", "value": ", ".join(mechanisms) if mechanisms else "Not clearly stated"},
        {"label": "Symptoms / function", "value": ", ".join(symptoms) if symptoms else "Not clearly stated"},
        {"label": "Symptoms specifically denied", "value": ", ".join(denied_symptoms) if denied_symptoms else "None clearly stated"},
        {"label": "Timing", "value": "Mentioned" if has_time else "Not stated"},
    ]
    timeline = {
        "mechanism": ", ".join(mechanisms) if mechanisms else "Mechanism not clearly stated.",
        "symptoms": ", ".join(symptoms) if symptoms else "Symptoms not clearly stated.",
        "time": "Timing mentioned." if has_time else "Timing not stated.",
        "clinical_concern": "Escalation review" if red_flags else ("Focused clinical review" if primary_anatomy and symptoms else "Incomplete information"),
    }
    follow_up = suggested_questions(text, anatomy or adjacent_anatomy, mechanisms, symptoms, denied_symptoms, has_time, input_quality, red_flags)
    considerations = build_clinical_considerations(primary_anatomy, anatomy or adjacent_anatomy, mechanisms, symptoms, red_flags)
    clinical_direction = build_clinical_direction(primary_anatomy, other_anatomy, mechanisms, symptoms, denied_symptoms, red_flags, considerations, input_quality)
    if (clinical_direction.get("action") or {}).get("level") == "Level 4":
        follow_up = urgent_non_delaying_questions(primary_anatomy, red_flags)
    elif not follow_up:
        follow_up = refinement_questions(primary_anatomy, (clinical_direction.get("action") or {}).get("level", ""))
    consistency = clinical_consistency_gate(primary_anatomy, anatomy_context, mechanisms, symptoms, denied_symptoms, red_flags, clinical_direction)
    if not consistency["ready_for_rendering"]:
        clinical_direction = withheld_clinical_direction(consistency)
    follow_up_display = follow_up_display_config(follow_up, (clinical_direction.get("action") or {}).get("level", ""))
    clinical_reasoning = clinical_reasoning_summary([primary_anatomy] if primary_anatomy else [], mechanisms, symptoms, denied_symptoms, red_flags, evidence_pack)
    extraction_confidence_items = extraction_confidence(anatomy or adjacent_anatomy, mechanisms, symptoms, denied_symptoms, has_time)
    if consistency["ready_for_rendering"]:
        llm_context = run_problem_llm_writer(
            extracted_cards,
            timeline,
            red_flags,
            follow_up,
            considerations,
            evidence_pack,
            clinical_reasoning,
            clinical_direction,
        )
    else:
        display = deterministic_problem_llm_fallback(extracted_cards, timeline, red_flags, evidence_pack, clinical_direction)
        llm_context = {
            "status": "SKIPPED_CONSISTENCY_FAILED",
            "llm_called": False,
            "llm_attempted": False,
            "display": display,
            "validation": {"verdict": "PASS", "errors": consistency["issues"]},
        }
        llm_context["display_metadata"] = problem_llm_display_metadata(llm_context)
    return {
        "status": "accepted",
        "headline": (clinical_direction.get("headline") or "Clinical review support ready"),
        "summary": (
            f"Primary review focus: {clinical_direction.get('most_compatible', 'structured musculoskeletal review context')}. "
            f"Recommended next step: {(clinical_direction.get('action') or {}).get('title', 'clinician review context is available')}."
        ),
        "extracted_cards": extracted_cards,
        "timeline": timeline,
        "red_flags": red_flags,
        "red_flag_escalation_message": escalation_message,
        "primary_anatomy": primary_anatomy,
        "adjacent_or_functional_anatomy": other_anatomy,
        "explicitly_denied_anatomy": denied_anatomy,
        "follow_up_questions": follow_up,
        "follow_up_display": follow_up_display,
        "clinical_considerations": considerations,
        "clinical_direction": clinical_direction,
        "clinical_consistency": consistency,
        "extraction_evidence": extraction_evidence,
        "clinical_reasoning": clinical_reasoning,
        "extraction_confidence": extraction_confidence_items,
        "evidence_completeness": completeness,
        "input_quality": input_quality,
        "evidence_pack": evidence_pack,
        "evidence_cards": evidence_pack.get("visible_cards") or evidence_pack["cards"],
        "main_evidence_cards": (evidence_pack.get("visible_cards") or evidence_pack["cards"])[:3],
        "additional_evidence_cards": (evidence_pack.get("visible_cards") or evidence_pack["cards"])[3:],
        "background_evidence_cards": evidence_pack.get("background_cards", []),
        "llm_context": llm_context,
        "explainability": [
            "The text was screened to confirm it describes a musculoskeletal health problem.",
            "Body-region, mechanism, symptom, timing, and red-flag keywords were extracted into structured fields.",
            "Missing information was converted into follow-up questions instead of being guessed.",
            "Clinical considerations were generated as review prompts, not as diagnoses.",
            "A controlled PubMed query was built from the structured fields and used to retrieve source cards where available.",
            "The LLM writer is allowed to summarize only the structured intake and PubMed source cards; invalid output falls back to deterministic text.",
        ],
    }


def validate_uploaded_xray(data: bytes, filename: str, content_type: str | None) -> Dict[str, Any]:
    blocking: List[str] = []
    warnings: List[str] = []
    limited_quality_flags: List[str] = []
    metrics: Dict[str, Any] = {
        "filename": filename or "uploaded image",
        "content_type": content_type or "unknown",
        "file_size_mb": round(len(data) / (1024 * 1024), 2),
    }

    suffix = Path(filename or "").suffix.lower()
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    allowed_content_types = {"image/jpeg", "image/png", "application/octet-stream", None, ""}

    if not data:
        return upload_result("rejected", "No image received", "Upload an X-ray image file before starting preflight.")
    if len(data) > MAX_UPLOAD_BYTES:
        blocking.append(f"File is too large ({metrics['file_size_mb']} MB). Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if suffix and suffix not in allowed_suffixes:
        blocking.append("Unsupported file type. Please upload PNG or JPEG.")
    if content_type not in allowed_content_types:
        warnings.append(f"Browser reported content type '{content_type}'. The file will still be checked by image decoding.")

    try:
        image = Image.open(BytesIO(data))
        if getattr(image, "is_animated", False):
            blocking.append("Animated or multi-frame images are not supported.")
        image = ImageOps.exif_transpose(image)
        original_mode = image.mode
        rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return upload_result(
            "rejected",
            "Unreadable image",
            "The uploaded file could not be decoded as a radiograph image.",
            metrics=metrics,
            blocking=["The file is not a readable PNG/JPEG image."],
            warnings=warnings,
        )

    width, height = rgb.size
    short_side = min(width, height)
    long_side = max(width, height)
    aspect_ratio = round(long_side / max(short_side, 1), 2)
    metrics.update(
        {
            "width_px": width,
            "height_px": height,
            "short_side_px": short_side,
            "long_side_px": long_side,
            "aspect_ratio": aspect_ratio,
            "color_mode": original_mode,
        }
    )

    if short_side < MIN_SHORT_SIDE:
        blocking.append(f"Image resolution is too low: short side is {short_side}px. Minimum required is {MIN_SHORT_SIDE}px.")
    elif short_side < RECOMMENDED_SHORT_SIDE:
        warnings.append(f"Image is acceptable but below the preferred resolution: short side is {short_side}px; {RECOMMENDED_SHORT_SIDE}px+ is preferred.")
    if long_side > MAX_LONG_SIDE:
        blocking.append(f"Image is extremely large: long side is {long_side}px. Please upload a smaller exported image.")
    if aspect_ratio > MAX_ASPECT_RATIO:
        blocking.append("Image aspect ratio is extreme. This may indicate a cropped strip, screenshot artifact, or wrong export.")
    elif aspect_ratio > WARN_ASPECT_RATIO:
        warnings.append("Image aspect ratio is unusual. Confirm that the radiograph is not severely cropped.")
    if width > height * 1.35:
        warnings.append("Image is landscape-oriented. The live pipeline may try orientation rescue, but confirm that the radiograph was exported correctly.")
    if original_mode not in {"L", "I", "I;16", "RGB"}:
        warnings.append(f"Image mode '{original_mode}' was converted for review. Prefer a clean grayscale PNG/JPEG export.")

    gray = np.array(rgb.convert("L"))
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    hsv = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    rg = rgb_arr[:, :, 0] - rgb_arr[:, :, 1]
    yb = 0.5 * (rgb_arr[:, :, 0] + rgb_arr[:, :, 1]) - rgb_arr[:, :, 2]
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    channel_spread = float(np.mean(np.max(rgb_arr, axis=2) - np.min(rgb_arr, axis=2)))
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    dark_pixel_ratio = float((gray < 20).mean())
    bright_pixel_ratio = float((gray > 245).mean())
    p1, p99 = np.percentile(gray, [1, 99])
    dynamic_range = float(p99 - p1)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics.update(
        {
            "mean_brightness": round(mean_brightness, 1),
            "contrast_std": round(contrast_std, 1),
            "dark_pixel_ratio": round(dark_pixel_ratio, 3),
            "bright_pixel_ratio": round(bright_pixel_ratio, 3),
            "dynamic_range_p1_p99": round(dynamic_range, 1),
            "sharpness_laplacian_var": round(laplacian_var, 1),
            "mean_color_saturation": round(mean_saturation, 1),
            "colorfulness_score": round(colorfulness, 1),
            "mean_rgb_channel_spread": round(channel_spread, 1),
        }
    )

    non_radiograph_like = False
    if mean_saturation > MAX_MEAN_SATURATION and colorfulness > MAX_COLORFULNESS:
        non_radiograph_like = True
        blocking.append(
            "Image appears to be a color/natural photograph rather than a clean radiograph export."
        )
    elif mean_saturation > WARN_MEAN_SATURATION or colorfulness > WARN_COLORFULNESS:
        warnings.append(
            "Image has noticeable color content. Prefer a clean grayscale radiograph export."
        )
    if channel_spread > 35:
        non_radiograph_like = True
        blocking.append(
            "RGB channels differ strongly. This is unusual for an X-ray and may indicate a non-radiograph photo or processed screenshot."
        )
    elif channel_spread > 18:
        warnings.append(
            "RGB channel differences are noticeable. Confirm this is an unmodified radiograph image."
        )

    visible_structure_too_weak = contrast_std < MIN_VISIBLE_CONTRAST_STD or dynamic_range < MIN_VISIBLE_DYNAMIC_RANGE
    extreme_exposure = mean_brightness < VERY_DARK_MEAN_BRIGHTNESS or mean_brightness > VERY_BRIGHT_MEAN_BRIGHTNESS
    washed_page_like = mean_brightness > 230 and contrast_std < 18 and dynamic_range < 60
    radiograph_like_grayscale = (
        not non_radiograph_like
        and mean_saturation <= WARN_MEAN_SATURATION
        and colorfulness <= WARN_COLORFULNESS
        and channel_spread <= 18
    )

    if washed_page_like:
        blocking.append(
            "Image looks like a washed-out screenshot or document page rather than a usable radiograph."
        )
    if mean_brightness < VERY_DARK_MEAN_BRIGHTNESS:
        blocking.append("Image is too dark for reliable review.")
    elif mean_brightness < 35:
        limited_quality_flags.append("dark_radiograph")
        warnings.append("Image appears dark. AI review can continue, but subtle findings may be missed.")
    elif mean_brightness < 45:
        warnings.append("Image appears dark. Prefer a clearer export if available.")
    if mean_brightness > VERY_BRIGHT_MEAN_BRIGHTNESS:
        blocking.append("Image is overexposed/washed out for reliable review.")
    elif mean_brightness > 225:
        limited_quality_flags.append("bright_radiograph")
        warnings.append("Image appears bright/washed. AI review can continue, but bone detail may be incomplete.")
    elif mean_brightness > 210:
        warnings.append("Image appears very bright. Confirm that bone detail is visible.")
    if visible_structure_too_weak and (non_radiograph_like or extreme_exposure):
        blocking.append("Image does not show enough visible radiographic structure for AI review.")
    elif contrast_std < LIMITED_QUALITY_CONTRAST_STD:
        limited_quality_flags.append("low_contrast_radiograph")
        warnings.append("Image contrast is low. AI review can continue as limited-quality output.")
    elif contrast_std < 28:
        warnings.append("Image contrast is limited. A higher-quality export is preferred.")
    if dynamic_range < LIMITED_QUALITY_DYNAMIC_RANGE:
        if visible_structure_too_weak and (non_radiograph_like or extreme_exposure):
            blocking.append("Image tonal range is too narrow for reliable AI review.")
        else:
            limited_quality_flags.append("narrow_tonal_range")
            warnings.append("Image tonal range is narrow. AI review can continue with limited-quality caution.")
    elif dynamic_range < 90:
        warnings.append("Image tonal range is limited; subtle fractures may be harder to review.")
    if laplacian_var < LIMITED_QUALITY_SHARPNESS:
        if visible_structure_too_weak and not radiograph_like_grayscale:
            blocking.append("Image appears severely blurred.")
        else:
            limited_quality_flags.append("blurred_radiograph")
            warnings.append("Image appears blurred. AI review can continue, but the result may be incomplete.")
    elif laplacian_var < 60:
        warnings.append("Image sharpness is limited. Fine fracture lines may be harder to inspect.")

    UPLOAD_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_id = uuid4().hex
    preview_path = UPLOAD_PREVIEW_DIR / f"{preview_id}.png"
    if blocking:
        warnings = [
            warning
            for warning in warnings
            if warning.startswith("Browser reported")
        ]
        limited_quality_flags = []
    limited_quality = bool(limited_quality_flags) and not blocking
    metrics["limited_quality_radiograph"] = limited_quality
    metrics["limited_quality_flags"] = limited_quality_flags
    if limited_quality:
        metrics["preflight_processing"] = "limited_quality_grayscale_clahe_normalization"
        warnings.insert(
            0,
            "Limited-quality radiograph accepted: AI output must be treated as incomplete screening support, not reassurance.",
        )
        normalize_limited_quality_radiograph(rgb).save(preview_path)
    else:
        metrics["preflight_processing"] = "standard_rgb_preview"
        rgb.save(preview_path)

    if blocking:
        result = upload_result(
            "rejected",
            "Image failed preflight",
            "The image was not accepted for AI analysis because one or more quality requirements failed.",
            metrics=metrics,
            blocking=blocking,
            warnings=warnings,
            preview_id=preview_id,
        )
        write_json_file(uploaded_preflight_metadata_path(preview_id), result)
        return result

    result = upload_result(
        "accepted",
        "Image accepted with limited-quality warning" if limited_quality else "Image passed preflight",
        (
            "The image appears to be a radiograph, but quality limitations may reduce AI sensitivity. It can continue to live analysis with caution."
            if limited_quality
            else "The image is technically acceptable and can continue to the live research analysis path."
        ),
        metrics=metrics,
        blocking=[],
        warnings=warnings,
        preview_id=preview_id,
    )
    write_json_file(uploaded_preflight_metadata_path(preview_id), result)
    return result


def management_view(management: Dict[str, Any]) -> Dict[str, Any]:
    direction = management.get("management_direction") or {}
    anatomy_specific = management.get("anatomy_specific_management") or {}
    suggestions = (management.get("case_management_suggestions") or {}).get("suggestions") or []
    care = management.get("care_guidance") or {}
    return {
        "summary": direction.get("summary") or management.get("why_this_section_is_shown") or "",
        "direction_bullets": list_value(direction, "direction_bullets", limit=3),
        "review_focus": anatomy_specific.get("review_focus") or "",
        "clinical_localizers": list_value(anatomy_specific, "clinical_localizers", limit=4)
        or list_value(care, "clinical_localizers", limit=4),
        "management_considerations": list_value(anatomy_specific, "management_considerations", limit=4)
        or list_value(care, "initial_management_considerations", limit=4),
        "suggestions": suggestions[:3],
        "clinical_inputs": list_value(management, "clinical_inputs_required", limit=6),
        "red_flags": list_value(management, "red_flags", limit=6),
        "ai_must_not_decide": list_value(management, "ai_must_not_decide", limit=8),
        "limitations": list_value(management, "limitations", limit=4),
    }


def pubmed_source_cards(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = contract.get("pubmed_sources") or {}
    if isinstance(sources, dict):
        cards = sources.get("source_cards") or []
        return cards if isinstance(cards, list) else []
    if isinstance(sources, list):
        return sources
    return []


def explainability_view(explainability: Dict[str, Any]) -> Dict[str, Any]:
    status = explainability.get("case_status_explanation") or {}
    stages = explainability.get("stage_summaries") or []
    return {
        "short_explanation": status.get("short_explanation") or "",
        "bottom_line": status.get("bottom_line") or "",
        "pipeline_flow": status.get("pipeline_flow") or [],
        "stage_summaries": stages,
        "case_specific_limitation": status.get("case_specific_limitation") or "",
    }


def top_model_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [card for card in cards if card.get("id") in {"fracture_candidate_support", "anatomy_support"}]


def percent_text(value: Any) -> str | None:
    try:
        if value is None:
            return None
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return None


def support_level(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if value_f >= 80:
        return "high"
    if value_f >= 50:
        return "moderate"
    if value_f >= 25:
        return "limited"
    return "low"


def support_card(card_id: str, label: str, value: Any, score_kind: str, display_note: str) -> Dict[str, Any]:
    value_text = percent_text(value)
    level = support_level(value)
    return {
        "id": card_id,
        "label": label,
        "support_percent": value,
        "support_text": value_text,
        "support_level": level,
        "support_level_label": f"{level.title()} model signal" if level != "unknown" else "Model signal unavailable",
        "score_kind": score_kind,
        "display_context": "live_external_image",
        "calibration_note": "Internal model support score; not a calibrated clinical probability.",
        "display_note": display_note,
    }


def normalize_source_cards(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        source_id = source.get("id") or source.get("source_id") or f"source_{index}"
        cards.append(
            {
                **source,
                "id": source_id,
                "title": source.get("title") or source_id,
                "source_type": source.get("source_type") or source.get("type") or "Evidence source",
                "url": source.get("url") or source.get("link"),
                "limitations": source.get("limitations") or source.get("source_limitations") or [],
            }
        )
    return cards


def live_display_contract_from_unified(unified: Dict[str, Any]) -> Dict[str, Any]:
    final = unified.get("final_assessment") or {}
    stages = unified.get("stages") or {}
    findings = unified.get("findings") or {}
    candidates = findings.get("candidates") or []
    primary_candidate = candidates[0] if candidates else {}
    primary_bbox = primary_candidate.get("bbox") or {}
    stage2b = stages.get("stage2b_anatomy") or {}
    stage2c = stages.get("stage2c_image_level_safety") or {}
    care = unified.get("care_guidance") or {}
    review = unified.get("review") or {}
    evidence = unified.get("evidence") or {}

    selected_label = stage2b.get("selected_label") or "unknown"
    selected_display = stage2b.get("selected_display_label") or selected_label
    selected_support = stage2b.get("selected_support_percent")
    top_probabilities = []
    for item in stage2b.get("top_probabilities") or []:
        label = item.get("label")
        percent = item.get("support_percent")
        if percent is None:
            percent = item.get("percent")
        top_probabilities.append(
            {
                "label": label,
                "display_label": label,
                "support_percent": percent,
                "support_text": percent_text(percent),
            }
        )

    model_cards: List[Dict[str, Any]] = []
    if primary_candidate:
        stage1 = primary_candidate.get("stage1") or {}
        verifier = primary_candidate.get("stage2a_verifier") or {}
        fracture_support = (
            verifier.get("mean_support_percent")
            or stage1.get("detector_support_percent")
            or primary_bbox.get("detector_support_percent")
        )
        model_cards.append(
            support_card(
                "fracture_candidate_support",
                "Fracture candidate model signal",
                fracture_support,
                "internal_candidate_support",
                "This score supports a retained review cue only; it does not confirm fracture.",
            )
        )
    if selected_support is not None:
        model_cards.append(
            support_card(
                "anatomy_support",
                "Anatomy label model signal",
                selected_support,
                "internal_anatomy_support",
                "This score supports body-region classification only.",
            )
        )
    if stage2c.get("suspicious_support_percent") is not None:
        model_cards.append(
            support_card(
                "stage2c_caution_support",
                "Image-level caution model signal",
                stage2c.get("suspicious_support_percent"),
                "internal_image_level_support",
                "This image-level score does not localize a fracture and does not exclude injury.",
            )
        )

    show_bbox = bool(primary_bbox.get("available") and primary_bbox.get("xyxy_pixels"))
    source_cards = normalize_source_cards(evidence.get("displayed_sources") or [])
    source_ids = [source.get("id") for source in source_cards if source.get("id")]
    suggestion_sources = source_ids[:3]
    suggestions = []
    for index, item in enumerate(care.get("initial_management_considerations") or [], start=1):
        suggestions.append(
            {
                "suggestion_id": f"live_management_context_{index}",
                "category": "management context",
                "priority": "review",
                "text": item,
                "rationale": "Generated from the locked Stage 3 care-guidance layer for clinician review.",
                "source_ids": suggestion_sources,
                "clinician_only": True,
                "is_direct_order": False,
                "requires_human_confirmation": True,
                "prohibited_interpretation": "Do not interpret this as a patient-specific treatment instruction.",
            }
        )

    state_id = final.get("state_id") or "unknown"
    anatomy_phrase = selected_display if selected_display != "unknown" else "the reviewed image"
    if state_id == "candidate_retained":
        management_summary = (
            f"A retained AI review cue is present around {anatomy_phrase}. "
            "Management context is shown only to orient clinician review."
        )
    elif state_id == "image_level_warning_without_bbox":
        management_summary = (
            f"No localized box was retained, but the image-level safety lane raised caution around {anatomy_phrase}. "
            "Review should remain full-image and clinician-led."
        )
    else:
        management_summary = (
            "No high-confidence localized cue was retained. This limits the AI contribution and must not be treated as reassurance."
        )

    stage_summaries = []
    for key, title in [
        ("stage1_detection", "Stage 1: candidate detection"),
        ("stage2a_verifier", "Stage 2A: candidate verifier"),
        ("stage2a5_artifact_suppression", "Stage 2A.5: artifact suppression"),
        ("stage2b_anatomy", "Stage 2B: anatomy classification"),
        ("stage2c_image_level_safety", "Stage 2C: image-level caution"),
        ("stage3_care_guidance", "Stage 3: evidence-linked report"),
    ]:
        stage = stages.get(key) or {}
        ran = bool(stage.get("ran") or stage.get("evaluated"))
        stage_summaries.append(
            {
                "stage_display_label": title,
                "user_explanation": (
                    f"This stage {'ran' if ran else 'was not triggered'} for this uploaded image. "
                    f"Status: {stage.get('status') or 'not available'}."
                ),
                "output": stage.get("status") or "not available",
                "limitations": [
                    "Live external-image outputs are technical review support and are not calibrated clinical probabilities."
                ],
            }
        )

    return {
        "version": "app_facing_live_upload_contract_v1",
        "case": {
            **(unified.get("case") or {}),
            "academic_prototype": True,
        },
        "input": unified.get("input") or {},
        "status": {
            "state_id": state_id,
            "status_tone": final.get("status_tone"),
            "headline": final.get("headline") or "Live uploaded-image review completed.",
            "subheadline": final.get("subheadline")
            or "External uploaded image; human verification is required.",
            "primary_action": final.get("primary_action") or "Review the full X-ray and AI cues with clinical context.",
            "requires_human_review": True,
            "source_state_id": state_id,
        },
        "visual": {
            "display_priority": "bbox_overlay" if show_bbox else "original_image_only",
            "show_primary_bbox": show_bbox,
            "primary_bbox": {
                "available": show_bbox,
                "xyxy_pixels": primary_bbox.get("xyxy_pixels"),
                "bbox_source": primary_bbox.get("bbox_source"),
                "detector_support_percent": primary_bbox.get("detector_support_percent"),
                "bbox_quality_percent_proxy": primary_bbox.get("bbox_quality_percent_proxy"),
                "localization_note": primary_bbox.get("localization_note"),
            },
            "heatmap_available": False,
            "show_heatmap_by_default": False,
            "heatmap_button_label": None,
            "heatmap_warning": None,
        },
        "candidate_regions": candidates,
        "model_signals": {
            "score_policy": "Scores are internal model support signals. They are not calibrated clinical probabilities, do not diagnose fracture, and must not be used to exclude injury.",
            "cards": model_cards,
        },
        "anatomy": {
            "selected_label": selected_label,
            "selected_display_label": selected_display,
            "selected_support_percent": selected_support,
            "parent_label": stage2b.get("parent_label"),
            "parent_display_label": stage2b.get("parent_display_label") or stage2b.get("parent_label"),
            "output_type": stage2b.get("output_type"),
            "top_probabilities": top_probabilities,
            "display_badges": ["External uploaded image"],
            "disclaimer": "Anatomy scores describe body-region classification only. They do not indicate whether a fracture is present or absent.",
        },
        "stage2c_caution": {
            "ran": bool(stage2c.get("ran")),
            "decision": stage2c.get("decision"),
            "warning_visible": state_id == "image_level_warning_without_bbox",
            "support_percent": stage2c.get("suspicious_support_percent"),
            "bbox_created": False,
            "suppressed_existing_candidate": False,
            "heatmap_available": False,
            "display_text": stage2c.get("not_run_reason") or "",
        },
        "management_context": {
            "enabled": True,
            "label": "Evidence-linked management context",
            "is_treatment_plan": False,
            "guidance_strength": care.get("guidance_level") or "clinician_review_orientation",
            "why_this_section_is_shown": management_summary,
            "management_direction": {
                "summary": management_summary,
                "direction_bullets": care.get("initial_management_considerations") or [],
                "decision_boundary": "This section may orient clinician review, but it must not decide diagnosis, immobilisation, weight-bearing, medication, procedure, follow-up, clearance, reassurance, or surgery.",
            },
            "case_management_suggestions": {
                "suggestions": suggestions,
            },
            "anatomy_specific_management": {
                "review_focus": "; ".join(review.get("top_focus") or []) or anatomy_phrase,
                "clinical_localizers": review.get("top_context_questions") or [],
                "management_considerations": care.get("initial_management_considerations") or [],
            },
            "clinical_inputs_required": review.get("top_context_questions") or [],
            "red_flags": care.get("red_flags_do_not_miss") or [],
            "ai_must_not_decide": care.get("blocked_outputs") or review.get("do_not_infer") or [],
            "limitations": review.get("uncertainty_notes") or [],
        },
        "pubmed_sources": {
            "source_cards": source_cards,
        },
        "explainability": {
            "case_status_explanation": {
                "short_explanation": "The uploaded image passed the image-quality check and then moved through the AI review pipeline. The system looked for marked areas, checked whether they should remain visible, estimated the body region, and prepared a conservative review summary.",
                "bottom_line": "The output is review support only. The original X-ray and clinical context remain decisive.",
                "pipeline_flow": [
                    "The image-quality gate accepted the uploaded file.",
                    "The visual review model searched for areas that may deserve closer inspection.",
                    "Additional checks removed weak or artifact-like marks where appropriate.",
                    "The anatomy model estimated the most likely body region.",
                    "The reporting layer prepared review guidance and evidence context.",
                ],
                "case_specific_limitation": "External uploaded images are outside the locked FracAtlas validation set.",
            },
            "stage_summaries": stage_summaries,
        },
        "safety": unified.get("safety") or {},
        "technical_audit": {
            "unified_contract": unified,
            "live_upload_note": "This result was generated from a user-uploaded image through the live adapter path.",
        },
    }


def clinical_overview_view(
    status: Dict[str, Any],
    anatomy: Dict[str, Any],
    candidate_regions: List[Dict[str, Any]],
    management_summary: Dict[str, Any],
) -> Dict[str, Any]:
    state_id = status.get("state_id") or "unknown"
    anatomy_label = anatomy.get("selected_display_label") or "the reviewed anatomy"
    cue_count = len(candidate_regions)

    if state_id == "candidate_retained":
        title = "Marked area retained for review"
        plain_summary = (
            f"The system retained {cue_count or 1} marked area"
            f"{'' if cue_count == 1 else 's'} around {anatomy_label}. "
            "This means the marked area deserves focused human review, not that a fracture has been diagnosed."
        )
        review_first = [
            "Inspect the marked region on the original X-ray.",
            "Then review the complete image/study for additional or competing findings.",
            "Check whether the visual cue matches focal pain, tenderness, mechanism, and examination.",
        ]
        does_not_mean = [
            "The AI has confirmed a fracture.",
            "The marked area is the exact fracture boundary.",
            "Treatment, immobilisation, discharge, or follow-up can be decided from this output alone.",
        ]
        priority = "Focused review recommended"
        takeaway_label = "Review marked area first"
        boundary_label = "Possible finding, not diagnosis"
    elif state_id == "image_level_warning_without_bbox":
        title = "Whole-image caution signal"
        plain_summary = (
            f"No reliable marked area was retained, but the whole-image safety check still raised caution around {anatomy_label}. "
            "This should prompt careful full-image review rather than interpretation of a single marked location."
        )
        review_first = [
            "Review the full radiograph/study instead of searching for a single AI mark.",
            "Use any available heatmap only as a broad attention aid.",
            "Compare the image-level caution with symptoms, examination, and the formal radiology review.",
        ]
        does_not_mean = [
            "A fracture has been localized.",
            "Any heatmap is a fracture contour or treatment target.",
            "The image can be treated as clear if no marked area is shown.",
        ]
        priority = "Full-image review recommended"
        takeaway_label = "Review full image"
        boundary_label = "Caution signal, not localization"
    elif state_id == "no_high_confidence_candidate_retained":
        title = "No high-confidence marked area retained"
        plain_summary = (
            "The system did not retain a high-confidence marked area for this case. "
            "This is a limited screening output and must not be used to rule out injury."
        )
        review_first = [
            "Review the original X-ray/study using normal clinical workflow.",
            "If symptoms remain focal or concerning, rely on clinical judgement and local imaging pathways.",
            "Treat the AI output as limited support, not reassurance.",
        ]
        does_not_mean = [
            "Injury has been excluded.",
            "The X-ray has been declared normal.",
            "No clinical or radiology review is needed.",
        ]
        priority = "Limited AI output"
        takeaway_label = "No retained AI mark"
        boundary_label = "Limited output, not reassurance"
    else:
        title = "Review output available"
        plain_summary = status.get("subheadline") or "The AI output requires human clinical interpretation."
        review_first = [
            "Review the original image and clinical context.",
            "Use the AI output only as support for human review.",
        ]
        does_not_mean = [
            "The AI has made a clinical diagnosis.",
            "The AI can replace formal review.",
        ]
        priority = "Human review required"
        takeaway_label = "Review required"
        boundary_label = "Support only"

    return {
        "title": title,
        "priority": priority,
        "takeaway_label": takeaway_label,
        "boundary_label": boundary_label,
        "plain_summary": plain_summary,
        "review_first": review_first,
        "does_not_mean": does_not_mean,
        "clinical_inputs": management_summary.get("clinical_inputs", [])[:4],
        "red_flags": management_summary.get("red_flags", [])[:4],
    }


def clinical_support_report_view(
    status: Dict[str, Any],
    anatomy: Dict[str, Any],
    management: Dict[str, Any],
    management_summary: Dict[str, Any],
) -> Dict[str, Any]:
    anatomy_label = anatomy.get("selected_display_label") or "the reviewed region"
    anatomy_specific = management.get("anatomy_specific_management") or {}
    review_focus = management_summary.get("review_focus") or anatomy_specific.get("review_focus") or ""

    if not review_focus:
        review_focus = "the original image, any displayed review cue, and the full clinical presentation"

    status_text = status.get("headline") or "AI review output available."
    if status.get("subheadline"):
        status_text = f"{status_text} {status['subheadline']}"

    return {
        "status_text": status_text,
        "focus_sentence": f"Use this section to focus human review around {review_focus}.",
        "anatomy_sentence": (
            f"The anatomy classifier selected {anatomy_label}. "
            "This describes body-region context only and does not indicate whether a fracture is present."
        ),
        "localizers": management_summary.get("clinical_localizers", [])[:4],
        "red_flags": management_summary.get("red_flags", [])[:5],
        "must_not_decide": management_summary.get("ai_must_not_decide", [])[:6]
        or [
            "diagnosis",
            "fracture exclusion",
            "immobilisation",
            "weight-bearing or mobility status",
            "medication",
            "follow-up or surgery",
        ],
    }


def evidence_context_view(management: Dict[str, Any], source_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    direction = management.get("management_direction") or {}
    suggestions = (management.get("case_management_suggestions") or {}).get("suggestions") or []
    source_lookup = {source.get("id"): source for source in source_cards if source.get("id")}

    evidence_items = []
    for suggestion in suggestions[:3]:
        source_ids = suggestion.get("source_ids") or []
        visible_sources = [source_id for source_id in source_ids if source_id in source_lookup][:3]
        evidence_items.append(
            {
                "category": (suggestion.get("category") or "management context").replace("-", " ").title(),
                "text": suggestion.get("text") or "",
                "rationale": suggestion.get("rationale") or "",
                "source_ids": visible_sources,
                "boundary": suggestion.get("prohibited_interpretation") or "",
            }
        )

    return {
        "shown": bool(management.get("enabled")),
        "summary": direction.get("summary") or management.get("why_this_section_is_shown") or "",
        "direction_bullets": list_value(direction, "direction_bullets", limit=3),
        "evidence_items": evidence_items,
        "source_count": len(source_cards),
        "section_note": (
            "This section uses PubMed/RAG context only for management discussion. "
            "It does not diagnose the image and does not create a patient-specific treatment plan."
        ),
    }


def image_review_view(
    status: Dict[str, Any],
    visual: Dict[str, Any],
    candidate_regions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    state_id = status.get("state_id") or "unknown"
    has_bbox = bool((visual.get("primary_bbox") or {}).get("available"))
    heatmap_available = bool(visual.get("heatmap_available"))

    if state_id == "candidate_retained" and candidate_regions:
        summary = (
            "A localized visual cue was retained and can be inspected with the overlay and zoom controls. "
            "Use it as a starting point for review, then inspect the full radiograph."
        )
        primary_instruction = "Start at the displayed review cue, then widen the review to the surrounding bone and the full image."
    elif state_id == "image_level_warning_without_bbox":
        summary = (
            "No reliable localized box was retained. The image-level safety lane raised caution, so this case should be reviewed as a whole-image finding."
        )
        primary_instruction = "Do not search for a fake box; inspect the complete radiograph and correlate with the clinical presentation."
    elif state_id == "no_high_confidence_candidate_retained":
        summary = (
            "No high-confidence localized cue was retained. This is a limited AI screening output, not reassurance and not fracture exclusion."
        )
        primary_instruction = "Use normal clinical/radiology review workflow and do not treat the absence of an overlay as an all-clear result."
    else:
        summary = "Image review output is available and requires human interpretation."
        primary_instruction = "Review the original image and clinical context."

    cues = []
    for index, candidate in enumerate(candidate_regions, start=1):
        stage1 = candidate.get("stage1") or {}
        bbox = candidate.get("bbox") or {}
        artifact = candidate.get("artifact_suppression") or {}
        support = stage1.get("detector_support_percent")
        if isinstance(support, (int, float)):
            support_text = f"{support:.1f}% AI support"
        else:
            support_text = "AI support unavailable"
        if artifact.get("available"):
            if artifact.get("suppressed"):
                artifact_text = "Artifact check suppressed this region."
            else:
                artifact_text = "Artifact check did not suppress this region."
        else:
            artifact_text = "Artifact check unavailable."
        cues.append(
            {
                "candidate_id": candidate.get("candidate_id") or f"C{index}",
                "title": f"Marked area {index}",
                "support_text": support_text,
                "plain_note": (
                    "This marked area was kept for closer human review. "
                    "It is a visual cue, not a diagnosis and not an exact fracture outline."
                ),
                "box_note": bbox.get("localization_note")
                or "The box is a review cue, not a precise fracture outline.",
                "artifact_text": artifact_text,
            }
        )

    return {
        "summary": summary,
        "primary_instruction": primary_instruction,
        "has_bbox": has_bbox,
        "heatmap_available": heatmap_available,
        "cue_count": len(cues),
        "cues": cues,
        "viewer_notes": [
            "Zoom can be used for closer visual inspection; image quality may degrade at high zoom.",
            "The overlay can be turned on/off, but hiding it does not change the AI result.",
            "Any displayed box is a review cue only, not a diagnosis, contour, or treatment target.",
        ],
    }


def markdown_path(image_id: str) -> Path:
    return GOLDEN_MARKDOWN_DIR / f"{Path(image_id).stem}.md"


def provenance_view(
    case: Dict[str, Any],
    contract: Dict[str, Any],
    external_notice: str | None = None,
) -> Dict[str, Any]:
    contract_case = contract.get("case") or {}
    technical = contract.get("technical_audit") or {}
    runner_metadata = technical.get("runner_metadata") or {}
    contract_audit = contract.get("contract_audit") or {}
    input_payload = contract.get("input") or {}

    analysis_mode = contract_case.get("analysis_mode") or ("live_full_pipeline" if external_notice else "locked_precomputed")
    is_live = analysis_mode == "live_full_pipeline" or bool(external_notice)
    source_payload = (
        contract_case.get("source_payload_type")
        or runner_metadata.get("runner")
        or technical.get("pipeline_version")
        or technical.get("source_payload_version")
        or contract.get("version")
        or "not recorded"
    )

    if is_live:
        mode_label = "Live uploaded analysis"
        mode_detail = "Generated from a user-uploaded image during this app session."
        badge_label = "Live analysis"
        source_label = "External uploaded image"
        validator_status = (
            "Preflight accepted; unified live contract returned."
            if contract_audit.get("schema_keys_present", True)
            else "Live contract validation incomplete."
        )
    else:
        mode_label = "Locked demo output"
        mode_detail = "Precomputed case used for stable demonstration and repeatable review."
        badge_label = "Demo case"
        source_label = contract_case.get("input_source") or case.get("gallery_group") or "Curated demo case"
        validator_status = "Locked app contract loaded."

    warnings = input_payload.get("warnings") or []
    badges = [
        "Live" if is_live else "Demo",
        "External image" if is_live else "Precomputed",
        "Human review required",
    ]
    if contract_case.get("out_of_distribution"):
        badges.append("Outside validation distribution")

    return {
        "mode_label": mode_label,
        "mode_detail": mode_detail,
        "badge_label": badge_label,
        "badges": badges,
        "details": [
            {"label": "Run type", "value": mode_label},
            {"label": "Image source", "value": source_label},
            {"label": "Contract", "value": contract.get("version") or "not recorded"},
            {"label": "Pipeline", "value": source_payload},
            {"label": "Validator", "value": validator_status},
            {"label": "Viewed", "value": display_timestamp()},
        ],
        "warnings": warnings[:2],
    }


def app_context_from_contract(
    case: Dict[str, Any],
    contract: Dict[str, Any],
    image_url: str,
    json_download_url: str,
    markdown_download_url: str | None = None,
    external_notice: str | None = None,
) -> Dict[str, Any]:
    image_path = safe_image_path(contract)
    dims = image_dimensions(image_path)
    status = contract.get("status") or {}
    visual = contract.get("visual") or {}
    anatomy = contract.get("anatomy") or {}
    model_signals = contract.get("model_signals") or {}
    model_cards = model_signals.get("cards") or []
    management = contract.get("management_context") or {}
    management_summary = management_view(management)
    source_cards = pubmed_source_cards(contract)
    explainability = contract.get("explainability") or {}
    candidate_regions = contract.get("candidate_regions") or []
    return {
        "case": case,
        "contract": contract,
        "image_url": image_url,
        "json_download_url": json_download_url,
        "markdown_download_url": markdown_download_url,
        "external_notice": external_notice,
        "provenance": provenance_view(case, contract, external_notice),
        "image_dimensions": dims,
        "bbox_percent": bbox_percent(contract, dims),
        "candidate_bboxes": candidate_bboxes_percent(candidate_regions, dims),
        "status_class": status_class(status.get("state_id", "")),
        "status": status,
        "visual": visual,
        "anatomy": anatomy,
        "model_cards": model_cards,
        "overview_model_cards": top_model_cards(model_cards),
        "candidate_regions": candidate_regions,
        "source_cards": source_cards,
        "management": management,
        "management_view": management_summary,
        "image_review": image_review_view(status, visual, candidate_regions),
        "clinical_overview": clinical_overview_view(
            status,
            anatomy,
            candidate_regions,
            management_summary,
        ),
        "clinical_support_report": clinical_support_report_view(status, anatomy, management, management_summary),
        "evidence_context": evidence_context_view(management, source_cards),
        "explainability": explainability,
        "explainability_view": explainability_view(explainability),
        "safety": contract.get("safety") or {},
        "technical_audit": contract.get("technical_audit") or {},
        "markdown_available": bool(markdown_download_url),
    }


def app_context(case: Dict[str, Any]) -> Dict[str, Any]:
    contract = load_contract_for_case(case)
    demo_id = case["demo_id"]
    md_path = markdown_path((contract.get("case") or {}).get("image_id", ""))
    return app_context_from_contract(
        case,
        contract,
        image_url=f"/image/{demo_id}",
        json_download_url=f"/download/{demo_id}/json",
        markdown_download_url=f"/download/{demo_id}/markdown" if md_path.exists() else None,
    )


def run_live_upload(preview_id: str) -> Dict[str, Any]:
    image_path = safe_uploaded_preview_path(preview_id)
    output_dir = UPLOAD_LIVE_OUTPUT_DIR / preview_id
    summary_path = output_dir / "summary.json"
    from live_full_pipeline_adapter import PIPELINE_VERSION as LIVE_PIPELINE_VERSION
    from live_full_pipeline_adapter import run_live_full_pipeline_one

    if summary_path.exists():
        cached_summary = read_json(summary_path)
        if cached_summary.get("pipeline_version") == LIVE_PIPELINE_VERSION:
            return cached_summary

    return run_live_full_pipeline_one(
        image_path=image_path,
        output_dir=output_dir,
        device=LIVE_ANALYSIS_DEVICE,
    )


def live_upload_context(preview_id: str) -> Dict[str, Any]:
    summary = run_live_upload(preview_id)
    preflight_metadata = read_uploaded_preflight_metadata(preview_id)
    preflight_metrics = preflight_metadata.get("metrics") or {}
    limited_quality = bool(preflight_metrics.get("limited_quality_radiograph"))
    unified_path = Path(((summary.get("outputs") or {}).get("unified_contract")) or "")
    if not unified_path.exists():
        raise RuntimeError("Live pipeline did not produce a unified contract.")
    unified = read_json(unified_path)
    display_contract = live_display_contract_from_unified(unified)
    image_id = (display_contract.get("case") or {}).get("image_id") or f"{preview_id}.png"
    short_id = preview_id[:8]
    case = {
        "demo_id": f"upload-{short_id}",
        "image_id": image_id,
        "display_title": "Uploaded X-ray live review",
        "display_subtitle": "External uploaded image processed through the live research pipeline.",
        "gallery_group": "Live upload",
    }
    external_notice = (
        "Uploaded-image review. Limited-quality radiograph accepted; AI output may be incomplete and must not be treated as reassurance."
        if limited_quality
        else (
            "Uploaded-image review. Results are advisory and require human verification; "
            "they are not a diagnosis or treatment decision."
        )
    )
    context = app_context_from_contract(
        case,
        display_contract,
        image_url=f"/uploaded-preview/{preview_id}",
        json_download_url=f"/download/upload/{preview_id}/json",
        markdown_download_url=None,
        external_notice=external_notice,
    )
    context["preflight_metadata"] = preflight_metadata
    context["limited_quality_upload"] = limited_quality
    context["limited_quality_flags"] = preflight_metrics.get("limited_quality_flags") or []
    return context


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})


@app.get("/demo")
def demo(request: Request):
    cases = []
    for case in curated_cases():
        contract = load_contract_for_case(case)
        cases.append(
            {
                **case,
                "state_id": (contract.get("status") or {}).get("state_id"),
                "headline": (contract.get("status") or {}).get("headline"),
                "anatomy": (contract.get("anatomy") or {}).get("selected_display_label"),
                "status_class": status_class((contract.get("status") or {}).get("state_id", "")),
            }
        )
    return templates.TemplateResponse(request, "demo.html", {"cases": cases})


@app.get("/review")
def review(request: Request):
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "upload_result": None,
            "upload_requirements": upload_gate_requirements(),
        },
    )


@app.post("/review")
async def review_upload(request: Request, xray_image: UploadFile = File(...)):
    data = await xray_image.read()
    result = validate_uploaded_xray(data, xray_image.filename or "", xray_image.content_type)
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "upload_result": result,
            "upload_requirements": upload_gate_requirements(),
        },
    )


@app.get("/uploaded-preview/{preview_id}")
def uploaded_preview(preview_id: str):
    return FileResponse(safe_uploaded_preview_path(preview_id))


@app.get("/upload/{preview_id}/analysis")
def uploaded_analysis_progress(request: Request, preview_id: str):
    safe_uploaded_preview_path(preview_id)
    return templates.TemplateResponse(
        request,
        "upload_analysis.html",
        {
            "preview_id": preview_id,
            "target_url": f"/upload/{preview_id}/result",
        },
    )


@app.get("/upload/{preview_id}/result")
def uploaded_result(request: Request, preview_id: str):
    try:
        return templates.TemplateResponse(request, "case.html", live_upload_context(preview_id))
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "upload_error.html",
            {
                "preview_id": preview_id,
                "error_message": str(exc),
            },
            status_code=500,
        )


@app.get("/problem-description")
def problem_description(request: Request):
    return templates.TemplateResponse(
        request,
        "problem_description.html",
        {
            "analysis": None,
            "description": "",
        },
    )


@app.post("/problem-description")
def problem_description_submit(request: Request, description: str = Form(...)):
    return templates.TemplateResponse(
        request,
        "problem_description.html",
        {
            "analysis": analyze_problem_description(description),
            "description": description,
        },
    )


@app.get("/case/{demo_id}/analysis")
def analysis(request: Request, demo_id: str):
    case = case_by_demo_id(demo_id)
    return templates.TemplateResponse(request, "analysis.html", {"case": case})


@app.get("/case/{demo_id}")
def case_result(request: Request, demo_id: str):
    case = case_by_demo_id(demo_id)
    return templates.TemplateResponse(request, "case.html", app_context(case))


@app.get("/image/{demo_id}")
def case_image(demo_id: str):
    case = case_by_demo_id(demo_id)
    contract = load_contract_for_case(case)
    return FileResponse(safe_image_path(contract))


@app.get("/download/{demo_id}/json")
def download_json(demo_id: str):
    case = case_by_demo_id(demo_id)
    return FileResponse(REPO_ROOT / case["app_contract_json"], filename=f"{Path(case['image_id']).stem}.app_contract.json")


@app.get("/download/upload/{preview_id}/json")
def download_upload_json(preview_id: str):
    summary = run_live_upload(preview_id)
    unified_path = Path(((summary.get("outputs") or {}).get("unified_contract")) or "")
    if not unified_path.exists():
        raise HTTPException(status_code=404, detail="Live unified contract not available")
    return FileResponse(unified_path, filename=f"{preview_id}.unified_live_contract.json")


@app.get("/download/{demo_id}/markdown")
def download_markdown(demo_id: str):
    case = case_by_demo_id(demo_id)
    path = markdown_path(case["image_id"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Markdown report not available")
    return FileResponse(path, filename=f"{Path(case['image_id']).stem}.md")


@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {})


@app.get("/safety")
def safety(request: Request):
    return templates.TemplateResponse(request, "safety.html", {})


@app.get("/upload")
def upload_placeholder():
    return RedirectResponse(url="/review", status_code=303)
