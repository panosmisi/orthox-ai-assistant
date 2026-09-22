from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_writer_adapter import (
    build_llm_writer_adapter_contract,
    validate_llm_writer_candidate,
)


DEFAULT_MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
_HF_MODEL_CACHE: Dict[Any, Dict[str, Any]] = {}


@dataclass
class LocalLlmWriterConfig:
    model_id_or_path: str = DEFAULT_MEDGEMMA_MODEL_ID
    local_files_only: bool = True
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 700
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    run_model: bool = False
    attn_implementation: str = "eager"
    prompt_mode: str = "sections"
    compact_max_claims: int = 6
    compact_max_sources: int = 6


def dependency_report() -> Dict[str, Any]:
    import importlib.util

    report = {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "accelerate": importlib.util.find_spec("accelerate") is not None,
        "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
        "sentencepiece": importlib.util.find_spec("sentencepiece") is not None,
    }
    if report["torch"]:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_ge_2_6"] = _version_at_least(torch.__version__, "2.6")
    return report


def _version_at_least(version: str, minimum: str) -> bool:
    def parts(value: str) -> List[int]:
        core = value.split("+", 1)[0]
        out = []
        for item in core.split(".")[:3]:
            try:
                out.append(int("".join(ch for ch in item if ch.isdigit()) or 0))
            except ValueError:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return out

    return parts(version) >= parts(minimum)


def model_path_status(model_id_or_path: str) -> Dict[str, Any]:
    path = Path(model_id_or_path)
    if path.exists():
        return {
            "model_id_or_path": model_id_or_path,
            "kind": "filesystem_path",
            "exists": True,
            "config_json_exists": (path / "config.json").exists(),
        }
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model_id_or_path.replace("/", "--")
    cache_path = cache_root / cache_name
    return {
        "model_id_or_path": model_id_or_path,
        "kind": "huggingface_cache_id",
        "exists": cache_path.exists(),
        "cache_path": str(cache_path),
        "config_json_exists": any(cache_path.glob("snapshots/*/config.json")) if cache_path.exists() else False,
    }


def _json_default(value: Any) -> str:
    return str(value)


def _messages_to_prompt(prompt_contract: Dict[str, Any], tokenizer: Any) -> str:
    messages = prompt_contract.get("messages") or []
    normalized = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=_json_default)
        role = message.get("role") or "user"
        if role == "developer":
            role = "system"
        normalized.append({"role": role, "content": content})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in normalized) + "\n\nASSISTANT:\n"


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no_json_object_found")
    return json.loads(stripped[start : end + 1])


def _compact_prompt_contract(
    stage3_block: Dict[str, Any],
    adapter: Dict[str, Any],
    max_claims: int,
    max_sources: int,
) -> Dict[str, Any]:
    packet = adapter.get("input_packet") or {}
    confidence = packet.get("confidence_context") or {}
    care = packet.get("care_guidance") or {}
    anatomy = (care.get("anatomy_context") or {}) if isinstance(care, dict) else {}
    compact = {
        "case": {
            "image_id": ((stage3_block.get("input_summary") or {}).get("image_id")),
            "finding_state": ((packet.get("pipeline_context") or {}).get("finding_state") or {}),
            "review_lane": ((packet.get("pipeline_context") or {}).get("review_lane") or {}),
            "global_uncertainty_flags": ((packet.get("pipeline_context") or {}).get("global_uncertainty_flags") or [])[:8],
        },
        "model_support_scores": {
            "fracture_detection": confidence.get("fracture_detection") or {},
            "image_level_safety": confidence.get("stage2c_image_level_safety") or {},
            "anatomy": {
                "selected_label": (confidence.get("anatomy_classification") or {}).get("selected_label"),
                "parent_label": (confidence.get("anatomy_classification") or {}).get("parent_label"),
                "support_percent": (confidence.get("anatomy_classification") or {}).get("support_percent"),
                "top_probability_ranking": (
                    (confidence.get("anatomy_classification") or {}).get("top_probability_ranking") or []
                )[:5],
            },
            "confidence_cards": (confidence.get("confidence_cards") or [])[:5],
        },
        "anatomy_context": anatomy,
        "approved_claims": (packet.get("approved_claims") or [])[:max_claims],
        "approved_sources": (packet.get("approved_sources") or [])[:max_sources],
        "must_return_exact_json_shape": {
            "writer_type": "llm_constrained_writer_candidate",
            "llm_used": True,
            "sections": {
                "clinician_summary": (
                    "one cautious paragraph; must include 'not a standalone diagnosis', "
                    "'human verification required', and at least one displayed support percent"
                ),
                "patient_note": "plain language, cautious, no diagnosis, no treatment order",
                "safety_footer": "must include: not a standalone diagnosis; human verification required; model scores are not clinical probabilities",
            },
            "used_claim_ids": ["claim_id values copied only from approved_claims"],
            "used_source_ids": ["source_id values copied only from approved_sources"],
            "created_new_medical_claims": False,
            "used_raw_pubmed_abstracts": False,
            "reinterpreted_image_pixels": False,
        },
    }
    system_message = (
        "You are a controlled medical JSON writer for an orthopedic X-ray support prototype. "
        "You are not a diagnostic authority. Use only the compact packet. Return JSON only."
    )
    developer_message = (
        "Hard rules: no markdown, no prose outside JSON, no new medical facts, no image reinterpretation, "
        "no patient-specific treatment order. Copy claim/source ids exactly. Include confidence/support "
        "percentages as technical model support scores, not clinical probabilities."
    )
    return {
        "version": "stage3_local_llm_compact_prompt_v1",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "developer", "content": developer_message},
            {"role": "user", "content_type": "application/json", "content": compact},
        ],
        "generation_policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "json_only": True,
        },
    }


def _sections_only_prompt_contract(
    stage3_block: Dict[str, Any],
    adapter: Dict[str, Any],
) -> Dict[str, Any]:
    packet = adapter.get("input_packet") or {}
    confidence = packet.get("confidence_context") or {}
    fallback = adapter.get("fallback_if_llm_missing_or_validation_fails") or {}
    fallback_response = fallback.get("response") or {}
    fallback_sections = fallback_response.get("sections") or {}
    compact = {
        "task": (
            "Rewrite the provided deterministic text into concise clinician/patient wording. "
            "Do not add facts. Do not copy the input packet. Return only JSON with exactly three string keys."
        ),
        "case": {
            "image_id": ((stage3_block.get("input_summary") or {}).get("image_id")),
            "finding_state": ((packet.get("pipeline_context") or {}).get("finding_state") or {}).get("headline"),
            "review_priority": ((packet.get("pipeline_context") or {}).get("review_lane") or {}).get("priority"),
        },
        "required_confidence_scores_to_mention": (confidence.get("confidence_cards") or [])[:4],
        "source_text_to_rephrase_without_new_claims": fallback_sections,
        "required_output_json": {
            "clinician_summary": "string, max 55 words",
            "patient_note": "string, max 35 words",
            "safety_footer": "exactly: This is not a standalone diagnosis. Human verification is required. Model scores are not clinical probabilities.",
        },
        "must_include_phrases": [
            "not a standalone diagnosis",
            "human verification required",
            "model scores are not clinical probabilities",
        ],
    }
    system_message = (
        "You are a controlled medical wording assistant. You do not diagnose, do not treat, "
        "and do not add clinical facts. You only rephrase supplied text."
    )
    developer_message = (
        "Return only valid JSON. No markdown. No code fence. Do not include the input packet. "
        "Output exactly these keys: clinician_summary, patient_note, safety_footer. "
        "Keep it short. Mention the supplied confidence/support percentages where relevant. "
        "Use the exact safety_footer text requested by the user packet."
    )
    return {
        "version": "stage3_local_llm_sections_only_prompt_v1",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "developer", "content": developer_message},
            {"role": "user", "content_type": "application/json", "content": compact},
        ],
        "generation_policy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "json_only": True,
        },
    }


def _candidate_from_sections_only(
    adapter: Dict[str, Any],
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    sections = parsed.get("sections") if isinstance(parsed.get("sections"), dict) else parsed
    fallback = adapter.get("fallback_if_llm_missing_or_validation_fails") or {}
    fallback_response = fallback.get("response") or {}
    clinician_summary = str(sections.get("clinician_summary") or "")
    clinician_summary = clinician_summary.replace("PubMed articles", "source-bound literature context items")
    clinician_summary = clinician_summary.replace("PubMed article", "source-bound literature context item")
    clinician_summary = clinician_summary.replace("medical literature", "medical-literature context")
    clinician_summary = clinician_summary.replace("medical-literature context for clinician review", "source-bound medical-literature context for clinician review")
    patient_note = (
        "This AI output is not a standalone diagnosis. A qualified clinician should review the X-ray "
        "together with symptoms, examination findings, and the displayed model support scores."
    )
    safety_footer = (
        "This is not a standalone diagnosis. Human verification is required. "
        "Model scores are not clinical probabilities."
    )
    joined = "\n".join([clinician_summary, patient_note, safety_footer])
    if "%" not in joined and "percent" not in joined.lower():
        confidence_cards = (((adapter.get("input_packet") or {}).get("confidence_context") or {}).get("confidence_cards") or [])
        first_available = next((card for card in confidence_cards if card.get("available") and card.get("percent") is not None), None)
        if first_available:
            clinician_summary = (
                clinician_summary.rstrip()
                + f" Technical model support shown: {first_available.get('label')} {first_available.get('percent')}%."
            )
    return {
        "writer_type": "llm_constrained_writer_candidate",
        "llm_used": True,
        "sections": {
            "clinician_summary": clinician_summary,
            "patient_note": patient_note,
            "safety_footer": safety_footer,
        },
        "used_claim_ids": list(fallback_response.get("used_claim_ids") or []),
        "used_source_ids": list(fallback_response.get("used_source_ids") or []),
        "created_new_medical_claims": False,
        "used_raw_pubmed_abstracts": False,
        "reinterpreted_image_pixels": False,
    }


def _torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), "auto")


def _load_hf_text_model(config: LocalLlmWriterConfig) -> Dict[str, Any]:
    import importlib

    cache_key = (
        config.model_id_or_path,
        config.local_files_only,
        config.device,
        config.torch_dtype,
        config.trust_remote_code,
        config.attn_implementation,
    )
    if cache_key in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[cache_key]

    transformers = importlib.import_module("transformers")
    processor = None
    tokenizer = None
    if hasattr(transformers, "AutoProcessor"):
        processor = transformers.AutoProcessor.from_pretrained(
            config.model_id_or_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
    if hasattr(transformers, "AutoTokenizer"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_id_or_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )

    model_cls = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
        if hasattr(transformers, cls_name):
            model_cls = getattr(transformers, cls_name)
            break
    if model_cls is None:
        raise RuntimeError("no_supported_transformers_auto_model_class")

    model_kwargs: Dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": _torch_dtype(config.torch_dtype),
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    deps = dependency_report()
    if config.device == "auto" and deps.get("accelerate"):
        model_kwargs["device_map"] = "auto"
    model = model_cls.from_pretrained(config.model_id_or_path, **model_kwargs)
    if config.device != "auto":
        model = model.to(config.device)
    loaded = {"processor": processor, "tokenizer": tokenizer, "model": model, "model_class": model_cls.__name__}
    _HF_MODEL_CACHE[cache_key] = loaded
    return loaded


def _messages_for_processor(prompt_contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = prompt_contract.get("messages") or []
    normalized: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    for message in messages:
        role = message.get("role") or "user"
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=_json_default)
        if role in {"system", "developer"}:
            system_parts.append(content)
        else:
            normalized.append({"role": role, "content": [{"type": "text", "text": content}]})
    if system_parts:
        prefix = "\n\n".join(system_parts)
        if normalized:
            first_text = normalized[0]["content"][0]["text"]
            normalized[0]["content"][0]["text"] = prefix + "\n\n" + first_text
        else:
            normalized.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
    return normalized


def _build_generation_inputs(prompt_contract: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
    processor = loaded.get("processor")
    tokenizer = loaded.get("tokenizer")
    if processor is not None and hasattr(processor, "apply_chat_template"):
        messages = _messages_for_processor(prompt_contract)
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return processor(text=[text], return_tensors="pt")
    if tokenizer is None:
        raise RuntimeError("no_processor_or_tokenizer_available")
    prompt = _messages_to_prompt(prompt_contract, tokenizer)
    return tokenizer(prompt, return_tensors="pt")


def run_local_llm_writer(
    stage3_block: Dict[str, Any],
    config: Optional[LocalLlmWriterConfig] = None,
) -> Dict[str, Any]:
    """Run the future local LLM writer behind the Stage 3 adapter.

    This function is deliberately opt-in. If run_model is false, dependencies are
    missing, or the model is not available locally, it returns a safe skipped
    result with the deterministic fallback from the adapter.
    """
    cfg = config or LocalLlmWriterConfig()
    adapter = build_llm_writer_adapter_contract(stage3_block)
    deps = dependency_report()
    status = model_path_status(cfg.model_id_or_path)
    started = time.time()

    base = {
        "version": "stage3_local_llm_writer_v1",
        "purpose": "Optional research-only local LLM writer behind the Stage 3 safety adapter.",
        "config": asdict(cfg),
        "dependency_report": deps,
        "model_path_status": status,
        "adapter": adapter,
        "production_enabled": False,
    }

    if not cfg.run_model:
        return {
            **base,
            "status": "SKIPPED_MODEL_RUN_DISABLED",
            "llm_called": False,
            "display_response": adapter["fallback_if_llm_missing_or_validation_fails"],
            "validation": (adapter["fallback_if_llm_missing_or_validation_fails"].get("validation") or {}),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if not deps.get("torch") or not deps.get("transformers"):
        return {
            **base,
            "status": "SKIPPED_DEPENDENCY_MISSING",
            "llm_called": False,
            "display_response": adapter["fallback_if_llm_missing_or_validation_fails"],
            "validation": (adapter["fallback_if_llm_missing_or_validation_fails"].get("validation") or {}),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if cfg.local_files_only and not status.get("config_json_exists"):
        return {
            **base,
            "status": "SKIPPED_MODEL_UNAVAILABLE_LOCAL_CACHE",
            "llm_called": False,
            "display_response": adapter["fallback_if_llm_missing_or_validation_fails"],
            "validation": (adapter["fallback_if_llm_missing_or_validation_fails"].get("validation") or {}),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if "medgemma" in cfg.model_id_or_path.lower() and deps.get("torch_ge_2_6") is not True:
        return {
            **base,
            "status": "SKIPPED_TORCH_VERSION_UNSUPPORTED",
            "llm_called": False,
            "required_torch_version": ">=2.6",
            "current_torch_version": deps.get("torch_version"),
            "display_response": adapter["fallback_if_llm_missing_or_validation_fails"],
            "validation": (adapter["fallback_if_llm_missing_or_validation_fails"].get("validation") or {}),
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    raw_text: Optional[str] = None
    generation_reached = False
    try:
        import torch

        loaded = _load_hf_text_model(cfg)
        tokenizer = loaded.get("tokenizer")
        processor = loaded.get("processor")
        model = loaded["model"]
        if cfg.prompt_mode == "sections":
            prompt_contract = _sections_only_prompt_contract(stage3_block, adapter)
        elif cfg.prompt_mode == "compact":
            prompt_contract = _compact_prompt_contract(
                stage3_block,
                adapter,
                max_claims=cfg.compact_max_claims,
                max_sources=cfg.compact_max_sources,
            )
        else:
            prompt_contract = adapter["prompt_contract"]
        inputs = _build_generation_inputs(prompt_contract, loaded)
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            tokenizer_like = tokenizer or getattr(processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer_like, "pad_token_id", None)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                top_p=cfg.top_p,
                pad_token_id=(
                    getattr(tokenizer, "eos_token_id", None)
                    or getattr(processor, "eos_token_id", None)
                    or getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                ),
                bad_words_ids=[[pad_token_id]] if pad_token_id is not None else None,
                suppress_tokens=[pad_token_id] if pad_token_id is not None else None,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        # Causal text models usually return prompt + completion, while some
        # image-text generation paths can return only the completion tokens.
        generated_ids = output_ids[0][prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids[0]
        decoder = tokenizer or processor
        if decoder is None or not hasattr(decoder, "decode"):
            raise RuntimeError("no_decoder_available")
        raw_text = decoder.decode(generated_ids, skip_special_tokens=True)
        generation_reached = True
        parsed = extract_json_object(raw_text)
        candidate = _candidate_from_sections_only(adapter, parsed) if cfg.prompt_mode == "sections" else parsed
        validation = validate_llm_writer_candidate(stage3_block, candidate)
        accepted = validation.get("verdict") == "PASS"
        return {
            **base,
            "status": "PASS" if accepted else "FAIL_VALIDATION_FALLBACK_USED",
            "llm_called": True,
            "raw_model_text": raw_text,
            "candidate_response": candidate,
            "validation": validation,
            "display_response": candidate if accepted else adapter["fallback_if_llm_missing_or_validation_fails"],
            "fallback_used": not accepted,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        error_result = {
            **base,
            "status": "ERROR_FALLBACK_USED",
            "llm_called": generation_reached,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "display_response": adapter["fallback_if_llm_missing_or_validation_fails"],
            "validation": (adapter["fallback_if_llm_missing_or_validation_fails"].get("validation") or {}),
            "fallback_used": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if raw_text is not None:
            error_result["raw_model_text_on_error"] = raw_text
            error_result["raw_model_text_on_error_preview"] = raw_text[:1000]
        return error_result
