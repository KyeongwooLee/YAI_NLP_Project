from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ProjectConfig, ensure_project_dirs
from src.data.persona_signal_extractor import build_persona_profile
from src.inference.router_v1 import RouterV1
from src.inference.router_v2 import RouterV2
from src.utils.model_resolver import resolve_base_model_name


def _load_router(config: ProjectConfig):
    v2_path = config.router_dir / "router_v2.pkl"
    v1_path = config.router_dir / "router_v1.pkl"
    if v2_path.exists():
        return "v2", RouterV2.load(v2_path)
    if v1_path.exists():
        return "v1", RouterV1.load(v1_path)
    raise FileNotFoundError(
        f"No router model found. Expected {v2_path} or {v1_path}."
    )


def _load_model_and_tokenizer(base_model_name: str, local_files_only: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise RuntimeError(
            "Tokenizer load failed. Use a local BASE_MODEL_NAME path or set LOCAL_FILES_ONLY=0."
        ) from error
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise RuntimeError(
            "Base model load failed. Use a local BASE_MODEL_NAME path or set LOCAL_FILES_ONLY=0."
        ) from error
    model.eval()
    return model, tokenizer


def _apply_domain_persona_adapters(
    model,
    domain_adapter_path: Path,
    persona_adapter_path: Path | None,
    domain_weight: float,
    persona_weight: float,
):
    peft_model = PeftModel.from_pretrained(model, str(domain_adapter_path), adapter_name="domain")
    peft_model.set_adapter("domain")

    if persona_adapter_path is not None and persona_adapter_path.exists():
        peft_model.load_adapter(str(persona_adapter_path), adapter_name="persona")
        try:
            peft_model.add_weighted_adapter(
                adapters=["domain", "persona"],
                weights=[domain_weight, persona_weight],
                adapter_name="active",
                combination_type="linear",
            )
            peft_model.set_adapter("active")
        except Exception:
            peft_model.set_adapter("domain")
    return peft_model


def _list_domain_adapter_candidates(config: ProjectConfig, preferred_domain: str) -> list[Path]:
    all_dirs = sorted([path for path in config.domain_adapters_dir.iterdir() if path.is_dir()])
    preferred = config.domain_adapters_dir / preferred_domain
    normal_dirs = [path for path in all_dirs if not path.name.startswith("merged_")]
    merged_dirs = [path for path in all_dirs if path.name.startswith("merged_")]

    ordered: list[Path] = []
    if preferred.exists():
        ordered.append(preferred)
    for path in normal_dirs:
        if path not in ordered:
            ordered.append(path)
    for path in merged_dirs:
        if path not in ordered:
            ordered.append(path)
    return ordered


def _build_prompt(query: str, persona_label: str) -> str:
    persona = build_persona_profile(persona_label)
    return (
        "You are a personal tutoring assistant.\n"
        f"Style instruction: {persona.style_instruction}\n"
        f"Student question: {query}\n"
        "Tutor answer:"
    )


def run_inference(
    query: str,
    *,
    config: ProjectConfig,
    max_new_tokens: int = 180,
    student_preference: str = "",
    teacher_preference: str = "",
) -> dict:
    ensure_project_dirs(config)
    config.base_model_name = resolve_base_model_name(config)
    router_name, router = _load_router(config)
    route = router.route(  # type: ignore[attr-defined]
        query,
        student_preference_hint=student_preference,
        teacher_preference_hint=teacher_preference,
    )

    domain_candidates = _list_domain_adapter_candidates(config, route.domain)
    persona_adapter_path = config.persona_adapters_dir / route.persona
    if not domain_candidates:
        raise FileNotFoundError("No domain adapters available for inference.")

    last_error: Exception | None = None
    domain_adapter_path: Path | None = None
    tokenizer = None
    model = None
    for candidate in domain_candidates:
        try:
            base_model, tokenizer = _load_model_and_tokenizer(
                config.base_model_name,
                local_files_only=config.local_files_only,
            )
            model = _apply_domain_persona_adapters(
                model=base_model,
                domain_adapter_path=candidate,
                persona_adapter_path=persona_adapter_path if persona_adapter_path.exists() else None,
                domain_weight=float(route.domain_weight),
                persona_weight=float(route.persona_weight),
            )
            domain_adapter_path = candidate
            break
        except Exception as error:
            last_error = error
            continue
    if model is None or domain_adapter_path is None:
        raise RuntimeError("Failed to load any domain adapter for inference.") from last_error
    if tokenizer is None:
        raise RuntimeError("Tokenizer was not loaded for inference.")

    prompt = _build_prompt(query, route.persona)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = decoded.split("Tutor answer:", maxsplit=1)[-1].strip()

    result = {
        "query": query,
        "student_preference": student_preference,
        "teacher_preference": teacher_preference,
        "answer": answer,
        "router": router_name,
        "selected_domain": route.domain,
        "selected_persona": route.persona,
        "domain_weight": float(route.domain_weight),
        "persona_weight": float(route.persona_weight),
        "domain_adapter_path": str(domain_adapter_path),
        "persona_adapter_path": str(persona_adapter_path),
    }
    output_path = config.logs_dir / "latest_inference.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routed inference.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--student-preference", type=str, default="")
    parser.add_argument("--teacher-preference", type=str, default="")
    args = parser.parse_args()
    config = ProjectConfig()
    result = run_inference(
        args.query,
        config=config,
        max_new_tokens=args.max_new_tokens,
        student_preference=args.student_preference,
        teacher_preference=args.teacher_preference,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
