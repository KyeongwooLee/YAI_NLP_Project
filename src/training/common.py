from __future__ import annotations

import json
import inspect
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import ProjectConfig
from src.data.loader import TrainingExample


def _device_supports_fp16() -> bool:
    return torch.cuda.is_available()


def _build_model_kwargs(config: ProjectConfig) -> dict:
    kwargs: dict = {"device_map": "auto"}
    if not config.use_4bit:
        return kwargs

    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        warnings.warn(
            "USE_4BIT=1 but bitsandbytes is not installed. Falling back to non-4bit loading.",
            stacklevel=2,
        )
        return kwargs

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception as error:
        warnings.warn(
            f"4bit quantization config failed ({error}). Falling back to non-4bit loading.",
            stacklevel=2,
        )
        return kwargs
    kwargs["quantization_config"] = quantization_config
    return kwargs


def load_tokenizer(model_name: str, local_files_only: bool):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise RuntimeError(
            "Tokenizer load failed. Set BASE_MODEL_NAME to a local path or set LOCAL_FILES_ONLY=0 for online download."
        ) from error
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, config: ProjectConfig):
    kwargs = _build_model_kwargs(config)
    kwargs["local_files_only"] = config.local_files_only
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as error:
        raise RuntimeError(
            "Base model load failed. Use a local checkpoint path or set LOCAL_FILES_ONLY=0 for online download."
        ) from error
    model.config.use_cache = False
    return model


def infer_lora_target_modules(model) -> list[str]:
    module_names = [name for name, _ in model.named_modules()]
    candidates = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    )
    selected = set()
    for name in module_names:
        for candidate in candidates:
            if name.endswith(candidate):
                selected.add(candidate)
    if not selected:
        selected = {"c_attn", "c_proj"}
    return sorted(selected)


def format_example_text(example: TrainingExample, persona_instruction: str | None = None) -> str:
    style_line = (
        f"Style guidance: {persona_instruction}\n"
        if persona_instruction and persona_instruction.strip()
        else ""
    )
    return (
        "### System\n"
        "You are a helpful tutoring assistant.\n"
        f"{style_line}"
        f"### Topic\n{example.topic}\n"
        f"### Conversation Context\n{example.prompt}\n"
        "### Teacher Response\n"
        f"{example.response}"
    )


def build_hf_dataset(
    examples: Iterable[TrainingExample],
    tokenizer,
    max_seq_length: int,
    persona_instruction: str | None = None,
) -> Dataset:
    texts = [format_example_text(example, persona_instruction) for example in examples]
    dataset = Dataset.from_dict({"text": texts})

    def _tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
        )
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def build_lora_config(
    model,
    rank: int,
    alpha: int,
    dropout: float,
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=infer_lora_target_modules(model),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_lora_adapter(
    *,
    config: ProjectConfig,
    adapter_output_dir: Path,
    train_examples: list[TrainingExample],
    eval_examples: list[TrainingExample],
    rank: int,
    alpha: int,
    dropout: float,
    persona_instruction: str | None = None,
    resume_adapter_path: Path | None = None,
) -> dict:
    if not train_examples:
        raise ValueError("train_examples is empty. Cannot train adapter.")

    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.base_model_name, config.local_files_only)
    base_model = load_base_model(config.base_model_name, config)
    if resume_adapter_path is not None and resume_adapter_path.exists():
        model = PeftModel.from_pretrained(
            base_model, str(resume_adapter_path), is_trainable=True
        )
    else:
        lora_config = build_lora_config(
            base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        model = get_peft_model(base_model, lora_config)

    train_dataset = build_hf_dataset(
        train_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        persona_instruction=persona_instruction,
    )
    eval_dataset = build_hf_dataset(
        eval_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        persona_instruction=persona_instruction,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_fp16 = _device_supports_fp16()
    training_kwargs = {
        "output_dir": str(adapter_output_dir / "trainer_outputs"),
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": max(config.save_steps, 20),
        "fp16": use_fp16,
        "bf16": False,
        "report_to": [],
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "seed": config.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    else:
        training_kwargs["eval_strategy"] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})

    model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    metadata = {
        "base_model_name": config.base_model_name,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "persona_instruction": persona_instruction or "",
        "metrics": metrics,
    }
    (adapter_output_dir / "train_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def save_examples_metadata(
    output_dir: Path,
    *,
    domain: str | None = None,
    persona: str | None = None,
    extra: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"domain": domain, "persona": persona}
    if extra:
        payload.update(extra)
    (output_dir / "adapter_meta.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def dump_config(config: ProjectConfig, output_path: Path) -> None:
    payload = asdict(config)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
