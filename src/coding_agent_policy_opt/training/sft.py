from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..data.gate import validate_decision_dataset
from ..hashing import canonical_json, sha256_json
from ..data.manifest import read_jsonl
from ..data.release import validate_dataset_release_manifest


from ..config import PINNED_MODEL_ID, PINNED_MODEL_REVISION


@dataclass(frozen=True)
class SFTDataConfig:
    dataset: str
    path: Path
    manifest: Path
    expected_samples: int
    allowed_split: str
    min_review_fraction: float
    replay_ledger: Path | None
    release_manifest: Path


@dataclass(frozen=True)
class SFTTrainingConfig:
    quantization: str
    load_in_4bit: bool
    compute_dtype: str
    double_quantization: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    max_sequence_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    epochs: int
    gradient_checkpointing: bool
    optimizer: str
    seed: int
    save_strategy: str


@dataclass(frozen=True)
class SFTConfig:
    schema_version: str
    profile: str
    experiment_id: str
    status: str
    model: dict[str, Any]
    data: SFTDataConfig
    training: SFTTrainingConfig
    output_dir: Path
    config_hash: str
    raw: dict[str, Any]


FROZEN_TRAINING_VALUES: dict[str, Any] = {
    "quantization": "nf4",
    "load_in_4bit": True,
    "compute_dtype": "bfloat16",
    "double_quantization": True,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "max_sequence_length": 4096,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0002,
    "scheduler": "cosine",
    "warmup_ratio": 0.03,
    "epochs": 2,
    "gradient_checkpointing": True,
    "optimizer": "paged_adamw_8bit",
    "seed": 42,
    "save_strategy": "epoch",
}


def load_sft_config(path: Path) -> SFTConfig:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SFT config must be a YAML mapping")
    if raw.get("schema_version") != "2.0.0" or raw.get("profile") != "mvp_qlora_v1":
        raise ValueError("SFT config must use schema 2.0.0 and profile mvp_qlora_v1")
    model = raw.get("model")
    data = raw.get("data")
    training = raw.get("training")
    if not all(isinstance(value, dict) for value in (model, data, training)):
        raise ValueError("model/data/training must be YAML mappings")
    if model.get("model_id") != PINNED_MODEL_ID:
        raise ValueError(f"model_id must be {PINNED_MODEL_ID}")
    if model.get("model_revision") != PINNED_MODEL_REVISION:
        raise ValueError("model_revision is not the frozen Qwen commit")
    if model.get("tokenizer_id") != PINNED_MODEL_ID:
        raise ValueError("tokenizer_id must match the frozen model")
    if model.get("tokenizer_revision") != PINNED_MODEL_REVISION:
        raise ValueError("tokenizer_revision must match the frozen model commit")
    if model.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must be false")
    for key, expected in FROZEN_TRAINING_VALUES.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be frozen to {expected!r}")
    if data.get("dataset") not in {"D_success", "D_policy"}:
        raise ValueError("data.dataset must be D_success or D_policy")
    if data.get("expected_samples") != 600:
        raise ValueError("data.expected_samples must be 600")
    if data.get("allowed_split") != "train":
        raise ValueError("SFT data must come from the train split")
    replay_ledger_value = data.get("replay_ledger")
    release_manifest_value = data.get("release_manifest")
    if not isinstance(release_manifest_value, str):
        raise ValueError("SFT data requires a dataset release manifest")
    if data.get("dataset") == "D_policy" and not isinstance(replay_ledger_value, str):
        raise ValueError("D_policy requires a corrected-sample replay ledger")
    if data.get("dataset") == "D_success" and replay_ledger_value is not None:
        raise ValueError("D_success must not declare a corrected-sample replay ledger")
    review_fraction = float(data.get("min_review_fraction"))
    if not 0.0 <= review_fraction <= 1.0:
        raise ValueError("data.min_review_fraction must be between 0 and 1")
    if raw.get("status") not in {"configured", "planned", "implemented", "verified"}:
        raise ValueError("status must be configured or a supported legacy lifecycle value")

    return SFTConfig(
        schema_version="2.0.0",
        profile="mvp_qlora_v1",
        experiment_id=str(raw["experiment_id"]),
        status=str(raw["status"]),
        model=dict(model),
        data=SFTDataConfig(
            dataset=str(data["dataset"]),
            path=Path(data["path"]),
            manifest=Path(data["manifest"]),
            expected_samples=int(data["expected_samples"]),
            allowed_split=str(data["allowed_split"]),
            min_review_fraction=review_fraction,
            replay_ledger=(
                Path(replay_ledger_value)
                if isinstance(replay_ledger_value, str)
                else None
            ),
            release_manifest=Path(release_manifest_value),
        ),
        training=SFTTrainingConfig(**training),
        output_dir=Path(raw["output_dir"]),
        config_hash=sha256_json(raw),
        raw=raw,
    )


def preflight_sft(config: SFTConfig) -> dict[str, Any]:
    data_report = validate_decision_dataset(
        config.data.path,
        config.data.dataset,
        expected_count=config.data.expected_samples,
        min_review_fraction=config.data.min_review_fraction,
        manifest_path=config.data.manifest,
        allowed_split=config.data.allowed_split,
        replay_ledger_path=config.data.replay_ledger,
    )
    release_report = validate_dataset_release_manifest(
        config.data.release_manifest,
        config.data.path,
        config.data.dataset,
        config.data.manifest,
        config.data.replay_ledger,
        expected_count=config.data.expected_samples,
    )
    return {
        "schema_version": "2.0.0",
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "dataset_gate": data_report.to_dict(),
        "dataset_release": release_report,
        "ready_for_tokenizer_preflight": data_report.valid and release_report["passed"],
    }


class ChatTemplateTokenizer(Protocol):
    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


def encode_decision_sample(
    tokenizer: ChatTemplateTokenizer,
    sample: dict[str, Any],
    max_sequence_length: int,
) -> dict[str, list[int]]:
    messages = sample["messages"]
    target_message = {"role": "assistant", "content": canonical_json(sample["target"])}
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            [*messages, target_message], tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"chat template is not prefix-preserving for sample {sample.get('sample_id')}"
        )
    if len(full_ids) > max_sequence_length:
        raise ValueError(
            f"sample {sample.get('sample_id')} has {len(full_ids)} tokens; "
            f"maximum is {max_sequence_length}"
        )
    supervised_tokens = len(full_ids) - len(prompt_ids)
    if supervised_tokens <= 0:
        raise ValueError(f"sample {sample.get('sample_id')} has no target tokens")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }


def analyze_tokenized_dataset(
    path: Path,
    tokenizer: ChatTemplateTokenizer,
    *,
    max_sequence_length: int = 4096,
) -> dict[str, Any]:
    samples = read_jsonl(path)
    total_sequence_tokens = 0
    total_target_tokens = 0
    maximum_sequence_tokens = 0
    for sample in samples:
        encoded = encode_decision_sample(tokenizer, sample, max_sequence_length)
        sequence_tokens = len(encoded["input_ids"])
        target_tokens = sum(label != -100 for label in encoded["labels"])
        total_sequence_tokens += sequence_tokens
        total_target_tokens += target_tokens
        maximum_sequence_tokens = max(maximum_sequence_tokens, sequence_tokens)
    return {
        "samples": len(samples),
        "total_sequence_tokens": total_sequence_tokens,
        "total_target_tokens": total_target_tokens,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "max_sequence_length": max_sequence_length,
    }


def run_tokenizer_preflight(config: SFTConfig) -> dict[str, Any]:
    preflight = preflight_sft(config)
    if not preflight["ready_for_tokenizer_preflight"]:
        raise ValueError("SFT dataset gate failed before tokenization")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers before tokenizer preflight") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        config.model["tokenizer_id"],
        revision=config.model["tokenizer_revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    token_counts = analyze_tokenized_dataset(
        config.data.path,
        tokenizer,
        max_sequence_length=config.training.max_sequence_length,
    )
    return {
        "schema_version": "2.0.0",
        "report_type": "tokenizer_preflight",
        "experiment_id": config.experiment_id,
        "dataset": config.data.dataset,
        "config_hash": config.config_hash,
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        **token_counts,
    }


def evaluate_token_budget_gate(
    success_report_path: Path,
    policy_report_path: Path,
) -> dict[str, Any]:
    success = json.loads(success_report_path.resolve().read_text(encoding="utf-8"))
    policy = json.loads(policy_report_path.resolve().read_text(encoding="utf-8"))
    for name, report, dataset in (
        ("success", success, "D_success"),
        ("policy", policy, "D_policy"),
    ):
        if report.get("report_type") != "tokenizer_preflight":
            raise ValueError(f"{name} report is not a tokenizer preflight")
        if report.get("dataset") != dataset:
            raise ValueError(f"{name} report has the wrong dataset")
        if report.get("samples") != 600:
            raise ValueError(f"{name} report must contain 600 samples")
    if success["tokenizer_id"] != policy["tokenizer_id"] or success[
        "tokenizer_revision"
    ] != policy["tokenizer_revision"]:
        raise ValueError("token budget comparison requires the same tokenizer revision")
    success_tokens = int(success["total_target_tokens"])
    policy_tokens = int(policy["total_target_tokens"])
    denominator = max(success_tokens, policy_tokens)
    relative_difference = (
        abs(success_tokens - policy_tokens) / denominator if denominator else 0.0
    )
    return {
        "schema_version": "2.0.0",
        "success_target_tokens": success_tokens,
        "policy_target_tokens": policy_tokens,
        "relative_difference": relative_difference,
        "maximum_relative_difference": 0.01,
        "passed": relative_difference <= 0.01,
    }


class DecisionDataset:
    def __init__(
        self,
        path: Path,
        tokenizer: ChatTemplateTokenizer,
        max_sequence_length: int,
    ) -> None:
        self.examples = [
            encode_decision_sample(tokenizer, sample, max_sequence_length)
            for sample in read_jsonl(path)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class TargetOnlyDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)

        def padded(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_length - len(values))

        return {
            "input_ids": torch.tensor(
                [padded(feature["input_ids"], self.pad_token_id) for feature in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [padded(feature["attention_mask"], 0) for feature in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [padded(feature["labels"], -100) for feature in features],
                dtype=torch.long,
            ),
        }


def run_sft_training(config: SFTConfig) -> dict[str, Any]:
    preflight = preflight_sft(config)
    if not preflight["ready_for_tokenizer_preflight"]:
        raise ValueError("SFT dataset gate failed")
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse SFT output directory: {output_dir}")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install the optional train dependencies before SFT") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen SFT profile requires BF16-capable CUDA hardware")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model["tokenizer_id"],
        revision=config.model["tokenizer_revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = DecisionDataset(
        config.data.path, tokenizer, config.training.max_sequence_length
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model["model_id"],
        revision=config.model["model_revision"],
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=config.training.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frozen_sft_config.json").write_text(
        json.dumps(config.raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.scheduler,
        warmup_ratio=config.training.warmup_ratio,
        num_train_epochs=config.training.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim=config.training.optimizer,
        save_strategy=config.training.save_strategy,
        logging_steps=1,
        seed=config.training.seed,
        data_seed=config.training.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=TargetOnlyDataCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    trainer.save_state()
    return {
        "schema_version": "2.0.0",
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "output_dir": str(output_dir),
        "samples": len(dataset),
        "train_metrics": train_result.metrics,
    }
