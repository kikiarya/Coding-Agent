from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import PINNED_MODEL_ID, PINNED_MODEL_REVISION
from ..hashing import sha256_json
from ..models import PolicyOutput
from .prompts import (
    agent_system_prompt,
    build_agent_messages,
    fit_messages_to_context,
    parse_agent_decision,
)

# Re-export prompt helpers for callers that historically imported from model_policy.
__all__ = [
    "DecodingConfig",
    "QwenPolicyConfig",
    "TransformersAgentPolicy",
    "TransformersPolicyRuntime",
    "agent_system_prompt",
    "build_agent_messages",
    "fit_messages_to_context",
    "load_qwen_policy_config",
    "load_transformers_policy",
    "load_transformers_runtime",
    "parse_agent_decision",
]


@dataclass(frozen=True)
class DecodingConfig:
    do_sample: bool
    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class QwenPolicyConfig:
    model: dict[str, Any]
    prompt_version: str
    max_context_tokens: int
    max_decision_output_tokens: int
    decoding: dict[str, DecodingConfig]
    config_hash: str


def load_qwen_policy_config(path: Path) -> QwenPolicyConfig:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Qwen policy config must be a YAML mapping")
    if raw.get("schema_version") != "2.0.0" or raw.get("profile") != "qwen_policy_v1":
        raise ValueError("Qwen policy config must use schema 2.0.0 and qwen_policy_v1")
    model = raw.get("model", {})
    prompt = raw.get("prompt", {})
    decoding = raw.get("decoding", {})
    frozen_model = {
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "tokenizer_id": PINNED_MODEL_ID,
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "trust_remote_code": False,
        "load_in_4bit": True,
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "double_quantization": True,
    }
    if model != frozen_model:
        raise ValueError("model block differs from the frozen Qwen NF4 policy")
    if prompt.get("version") != "agent_prompt_v1":
        raise ValueError("prompt.version must be agent_prompt_v1")
    if prompt.get("max_context_tokens") != 16_384:
        raise ValueError("max_context_tokens must be 16384")
    if prompt.get("max_decision_output_tokens") != 1_024:
        raise ValueError("max_decision_output_tokens must be 1024")
    expected_decoding = {
        "evaluation": DecodingConfig(False, 0.0, 1.0, 42),
        "collection_seed_42": DecodingConfig(True, 0.7, 0.95, 42),
        "collection_seed_43": DecodingConfig(True, 0.7, 0.95, 43),
    }
    parsed_decoding: dict[str, DecodingConfig] = {}
    for name, expected in expected_decoding.items():
        value = decoding.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"decoding.{name} is missing")
        parsed = DecodingConfig(
            do_sample=value.get("do_sample"),
            temperature=float(value.get("temperature")),
            top_p=float(value.get("top_p")),
            seed=int(value.get("seed")),
        )
        if parsed != expected:
            raise ValueError(f"decoding.{name} differs from the frozen profile")
        parsed_decoding[name] = parsed
    if set(decoding) != set(expected_decoding):
        raise ValueError("decoding contains an unregistered mode")
    return QwenPolicyConfig(
        model=dict(model),
        prompt_version=prompt["version"],
        max_context_tokens=prompt["max_context_tokens"],
        max_decision_output_tokens=prompt["max_decision_output_tokens"],
        decoding=parsed_decoding,
        config_hash=sha256_json(raw),
    )


class TransformersAgentPolicy:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        config: QwenPolicyConfig,
        decoding_mode: str,
        torch_module: Any,
    ) -> None:
        if decoding_mode not in config.decoding:
            raise ValueError(f"Unknown decoding mode: {decoding_mode}")
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.decoding = config.decoding[decoding_mode]
        self.torch = torch_module
        # Seed once per episode. Re-seeding each decision would restart the RNG
        # stream and change the intended rollout sampling distribution.
        self.torch.manual_seed(self.decoding.seed)

    def decide(self, context: dict[str, Any]) -> PolicyOutput:
        max_input = (
            self.config.max_context_tokens - self.config.max_decision_output_tokens
        )
        _, input_ids, _ = fit_messages_to_context(
            self.tokenizer, context, max_input_tokens=max_input
        )
        device = getattr(self.model, "device", "cuda:0")
        model_input = self.torch.tensor([input_ids], dtype=self.torch.long, device=device)
        generation: dict[str, Any] = {
            "max_new_tokens": self.config.max_decision_output_tokens,
            "do_sample": self.decoding.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.decoding.do_sample:
            generation.update(
                {"temperature": self.decoding.temperature, "top_p": self.decoding.top_p}
            )
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(model_input, **generation)
        latency_ms = max(round((time.monotonic() - started) * 1000), 0)
        generated_ids = output[0][len(input_ids) :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return PolicyOutput(
            payload=parse_agent_decision(text),
            input_tokens=len(input_ids),
            output_tokens=len(generated_ids),
            latency_ms=latency_ms,
        )


class TransformersPolicyRuntime:
    def __init__(self, tokenizer: Any, model: Any, config: QwenPolicyConfig, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.torch = torch_module

    def policy(self, decoding_mode: str) -> TransformersAgentPolicy:
        return TransformersAgentPolicy(
            self.tokenizer,
            self.model,
            self.config,
            decoding_mode,
            self.torch,
        )


def load_transformers_runtime(
    config_path: Path,
    *,
    adapter_path: Path | None = None,
) -> TransformersPolicyRuntime:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the optional inference dependencies") from exc
    config = load_qwen_policy_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model["tokenizer_id"],
        revision=config.model["tokenizer_revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load an SFT adapter") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path.resolve()))
    model.eval()
    return TransformersPolicyRuntime(tokenizer, model, config, torch)


def load_transformers_policy(
    config_path: Path,
    decoding_mode: str,
    *,
    adapter_path: Path | None = None,
) -> TransformersAgentPolicy:
    return load_transformers_runtime(
        config_path,
        adapter_path=adapter_path,
    ).policy(decoding_mode)
