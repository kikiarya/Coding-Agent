from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .hashing import sha256_json

PINNED_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
PINNED_MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


@dataclass(frozen=True)
class RunConfig:
    seed: int
    artifacts_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_revision: str
    adapter_id: str | None
    prompt_version: str


@dataclass(frozen=True)
class HarnessConfig:
    harness_version: str
    max_tool_steps: int
    max_test_calls: int
    max_total_tokens: int
    max_context_tokens: int
    max_decision_output_tokens: int
    tool_timeout_seconds: int
    task_timeout_seconds: int


@dataclass(frozen=True)
class HarnessPolicyConfig:
    variant: str
    history_limit: int
    verification_mode: str
    recovery_mode: str


@dataclass(frozen=True)
class RepositoryConfig:
    test_command: tuple[str, ...]
    default_test_args: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    run: RunConfig
    model: ModelConfig
    harness: HarnessConfig
    harness_policy: HarnessPolicyConfig
    repository: RepositoryConfig
    config_hash: str
    raw: dict[str, Any]


def experiment_config_with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """Create an auditable per-rollout config without mutating the frozen source config."""
    raw = deepcopy(config.raw)
    raw["run"]["seed"] = seed
    return replace(
        config,
        run=replace(config.run, seed=seed),
        config_hash=sha256_json(raw),
        raw=raw,
    )


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    if raw.get("schema_version") != "2.0.0":
        raise ValueError("schema_version must be 2.0.0")

    run = raw.get("run", {})
    model = raw.get("model", {})
    harness = raw.get("harness", {})
    repository = raw.get("repository", {})
    harness_policy = raw.get("harness_policy", {})
    if not all(isinstance(value, dict) for value in (run, model, harness, repository, harness_policy)):
        raise ValueError("run/model/harness/repository must be YAML mappings")

    policy_variant = str(harness_policy.get("variant", "static_v1"))
    if policy_variant not in {"static_v1", "adaptive_v1"}:
        raise ValueError("harness_policy.variant must be static_v1 or adaptive_v1")
    history_limit = _positive_int(
        harness_policy.get("history_limit", 1_000_000),
        "harness_policy.history_limit",
    )
    verification_mode = str(harness_policy.get("verification_mode", "advisory"))
    recovery_mode = str(harness_policy.get("recovery_mode", "advisory"))
    if verification_mode not in {"advisory", "required"}:
        raise ValueError("harness_policy.verification_mode must be advisory or required")
    if recovery_mode not in {"advisory", "required"}:
        raise ValueError("harness_policy.recovery_mode must be advisory or required")

    test_command = repository.get("test_command")
    default_test_args = repository.get("default_test_args", [])
    if not isinstance(test_command, list) or not test_command or not all(
        isinstance(item, str) and item for item in test_command
    ):
        raise ValueError("repository.test_command must be a non-empty string list")
    if not isinstance(default_test_args, list) or not all(
        isinstance(item, str) and item for item in default_test_args
    ):
        raise ValueError("repository.default_test_args must be a string list")

    return ExperimentConfig(
        schema_version="2.0.0",
        run=RunConfig(seed=int(run["seed"]), artifacts_dir=Path(run["artifacts_dir"])),
        model=ModelConfig(
            model_id=str(model["model_id"]),
            model_revision=str(model["model_revision"]),
            adapter_id=model.get("adapter_id"),
            prompt_version=str(model["prompt_version"]),
        ),
        harness=HarnessConfig(
            harness_version=str(harness["harness_version"]),
            max_tool_steps=_positive_int(harness["max_tool_steps"], "max_tool_steps"),
            max_test_calls=_positive_int(harness["max_test_calls"], "max_test_calls"),
            max_total_tokens=_positive_int(harness["max_total_tokens"], "max_total_tokens"),
            max_context_tokens=_positive_int(harness["max_context_tokens"], "max_context_tokens"),
            max_decision_output_tokens=_positive_int(
                harness["max_decision_output_tokens"], "max_decision_output_tokens"
            ),
            tool_timeout_seconds=_positive_int(
                harness["tool_timeout_seconds"], "tool_timeout_seconds"
            ),
            task_timeout_seconds=_positive_int(
                harness["task_timeout_seconds"], "task_timeout_seconds"
            ),
        ),
        harness_policy=HarnessPolicyConfig(
            variant=policy_variant,
            history_limit=history_limit,
            verification_mode=verification_mode,
            recovery_mode=recovery_mode,
        ),
        repository=RepositoryConfig(
            test_command=tuple(test_command),
            default_test_args=tuple(default_test_args),
        ),
        config_hash=sha256_json(raw),
        raw=raw,
    )
