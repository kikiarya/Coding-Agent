from __future__ import annotations

# Merged experiment-stage gates: harness smoke, SFT selection, RL entry.
# Runtime/training readiness lives in training.preflight.

# --- from harness_gate.py ---

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import ValidationError
import yaml

from ..config import PINNED_MODEL_REVISION
from ..data.manifest import read_jsonl
from ..hashing import sha256_json
from ..schema import SchemaRegistry
from .rewards import RewardWeights


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.resolve()}")
    return value


def _validate_run(run_dir: Path, registry: SchemaRegistry) -> list[str]:
    errors: list[str] = []
    resolved = run_dir.resolve()
    required = ("run.json", "steps.jsonl", "episodes.jsonl")
    for name in required:
        if not (resolved / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors

    try:
        run_metadata = _load_json(resolved / "run.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid run.json: {exc}")
        return errors
    for field in ("schema_version", "run_id", "task_id", "experiment_config_hash"):
        if not run_metadata.get(field):
            errors.append(f"run.json missing {field}")

    for schema_name, filename in (
        ("step_record", "steps.jsonl"),
        ("episode_record", "episodes.jsonl"),
    ):
        try:
            values = read_jsonl(resolved / filename)
        except (OSError, ValueError) as exc:
            errors.append(f"invalid {filename}: {exc}")
            continue
        if not values:
            errors.append(f"{filename} contains no records")
            continue
        if schema_name == "episode_record" and len(values) != 1:
            errors.append(f"episodes.jsonl must contain exactly one record; found {len(values)}")
        for index, value in enumerate(values, start=1):
            try:
                registry.validate(schema_name, value)
            except ValidationError as exc:
                errors.append(f"{filename}:{index}: {exc.message}")
        if schema_name == "episode_record" and values:
            episode = values[0]
            for field in ("initial_evaluator_artifact", "evaluator_artifact"):
                artifact = episode.get(field)
                if not isinstance(artifact, str) or not (resolved / artifact).is_file():
                    errors.append(f"missing referenced {field}: {artifact!r}")
    return errors


def evaluate_harness_smoke_gate(
    stability_report_path: Path,
    rollout_report_path: Path,
    *,
    expected_tasks: int = 20,
) -> dict[str, Any]:
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    stability = _load_json(stability_report_path)
    rollout = _load_json(rollout_report_path)
    if stability.get("report_type") != "swesmith_stability_gate":
        raise ValueError("stability report has the wrong report_type")
    if rollout.get("report_type") != "swesmith_rollout_batch":
        raise ValueError("rollout report has the wrong report_type")
    if stability.get("split") != "smoke" or rollout.get("split") != "smoke":
        raise ValueError("Harness Smoke Gate accepts only smoke reports")

    minimum_coverage = math.ceil(expected_tasks * 0.95)
    results = rollout.get("results")
    if not isinstance(results, list):
        raise ValueError("rollout report results must be a list")
    registry = SchemaRegistry()
    run_checks: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("run_dir"), str):
            run_checks.append({"run_dir": None, "valid": False, "errors": ["invalid rollout result"]})
            continue
        errors = _validate_run(Path(result["run_dir"]), registry)
        run_checks.append(
            {"run_dir": result["run_dir"], "valid": not errors, "errors": errors}
        )

    task_ids = {
        str(result.get("task_id"))
        for result in results
        if isinstance(result, dict) and result.get("completed")
    }
    completed = sum(
        bool(result.get("completed")) for result in results if isinstance(result, dict)
    )
    cleanup_complete = bool(
        results
        and all(
            result.get("cleanup_ok") is True
            for result in results
            if isinstance(result, dict) and result.get("completed")
        )
    )
    schema_complete = bool(run_checks and all(item["valid"] for item in run_checks))
    checks = {
        "stability_gate_passed": stability.get("passed") is True,
        "stability_requested_count": stability.get("requested_tasks") == expected_tasks,
        "initialization_rate": float(stability.get("initialization_rate", 0.0)) >= 0.95,
        "initial_test_consistency_rate": float(
            stability.get("initial_test_consistency_rate", 0.0)
        )
        >= 0.95,
        "stability_schema_completeness": float(
            stability.get("schema_completeness", 0.0)
        )
        == 1.0,
        "rollout_task_coverage": len(task_ids) >= minimum_coverage,
        "rollout_completion": completed == len(results) and completed >= minimum_coverage,
        "rollout_batch_passed": rollout.get("passed") is True,
        "trajectory_schema_completeness": schema_complete,
        "container_cleanup": cleanup_complete,
    }
    return {
        "schema_version": "2.0.0",
        "report_type": "harness_smoke_gate",
        "expected_tasks": expected_tasks,
        "minimum_tasks_at_95_percent": minimum_coverage,
        "completed_tasks": len(task_ids),
        "completed_rollouts": completed,
        "schema_valid_runs": sum(bool(item["valid"]) for item in run_checks),
        "checks": checks,
        "passed": all(checks.values()),
        "run_checks": run_checks,
    }

def load_evaluation_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_type") != "evaluation_summary":
        raise ValueError(f"Not an evaluation summary: {path}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Evaluation summary has no metrics object: {path}")
    return value


def _task_ids(report: dict[str, Any]) -> set[str]:
    tasks = report["metrics"].get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Evaluation summary must retain task-level rows")
    task_ids = {task["task_id"] for task in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("Evaluation summary contains duplicate task_id rows")
    return task_ids


def select_checkpoint(candidates_path: Path) -> dict[str, Any]:
    registry = SchemaRegistry()
    candidates: list[dict[str, Any]] = []
    expected_tasks: set[str] | None = None
    seen_epochs: set[int] = set()
    for line_number, candidate in enumerate(read_jsonl(candidates_path), start=1):
        registry.validate("checkpoint_candidate", candidate)
        epoch = candidate["epoch"]
        if epoch in seen_epochs:
            raise ValueError(f"line {line_number}: duplicate epoch {epoch}")
        seen_epochs.add(epoch)
        report = load_evaluation_summary(Path(candidate["evaluation_report"]))
        tasks = _task_ids(report)
        if expected_tasks is None:
            expected_tasks = tasks
        elif tasks != expected_tasks:
            raise ValueError("all checkpoint candidates must use the identical dev task set")
        metrics = report["metrics"]
        resolve_rate = metrics.get("resolve_rate")
        verification_rate = metrics.get("post_patch_verification_rate")
        total_tokens = (
            metrics.get("costs", {})
            .get("all_valid", {})
            .get("total_tokens", {})
            .get("mean")
        )
        if resolve_rate is None or verification_rate is None or total_tokens is None:
            raise ValueError(f"candidate epoch {epoch} lacks a checkpoint-selection metric")
        candidates.append(
            {
                **candidate,
                "resolve_rate": float(resolve_rate),
                "resolved_count": int(metrics["resolved_count"]),
                "post_patch_verification_rate": float(verification_rate),
                "average_total_tokens": float(total_tokens),
                "dev_task_count": len(tasks),
            }
        )
    if not candidates:
        raise ValueError("checkpoint candidate file is empty")

    ranking = sorted(
        candidates,
        key=lambda item: (
            -item["resolve_rate"],
            -item["post_patch_verification_rate"],
            item["average_total_tokens"],
            item["epoch"],
        ),
    )
    return {
        "schema_version": "2.0.0",
        "report_type": "checkpoint_selection",
        "selection_rule": [
            "resolve_rate_desc",
            "post_patch_verification_rate_desc",
            "average_total_tokens_asc",
            "epoch_asc",
        ],
        "selected": ranking[0],
        "ranking": ranking,
    }


def evaluate_sft_gate(
    success_report_path: Path,
    policy_report_path: Path,
    comparison_report_path: Path,
) -> dict[str, Any]:
    success = load_evaluation_summary(success_report_path)
    policy = load_evaluation_summary(policy_report_path)
    if _task_ids(success) != _task_ids(policy):
        raise ValueError("SFT gate requires identical task_id sets")
    comparison = json.loads(
        comparison_report_path.resolve().read_text(encoding="utf-8")
    )
    if comparison.get("report_type") != "paired_model_comparison":
        raise ValueError("SFT gate requires a paired_model_comparison report")
    if comparison.get("bootstrap_samples") != 10_000 or comparison.get("seed") != 42:
        raise ValueError("SFT gate requires 10,000 bootstrap samples with seed 42")
    paired = comparison.get("comparison", {})
    if paired.get("task_count") != len(_task_ids(success)):
        raise ValueError("Paired comparison task count differs from SFT summaries")
    bootstrap = paired.get("paired_bootstrap", {})
    required_bootstrap = {
        "resolve_rate",
        "post_patch_verification_rate",
        "duplicate_action_rate",
        "invalid_action_rate",
    }
    if not required_bootstrap.issubset(bootstrap):
        raise ValueError("Paired comparison is missing an SFT gate metric")
    success_metrics = success["metrics"]
    policy_metrics = policy["metrics"]
    required = [
        "resolve_rate",
        "post_patch_verification_rate",
        "duplicate_action_rate",
        "invalid_action_rate",
    ]
    if any(success_metrics.get(metric) is None for metric in required) or any(
        policy_metrics.get(metric) is None for metric in required
    ):
        raise ValueError("SFT gate metrics must have non-zero denominators")

    resolve_change = bootstrap["resolve_rate"]["difference_candidate_minus_baseline"]
    verification_change = bootstrap["post_patch_verification_rate"][
        "difference_candidate_minus_baseline"
    ]
    success_duplicate = bootstrap["duplicate_action_rate"]["baseline"]
    policy_duplicate = bootstrap["duplicate_action_rate"]["candidate"]
    invalid_change = bootstrap["invalid_action_rate"][
        "difference_candidate_minus_baseline"
    ]
    if any(
        value is None
        for value in (
            resolve_change,
            verification_change,
            success_duplicate,
            policy_duplicate,
            invalid_change,
        )
    ):
        raise ValueError("SFT gate paired metrics must have non-zero joint denominators")
    duplicate_relative_reduction = (
        (success_duplicate - policy_duplicate) / success_duplicate
        if success_duplicate > 0
        else None
    )
    resolve_gate = resolve_change >= -0.05
    verification_gate = verification_change >= 0.10
    duplicate_gate = (
        duplicate_relative_reduction is not None
        and duplicate_relative_reduction >= 0.20
    )
    mechanism_gate = verification_gate or duplicate_gate
    invalid_gate = invalid_change <= 0.05

    def confidence_interval(metric: str) -> list[float] | None:
        value = bootstrap[metric].get("confidence_interval_95")
        return value if isinstance(value, list) and len(value) == 2 else None

    resolve_ci = confidence_interval("resolve_rate")
    verification_ci = confidence_interval("post_patch_verification_rate")
    duplicate_ci = confidence_interval("duplicate_action_rate")
    invalid_ci = confidence_interval("invalid_action_rate")
    statistical_support = bool(
        resolve_ci is not None
        and resolve_ci[0] >= -0.05
        and invalid_ci is not None
        and invalid_ci[1] <= 0.05
        and (
            verification_gate
            and verification_ci is not None
            and verification_ci[0] > 0
            or duplicate_gate
            and duplicate_ci is not None
            and duplicate_ci[1] < 0
        )
    )
    representative_failures = [
        task["task_id"]
        for task in policy_metrics.get("tasks", [])
        if task.get("environment_valid") is True and task.get("resolved") is False
    ][:10]
    return {
        "schema_version": "2.0.0",
        "report_type": "sft_gate",
        "task_count": len(_task_ids(success)),
        "thresholds": {
            "minimum_resolve_change": -0.05,
            "minimum_verification_change": 0.10,
            "minimum_duplicate_relative_reduction": 0.20,
            "maximum_invalid_action_rate_change": 0.05,
        },
        "observed": {
            "resolve_change_policy_minus_success": resolve_change,
            "verification_change_policy_minus_success": verification_change,
            "duplicate_relative_reduction": duplicate_relative_reduction,
            "invalid_action_rate_change": invalid_change,
        },
        "checks": {
            "resolve_non_inferiority": resolve_gate,
            "verification_improvement": verification_gate,
            "duplicate_reduction": duplicate_gate,
            "mechanism_or_gate": mechanism_gate,
            "invalid_action_non_degradation": invalid_gate,
        },
        "paired_statistics": paired,
        "statistical_support": statistical_support,
        "claim_status": "supported" if statistical_support else "exploratory",
        "representative_failure_task_ids": representative_failures,
        "passed": resolve_gate and mechanism_gate and invalid_gate,
    }

PINNED_TRL_REVISION = "e17a8ae6f10945c2f7a3d31e0267ec05f29d6f20"
FROZEN_REWARD = RewardWeights().to_config()


def load_rl_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RL config must be a YAML mapping")
    if raw.get("schema_version") != "2.0.0":
        raise ValueError("RL config schema_version must be 2.0.0")
    if raw.get("profile") != "conditional_agentic_grpo_v1":
        raise ValueError("RL config profile must be conditional_agentic_grpo_v1")
    if raw.get("status") not in {"gated", "conditional_plan"}:
        raise ValueError("RL status must require gate validation")
    trainer = raw.get("trainer", {})
    profiles = raw.get("profiles", {})
    environment = raw.get("environment", {})
    harness_policy = raw.get("harness_policy", {})
    gate = raw.get("gate", {})
    if trainer.get("revision") != PINNED_TRL_REVISION:
        raise ValueError("TRL revision is not frozen")
    if trainer.get("minimum_transformers") != "5.2.0":
        raise ValueError("environment_factory requires transformers>=5.2.0")
    if trainer.get("vllm_mode") != "colocate" or trainer.get("vllm_enable_sleep_mode") is not True:
        raise ValueError("formal GRPO must use colocate and sleep mode")
    frozen_trainer = {
        "initialization": "policy_sft",
        "peft": "qlora",
        "num_generations": 4,
        "max_tool_calling_iterations": 12,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "kl_beta": 0.001,
        "seed": 42,
    }
    for key, expected in frozen_trainer.items():
        if trainer.get(key) != expected:
            raise ValueError(f"trainer.{key} differs from the frozen V2 value")
    expected_profiles = {
        "smoke": {"stable_train_tasks": 8, "update_steps": 50},
        "formal": {
            "stable_train_tasks": 64,
            "update_steps": 500,
            "eval_every_steps": 50,
            "stop_after_evals_without_improvement": 3,
            "maximum_invalid_action_rate": 0.05,
            "maximum_resolve_drop": 0.05,
        },
    }
    if profiles != expected_profiles:
        raise ValueError("GRPO smoke/formal profiles differ from the frozen V2 values")
    if raw.get("reward") != FROZEN_REWARD:
        raise ValueError("RL reward differs from the frozen V2 reward")
    if environment.get("max_tool_steps") != 30:
        raise ValueError("environment max_tool_steps must be 30")
    if environment.get("max_test_calls") != 8:
        raise ValueError("max_test_calls must be 8")
    if environment.get("independent_environment_per_rollout") is not True:
        raise ValueError("each rollout must use an independent environment")
    expected_harness_policy = {
        "variant": "adaptive_v1",
        "history_limit": 8,
        "verification_mode": "required",
        "recovery_mode": "required",
    }
    if harness_policy != expected_harness_policy:
        raise ValueError("GRPO must use the frozen adaptive Harness Policy profile")
    if gate.get("minimum_gpu_memory_gb") != 80:
        raise ValueError("RL gate requires an 80GB GPU")
    if gate.get("smoke_tasks") != 8 or gate.get("audit_trajectories") != 20:
        raise ValueError("RL smoke/audit task counts differ from V2")
    if gate.get("require_chat_template_preflight") is not True:
        raise ValueError("RL gate must require chat-template preflight")
    return raw | {"config_hash": sha256_json(raw)}


def evaluate_rl_gate(evidence_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_rl_config(config_path)
    evidence = json.loads(evidence_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("RL gate evidence must be a JSON object")
    SchemaRegistry().validate("rl_gate_evidence", evidence)
    sft_gate_path = Path(evidence["sft_gate_report"]).resolve()
    sft_gate = json.loads(sft_gate_path.read_text(encoding="utf-8"))
    template_path = Path(evidence["chat_template_preflight_report"]).resolve()
    template_report = json.loads(template_path.read_text(encoding="utf-8"))
    checkpoint = Path(evidence["policy_checkpoint"]).resolve()
    gate_config = config["gate"]
    checks = {
        "sft_gate_passed": sft_gate.get("passed") is True,
        "policy_checkpoint_exists": checkpoint.exists(),
        "policy_checkpoint_frozen": evidence["policy_checkpoint_frozen"] is True,
        "chat_template_prefix_preserving": bool(
            template_report.get("report_type") == "grpo_chat_template_preflight"
            and template_report.get("passed") is True
            and template_report.get("tools_passed") == 6
            and template_report.get("tokenizer_revision") == PINNED_MODEL_REVISION
        ),
        "gpu_memory": evidence["gpu_memory_gb"] >= gate_config["minimum_gpu_memory_gb"],
        "smoke_tasks": evidence["smoke_tasks_completed"] >= gate_config["smoke_tasks"],
        "no_smoke_oom": evidence["smoke_oom_count"] == 0,
        "no_gradient_anomaly": evidence["smoke_gradient_anomaly_count"] == 0,
        "no_container_state_leakage": evidence["smoke_state_isolation_passed"] is True,
        "audit_coverage": evidence["audit_trajectories"] >= gate_config["audit_trajectories"],
        "no_reward_hacking": evidence["reward_hacking_findings"] == 0,
        "reward_component_logs": evidence["reward_component_logs"] is True,
        "kl_loss_logs": evidence["kl_loss_logs"] is True,
        "loss_logs": evidence["loss_logs"] is True,
        "invalid_action_logs": evidence["invalid_action_logs"] is True,
        "episode_length_logs": evidence["episode_length_logs"] is True,
        "same_harness_and_budget": evidence["same_harness_and_budget"] is True,
    }
    return {
        "schema_version": "2.0.0",
        "report_type": "rl_gate",
        "config_hash": config["config_hash"],
        "checks": checks,
        "passed": all(checks.values()),
    }
