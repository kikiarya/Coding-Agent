from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from ..config import ExperimentConfig, load_experiment_config
from ..data.manifest import EXTERNAL_SUBSET_COUNT, read_jsonl
from .metrics import aggregate_metrics, collect_task_metrics, load_gold_files
from .stats import compare_models


ARM_NAMES = ("base", "model_only", "harness_only", "joint")


@dataclass(frozen=True)
class TwoLayerArm:
    name: str
    model_policy: str
    harness_policy: str
    expected_adapter: bool


ARM_DEFINITIONS = {
    "base": TwoLayerArm("base", "base_model", "static_v1", False),
    "model_only": TwoLayerArm("model_only", "trained_model", "static_v1", True),
    "harness_only": TwoLayerArm("harness_only", "base_model", "adaptive_v1", False),
    "joint": TwoLayerArm("joint", "trained_model", "adaptive_v1", True),
}


def _runtime_controls(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir.resolve() / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Run metadata has no config mapping: {run_dir}")
    model = config.get("model", {})
    harness = config.get("harness", {})
    repository = config.get("repository", {})
    policy = metadata.get("harness_policy", config.get("harness_policy", {}))
    if not all(isinstance(value, dict) for value in (model, harness, repository, policy)):
        raise ValueError(f"Run metadata has malformed control blocks: {run_dir}")
    return {
        "model_id": model.get("model_id"),
        "model_revision": model.get("model_revision"),
        "prompt_version": model.get("prompt_version"),
        "adapter_id": model.get("adapter_id"),
        "harness": harness,
        "repository": repository,
        "harness_policy": policy,
    }


def _validate_runtime_arms(
    arm_run_dirs: dict[str, list[Path]],
) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for arm in ARM_NAMES:
        expected = ARM_DEFINITIONS[arm]
        for run_dir in arm_run_dirs[arm]:
            value = _runtime_controls(run_dir)
            if bool(value["adapter_id"]) != expected.expected_adapter:
                raise ValueError(
                    f"Run {run_dir} does not match {arm}: adapter presence is wrong"
                )
            if value["harness_policy"].get("variant") != expected.harness_policy:
                raise ValueError(
                    f"Run {run_dir} does not match {arm}: Harness Policy variant is wrong"
                )
            invariant = {
                "model_id": value["model_id"],
                "model_revision": value["model_revision"],
                "prompt_version": value["prompt_version"],
                "harness": value["harness"],
                "repository": value["repository"],
            }
            if reference is None:
                reference = invariant
            elif invariant != reference:
                raise ValueError(
                    f"Run {run_dir} changes a frozen two-layer control variable"
                )
            controls.setdefault(arm, value)
    return controls


def classify_arm(config: ExperimentConfig) -> str:
    """Classify a frozen config while keeping the four-arm interpretation explicit."""

    has_adapter = bool(config.model.adapter_id)
    variant = config.harness_policy.variant
    if not has_adapter and variant == "static_v1":
        return "base"
    if has_adapter and variant == "static_v1":
        return "model_only"
    if not has_adapter and variant == "adaptive_v1":
        return "harness_only"
    if has_adapter and variant == "adaptive_v1":
        return "joint"
    raise ValueError(
        "Unsupported two-layer arm: adapter presence and harness policy variant "
        f"are not a known combination ({has_adapter=}, {variant=})"
    )


def validate_two_layer_configs(config_paths: dict[str, Path]) -> dict[str, Any]:
    """Check that the four arms differ only in the intended optimization layer."""

    missing = sorted(set(ARM_NAMES) - set(config_paths))
    extra = sorted(set(config_paths) - set(ARM_NAMES))
    if missing or extra:
        raise ValueError(f"Two-layer configs must contain exactly {ARM_NAMES}; {missing=}, {extra=}")

    configs = {name: load_experiment_config(path) for name, path in config_paths.items()}
    classifications = {name: classify_arm(config) for name, config in configs.items()}
    if classifications != {name: name for name in ARM_NAMES}:
        raise ValueError(
            "Two-layer config names do not match their adapter/policy combinations: "
            f"{classifications}"
        )

    reference = configs["base"]
    mismatches: dict[str, list[str]] = {}
    for name, config in configs.items():
        differences: list[str] = []
        if config.model.model_id != reference.model.model_id:
            differences.append("model.model_id")
        if config.model.model_revision != reference.model.model_revision:
            differences.append("model.model_revision")
        if config.model.prompt_version != reference.model.prompt_version:
            differences.append("model.prompt_version")
        if config.harness != reference.harness:
            differences.append("harness")
        if config.repository != reference.repository:
            differences.append("repository")
        if config.run.seed != reference.run.seed:
            differences.append("run.seed")
        if differences:
            mismatches[name] = differences
    if mismatches:
        raise ValueError(f"Two-layer configs have unfrozen control variables: {mismatches}")

    return {
        "schema_version": "2.0.0",
        "report_type": "two_layer_config_validation",
        "passed": True,
        "arms": {
            name: {
                "config": str(config_paths[name].resolve()),
                "config_hash": configs[name].config_hash,
                "model_policy": ARM_DEFINITIONS[name].model_policy,
                "harness_policy": configs[name].harness_policy.variant,
                "adapter_id": configs[name].model.adapter_id,
            }
            for name in ARM_NAMES
        },
        "frozen_controls": {
            "model_id": reference.model.model_id,
            "model_revision": reference.model.model_revision,
            "prompt_version": reference.model.prompt_version,
            "harness": asdict(reference.harness),
            "repository": asdict(reference.repository),
            "seed": reference.run.seed,
        },
    }


def compare_two_layer_arms(
    arm_run_dirs: dict[str, Iterable[Path]],
    *,
    gold_files: Path | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    expected_task_count: int | None = None,
    expected_split: str | None = None,
) -> dict[str, Any]:
    """Produce paired ablations for Base/Model-only/Harness-only/Joint.

    Each arm must contain exactly one run per task. The evaluator intentionally
    refuses duplicate task IDs so repeated seeds must be summarized separately
    or reduced by a pre-registered aggregation rule before this command.
    """

    missing = sorted(set(ARM_NAMES) - set(arm_run_dirs))
    extra = sorted(set(arm_run_dirs) - set(ARM_NAMES))
    if missing or extra:
        raise ValueError(f"Arm runs must contain exactly {ARM_NAMES}; {missing=}, {extra=}")
    normalized_run_dirs = {name: list(arm_run_dirs[name]) for name in ARM_NAMES}
    runtime_controls = _validate_runtime_arms(normalized_run_dirs)
    gold = load_gold_files(gold_files) if gold_files is not None else None
    tasks = {
        name: collect_task_metrics(normalized_run_dirs[name], gold_by_task=gold)
        for name in ARM_NAMES
    }
    if expected_task_count is not None:
        if expected_task_count <= 0:
            raise ValueError("expected_task_count must be positive")
        wrong_counts = {
            name: len(values)
            for name, values in tasks.items()
            if len(values) != expected_task_count
        }
        if wrong_counts:
            raise ValueError(
                "Two-layer comparison has the wrong task count: "
                f"expected={expected_task_count}, actual={wrong_counts}"
            )
    if expected_split is not None:
        if not expected_split:
            raise ValueError("expected_split must be non-empty")
        for arm in ARM_NAMES:
            for run_dir in normalized_run_dirs[arm]:
                episodes = read_jsonl(run_dir.resolve() / "episodes.jsonl")
                splits = {episode.get("split") for episode in episodes}
                if splits != {expected_split}:
                    found = sorted(value for value in splits if value is not None)
                    raise ValueError(
                        f"Run {run_dir} does not match expected split={expected_split!r}: "
                        f"found={found}"
                    )
    comparisons = {
        "base_vs_model_only": compare_models(
            tasks["base"], tasks["model_only"], samples=bootstrap_samples, seed=seed
        ),
        "base_vs_harness_only": compare_models(
            tasks["base"], tasks["harness_only"], samples=bootstrap_samples, seed=seed
        ),
        "base_vs_joint": compare_models(
            tasks["base"], tasks["joint"], samples=bootstrap_samples, seed=seed
        ),
        "model_only_vs_joint": compare_models(
            tasks["model_only"], tasks["joint"], samples=bootstrap_samples, seed=seed
        ),
        "harness_only_vs_joint": compare_models(
            tasks["harness_only"], tasks["joint"], samples=bootstrap_samples, seed=seed
        ),
    }
    aggregate = {name: aggregate_metrics(value) for name, value in tasks.items()}
    base_calls = aggregate["base"]["costs"]["all_valid"]["tool_calls"]["mean"]
    joint_calls = aggregate["joint"]["costs"]["all_valid"]["tool_calls"]["mean"]
    tool_call_reduction = (
        (base_calls - joint_calls) / base_calls
        if base_calls is not None and joint_calls is not None and base_calls > 0
        else None
    )
    resume_eligible = (
        expected_task_count == EXTERNAL_SUBSET_COUNT and expected_split == "external"
    )
    return {
        "schema_version": "2.0.0",
        "report_type": "two_layer_ablation",
        "arms": list(ARM_NAMES),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "evaluation_scope": {
            "expected_task_count": expected_task_count,
            "expected_split": expected_split,
            "task_count_by_arm": {
                name: len(values) for name, values in tasks.items()
            },
        },
        "resume_evidence": {
            "eligible": resume_eligible,
            "required_task_count": EXTERNAL_SUBSET_COUNT,
            "required_split": "external",
            "base_resolved_count": aggregate["base"]["resolved_count"],
            "joint_resolved_count": aggregate["joint"]["resolved_count"],
            "joint_vs_base_resolve_rate_change_pp": (
                comparisons["base_vs_joint"]["paired_bootstrap"]["resolve_rate"][
                    "difference_candidate_minus_baseline"
                ]
                * 100
                if comparisons["base_vs_joint"]["paired_bootstrap"]["resolve_rate"][
                    "difference_candidate_minus_baseline"
                ]
                is not None
                else None
            ),
            "joint_vs_base_recovery_rate_change_pp": (
                comparisons["base_vs_joint"]["paired_bootstrap"][
                    "episode_recovery_rate"
                ]["difference_candidate_minus_baseline"]
                * 100
                if comparisons["base_vs_joint"]["paired_bootstrap"][
                    "episode_recovery_rate"
                ]["difference_candidate_minus_baseline"]
                is not None
                else None
            ),
            "joint_vs_base_average_tool_call_reduction_pct": (
                tool_call_reduction * 100
                if tool_call_reduction is not None
                else None
            ),
        },
        "runtime_controls": runtime_controls,
        "aggregate": aggregate,
        "joint_vs_base_resume_metrics": {
            "resolve_rate_change": comparisons["base_vs_joint"]["paired_bootstrap"]["resolve_rate"][
                "difference_candidate_minus_baseline"
            ],
            "recovery_rate_change": comparisons["base_vs_joint"]["paired_bootstrap"][
                "episode_recovery_rate"
            ]["difference_candidate_minus_baseline"],
            "average_tool_call_reduction": tool_call_reduction,
        },
        "comparisons": comparisons,
    }
