from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .harness.artifacts import RunArtifactStore
from .harness.checkpoint import replay_latest_checkpoint
from .agent.api_policy import (
    load_api_model_policy_config,
    resolve_api_model_environment,
)
from .harness.runner import run_api_repository
from .config import load_experiment_config
from .data.gate import validate_decision_dataset
from .data.builder import (
    apply_review_decisions,
    assemble_policy_dataset,
    assemble_success_dataset,
    select_review_queue,
)
from .harness.environment import RepositoryEnvironment, RepositoryProfile
from .evaluation.exploration import analyze_exploration
from .evaluation.gates import evaluate_harness_smoke_gate
from .harness.runner import AgentHarness
from .harness.policy import select_harness_candidate
from .data.manifest import freeze_external_manifest, merge_task_manifests, validate_manifest
from .evaluation.metrics import aggregate_metrics, collect_task_metrics, load_gold_files
from .evaluation.product import (
    export_failure_cases,
    export_trace,
    markdown_report,
    reproducibility_report,
)
from .agent.model_policy import load_qwen_policy_config
from .agent.model_policy import load_transformers_runtime
from .agent.policy import ScriptedPolicy
from .evaluation.rewards import compute_trajectory_reward
from .harness.replay import replay_corrected_samples
from .data.release import build_dataset_release_manifest
from .swesmith.rollout import run_base_collection, run_swesmith_rollouts
from .evaluation.gates import evaluate_rl_gate, load_rl_config
from .swesmith.materializer import SweSmithRolloutMaterializer
from .training.preflight import run_grpo_chat_template_preflight
from .training.grpo import (
    build_grpo_run_plan,
    run_grpo_training,
    run_grpo_smoke_training,
    write_grpo_run_plan,
)
from .training.rft import build_rft_dataset, select_rft_rollouts
from .evaluation.gates import evaluate_sft_gate, select_checkpoint
from .harness.smoke import create_smoke_repository, smoke_decisions
from .training.sft import (
    evaluate_token_budget_gate,
    load_sft_config,
    preflight_sft,
    run_sft_training,
    run_tokenizer_preflight,
)
from .evaluation.stats import compare_models
from .swesmith.stability import (
    stabilize_external_candidates,
    stabilize_swesmith_tasks,
)
from .swesmith.external import (
    build_external_inventory,
    build_external_source_inventory,
)
from .evaluation.two_layer import compare_two_layer_arms, validate_two_layer_configs
from .swesmith.config import (
    build_swesmith_inventory,
    fetch_swesmith_source,
    load_swesmith_environment_config,
    preflight_linux_server_environment,
    resolve_server_runtime_root,
)
from .swesmith.environment import preflight_swesmith_worker
from .trajectory.samples import write_success_samples


def _unique_run_name(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def run_web(host: str, port: int) -> int:
    """Start the optional localhost Web control plane."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Web dependencies are not installed; run `pip install -e .[web]`."
        ) from exc
    uvicorn.run(
        "coding_agent_policy_opt.web.app:app",
        host=host,
        port=port,
        reload=False,
    )
    return 0


def run_smoke(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    run_name = _unique_run_name("smoke")
    artifacts_root = config.run.artifacts_dir.resolve()
    run_dir = artifacts_root / "runs" / run_name
    workspace_dir = artifacts_root / "workspaces" / run_name
    workspace_dir.parent.mkdir(parents=True, exist_ok=True)

    task = create_smoke_repository(workspace_dir)
    profile = RepositoryProfile(
        test_command=config.repository.test_command,
        default_test_args=config.repository.default_test_args,
        timeout_seconds=config.harness.tool_timeout_seconds,
    )
    environment = RepositoryEnvironment(workspace_dir, profile)
    store = RunArtifactStore(run_dir)
    harness = AgentHarness(config, environment, store)
    episode = harness.run(task, ScriptedPolicy(smoke_decisions()))

    summary = {
        "run_dir": str(run_dir),
        "workspace_dir": str(workspace_dir),
        "episode": episode.to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if episode.resolved else 1


def validate_task_manifest(path: Path) -> int:
    report = validate_manifest(path)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.valid else 1


def merge_mvp_manifests(train: Path, dev: Path, test: Path, output: Path) -> int:
    report = merge_task_manifests(
        [train, dev, test],
        output,
        expected_counts={"train": 100, "dev": 20, "test": 40},
    )
    _emit_json(report)
    return 0


def freeze_external_tasks(
    candidates: Path,
    manifest_output: Path,
    exclusion_report_output: Path,
    count: int,
) -> int:
    report = freeze_external_manifest(
        candidates,
        manifest_output,
        exclusion_report_output,
        count=count,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def build_success_data(run_dir: Path, output: Path) -> int:
    count = write_success_samples(run_dir, output)
    print(
        json.dumps(
            {"dataset": "D_success", "samples": count, "output": str(output.resolve())},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_data(
    path: Path,
    dataset: str,
    expected_count: int | None,
    require_reviewed: bool,
    min_review_fraction: float | None,
    manifest: Path | None,
    allowed_split: str | None,
    replay_ledger: Path | None,
) -> int:
    report = validate_decision_dataset(
        path,
        dataset,
        expected_count=expected_count,
        require_reviewed=require_reviewed,
        min_review_fraction=min_review_fraction,
        manifest_path=manifest,
        allowed_split=allowed_split,
        replay_ledger_path=replay_ledger,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.valid else 1


def replay_corrected_data(
    dataset: Path,
    run_dirs: list[Path],
    inventory: Path,
    oracle: Path,
    manifest: Path,
    environment_config_path: Path,
    output: Path,
    ledger_output: Path,
) -> int:
    environment_config = load_swesmith_environment_config(environment_config_path)
    materializer = SweSmithRolloutMaterializer(
        inventory,
        oracle,
        manifest,
        environment_config,
        split="train",
    )
    report = replay_corrected_samples(
        dataset,
        run_dirs,
        output,
        ledger_output,
        materializer,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def release_dataset(
    dataset_path: Path,
    dataset: str,
    task_manifest: Path,
    output: Path,
    expected_count: int,
    replay_ledger: Path | None,
) -> int:
    report = build_dataset_release_manifest(
        dataset_path,
        dataset,
        task_manifest,
        output,
        expected_count=expected_count,
        replay_ledger_path=replay_ledger,
    )
    _emit_json(
        {
            "schema_version": "2.0.0",
            "dataset": report["dataset"],
            "sample_count": report["sample_count"],
            "dataset_sha256": report["dataset_sha256"],
            "output": str(output.resolve()),
        }
    )
    return 0


def _emit_json(value: dict, output: Path | None = None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    resolved = output.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    print(json.dumps({"output": str(resolved)}, ensure_ascii=False, indent=2))


def summarize_evaluation(
    run_dirs: list[Path], gold_files: Path | None, output: Path | None
) -> int:
    gold = load_gold_files(gold_files) if gold_files is not None else None
    tasks = collect_task_metrics(run_dirs, gold_by_task=gold)
    policies: list[dict] = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir.resolve() / "run.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Run metadata must be an object: {run_dir}")
        policy = metadata.get("harness_policy")
        if not isinstance(policy, dict):
            config = metadata.get("config", {})
            policy = config.get("harness_policy") if isinstance(config, dict) else None
        if isinstance(policy, dict) and policy not in policies:
            policies.append(policy)
    if len(policies) > 1:
        raise ValueError("Evaluation summary combines multiple Harness Policy candidates")
    report = {
        "schema_version": "2.0.0",
        "report_type": "evaluation_summary",
        "metrics": aggregate_metrics(tasks),
    }
    if policies:
        report["harness_policy"] = policies[0]
    _emit_json(report, output)
    return 0


def compare_evaluations(
    baseline_run_dirs: list[Path],
    candidate_run_dirs: list[Path],
    gold_files: Path | None,
    bootstrap_samples: int,
    seed: int,
    output: Path | None,
) -> int:
    gold = load_gold_files(gold_files) if gold_files is not None else None
    baseline = collect_task_metrics(baseline_run_dirs, gold_by_task=gold)
    candidate = collect_task_metrics(candidate_run_dirs, gold_by_task=gold)
    report = {
        "schema_version": "2.0.0",
        "report_type": "paired_model_comparison",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "comparison": compare_models(
            baseline, candidate, samples=bootstrap_samples, seed=seed
        ),
    }
    _emit_json(report, output)
    return 0


def validate_two_layer_configs_command(config_paths: dict[str, Path]) -> int:
    _emit_json(validate_two_layer_configs(config_paths))
    return 0


def compare_two_layer_evaluations(
    arm_run_dirs: dict[str, list[Path]],
    gold_files: Path | None,
    bootstrap_samples: int,
    seed: int,
    expected_task_count: int | None,
    expected_split: str | None,
    output: Path | None,
) -> int:
    report = compare_two_layer_arms(
        arm_run_dirs,
        gold_files=gold_files,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_task_count=expected_task_count,
        expected_split=expected_split,
    )
    _emit_json(report, output)
    return 0


def replay_checkpoint(
    run_dir: Path,
    inventory: Path,
    oracle: Path,
    manifest: Path,
    environment_config_path: Path,
    output: Path | None,
) -> int:
    metadata = json.loads((run_dir.resolve() / "run.json").read_text(encoding="utf-8"))
    split = metadata.get("split")
    if not isinstance(split, str):
        raise ValueError("Checkpoint source run metadata must contain a split")
    environment_config = load_swesmith_environment_config(environment_config_path)
    materializer = SweSmithRolloutMaterializer(
        inventory,
        oracle,
        manifest,
        environment_config,
        split=split,
    )
    report = replay_latest_checkpoint(run_dir, materializer)
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def analyze_exploration_command(
    run_dirs: list[Path], gold_files: Path | None, output: Path | None
) -> int:
    gold = load_gold_files(gold_files) if gold_files is not None else None
    report = analyze_exploration(run_dirs, gold_by_task=gold)
    _emit_json(report, output)
    return 0


def select_harness_candidate_command(
    dev_report_paths: list[Path], output: Path | None
) -> int:
    """Select a frozen Harness candidate from dev-only evaluation summaries."""

    reports: list[dict] = []
    for path in dev_report_paths:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Harness dev report must be an object: {path}")
        if value.get("report_type") == "evaluation_summary":
            policy = value.get("harness_policy")
            metrics = value.get("metrics")
            if not isinstance(policy, dict) or not isinstance(metrics, dict):
                raise ValueError(
                    "evaluation_summary must contain harness_policy and metrics"
                )
            value = {"harness_policy": policy, "metrics": metrics}
        reports.append(value)
    _emit_json(select_harness_candidate(reports), output)
    return 0


def select_rft_rollouts_command(run_dirs: list[Path], output: Path | None) -> int:
    report = select_rft_rollouts(run_dirs)
    _emit_json(report, output)
    return 0


def build_rft_dataset_command(
    run_dirs: list[Path],
    dataset_output: Path,
    report_output: Path | None,
    count: int,
    seed: int,
) -> int:
    report = build_rft_dataset(run_dirs, dataset_output, count=count, seed=seed)
    _emit_json(report, report_output)
    return 0


def validate_sft_config(config_path: Path) -> int:
    config = load_sft_config(config_path)
    _emit_json(
        {
            "schema_version": config.schema_version,
            "experiment_id": config.experiment_id,
            "config_hash": config.config_hash,
            "frozen_profile": config.profile,
            "status": config.status,
        }
    )
    return 0


def run_sft_preflight(config_path: Path, output: Path | None) -> int:
    report = preflight_sft(load_sft_config(config_path))
    _emit_json(report, output)
    return 0 if report["dataset_gate"]["valid"] else 1


def train_sft(config_path: Path, output: Path | None) -> int:
    result = run_sft_training(load_sft_config(config_path))
    _emit_json(result, output)
    return 0


def select_sft_checkpoint(candidates: Path, output: Path | None) -> int:
    _emit_json(select_checkpoint(candidates), output)
    return 0


def check_sft_gate(
    success_report: Path,
    policy_report: Path,
    comparison_report: Path,
    output: Path | None,
) -> int:
    report = evaluate_sft_gate(success_report, policy_report, comparison_report)
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def preflight_environment(config_path: Path, output: Path | None) -> int:
    report = {
        "schema_version": "2.0.0",
        "status": "refused",
        "ready": False,
        "config": str(config_path),
        "reason": (
            "Local Windows/WSL environment operations are disabled to protect "
            "the user's D drive; use 'capo server preflight' on the rented Linux server"
        ),
    }
    _emit_json(report, output)
    return 1


def bootstrap_environment(config_path: Path) -> int:
    _emit_json(
        {
            "schema_version": "2.0.0",
            "status": "refused",
            "config": str(config_path),
            "reason": (
                "Local Windows/WSL bootstrap is disabled and will not inspect or "
                "modify D-drive paths; use 'capo server bootstrap' on the rented server"
            ),
        }
    )
    return 1


def preflight_server_environment(
    config_path: Path,
    output: Path | None,
    *,
    base_only: bool,
) -> int:
    config = load_swesmith_environment_config(config_path)
    report = preflight_linux_server_environment(config)
    if report["ready"] and not base_only:
        try:
            worker_report = preflight_swesmith_worker(config)
        except Exception as exc:
            worker_report = {
                "ready": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "checks": {},
                "details": {},
            }
        report["worker"] = worker_report
        report["ready"] = bool(report["ready"] and worker_report.get("ready"))
    elif base_only:
        report["worker"] = {"skipped": True, "reason": "base-only preflight"}
    else:
        report["worker"] = {
            "ready": False,
            "skipped": True,
            "reason": "Linux server base preflight failed",
        }
    _emit_json(report, output)
    return 0 if report["ready"] else 1


def bootstrap_server_environment(config_path: Path) -> int:
    config = load_swesmith_environment_config(config_path)
    base = preflight_linux_server_environment(config)
    if not base["ready"]:
        _emit_json(
            {
                "schema_version": "2.0.0",
                "status": "blocked",
                "preflight": base,
            }
        )
        return 1
    runtime_root = resolve_server_runtime_root(config)
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "bootstrap_swesmith_server.sh"
    result = subprocess.run(
        ["bash", str(script), str(project_root), str(runtime_root)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=3600,
    )
    report = {
        "schema_version": "2.0.0",
        "status": "completed" if result.returncode == 0 else "failed",
        "runtime_root": str(runtime_root),
        "exit_code": result.returncode,
        "stdout": result.stdout[-20_000:],
        "stderr": result.stderr[-20_000:],
    }
    _emit_json(report)
    return 0 if result.returncode == 0 else 1


def create_swesmith_inventory(
    source: Path,
    config_path: Path,
    inventory_output: Path,
    oracle_output: Path,
) -> int:
    report = build_swesmith_inventory(
        source,
        inventory_output,
        oracle_output,
        load_swesmith_environment_config(config_path),
    )
    _emit_json(report)
    return 0


def export_trace_command(run_dir: Path, output: Path) -> int:
    _emit_json(export_trace(run_dir, output))
    return 0


def export_failure_cases_command(run_dirs: list[Path], output_root: Path, output: Path | None) -> int:
    report = export_failure_cases(run_dirs, output_root)
    _emit_json(report, output)
    return 0


def reproducibility_report_command(
    run_dirs: list[Path], output: Path, gold_files: Path | None
) -> int:
    _emit_json(reproducibility_report(run_dirs, output, gold_files))
    return 0


def markdown_report_command(
    run_dirs: list[Path], output: Path, gold_files: Path | None
) -> int:
    _emit_json(markdown_report(run_dirs, output, gold_files))
    return 0


def create_external_inventory(
    source: Path,
    candidates: Path,
    freeze_report: Path,
    manifest: Path,
    inventory_output: Path,
    oracle_output: Path,
    config_path: Path,
    dataset_revision: str,
) -> int:
    load_swesmith_environment_config(config_path)
    report = build_external_inventory(
        source,
        candidates,
        freeze_report,
        manifest,
        inventory_output,
        oracle_output,
        dataset_revision=dataset_revision,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def create_external_source_inventory(
    source: Path,
    inventory_output: Path,
    oracle_output: Path,
    dataset_revision: str,
) -> int:
    report = build_external_source_inventory(
        source,
        inventory_output,
        oracle_output,
        dataset_revision=dataset_revision,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def fetch_swesmith_dataset(output: Path, config_path: Path) -> int:
    _emit_json(
        fetch_swesmith_source(
            output, load_swesmith_environment_config(config_path)
        )
    )
    return 0


def stabilize_swesmith(
    inventory: Path,
    oracle: Path,
    manifest_output: Path,
    report_output: Path,
    config_path: Path,
    split: str,
    tool_timeout_seconds: int,
) -> int:
    report = stabilize_swesmith_tasks(
        inventory,
        oracle,
        manifest_output,
        report_output,
        load_swesmith_environment_config(config_path),
        split=split,
        tool_timeout_seconds=tool_timeout_seconds,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def stabilize_external(
    inventory: Path,
    oracle: Path,
    candidates_output: Path,
    report_output: Path,
    config_path: Path,
    tool_timeout_seconds: int,
) -> int:
    report = stabilize_external_candidates(
        inventory,
        oracle,
        candidates_output,
        report_output,
        load_swesmith_environment_config(config_path),
        tool_timeout_seconds=tool_timeout_seconds,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def check_harness_gate(
    stability_report: Path,
    rollout_report: Path,
    output: Path | None,
    expected_tasks: int,
) -> int:
    report = evaluate_harness_smoke_gate(
        stability_report, rollout_report, expected_tasks=expected_tasks
    )
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def collect_base_rollouts(
    inventory: Path,
    oracle: Path,
    manifest: Path,
    report_output: Path,
    experiment_config_path: Path,
    environment_config_path: Path,
    model_config_path: Path,
) -> int:
    experiment = load_experiment_config(experiment_config_path)
    model_config = load_qwen_policy_config(model_config_path)
    if experiment.model.model_id != model_config.model["model_id"]:
        raise ValueError("Experiment and Qwen policy model IDs differ")
    if experiment.model.model_revision != model_config.model["model_revision"]:
        raise ValueError("Experiment and Qwen policy model revisions differ")
    if experiment.model.adapter_id is not None:
        raise ValueError("Base collection must not load an adapter")
    runtime = load_transformers_runtime(model_config_path)
    report = run_base_collection(
        inventory,
        oracle,
        manifest,
        report_output,
        experiment,
        load_swesmith_environment_config(environment_config_path),
        lambda seed: runtime.policy(f"collection_seed_{seed}"),
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def evaluate_model_rollouts(
    inventory: Path,
    oracle: Path,
    manifest: Path,
    report_output: Path,
    experiment_config_path: Path,
    environment_config_path: Path,
    model_config_path: Path,
    split: str,
    run_label: str,
    adapter_path: Path | None,
) -> int:
    experiment = load_experiment_config(experiment_config_path)
    model_config = load_qwen_policy_config(model_config_path)
    if experiment.model.model_id != model_config.model["model_id"]:
        raise ValueError("Experiment and Qwen policy model IDs differ")
    if experiment.model.model_revision != model_config.model["model_revision"]:
        raise ValueError("Experiment and Qwen policy model revisions differ")
    if (experiment.model.adapter_id is None) != (adapter_path is None):
        raise ValueError(
            "adapter_path must be present exactly when experiment.model.adapter_id is set"
        )
    runtime = load_transformers_runtime(model_config_path, adapter_path=adapter_path)
    report = run_swesmith_rollouts(
        inventory,
        oracle,
        manifest,
        report_output,
        experiment,
        load_swesmith_environment_config(environment_config_path),
        lambda seed: runtime.policy("evaluation"),
        split=split,
        seeds=(42,),
        run_label=run_label,
    )
    _emit_json(report)
    return 0 if report["passed"] else 1


def validate_rl_config(config_path: Path) -> int:
    config = load_rl_config(config_path)
    _emit_json(
        {
            "schema_version": config["schema_version"],
            "profile": config["profile"],
            "status": config["status"],
            "config_hash": config["config_hash"],
        }
    )
    return 0


def check_rl_gate(evidence: Path, config_path: Path, output: Path | None) -> int:
    report = evaluate_rl_gate(evidence, config_path)
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def grpo_template_preflight(model_config_path: Path, output: Path | None) -> int:
    report = run_grpo_chat_template_preflight(model_config_path)
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def create_grpo_plan(
    manifest: Path,
    checkpoint: Path,
    sft_gate_report: Path,
    template_preflight_report: Path,
    model_config: Path,
    rl_config: Path,
    run_output_dir: Path,
    profile: str,
    output: Path,
    rl_gate_report: Path | None,
) -> int:
    plan = build_grpo_run_plan(
        manifest,
        checkpoint,
        sft_gate_report,
        template_preflight_report,
        model_config,
        rl_config,
        run_output_dir,
        profile=profile,
        rl_gate_report_path=rl_gate_report,
    )
    write_grpo_run_plan(plan, output)
    _emit_json(
        {
            "schema_version": "2.0.0",
            "profile": plan["profile"],
            "task_count": plan["task_count"],
            "update_steps": plan["update_steps"],
            "plan_hash": plan["plan_hash"],
            "output": str(output.resolve()),
        }
    )
    return 0


def train_grpo_smoke(args: argparse.Namespace) -> int:
    report = run_grpo_smoke_training(
        args.inventory,
        args.oracle,
        args.manifest,
        args.checkpoint,
        args.sft_gate_report,
        args.template_preflight_report,
        args.model_config,
        args.config,
        args.environment_config,
        args.output_dir,
    )
    _emit_json(report)
    return 0 if report["completed"] else 1


def train_grpo(args: argparse.Namespace) -> int:
    report = run_grpo_training(
        args.inventory,
        args.oracle,
        args.manifest,
        args.checkpoint,
        args.sft_gate_report,
        args.template_preflight_report,
        args.model_config,
        args.config,
        args.environment_config,
        args.output_dir,
        profile=args.profile,
        rl_gate_report_path=args.rl_gate_report,
    )
    _emit_json(report)
    return 0 if report["completed"] else 1


def calculate_reward(
    episode_path: Path,
    max_tool_steps: int,
    recovery_ablation: bool,
    audit_passed: bool,
) -> int:
    episode = json.loads(episode_path.resolve().read_text(encoding="utf-8"))
    result = compute_trajectory_reward(
        episode,
        max_tool_steps=max_tool_steps,
        include_recovery_ablation=recovery_ablation,
        reward_hacking_audit_passed=audit_passed,
    )
    _emit_json({"schema_version": "2.0.0", "reward": result.to_dict()})
    return 0


def validate_model_policy_config(config_path: Path) -> int:
    config = load_qwen_policy_config(config_path)
    _emit_json(
        {
            "schema_version": "2.0.0",
            "profile": "qwen_policy_v1",
            "model_id": config.model["model_id"],
            "model_revision": config.model["model_revision"],
            "prompt_version": config.prompt_version,
            "decoding_modes": sorted(config.decoding),
            "config_hash": config.config_hash,
        }
    )
    return 0


def preflight_api_model_policy(config_path: Path, output: Path | None) -> int:
    config = load_api_model_policy_config(config_path)
    try:
        base_url, _api_key, model_id = resolve_api_model_environment(
            config, os.environ
        )
        report = {
            "schema_version": "2.0.0",
            "profile": "api_policy_v1",
            "ready": True,
            "transport": "openai_chat_completions",
            "base_url": base_url,
            "model_id": model_id,
            "api_key_env": config.api_key_env,
            "api_key_present": True,
            "network_request_sent": False,
            "config_hash": config.config_hash,
        }
    except ValueError as exc:
        report = {
            "schema_version": "2.0.0",
            "profile": "api_policy_v1",
            "ready": False,
            "error": str(exc),
            "network_request_sent": False,
            "config_hash": config.config_hash,
        }
    _emit_json(report, output)
    return 0 if report["ready"] else 1


def run_api_repository_command(args: argparse.Namespace) -> int:
    def split_command(value: str) -> tuple[str, ...]:
        # posix=False preserves Windows backslashes; remove only syntactic
        # quote characters around a token after splitting.
        return tuple(item.strip("\"'") for item in shlex.split(value, posix=False))

    report = run_api_repository(
        repo=args.repo,
        issue=args.issue,
        run_dir=args.run_dir,
        api_config=args.config,
        task_id=args.task_id,
        test_command=split_command(args.test_command),
        default_test_args=split_command(args.test_args),
        seed=args.seed,
        harness_policy=args.harness_policy,
        history_limit=args.history_limit,
        verification_mode=args.verification_mode,
        recovery_mode=args.recovery_mode,
        recovery_profile=args.recovery_profile,
        allow_dirty=args.allow_dirty,
    )
    _emit_json(report)
    return 0 if report["episode"]["resolved"] else 1


def assemble_success_data(inputs: list[Path], output: Path, count: int, seed: int) -> int:
    _emit_json(assemble_success_dataset(inputs, output, count=count, seed=seed))
    return 0


def assemble_policy_data(inputs: list[Path], output: Path, seed: int) -> int:
    _emit_json(assemble_policy_dataset(inputs, output, seed=seed))
    return 0


def build_review_queue(
    dataset: Path, output: Path, fraction: float, seed: int
) -> int:
    _emit_json(
        select_review_queue(dataset, output, fraction=fraction, seed=seed)
    )
    return 0


def merge_review_decisions(dataset: Path, reviews: Path, output: Path) -> int:
    _emit_json(apply_review_decisions(dataset, reviews, output))
    return 0


def tokenizer_preflight(config_path: Path, output: Path | None) -> int:
    _emit_json(run_tokenizer_preflight(load_sft_config(config_path)), output)
    return 0


def token_budget_gate(
    success_report: Path, policy_report: Path, output: Path | None
) -> int:
    report = evaluate_token_budget_gate(success_report, policy_report)
    _emit_json(report, output)
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Run the deterministic Phase 0 smoke task")
    smoke.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/smoke.yaml"),
        help="Experiment YAML path",
    )

    manifest = subparsers.add_parser("manifest", help="Task manifest operations")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_validate = manifest_subparsers.add_parser(
        "validate", help="Validate schema and split isolation"
    )
    manifest_validate.add_argument("path", type=Path)
    manifest_merge = manifest_subparsers.add_parser(
        "merge-mvp", help="Merge exact 100/20/40 train/dev/test manifests"
    )
    manifest_merge.add_argument("--train", type=Path, required=True)
    manifest_merge.add_argument("--dev", type=Path, required=True)
    manifest_merge.add_argument("--test", type=Path, required=True)
    manifest_merge.add_argument("--output", type=Path, required=True)
    manifest_external = manifest_subparsers.add_parser(
        "freeze-external",
        help="Freeze a SHA-256-selected external subset after recording exclusions",
    )
    manifest_external.add_argument("--candidates", type=Path, required=True)
    manifest_external.add_argument("--manifest-output", type=Path, required=True)
    manifest_external.add_argument(
        "--exclusion-report-output", type=Path, required=True
    )
    manifest_external.add_argument("--count", type=int, default=50)

    data = subparsers.add_parser("data", help="Decision dataset operations")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    data_build = data_subparsers.add_parser(
        "build-success", help="Build D_success samples from a resolved run"
    )
    data_build.add_argument("--run-dir", type=Path, required=True)
    data_build.add_argument("--output", type=Path, required=True)

    data_validate = data_subparsers.add_parser(
        "validate", help="Validate a decision dataset and its provenance"
    )
    data_validate.add_argument("--path", type=Path, required=True)
    data_validate.add_argument("--dataset", choices=["D_success", "D_policy"], required=True)
    data_validate.add_argument("--expected-count", type=int)
    data_validate.add_argument("--require-reviewed", action="store_true")
    data_validate.add_argument("--min-review-fraction", type=float)
    data_validate.add_argument("--manifest", type=Path)
    data_validate.add_argument(
        "--allowed-split", choices=["smoke", "train", "dev", "test", "external"]
    )
    data_validate.add_argument("--replay-ledger", type=Path)
    data_assemble_success = data_subparsers.add_parser(
        "assemble-success", help="Deterministically select an exact D_success dataset"
    )
    data_assemble_success.add_argument("--input", type=Path, action="append", required=True)
    data_assemble_success.add_argument("--output", type=Path, required=True)
    data_assemble_success.add_argument("--count", type=int, default=600)
    data_assemble_success.add_argument("--seed", type=int, default=42)
    data_assemble_policy = data_subparsers.add_parser(
        "assemble-policy", help="Select the frozen 240/150/150/60 D_policy quotas"
    )
    data_assemble_policy.add_argument("--input", type=Path, action="append", required=True)
    data_assemble_policy.add_argument("--output", type=Path, required=True)
    data_assemble_policy.add_argument("--seed", type=int, default=42)
    data_review = data_subparsers.add_parser(
        "review-queue", help="Create a deterministic review queue"
    )
    data_review.add_argument("--dataset", type=Path, required=True)
    data_review.add_argument("--output", type=Path, required=True)
    data_review.add_argument("--fraction", type=float, default=0.20)
    data_review.add_argument("--seed", type=int, default=42)
    data_apply_reviews = data_subparsers.add_parser(
        "apply-reviews", help="Apply immutable human review decisions to a new dataset file"
    )
    data_apply_reviews.add_argument("--dataset", type=Path, required=True)
    data_apply_reviews.add_argument("--reviews", type=Path, required=True)
    data_apply_reviews.add_argument("--output", type=Path, required=True)
    data_replay = data_subparsers.add_parser(
        "replay-corrected",
        help="Replay every corrected target from its exact source state in a fresh sandbox",
    )
    data_replay.add_argument("--dataset", type=Path, required=True)
    data_replay.add_argument("--run-dir", type=Path, action="append", required=True)
    data_replay.add_argument("--inventory", type=Path, required=True)
    data_replay.add_argument("--oracle", type=Path, required=True)
    data_replay.add_argument("--manifest", type=Path, required=True)
    data_replay.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    data_replay.add_argument("--output", type=Path, required=True)
    data_replay.add_argument("--ledger-output", type=Path, required=True)
    data_release = data_subparsers.add_parser(
        "release-manifest",
        help="Freeze dataset, task-manifest, replay, review, and sample-hash evidence",
    )
    data_release.add_argument("--path", type=Path, required=True)
    data_release.add_argument(
        "--dataset", choices=["D_success", "D_policy"], required=True
    )
    data_release.add_argument("--task-manifest", type=Path, required=True)
    data_release.add_argument("--output", type=Path, required=True)
    data_release.add_argument("--expected-count", type=int, default=600)
    data_release.add_argument("--replay-ledger", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluation metrics and statistics")
    evaluate_subparsers = evaluate.add_subparsers(
        dest="evaluate_command", required=True
    )
    evaluate_summary = evaluate_subparsers.add_parser(
        "summarize", help="Aggregate metrics from run artifacts"
    )
    evaluate_summary.add_argument("--run-dir", type=Path, action="append", required=True)
    evaluate_summary.add_argument("--gold-files", type=Path)
    evaluate_summary.add_argument("--output", type=Path)

    evaluate_compare = evaluate_subparsers.add_parser(
        "compare", help="Run paired model statistics on identical tasks"
    )
    evaluate_compare.add_argument(
        "--baseline-run-dir", type=Path, action="append", required=True
    )
    evaluate_compare.add_argument(
        "--candidate-run-dir", type=Path, action="append", required=True
    )
    evaluate_compare.add_argument("--gold-files", type=Path)
    evaluate_compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate_compare.add_argument("--seed", type=int, default=42)
    evaluate_compare.add_argument("--output", type=Path)
    evaluate_failures = evaluate_subparsers.add_parser(
        "failure-cases", help="Export trace-linked failure cases by category"
    )
    evaluate_failures.add_argument("--run-dir", type=Path, action="append", required=True)
    evaluate_failures.add_argument("--output-root", type=Path, required=True)
    evaluate_failures.add_argument("--output", type=Path)
    evaluate_repro = evaluate_subparsers.add_parser(
        "reproducibility", help="Freeze run metadata and regenerated metrics"
    )
    evaluate_repro.add_argument("--run-dir", type=Path, action="append", required=True)
    evaluate_repro.add_argument("--gold-files", type=Path)
    evaluate_repro.add_argument("--output", type=Path, required=True)
    evaluate_report = evaluate_subparsers.add_parser(
        "report", help="Write a measured/unverified Markdown evaluation report"
    )
    evaluate_report.add_argument("--run-dir", type=Path, action="append", required=True)
    evaluate_report.add_argument("--gold-files", type=Path)
    evaluate_report.add_argument("--output", type=Path, required=True)

    trace = subparsers.add_parser("trace", help="Export durable Agent traces")
    trace_subparsers = trace.add_subparsers(dest="trace_command", required=True)
    trace_export = trace_subparsers.add_parser("export", help="Export one run as a normalized trace")
    trace_export.add_argument("--run-dir", type=Path, required=True)
    trace_export.add_argument("--output", type=Path, required=True)

    two_layer = subparsers.add_parser(
        "two-layer", help="Two-layer Model Policy/Harness Policy ablations"
    )
    two_layer_subparsers = two_layer.add_subparsers(
        dest="two_layer_command", required=True
    )
    two_layer_validate = two_layer_subparsers.add_parser(
        "validate-configs", help="Validate the frozen four-arm experiment matrix"
    )
    for arm in ("base", "model-only", "harness-only", "joint"):
        two_layer_validate.add_argument(
            f"--{arm}-config", type=Path, required=True
        )
    two_layer_compare = two_layer_subparsers.add_parser(
        "compare", help="Compare Base/Model-only/Harness-only/Joint run artifacts"
    )
    for arm in ("base", "model-only", "harness-only", "joint"):
        two_layer_compare.add_argument(
            f"--{arm}-run-dir", type=Path, action="append", required=True
        )
    two_layer_compare.add_argument("--gold-files", type=Path)
    two_layer_compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    two_layer_compare.add_argument("--seed", type=int, default=42)
    two_layer_compare.add_argument(
        "--expected-task-count",
        type=int,
        help="Refuse comparison unless every arm contains exactly this many tasks",
    )
    two_layer_compare.add_argument(
        "--expected-split",
        choices=["smoke", "train", "dev", "test", "external"],
        help="Refuse comparison when an episode belongs to another split",
    )
    two_layer_compare.add_argument("--output", type=Path)
    two_layer_checkpoint = two_layer_subparsers.add_parser(
        "checkpoint-replay",
        help="Replay the latest run checkpoint in a fresh SWE-smith environment",
    )
    two_layer_checkpoint.add_argument("--run-dir", type=Path, required=True)
    two_layer_checkpoint.add_argument("--inventory", type=Path, required=True)
    two_layer_checkpoint.add_argument("--oracle", type=Path, required=True)
    two_layer_checkpoint.add_argument("--manifest", type=Path, required=True)
    two_layer_checkpoint.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    two_layer_checkpoint.add_argument("--output", type=Path)

    explore = subparsers.add_parser(
        "explore", help="Analyze repository localization and context traces"
    )
    explore_subparsers = explore.add_subparsers(
        dest="explore_command", required=True
    )
    explore_analyze = explore_subparsers.add_parser(
        "analyze", help="Produce a SWE-Explore-style trace analysis report"
    )
    explore_analyze.add_argument("--run-dir", type=Path, action="append", required=True)
    explore_analyze.add_argument("--gold-files", type=Path)
    explore_analyze.add_argument("--output", type=Path)

    harness = subparsers.add_parser(
        "harness", help="Harness Policy candidate selection"
    )
    harness_subparsers = harness.add_subparsers(
        dest="harness_command", required=True
    )
    harness_select = harness_subparsers.add_parser(
        "select-candidate",
        help="Select one registered Harness candidate from dev reports",
    )
    harness_select.add_argument(
        "--dev-report", type=Path, action="append", required=True
    )
    harness_select.add_argument("--output", type=Path)

    rft = subparsers.add_parser(
        "rft", help="Rejection-sampling fine-tuning data selection"
    )
    rft_subparsers = rft.add_subparsers(dest="rft_command", required=True)
    rft_select = rft_subparsers.add_parser(
        "select", help="Filter sampled train rollouts using evaluator evidence"
    )
    rft_select.add_argument("--run-dir", type=Path, action="append", required=True)
    rft_select.add_argument("--output", type=Path)
    rft_build = rft_subparsers.add_parser(
        "build-dataset", help="Build D_success from selected RFT rollouts"
    )
    rft_build.add_argument("--run-dir", type=Path, action="append", required=True)
    rft_build.add_argument("--dataset-output", type=Path, required=True)
    rft_build.add_argument("--report-output", type=Path)
    rft_build.add_argument("--count", type=int, default=600)
    rft_build.add_argument("--seed", type=int, default=42)

    sft = subparsers.add_parser("sft", help="Frozen QLoRA SFT operations")
    sft_subparsers = sft.add_subparsers(dest="sft_command", required=True)
    sft_validate = sft_subparsers.add_parser(
        "validate-config", help="Validate the frozen MVP QLoRA profile"
    )
    sft_validate.add_argument("--config", type=Path, required=True)
    sft_preflight = sft_subparsers.add_parser(
        "preflight", help="Run manifest, sample-count, split, and review gates"
    )
    sft_preflight.add_argument("--config", type=Path, required=True)
    sft_preflight.add_argument("--output", type=Path)
    sft_train = sft_subparsers.add_parser(
        "train", help="Run target-only 4-bit QLoRA SFT on CUDA"
    )
    sft_train.add_argument("--config", type=Path, required=True)
    sft_train.add_argument("--output", type=Path)
    sft_select = sft_subparsers.add_parser(
        "select-checkpoint", help="Select a dev checkpoint using the frozen rule"
    )
    sft_select.add_argument("--candidates", type=Path, required=True)
    sft_select.add_argument("--output", type=Path)
    sft_gate = sft_subparsers.add_parser(
        "gate", help="Evaluate the frozen Policy-SFT advancement thresholds"
    )
    sft_gate.add_argument("--success-report", type=Path, required=True)
    sft_gate.add_argument("--policy-report", type=Path, required=True)
    sft_gate.add_argument("--comparison-report", type=Path, required=True)
    sft_gate.add_argument("--output", type=Path)
    sft_tokenizer = sft_subparsers.add_parser(
        "tokenizer-preflight", help="Verify prefix preservation, 4K lengths, and token totals"
    )
    sft_tokenizer.add_argument("--config", type=Path, required=True)
    sft_tokenizer.add_argument("--output", type=Path)
    sft_token_gate = sft_subparsers.add_parser(
        "token-budget-gate", help="Require D_success/D_policy target tokens within one percent"
    )
    sft_token_gate.add_argument("--success-report", type=Path, required=True)
    sft_token_gate.add_argument("--policy-report", type=Path, required=True)
    sft_token_gate.add_argument("--output", type=Path)

    server = subparsers.add_parser(
        "server", help="Native Linux GPU server environment operations"
    )
    server_subparsers = server.add_subparsers(dest="server_command", required=True)
    server_preflight = server_subparsers.add_parser(
        "preflight", help="Verify the rented native Linux GPU server"
    )
    server_preflight.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    server_preflight.add_argument("--output", type=Path)
    server_preflight.add_argument(
        "--base-only",
        action="store_true",
        help="Skip the dependency worker check before first bootstrap",
    )
    server_bootstrap = server_subparsers.add_parser(
        "bootstrap", help="Install pinned dependencies only under CAPO_RUNTIME_ROOT"
    )
    server_bootstrap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )

    environment = subparsers.add_parser(
        "environment", help="Disabled local Windows/WSL operations"
    )
    environment_subparsers = environment.add_subparsers(
        dest="environment_command", required=True
    )
    environment_preflight = environment_subparsers.add_parser(
        "preflight", help="Refuse legacy local environment access"
    )
    environment_preflight.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith.yaml"),
    )
    environment_preflight.add_argument("--output", type=Path)
    environment_bootstrap = environment_subparsers.add_parser(
        "bootstrap", help="Refuse legacy local bootstrap"
    )
    environment_bootstrap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith.yaml"),
    )

    swesmith = subparsers.add_parser("swesmith", help="SWE-smith data operations")
    swesmith_subparsers = swesmith.add_subparsers(
        dest="swesmith_command", required=True
    )
    swesmith_inventory = swesmith_subparsers.add_parser(
        "build-inventory", help="Build public inventory and isolated oracle JSONL"
    )
    swesmith_inventory.add_argument("--source", type=Path, required=True)
    swesmith_inventory.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_inventory.add_argument("--inventory-output", type=Path, required=True)
    swesmith_inventory.add_argument("--oracle-output", type=Path, required=True)
    swesmith_fetch = swesmith_subparsers.add_parser(
        "fetch-source", help="Fetch the frozen SWE-smith train split into immutable JSONL"
    )
    swesmith_fetch.add_argument("--output", type=Path, required=True)
    swesmith_fetch.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_external_source = swesmith_subparsers.add_parser(
        "build-external-source-inventory",
        help="Build provisional external inventory/oracle before stability filtering",
    )
    swesmith_external_source.add_argument("--source", type=Path, required=True)
    swesmith_external_source.add_argument("--inventory-output", type=Path, required=True)
    swesmith_external_source.add_argument("--oracle-output", type=Path, required=True)
    swesmith_external_source.add_argument(
        "--dataset-revision",
        required=True,
        help="Pinned 40-character revision of the SWE-bench Verified source",
    )
    swesmith_external = swesmith_subparsers.add_parser(
        "build-external-inventory",
        help="Build executable inventory/oracle from a frozen external subset",
    )
    swesmith_external.add_argument("--source", type=Path, required=True)
    swesmith_external.add_argument("--candidates", type=Path, required=True)
    swesmith_external.add_argument("--freeze-report", type=Path, required=True)
    swesmith_external.add_argument("--manifest", type=Path, required=True)
    swesmith_external.add_argument("--inventory-output", type=Path, required=True)
    swesmith_external.add_argument("--oracle-output", type=Path, required=True)
    swesmith_external.add_argument(
        "--dataset-revision",
        required=True,
        help="Pinned 40-character revision of the SWE-bench Verified source",
    )
    swesmith_external.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_stabilize = swesmith_subparsers.add_parser(
        "stabilize", help="Run two initial evaluations and freeze a stable task manifest"
    )
    swesmith_stabilize.add_argument("--inventory", type=Path, required=True)
    swesmith_stabilize.add_argument("--oracle", type=Path, required=True)
    swesmith_stabilize.add_argument("--manifest-output", type=Path, required=True)
    swesmith_stabilize.add_argument("--report-output", type=Path, required=True)
    swesmith_stabilize.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_stabilize.add_argument(
        "--split", choices=["smoke", "train", "dev", "test"], required=True
    )
    swesmith_stabilize.add_argument("--tool-timeout-seconds", type=int, default=600)
    swesmith_external_stabilize = swesmith_subparsers.add_parser(
        "stabilize-external",
        help="Run two initial evaluations for every provisional external task",
    )
    swesmith_external_stabilize.add_argument("--inventory", type=Path, required=True)
    swesmith_external_stabilize.add_argument("--oracle", type=Path, required=True)
    swesmith_external_stabilize.add_argument(
        "--candidates-output", type=Path, required=True
    )
    swesmith_external_stabilize.add_argument("--report-output", type=Path, required=True)
    swesmith_external_stabilize.add_argument(
        "--config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_external_stabilize.add_argument(
        "--tool-timeout-seconds", type=int, default=600
    )
    swesmith_harness_gate = swesmith_subparsers.add_parser(
        "harness-gate", help="Combine smoke stability, rollout, schema, and cleanup evidence"
    )
    swesmith_harness_gate.add_argument(
        "--stability-report", type=Path, required=True
    )
    swesmith_harness_gate.add_argument("--rollout-report", type=Path, required=True)
    swesmith_harness_gate.add_argument("--output", type=Path)
    swesmith_harness_gate.add_argument("--expected-tasks", type=int, default=20)
    swesmith_collect = swesmith_subparsers.add_parser(
        "collect-base", help="Run the frozen seed 42/43 Base collection on train tasks"
    )
    swesmith_collect.add_argument("--inventory", type=Path, required=True)
    swesmith_collect.add_argument("--oracle", type=Path, required=True)
    swesmith_collect.add_argument("--manifest", type=Path, required=True)
    swesmith_collect.add_argument("--report-output", type=Path, required=True)
    swesmith_collect.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiment/base_collection.yaml"),
    )
    swesmith_collect.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_collect.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/qwen_policy.yaml"),
    )
    swesmith_evaluate = swesmith_subparsers.add_parser(
        "evaluate-model", help="Run deterministic evaluation decoding on a frozen split"
    )
    swesmith_evaluate.add_argument("--inventory", type=Path, required=True)
    swesmith_evaluate.add_argument("--oracle", type=Path, required=True)
    swesmith_evaluate.add_argument("--manifest", type=Path, required=True)
    swesmith_evaluate.add_argument("--report-output", type=Path, required=True)
    swesmith_evaluate.add_argument("--experiment-config", type=Path, required=True)
    swesmith_evaluate.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    swesmith_evaluate.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/qwen_policy.yaml"),
    )
    swesmith_evaluate.add_argument(
        "--split", choices=["smoke", "dev", "test", "external"], required=True
    )
    swesmith_evaluate.add_argument("--run-label", required=True)
    swesmith_evaluate.add_argument("--adapter-path", type=Path)

    rl = subparsers.add_parser("rl", help="Conditional Agentic GRPO operations")
    rl_subparsers = rl.add_subparsers(dest="rl_command", required=True)
    rl_validate = rl_subparsers.add_parser(
        "validate-config", help="Validate the conditional frozen GRPO config"
    )
    rl_validate.add_argument(
        "--config", type=Path, default=Path("configs/rl/grpo.yaml")
    )
    rl_gate = rl_subparsers.add_parser(
        "gate", help="Evaluate whether formal 7B Agentic GRPO may start"
    )
    rl_gate.add_argument("--evidence", type=Path, required=True)
    rl_gate.add_argument(
        "--config", type=Path, default=Path("configs/rl/grpo.yaml")
    )
    rl_gate.add_argument("--output", type=Path)
    rl_template = rl_subparsers.add_parser(
        "template-preflight",
        help="Verify strict token-prefix preservation for all six registered tools",
    )
    rl_template.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/qwen_policy.yaml"),
    )
    rl_template.add_argument("--output", type=Path)
    rl_plan = rl_subparsers.add_parser(
        "plan", help="Freeze deterministic task, checkpoint, config, and gate evidence"
    )
    rl_plan.add_argument("--manifest", type=Path, required=True)
    rl_plan.add_argument("--checkpoint", type=Path, required=True)
    rl_plan.add_argument("--sft-gate-report", type=Path, required=True)
    rl_plan.add_argument("--template-preflight-report", type=Path, required=True)
    rl_plan.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen_policy.yaml")
    )
    rl_plan.add_argument("--config", type=Path, default=Path("configs/rl/grpo.yaml"))
    rl_plan.add_argument("--run-output-dir", type=Path, required=True)
    rl_plan.add_argument("--profile", choices=["smoke", "formal"], required=True)
    rl_plan.add_argument("--output", type=Path, required=True)
    rl_plan.add_argument("--rl-gate-report", type=Path)
    rl_smoke_train = rl_subparsers.add_parser(
        "smoke-train", help="Run the gated 8-task, 50-update Agentic GRPO smoke"
    )
    rl_smoke_train.add_argument("--inventory", type=Path, required=True)
    rl_smoke_train.add_argument("--oracle", type=Path, required=True)
    rl_smoke_train.add_argument("--manifest", type=Path, required=True)
    rl_smoke_train.add_argument("--checkpoint", type=Path, required=True)
    rl_smoke_train.add_argument("--sft-gate-report", type=Path, required=True)
    rl_smoke_train.add_argument(
        "--template-preflight-report", type=Path, required=True
    )
    rl_smoke_train.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen_policy.yaml")
    )
    rl_smoke_train.add_argument(
        "--config", type=Path, default=Path("configs/rl/grpo.yaml")
    )
    rl_smoke_train.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    rl_smoke_train.add_argument("--output-dir", type=Path, required=True)
    rl_train = rl_subparsers.add_parser(
        "train", help="Run the gated smoke or formal Agentic GRPO profile"
    )
    rl_train.add_argument("--inventory", type=Path, required=True)
    rl_train.add_argument("--oracle", type=Path, required=True)
    rl_train.add_argument("--manifest", type=Path, required=True)
    rl_train.add_argument("--checkpoint", type=Path, required=True)
    rl_train.add_argument("--sft-gate-report", type=Path, required=True)
    rl_train.add_argument("--template-preflight-report", type=Path, required=True)
    rl_train.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen_policy.yaml")
    )
    rl_train.add_argument(
        "--config", type=Path, default=Path("configs/rl/grpo.yaml")
    )
    rl_train.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/environment/swesmith_server.yaml"),
    )
    rl_train.add_argument("--output-dir", type=Path, required=True)
    rl_train.add_argument("--profile", choices=["smoke", "formal"], required=True)
    rl_train.add_argument(
        "--rl-gate-report",
        type=Path,
        help="Required for the formal profile; must be a passed RL Gate report",
    )
    rl_reward = rl_subparsers.add_parser(
        "reward", help="Calculate the frozen trajectory reward from an episode JSON"
    )
    rl_reward.add_argument("--episode", type=Path, required=True)
    rl_reward.add_argument("--max-tool-steps", type=int, default=30)
    rl_reward.add_argument("--recovery-ablation", action="store_true")
    rl_reward.add_argument("--audit-passed", action="store_true")

    model = subparsers.add_parser("model", help="Qwen agent policy operations")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_validate = model_subparsers.add_parser(
        "validate-config", help="Validate the frozen model, prompt, and decoding profile"
    )
    model_validate.add_argument(
        "--config", type=Path, default=Path("configs/model/qwen_policy.yaml")
    )
    model_api_preflight = model_subparsers.add_parser(
        "api-preflight",
        help="Validate API policy environment without sending a network request",
    )
    model_api_preflight.add_argument(
        "--config", type=Path, default=Path("configs/model/api_policy.yaml")
    )
    model_api_preflight.add_argument("--output", type=Path)
    model_api_run = model_subparsers.add_parser(
        "api-run",
        help="Run one API-backed agent episode in an explicit repository worktree",
    )
    model_api_run.add_argument("--repo", type=Path, required=True)
    model_api_run.add_argument("--issue", required=True)
    model_api_run.add_argument("--run-dir", type=Path, required=True)
    model_api_run.add_argument(
        "--config", type=Path, default=Path("configs/model/api_policy.yaml")
    )
    model_api_run.add_argument("--task-id", default="api_task")
    model_api_run.add_argument(
        "--test-command", default="{python} -m pytest",
        help="Test executable and fixed arguments as one command-line string",
    )
    model_api_run.add_argument(
        "--test-args", default="-q",
        help="Default test arguments as one command-line string",
    )
    model_api_run.add_argument("--seed", type=int, default=42)
    model_api_run.add_argument(
        "--harness-policy", choices=["static_v1", "adaptive_v1"], default="adaptive_v1"
    )
    model_api_run.add_argument("--history-limit", type=int, default=8)
    model_api_run.add_argument("--recovery-profile", choices=["baseline", "record", "evidence", "full"], default="full")
    model_api_run.add_argument(
        "--verification-mode", choices=["advisory", "required"], default="required"
    )
    model_api_run.add_argument(
        "--recovery-mode", choices=["advisory", "required"], default="required"
    )
    model_api_run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow execution in a dirty worktree; use only when changes are intentional",
    )
    web = subparsers.add_parser(
        "web", help="Start the localhost Web control plane (optional web extra)"
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "web":
        return run_web(args.host, args.port)
    if args.command == "smoke":
        return run_smoke(args.config)
    if args.command == "manifest" and args.manifest_command == "validate":
        return validate_task_manifest(args.path)
    if args.command == "manifest" and args.manifest_command == "merge-mvp":
        return merge_mvp_manifests(args.train, args.dev, args.test, args.output)
    if args.command == "manifest" and args.manifest_command == "freeze-external":
        return freeze_external_tasks(
            args.candidates,
            args.manifest_output,
            args.exclusion_report_output,
            args.count,
        )
    if args.command == "data" and args.data_command == "build-success":
        return build_success_data(args.run_dir, args.output)
    if args.command == "data" and args.data_command == "validate":
        return validate_data(
            args.path,
            args.dataset,
            args.expected_count,
            args.require_reviewed,
            args.min_review_fraction,
            args.manifest,
            args.allowed_split,
            args.replay_ledger,
        )
    if args.command == "data" and args.data_command == "assemble-success":
        return assemble_success_data(args.input, args.output, args.count, args.seed)
    if args.command == "data" and args.data_command == "assemble-policy":
        return assemble_policy_data(args.input, args.output, args.seed)
    if args.command == "data" and args.data_command == "review-queue":
        return build_review_queue(
            args.dataset, args.output, args.fraction, args.seed
        )
    if args.command == "data" and args.data_command == "apply-reviews":
        return merge_review_decisions(args.dataset, args.reviews, args.output)
    if args.command == "data" and args.data_command == "replay-corrected":
        return replay_corrected_data(
            args.dataset,
            args.run_dir,
            args.inventory,
            args.oracle,
            args.manifest,
            args.environment_config,
            args.output,
            args.ledger_output,
        )
    if args.command == "data" and args.data_command == "release-manifest":
        return release_dataset(
            args.path,
            args.dataset,
            args.task_manifest,
            args.output,
            args.expected_count,
            args.replay_ledger,
        )
    if args.command == "evaluate" and args.evaluate_command == "summarize":
        return summarize_evaluation(args.run_dir, args.gold_files, args.output)
    if args.command == "evaluate" and args.evaluate_command == "compare":
        return compare_evaluations(
            args.baseline_run_dir,
            args.candidate_run_dir,
            args.gold_files,
            args.bootstrap_samples,
            args.seed,
            args.output,
        )
    if args.command == "evaluate" and args.evaluate_command == "failure-cases":
        return export_failure_cases_command(args.run_dir, args.output_root, args.output)
    if args.command == "evaluate" and args.evaluate_command == "reproducibility":
        return reproducibility_report_command(args.run_dir, args.output, args.gold_files)
    if args.command == "evaluate" and args.evaluate_command == "report":
        return markdown_report_command(args.run_dir, args.output, args.gold_files)
    if args.command == "trace" and args.trace_command == "export":
        return export_trace_command(args.run_dir, args.output)
    if args.command == "two-layer" and args.two_layer_command == "validate-configs":
        return validate_two_layer_configs_command(
            {
                "base": args.base_config,
                "model_only": args.model_only_config,
                "harness_only": args.harness_only_config,
                "joint": args.joint_config,
            }
        )
    if args.command == "two-layer" and args.two_layer_command == "compare":
        return compare_two_layer_evaluations(
            {
                "base": args.base_run_dir,
                "model_only": args.model_only_run_dir,
                "harness_only": args.harness_only_run_dir,
                "joint": args.joint_run_dir,
            },
            args.gold_files,
            args.bootstrap_samples,
            args.seed,
            args.expected_task_count,
            args.expected_split,
            args.output,
        )
    if args.command == "two-layer" and args.two_layer_command == "checkpoint-replay":
        return replay_checkpoint(
            args.run_dir,
            args.inventory,
            args.oracle,
            args.manifest,
            args.environment_config,
            args.output,
        )
    if args.command == "explore" and args.explore_command == "analyze":
        return analyze_exploration_command(args.run_dir, args.gold_files, args.output)
    if args.command == "harness" and args.harness_command == "select-candidate":
        return select_harness_candidate_command(args.dev_report, args.output)
    if args.command == "rft" and args.rft_command == "select":
        return select_rft_rollouts_command(args.run_dir, args.output)
    if args.command == "rft" and args.rft_command == "build-dataset":
        return build_rft_dataset_command(
            args.run_dir,
            args.dataset_output,
            args.report_output,
            args.count,
            args.seed,
        )
    if args.command == "sft" and args.sft_command == "validate-config":
        return validate_sft_config(args.config)
    if args.command == "sft" and args.sft_command == "preflight":
        return run_sft_preflight(args.config, args.output)
    if args.command == "sft" and args.sft_command == "train":
        return train_sft(args.config, args.output)
    if args.command == "sft" and args.sft_command == "select-checkpoint":
        return select_sft_checkpoint(args.candidates, args.output)
    if args.command == "sft" and args.sft_command == "gate":
        return check_sft_gate(
            args.success_report,
            args.policy_report,
            args.comparison_report,
            args.output,
        )
    if args.command == "sft" and args.sft_command == "tokenizer-preflight":
        return tokenizer_preflight(args.config, args.output)
    if args.command == "sft" and args.sft_command == "token-budget-gate":
        return token_budget_gate(
            args.success_report, args.policy_report, args.output
        )
    if args.command == "server" and args.server_command == "preflight":
        return preflight_server_environment(
            args.config, args.output, base_only=args.base_only
        )
    if args.command == "server" and args.server_command == "bootstrap":
        return bootstrap_server_environment(args.config)
    if args.command == "environment" and args.environment_command == "preflight":
        return preflight_environment(args.config, args.output)
    if args.command == "environment" and args.environment_command == "bootstrap":
        return bootstrap_environment(args.config)
    if args.command == "swesmith" and args.swesmith_command == "build-inventory":
        return create_swesmith_inventory(
            args.source,
            args.config,
            args.inventory_output,
            args.oracle_output,
        )
    if args.command == "swesmith" and args.swesmith_command == "fetch-source":
        return fetch_swesmith_dataset(args.output, args.config)
    if (
        args.command == "swesmith"
        and args.swesmith_command == "build-external-source-inventory"
    ):
        return create_external_source_inventory(
            args.source,
            args.inventory_output,
            args.oracle_output,
            args.dataset_revision,
        )
    if args.command == "swesmith" and args.swesmith_command == "build-external-inventory":
        return create_external_inventory(
            args.source,
            args.candidates,
            args.freeze_report,
            args.manifest,
            args.inventory_output,
            args.oracle_output,
            args.config,
            args.dataset_revision,
        )
    if args.command == "swesmith" and args.swesmith_command == "stabilize":
        return stabilize_swesmith(
            args.inventory,
            args.oracle,
            args.manifest_output,
            args.report_output,
            args.config,
            args.split,
            args.tool_timeout_seconds,
        )
    if args.command == "swesmith" and args.swesmith_command == "stabilize-external":
        return stabilize_external(
            args.inventory,
            args.oracle,
            args.candidates_output,
            args.report_output,
            args.config,
            args.tool_timeout_seconds,
        )
    if args.command == "swesmith" and args.swesmith_command == "harness-gate":
        return check_harness_gate(
            args.stability_report,
            args.rollout_report,
            args.output,
            args.expected_tasks,
        )
    if args.command == "swesmith" and args.swesmith_command == "collect-base":
        return collect_base_rollouts(
            args.inventory,
            args.oracle,
            args.manifest,
            args.report_output,
            args.experiment_config,
            args.environment_config,
            args.model_config,
        )
    if args.command == "swesmith" and args.swesmith_command == "evaluate-model":
        return evaluate_model_rollouts(
            args.inventory,
            args.oracle,
            args.manifest,
            args.report_output,
            args.experiment_config,
            args.environment_config,
            args.model_config,
            args.split,
            args.run_label,
            args.adapter_path,
        )
    if args.command == "rl" and args.rl_command == "validate-config":
        return validate_rl_config(args.config)
    if args.command == "rl" and args.rl_command == "gate":
        return check_rl_gate(args.evidence, args.config, args.output)
    if args.command == "rl" and args.rl_command == "template-preflight":
        return grpo_template_preflight(args.model_config, args.output)
    if args.command == "rl" and args.rl_command == "plan":
        return create_grpo_plan(
            args.manifest,
            args.checkpoint,
            args.sft_gate_report,
            args.template_preflight_report,
            args.model_config,
            args.config,
            args.run_output_dir,
            args.profile,
            args.output,
            args.rl_gate_report,
        )
    if args.command == "rl" and args.rl_command == "smoke-train":
        return train_grpo_smoke(args)
    if args.command == "rl" and args.rl_command == "train":
        return train_grpo(args)
    if args.command == "rl" and args.rl_command == "reward":
        return calculate_reward(
            args.episode,
            args.max_tool_steps,
            args.recovery_ablation,
            args.audit_passed,
        )
    if args.command == "model" and args.model_command == "validate-config":
        return validate_model_policy_config(args.config)
    if args.command == "model" and args.model_command == "api-preflight":
        return preflight_api_model_policy(args.config, args.output)
    if args.command == "model" and args.model_command == "api-run":
        return run_api_repository_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
