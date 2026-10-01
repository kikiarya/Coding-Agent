from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from ..hashing import sha256_json, sha256_text
from ..data.manifest import TaskManifestEntry, read_jsonl, validate_manifest
from ..agent.model_policy import load_qwen_policy_config
from .grpo_environment import CodingGRPOEnvironment, GRPOEnvironmentConfig
from ..config import PINNED_MODEL_REVISION
from ..evaluation.gates import load_rl_config
from ..evaluation.rewards import RewardWeights
from ..swesmith.materializer import SweSmithRolloutMaterializer
from .preflight import grpo_chat_template_preflight
from ..swesmith.config import load_swesmith_environment_config


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256(path.resolve().read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _directory_hash(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {resolved}")
    entries: list[dict[str, str]] = []
    for file_path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": file_path.relative_to(resolved).as_posix(),
                "sha256": _file_hash(file_path),
            }
        )
    if not entries:
        raise ValueError("Checkpoint directory contains no files")
    return sha256_json(entries)


def _load_passed_report(path: Path, report_type: str) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_type") != report_type:
        raise ValueError(f"Expected {report_type} report: {path.resolve()}")
    if value.get("passed") is not True:
        raise ValueError(f"Required report did not pass: {path.resolve()}")
    return value


def build_grpo_run_plan(
    manifest_path: Path,
    checkpoint_path: Path,
    sft_gate_report_path: Path,
    template_preflight_report_path: Path,
    model_config_path: Path,
    rl_config_path: Path,
    output_dir: Path,
    *,
    profile: str = "smoke",
    rl_gate_report_path: Path | None = None,
) -> dict[str, Any]:
    if profile not in {"smoke", "formal"}:
        raise ValueError("profile must be smoke or formal")
    if output_dir.resolve().exists():
        raise FileExistsError(f"Refusing to reuse RL output directory: {output_dir.resolve()}")
    manifest_report = validate_manifest(manifest_path)
    if not manifest_report.valid:
        raise ValueError(f"Task manifest is invalid: {manifest_report.errors}")
    rl_config = load_rl_config(rl_config_path)
    model_config = load_qwen_policy_config(model_config_path)
    _load_passed_report(sft_gate_report_path, "sft_gate")
    template_report = _load_passed_report(
        template_preflight_report_path, "grpo_chat_template_preflight"
    )
    if (
        template_report.get("tools_passed") != 6
        or template_report.get("tokenizer_revision") != PINNED_MODEL_REVISION
    ):
        raise ValueError("Template preflight does not cover the pinned tokenizer and six tools")
    checkpoint = checkpoint_path.resolve()
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("Policy-SFT checkpoint is missing adapter_config.json")
    if profile == "formal":
        if rl_gate_report_path is None:
            raise ValueError("Formal GRPO plan requires a passed RL Gate report")
        _load_passed_report(rl_gate_report_path, "rl_gate")

    train_entries: list[TaskManifestEntry] = []
    for raw in read_jsonl(manifest_path):
        entry = TaskManifestEntry.from_dict(raw)
        if entry.split == "train":
            if entry.stable_repeats < 2:
                raise ValueError(f"Train task is not stable: {entry.task_id}")
            train_entries.append(entry)
    profile_config = rl_config["profiles"][profile]
    task_count = int(profile_config["stable_train_tasks"])
    if len(train_entries) < task_count:
        raise ValueError(
            f"GRPO {profile} requires {task_count} stable train tasks; found {len(train_entries)}"
        )
    seed = int(rl_config["trainer"]["seed"])
    selected = sorted(
        train_entries,
        key=lambda entry: sha256_text(f"{seed}:{entry.task_id}"),
    )[:task_count]
    plan: dict[str, Any] = {
        "schema_version": "2.0.0",
        "report_type": "grpo_run_plan",
        "status": "ready_for_runtime_preflight",
        "profile": profile,
        "rl_config_hash": rl_config["config_hash"],
        "model_config_hash": model_config.config_hash,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_hash(manifest_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_hash": _directory_hash(checkpoint),
        "sft_gate_report": str(sft_gate_report_path.resolve()),
        "sft_gate_report_sha256": _file_hash(sft_gate_report_path),
        "template_preflight_report": str(template_preflight_report_path.resolve()),
        "template_preflight_report_sha256": _file_hash(
            template_preflight_report_path
        ),
        "rl_gate_report": (
            str(rl_gate_report_path.resolve()) if rl_gate_report_path is not None else None
        ),
        "rl_gate_report_sha256": (
            _file_hash(rl_gate_report_path) if rl_gate_report_path is not None else None
        ),
        "selected_task_ids": [entry.task_id for entry in selected],
        "task_count": task_count,
        "update_steps": int(profile_config["update_steps"]),
        "trainer": dict(rl_config["trainer"]),
        "environment": dict(rl_config["environment"]),
        "harness_policy": dict(rl_config["harness_policy"]),
        "output_dir": str(output_dir.resolve()),
    }
    plan["plan_hash"] = sha256_json(plan)
    return plan


def write_grpo_run_plan(plan: dict[str, Any], output_path: Path) -> None:
    resolved = output_path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


class JsonlTrajectorySink:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()

    def __call__(self, value: dict[str, Any]) -> None:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)


def run_grpo_training(
    inventory_path: Path,
    oracle_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    sft_gate_report_path: Path,
    template_preflight_report_path: Path,
    model_config_path: Path,
    rl_config_path: Path,
    environment_config_path: Path,
    output_dir: Path,
    *,
    profile: str = "smoke",
    rl_gate_report_path: Path | None = None,
) -> dict[str, Any]:
    plan = build_grpo_run_plan(
        manifest_path,
        checkpoint_path,
        sft_gate_report_path,
        template_preflight_report_path,
        model_config_path,
        rl_config_path,
        output_dir,
        profile=profile,
        rl_gate_report_path=rl_gate_report_path,
    )
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install the pinned optional RL dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"GRPO {profile} requires CUDA")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if memory_gb < 80:
        raise RuntimeError(
            f"GRPO {profile} requires at least 80GB VRAM; found {memory_gb:.2f}GB"
        )

    model_config = load_qwen_policy_config(model_config_path)
    rl_config = load_rl_config(rl_config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model["tokenizer_id"],
        revision=model_config.model["tokenizer_revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    runtime_template = grpo_chat_template_preflight(tokenizer, model_config)
    if not runtime_template["passed"]:
        raise RuntimeError("Runtime tokenizer failed GRPO chat-template preflight")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model["model_id"],
        revision=model_config.model["model_revision"],
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    model = PeftModel.from_pretrained(
        base_model, str(checkpoint_path.resolve()), is_trainable=True
    )

    environment_config = load_swesmith_environment_config(environment_config_path)
    materializer = SweSmithRolloutMaterializer(
        inventory_path,
        oracle_path,
        manifest_path,
        environment_config,
        split="train",
    )
    sink = JsonlTrajectorySink(output_dir / "trajectory_audit.jsonl")
    environment_profile = GRPOEnvironmentConfig(
        max_tool_steps=int(rl_config["environment"]["max_tool_steps"]),
        max_test_calls=int(rl_config["environment"]["max_test_calls"]),
        harness_policy_variant=str(rl_config["harness_policy"]["variant"]),
        harness_history_limit=int(rl_config["harness_policy"]["history_limit"]),
        harness_verification_mode=str(rl_config["harness_policy"]["verification_mode"]),
        harness_recovery_mode=str(rl_config["harness_policy"]["recovery_mode"]),
        reward_weights=RewardWeights.from_config(rl_config["reward"]),
    )

    def environment_factory() -> CodingGRPOEnvironment:
        return CodingGRPOEnvironment(
            materializer,
            environment_profile,
            audit_sink=sink,
        )

    prompt = [
        {
            "role": "user",
            "content": "Solve the assigned repository issue using only registered tools. ",
        }
    ]
    dataset = Dataset.from_list(
        [{"prompt": prompt, "task_id": task_id} for task_id in plan["selected_task_ids"]]
    )
    trainer_config = rl_config["trainer"]
    output_dir.resolve().mkdir(parents=True, exist_ok=False)
    write_grpo_run_plan(plan, output_dir / "run_plan.json")
    args = GRPOConfig(
        output_dir=str(output_dir.resolve()),
        max_steps=plan["update_steps"],
        num_generations=trainer_config["num_generations"],
        max_tool_calling_iterations=trainer_config["max_tool_calling_iterations"],
        per_device_train_batch_size=trainer_config["per_device_batch_size"],
        gradient_accumulation_steps=trainer_config["gradient_accumulation_steps"],
        learning_rate=trainer_config["learning_rate"],
        beta=trainer_config["kl_beta"],
        max_completion_length=1024,
        use_vllm=True,
        vllm_mode=trainer_config["vllm_mode"],
        vllm_enable_sleep_mode=trainer_config["vllm_enable_sleep_mode"],
        bf16=True,
        seed=trainer_config["seed"],
        logging_steps=1,
        save_steps=plan["update_steps"],
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=environment_factory,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    result = {
        "schema_version": "2.0.0",
        "report_type": f"grpo_{profile}_result",
        "profile": profile,
        "plan_hash": plan["plan_hash"],
        "runtime_template_preflight": runtime_template,
        "gpu_memory_gb": memory_gb,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(train_result.training_loss),
        "trajectory_audit": str((output_dir / "trajectory_audit.jsonl").resolve()),
        "checkpoint": str((output_dir / "checkpoint-final").resolve()),
        "completed": int(trainer.state.global_step) == plan["update_steps"],
    }
    with (output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def run_grpo_smoke_training(
    inventory_path: Path,
    oracle_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    sft_gate_report_path: Path,
    template_preflight_report_path: Path,
    model_config_path: Path,
    rl_config_path: Path,
    environment_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the gated smoke profile."""

    return run_grpo_training(
        inventory_path,
        oracle_path,
        manifest_path,
        checkpoint_path,
        sft_gate_report_path,
        template_preflight_report_path,
        model_config_path,
        rl_config_path,
        environment_config_path,
        output_dir,
        profile="smoke",
    )
