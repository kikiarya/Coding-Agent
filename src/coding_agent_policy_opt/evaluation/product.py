from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import aggregate_metrics, collect_task_metrics, load_gold_files
from ..data.manifest import read_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _write_new(path: Path, value: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_trace(run_dir: Path, output: Path) -> dict[str, Any]:
    root = run_dir.resolve()
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    steps = _read_jsonl(root / "steps.jsonl")
    episodes = _read_jsonl(root / "episodes.jsonl")
    checkpoints = _read_jsonl(root / "checkpoints.jsonl")
    events = _read_jsonl(root / "events.jsonl")
    timeline = []
    for step in steps:
        timeline.append({
            "step_index": step.get("step_index"),
            "episode_id": step.get("episode_id"),
            "action": step.get("action"),
            "tool_result": step.get("tool_result"),
            "test_state": step.get("test_state"),
            "latency_ms": step.get("latency_ms", 0),
            "tool_latency_ms": step.get("tool_latency_ms", 0),
            "input_tokens": step.get("input_tokens", 0),
            "output_tokens": step.get("output_tokens", 0),
            "labels": step.get("automatic_labels", []),
            "checkpoint": next((item for item in checkpoints
                                if item.get("episode_id") == step.get("episode_id")
                                and item.get("step_index") == step.get("step_index")), None),
        })
    trace = {
        "schema_version": "1.0.0",
        "report_type": "agent_trace",
        "run": metadata,
        "episodes": episodes,
        "timeline": timeline,
        "events": events,
    }
    _write_new(output, trace)
    return {"output": str(output.resolve()), "steps": len(timeline), "events": len(events)}


def export_failure_cases(run_dirs: Iterable[Path], output_root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    outputs: list[str] = []
    for run_dir in run_dirs:
        root = run_dir.resolve()
        episodes = {item.get("episode_id"): item for item in _read_jsonl(root / "episodes.jsonl")}
        for step in _read_jsonl(root / "steps.jsonl"):
            labels = set(step.get("automatic_labels", []))
            if step.get("tool_result", {}).get("error_code"):
                labels.add("invalid_tool")
            if not labels:
                continue
            categories: set[str] = set()
            if "RECOVERY_OPPORTUNITY" in labels or "RECOVERED" in labels:
                categories.add("recovery")
            if "PREMATURE_STOP" in labels:
                categories.add("premature_finish")
            if "INVALID_ACTION" in labels or "invalid_tool" in labels:
                categories.add("invalid_tool")
            if "LOCALIZATION_MISS" in labels:
                categories.add("localization")
            for category in categories:
                case = {
                    "schema_version": "1.0.0",
                    "run_dir": str(root),
                    "run_id": step.get("run_id"),
                    "episode_id": step.get("episode_id"),
                    "task_id": step.get("task_id"),
                    "step_index": step.get("step_index"),
                    "labels": sorted(labels),
                    "action": step.get("action"),
                    "tool_result": step.get("tool_result"),
                    "episode": episodes.get(step.get("episode_id")),
                }
                output = output_root / category / f"{step.get('run_id', 'run')}_{step.get('step_index', 0)}.json"
                _write_new(output, case)
                outputs.append(str(output.resolve()))
                counts[category] = counts.get(category, 0) + 1
    return {"report_type": "failure_cases", "counts": counts, "outputs": outputs}


def reproducibility_report(run_dirs: Iterable[Path], output: Path, gold_files: Path | None = None) -> dict[str, Any]:
    run_list = [path.resolve() for path in run_dirs]
    gold = load_gold_files(gold_files) if gold_files else None
    tasks = collect_task_metrics(run_list, gold_by_task=gold)
    report = {
        "schema_version": "1.0.0",
        "report_type": "reproducibility",
        "run_count": len(run_list),
        "runs": [
            {
                "run_dir": str(path),
                "run_metadata": json.loads((path / "run.json").read_text(encoding="utf-8"))
                if (path / "run.json").is_file() else None,
            }
            for path in run_list
        ],
        "metrics": aggregate_metrics(tasks),
        "evidence_boundary": {
            "status": "measured" if tasks else "unverified",
            "smoke_is_not_benchmark": True,
        },
    }
    _write_new(output, report)
    return {"output": str(output.resolve()), "run_count": len(run_list)}


def markdown_report(run_dirs: Iterable[Path], output: Path, gold_files: Path | None = None) -> dict[str, Any]:
    run_list = [path.resolve() for path in run_dirs]
    gold = load_gold_files(gold_files) if gold_files else None
    metrics = aggregate_metrics(collect_task_metrics(run_list, gold_by_task=gold))
    rows = [
        ("Resolve Rate", metrics.get("resolve_rate")),
        ("Test Pass Rate", metrics.get("test_pass_rate")),
        ("Recovery Rate", metrics.get("episode_recovery_rate")),
        ("Premature Finish Rate", metrics.get("premature_stop_rate")),
        ("Duplicate Action Rate", metrics.get("duplicate_action_rate")),
        ("Invalid Tool Rate", metrics.get("invalid_action_rate")),
        ("Localization@5", metrics.get("localization_at_5")),
        ("Replan Count", metrics.get("replan_count")),
    ]
    lines = [
        "# CAPO Evaluation Report", "", 
        "> Status boundary: measured values only. Missing evidence is `Unverified`.", "",
        "| Metric | Value | Status |", "|---|---:|---|",
    ]
    for name, value in rows:
        status = "Measured" if value is not None else "Unverified"
        rendered = "Unverified" if value is None else str(value)
        lines.append(f"| {name} | {rendered} | {status} |")
    lines.extend(["", f"Run count: `{len(run_list)}`", "", "Source runs:"])
    lines.extend(f"- `{path}`" for path in run_list)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "run_count": len(run_list)}
