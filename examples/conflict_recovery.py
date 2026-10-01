"""Deterministic human-edit recovery demo; no model API or credentials needed.

Run from the project root after installing the package:
    python examples/conflict_recovery.py --output artifacts/conflict-demo
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from coding_agent_policy_opt.config import load_experiment_config
from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment, RepositoryProfile
from coding_agent_policy_opt.harness.policy import AdaptiveHarnessPolicy
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.harness.smoke import create_smoke_repository, smoke_decisions
from coding_agent_policy_opt.models import PolicyOutput


class ConflictDemoPolicy:
    def __init__(self, root: Path):
        self.root = root
        self.calls = 0
        self.decisions = smoke_decisions()
        self.decisions[2]["action"]["args"]["patch"] = self.decisions[2]["action"]["args"]["patch"].replace(
            "@@ -1,3 +1,3 @@", "@@ -3,3 +3,3 @@"
        )
        self.baseline_accepts_stale_patch = False

    def decide(self, context):
        if self.calls == 1:
            source = self.root / "calculator.py"
            source.write_bytes(source.read_bytes().replace(b"# owner: agent", b"# owner: human"))
            result = subprocess.run(
                ["git", "apply", "--check", "-"], cwd=self.root,
                input=self.decisions[2]["action"]["args"]["patch"].encode(), capture_output=True,
            )
            self.baseline_accepts_stale_patch = result.returncode == 0
        sequence = [1, 2, 1, 2, 3, 4, 5]
        if self.calls >= len(sequence):
            return PolicyOutput(payload={
                "intent": "Stop the scripted demo after an unexpected result.",
                "action": {"type": "finish", "args": {"status": "give_up", "summary": "Inspect run artifacts."}},
            })
        decision = self.decisions[sequence[self.calls]]
        self.calls += 1
        return PolicyOutput(payload=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = output / "repo"
    task = create_smoke_repository(root)
    source = root / "calculator.py"
    source.write_bytes(b"# owner: agent\n\n" + source.read_bytes())
    config = load_experiment_config(Path(__file__).resolve().parents[1] / "configs/experiment/smoke.yaml")
    env = RepositoryEnvironment(root, RepositoryProfile(
        config.repository.test_command, config.repository.default_test_args, 30,
    ))
    policy = ConflictDemoPolicy(root)
    episode = AgentHarness(config, env, RunArtifactStore(output / "run"),
                           harness_policy=AdaptiveHarnessPolicy()).run(task, policy)
    steps = [json.loads(line) for line in (output / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {
        "mode": "scripted_no_model",
        "git_accepts_stale_patch": policy.baseline_accepts_stale_patch,
        "guard_rejected_stale_patch": any(s["tool_result"].get("error_code") == "PATCH_CONFLICT" for s in steps),
        "human_edit_preserved": b"# owner: human" in source.read_bytes(),
        "task_resolved": episode.resolved,
        "steps": len(steps),
        "run_directory": str(output / "run"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(report[key] for key in (
        "git_accepts_stale_patch", "guard_rejected_stale_patch", "human_edit_preserved", "task_resolved",
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
