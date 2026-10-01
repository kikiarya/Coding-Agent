"""Exercise candidate recovery through the real runner using a scripted policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_agent_policy_opt.config import load_experiment_config
from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment, RepositoryProfile
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.harness.smoke import create_smoke_repository, smoke_decisions
from coding_agent_policy_opt.models import PolicyOutput


class RecoveryDemoPolicy:
    def __init__(self):
        self.index = 0
        self.best = None
        self.saw_regression = False

    def decide(self, context):
        sequence = [1, 2, 3, 1, "regress", 3, 1, "restore", 3, 4, 5]
        step = sequence[self.index] if self.index < len(sequence) else 5
        self.index += 1
        decisions = smoke_decisions()
        recovery = context["environment_state"]["recovery"]
        if step == "regress":
            self.best = recovery["roles"]["best_observed"]
            text = decisions[2]["action"]["args"]["patch"]
            text = text.replace("-    return a - b", "-    return a + b").replace("+    return a + b", "+    return a * b")
            value = {"intent": "Inject a controlled edit regression.",
                     "action": {"type": "apply_patch", "args": {"patch": text}}}
        elif step == "restore":
            self.saw_regression = recovery["regression"]
            value = {"intent": "Restore the earlier observed candidate after rereading.",
                     "action": {"type": "apply_patch", "args": {
                         "candidate_id": self.best,
                         "workspace_version": context["environment_state"]["workspace_version"]}}}
        else:
            value = decisions[step]
        return PolicyOutput(payload=value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = create_smoke_repository(output / "repo")
    config = load_experiment_config(Path(__file__).resolve().parents[1] / "configs/experiment/smoke.yaml")
    environment = RepositoryEnvironment(output / "repo", RepositoryProfile(
        config.repository.test_command, config.repository.default_test_args, 30))
    policy = RecoveryDemoPolicy()
    episode = AgentHarness(config, environment, RunArtifactStore(output / "run")).run(task, policy)
    report = {"mode": "scripted_no_model", "harness": "static_v1",
              "regression_observed": policy.saw_regression,
              "candidate_switches": environment.recovery.switches,
              "candidate_count": len(environment.recovery.candidates),
              "resolved": episode.resolved, "steps": episode.total_steps,
              "agent_test_calls": episode.test_calls, "evaluator_test_calls": 2}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (report["resolved"] and report["regression_observed"] and report["candidate_switches"] == 1):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
