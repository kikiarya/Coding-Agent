from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.evaluation.exploration import analyze_exploration
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


class ExplorationAnalysisTest(unittest.TestCase):
    def test_trace_analysis_separates_localization_and_context_metrics(self) -> None:
        fixture = SmokeRepository()
        temporary = tempfile.TemporaryDirectory(prefix="capo_exploration_")
        try:
            run_dir = Path(temporary.name) / "run"
            AgentHarness(
                fixture.config,
                fixture.environment,
                RunArtifactStore(run_dir),
            ).run(fixture.task, ScriptedPolicy(smoke_decisions()))
            report = analyze_exploration(
                [run_dir],
                gold_by_task={fixture.task.task_id: {"calculator.py"}},
            )
            self.assertEqual(report["task_count"], 1)
            self.assertEqual(report["localization_at_5"], 1.0)
            self.assertEqual(report["mean_first_relevant_action_index"], 1)
            self.assertEqual(report["mean_search_calls"], 1)
            self.assertEqual(report["context_trim_rate"], 0.0)
            self.assertEqual(report["harness_gate_count"], 0)
        finally:
            fixture.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
