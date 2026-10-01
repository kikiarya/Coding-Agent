from __future__ import annotations

import unittest

from coding_agent_policy_opt.schema import SchemaRegistry
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SMOKE_CONFIG
from coding_agent_policy_opt.config import load_experiment_config


class ConfigAndSchemaTest(unittest.TestCase):
    def test_smoke_config_has_frozen_phase_zero_budgets(self) -> None:
        config = load_experiment_config(SMOKE_CONFIG)

        self.assertEqual(config.schema_version, "2.0.0")
        self.assertEqual(config.harness.max_tool_steps, 30)
        self.assertEqual(config.harness.max_test_calls, 8)
        self.assertEqual(config.harness.max_context_tokens, 16384)
        self.assertEqual(config.harness.max_decision_output_tokens, 1024)
        self.assertTrue(config.config_hash.startswith("sha256:"))

    def test_all_smoke_decisions_match_agent_action_schema(self) -> None:
        schemas = SchemaRegistry()

        for decision in smoke_decisions():
            schemas.validate("agent_action", decision)

    def test_schema_rejects_unregistered_tool(self) -> None:
        schemas = SchemaRegistry()
        invalid = {
            "intent": "Run an arbitrary command.",
            "action": {"type": "bash", "args": {"command": "dir"}},
        }

        with self.assertRaises(Exception):
            schemas.validate("agent_action", invalid)


if __name__ == "__main__":
    unittest.main()

