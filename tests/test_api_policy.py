from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from coding_agent_policy_opt.agent.api_policy import (
    APICompletion,
    APIModelPolicy,
    estimate_chat_tokens,
    fit_api_messages_to_context,
    load_api_model_policy,
    load_api_model_policy_config,
)

from tests.helpers import PROJECT_ROOT


def agent_context() -> dict[str, Any]:
    return {
        "task": {
            "task_id": "task-api-1",
            "repo_id": "owner/repo",
            "issue": "Fix the parser bug and verify the tests.",
        },
        "step_index": 2,
        "environment_state": {
            "base_commit": "a" * 40,
            "diff_hash": "sha256:" + "0" * 64,
            "test_state": {"status": "failed", "fail_to_pass_count": 1},
        },
        "history": [
            {
                "assistant": {
                    "intent": "Inspect the old implementation.",
                    "action": {
                        "type": "read_file",
                        "args": {"path": "parser.py", "start_line": 1, "end_line": 40},
                    },
                },
                "observation": {
                    "tool": "read_file",
                    "data": {"content": "OLDEST" * 400},
                },
            },
            {
                "assistant": {
                    "intent": "Run the parser tests.",
                    "action": {"type": "run_tests", "args": {}},
                },
                "observation": {
                    "tool": "run_tests",
                    "data": {"content": "LATEST", "failed": 1},
                },
            },
        ],
    }


class FakeAPIClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> APICompletion:
        self.calls.append(
            {
                "messages": messages,
                "model_id": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return APICompletion(self.content, prompt_tokens=321, completion_tokens=17)


class APIModelPolicyConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "model" / "api_policy.yaml"

    def test_config_contains_only_environment_references_not_secrets(self) -> None:
        config = load_api_model_policy_config(self.path)

        self.assertEqual(config.base_url_env, "CAPO_API_BASE_URL")
        self.assertEqual(config.api_key_env, "CAPO_API_KEY")
        self.assertEqual(config.model_id_env, "CAPO_API_MODEL")
        self.assertEqual(config.max_context_tokens, 16_384)

    def test_literal_api_key_field_is_rejected(self) -> None:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        raw["endpoint"]["api_key"] = "do-not-store-secrets"
        with tempfile.TemporaryDirectory(prefix="capo_api_config_") as temporary:
            path = Path(temporary) / "api.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "environment references"):
                load_api_model_policy_config(path)

    def test_remote_plain_http_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            load_api_model_policy(
                self.path,
                environ={
                    "CAPO_API_BASE_URL": "http://api.example.com/v1",
                    "CAPO_API_KEY": "secret-value",
                    "CAPO_API_MODEL": "model-name",
                },
            )

    def test_local_vllm_http_endpoint_is_allowed(self) -> None:
        policy = load_api_model_policy(
            self.path,
            environ={
                "CAPO_API_BASE_URL": "http://127.0.0.1:8000/v1",
                "CAPO_API_KEY": "local-dummy-key",
                "CAPO_API_MODEL": "local-model",
            },
        )

        self.assertEqual(policy.model_id, "local-model")
        self.assertNotIn("local-dummy-key", repr(policy.client))


class APIModelPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_api_model_policy_config(
            PROJECT_ROOT / "configs" / "model" / "api_policy.yaml"
        )

    def test_policy_uses_live_context_and_returns_dynamic_agent_action(self) -> None:
        client = FakeAPIClient(
            '{"intent":"Re-run tests after inspecting the failure.",'
            '"action":{"type":"run_tests","args":{}}}'
        )
        policy = APIModelPolicy(client, self.config, "api-model")

        output = policy.decide(agent_context())
        call = client.calls[0]
        rendered = "\n".join(item["content"] for item in call["messages"])

        self.assertEqual(output.payload["action"]["type"], "run_tests")
        self.assertEqual(output.input_tokens, 321)
        self.assertEqual(output.output_tokens, 17)
        self.assertIn("Fix the parser bug", rendered)
        self.assertIn("LATEST", rendered)
        self.assertNotIn("tool", [item["role"] for item in call["messages"]])
        self.assertIn("tool_observation", rendered)
        self.assertEqual(call["model_id"], "api-model")

    def test_invalid_model_output_is_not_silently_repaired(self) -> None:
        client = FakeAPIClient("I think you should inspect parser.py")
        policy = APIModelPolicy(client, self.config, "api-model")

        output = policy.decide(agent_context())

        self.assertIn("_raw_model_output", output.payload)

    def test_context_fitting_drops_oldest_turn_and_keeps_latest_state(self) -> None:
        value = agent_context()
        without_oldest, _, _ = fit_api_messages_to_context(
            {**value, "history": value["history"][1:]},
            max_input_tokens=100_000,
        )
        budget = estimate_chat_tokens(without_oldest) + 150

        messages, _, dropped = fit_api_messages_to_context(
            value,
            max_input_tokens=budget,
        )
        rendered = "\n".join(item["content"] for item in messages)

        self.assertEqual(dropped, 1)
        self.assertNotIn("OLDEST", rendered)
        self.assertIn("LATEST", rendered)
        self.assertIn("truncated_history_observations", rendered)


if __name__ == "__main__":
    unittest.main()
