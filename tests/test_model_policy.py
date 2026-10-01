from __future__ import annotations

import unittest

from coding_agent_policy_opt.agent.model_policy import (
    agent_system_prompt,
    build_agent_messages,
    fit_messages_to_context,
    load_qwen_policy_config,
    parse_agent_decision,
)
from coding_agent_policy_opt.training.sft import PINNED_MODEL_REVISION

from tests.helpers import PROJECT_ROOT


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize:
            raise AssertionError("test tokenizer requires tokenize=True")
        text = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


def context() -> dict:
    return {
        "task": {
            "task_id": "task-1",
            "repo_id": "owner/repo",
            "issue": "Fix the parser bug.",
        },
        "step_index": 2,
        "environment_state": {
            "base_commit": "a" * 40,
            "diff_hash": "sha256:" + "0" * 64,
            "test_state": {"status": "failed", "fail_to_pass_count": 1},
        },
        "history": [
            {
                "assistant": {"intent": "old", "action": {"type": "read_file", "args": {}}},
                "observation": {"data": {"content": "OLDEST" * 50}},
            },
            {
                "assistant": {"intent": "new", "action": {"type": "run_tests", "args": {}}},
                "observation": {"data": {"content": "LATEST"}},
            },
        ],
    }


class ModelPolicyConfigTest(unittest.TestCase):
    def test_config_pins_model_tokenizer_and_three_decoding_modes(self) -> None:
        config = load_qwen_policy_config(
            PROJECT_ROOT / "configs" / "model" / "qwen_policy.yaml"
        )

        self.assertEqual(config.model["model_revision"], PINNED_MODEL_REVISION)
        self.assertEqual(config.model["tokenizer_revision"], PINNED_MODEL_REVISION)
        self.assertFalse(config.decoding["evaluation"].do_sample)
        self.assertEqual(config.decoding["collection_seed_42"].temperature, 0.7)
        self.assertEqual(config.decoding["collection_seed_43"].seed, 43)


class PromptConstructionTest(unittest.TestCase):
    def test_system_prompt_contains_registered_schema_and_no_shell_tool(self) -> None:
        prompt = agent_system_prompt()

        self.assertIn('"search_code"', prompt)
        self.assertIn('"apply_patch"', prompt)
        self.assertNotIn('"bash"', prompt)

    def test_system_prompt_schema_is_cropped_to_harness_tool_surface(self) -> None:
        prompt = agent_system_prompt(allowed_tools=("read_file", "run_tests", "finish"))
        self.assertIn('"read_file"', prompt)
        self.assertIn('"run_tests"', prompt)
        self.assertIn('"finish"', prompt)
        self.assertNotIn('"search_code"', prompt)

    def test_messages_include_full_observation_and_current_state(self) -> None:
        messages = build_agent_messages(context())
        rendered = "\n".join(message["content"] for message in messages)

        self.assertIn("LATEST", rendered)
        self.assertIn("fail_to_pass_count", rendered)
        self.assertIn("Fix the parser bug", rendered)

    def test_context_fit_drops_oldest_turn_pairs_but_keeps_issue_and_latest(self) -> None:
        tokenizer = CharacterTokenizer()
        value = context()
        no_oldest = build_agent_messages(
            value,
            history=value["history"][1:],
            dropped_history=value["history"][:1],
        )
        budget = len(
            tokenizer.apply_chat_template(
                no_oldest, tokenize=True, add_generation_prompt=True
            )
        )

        messages, input_ids, dropped = fit_messages_to_context(
            tokenizer, value, max_input_tokens=budget
        )
        rendered = "\n".join(message["content"] for message in messages)

        self.assertEqual(dropped, 1)
        self.assertEqual(len(input_ids), budget)
        self.assertNotIn("OLDEST", rendered)
        self.assertIn("truncated_history_observations", rendered)
        self.assertIn("observation_utf8_bytes", rendered)
        self.assertIn("sha256:", rendered)
        self.assertIn("LATEST", rendered)
        self.assertIn("Fix the parser bug", rendered)

    def test_non_json_or_markdown_fenced_output_is_not_repaired(self) -> None:
        valid = parse_agent_decision('{"intent":"x","action":{}}')
        fenced = parse_agent_decision('```json\n{"intent":"x"}\n```')

        self.assertEqual(valid["intent"], "x")
        self.assertIn("_raw_model_output", fenced)


if __name__ == "__main__":
    unittest.main()
