from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from coding_agent_policy_opt.training.sft import (
    PINNED_MODEL_REVISION,
    analyze_tokenized_dataset,
    encode_decision_sample,
    evaluate_token_budget_gate,
    load_sft_config,
)
from coding_agent_policy_opt.data.manifest import write_jsonl

from tests.helpers import PROJECT_ROOT


class PrefixTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.assert_tokenize(tokenize)
        text = "".join(
            f"<{message['role']}>{message['content']}|" for message in conversation
        )
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))

    @staticmethod
    def assert_tokenize(tokenize: bool) -> None:
        if not tokenize:
            raise AssertionError("test tokenizer only supports tokenize=True")


class NonPrefixTokenizer(PrefixTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        values = super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return ([255] + values) if not add_generation_prompt else values


def sample() -> dict:
    return {
        "sample_id": "sample-1",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Fix the bug."},
        ],
        "target": {
            "intent": "Read the file.",
            "action": {
                "type": "read_file",
                "args": {"path": "a.py", "start_line": 1, "end_line": 10},
            },
        },
    }


class SFTConfigTest(unittest.TestCase):
    def test_success_and_policy_configs_share_frozen_base_and_training_profile(self) -> None:
        success = load_sft_config(PROJECT_ROOT / "configs" / "sft" / "success_sft.yaml")
        policy = load_sft_config(PROJECT_ROOT / "configs" / "sft" / "policy_sft.yaml")

        self.assertEqual(success.model["model_revision"], PINNED_MODEL_REVISION)
        self.assertEqual(policy.model["model_revision"], PINNED_MODEL_REVISION)
        self.assertEqual(success.training, policy.training)
        self.assertEqual(success.data.dataset, "D_success")
        self.assertEqual(policy.data.dataset, "D_policy")
        self.assertNotEqual(success.output_dir, policy.output_dir)

    def test_loader_rejects_silent_hyperparameter_drift(self) -> None:
        source = PROJECT_ROOT / "configs" / "sft" / "success_sft.yaml"
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        value["training"]["lora_rank"] = 16
        with tempfile.TemporaryDirectory(prefix="capo_sft_config_") as temporary:
            path = Path(temporary) / "drift.yaml"
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lora_rank"):
                load_sft_config(path)


class SFTEncodingTest(unittest.TestCase):
    def test_only_target_assistant_tokens_receive_loss(self) -> None:
        encoded = encode_decision_sample(PrefixTokenizer(), sample(), 4096)

        first_target = next(
            index for index, label in enumerate(encoded["labels"]) if label != -100
        )
        self.assertGreater(first_target, 0)
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_target]))
        self.assertEqual(
            encoded["labels"][first_target:], encoded["input_ids"][first_target:]
        )
        self.assertEqual(len(encoded["input_ids"]), len(encoded["attention_mask"]))

    def test_non_prefix_preserving_template_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not prefix-preserving"):
            encode_decision_sample(NonPrefixTokenizer(), sample(), 4096)

    def test_overlength_sample_is_rejected_not_silently_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum is 10"):
            encode_decision_sample(PrefixTokenizer(), sample(), 10)

    def test_tokenization_report_counts_only_supervised_target_tokens(self) -> None:
        with tempfile.TemporaryDirectory(prefix="capo_token_report_") as temporary:
            path = Path(temporary) / "samples.jsonl"
            write_jsonl(path, [sample()])

            report = analyze_tokenized_dataset(path, PrefixTokenizer())

        self.assertEqual(report["samples"], 1)
        self.assertGreater(report["total_target_tokens"], 0)
        self.assertGreater(
            report["total_sequence_tokens"], report["total_target_tokens"]
        )

    def test_token_budget_gate_uses_one_percent_target_token_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="capo_token_gate_") as temporary:
            root = Path(temporary)
            success = root / "success.json"
            policy = root / "policy.json"
            base = {
                "schema_version": "2.0.0",
                "report_type": "tokenizer_preflight",
                "samples": 600,
                "tokenizer_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "tokenizer_revision": PINNED_MODEL_REVISION,
            }
            success.write_text(
                json.dumps(base | {"dataset": "D_success", "total_target_tokens": 10_000}),
                encoding="utf-8",
            )
            policy.write_text(
                json.dumps(base | {"dataset": "D_policy", "total_target_tokens": 10_050}),
                encoding="utf-8",
            )

            passed = evaluate_token_budget_gate(success, policy)
            policy.write_text(
                json.dumps(base | {"dataset": "D_policy", "total_target_tokens": 10_200}),
                encoding="utf-8",
            )
            failed = evaluate_token_budget_gate(success, policy)

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
