from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from ..hashing import canonical_json, sha256_json
from ..agent.model_policy import QwenPolicyConfig, load_qwen_policy_config
from ..schema import SchemaRegistry


TOOL_CASES: dict[str, dict[str, Any]] = {
    "search_code": {"query": "class Parser", "path": ".", "glob": "*.py", "max_results": 20},
    "read_file": {"path": "src/parser.py", "start_line": 1, "end_line": 40},
    "apply_patch": {
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    },
    "run_tests": {"target": "tests/test_parser.py", "extra_args": ["-q"]},
    "inspect_diff": {},
    "finish": {"status": "success", "summary": "Patch verified."},
}

_SCHEMA_KEYS = {
    "search_code": "searchCode",
    "read_file": "readFile",
    "apply_patch": "applyPatch",
    "run_tests": "runTests",
    "inspect_diff": "inspectDiff",
    "finish": "finish",
}


class ToolChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


def registered_tool_schemas(
    schemas: SchemaRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = schemas or SchemaRegistry()
    decision_schema = registry.schema("agent_action")
    definitions = decision_schema["$defs"]
    tools: list[dict[str, Any]] = []
    for name, definition_key in _SCHEMA_KEYS.items():
        action_definition = definitions[definition_key]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Registered Coding Harness tool: {name}.",
                    "parameters": deepcopy(action_definition["properties"]["args"]),
                },
            }
        )
    return tools


def grpo_chat_template_preflight(
    tokenizer: ToolChatTokenizer,
    config: QwenPolicyConfig,
) -> dict[str, Any]:
    tools = registered_tool_schemas()
    cases: list[dict[str, Any]] = []
    for tool_name, arguments in TOOL_CASES.items():
        before_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use only registered tools."},
            {"role": "user", "content": "Fix the assigned repository issue."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            },
        ]
        after_messages = [
            *before_messages,
            {
                "role": "tool",
                "name": tool_name,
                "content": canonical_json(
                    {"ok": True, "tool": tool_name, "observation_hash": "sha256:" + "0" * 64}
                ),
            },
        ]
        before_ids = list(
            tokenizer.apply_chat_template(
                before_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        after_ids = list(
            tokenizer.apply_chat_template(
                after_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        prefix_preserving = bool(
            len(after_ids) > len(before_ids)
            and after_ids[: len(before_ids)] == before_ids
        )
        cases.append(
            {
                "tool": tool_name,
                "before_tokens": len(before_ids),
                "after_tokens": len(after_ids),
                "prefix_preserving": prefix_preserving,
            }
        )
    passed_count = sum(case["prefix_preserving"] for case in cases)
    return {
        "schema_version": "2.0.0",
        "report_type": "grpo_chat_template_preflight",
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "prompt_version": config.prompt_version,
        "tool_schema_hash": sha256_json(tools),
        "tools_checked": len(cases),
        "tools_passed": passed_count,
        "passed": passed_count == len(TOOL_CASES),
        "cases": cases,
    }


def run_grpo_chat_template_preflight(model_config_path: Path) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional RL dependencies before template preflight") from exc
    config = load_qwen_policy_config(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model["tokenizer_id"],
        revision=config.model["tokenizer_revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    return grpo_chat_template_preflight(tokenizer, config)
