from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ..hashing import canonical_json, sha256_json
from ..schema import SchemaRegistry


def agent_system_prompt(
    schemas: SchemaRegistry | None = None,
    *,
    allowed_tools: list[str] | tuple[str, ...] | None = None,
) -> str:
    registry = schemas or SchemaRegistry()
    action_schema = registry.schema("agent_action")
    visible_schema = deepcopy(action_schema)
    if allowed_tools is not None:
        allowed = set(allowed_tools)
        visible_refs = []
        visible_defs = {}
        for reference in action_schema.get("properties", {}).get("action", {}).get("oneOf", []):
            definition_name = reference.get("$ref", "").rsplit("/", 1)[-1]
            definition = action_schema.get("$defs", {}).get(definition_name, {})
            action_type = (
                definition.get("properties", {}).get("type", {}).get("const")
                if isinstance(definition, dict)
                else None
            )
            if action_type in allowed:
                visible_refs.append(reference)
                visible_defs[definition_name] = definition
        if not visible_refs:
            raise ValueError("Harness Policy must expose at least one registered tool")
        visible_schema["properties"]["action"]["oneOf"] = visible_refs
        visible_schema["$defs"] = visible_defs
    registered_tools = [
        definition["properties"]["type"]["const"]
        for definition in visible_schema.get("$defs", {}).values()
        if isinstance(definition, dict)
        and isinstance(definition.get("properties"), dict)
        and isinstance(definition["properties"].get("type"), dict)
        and "const" in definition["properties"]["type"]
    ]
    tool_surface = ", ".join(allowed_tools or sorted(registered_tools))
    return (
        "You are a repository coding agent. Return exactly one JSON object matching the "
        "AgentDecision schema. Do not use markdown fences, unregistered tools, shell commands, "
        "or hidden chain-of-thought. The intent field is one concise auditable decision summary. "
        "Work only inside the task repository, use test feedback, and verify the final patch before "
        "finish(status=\"success\").\n"
        "Before modifying an existing file, read_file in the current session. "
        "Reads include a full-file content_hash even for a partial line range. "
        "On PATCH_CONFLICT or FILE_NOT_READ, read every listed file and regenerate the patch "
        "against its current contents, preserving external edits. Never blindly retry a stale patch. "
        "After a successful patch or session resume, read again before further edits. "
        "Validation evidence belongs to a workspace version; stale or unknown evidence requires new tests. "
        "inspect_diff includes bounded candidates. To switch, read all affected files, then call apply_patch "
        "with candidate_id and the current workspace_version instead of patch. Switching requires fresh tests "
        "and is refused when external edits would be overwritten. Candidates are observed alternatives, not proven fixes. "
        "Only ordinary text changes, additions and deletions are supported by the local patch guard. "
        "If requirements and external edits cannot be reconciled, finish with a non-success status "
        "and explain the decision needed from the user.\n"
        f"The Harness Policy exposes these tools for this turn: {tool_surface}. "
        "Do not select a tool outside that surface.\nAgentDecision JSON Schema:\n"
        + canonical_json(visible_schema)
    )


def build_agent_messages(
    context: dict[str, Any],
    *,
    history: list[dict[str, Any]] | None = None,
    dropped_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    task = context["task"]
    selected_history = context.get("history", []) if history is None else history
    task_context: dict[str, Any] = {
        "task_id": task["task_id"],
        "repo_id": task["repo_id"],
        "issue": task["issue"],
    }
    harness_policy = context.get("harness_policy")
    if isinstance(harness_policy, dict):
        task_context["harness_policy"] = harness_policy
    if dropped_history:
        summaries: list[dict[str, Any]] = []
        for index, turn in enumerate(dropped_history):
            observation = turn.get("observation")
            rendered = canonical_json(observation)
            observation_type = (
                observation.get("tool", "unknown")
                if isinstance(observation, dict)
                else type(observation).__name__
            )
            observation_hash = (
                observation.get("observation_hash")
                if isinstance(observation, dict)
                else None
            )
            summaries.append(
                {
                    "history_index": index,
                    "observation_type": observation_type,
                    "observation_hash": observation_hash or sha256_json(observation),
                    "observation_utf8_bytes": len(rendered.encode("utf-8")),
                }
            )
        task_context["truncated_history_observations"] = summaries
    messages = [
        {
            "role": "system",
            "content": agent_system_prompt(
                allowed_tools=(
                    harness_policy.get("allowed_tools")
                    if isinstance(harness_policy, dict)
                    else None
                )
            ),
        },
        {
            "role": "user",
            "content": canonical_json(task_context),
        },
    ]
    for turn in selected_history:
        assistant = turn.get("assistant")
        assistant_content = (
            canonical_json(assistant) if not isinstance(assistant, str) else assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append(
            {"role": "tool", "content": canonical_json(turn["observation"])}
        )
    messages.append(
        {
            "role": "user",
            "content": canonical_json(
                {
                    "instruction": "Select the next registered action.",
                    "step_index": context["step_index"],
                    "current_environment_state": context["environment_state"],
                }
            ),
        }
    )
    return messages


def fit_messages_to_context(
    tokenizer: Any,
    context: dict[str, Any],
    *,
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], list[int], int]:
    history = list(context.get("history", []))
    for dropped in range(len(history) + 1):
        messages = build_agent_messages(
            context,
            history=history[dropped:],
            dropped_history=history[:dropped],
        )
        input_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if len(input_ids) <= max_input_tokens:
            return messages, input_ids, dropped
    raise ValueError(
        "System prompt, issue, and current state exceed the frozen model input budget"
    )


def parse_agent_decision(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"_raw_model_output": stripped}
