from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import yaml

from ..hashing import canonical_json, sha256_json
from .prompts import build_agent_messages, parse_agent_decision
from ..models import PolicyOutput


@dataclass(frozen=True)
class APIModelPolicyConfig:
    base_url_env: str
    api_key_env: str
    model_id_env: str
    timeout_seconds: int
    max_context_tokens: int
    max_decision_output_tokens: int
    temperature: float
    top_p: float
    config_hash: str


@dataclass(frozen=True)
class APICompletion:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class APIChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> APICompletion: ...


def _require_env_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty environment variable name")
    if not value.replace("_", "A").isalnum() or not value[0].isalpha():
        raise ValueError(f"{field} contains unsupported characters")
    if value.upper() != value:
        raise ValueError(f"{field} must use an uppercase environment variable name")
    return value


def load_api_model_policy_config(path: Path) -> APIModelPolicyConfig:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("API policy config must be a YAML mapping")
    if raw.get("schema_version") != "2.0.0" or raw.get("profile") != "api_policy_v1":
        raise ValueError("API policy config must use schema 2.0.0 and api_policy_v1")
    if raw.get("transport") != "openai_chat_completions":
        raise ValueError("Only the OpenAI-compatible chat completions transport is registered")

    endpoint = raw.get("endpoint", {})
    context = raw.get("context", {})
    decoding = raw.get("decoding", {})
    if not isinstance(endpoint, dict) or not isinstance(context, dict) or not isinstance(decoding, dict):
        raise ValueError("endpoint, context, and decoding must be mappings")
    allowed_endpoint = {
        "base_url_env",
        "api_key_env",
        "model_id_env",
        "timeout_seconds",
    }
    if set(endpoint) != allowed_endpoint:
        raise ValueError("endpoint must contain only environment references and timeout_seconds")

    timeout_seconds = endpoint.get("timeout_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= 600
    ):
        raise ValueError("endpoint.timeout_seconds must be between 1 and 600")
    max_context_tokens = context.get("max_context_tokens")
    max_output_tokens = context.get("max_decision_output_tokens")
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens < 2_048
    ):
        raise ValueError("context.max_context_tokens must be at least 2048")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or not 64 <= max_output_tokens < max_context_tokens
    ):
        raise ValueError("context.max_decision_output_tokens is invalid")
    if set(context) != {"max_context_tokens", "max_decision_output_tokens"}:
        raise ValueError("context contains unregistered fields")

    temperature = decoding.get("temperature")
    top_p = decoding.get("top_p")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= float(temperature) <= 2
    ):
        raise ValueError("decoding.temperature must be between 0 and 2")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("decoding.top_p must be between 0 and 1")
    if set(decoding) != {"temperature", "top_p"}:
        raise ValueError("decoding contains unregistered fields")

    return APIModelPolicyConfig(
        base_url_env=_require_env_name(endpoint.get("base_url_env"), "endpoint.base_url_env"),
        api_key_env=_require_env_name(endpoint.get("api_key_env"), "endpoint.api_key_env"),
        model_id_env=_require_env_name(endpoint.get("model_id_env"), "endpoint.model_id_env"),
        timeout_seconds=timeout_seconds,
        max_context_tokens=max_context_tokens,
        max_decision_output_tokens=max_output_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        config_hash=sha256_json(raw),
    )


def resolve_api_model_environment(
    config: APIModelPolicyConfig,
    environ: Mapping[str, str],
) -> tuple[str, str, str]:
    values: list[str] = []
    for name in (config.base_url_env, config.api_key_env, config.model_id_env):
        value = environ.get(name, "").strip()
        if not value:
            raise ValueError(f"Required environment variable is not set: {name}")
        values.append(value)
    base_url, api_key, model_id = values
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API base URL must be an absolute HTTP(S) URL")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and parsed.hostname not in local_hosts:
        raise ValueError("Remote API endpoints must use HTTPS")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("API base URL must not contain credentials, query, or fragment")
    return base_url.rstrip("/"), api_key, model_id


class OpenAICompatibleChatClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> APICompletion:
        payload = json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"API request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"API request failed: {type(exc).__name__}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("API response does not contain choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("API response content must be a non-empty string")
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return APICompletion(
            content=content,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=(
                completion_tokens if isinstance(completion_tokens, int) else None
            ),
        )


def _chat_compatible_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    compatible: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "tool":
            compatible.append(
                {
                    "role": "user",
                    "content": canonical_json(
                        {"tool_observation": json.loads(message["content"])}
                    ),
                }
            )
        else:
            compatible.append(dict(message))
    return compatible


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    # Product-mode fallback when the remote provider does not expose its tokenizer.
    return sum(4 + max(1, (len(item["content"].encode("utf-8")) + 2) // 3) for item in messages)


def fit_api_messages_to_context(
    context: dict[str, Any],
    *,
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], int, int]:
    history = list(context.get("history", []))
    for dropped in range(len(history) + 1):
        messages = _chat_compatible_messages(
            build_agent_messages(
                context,
                history=history[dropped:],
                dropped_history=history[:dropped],
            )
        )
        estimated_tokens = estimate_chat_tokens(messages)
        if estimated_tokens <= max_input_tokens:
            return messages, estimated_tokens, dropped
    raise ValueError("System prompt, issue, and current state exceed the API input budget")


class APIModelPolicy:
    def __init__(
        self,
        client: APIChatClient,
        config: APIModelPolicyConfig,
        model_id: str,
    ) -> None:
        self.client = client
        self.config = config
        self.model_id = model_id

    def decide(self, context: dict[str, Any]) -> PolicyOutput:
        max_input = self.config.max_context_tokens - self.config.max_decision_output_tokens
        messages, estimated_input_tokens, _ = fit_api_messages_to_context(
            context,
            max_input_tokens=max_input,
        )
        started = time.monotonic()
        completion = self.client.complete(
            messages,
            model_id=self.model_id,
            max_tokens=self.config.max_decision_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        latency_ms = max(round((time.monotonic() - started) * 1000), 0)
        output_tokens = completion.completion_tokens
        if output_tokens is None:
            output_tokens = max(1, (len(completion.content.encode("utf-8")) + 2) // 3)
        return PolicyOutput(
            payload=parse_agent_decision(completion.content),
            input_tokens=completion.prompt_tokens or estimated_input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )


def load_api_model_policy(
    config_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> APIModelPolicy:
    config = load_api_model_policy_config(config_path)
    base_url, api_key, model_id = resolve_api_model_environment(
        config,
        os.environ if environ is None else environ,
    )
    client = OpenAICompatibleChatClient(base_url, api_key, config.timeout_seconds)
    return APIModelPolicy(client, config, model_id)
