from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..models import PolicyOutput


class AgentPolicy(Protocol):
    def decide(self, context: dict[str, Any]) -> PolicyOutput:
        """Return one structured agent decision for the current context."""


class ScriptedPolicy:
    """Deterministic policy used to verify the harness without a model runtime."""

    def __init__(self, decisions: Sequence[dict[str, Any]]) -> None:
        self._decisions = list(decisions)
        self._index = 0

    def decide(self, context: dict[str, Any]) -> PolicyOutput:
        del context
        if self._index >= len(self._decisions):
            payload = {
                "intent": "No scripted decision remains.",
                "action": {
                    "type": "finish",
                    "args": {"status": "give_up", "summary": "Script exhausted."},
                },
            }
        else:
            payload = self._decisions[self._index]
            self._index += 1
        return PolicyOutput(payload=payload)

