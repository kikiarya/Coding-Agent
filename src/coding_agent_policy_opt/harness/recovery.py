"""Version-bound validation and a bounded, serial candidate history.

Snapshots contain ordinary UTF-8 text only. They are not write authorization:
switching is performed through the environment's existing patch checks.
"""
from __future__ import annotations

import difflib
from copy import deepcopy
from typing import Any

from ..hashing import sha256_json


RECOVERY_PROFILES = {
    "baseline": {"candidates": False, "evidence_guidance": False, "candidate_switching": False},
    "record": {"candidates": True, "evidence_guidance": False, "candidate_switching": False},
    "evidence": {"candidates": False, "evidence_guidance": True, "candidate_switching": False},
    "full": {"candidates": True, "evidence_guidance": True, "candidate_switching": True},
}


class RecoveryState:
    VERSION = 1

    def __init__(self, *, candidates_enabled: bool = True) -> None:
        self.candidates_enabled = candidates_enabled
        self.evidence: dict[str, Any] | None = None
        self.candidates: dict[str, dict[str, Any]] = {}
        self.roles: dict[str, str | None] = dict(current=None, best_observed=None, backup=None)
        self.switches = 0
        self.regression = False

    def observe(self, workspace: str, protocol: str) -> dict[str, Any] | None:
        if self.evidence is None:
            return None
        value = deepcopy(self.evidence)
        if value["workspace_version"] != workspace or value["protocol"] != protocol:
            value["validity"] = "stale"
        return value

    def validated(self, *, before: str, after: str, protocol: str,
                  command: list[str], state: dict[str, Any], stable: bool,
                  files: dict[str, str] | None) -> dict[str, Any]:
        evidence = {"workspace_version": after, "protocol": protocol,
                    "command": command, "test_state": state,
                    "validity": "valid" if stable and before == after else "unknown"}
        evidence["id"] = sha256_json(evidence)
        self.evidence = evidence
        self.regression = False
        best = deepcopy(self.candidates.get(self.roles["best_observed"] or ""))
        count = state.get("fail_to_pass_count")
        if best and best.get("evidence"):
            old = best["evidence"]
            old_count = old["test_state"].get("fail_to_pass_count")
            self.regression = bool(evidence["validity"] == "valid"
                                   and old["protocol"] == protocol
                                   and old["command"] == command
                                   and count is not None and old_count is not None
                                   and count > old_count)
        if files is not None and evidence["validity"] == "valid":
            candidate = self.capture(after, files)
            if candidate is not None:
                candidate["evidence"] = deepcopy(evidence)
                old_evidence = best.get("evidence") if best else None
                comparable = bool(old_evidence and old_evidence["protocol"] == protocol
                                  and old_evidence["command"] == command)
                old_count = old_evidence["test_state"].get("fail_to_pass_count") if comparable else None
                if count is not None and (not comparable or old_count is None or count < old_count):
                    self.roles["best_observed"] = after
                self._prune()
        return deepcopy(evidence)

    def capture(self, workspace: str, files: dict[str, str]) -> dict[str, Any] | None:
        if not self.candidates_enabled:
            return None
        previous = self.roles["current"]
        if previous != workspace:
            self.roles["backup"] = previous
        self.roles["current"] = workspace
        self.candidates.setdefault(workspace, {"id": workspace, "files": deepcopy(files), "evidence": None})
        self._prune()
        return self.candidates[workspace]

    def _prune(self) -> None:
        keep = set(self.roles.values())
        self.candidates = {key: value for key, value in self.candidates.items() if key in keep}

    def summary(self, workspace: str, protocol: str, *, guidance: bool = True,
                expose_candidates: bool = True) -> dict[str, Any]:
        evidence = self.observe(workspace, protocol)
        return {"evidence": evidence if guidance else None,
                "regression": self.regression if guidance else False,
                "roles": dict(self.roles) if expose_candidates else dict.fromkeys(self.roles),
                "switches": self.switches,
                "candidates": [{"id": c["id"], "files": sorted(c["files"]),
                                "evidence": c["evidence"] if guidance else None}
                               for c in self.candidates.values()] if expose_candidates else []}

    def export(self) -> dict[str, Any]:
        return deepcopy({"version": self.VERSION, "evidence": self.evidence,
                         "candidates": self.candidates, "roles": self.roles,
                         "switches": self.switches, "regression": self.regression})

    def restore(self, value: dict[str, Any]) -> None:
        if value.get("version") != self.VERSION:
            raise ValueError("Unsupported recovery state version")
        candidates = value.get("candidates", {})
        roles = value.get("roles", {})
        if len(candidates) > 3 or set(roles) != set(self.roles):
            raise ValueError("Invalid bounded candidate state")
        if any(item is not None and item not in candidates for item in roles.values()):
            raise ValueError("Missing candidate referenced by role")
        self.evidence = deepcopy(value.get("evidence"))
        self.candidates = deepcopy(candidates) if self.candidates_enabled else {}
        self.roles = deepcopy(roles) if self.candidates_enabled else dict.fromkeys(self.roles)
        self.switches = int(value.get("switches", 0))
        self.regression = bool(value.get("regression", False))

    def patch_to(self, candidate_id: str, current: dict[str, str]) -> str:
        if candidate_id not in self.candidates:
            raise ValueError("Unknown or evicted candidate")
        target = self.candidates[candidate_id]["files"]
        chunks: list[str] = []
        for path in sorted(current.keys() | target.keys()):
            old, new = current.get(path), target.get(path)
            if old == new:
                continue
            # Keep the generated diff within the ordinary-text patch contract.
            if any(char in path for char in ('\n', '\r', '"', '\t')):
                raise ValueError("Candidate path is not supported by text patches")
            if any(text and not text.endswith('\n') for text in (old, new)):
                raise ValueError("Candidate switching requires newline-terminated text")
            chunks.append(f"diff --git a/{path} b/{path}\n")
            chunks.extend(difflib.unified_diff((old or '').splitlines(keepends=True),
                                              (new or '').splitlines(keepends=True),
                                              fromfile=f'a/{path}' if old is not None else '/dev/null',
                                              tofile=f'b/{path}' if new is not None else '/dev/null'))
        return ''.join(chunks)
