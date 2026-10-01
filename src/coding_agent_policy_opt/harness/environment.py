from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..hashing import sha256_json, sha256_text
from ..models import ActionType, AgentAction, ErrorCode, TestState, TestStatus, ToolResult
from .recovery import RecoveryState


class AgentEnvironment(Protocol):
    """Minimal stateful environment contract consumed by the agent harness."""

    base_commit: str
    last_test_state: TestState | None

    def state_payload(self) -> dict[str, Any]: ...

    def diff_hash(self) -> str: ...

    def execute(self, action: AgentAction) -> ToolResult: ...

    def evaluate(self) -> ToolResult: ...


@dataclass(frozen=True)
class RepositoryProfile:
    test_command: tuple[str, ...]
    default_test_args: tuple[str, ...]
    timeout_seconds: int

    def resolved_test_command(self) -> list[str]:
        return [sys.executable if item == "{python}" else item for item in self.test_command]


class RepositoryEnvironment:
    def __init__(self, repo_root: Path, profile: RepositoryProfile) -> None:
        self.repo_root = repo_root.resolve()
        self.profile = profile
        if not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        if not (self.repo_root / ".git").exists():
            raise ValueError(f"Repository root is not a git worktree: {self.repo_root}")
        self.base_commit = self._git("rev-parse", "HEAD").stdout.strip()
        self.last_test_state: TestState | None = None
        # Session-local evidence: a resumed process must obtain fresh reads.
        self._read_versions: dict[str, str | None] = {}
        self._patch_conflicts: dict[str, dict[str, Any]] = {}
        self.recovery = RecoveryState()
        self.evidence_guidance = True
        self.candidate_switching = True

    def _git(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )

    def _safe_path(self, relative_path: str, *, must_exist: bool = True) -> Path:
        if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
            raise ValueError("Path must be a non-empty string")
        path = Path(relative_path)
        if path.is_absolute():
            raise PermissionError("Absolute paths are not allowed")
        if any(part.lower() == ".git" for part in path.parts):
            raise PermissionError("Git metadata is not part of the task repository interface")
        resolved = (self.repo_root / path).resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise PermissionError("Path is outside the task repository") from exc
        if must_exist and not resolved.exists():
            raise FileNotFoundError(relative_path)
        return resolved

    def diff_text(self) -> str:
        result = self._git("diff", "--binary", "--no-ext-diff")
        return result.stdout

    def diff_hash(self) -> str:
        return sha256_text(self.diff_text())

    def state_payload(self) -> dict[str, Any]:
        return {
            "base_commit": self.base_commit,
            "diff_hash": self.diff_hash(),
            "test_state": self.last_test_state.to_dict() if self.last_test_state else None,
            "patch_conflicts": list(self._patch_conflicts.values()),
            "write_precondition": "read_existing_files_in_this_session",
            "read_evidence_hash": sha256_json(self._read_versions),
            "workspace_version": self.workspace_version(),
            "recovery": self.recovery.summary(self.workspace_version(), self.validation_protocol(),
                                               guidance=self.evidence_guidance,
                                               expose_candidates=self.candidate_switching),
        }

    def workspace_files(self) -> dict[str, dict[str, Any]]:
        listing = self._git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        if listing.returncode:
            raise ValueError("Cannot enumerate workspace files")
        files: dict[str, dict[str, Any]] = {}
        for relative in sorted(set(listing.stdout.split("\x00")) - {""}):
            lexical = self.repo_root / relative
            if lexical.is_symlink():
                files[relative] = {"kind": "symlink", "hash": sha256_text(os.readlink(lexical))}
                continue
            path = self._safe_path(relative, must_exist=False)
            if path.is_file():
                files[relative] = {"kind": "file", "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "executable": bool(path.stat().st_mode & 0o111)}
            elif path.exists():
                files[relative] = {"kind": "unsupported"}
        return files

    def workspace_version(self) -> str:
        return sha256_json({"head": self._git("rev-parse", "HEAD").stdout.strip(),
                            "files": self.workspace_files()})

    def durable_state_payload(self) -> dict[str, Any]:
        return {"state_contract": "workspace-v1", "base_commit": self.base_commit,
                "workspace_version": self.workspace_version(), "diff_hash": self.diff_hash()}

    def validation_protocol(self) -> str:
        # Hash, never serialize, process environment values (which may contain secrets).
        return sha256_json({"command": self.profile.resolved_test_command(),
                            "default_args": self.profile.default_test_args,
                            "python": sys.version, "environment": dict(os.environ)})

    def _candidate_files(self) -> dict[str, str] | None:
        files: dict[str, str] = {}
        size = 0
        for relative, metadata in self.workspace_files().items():
            if metadata["kind"] != "file":
                return None
            path = self._safe_path(relative)
            size += path.stat().st_size
            if size > 8 * 1024 * 1024:
                return None
            try:
                content = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                return None
            if "\x00" in content:
                return None
            files[relative] = content
        return files

    def _capture_candidate(self) -> None:
        if self.recovery.candidates_enabled:
            before = self.workspace_version()
            files = self._candidate_files()
            if files is not None and self.workspace_version() == before:
                self.recovery.capture(before, files)

    def _invalidate_external_candidates(self) -> None:
        current = self.recovery.roles["current"]
        if current is not None and current != self.workspace_version():
            # Reading or testing external edits must not silently authorize rolling
            # them back later through a candidate from the old workspace lineage.
            self.recovery.candidates.clear()
            self.recovery.roles = dict.fromkeys(self.recovery.roles)
            self.recovery.regression = False

    def export_recovery_state(self) -> dict[str, Any]:
        return self.recovery.export()

    def restore_recovery_state(self, value: dict[str, Any]) -> None:
        self.recovery.restore(value)
        self.reset_read_evidence()

    def _switch_candidate(self, args: dict[str, Any]) -> ToolResult:
        if not self.candidate_switching:
            raise ValueError("Candidate switching is disabled for this experiment")
        current = self.workspace_version()
        if current != args["workspace_version"] or current != self.recovery.roles["current"]:
            return ToolResult.failure("apply_patch", ErrorCode.PATCH_CONFLICT,
                                      "Workspace changed outside the recorded candidate. Preserve external edits and replan.")
        if self.recovery.switches >= 3:
            return ToolResult.failure("apply_patch", ErrorCode.BUDGET_EXHAUSTED, "Candidate switch limit reached")
        files = self._candidate_files()
        if files is None:
            raise ValueError("Candidate switching supports bounded UTF-8 text workspaces only")
        patch = self.recovery.patch_to(args["candidate_id"], files)
        if not patch:
            raise ValueError("Candidate is already current")
        result = self._apply_patch({"patch": patch, "_expected_workspace": current})
        if result.ok:
            self.recovery.switches += 1
            return ToolResult.success("apply_patch", {
                **result.data, "restored_candidate": args["candidate_id"], "recovery_action": "run_tests",
            }, state_changed=result.state_changed)
        return result

    def state_hash(self) -> str:
        return sha256_json(self.state_payload())

    def reset_read_evidence(self) -> None:
        self._read_versions.clear()

    def execute(self, action: AgentAction) -> ToolResult:
        try:
            if not (action.type is ActionType.APPLY_PATCH and "candidate_id" in action.args):
                self._invalidate_external_candidates()
            if action.type is ActionType.SEARCH_CODE:
                return self._search_code(action.args)
            if action.type is ActionType.READ_FILE:
                return self._read_file(action.args)
            if action.type is ActionType.APPLY_PATCH:
                if "candidate_id" in action.args:
                    return self._switch_candidate(action.args)
                return self._apply_patch(action.args)
            if action.type is ActionType.RUN_TESTS:
                return self._run_tests(action.args)
            if action.type is ActionType.INSPECT_DIFF:
                return self._inspect_diff()
            if action.type is ActionType.FINISH:
                if action.args.get("status") == "success" and self._patch_conflicts:
                    return ToolResult.failure("finish", ErrorCode.PATCH_CONFLICT,
                                              "Resolve pending file conflicts before reporting success.",
                                              data={"conflicts": list(self._patch_conflicts.values())})
                evidence = self.recovery.observe(self.workspace_version(), self.validation_protocol())
                if action.args.get("status") == "success" and (
                    not evidence or evidence["validity"] != "valid"
                    or evidence["test_state"]["status"] != TestStatus.PASSED.value
                ):
                    return ToolResult.failure("finish", ErrorCode.VERIFICATION_REQUIRED,
                                              "Run passing tests on the current workspace before success.",
                                              data={"recovery_action": "run_tests", "evidence": evidence})
                return ToolResult.success("finish", {"accepted": True})
            return ToolResult.failure(
                action.type.value,
                ErrorCode.UNKNOWN_TOOL,
                f"Unknown tool: {action.type.value}",
            )
        except PermissionError as exc:
            return ToolResult.failure(
                action.type.value,
                ErrorCode.PATH_OUTSIDE_REPOSITORY,
                str(exc),
            )
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            return ToolResult.failure(
                action.type.value,
                ErrorCode.INVALID_ARGUMENT,
                str(exc),
                retryable=True,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                action.type.value,
                ErrorCode.TOOL_TIMEOUT,
                f"Tool exceeded {self.profile.timeout_seconds} seconds",
                retryable=True,
            )
        except OSError as exc:
            return ToolResult.failure(
                action.type.value,
                ErrorCode.ENVIRONMENT_ERROR,
                str(exc),
                retryable=True,
            )

    def _search_code(self, args: dict[str, Any]) -> ToolResult:
        query = args["query"]
        root = self._safe_path(args["path"])
        glob = args.get("glob", "*")
        max_results = int(args.get("max_results", 50))
        if max_results < 1 or max_results > 200:
            raise ValueError("max_results must be between 1 and 200")

        candidates = [root] if root.is_file() else root.rglob(glob)
        matches: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(matches) >= max_results:
                break
            if not candidate.is_file() or ".git" in candidate.parts:
                continue
            safe_candidate = self._safe_path(str(candidate.relative_to(self.repo_root)))
            try:
                lines = safe_candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        {
                            "path": safe_candidate.relative_to(self.repo_root).as_posix(),
                            "line": line_number,
                            "text": line[:1000],
                        }
                    )
                    if len(matches) >= max_results:
                        break
        return ToolResult.success(
            "search_code",
            {"query": query, "matches": matches, "truncated": len(matches) >= max_results},
        )

    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._safe_path(args["path"], must_exist=False)
        if path.exists() and not path.is_file():
            raise IsADirectoryError(args["path"])
        start_line = int(args["start_line"])
        end_line = int(args["end_line"])
        if start_line < 1 or end_line < start_line:
            raise ValueError("Invalid line range")
        if end_line - start_line + 1 > 400:
            raise ValueError("read_file is limited to 400 lines")
        relative = path.relative_to(self.repo_root).as_posix()
        raw = path.read_bytes() if path.exists() else None
        version = hashlib.sha256(raw).hexdigest() if raw is not None else None
        lines = raw.decode("utf-8").splitlines() if raw is not None else []
        evidence_changed = (
            relative not in self._read_versions
            or self._read_versions[relative] != version
            or relative in self._patch_conflicts
        )
        self._read_versions[relative] = version
        self._patch_conflicts.pop(relative, None)
        selected = [
            {"line": index, "text": lines[index - 1]}
            for index in range(start_line, min(end_line, len(lines)) + 1)
        ]
        return ToolResult.success(
            "read_file",
            {
                "path": path.relative_to(self.repo_root).as_posix(),
                "start_line": start_line,
                "end_line": min(end_line, len(lines)),
                "total_lines": len(lines),
                "lines": selected,
                "exists": raw is not None,
                "content_hash": version,
            },
            state_changed=evidence_changed,
        )

    def _validate_patch_paths(self, patch: str) -> tuple[str, ...]:
        if not patch.startswith("diff --git "):
            raise ValueError("Patch must be a git-style unified diff")
        for line in patch.splitlines():
            if line in {
                "new file mode 120000",
                "new mode 120000",
                "new file mode 160000",
                "new mode 160000",
            }:
                raise ValueError("Symlink and submodule patches are not allowed")
        headers = 0
        paths: set[str] = set()
        for line in patch.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            headers += 1
            patch_path = line[4:].split("\t", 1)[0].strip().strip('"')
            if patch_path == "/dev/null":
                continue
            if patch_path.startswith(("a/", "b/")):
                patch_path = patch_path[2:]
            self._safe_path(patch_path, must_exist=False)
            paths.add(patch_path)
        if headers < 2:
            raise ValueError("Patch is missing unified diff file headers")
        return tuple(sorted(paths))

    def _patch_preconditions(self, patch: str) -> dict[str, str | None]:
        """Accept only ordinary text modifications, additions and deletions.

        Validate every diff section, not just the union of ---/+++ headers:
        metadata-only sections otherwise bypass the version guard.
        """
        expected: dict[str, str | None] = {}
        seen: set[str] = set()
        for section in re.split(r"^diff --git ", patch, flags=re.MULTILINE)[1:]:
            lines = section.splitlines()
            if not lines:
                raise ValueError("Empty diff section")
            match = re.fullmatch(r"a/(.+) b/(.+)", lines[0])
            if not match or match[1] != match[2]:
                raise ValueError("Only same-path text diffs are supported; split renames into delete/add")
            relative = match[1]
            if any(line.startswith(("rename ", "copy ", "old mode ", "new mode ", "GIT binary", "Binary files")) for line in lines):
                raise ValueError("Rename, copy, mode-only and binary patches are not supported")
            old = [line[4:] for line in lines if line.startswith("--- ")]
            new = [line[4:] for line in lines if line.startswith("+++ ")]
            if len(old) != 1 or len(new) != 1 or not any(line.startswith("@@ ") for line in lines):
                raise ValueError("Every diff section must contain one pair of file headers and text hunks")
            if old[0] not in ("/dev/null", "a/" + relative) or new[0] not in ("/dev/null", "b/" + relative) or old[0] == new[0]:
                raise ValueError("Diff headers do not match the affected file")
            path = self._safe_path(relative, must_exist=False)
            lexical = self.repo_root / relative
            if any(part.is_symlink() for part in (lexical, *lexical.parents)):
                raise ValueError("Patching through symbolic links is not supported")
            key = path.relative_to(self.repo_root).as_posix()
            if key in seen:
                raise ValueError("A file may occur only once in a patch")
            seen.add(key)
            if old[0] == "/dev/null":
                expected[key] = None
            elif key not in self._read_versions:
                self._patch_conflicts[key] = {"path": key, "reason": "not_read"}
            else:
                expected[key] = self._read_versions[key]
        return expected

    def _check_patch_versions(self, expected: dict[str, str | None]) -> ToolResult | None:
        for relative, version in expected.items():
            path = self._safe_path(relative, must_exist=False)
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if current != version or (path.exists() and not path.is_file()):
                self._patch_conflicts[relative] = {
                    "path": relative,
                    "reason": "created" if version is None else "deleted" if not path.exists() else "modified",
                    "expected_hash": version, "current_hash": current,
                }
        if not self._patch_conflicts:
            return None
        conflicts = list(self._patch_conflicts.values())
        return ToolResult.failure(
            "apply_patch",
            ErrorCode.FILE_NOT_READ if all(item["reason"] == "not_read" for item in conflicts) else ErrorCode.PATCH_CONFLICT,
            "No patch applied. Read the listed files, preserve external edits, and regenerate the patch. Do not retry the old patch blindly.",
            retryable=True,
            data={"conflicts": conflicts, "recovery_action": "read_then_replan", "patch_applied": False},
        )

    def _apply_patch(self, args: dict[str, Any]) -> ToolResult:
        patch = args["patch"]
        patch_paths = self._validate_patch_paths(patch)
        expected = self._patch_preconditions(patch)
        conflict = self._check_patch_versions(expected)
        if conflict is not None:
            return conflict
        before_hash = self.diff_hash()
        # Preflight the complete group; never use --reject (partial application).
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=self.repo_root, input=patch.encode("utf-8"), capture_output=True, check=False,
        )
        if check.returncode:
            return ToolResult.failure("apply_patch", ErrorCode.PATCH_APPLY_FAILED,
                                      check.stderr.decode("utf-8", errors="replace").strip(), retryable=True)
        # Recheck after the subprocess: an editor may have saved during preflight.
        self._patch_preconditions(patch)
        conflict = self._check_patch_versions(expected)
        if conflict is not None:
            return conflict
        if args.get("_expected_workspace") and self.workspace_version() != args["_expected_workspace"]:
            return ToolResult.failure("apply_patch", ErrorCode.PATCH_CONFLICT, "Workspace changed during candidate switch")
        # Feed bytes so Windows does not translate LF in the unified diff to CRLF.
        raw_result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.repo_root,
            input=patch.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if raw_result.returncode != 0:
            return ToolResult.failure(
                "apply_patch",
                ErrorCode.PATCH_APPLY_FAILED,
                raw_result.stderr.decode("utf-8", errors="replace").strip()
                or "git apply failed",
                retryable=True,
            )
        # Do not treat generated output as a new model observation.
        for relative in expected:
            self._read_versions.pop(relative, None)
        # Intent-to-add only files created by this patch. Tests may have produced
        # unrelated untracked files which must never enter the evaluator patch.
        new_paths: list[str] = []
        for relative in patch_paths:
            path = self._safe_path(relative, must_exist=False)
            tracked = self._git("ls-files", "--error-unmatch", "--", relative)
            if path.exists() and tracked.returncode != 0:
                new_paths.append(relative)
        if new_paths:
            intent_result = self._git("add", "--intent-to-add", "--", *new_paths)
            if intent_result.returncode != 0:
                return ToolResult.failure(
                    "apply_patch",
                    ErrorCode.ENVIRONMENT_ERROR,
                    intent_result.stderr.strip() or "git add --intent-to-add failed",
                    retryable=True,
                    data={"patch_applied": True, "recovery_action": "inspect_diff", "changed_files": list(patch_paths)},
                    state_changed=True,
                )
        changed = [
            item
            for item in self._git("diff", "--name-only", "-z").stdout.split("\x00")
            if item
        ]
        for relative in changed:
            self._safe_path(relative, must_exist=False)
        after_hash = self.diff_hash()
        if self.recovery.evidence is not None:
            self.recovery.evidence["validity"] = "stale"
        self._capture_candidate()
        return ToolResult.success(
            "apply_patch",
            {"changed_files": changed, "diff_hash": after_hash, "checked_versions": expected},
            state_changed=after_hash != before_hash,
        )

    @staticmethod
    def _validate_process_arg(value: str) -> None:
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("Test arguments must be non-empty single-line strings")

    @staticmethod
    def _parse_failure_count(output: str, returncode: int) -> int | None:
        if returncode == 0:
            return 0
        pytest_match = re.search(r"(?:^|\s)(\d+) failed(?:,|\s|$)", output)
        if pytest_match:
            return int(pytest_match.group(1))
        unittest_match = re.search(r"FAILED \(([^)]*)\)", output)
        if unittest_match:
            total = 0
            for key in ("failures", "errors"):
                match = re.search(rf"{key}=(\d+)", unittest_match.group(1))
                if match:
                    total += int(match.group(1))
            return max(total, 1)
        return 1

    def _run_tests(self, args: dict[str, Any]) -> ToolResult:
        self._invalidate_external_candidates()
        target = args.get("target")
        extra_args = list(args.get("extra_args", []))
        if target is not None:
            self._validate_process_arg(target)
        for item in extra_args:
            self._validate_process_arg(item)

        command = self.profile.resolved_test_command()
        command.extend([target] if target else self.profile.default_test_args)
        command.extend(extra_args)
        before = self.workspace_version()
        protocol = self.validation_protocol()
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="capo_pycache_") as pycache:
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=self.profile.timeout_seconds,
                    check=False,
                    env={
                        **os.environ,
                        "PYTHONUTF8": "1",
                        "PYTHONPYCACHEPREFIX": pycache,
                    },
                )
        except subprocess.TimeoutExpired as exc:
            state = TestState(TestStatus.TIMEOUT, None, None)
            changed = state != self.last_test_state
            self.last_test_state = state
            self.recovery.validated(before=before, after=self.workspace_version(), protocol=protocol,
                                    command=command, state=state.to_dict(), stable=False, files=None)
            return ToolResult.failure(
                "run_tests",
                ErrorCode.TOOL_TIMEOUT,
                f"Tests exceeded {self.profile.timeout_seconds} seconds",
                retryable=True,
                data={
                    "command": command,
                    "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "",
                    "test_state": state.to_dict(),
                    "state_changed": changed,
                },
            )

        output = f"{result.stdout}\n{result.stderr}".strip()
        failure_count = self._parse_failure_count(output, result.returncode)
        status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED
        state = TestState(status, failure_count, 0)
        changed = state != self.last_test_state
        self.last_test_state = state
        after = self.workspace_version()
        stable = protocol == self.validation_protocol() and before == after
        files = self._candidate_files() if stable and self.recovery.candidates_enabled else None
        stable = stable and self.workspace_version() == after
        evidence = self.recovery.validated(before=before, after=after, protocol=protocol,
                                            command=command, state=state.to_dict(), stable=stable, files=files)
        if not stable:
            return ToolResult.failure("run_tests", ErrorCode.ENVIRONMENT_ERROR,
                                      "Workspace or environment changed during validation; result is not current evidence.",
                                      data={"command": command, "output": output[-20000:], "evidence": evidence})
        return ToolResult.success(
            "run_tests",
            {
                "command": command,
                "exit_code": result.returncode,
                "output": output[-20000:],
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "test_state": state.to_dict(),
                **({"evidence": evidence, "regression": self.recovery.regression}
                   if self.evidence_guidance else {}),
            },
            state_changed=changed,
        )

    def _inspect_diff(self) -> ToolResult:
        diff = self.diff_text()
        stat = self._git("diff", "--stat").stdout
        return ToolResult.success(
            "inspect_diff",
            {"stat": stat, "diff": diff[-50000:], "diff_hash": sha256_text(diff),
             "workspace_version": self.workspace_version(),
             "recovery": self.recovery.summary(self.workspace_version(), self.validation_protocol(),
                                                guidance=self.evidence_guidance,
                                                expose_candidates=self.candidate_switching)},
        )

    def evaluate(self) -> ToolResult:
        return self._run_tests({})
