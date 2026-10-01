from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import PurePosixPath
from typing import Any


PROTOCOL_VERSION = "1.0.0"
MAX_OUTPUT_CHARS = 20_000


def _safe_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Path must be a non-empty string")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise PermissionError("Path is outside the task repository")
    if any(part.lower() == ".git" for part in path.parts):
        raise PermissionError("Git metadata is not part of the task repository interface")
    return path.as_posix()


def _patch_paths(patch: str) -> tuple[str, ...]:
    if not isinstance(patch, str) or not patch.startswith("diff --git "):
        raise ValueError("Patch must be a git-style unified diff")
    paths: set[str] = set()
    headers = 0
    for line in patch.splitlines():
        if line in {
            "new file mode 120000",
            "new mode 120000",
            "new file mode 160000",
            "new mode 160000",
        }:
            raise ValueError("Symlink and submodule patches are not allowed")
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        headers += 1
        value = line[4:].split("\t", 1)[0].strip().strip('"')
        if value == "/dev/null":
            continue
        if value.startswith(("a/", "b/")):
            value = value[2:]
        paths.add(_safe_relpath(value))
    if headers < 2:
        raise ValueError("Patch is missing unified diff file headers")
    return tuple(sorted(paths))


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _exec(container: Any, command: str | list[str], *, workdir: str | None = None) -> tuple[int, str]:
    from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR

    result = container.exec_run(
        command,
        workdir=workdir or DOCKER_WORKDIR,
        user=DOCKER_USER,
    )
    return int(result.exit_code), _decode(result.output)


def _put_file(container: Any, destination_dir: str, name: str, content: bytes) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(content))
    stream.seek(0)
    if not container.put_archive(destination_dir, stream.read()):
        raise RuntimeError(f"Failed to copy {name} into the container")


def _git_diff(container: Any) -> str:
    code, output = _exec(container, ["git", "diff", "--binary", "--no-ext-diff"])
    if code != 0:
        raise RuntimeError(output.strip() or "git diff failed")
    return output


def _diff_hash(diff: str) -> str:
    return "sha256:" + hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _apply_patch(container: Any, patch: str) -> tuple[bool, str, list[str]]:
    patch_paths = _patch_paths(patch)
    before = _git_diff(container)
    filename = f"capo_patch_{os.getpid()}.diff"
    destination = f"/tmp/{filename}"
    _put_file(container, "/tmp", filename, patch.encode("utf-8"))
    try:
        code, output = _exec(
            container,
            ["git", "apply", "--whitespace=nowarn", destination],
        )
    finally:
        _exec(container, ["rm", "-f", destination], workdir="/")
    if code != 0:
        return False, output.strip() or "git apply failed", []
    new_paths: list[str] = []
    for path in patch_paths:
        exists_code, _ = _exec(container, ["test", "-e", path])
        tracked_code, _ = _exec(
            container, ["git", "ls-files", "--error-unmatch", "--", path]
        )
        if exists_code == 0 and tracked_code != 0:
            new_paths.append(path)
    if new_paths:
        code, output = _exec(
            container, ["git", "add", "--intent-to-add", "--", *new_paths]
        )
        if code != 0:
            _put_file(container, "/tmp", filename, patch.encode("utf-8"))
            try:
                _exec(
                    container,
                    ["git", "apply", "--reverse", "--whitespace=nowarn", destination],
                )
            finally:
                _exec(container, ["rm", "-f", destination], workdir="/")
            raise RuntimeError(output.strip() or "git add --intent-to-add failed")
    code, output = _exec(container, ["git", "diff", "--name-only", "-z"])
    if code != 0:
        raise RuntimeError(output.strip() or "git diff --name-only failed")
    changed = [item for item in output.split("\x00") if item]
    after = _git_diff(container)
    return before != after, "", changed


def _parse_search_output(output: str, repo_root: str, max_results: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    pattern = re.compile(r"^(.*?):(\d+):(.*)$")
    prefix = repo_root.rstrip("/") + "/"
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        path = match.group(1)
        if path.startswith(prefix):
            path = path[len(prefix) :]
        matches.append(
            {
                "path": path,
                "line": int(match.group(2)),
                "text": match.group(3)[:1000],
            }
        )
        if len(matches) >= max_results:
            break
    return matches


def _search(container: Any, args: dict[str, Any]) -> dict[str, Any]:
    from swebench.harness.constants import DOCKER_WORKDIR

    query = args.get("query")
    if not isinstance(query, str) or not query or "\x00" in query:
        raise ValueError("query must be a non-empty string")
    path = _safe_relpath(args.get("path"))
    glob = args.get("glob", "*")
    if not isinstance(glob, str) or not glob or "\x00" in glob or "/" in glob:
        raise ValueError("glob must be a non-empty filename pattern")
    max_results = int(args.get("max_results", 50))
    if max_results < 1 or max_results > 200:
        raise ValueError("max_results must be between 1 and 200")
    root = f"{DOCKER_WORKDIR}/{path}" if path != "." else DOCKER_WORKDIR
    script = (
        f"grep -R -n -I -F --include={shlex.quote(glob)} --exclude-dir=.git -- "
        f"{shlex.quote(query)} {shlex.quote(root)} 2>/dev/null | head -n {max_results}"
    )
    code, output = _exec(container, ["/bin/bash", "-lc", script], workdir="/")
    if code not in {0, 1}:
        raise RuntimeError(output.strip() or "search_code failed")
    matches = _parse_search_output(output, DOCKER_WORKDIR, max_results)
    return {"query": query, "matches": matches, "truncated": len(matches) >= max_results}


def _read_file(container: Any, args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_relpath(args.get("path"))
    start_line = int(args.get("start_line"))
    end_line = int(args.get("end_line"))
    if start_line < 1 or end_line < start_line or end_line - start_line + 1 > 400:
        raise ValueError("Invalid line range; read_file is limited to 400 lines")
    code, total_output = _exec(container, ["awk", "END {print NR}", path])
    if code != 0:
        raise ValueError(total_output.strip() or f"File does not exist: {path}")
    total_lines = int(total_output.strip().split()[0])
    code, output = _exec(container, ["sed", "-n", f"{start_line},{end_line}p", "--", path])
    if code != 0:
        raise RuntimeError(output.strip() or "read_file failed")
    lines = [
        {"line": number, "text": text}
        for number, text in enumerate(output.splitlines(), start=start_line)
    ]
    return {
        "path": path,
        "start_line": start_line,
        "end_line": min(end_line, total_lines),
        "total_lines": total_lines,
        "lines": lines,
    }


def _test_result(container: Any, instance: dict[str, Any], args: dict[str, Any], timeout: int) -> dict[str, Any]:
    from swebench.harness.constants import DOCKER_WORKDIR, FAIL_TO_PASS, PASS_TO_PASS
    from swebench.harness.docker_utils import exec_run_with_timeout
    from swesmith.harness.grading import get_eval_tests_report
    from swesmith.profiles import registry

    profile = registry.get_from_inst(instance)
    command, _ = profile.get_test_cmd(instance, f2p_only=False)
    target = args.get("target")
    extra_args = args.get("extra_args", [])
    if target is not None:
        if not isinstance(target, str) or not target or any(ch in target for ch in "\x00\r\n"):
            raise ValueError("target must be a non-empty single-line string")
        command += " " + shlex.quote(target)
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) and item and not any(ch in item for ch in "\x00\r\n")
        for item in extra_args
    ):
        raise ValueError("extra_args must contain non-empty single-line strings")
    if extra_args:
        command += " " + " ".join(shlex.quote(item) for item in extra_args)

    script_name = f"capo_eval_{os.getpid()}.sh"
    script = f"#!/bin/bash\nset -uo pipefail\ncd {shlex.quote(DOCKER_WORKDIR)}\n{command}\n"
    _put_file(container, "/tmp", script_name, script.encode("utf-8"))
    started = time.perf_counter()
    try:
        output, timed_out, runtime = exec_run_with_timeout(
            container,
            f"/bin/bash /tmp/{script_name}",
            timeout=timeout,
        )
    finally:
        _exec(container, ["rm", "-f", f"/tmp/{script_name}"], workdir="/")
    if timed_out:
        return {
            "ok": False,
            "tool": "run_tests",
            "error_code": "TOOL_TIMEOUT",
            "message": f"Tests exceeded {timeout} seconds",
            "retryable": True,
            "state_changed": False,
            "data": {
                "command": command,
                "output": output[-MAX_OUTPUT_CHARS:],
                "duration_ms": round(runtime * 1000),
                "test_state": {
                    "status": "timeout",
                    "fail_to_pass_count": None,
                    "pass_to_pass_failures": None,
                },
            },
        }
    status_map = profile.log_parser(output)
    report = get_eval_tests_report(status_map, instance)
    f2p_failures = len(report[FAIL_TO_PASS]["failure"])
    p2p_failures = len(report[PASS_TO_PASS]["failure"])
    return {
        "ok": True,
        "tool": "run_tests",
        "state_changed": False,
        "data": {
            "command": command,
            "output": output[-MAX_OUTPUT_CHARS:],
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "test_state": {
                "status": "passed" if f2p_failures == 0 and p2p_failures == 0 else "failed",
                "fail_to_pass_count": f2p_failures,
                "pass_to_pass_failures": p2p_failures,
            },
            "official_test_report": report,
        },
    }


def _tool_failure(tool: str, code: str, message: str, *, retryable: bool = True) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "error_code": code,
        "message": message,
        "retryable": retryable,
        "state_changed": False,
        "data": {},
    }


def _execute(container: Any, instance: dict[str, Any], action: dict[str, Any], timeout: int) -> dict[str, Any]:
    action_type = action.get("type")
    args = action.get("args")
    if not isinstance(args, dict):
        return _tool_failure(str(action_type), "INVALID_ARGUMENT", "args must be an object")
    try:
        if action_type == "search_code":
            result = {"ok": True, "tool": action_type, "data": _search(container, args), "state_changed": False}
        elif action_type == "read_file":
            result = {"ok": True, "tool": action_type, "data": _read_file(container, args), "state_changed": False}
        elif action_type == "apply_patch":
            changed, error, changed_files = _apply_patch(container, args.get("patch"))
            if error:
                return _tool_failure(action_type, "PATCH_APPLY_FAILED", error)
            diff = _git_diff(container)
            result = {
                "ok": True,
                "tool": action_type,
                "data": {"changed_files": changed_files, "diff_hash": _diff_hash(diff)},
                "state_changed": changed,
            }
        elif action_type == "run_tests":
            result = _test_result(container, instance, args, timeout)
        elif action_type == "inspect_diff":
            diff = _git_diff(container)
            code, stat = _exec(container, ["git", "diff", "--stat"])
            if code != 0:
                raise RuntimeError(stat.strip() or "git diff --stat failed")
            result = {
                "ok": True,
                "tool": action_type,
                "data": {"stat": stat, "diff": diff[-50_000:], "diff_hash": _diff_hash(diff)},
                "state_changed": False,
            }
        else:
            result = _tool_failure(str(action_type), "UNKNOWN_TOOL", "Unregistered tool")
    except PermissionError as exc:
        result = _tool_failure(str(action_type), "PATH_OUTSIDE_REPOSITORY", str(exc))
    except (TypeError, ValueError) as exc:
        result = _tool_failure(str(action_type), "INVALID_ARGUMENT", str(exc))
    except Exception as exc:
        result = _tool_failure(str(action_type), "ENVIRONMENT_ERROR", str(exc))
    return result


def _checkout_bug_stage(container: Any) -> str:
    code, output = _exec(container, ["git", "checkout", "HEAD~1"])
    if code != 0:
        raise RuntimeError(output.strip() or "Failed to restore the SWE-smith bug stage")
    code, output = _exec(container, ["git", "rev-parse", "HEAD"])
    if code != 0 or not output.strip():
        raise RuntimeError(output.strip() or "Unable to resolve the materialized base commit")
    return output.strip()


def _container_metadata(container: Any) -> dict[str, Any]:
    container.reload()
    image = container.image
    image.reload()
    attrs = image.attrs
    digests = attrs.get("RepoDigests") or []
    return {
        "container_id": container.id,
        "image_id": image.id,
        "image_digests": sorted(str(item) for item in digests),
        "image_tags": sorted(str(item) for item in image.tags),
    }


def _materialize(instance: dict[str, Any]) -> dict[str, Any]:
    from swesmith.profiles import registry

    profile = registry.get_from_inst(instance)
    container = profile.get_container(instance)
    try:
        base_commit = _checkout_bug_stage(container)
        metadata = _container_metadata(container)
        diff = _git_diff(container)
        test_command, _ = profile.get_test_cmd(instance, f2p_only=False)
        return {
            **metadata,
            "base_commit": base_commit,
            "test_command": test_command,
            "profile_timeout_seconds": int(profile.timeout),
            "diff_hash": _diff_hash(diff),
        }
    except Exception:
        container.remove(force=True)
        raise


def _evaluate(container: Any, instance: dict[str, Any], expected_image_id: str, timeout: int) -> dict[str, Any]:
    from swesmith.profiles import registry

    active_diff = _git_diff(container)
    profile = registry.get_from_inst(instance)
    evaluator = profile.get_container(instance)
    try:
        _checkout_bug_stage(evaluator)
        metadata = _container_metadata(evaluator)
        if expected_image_id and metadata["image_id"] != expected_image_id:
            raise RuntimeError("Evaluator image ID differs from the materialized agent image")
        if active_diff:
            changed, error, _ = _apply_patch(evaluator, active_diff)
            if error or not changed:
                raise RuntimeError(error or "Evaluator patch did not change the clean container")
        f2p_files, p2p_files = profile.get_test_files(instance)
        test_files = sorted({_safe_relpath(str(path)) for path in [*f2p_files, *p2p_files]})
        if test_files:
            code, output = _exec(evaluator, ["git", "checkout", "--", *test_files])
            if code != 0:
                raise RuntimeError(output.strip() or "Failed to restore evaluator test files")
        result = _test_result(evaluator, instance, {}, timeout)
        result["tool"] = "evaluator"
        result["data"]["agent_diff_hash"] = _diff_hash(active_diff)
        result["data"]["restored_test_files"] = test_files
        return result
    finally:
        evaluator.remove(force=True)


def _installed_commit() -> tuple[str | None, str | None]:
    direct_url: dict[str, Any] = {}
    try:
        distribution = importlib.metadata.distribution("SWE-smith")
        raw = distribution.read_text("direct_url.json")
        direct_url = json.loads(raw) if raw else {}
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    vcs_commit = direct_url.get("vcs_info", {}).get("commit_id")
    import swesmith

    module_path = os.path.realpath(swesmith.__file__)
    current = os.path.dirname(module_path)
    while current and current != os.path.dirname(current):
        if os.path.isdir(os.path.join(current, ".git")):
            result = subprocess.run(
                ["git", "-C", current, "rev-parse", "HEAD"],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip(), module_path
        current = os.path.dirname(current)
    return str(vcs_commit) if vcs_commit else None, module_path


def _dependency_preflight(expected_revision: str) -> dict[str, Any]:
    import docker
    import swesmith

    installed_commit, module_path = _installed_commit()
    client = docker.from_env()
    ping = bool(client.ping())
    checks = {
        "docker_sdk": True,
        "docker_engine": ping,
        "swesmith_import": bool(swesmith),
        "repository_revision": installed_commit == expected_revision,
    }
    return {
        "schema_version": "2.0.0",
        "checks": checks,
        "details": {
            "swesmith_module": module_path,
            "installed_repository_revision": installed_commit,
            "expected_repository_revision": expected_revision,
            "docker_server_version": client.version().get("Version"),
        },
        "errors": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Worker protocol version mismatch")
    op = request.get("op")
    if op == "dependency_preflight":
        return {
            "data": _dependency_preflight(
                str(request.get("expected_repository_revision", ""))
            )
        }
    if op == "materialize":
        return {"data": _materialize(dict(request["instance"]))}

    import docker

    container_id = request.get("container_id")
    if not isinstance(container_id, str) or not container_id:
        raise ValueError("container_id is required")
    client = docker.from_env()
    container = client.containers.get(container_id)
    if op == "close":
        container.remove(force=True)
        return {"data": {"removed_container_id": container_id}}

    instance = dict(request["instance"])
    timeout = int(request.get("timeout_seconds", 900))
    if timeout < 1 or timeout > 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600")
    if op == "execute":
        result = _execute(container, instance, dict(request["action"]), timeout)
    elif op == "evaluate":
        result = _evaluate(
            container,
            instance,
            str(request.get("expected_image_id", "")),
            timeout,
        )
    else:
        raise ValueError(f"Unknown worker operation: {op}")
    diff = _git_diff(container)
    test_state = result.get("data", {}).get("test_state")
    return {
        "tool_result": result,
        "diff_hash": _diff_hash(diff),
        "test_state": test_state,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("Worker request must be an object")
        with contextlib.redirect_stdout(sys.stderr):
            data = _dispatch(request)
        response = {"protocol_version": PROTOCOL_VERSION, "ok": True, **data}
    except Exception as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8)[-8000:],
            },
        }
    sys.stdout.write(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
