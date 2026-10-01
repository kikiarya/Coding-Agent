from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..models import TaskSpec


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def create_smoke_repository(repo_root: Path) -> TaskSpec:
    repo_root = repo_root.resolve()
    repo_root.mkdir(parents=True, exist_ok=False)
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()

    (repo_root / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Return the sum of two integers.\"\"\"\n"
        "    return a - b\n",
        encoding="utf-8",
        newline="\n",
    )
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import add\n\n\n"
        "class CalculatorTest(unittest.TestCase):\n"
        "    def test_adds_positive_numbers(self) -> None:\n"
        "        self.assertEqual(add(2, 3), 5)\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    unittest.main()\n",
        encoding="utf-8",
        newline="\n",
    )

    _run_git(repo_root, "init", "--initial-branch=main")
    _run_git(repo_root, "config", "user.name", "CAPO Smoke")
    _run_git(repo_root, "config", "user.email", "smoke@example.invalid")
    _run_git(repo_root, "config", "core.autocrlf", "false")
    _run_git(repo_root, "add", "calculator.py", "tests")
    _run_git(repo_root, "commit", "-m", "Create failing calculator task")
    base_commit = _run_git(repo_root, "rev-parse", "HEAD")

    return TaskSpec(
        task_id="smoke_calculator_add",
        split="smoke",
        repo_id="local/smoke-calculator",
        issue=(
            "The add function in calculator.py returns the wrong result. "
            "Fix the implementation and verify the calculator tests."
        ),
        base_commit=base_commit,
    )


def smoke_decisions() -> list[dict[str, Any]]:
    patch = (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "     \"\"\"Return the sum of two integers.\"\"\"\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )
    return [
        {
            "intent": "Locate the add implementation before editing.",
            "action": {
                "type": "search_code",
                "args": {"query": "def add", "path": ".", "glob": "*.py"},
            },
        },
        {
            "intent": "Read the implementation and surrounding context.",
            "action": {
                "type": "read_file",
                "args": {"path": "calculator.py", "start_line": 1, "end_line": 20},
            },
        },
        {
            "intent": "Replace subtraction with addition using a minimal patch.",
            "action": {"type": "apply_patch", "args": {"patch": patch}},
        },
        {
            "intent": "Run the targeted calculator test after changing the code.",
            "action": {
                "type": "run_tests",
                "args": {"target": "tests.test_calculator", "extra_args": ["-v"]},
            },
        },
        {
            "intent": "Inspect the final diff before finishing.",
            "action": {"type": "inspect_diff", "args": {}},
        },
        {
            "intent": "The targeted test passes and the diff is minimal.",
            "action": {
                "type": "finish",
                "args": {"status": "success", "summary": "Fixed add and verified tests."},
            },
        },
    ]

