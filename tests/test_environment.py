from __future__ import annotations

import unittest

from coding_agent_policy_opt.models import ActionType, AgentAction, ErrorCode, TestStatus
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


class RepositoryEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SmokeRepository()
        self.environment = self.fixture.environment

    def tearDown(self) -> None:
        self.fixture.close()

    def test_search_and_read_are_repo_scoped(self) -> None:
        search = self.environment.execute(
            AgentAction(
                ActionType.SEARCH_CODE,
                {"query": "def add", "path": ".", "glob": "*.py"},
            )
        )
        self.assertTrue(search.ok)
        self.assertEqual(search.data["matches"][0]["path"], "calculator.py")

        read = self.environment.execute(
            AgentAction(
                ActionType.READ_FILE,
                {"path": "calculator.py", "start_line": 1, "end_line": 20},
            )
        )
        self.assertTrue(read.ok)
        self.assertIn("    return a - b", [line["text"] for line in read.data["lines"]])

    def test_path_traversal_is_rejected(self) -> None:
        result = self.environment.execute(
            AgentAction(
                ActionType.READ_FILE,
                {"path": "../outside.txt", "start_line": 1, "end_line": 1},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.PATH_OUTSIDE_REPOSITORY)

    def test_git_metadata_is_not_exposed(self) -> None:
        result = self.environment.execute(
            AgentAction(
                ActionType.READ_FILE,
                {"path": ".git/config", "start_line": 1, "end_line": 20},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.PATH_OUTSIDE_REPOSITORY)

    def test_patch_then_targeted_test_resolves_bug(self) -> None:
        self.environment.execute(AgentAction.from_dict(smoke_decisions()[1]["action"]))
        patch_action = AgentAction.from_dict(smoke_decisions()[2]["action"])
        patch_result = self.environment.execute(patch_action)
        self.assertTrue(patch_result.ok)
        self.assertTrue(patch_result.state_changed)
        self.assertEqual(patch_result.data["changed_files"], ["calculator.py"])

        test_result = self.environment.execute(
            AgentAction(
                ActionType.RUN_TESTS,
                {"target": "tests.test_calculator", "extra_args": ["-v"]},
            )
        )
        self.assertTrue(test_result.ok)
        self.assertEqual(test_result.data["test_state"]["status"], TestStatus.PASSED.value)
        self.assertEqual(test_result.data["test_state"]["fail_to_pass_count"], 0)

    def test_non_unified_patch_is_rejected_without_change(self) -> None:
        before = self.environment.diff_hash()
        result = self.environment.execute(
            AgentAction(ActionType.APPLY_PATCH, {"patch": "calculator.py: return a + b"})
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENT)
        self.assertEqual(self.environment.diff_hash(), before)

    def test_new_file_is_visible_in_diff_and_state_hash(self) -> None:
        (self.fixture.root / "unrelated-generated.log").write_text(
            "not part of the model patch\n", encoding="utf-8"
        )
        before = self.environment.diff_hash()
        patch = (
            "diff --git a/new_module.py b/new_module.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new_module.py\n"
            "@@ -0,0 +1 @@\n"
            "+VALUE = 42\n"
        )

        result = self.environment.execute(
            AgentAction(ActionType.APPLY_PATCH, {"patch": patch})
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.state_changed)
        self.assertIn("new_module.py", result.data["changed_files"])
        self.assertNotEqual(self.environment.diff_hash(), before)
        self.assertIn("new file mode", self.environment.diff_text())
        self.assertNotIn("unrelated-generated.log", self.environment.diff_text())

    def test_symlink_patch_is_rejected_before_worktree_change(self) -> None:
        before = self.environment.diff_hash()
        patch = (
            "diff --git a/link b/link\n"
            "new file mode 120000\n"
            "--- /dev/null\n"
            "+++ b/link\n"
            "@@ -0,0 +1 @@\n"
            "+../outside\n"
        )

        result = self.environment.execute(
            AgentAction(ActionType.APPLY_PATCH, {"patch": patch})
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENT)
        self.assertEqual(self.environment.diff_hash(), before)


if __name__ == "__main__":
    unittest.main()
