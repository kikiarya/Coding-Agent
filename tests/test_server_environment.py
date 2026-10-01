from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess

from coding_agent_policy_opt.cli import build_parser, preflight_environment
from coding_agent_policy_opt.swesmith.config import (
    PINNED_SWESMITH_DATASET_REVISION,
    PINNED_SWESMITH_REPOSITORY_REVISION,
    load_swesmith_environment_config,
    preflight_linux_server_environment,
    preflight_wsl_environment,
    resolve_server_runtime_root,
)

from tests.helpers import PROJECT_ROOT


class ServerEnvironmentConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            PROJECT_ROOT / "configs" / "environment" / "swesmith_server.yaml"
        )
        self.config = load_swesmith_environment_config(self.path)

    def test_server_profile_freezes_sources_hardware_and_storage_gate(self) -> None:
        self.assertEqual(self.config.profile, "swesmith_linux_server_v1")
        self.assertEqual(self.config.platform, "linux_server")
        self.assertEqual(
            self.config.repository_revision, PINNED_SWESMITH_REPOSITORY_REVISION
        )
        self.assertEqual(
            self.config.dataset_revision, PINNED_SWESMITH_DATASET_REVISION
        )
        self.assertEqual(self.config.runtime_root_env, "CAPO_RUNTIME_ROOT")
        self.assertEqual(self.config.minimum_free_gib, 200)
        self.assertEqual(self.config.minimum_gpu_memory_gib, 24)
        self.assertTrue(self.config.require_native_linux)
        self.assertTrue(self.config.require_cuda)
        self.assertIsNone(self.config.docker_data_root)
        self.assertIsNone(self.config.dependencies_root)

    def test_server_runtime_root_rejects_windows_and_d_drive_paths(self) -> None:
        invalid = [r"D:\capo", r"C:\capo", "relative/capo"]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_server_runtime_root(
                        self.config, environ={"CAPO_RUNTIME_ROOT": value}
                    )

    @patch("coding_agent_policy_opt.swesmith.config.subprocess.run")
    @patch("coding_agent_policy_opt.swesmith.config.shutil.disk_usage")
    @patch("coding_agent_policy_opt.swesmith.config.platform.system", return_value="Windows")
    def test_server_preflight_on_windows_stops_before_any_probe(
        self, _system: object, disk_usage: object, run: object
    ) -> None:
        report = preflight_linux_server_environment(self.config)

        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"], {"native_linux": False})
        disk_usage.assert_not_called()
        run.assert_not_called()

    @patch("coding_agent_policy_opt.swesmith.config.Path.read_text", return_value="Linux")
    @patch("coding_agent_policy_opt.swesmith.config.platform.system", return_value="Linux")
    def test_missing_server_commands_are_reported_not_raised(
        self, _system: object, _read_text: object
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="capo_server_") as temporary:
            with patch(
                "coding_agent_policy_opt.swesmith.config.resolve_server_runtime_root",
                return_value=Path(temporary),
            ), patch(
                "coding_agent_policy_opt.swesmith.config.shutil.disk_usage",
                return_value=SimpleNamespace(free=300 * 1024**3),
            ), patch(
                "coding_agent_policy_opt.swesmith.config.subprocess.run",
                side_effect=FileNotFoundError("missing command"),
            ):
                report = preflight_linux_server_environment(self.config)

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["docker_engine"])
        self.assertFalse(report["checks"]["cuda_gpu"])

    @patch("coding_agent_policy_opt.swesmith.config.Path.read_text", return_value="Linux")
    @patch("coding_agent_policy_opt.swesmith.config.platform.system", return_value="Linux")
    def test_marketed_24gb_gpu_passes_with_driver_reserved_memory(
        self, _system: object, _read_text: object
    ) -> None:
        def command_result(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if command[0] == "python3":
                output = "3.10\n"
            elif command[0] == "git":
                output = "git version 2.43\n"
            elif command[0] == "docker":
                output = "linux|x86_64|27.0\n"
            else:
                output = "24564\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        with tempfile.TemporaryDirectory(prefix="capo_server_") as temporary:
            with patch(
                "coding_agent_policy_opt.swesmith.config.resolve_server_runtime_root",
                return_value=Path(temporary),
            ), patch(
                "coding_agent_policy_opt.swesmith.config.shutil.disk_usage",
                return_value=SimpleNamespace(free=300 * 1024**3),
            ), patch(
                "coding_agent_policy_opt.swesmith.config.subprocess.run",
                side_effect=command_result,
            ):
                report = preflight_linux_server_environment(self.config)

        self.assertTrue(report["ready"])
        self.assertTrue(report["checks"]["gpu_memory"])

    def test_formal_cli_defaults_to_server_config(self) -> None:
        parser = build_parser()
        fetch = parser.parse_args(["swesmith", "fetch-source", "--output", "x"])
        smoke = parser.parse_args(
            [
                "rl",
                "smoke-train",
                "--inventory",
                "i",
                "--oracle",
                "o",
                "--manifest",
                "m",
                "--checkpoint",
                "c",
                "--sft-gate-report",
                "s",
                "--template-preflight-report",
                "t",
                "--output-dir",
                "out",
            ]
        )

        expected = Path("configs/environment/swesmith_server.yaml")
        self.assertEqual(fetch.config, expected)
        self.assertEqual(smoke.environment_config, expected)

    def test_external_evaluation_split_is_reachable_from_cli(self) -> None:
        parser = build_parser()
        evaluation = parser.parse_args(
            [
                "swesmith",
                "evaluate-model",
                "--inventory",
                "inventory.jsonl",
                "--oracle",
                "oracle.jsonl",
                "--manifest",
                "external_50.jsonl",
                "--report-output",
                "external_report.json",
                "--experiment-config",
                "evaluation.yaml",
                "--split",
                "external",
                "--run-label",
                "joint_external",
            ]
        )

        self.assertEqual(evaluation.split, "external")


class LocalDriveProtectionTest(unittest.TestCase):
    @patch("coding_agent_policy_opt.swesmith.config.subprocess.run")
    @patch("coding_agent_policy_opt.swesmith.config.shutil.disk_usage")
    def test_legacy_library_preflight_refuses_without_system_probe(
        self, disk_usage: object, run: object
    ) -> None:
        config = load_swesmith_environment_config(
            PROJECT_ROOT / "configs" / "environment" / "swesmith.yaml"
        )
        report = preflight_wsl_environment(config)

        self.assertEqual(report["status"], "refused")
        self.assertFalse(report["ready"])
        disk_usage.assert_not_called()
        run.assert_not_called()

    @patch(
        "coding_agent_policy_opt.cli.load_swesmith_environment_config",
        side_effect=AssertionError("legacy config must not be loaded"),
    )
    def test_legacy_preflight_refuses_without_loading_or_probing(self, _load: object) -> None:
        with tempfile.TemporaryDirectory(prefix="capo_protection_") as temporary:
            output = Path(temporary) / "report.json"
            exit_code = preflight_environment(Path("legacy.yaml"), output)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "refused")
        self.assertIn("D drive", report["reason"])
        _load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
