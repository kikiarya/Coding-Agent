from __future__ import annotations

import tempfile
from pathlib import Path

from coding_agent_policy_opt.config import ExperimentConfig, load_experiment_config
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment, RepositoryProfile
from coding_agent_policy_opt.models import TaskSpec
from coding_agent_policy_opt.harness.smoke import create_smoke_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "smoke.yaml"


class SmokeRepository:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="capo_test_")
        self.root = Path(self._temporary.name) / "repo"
        self.task: TaskSpec = create_smoke_repository(self.root)
        self.config: ExperimentConfig = load_experiment_config(SMOKE_CONFIG)
        self.profile = RepositoryProfile(
            test_command=self.config.repository.test_command,
            default_test_args=self.config.repository.default_test_args,
            timeout_seconds=30,
        )
        self.environment = RepositoryEnvironment(self.root, self.profile)

    def close(self) -> None:
        self._temporary.cleanup()

