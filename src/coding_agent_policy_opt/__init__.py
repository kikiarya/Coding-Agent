"""Repository-level coding agent policy optimization harness."""

from .config import ExperimentConfig, load_experiment_config
from .evaluation.exploration import analyze_exploration
from .evaluation.two_layer import compare_two_layer_arms, validate_two_layer_configs
from .harness.checkpoint import replay_latest_checkpoint
from .harness.runner import AgentHarness
from .models import EpisodeRecord, StepRecord
from .training.rft import build_rft_dataset, select_rft_rollouts

__all__ = [
    "AgentHarness",
    "analyze_exploration",
    "build_rft_dataset",
    "EpisodeRecord",
    "ExperimentConfig",
    "StepRecord",
    "compare_two_layer_arms",
    "load_experiment_config",
    "replay_latest_checkpoint",
    "select_rft_rollouts",
    "validate_two_layer_configs",
]

__version__ = "0.1.0"
