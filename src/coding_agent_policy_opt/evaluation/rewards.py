from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _bounded(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class RewardWeights:
    """Frozen scalarization used by the online Model Policy objective.

    Keeping the weights as a typed value makes the YAML experiment contract
    executable instead of leaving a second, potentially drifting copy of the
    objective embedded in the reward function.
    """

    success: float = 1.0
    progress: float = 0.2
    verification: float = 0.05
    duplicate: float = -0.05
    invalid: float = -0.05
    normalized_steps: float = -0.05
    recovery_ablation: float = 0.05

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "RewardWeights":
        expected = {
            "success",
            "progress",
            "verification",
            "duplicate",
            "normalized_steps",
            "recovery_ablation",
            "recovery_ablation_enabled",
        }
        if set(value) != expected:
            raise ValueError(
                "RL reward config must contain exactly the frozen reward fields"
            )
        if value["recovery_ablation_enabled"] is not False:
            raise ValueError(
                "Recovery reward remains an explicit ablation and must be disabled"
            )
        weights = cls(
            success=float(value["success"]),
            progress=float(value["progress"]),
            verification=float(value["verification"]),
            duplicate=float(value["duplicate"]),
            normalized_steps=float(value["normalized_steps"]),
            recovery_ablation=float(value["recovery_ablation"]),
        )
        if weights.to_config()["recovery_ablation_enabled"] is not False:
            raise AssertionError("unreachable")
        return weights

    def to_config(self) -> dict[str, float | bool]:
        return {
            "success": self.success,
            "progress": self.progress,
            "verification": self.verification,
            "duplicate": self.duplicate,
            "normalized_steps": self.normalized_steps,
            "recovery_ablation": self.recovery_ablation,
            "recovery_ablation_enabled": False,
        }


@dataclass(frozen=True)
class RewardComponents:
    success: float
    progress: float
    verification: float
    duplicate: float
    invalid: float
    normalized_steps: float
    recovery: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_trajectory_reward(
    episode: Mapping[str, Any],
    *,
    max_tool_steps: int,
    weights: RewardWeights | None = None,
    include_recovery_ablation: bool = False,
    reward_hacking_audit_passed: bool = False,
) -> RewardComponents:
    if max_tool_steps <= 0:
        raise ValueError("max_tool_steps must be positive")
    if include_recovery_ablation and not reward_hacking_audit_passed:
        raise ValueError("recovery reward requires a passed reward-hacking audit")
    reward_weights = weights or RewardWeights()

    success = _bounded(float(bool(episode.get("resolved"))))
    initial = episode.get("initial_fail_to_pass_count")
    best = episode.get("best_fail_to_pass_count")
    if isinstance(initial, int) and not isinstance(initial, bool) and initial > 0:
        best_count = best if isinstance(best, int) and not isinstance(best, bool) else initial
        progress = _bounded((initial - max(best_count, 0)) / initial)
    else:
        progress = 0.0

    patch_groups = max(int(episode.get("patch_groups", 0)), 0)
    verified_patch_groups = max(int(episode.get("verified_patch_groups", 0)), 0)
    verification = float(
        patch_groups > 0 and verified_patch_groups == patch_groups
    )
    total_steps = max(int(episode.get("total_steps", 0)), 0)
    invalid_actions = max(int(episode.get("invalid_actions", 0)), 0)
    total_tool_actions = max(total_steps - invalid_actions, 1)
    duplicate = _bounded(
        max(int(episode.get("duplicate_actions", 0)), 0) / total_tool_actions
    )
    invalid = _bounded(
        max(int(episode.get("invalid_actions", 0)), 0) / max(total_steps, 1)
    )
    normalized_steps = _bounded(total_steps / max_tool_steps)
    recovery = float(
        include_recovery_ablation
        and max(int(episode.get("recovered_episodes", 0)), 0) > 0
    )
    total = (
        reward_weights.success * success
        + reward_weights.progress * progress
        + reward_weights.verification * verification
        + reward_weights.duplicate * duplicate
        + reward_weights.invalid * invalid
        + reward_weights.normalized_steps * normalized_steps
        + reward_weights.recovery_ablation * recovery
    )
    return RewardComponents(
        success=success,
        progress=progress,
        verification=verification,
        duplicate=duplicate,
        invalid=invalid,
        normalized_steps=normalized_steps,
        recovery=recovery,
        total=total,
    )
