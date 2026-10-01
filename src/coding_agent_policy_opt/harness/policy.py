from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .tools import REGISTERED_TOOLS


def enumerate_harness_candidates() -> list[dict[str, Any]]:
    """Return the pre-registered Harness Policy search space for dev tuning.

    Candidate selection is deliberately discrete and auditable. It is run only
    on dev; the selected values are copied into the frozen test configuration.
    """

    candidates: list[dict[str, Any]] = [
        {
            "variant": "static_v1",
            "history_limit": 1_000_000,
            "verification_mode": "advisory",
            "recovery_mode": "advisory",
        }
    ]
    for history_limit in (4, 8, 12):
        for verification_mode in ("advisory", "required"):
            for recovery_mode in ("advisory", "required"):
                candidates.append(
                    {
                        "variant": "adaptive_v1",
                        "history_limit": history_limit,
                        "verification_mode": verification_mode,
                        "recovery_mode": recovery_mode,
                    }
                )
    return candidates


def select_harness_candidate(dev_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one Harness candidate from dev summaries using a frozen ranking."""

    if not dev_reports:
        raise ValueError("dev_reports must not be empty")
    candidates = enumerate_harness_candidates()
    allowed = {tuple(sorted(candidate.items())) for candidate in candidates}
    static_reports = [
        report
        for report in dev_reports
        if isinstance(report.get("harness_policy"), dict)
        and report["harness_policy"].get("variant") == "static_v1"
    ]
    if len(static_reports) != 1:
        raise ValueError(
            "dev_reports must contain exactly one static_v1 baseline for the "
            "environment-validity gate"
        )
    static_metrics = static_reports[0].get("metrics")
    if not isinstance(static_metrics, dict):
        raise ValueError("static_v1 baseline must contain a metrics object")
    static_validity = static_metrics.get("environment_validity_rate")
    if static_validity is None:
        raise ValueError("static_v1 baseline must report environment_validity_rate")
    static_validity = float(static_validity)
    prepared_reports: list[dict[str, Any]] = []
    excluded_reports: list[dict[str, Any]] = []
    for report in dev_reports:
        config = report.get("harness_policy")
        if not isinstance(config, dict) or tuple(sorted(config.items())) not in allowed:
            raise ValueError("dev report contains a Harness Policy outside the frozen candidate set")
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("dev report must contain a metrics object")
        validity = metrics.get("environment_validity_rate")
        if validity is None:
            raise ValueError("dev report must have environment_validity_rate")
        validity = float(validity)
        if validity <= 0:
            excluded_reports.append(
                {"harness_policy": config, "reason": "zero_environment_validity"}
            )
            continue
        if validity + 1e-12 < static_validity:
            excluded_reports.append(
                {
                    "harness_policy": config,
                    "reason": "environment_validity_below_static_baseline",
                    "environment_validity_rate": validity,
                    "static_baseline_rate": static_validity,
                }
            )
            continue
        prepared_reports.append(
            {
                **report,
                "average_tool_calls": (
                    metrics.get("costs", {})
                    .get("all_valid", {})
                    .get("tool_calls", {})
                    .get("mean")
                ),
            }
        )

    if not prepared_reports:
        raise ValueError("No Harness candidate passed the environment-validity gate")

    def score(report: dict[str, Any]) -> tuple[float, float, float, float, float, float, tuple[tuple[str, Any], ...]]:
        metrics = report["metrics"]
        missing = [
            key
            for key in (
                "resolve_rate",
                "episode_recovery_rate",
                "premature_stop_rate",
                "invalid_action_rate",
                "duplicate_action_rate",
            )
            if metrics.get(key) is None
        ]
        if missing or report["average_tool_calls"] is None:
            raise ValueError(f"dev report lacks Harness selection metrics: {missing}")
        config = tuple(sorted(report["harness_policy"].items()))
        return (
            -float(metrics["resolve_rate"]),
            -float(metrics["episode_recovery_rate"] or 0.0),
            float(metrics["premature_stop_rate"]),
            float(metrics["invalid_action_rate"]),
            float(metrics["duplicate_action_rate"]),
            float(report["average_tool_calls"]),
            config,
        )

    ranking = sorted(prepared_reports, key=score)
    return {
        "schema_version": "2.0.0",
        "report_type": "harness_policy_selection",
        "selection_split": "dev",
        "selection_rule": [
            "environment_validity_at_least_static_baseline",
            "resolve_rate_desc",
            "episode_recovery_rate_desc",
            "premature_stop_rate_asc",
            "invalid_action_rate_asc",
            "duplicate_action_rate_asc",
            "average_tool_calls_asc",
            "candidate_config_lexicographic",
        ],
        "static_baseline_environment_validity_rate": static_validity,
        "selected": ranking[0],
        "ranking": ranking,
        "excluded": excluded_reports,
    }


@dataclass(frozen=True)
class HarnessPolicyDecision:
    """A deterministic control decision made outside the language model."""

    variant: str
    allowed_tools: tuple[str, ...]
    history_limit: int
    context_strategy: str
    verification_required: bool
    recovery_required: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_tools"] = list(self.allowed_tools)
        value["reason_codes"] = list(self.reason_codes)
        return value


class HarnessPolicy(Protocol):
    def decide(self, context: dict[str, Any]) -> HarnessPolicyDecision:
        """Select the tool surface and execution guard for one agent turn."""

    def describe(self) -> dict[str, Any]:
        """Return an auditable, non-secret policy description."""


class StaticHarnessPolicy:
    """V0 control: expose every registered tool and do not intervene."""

    def __init__(self, *, history_limit: int = 8) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self.history_limit = history_limit

    def decide(self, context: dict[str, Any]) -> HarnessPolicyDecision:
        del context
        return HarnessPolicyDecision(
            variant="static_v1",
            allowed_tools=REGISTERED_TOOLS,
            history_limit=self.history_limit,
            context_strategy="full_recent_history",
            verification_required=False,
            recovery_required=False,
            reason_codes=("STATIC_TOOL_SURFACE",),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "variant": "static_v1",
            "history_limit": self.history_limit,
            "verification_mode": "advisory",
            "recovery_mode": "advisory",
        }


class AdaptiveHarnessPolicy:
    """Rule-based Harness Policy used as the first optimizable policy.

    The policy never invents tools or executes code. It only changes the model's
    visible action surface and adds guards based on observable task state.
    """

    def __init__(
        self,
        *,
        history_limit: int = 8,
        require_verification: bool = True,
        require_recovery: bool = True,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self.history_limit = history_limit
        self.require_verification = require_verification
        self.require_recovery = require_recovery

    @staticmethod
    def _state(context: dict[str, Any]) -> dict[str, Any]:
        value = context.get("policy_state", {})
        return value if isinstance(value, dict) else {}

    def decide(self, context: dict[str, Any]) -> HarnessPolicyDecision:
        state = self._state(context)
        patch_pending = bool(state.get("patch_pending_verification"))
        recovery_active = bool(state.get("recovery_active"))
        last_error = state.get("last_error_code")
        last_action = state.get("last_action_type")

        if context.get("environment_state", {}).get("patch_conflicts"):
            return HarnessPolicyDecision(
                variant="adaptive_v1",
                allowed_tools=("read_file", "inspect_diff", "finish"),
                history_limit=self.history_limit,
                context_strategy="conflict_replanning",
                verification_required=False,
                recovery_required=True,
                reason_codes=("EXTERNAL_EDIT_CONFLICT", "READ_BEFORE_REPLAN"),
            )

        recovery = context.get("environment_state", {}).get("recovery", {})
        evidence = recovery.get("evidence")
        if evidence and evidence.get("validity") != "valid":
            return HarnessPolicyDecision(
                variant="adaptive_v1", allowed_tools=REGISTERED_TOOLS,
                history_limit=self.history_limit, context_strategy="evidence_revalidation",
                verification_required=True, recovery_required=False,
                reason_codes=("STALE_VALIDATION_EVIDENCE", "RETEST_CURRENT_WORKSPACE"),
            )
        if recovery.get("regression"):
            return HarnessPolicyDecision(
                variant="adaptive_v1", allowed_tools=REGISTERED_TOOLS,
                history_limit=self.history_limit, context_strategy="candidate_recovery",
                verification_required=False, recovery_required=self.require_recovery,
                reason_codes=("OBSERVED_TEST_REGRESSION", "INSPECT_CANDIDATES_OR_REPAIR"),
            )

        if recovery_active:
            allowed = (
                "search_code",
                "read_file",
                "apply_patch",
                "run_tests",
                "inspect_diff",
                "finish",
            )
            return HarnessPolicyDecision(
                variant="adaptive_v1",
                allowed_tools=allowed,
                history_limit=min(self.history_limit, 6),
                context_strategy="recovery_focus",
                verification_required=False,
                recovery_required=self.require_recovery,
                reason_codes=("TEST_FAILURE_RECOVERY", "HIDE_PREMATURE_FINISH"),
            )

        if patch_pending:
            allowed = (
                "read_file",
                "apply_patch",
                "run_tests",
                "inspect_diff",
                "finish",
            )
            return HarnessPolicyDecision(
                variant="adaptive_v1",
                allowed_tools=allowed,
                history_limit=min(self.history_limit, 5),
                context_strategy="verification_focus",
                verification_required=self.require_verification,
                recovery_required=False,
                reason_codes=("PATCH_PENDING_VERIFICATION", "HIDE_PREMATURE_FINISH"),
            )

        if last_action == "run_tests" and state.get("last_test_passed") is True:
            allowed = ("inspect_diff", "read_file", "run_tests", "finish")
            return HarnessPolicyDecision(
                variant="adaptive_v1",
                allowed_tools=allowed,
                history_limit=min(self.history_limit, 4),
                context_strategy="completion_focus",
                verification_required=False,
                recovery_required=False,
                reason_codes=("TEST_PASSED", "PREFER_COMPLETION"),
            )

        reasons = ["LOCALIZATION_FIRST"]
        if last_error:
            reasons.append("RETRY_AFTER_TOOL_ERROR")
        return HarnessPolicyDecision(
            variant="adaptive_v1",
            allowed_tools=(
                "search_code",
                "read_file",
                "apply_patch",
                "run_tests",
                "inspect_diff",
                "finish",
            ),
            history_limit=self.history_limit,
            context_strategy="localization_and_planning",
            verification_required=False,
            recovery_required=False,
            reason_codes=tuple(reasons),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "variant": "adaptive_v1",
            "history_limit": self.history_limit,
            "verification_mode": "required" if self.require_verification else "advisory",
            "recovery_mode": "required" if self.require_recovery else "advisory",
        }


def build_harness_policy(config: Any) -> HarnessPolicy:
    """Build the runtime policy from the experiment's frozen config."""

    value = getattr(config, "harness_policy", None)
    variant = getattr(value, "variant", "static_v1")
    history_limit = int(getattr(value, "history_limit", 1_000_000))
    if variant == "static_v1":
        return StaticHarnessPolicy(history_limit=history_limit)
    if variant == "adaptive_v1":
        return AdaptiveHarnessPolicy(
            history_limit=history_limit,
            require_verification=getattr(value, "verification_mode", "required") == "required",
            require_recovery=getattr(value, "recovery_mode", "required") == "required",
        )
    raise ValueError(f"Unknown harness policy variant: {variant}")
