"""The agent loop and runtime state primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

from .actions import Action, ActionParser, Observation
from .compaction import prune_window
from .executor import ExecResult, Executor
from .model import Message, Model, ModelError, ModelResponse
from .policy import CommandPolicy

BASE_SYSTEM_PROMPT = (
    "You are an autonomous terminal assistant working inside an isolated Linux sandbox. "
    "The working directory is /workspace. Files persist for the whole task, and there is no network access \n\n"
    "If a request needs no commands (a greeting, an explanation), reply in plain text and "
    "take no action. Otherwise work step by step: run a command, read its outputs, "
    "and then  decide the next step.\n\n"
    "When you finish, report a status: done (completed and verified), impossible (cannot be done as specified; say why), or gave_up (you could not complete it). Never claim "
    "success you have not verified."
)


@dataclass(frozen=True)
class AgentConfig:
    """Execution limits and runtime parameters."""

    max_steps: int = 25
    max_format_errors: int = 3  # consecutive turns containing an invalid action
    max_history_messages: int = 12
    command_timeout: int = 30
    max_output_chars: int = 2000
    step_delay: float = 1.0
    seed: int | None = None


class StopReason(StrEnum):
    """Reason the agent loop terminated"""

    COMPLETED = "completed"
    CONVERSATIONAL = "conversational"
    STEP_LIMIT = "step_limit"
    FORMAT_ERRORS = "format_errors"
    API_ERROR = "api_error"
    EMPTY_RESPONSE = "empty_response"


@dataclass
class StepRecord:
    """Record of a single interaction turn."""

    step: int
    response: ModelResponse
    actions: list[Action]
    observation: list[Observation]
    history_pruned: tuple[int, int] | None = None


@dataclass
class Trajectory:
    """Complete execution history and outcome of a task."""

    task: str
    steps: list[StepRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    stop_reason: StopReason | None = None
    finish_status: str | None = None  # done | impossible | gave_up
    summary: str | None = None
    error: str | None = None

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens consumed across all steps."""
        return sum(s.response.usage.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens consumed across all steps."""
        return sum(s.response.usage.completion_tokens for s in self.steps)


def truncate_output(output: str, max_chars: int) -> str:
    """Truncate output text from the middle if it exceeds max_chars"""
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    omitted = len(output) - max_chars
    return f"{output[:half]}\n\n... [output truncated: {omitted} characters omitted] .. \n\n{output[-half:]}"


def format_exec_result(result: ExecResult, timeout: int, max_chars: int) -> str:
    """Format command execution results, truncation, timeouts and exit codes."""
    output = truncate_output(result.output, max_chars)
    if result.timed_out:
        text = f"Error: the command timed out after {timeout}s and was killed."
        return f"{text}\nPartial output:\n{output}" if output else text
    text = output or "[command finished with no output]"
    if result.exit_code != 0:
        text += f"\n[exit code: {result.exit_code}]"
    return text


def apply_action(
    action: Action,
    *,
    executor: Executor,
    policy: CommandPolicy,
    timeout: int = 30,
    max_output_chars: int = 2000,
) -> Observation:
    """Validate, policy-check, and execute an agent action."""

    if action.kind == "invalid":
        return Observation(text=f"Error: {action.error}", is_error=True)

    if action.kind == "finish":
        return Observation(
            text=f"Acknowledged: task marked '{action.status}'", done=True
        )

    decision = policy.check(action.command)
    if not decision.allow:
        return Observation(
            text=f"Command blocked by policy: {decision.reason}",
            is_error=True,
            flagged=decision.flagged,
            policy_reason=decision.reason,
        )

    result = executor.execute(action.command, timeout=timeout)
    return Observation(
        text=format_exec_result(result, timeout, max_output_chars),
        exec_result=result,
        flagged=decision.flagged,
        policy_reason=decision.reason,
    )


class Agent:
    """Autonomous agent orchestrating LLM interactions, tool calls, and execution state."""

    def __init__(
        self,
        model: Model,
        executor: Executor,
        parser: ActionParser,
        policy: CommandPolicy,
        config: AgentConfig = AgentConfig(),
        system_prompt: str = BASE_SYSTEM_PROMPT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.executor = executor
        self.parser = parser
        self.policy = policy
        self.config = config
        self.system = f"{system_prompt}\n\n{parser.system_suffix()}"
        self._sleep = sleep

    def run(
        self, task: str, on_step: Callable[[StepRecord], None] | None = None
    ) -> Trajectory:
        """Execute the multi-turn agent loop for a given task until completion or failure"""

        cfg = self.config
        trajectory = Trajectory(task=task)
        messages: list[Message] = [Message(role="user", text=task)]
        format_errors = 0

        def stop(reason: StopReason, **fields) -> Trajectory:
            """Finalize and return the trajectory with a terminal reason and metadata"""
            trajectory.stop_reason = reason
            trajectory.messages = messages
            for key, value in fields.items():
                setattr(trajectory, key, value)
            return trajectory

        for step in range(1, cfg.max_steps + 1):
            before = len(messages)
            messages = prune_window(messages, cfg.max_history_messages)
            pruned = (before, len(messages)) if len(messages) < before else None

            try:
                response = self.model.complete(
                    self.system, messages, self.parser.tools(), seed=cfg.seed
                )
            except ModelError as e:
                return stop(StopReason.API_ERROR, error=str(e))

            message = response.message
            messages.append(message)

            actions = self.parser.parse(message)
            observations = [
                apply_action(
                    action,
                    executor=self.executor,
                    policy=self.policy,
                    timeout=cfg.command_timeout,
                    max_output_chars=cfg.max_output_chars,
                )
                for action in actions
            ]

            if actions:
                messages.append(self.parser.observation_message(actions, observations))

            record = StepRecord(step, response, actions, observations, pruned)
            trajectory.steps.append(record)

            if on_step:
                on_step(record)

            if not actions:
                if not message.text and not message.tool_calls:
                    return stop(StopReason.EMPTY_RESPONSE)
                return stop(StopReason.CONVERSATIONAL)

            finish = next(
                (action for action in actions if action.kind == "finish"), None
            )
            if finish:
                return stop(
                    StopReason.COMPLETED,
                    finish_status=finish.status,
                    summary=finish.summary,
                )

            if any(action.kind == "invalid" for action in actions):
                format_errors += 1

                if format_errors >= cfg.max_format_errors:
                    return stop(StopReason.FORMAT_ERRORS)
            else:
                format_errors = 0

            if cfg.step_delay:
                self._sleep(cfg.step_delay)

        return stop(StopReason.STEP_LIMIT)
