"""
Turning model messages into actions, and results back into messages
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import re
from dataclasses import dataclass
from typing import Literal

from .model import Message, ToolResult, ToolSpec

FINISH_STATUSES = ("done", "impossible", "gave_up")


@dataclass(frozen=True)
class Action:
    """Represents an intent extracted from a model message."""

    kind: Literal["bash", "finish", "invalid"]
    command: str = ""
    status: str = ""
    summary: str = ""
    error: str = ""
    call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class Observation:
    """What one action produced. `apply_action` in agent.py builds these."""

    text: str
    exec_result: object | None = None
    done: bool = False
    is_error: bool = False
    flagged: bool = False
    policy_reason: str = ""


class ActionParser(ABC):
    name: str

    @abstractmethod
    def tools(self) -> list[ToolSpec] | None:
        """Tool schema to send with each requiest, or None."""

    @abstractmethod
    def system_suffix(self) -> str:
        """Format instructions appended to the system prompt."""

    @abstractmethod
    def parse(self, message: Message) -> list[Action]:
        """Actions in this message. Empty list = a plain text reply (no action)"""

    @abstractmethod
    def observation_message(
        self, actions: list[Action], observation: list[Observation]
    ) -> Message:
        """Package observations the way this parser's protocol expects."""


EXECUTE_BASH = ToolSpec(
    name="execute_bash",
    description=(
        "Run a shell command inside the isolated Docker sandbox and return its combined"
        "stdout/stderr, plus the exit code if it is non-zero. "
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."}
        },
        "required": ["command"],
    },
)


TASK_COMPLETE = ToolSpec(
    name="task_complete",
    description="Call this once to end the task, with a status and a short summary.",
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": list(FINISH_STATUSES),
                "description": (
                    "done: the task is complete and verified. "
                    "impossible: the task cannot be done as specified. "
                    "gave_up: you could not complete it."
                ),
            },
            "summary": {"type": "string", "description": "What was done, or why not."},
        },
        "required": ["status", "summary"],
    },
)


def _finish_action(status: object, summary: object, **ids) -> Action:
    if status not in FINISH_STATUSES:
        return Action(
            kind="invalid",
            error=f"task_complete status must be one of {','.join(FINISH_STATUSES)}; got {status!r}",
        )
    return Action(kind="finish", status=str(status), summary=str(summary or ""), **ids)


class ToolCallParser(ActionParser):
    name = "tool_call"

    def tools(self) -> list[ToolSpec]:
        return [EXECUTE_BASH, TASK_COMPLETE]

    def system_suffix(self) -> str:
        return (
            "Use the execute_bash tool to run commands. To finish, call task_complete with a"
            "status and a summary. Do not call task_complete in the same turn as execute_bash"
        )

    def parse(self, message: Message) -> list[Action]:
        actions = []
        for call in message.tool_calls:
            ids = {"call_id": call.id, "tool_name": call.name}
            if call.name == "execute_bash":
                command = call.args.get("command")
                if isinstance(command, str) and command.strip():
                    actions.append(Action(kind="bash", command=command, **ids))
            elif call.name == "task_complete":
                actions.append(
                    _finish_action(call.args.get("status"), call.args.get("summary"))
                )
            else:
                actions.append(
                    Action(
                        kind="invalid", error=f"Unknown tool '{call.name}'. ", **ids
                    ),
                )

        return actions

    def observation_message(self, actions, observation) -> Message:
        # Every tool call gets exactly one result, including invalid ones.
        # so no call is ever left unanswered.
        return Message(
            role="tool",
            tool_results=[
                ToolResult(call_id=a.call_id, name=a.tool_name or "", content=o.text)
                for a, o in zip(actions, observation)
            ],
        )


BLOCK_PATTERN = re.compile(r"```(bash-action|task-complete)[^\n]*\n(.*?)```", re.DOTALL)


REGEX_FORMAT_REMINDER = (
    "Reply with exactly one action block. To run a command:\n"
    "```bash-action\n<command>\n```\n"
    "To finish (first line is the status: done, impossible, or gave_up):\n"
    "```task-complete\ndone\n<summary>\n```"
)


class RegexParser(ActionParser):
    name = "regex"

    def tools(self) -> None:
        return None

    def system_suffix(self) -> str:
        return (
            REGEX_FORMAT_REMINDER + "\nNever put more than one action block in a reply."
        )

    def parse(self, message: Message) -> list[Action]:
        blocks = BLOCK_PATTERN.findall(message.text or "")
        if not blocks:
            return []
        if len(blocks) > 1:
            return [
                Action(
                    kind="invalid",
                    error=f"Found {len(blocks)} action blocks; expected exactly one.",
                )
            ]

        tag, body = blocks[0]
        body = body.strip()
        if tag == "bash-action":
            if not body:
                return [Action(kind="invalid", error="The bash-action block is empty.")]
            return [Action(kind="bash", command=body)]

        status, _, summary = body.partition("\n")
        return [_finish_action(status.strip(), summary.strip())]

    def observation_message(self, actions, observations) -> Message:
        observation = observations[0]
        text = observation.text
        if actions[0].kind == "invalid":
            text = f"{text}\n\n{REGEX_FORMAT_REMINDER}"
        return Message(role="user", text=text)
