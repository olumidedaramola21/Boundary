"""
Command policy: what to do when a command looks destructive

"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

DANGEROUS_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[fdx]{1,3}\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd[a-z]",
)

COMPILED_DEFAULT_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in DANGEROUS_PATTERNS),
    re.IGNORECASE,
)


class PolicyMode(StrEnum):
    """Enforcement behavior mode for detected dangerous commands."""

    INTERACTIVE = "interactive"
    LOG_ONLY = "log_only"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    """Outcome of evaluating a command against a safety policy"""

    allow: bool
    reason: str = ""
    flagged: bool = False


class CommandPolicy:
    """Evaluate shell commands against dangerous patterns and enforces safety policies"""

    def __init__(
        self,
        mode: PolicyMode = PolicyMode.INTERACTIVE,
        patterns: tuple[str, ...] | None = None,
        ask: Callable[[str], str] = input,
    ) -> None:
        """Initialize the policy with an enforcement mode, regex patterns, and prompt handler"""
        self.mode = PolicyMode(mode)
        self.ask = ask
        if patterns is None:
            self.pattern = COMPILED_DEFAULT_PATTERN
        else:
            self.pattern = re.compile(
                "|".join(f"(?:{pattern})" for pattern in patterns),
                re.IGNORECASE,
            )

    def check(self, command: str) -> Decision:
        """Check whether a command matches risky patterns and return an enforcement decision"""
        match = self.pattern.search(command)
        if match is None:
            return Decision(allow=True)

        reason = f"matches dangerous pattern {match.pattern!r}"
        match self.mode:
            case PolicyMode.LOG_ONLY:
                return Decision(allow=True, reason=reason, flagged=True)

            case PolicyMode.DENY:
                return Decision(allow=False, reason=reason, flagged=True)

            case PolicyMode.INTERACTIVE:
                answer = self.ask(
                    f"\n[SAFETY] Potentially destructive commands:\n   ${command}\nAllow it? (y/N)"
                )

                allowed = answer.strip().lower() in ("y", "yes")
                verdict = "approved by user" if allowed else "denied by user"
                return Decision(
                    allow=allowed,
                    reason=f"{reason} {verdict}",
                    flagged=True,
                )
