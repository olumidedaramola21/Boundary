"""
Executor interface for running shell commands and returning structured results.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class ExecResult:
    """Result of a command execution"""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


class Executor(ABC):
    """Abstract base class for command execution environments."""

    @abstractmethod
    def execute(self, command: str, timeout: int = 30) -> ExecResult:
        """
        Run a command, Must return an ExecResult with timed_out=True rather than raising.
        """

    def setup(self) -> None:
        """Initialize the execution environment"""

    def teardown(self) -> None:
        """Clean up the execution environment"""

    def __enter__(self) -> Self:
        """Enter the runtime context and initialize resources."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, ex_tb) -> None:
        """Exit the runtime context, and release resources."""
        self.teardown()
