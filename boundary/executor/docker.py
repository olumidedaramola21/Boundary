from __future__ import annotations
import subprocess
import time
from dataclasses import dataclass, field

from .base import Executor, ExecResult

SAFE_ENV: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS": "off",
    "TQDM_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
}

TIMEOUT_EXIT_CODE = 124
KILLED_EXIT_CODE = 137
KILL_GRACE_SECONDS = 2


@dataclass(frozen=True)
class SandboxProfile:

    name: str
    image: str = "python:3.12-slim"
    network: str = "none"
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 64
    workspace: str = "/workspace"
    cap_drop_all: bool = False
    no_new_privileges: bool = False
    user: str | None = None
    read_only_root: bool = False
    tmpfs_workspace_size: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Safety floor: the process and memory limits can never be switched off,
        # not even for red-team baslines. An uncontained fork bomb can take WSL down
        if not isinstance(self.pids_limit, int) or not 0 < self.pids_limit <= 4096:
            raise ValueError(
                f"pids_limit must be between 1 and 4096, got {self.pids_limit}"
            )

        if not self.memory:
            raise ValueError("Memory limit is mandatory")


DEV = SandboxProfile(name="dev")

HARDENED = SandboxProfile(
    name="hardened",
    cap_drop_all=True,
    no_new_privileges=True,
    user="1000:1000",
    read_only_root=True,
    tmpfs_workspace_size="200m",
    extra_env={"HOME": "/tmp"},
)

PROFILES: dict[str, SandboxProfile] = {p.name: p for p in (DEV, HARDENED)}


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class DockerExecutor(Executor):

    def __init__(
        self,
        profile: SandboxProfile = DEV,
        container_name: str = "agent-sandbox",
    ) -> None:

        self.profile = profile
        self.container_name = container_name

    def run_args(self) -> list[str]:
        p = self.profile
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--network",
            p.network,
            "--memory",
            p.memory,
            "--memory-swap",
            p.memory,
            "--cpus",
            p.cpus,
            "--pids-limit",
            str(p.pids_limit),
            "-w",
            p.workspace,
        ]

        for key, value in {**SAFE_ENV, **p.extra_env}.items():
            args += ["-e", f"{key}={value}"]

        if p.cap_drop_all:
            args += ["--cap-drop", "ALL"]
        if p.no_new_privileges:
            args += ["--security-opt", "no-new-priveleges:true"]
        if p.user:
            args += ["--user", p.user]
        if p.read_only_root:
            args += ["--read-only", "--tmpfs", "/tmp:rw,exec,size=64m"]
        if p.tmpfs_workspace_size:
            options = f"rw,exec,size={p.tmpfs_workspace_size}"
            if p.user:
                uid, _, gid = p.user.partition(":")
                options += f",uid={uid},gid={gid or uid}"
            args += ["--tmpfs", f"{p.workspace}:{options}"]

        args += [p.image, "sleep", "infinity"]
        return args

    def setup(self) -> None:
        self._remove()
        try:
            proc = subprocess.run(self.run_args(), capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "docker CLI not found. Run Boundary from inside WSL."
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to start sandbox container: {proc.stderr.strip()}"
            )

    def execute(self, command, timeout=30) -> ExecResult:
        args = [
            "docker",
            "exec",
            self.container_name,
            "timeout",
            "-k",
            str(KILL_GRACE_SECONDS),
            str(timeout),
            "sh",
            "-c",
            command,
        ]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + KILL_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                stdout=_text(e.stdout),
                stderr=_text(e.stderr),
                exit_code=TIMEOUT_EXIT_CODE,
                duration_ms=_elapsed_ms(start),
            )

        duration_ms = _elapsed_ms(start)
        timed_out = proc.returncode == TIMEOUT_EXIT_CODE or (
            proc.returncode == KILLED_EXIT_CODE and duration_ms >= timeout * 1000
        )
        return ExecResult(
            proc.stdout, proc.stderr, proc.returncode, duration_ms, timed_out
        )

    def teardown(self):
        self._remove()

    def _remove(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
