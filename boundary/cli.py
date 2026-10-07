"""
A thin wrapper around Agent.run()
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .actions import RegexParser, ToolCallParser
from .agent import Agent, AgentConfig, StepRecord
from .executor import PROFILES, DockerExecutor
from .model import GeminiModel, SamplingParams
from .policy import CommandPolicy, PolicyMode

PARSERS = {"tool_call": ToolCallParser, "regex": RegexParser}
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def print_step(record: StepRecord) -> None:
    if record.history_pruned:
        before, after = record.history_pruned
        print(f"[history pruned: {before} -> {after}]")

    if record.response.message.text:
        print(f"\nAgent:\n{record.response.message.text}")

    for action, observation in zip(record.actions, record.observation):
        if action.kind == "bash":
            print(f"\n[Step {record.step} | Running in Docker ]: {action.command}")
            print(f"Output:\n{observation.text}")
        elif action.kind == "finish":
            print(f"\n[Task {action.status}]: {action.summary}")


def print_retry(attempt: int, code: int | None, delay: float) -> None:
    print(f"\n[API {code}] retry {attempt} in {delay:.0f}s...")


def with_session_context(task: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return task
    lines = "\n".join(f"- User: {q}\n You: {a}" for q, a in history[-5:])
    return f"Earlier in this conversation:\n{lines}\n\nCurrent result:  {task}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="Boundary")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--parser", choices=sorted(PARSERS), default="tool_call")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="dev")
    ap.add_argument(
        "--policy", choices=[mode.value for mode in PolicyMode], default="interactive"
    )
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--session-memory", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv()

    model = GeminiModel(
        args.model, sampling=SamplingParams(temperature=0.2), on_retry=print_retry
    )

    config = AgentConfig(max_steps=args.max_steps, seed=args.seed)

    print(
        f"Boundary | model={args.model} "
        f"parser={args.parser} "
        f"profile={args.profile} "
        f"policy={args.policy}"
    )

    print("Type 'quit' to close ")

    history: list[tuple[str, str]] = []

    with DockerExecutor(profile=PROFILES[args.profile]) as executor:
        print(f"[sandbox '{executor.container_name}' started ]")
        agent = Agent(
            model, executor, PARSERS[args.parser](), CommandPolicy(args.policy), config
        )

        while True:
            try:
                task = input("\nUser > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not task:
                continue
            if task.lower() in ("quit", "exit", "stop"):
                print("Goodbye!")
                break

            prompt = (
                with_session_context(task, history) if args.session_memory else task
            )

            trajectory = agent.run(prompt, on_step=print_step)

            line = f"[stop reason: {trajectory.stop_reason}]"

            if trajectory.finish_status:
                line += f"  | status: {trajectory.finish_status} "

            print(
                f"{line}  | tokens: "
                f"{trajectory.prompt_tokens} in / "
                f"{trajectory.completion_tokens} out"
            )

            if trajectory.error:
                print(f"[error: {trajectory.error}]")

            last_text = trajectory.summary or (
                trajectory.steps[-1].response.message.text if trajectory.steps else ""
            )
            history.append((task, last_text or "(no reply)"))

        print("[sandbox removed]")


if __name__ == "__main__":
    main()
