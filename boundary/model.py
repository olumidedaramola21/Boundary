"""
Provider-neutral messages types, and model adapter interfaces.
"""

from __future__ import annotations

import os
import re
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    """Represents a tool invocation request made by a model"""

    id: str | None
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """Represents the execution outcome of a tool call"""

    call_id: str | None
    name: str
    content: str


@dataclass
class Message:
    """Standardized conversation message passed across Boundary components."""

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    raw: Any = None


@dataclass(frozen=True)
class ToolSpec:
    """Schema definition for a tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SamplingParams:
    """Generation hyperparameters controlling model output randomness."""

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


@dataclass
class Usage:
    """Token consumption metrics for a single model completion"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class ModelResponse:
    """Normalized response payload returned by a Model adapter."""

    message: Message
    usage: Usage
    latency_ms: int
    sampling: SamplingParams
    attempts: int = 1
    finish_reason: str | None = None
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None


class ModelError(Exception):
    """Raised when an unrecoverable model provider failure occurs"""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class Model(ABC):
    """Abstract base adapter interface for LLM providers"""

    name: str

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        seed: int | None = None,
    ) -> ModelResponse:
        """Execute a completion request against the provider"""


RETRYABLE_CODES = (429, 500, 503)
RetryCallback = Callable[
    [int, int | None, float], None
]  # (attempt, code, delay_seconds)


@dataclass(frozen=True)
class QuotaInfo:
    """Parsed quota and rate-limit metadata extracted from provider errors"""

    retry_delay: float | None = None
    per_day: bool = False


def parse_quota_info(error_json: Any, message: str = "") -> QuotaInfo:
    """Extract backoff delay and quota exhaustion details from API errors."""

    retry_delay: float | None = None
    per_day = False
    details = []
    if isinstance(error_json, dict):
        details = (error_json.get("error") or {}).get("details") or []
    for item in details:
        kind = str(item.get("@type", ""))
        if kind.endswith("RetryInfo") and item.get("retryDelay"):
            try:
                retry_delay = float(str(item["retryDelay"]).rstrip("s"))
            except ValueError:
                pass
        if kind.endswith("QuotaFailure"):
            for violation in item.get("violations") or []:
                if "PerDay" in str(violation.get("quotaId", "")):
                    per_day = True
    if retry_delay is None:
        match = re.search(r"retry in ([\d.]+)s", message or "")
        if match:
            retry_delay = float(match.group(1))
    return QuotaInfo(retry_delay=retry_delay, per_day=per_day)


class GeminiModel(Model):
    """Adapter implementation for Google Gemini models via the GenAI SDK"""

    def __init__(
        self,
        name: str,
        *,
        api_key: str | None = None,
        sampling: SamplingParams = SamplingParams(),
        max_retries: int = 5,
        base_delay: float = 5.0,
        max_delay: float = 60.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: RetryCallback | None = None,
    ) -> None:
        from google import genai  # imported here so the rest of boundary never needs it

        self.name = name
        self.sampling = sampling
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.client = client or genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY")
        )
        self._sleep = sleep
        self._on_retry = on_retry

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        seed: int | None = None,
    ) -> ModelResponse:
        """Generate a model completion for the given conversational turn."""
        from google.genai import types

        sampling = (
            replace(self.sampling, seed=seed) if seed is not None else self.sampling
        )

        tool_config = (
            [types.Tool(function_declarations=[self._declaration(t) for t in tools])]
            if tools
            else None
        )

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tool_config,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            seed=sampling.seed,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        contents = [
            self.to_content(m) for m in messages
        ]  # provider-neutral message -> sdk specific types.Content

        start = time.monotonic()
        response, attempts = self._generate(contents, config)
        latency_ms = int((time.monotonic() - start) * 1000)
        return self.from_response(response, latency_ms, sampling, attempts)

    @staticmethod
    def to_content(message: Message):
        """Convert a provider-neutral Message into a Gemini types.Content Object."""
        from google.genai import types

        if message.role == "assistant" and isinstance(message.raw, types.Content):
            return message.raw

        if message.role == "tool":
            return types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=tool_result.call_id,
                            name=tool_result.name,
                            response={"output": tool_result.content},
                        )
                    )
                    for tool_result in message.tool_results
                ],
            )

        if message.role == "assistant":
            parts = [types.Part(text=message.text)] if message.text else []
            parts.extend(
                types.Part(
                    function_call=types.FunctionCall(
                        id=tool_call.id, name=tool_call.name, args=tool_call.args
                    )
                )
                for tool_call in message.tool_calls
            )

            return types.Content(role="model", parts=parts)

        return types.Content(role="user", parts=[types.Part(text=message.text)])

    @staticmethod
    def from_response(
        response, latency_ms: int, sampling: SamplingParams, attempts: int = 1
    ) -> ModelResponse:
        """Converts a Gemini API response into a standardized ModelResponse."""

        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        parts = (content.parts or []) if content else []

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in parts:
            if part.text and not getattr(part, "thought", False):
                text_chunks.append(part.text)
            elif part.function_call:
                function_call = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=function_call.id,
                        name=function_call.name,
                        args=dict(function_call.args or {}),
                    )
                )

        meta = response.usage_metadata
        usage = Usage(
            prompt_tokens=(meta.prompt_token_count or 0) if meta else 0,
            completion_tokens=(meta.candidates_token_count or 0) if meta else 0,
            thinking_tokens=(meta.thoughts_token_count or 0) if meta else 0,
        )
        finish = getattr(candidate, "finish_reason", None) if candidate else None
        return ModelResponse(
            message=Message(
                role="assistant",
                text="".join(text_chunks),
                tool_calls=tool_calls,
                raw=content,
            ),
            usage=usage,
            latency_ms=latency_ms,
            sampling=sampling,
            attempts=attempts,
            finish_reason=(
                str(getattr(finish, "value", finish)) if finish is not None else None
            ),
        )

    @staticmethod
    def _declaration(spec: ToolSpec):
        """Convert a ToolSpec into a Gemini FunctionDeclaration schema"""
        from google.genai import types

        return types.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            parameters_json_schema=spec.parameters,
        )

    def _generate(self, contents, config):
        """Execute generate_content with rate-limit and exponential backoff"""
        from google.genai import errors

        backoff = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.name, contents=contents, config=config
                )
                return response, attempt

            except errors.APIError as e:
                quota = parse_quota_info(e.details, e.message or "")
                if e.code == 429 and quota.per_day:
                    raise ModelError(
                        f"daily quota exhausted for {self.name}; retrying won't help until it resets",
                        code=429,
                    ) from e
                if e.code not in RETRYABLE_CODES or attempt == self.max_retries:
                    raise ModelError(
                        f"{e.code} {e.status}: {e.message}", code=e.code
                    ) from e

                if quota.retry_delay is not None:
                    delay = quota.retry_delay + 1
                else:
                    delay = backoff

                delay = min(delay, self.max_delay)

                if self._on_retry:
                    self._on_retry(attempt, e.code, delay)
                self._sleep(delay)
                backoff = min(backoff * 2, self.max_delay)

        raise ModelError("Unreachable")
