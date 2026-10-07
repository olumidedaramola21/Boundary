"""
Context Management
"""

from __future__ import annotations

from .model import Message


def prune_window(messages: list[Message], max_messages: int) -> list[Message]:
    """
    Keep the task message plus the most recent `max_messages` messages.

    Invariant: the kept tail always start on an assistant message, so a tool result (or regex observation) is never separated from the action that caused it.

    """
    if len(messages) <= max_messages + 1:
        return messages

    start = len(messages) - max_messages

    while start > 1 and messages[start].role != "assistant":
        start -= 1
        
    return messages[:1] + messages[start:]
