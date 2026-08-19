"""Inject a background completion into a retained gateway session."""

from __future__ import annotations

from typing import Any


async def deliver_wake(
    adapter: Any,
    *,
    text: str,
    source: Any,
) -> None:
    """Deliver an internal synthetic turn through a messaging adapter."""
    if source is None:
        raise ValueError("deliver_wake requires a SessionSource")

    from gateway.platforms.base import MessageEvent, MessageType

    await adapter.handle_message(
        MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
        )
    )
