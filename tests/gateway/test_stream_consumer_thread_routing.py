"""Shared stream-consumer reply/thread routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer


def _adapter(*, max_length=4096):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="msg-1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True)
    )
    adapter.MAX_MESSAGE_LENGTH = max_length
    return adapter


@pytest.mark.asyncio
async def test_first_stream_send_uses_initial_reply_id():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat",
        metadata={"thread_id": "thread"},
        initial_reply_to_id="source-message",
    )

    await consumer._send_or_edit("Hello")

    assert adapter.send.await_args.kwargs["reply_to"] == "source-message"


@pytest.mark.asyncio
async def test_subsequent_stream_update_edits_first_message():
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat",
        initial_reply_to_id="source-message",
    )

    await consumer._send_or_edit("Hello")
    await consumer._send_or_edit("Hello again")

    adapter.edit_message.assert_awaited_once()
    assert adapter.edit_message.await_args.kwargs["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_first_overflow_chunk_keeps_initial_reply_id():
    adapter = _adapter(max_length=10)
    consumer = GatewayStreamConsumer(
        adapter,
        "chat",
        initial_reply_to_id="source-message",
    )
    consumer._accumulated = "A" * 100
    consumer._current_edit_interval = 999

    await consumer._send_new_chunk("chunk", consumer._initial_reply_to_id)

    assert adapter.send.await_args.kwargs["reply_to"] == "source-message"
