"""Retained Telegram text-batching behavior."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
        ),
    )


def _make_telegram_adapter():
    from gateway.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestTelegramAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_telegram_adapter()
        adapter._enqueue_text_event(_make_event("short msg"))

        await asyncio.sleep(0.15)

        adapter.handle_message.assert_called_once()
