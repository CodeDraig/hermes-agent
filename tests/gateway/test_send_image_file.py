"""Retained media extraction and Telegram file-send behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.telegram.adapter import TelegramAdapter


def test_png_media_tag_is_extracted():
    content = "Here is the screenshot:\nMEDIA:/tmp/shot.png"

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == [("/tmp/shot.png", False)]
    assert "MEDIA:" not in cleaned


@pytest.mark.asyncio
async def test_telegram_sends_local_image_as_photo(tmp_path):
    image = tmp_path / "screenshot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_photo = AsyncMock(
        return_value=MagicMock(message_id=42)
    )

    result = await adapter.send_image_file("12345", str(image))

    assert result.success
    assert result.message_id == "42"
    adapter._bot.send_photo.assert_awaited_once()


def test_telegram_image_send_requires_connection():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = None

    result = asyncio.run(adapter.send_image_file("12345", "/tmp/image.png"))

    assert not result.success
    assert "Not connected" in result.error
