"""Retained Telegram and Mattermost outbound-delivery tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, _send_to_platform, resolve_send_target


def test_parse_telegram_topic_target():
    assert _parse_target_ref("telegram", "-100123:77") == ("-100123", "77", True)


def test_parse_mattermost_channel_id():
    assert _parse_target_ref("mattermost", "channel_abc") == ("channel_abc", None, True)


def test_unknown_platform_is_rejected():
    assert resolve_send_target("unsupported", "123")[2] == "Unknown platform: unsupported"


def test_mattermost_send_uses_internal_sender_and_preserves_thread():
    sender = AsyncMock(return_value={"success": True, "message_id": "post-1"})
    config = SimpleNamespace(enabled=True, token="token", extra={})
    with patch("gateway.platforms.mattermost.standalone_send", sender):
        result = asyncio.run(_send_to_platform(
            Platform.MATTERMOST, config, "channel-1", "hello", thread_id="root-1"
        ))
    assert result["success"] is True
    sender.assert_awaited_once()
    assert sender.await_args.kwargs["thread_id"] == "root-1"


def test_telegram_send_uses_internal_sender():
    sender = AsyncMock(return_value={"success": True, "message_id": "42"})
    config = SimpleNamespace(enabled=True, token="token", extra={})
    with patch("gateway.platforms.telegram.adapter.standalone_send", sender):
        result = asyncio.run(_send_to_platform(
            Platform.TELEGRAM, config, "123", "hello", thread_id="7"
        ))
    assert result["success"] is True
    sender.assert_awaited_once()
