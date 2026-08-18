"""Retained Telegram, Mattermost, and local delivery routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import SendResult
from gateway.session import SessionSource


def test_explicit_telegram_topic_parses():
    target = DeliveryTarget.parse("telegram:-100:42")
    assert target.platform == Platform.TELEGRAM
    assert target.chat_id == "-100"
    assert target.thread_id == "42"
    assert target.is_explicit


def test_mattermost_channel_id_preserves_case():
    target = DeliveryTarget.parse("MATTERMOST:ChannelABC")
    assert target.platform == Platform.MATTERMOST
    assert target.chat_id == "ChannelABC"


def test_origin_preserves_thread():
    source = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="channel",
        thread_id="root",
    )
    target = DeliveryTarget.parse("origin", origin=source)
    assert target.to_string() == "origin"
    assert target.thread_id == "root"


@pytest.mark.asyncio
async def test_mattermost_delivery_uses_live_adapter():
    adapter = SimpleNamespace(
        splits_long_messages=True,
        send=AsyncMock(return_value=SendResult(success=True, message_id="post")),
    )
    config = GatewayConfig(
        platforms={Platform.MATTERMOST: PlatformConfig(enabled=True, token="token")}
    )
    router = DeliveryRouter(config, adapters={Platform.MATTERMOST: adapter})

    result = await router.deliver(
        "hello",
        [DeliveryTarget(platform=Platform.MATTERMOST, chat_id="channel")],
    )

    assert result["mattermost:channel"]["success"] is True
    adapter.send.assert_awaited_once()


def test_local_delivery_writes_output(tmp_path):
    router = DeliveryRouter(GatewayConfig())
    router.output_dir = tmp_path

    result = router._deliver_local("hello", "job", "Job", None)

    assert "hello" in open(result["path"], encoding="utf-8").read()
