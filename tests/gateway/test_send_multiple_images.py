"""Retained multi-image behavior for the base, Telegram, and Mattermost adapters."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.mattermost import MattermostAdapter
from gateway.platforms.telegram.adapter import TelegramAdapter


class StubAdapter(BasePlatformAdapter):
    name = "stub"

    def __init__(self):
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, **kwargs):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}

    async def send_image(self, chat_id, image_url, caption=None, **kwargs):
        self.sent.append(("image", image_url))
        return SendResult(success=True)

    async def send_animation(self, chat_id, animation_url, caption=None, **kwargs):
        self.sent.append(("animation", animation_url))
        return SendResult(success=True)

    async def send_image_file(self, chat_id, image_path, caption=None, **kwargs):
        self.sent.append(("file", image_path))
        return SendResult(success=True)


@pytest.mark.asyncio
async def test_base_adapter_falls_back_to_per_image_send():
    adapter = StubAdapter()

    await adapter.send_multiple_images(
        "chat",
        [("https://x/a.png", ""), ("file:///tmp/b.png", ""), ("https://x/c.gif", "")],
    )

    assert [kind for kind, _ in adapter.sent] == ["image", "file", "animation"]


@pytest.mark.asyncio
async def test_telegram_chunks_media_groups_at_ten(monkeypatch):
    import telegram

    monkeypatch.setattr(
        telegram,
        "InputMediaPhoto",
        MagicMock(side_effect=lambda media, caption=None: {"media": media}),
    )
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_media_group = AsyncMock(
        return_value=[MagicMock(message_id=1)]
    )

    await adapter.send_multiple_images(
        "12345", [(f"https://x/{index}.png", "") for index in range(15)]
    )

    assert [len(call.kwargs["media"]) for call in adapter._bot.send_media_group.await_args_list] == [10, 5]


@pytest.mark.asyncio
async def test_mattermost_uploads_files_in_one_post(tmp_path):
    adapter = object.__new__(MattermostAdapter)
    adapter._base_url = "https://mm.example.com"
    adapter._token = "fake"
    adapter._session = MagicMock()
    adapter._reply_mode = "thread"
    adapter._api_post = AsyncMock(return_value={"id": "post123"})
    adapter._upload_file = AsyncMock(side_effect=["one", "two"])
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"\x89PNG")
        paths.append(path)

    await adapter.send_multiple_images(
        "channel", [(f"file://{path}", "") for path in paths]
    )

    payload = adapter._api_post.await_args.args[1]
    assert payload["file_ids"] == ["one", "two"]
