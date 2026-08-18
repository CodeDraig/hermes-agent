"""Channel-directory behavior shared by retained messaging adapters."""

import asyncio
import json
from unittest.mock import patch

from gateway.channel_directory import (
    build_channel_directory,
    format_directory_for_display,
    load_directory,
    lookup_channel_type,
    resolve_channel_name,
)
from gateway.config import Platform


def test_missing_directory_is_empty(tmp_path):
    with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "missing.json"):
        assert load_directory() == {"updated_at": None, "platforms": {}}


def test_directory_collects_adapter_channels(tmp_path):
    class Adapter:
        async def list_channels(self):
            return [{"id": "channel-1", "name": "Engineering", "type": "channel"}]

    path = tmp_path / "channels.json"
    with patch("gateway.channel_directory.DIRECTORY_PATH", path):
        result = asyncio.run(
            build_channel_directory({Platform.MATTERMOST: Adapter()})
        )

    assert result["platforms"]["mattermost"][0]["id"] == "channel-1"


def test_resolve_and_lookup_use_cached_directory(tmp_path):
    path = tmp_path / "channels.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "now",
                "platforms": {
                    "telegram": [
                        {"id": "-100", "name": "Family", "type": "group"}
                    ]
                },
            }
        )
    )
    with patch("gateway.channel_directory.DIRECTORY_PATH", path):
        assert resolve_channel_name("telegram", "Family") == "-100"
        assert lookup_channel_type("telegram", "-100") == "group"
        assert "Family" in format_directory_for_display()
