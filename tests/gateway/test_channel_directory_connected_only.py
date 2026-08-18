"""Session-based channel discovery must not resurrect disconnected platforms.

Surgical reapply of the directory portion of PR #25959: historical session
origins for platforms with no connected adapter must not become active
send_message targets."""

import asyncio
from unittest.mock import patch

from gateway.channel_directory import build_channel_directory
from gateway.platforms.base import Platform


def test_does_not_resurrect_disconnected_platforms_from_session_history(tmp_path, monkeypatch):
    cache_file = tmp_path / "channel_directory.json"

    calls = []

    def fake_build_from_sessions(plat_name):
        calls.append(plat_name)
        return {"channels": [{"id": "1", "name": "old"}]}

    with patch("gateway.channel_directory._build_from_sessions", side_effect=fake_build_from_sessions), \
         patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
        # Only Telegram is connected; Mattermost is not.
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: object()}))

    plats = directory["platforms"]
    assert "telegram" in plats
    # Disconnected platforms must not appear via session discovery.
    assert "mattermost" not in plats
    assert set(calls) <= {"telegram"}

