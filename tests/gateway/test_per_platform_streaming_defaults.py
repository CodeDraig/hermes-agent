"""Per-platform streaming defaults.

Streaming is smooth on Telegram (native sendMessageDraft) but flickers on
edit-only platforms like Mattermost (repeated edits). The shipped
defaults encode that: display.platforms.telegram.streaming=true,
.mattermost.streaming=false. These are gap-fillers (user values win via
deep-merge).
"""

from __future__ import annotations


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"]["streaming"] is True
    assert plats["mattermost"]["streaming"] is False

