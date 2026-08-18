"""Target parsing retained for Telegram and Mattermost."""

from unittest.mock import patch

from tools.send_message_tool import resolve_send_target


def test_telegram_username_target():
    chat_id, thread_id, error = resolve_send_target("telegram", "@HermesBot")
    assert chat_id == "@HermesBot"
    assert thread_id is None
    assert error is None


def test_mattermost_id_is_explicit():
    assert resolve_send_target("mattermost", "channel_abc") == (
        "channel_abc", None, None
    )


def test_telegram_name_resolves_from_directory():
    with patch("gateway.channel_directory.resolve_channel_name", return_value="-10042"):
        assert resolve_send_target("telegram", "ops") == ("-10042", None, None)


def test_removed_platform_is_unknown():
    assert resolve_send_target("unsupported", "C123") == (
        None, None, "Unknown platform: unsupported"
    )
