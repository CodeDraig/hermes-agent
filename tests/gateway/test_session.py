"""Retained gateway session contracts."""

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import (
    SessionSource,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
)


def test_session_source_round_trip_preserves_retained_routing():
    source = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="channel",
        chat_name="engineering",
        chat_type="thread",
        user_id="user",
        thread_id="root-post",
        profile="work",
    )

    restored = SessionSource.from_dict(source.to_dict())

    assert restored == source


def test_telegram_topic_session_key_is_stable():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        thread_id="42",
        user_id="7",
    )

    assert build_session_key(source) == "agent:main:telegram:group:-100123:42"


def test_mattermost_dm_sessions_include_conversation_id():
    first = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="dm-one",
        chat_type="dm",
        user_id="user",
    )
    second = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="dm-two",
        chat_type="dm",
        user_id="user",
    )

    assert build_session_key(first) != build_session_key(second)


def test_context_prompt_names_retained_platform_and_home():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="123",
    )
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="123",
                    name="Home",
                ),
            )
        }
    )
    context = build_session_context(source, config)

    prompt = build_session_context_prompt(context)

    assert "Telegram" in prompt
    assert "Home" in prompt
