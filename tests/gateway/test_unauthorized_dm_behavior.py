from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "MATTERMOST_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "MATTERMOST_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(
    platform: Platform,
    user_id: str,
    chat_id: str,
    *,
    profile: str | None = None,
) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type="dm",
            profile=profile,
        ),
    )


def _make_runner(platform: Platform, config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    # Attributes required by _handle_message for the authorized-user path
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner, adapter






def test_telegram_group_users_legacy_chat_ids_still_authorize(monkeypatch):
    """Backward-compat: PR #15027 shipped TELEGRAM_GROUP_ALLOWED_USERS as a
    chat-ID allowlist. PR #17686 renamed it to sender IDs and added
    TELEGRAM_GROUP_ALLOWED_CHATS. Users on the old guidance must keep working:
    chat-ID-shaped values (starting with "-") in the _USERS var are honored as
    chat IDs with a deprecation warning.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="forum",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_users_mixed_sender_and_legacy_chat(monkeypatch):
    """Mixed values: positive user ID gates senders; negative chat ID gates chat."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "999,-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    # Legacy chat ID path: any sender in the listed chat is authorized
    legacy_chat_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="group",
    )
    assert runner._is_user_authorized(legacy_chat_source) is True

    # Sender path: listed sender user ID authorized in any group
    sender_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1009999999999",
        user_name="tester",
        chat_type="group",
    )
    assert runner._is_user_authorized(sender_source) is True


@pytest.mark.asyncio
async def test_unauthorized_dm_pairs_by_default(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)
    runner.pairing_store.generate_code.return_value = "ABC12DEF"

    result = await runner._handle_message(
        _make_event(
            Platform.TELEGRAM,
            "999999999",
            "999999999",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_called_once_with(
        "telegram",
        "999999999",
        "tester",
    )
    adapter.send.assert_awaited_once()
    assert "ABC12DEF" in adapter.send.await_args.args[1]


@pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Allowlist-configured platforms default to "ignore" for unauthorized users
# (#9337: Signal gateway sends pairing spam when allowlist is configured)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_telegram_with_allowlist_ignores_unauthorized_dm(monkeypatch):
    """Same behavior for Telegram: allowlist ⟹ ignore unauthorized DMs."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111111111")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    result = await runner._handle_message(
        _make_event(Platform.TELEGRAM, "999999999", "999999999")
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_allowlist_ignores_unauthorized_dm(monkeypatch):
    """GATEWAY_ALLOWED_USERS also triggers the 'ignore' behavior."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "111111111")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    result = await runner._handle_message(
        _make_event(Platform.TELEGRAM, "+15559999999", "+15559999999")
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


def test_allowlist_authorized_user_returns_ignore_for_unauthorized(monkeypatch):
    """_get_unauthorized_dm_behavior returns 'ignore' when allowlist is set.

    We test the resolver directly.  The full _handle_message path for
    authorized users is covered by the integration tests in this module.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111111111")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, _adapter = _make_runner(Platform.TELEGRAM, config)

    behavior = runner._get_unauthorized_dm_behavior(Platform.TELEGRAM)
    assert behavior == "ignore"


def test_get_unauthorized_dm_behavior_no_allowlist_returns_pair(monkeypatch):
    """Without any allowlist, 'pair' is still the default."""
    _clear_auth_env(monkeypatch)

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, _adapter = _make_runner(Platform.TELEGRAM, config)

    behavior = runner._get_unauthorized_dm_behavior(Platform.TELEGRAM)
    assert behavior == "pair"
