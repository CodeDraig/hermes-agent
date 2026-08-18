"""Tests for /restart notification — the gateway notifies the requester on comeback."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ── restart marker helpers ───────────────────────────────────────────────


def test_planned_restart_notification_pending_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    marker = tmp_path / ".restart_pending.json"

    assert gateway_run._planned_restart_notification_pending() is False
    marker.write_text("{}")
    assert gateway_run._planned_restart_notification_pending() is True

    gateway_run._clear_planned_restart_notification()

    assert gateway_run._planned_restart_notification_pending() is False


# ── _handle_restart_command writes .restart_notify.json ──────────────────


@pytest.mark.asyncio
async def test_restart_command_writes_notify_file(tmp_path, monkeypatch):
    """When /restart fires, the requester's routing info is persisted to disk."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    result = await runner._handle_restart_command(event)
    assert "Restarting" in result

    notify_path = tmp_path / ".restart_notify.json"
    assert notify_path.exists()
    data = json.loads(notify_path.read_text())
    assert data["platform"] == "telegram"
    assert data["chat_id"] == "42"
    assert data["chat_type"] == "dm"
    assert data["message_id"] == "m1"
    assert "thread_id" not in data  # no thread → omitted


@pytest.mark.asyncio
async def test_restart_command_uses_atomic_json_writes_for_marker_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    calls = []

    def _fake_atomic_json_write(path, payload, **kwargs):
        calls.append((Path(path).name, payload, kwargs))

    # _handle_restart_command lives in gateway/slash_commands.py (extracted from
    # run.py); it uses that module's top-level atomic_json_write import.
    import gateway.slash_commands as gateway_slash
    monkeypatch.setattr(gateway_slash, "atomic_json_write", _fake_atomic_json_write)
    monkeypatch.setattr(gateway_run, "atomic_json_write", _fake_atomic_json_write)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    await runner._handle_restart_command(event)

    names = [name for name, _payload, _kwargs in calls]
    assert names == [".restart_notify.json", ".restart_last_processed.json"]
    assert calls[0][1]["chat_id"] == "42"
    assert calls[1][1]["platform"] == "telegram"


@pytest.mark.asyncio
async def test_sethome_updates_running_config_for_same_process_restart(tmp_path, monkeypatch):
    """/sethome persists to env and updates in-memory config before restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    saved = {}

    def _fake_save_env_value(key, value):
        saved[key] = value

    monkeypatch.setattr("hermes_cli.config.save_env_value", _fake_save_env_value)
    monkeypatch.setattr("gateway.slash_commands.persist_home_channel", lambda home, **kwargs: None)

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="home-42")
    source.chat_name = "Ops Home"
    event = MessageEvent(
        text="/sethome",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-home",
    )

    result = await runner._handle_set_home_command(event)

    home = runner.config.get_home_channel(Platform.TELEGRAM)
    assert "Home channel set" in result
    assert saved["TELEGRAM_HOME_CHANNEL"] == "home-42"
    assert home is not None
    assert home.chat_id == "home-42"
    assert home.name == "Ops Home"


@pytest.mark.asyncio
async def test_sethome_preserves_thread_target_for_same_process_restart(tmp_path, monkeypatch):
    """/sethome from a topic/thread stores the thread-aware home target."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    saved = {}

    def _fake_save_env_value(key, value):
        saved[key] = value

    monkeypatch.setattr("hermes_cli.config.save_env_value", _fake_save_env_value)
    monkeypatch.setattr("gateway.slash_commands.persist_home_channel", lambda home, **kwargs: None)

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="parent-42", thread_id="topic-7")
    source.chat_name = "Ops Topic"
    event = MessageEvent(
        text="/sethome",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-home-thread",
    )

    result = await runner._handle_set_home_command(event)

    home = runner.config.get_home_channel(Platform.TELEGRAM)
    assert "Home channel set" in result
    assert saved["TELEGRAM_HOME_CHANNEL"] == "parent-42"
    assert saved["TELEGRAM_HOME_CHANNEL_THREAD_ID"] == "topic-7"
    assert home is not None
    assert home.chat_id == "parent-42"
    assert home.thread_id == "topic-7"


# ── home-channel startup notifications ─────────────────────────────────────


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_preserves_thread_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="parent-42",
        name="Ops Topic",
        thread_id="777",
    )
    # Declare the DM-topic lookup on the adapter CLASS, not the instance.
    # _is_telegram_dm_topic_target resolves _get_dm_topic_info via type(adapter)
    # so a MagicMock auto-attribute (instance-level) is intentionally ignored;
    # a real adapter exposes the method on its class. Mirrors the fake-adapter
    # pattern in test_telegram_topic_mode.py.
    class _DmTopicAdapter(type(adapter)):
        def _get_dm_topic_info(self, chat_id, thread_id):
            return {"name": "Ops Topic"}

    adapter.__class__ = _DmTopicAdapter
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "parent-42", "777")}
    adapter.send.assert_called_once_with(
        "parent-42",
        "♻️ Gateway online — Hermes is back and ready.",
        metadata={
            "thread_id": "777",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "777",
        },
    )


# ── _send_restart_notification ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_restart_notification_logs_warning_on_sendresult_failure(
    tmp_path, monkeypatch, caplog
):
    """Adapter that returns SendResult(success=False) must log a WARNING, not INFO.

    Regression guard: adapter.send() catches provider errors (e.g. Telegram
    "Chat not found") and returns SendResult(success=False) rather than
    raising. The caller previously ignored the return value and always
    logged "Sent restart notification to ..." at INFO — masking real
    delivery failures behind a fake success line.
    """
    from gateway.platforms.base import SendResult

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(
        return_value=SendResult(success=False, error="Chat not found"),
    )

    with caplog.at_level("DEBUG", logger="gateway.run"):
        delivered_target = await runner._send_restart_notification()

    success_lines = [
        r for r in caplog.records
        if r.levelname == "INFO" and "Sent restart notification" in r.getMessage()
    ]
    warning_lines = [
        r for r in caplog.records
        if r.levelname == "WARNING"
        and "was not delivered" in r.getMessage()
        and "Chat not found" in r.getMessage()
    ]
    assert delivered_target is None
    assert not success_lines, (
        "Expected no INFO 'Sent restart notification' line when send failed, "
        f"got: {[r.getMessage() for r in success_lines]}"
    )
    assert warning_lines, (
        "Expected a WARNING line mentioning the failure; "
        f"got records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    # Still cleans up.
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_logs_info_on_sendresult_success(
    tmp_path, monkeypatch, caplog
):
    """Adapter returning SendResult(success=True) keeps the INFO log line."""
    from gateway.platforms.base import SendResult

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m-1"))

    with caplog.at_level("DEBUG", logger="gateway.run"):
        delivered_target = await runner._send_restart_notification()

    success_lines = [
        r for r in caplog.records
        if r.levelname == "INFO" and "Sent restart notification" in r.getMessage()
    ]
    assert delivered_target == ("telegram", "42", None)
    assert success_lines, (
        "Expected INFO 'Sent restart notification' when send succeeded; "
        f"got records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_shutdown_notifications_use_cached_live_thread_source_when_origin_missing():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="parent-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=None)
    runner._cache_session_source(session_key, source)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="shutdown"))

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once_with(
        "parent-42",
        "⚠️ Gateway shutting down — Your current task will be interrupted.",
        metadata={"thread_id": "topic-7"},
    )


@pytest.mark.asyncio
async def test_shutdown_notifications_are_fully_muted_when_flag_disabled():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)
    adapter.send = AsyncMock()

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_not_awaited()

