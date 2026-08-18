"""Shared end-to-end gateway fixtures for Telegram and Mattermost."""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.mattermost import MattermostAdapter
from gateway.platforms.telegram.adapter import TelegramAdapter
from gateway.session import SessionEntry, SessionSource, build_session_key


def make_source(platform: Platform, chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1", chat_type: str = "dm") -> SessionSource:
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id, user_name="e2e_tester", chat_type=chat_type)


def make_session_entry(platform: Platform, source: SessionSource | None = None):
    source = source or make_source(platform)
    return SessionEntry(session_key=build_session_key(source), session_id=f"sess-{uuid.uuid4().hex[:8]}", created_at=datetime.now(), updated_at=datetime.now(), platform=platform, chat_type="dm")


def make_event(platform: Platform, text: str = "/help", **kwargs) -> MessageEvent:
    return MessageEvent(text=text, source=make_source(platform, **kwargs), message_id=f"msg-{uuid.uuid4().hex[:8]}")


def make_runner(platform: Platform, session_entry=None):
    from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    from gateway.run import GatewayRunner

    session_entry = session_entry or make_session_entry(platform)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, token="e2e-test-token")})
    runner.adapters = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._running_agents_ts = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_providers = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_message_with_agent = AsyncMock(return_value="agent-handled-default")
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **kw: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._read_user_config = lambda: {"approvals": {"destructive_slash_confirm": False}}
    runner._reset_notice_session_info = lambda source: ""
    runner.pairing_store = MagicMock()
    runner.pairing_store._is_rate_limited.return_value = False
    runner.pairing_store.generate_code.return_value = "ABC123"
    return runner


def make_adapter(platform: Platform, runner=None):
    runner = runner or make_runner(platform)
    config = PlatformConfig(enabled=True, token="e2e-test-token", extra={"url": "https://mattermost.example.test"})
    adapter = TelegramAdapter(config) if platform == Platform.TELEGRAM else MattermostAdapter(config)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-response"))
    adapter.send_typing = AsyncMock()
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[platform] = adapter
    return adapter


async def send_and_capture(adapter, text: str, platform: Platform, **event_kwargs):
    event = make_event(platform, text, **event_kwargs)
    adapter.send.reset_mock()
    await adapter.handle_message(event)
    for _ in range(40):
        if adapter.send.called:
            break
        await asyncio.sleep(0.05)
    return adapter.send


@pytest.fixture(params=[Platform.TELEGRAM, Platform.MATTERMOST], ids=["telegram", "mattermost"])
def platform(request):
    return request.param


@pytest.fixture
def source(platform):
    return make_source(platform)


@pytest.fixture
def session_entry(platform, source):
    return make_session_entry(platform, source)


@pytest.fixture
def runner(platform, session_entry):
    return make_runner(platform, session_entry)


@pytest.fixture
def adapter(platform, runner):
    return make_adapter(platform, runner)
