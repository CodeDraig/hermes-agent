"""Tests for /resume gateway slash command.

Tests the _handle_resume_command handler (switch to a previously-named session)
across gateway messenger platforms.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_event(text="/resume", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _session_key_for_event(event):
    """Get the session key that build_session_key produces for an event."""
    return build_session_key(event.source)


def _make_runner(session_db=None, current_session_id="current_session_001",
                 event=None):
    """Create a bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(platforms={})
    runner._voice_mode = {}
    # Gateway holds the async facade; the slash handlers await it.
    if session_db is not None:
        from hermes_state import AsyncSessionDB
        session_db = AsyncSessionDB(session_db)
    runner._session_db = session_db
    runner._running_agents = {}
    runner._is_user_authorized = lambda _source: True

    # Compute the real session key if an event is provided
    session_key = build_session_key(event.source) if event else "agent:main:telegram:dm"

    # Mock session_store that returns a session entry with a known session_id
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = current_session_id
    mock_session_entry.session_key = session_key
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    mock_store.load_transcript.return_value = []
    mock_store.switch_session.return_value = mock_session_entry
    runner.session_store = mock_store

    return runner


# ---------------------------------------------------------------------------
# _handle_resume_command
# ---------------------------------------------------------------------------


class TestHandleResumeCommand:
    """Tests for GatewayRunner._handle_resume_command."""

    @pytest.mark.asyncio
    async def test_no_session_db(self):
        """Returns error when session database is unavailable."""
        runner = _make_runner(session_db=None)
        event = _make_event(text="/resume My Project")
        result = await runner._handle_resume_command(event)
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_list_named_sessions_when_no_arg(self, tmp_path):
        """With no argument, lists recently titled sessions."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.create_session(
            "sess_002", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "Research" in result
        assert "Coding" in result
        assert "Named Sessions" in result
        assert "1." in result
        assert "2." in result
        assert "/resume 1" in result
        db.close()


    @pytest.mark.asyncio
    async def test_resume_clears_session_model_overrides(self, tmp_path):
        """Resume must not carry a previous session's /model override into the
        restored conversation, while leaving other chats' overrides intact (#10702)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        key = _session_key_for_event(event)
        runner._session_model_overrides = {
            key: {"model": "gpt-5", "provider": "openai"},
            "agent:main:telegram:dm:other": {"model": "keep-me"},
        }
        runner._pending_model_notes = {
            key: "[Note: switched to gpt-5]",
            "agent:main:telegram:dm:other": "[Note: keep-me]",
        }

        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        # The resumed chat's override + pending note are cleared...
        assert key not in runner._session_model_overrides
        assert key not in runner._pending_model_notes
        # ...but an unrelated chat's state is untouched.
        assert runner._session_model_overrides["agent:main:telegram:dm:other"] == {"model": "keep-me"}
        assert runner._pending_model_notes["agent:main:telegram:dm:other"] == "[Note: keep-me]"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_clears_last_resolved_model(self, tmp_path):
        """Resume must also clear the resumed chat's cached last-resolved
        model, so the restored conversation re-resolves from current config
        instead of a value cached before the switch (mirrors /new and the
        compression-exhausted auto-reset, #58403), while leaving other
        chats' cache entries intact."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        key = _session_key_for_event(event)
        runner._last_resolved_model = {
            key: "gpt-5",
            "agent:main:telegram:dm:other": "keep-me",
        }

        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        assert key not in runner._last_resolved_model
        assert runner._last_resolved_model["agent:main:telegram:dm:other"] == "keep-me"
        db.close()


    @pytest.mark.asyncio
    async def test_resume_follows_compression_continuation(self, tmp_path):
        """Gateway /resume should reopen the live descendant after compression."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("compressed_root", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("compressed_root", "Compressed Work")
        db.end_session("compressed_root", "compression")
        db.create_session("compressed_child", "telegram", user_id="12345", chat_id="67890", parent_session_id="compressed_root")
        db.append_message("compressed_child", "user", "hello from continuation")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Compressed Work")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        runner.session_store.load_transcript.side_effect = (
            lambda session_id: [{"role": "user", "content": "hello from continuation"}]
            if session_id == "compressed_child"
            else []
        )

        result = await runner._handle_resume_command(event)

        assert "Resumed session" in result
        assert "(1 message)" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "compressed_child"
        runner.session_store.load_transcript.assert_called_with("compressed_child")
        db.close()


    @pytest.mark.asyncio
    async def test_resume_evicts_cached_agent(self, tmp_path):
        """Gateway /resume evicts the cached AIAgent so the next message
        rebuilds with the correct session_id end-to-end — mirrors /branch
        and /reset. Without this, the cached agent's memory provider keeps
        writing into the wrong session. See #6672.
        """
        import threading
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        # Seed the cache with a fake agent
        real_key = _session_key_for_event(event)
        runner._agent_cache = {real_key: (MagicMock(), object())}
        runner._agent_cache_lock = threading.RLock()

        await runner._handle_resume_command(event)

        assert real_key not in runner._agent_cache
        db.close()


    @pytest.mark.asyncio
    async def test_bare_resume_lists_exact_lane_before_limit(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        lane_key = _session_key_for_event(event)
        for i in range(3):
            sid = f"lane_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Lane Work {i}")
        for i in range(12):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert "Lane Work 0" in result
        assert "Lane Work 1" in result
        assert "Lane Work 2" in result
        assert "Foreign Work" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_bare_resume_admin_all_preserves_same_platform_widening(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume --all")
        db.create_session(
            "other_lane", "telegram",
            session_key="agent:main:telegram:dm:other",
            user_id="other-user", chat_id="other",
        )
        db.set_session_title("other_lane", "Other Lane Work")

        runner = _make_runner(session_db=db, event=event)
        runner._resume_caller_is_admin = lambda _source: True
        result = await runner._handle_resume_command(event)

        assert "Other Lane Work" in result
        db.close()

    @pytest.mark.asyncio
    async def test_numeric_resume_fallback_uses_exact_lane_candidates(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume 2")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "lane_older", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("lane_older", "Lane Older")
        db.create_session(
            "lane_newer", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("lane_newer", "Lane Newer")
        for i in range(12):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")
        db.create_session(
            "current_session_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )

        runner = _make_runner(
            session_db=db, current_session_id="current_session_001", event=event
        )
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        runner.session_store.switch_session.assert_called_once()
        assert runner.session_store.switch_session.call_args[0][1] == "lane_older"
        db.close()

    @pytest.mark.asyncio
    async def test_bare_resume_normalizes_telegram_lobby_source_to_bound_topic(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        topic_source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="topic-42",
        )
        topic_key = build_session_key(topic_source)
        db.create_session(
            "topic_session", "telegram", session_key=topic_key,
            user_id="12345", chat_id="67890", chat_type="dm",
            thread_id="topic-42",
        )
        db.set_session_title("topic_session", "Recovered Topic Work")
        db.enable_telegram_topic_mode(chat_id="67890", user_id="12345")
        db.bind_telegram_topic(
            chat_id="67890",
            thread_id="topic-42",
            user_id="12345",
            session_key=topic_key,
            session_id="topic_session",
        )
        lobby_key = _session_key_for_event(event)
        db.create_session(
            "lobby_session", "telegram", session_key=lobby_key,
            user_id="12345", chat_id="67890", chat_type="dm",
        )
        db.set_session_title("lobby_session", "Lobby Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert "Recovered Topic Work" in result
        assert "Lobby Work" not in result
        db.close()


class TestHandleSessionsCommand:
    """Tests for GatewayRunner._handle_sessions_command."""

    @pytest.mark.asyncio
    async def test_sessions_full_keeps_legacy_reset_child_after_parent_resume(
        self, tmp_path
    ):
        import json

        from gateway.config import GatewayConfig
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import AsyncSessionDB

        event = _make_event(text="/sessions full")
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(),
        )
        db = store._db
        assert db is not None

        root = store.get_or_create_session(event.source)
        root_id = root.session_id
        db.set_session_title(root_id, "Legacy reset parent")
        child = store.reset_session(root.session_key)
        assert child is not None
        child_id = child.session_id
        db.set_session_title(child_id, "Legacy reset child")
        # Reproduce the on-disk shape from before _reset_from existed.
        db._conn.execute(
            "UPDATE sessions SET model_config = NULL WHERE id = ?",
            (child_id,),
        )
        db._conn.commit()

        runner = _make_runner(session_db=None, event=event)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = AsyncSessionDB(db)

        before_resume = await runner._handle_sessions_command(event)
        assert "Legacy reset parent" in before_resume

        switched = store.switch_session(root.session_key, root_id)
        assert switched is not None
        after_resume = await runner._handle_sessions_command(event)

        assert "Legacy reset child" in after_resume
        assert "Legacy reset parent" not in after_resume
        child_row = db.get_session(child_id)
        assert child_row is not None
        assert json.loads(child_row["model_config"])["_reset_from"] == root_id
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_full_lists_conversations_created_by_gateway_resets(
        self, tmp_path
    ):
        import json

        from gateway.config import GatewayConfig
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import AsyncSessionDB

        event = _make_event(text="/sessions full")
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(),
        )
        db = store._db
        assert db is not None

        entry = store.get_or_create_session(event.source)
        db.set_session_title(entry.session_id, "Greeting via Telegram")
        for title in (
            "Store memories with priority",
            "Extract AI news to Telegram",
            "Current Telegram work",
        ):
            previous_id = entry.session_id
            entry = store.reset_session(entry.session_key)
            assert entry is not None
            db.set_session_title(entry.session_id, title)
            reset_row = db.get_session(entry.session_id)
            assert reset_row is not None
            assert json.loads(reset_row["model_config"])["_reset_from"] == previous_id

        # The gateway creates the identity row before the agent exists. Its
        # first-turn create_session upsert must enrich the marker-only config,
        # while later bare/retry upserts must not replace the established data.
        db.create_session(
            entry.session_id,
            "telegram",
            model_config={"max_iterations": 60},
        )
        enriched = json.loads(db.get_session(entry.session_id)["model_config"])
        assert enriched == {
            "max_iterations": 60,
            "_reset_from": previous_id,
        }
        db.create_session(
            entry.session_id,
            "telegram",
            model_config={"max_iterations": 999},
        )
        assert json.loads(db.get_session(entry.session_id)["model_config"]) == enriched

        runner = _make_runner(session_db=None, event=event)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = AsyncSessionDB(db)

        result = await runner._handle_sessions_command(event)

        assert "Greeting via Telegram" in result
        assert "Store memories with priority" in result
        assert "Extract AI news to Telegram" in result
        assert "Current Telegram work" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_busy_platform_lists_exact_lane_and_excludes_current_tip(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        for i in range(11):
            sid = f"lane_root_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Lane Work {i}")

        db.create_session(
            "current_root", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("current_root", "Current compressed root")
        db.end_session("current_root", "compression")
        db.create_session(
            "current_tip", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890", parent_session_id="current_root",
        )
        db.set_session_title("current_tip", "Current compressed tip")

        for i in range(60):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")

        runner = _make_runner(
            session_db=db, current_session_id="current_tip", event=event
        )
        result = await runner._handle_sessions_command(event)

        assert result.count("Lane Work") == 10
        assert "Lane Work 1" in result
        assert "Lane Work 0" not in result
        assert "Foreign Work" not in result
        assert "current_tip" not in result
        assert "current_root" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_admin_all_preserves_cross_origin_widening(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions all")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "tg_named", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("tg_named", "Telegram Work")
        db.create_session(
            "mattermost_named", "mattermost",
            session_key="agent:main:mattermost:dm:other",
            user_id="other-user", chat_id="other",
        )
        db.set_session_title("mattermost_named", "Mattermost Work")

        runner = _make_runner(session_db=db, event=event)
        runner._resume_caller_is_admin = lambda _source: True
        result = await runner._handle_sessions_command(event)

        assert "Telegram Work" in result
        assert "Mattermost Work" in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_normalizes_telegram_lobby_source_to_bound_topic(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        topic_source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="topic-42",
        )
        topic_key = build_session_key(topic_source)
        db.create_session(
            "topic_session", "telegram", session_key=topic_key,
            user_id="12345", chat_id="67890", chat_type="dm",
            thread_id="topic-42",
        )
        db.set_session_title("topic_session", "Recovered Topic Work")
        db.enable_telegram_topic_mode(chat_id="67890", user_id="12345")
        db.bind_telegram_topic(
            chat_id="67890",
            thread_id="topic-42",
            user_id="12345",
            session_key=topic_key,
            session_id="topic_session",
        )
        lobby_key = _session_key_for_event(event)
        db.create_session(
            "lobby_session", "telegram", session_key=lobby_key,
            user_id="12345", chat_id="67890", chat_type="dm",
        )
        db.set_session_title("lobby_session", "Lobby Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Recovered Topic Work" in result
        assert "Lobby Work" not in result
        db.close()



    @pytest.mark.asyncio
    async def test_sessions_all_does_not_leak_cross_origin_for_non_admin(self, tmp_path):
        """`/sessions all` from a non-admin caller must stay scoped to the
        caller's own origin — it must NOT enumerate other origins' sessions
        (the enumeration half of the /resume IDOR). Cross-origin listing is
        gated behind an explicitly-configured admin, which the default test
        config is not."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions all full")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "tg_named", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("tg_named", "Telegram Work")
        db.create_session("mattermost_unnamed", "mattermost")  # other origin
        db.append_message("mattermost_unnamed", "user", "mattermost first prompt")

        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        # Caller's own Telegram session is shown; the cross-origin Mattermost
        # session is NOT leaked even with `all`.
        assert "Telegram Work" in result
        assert "mattermost_unnamed" not in result
        assert "Mattermost" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_search_finds_older_titled_session(self, tmp_path):
        """`/sessions search <query>` matches titles beyond the recent-10 list
        and orders by activity, keeping the caller's own scope."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions search an94")
        lane_key = _session_key_for_event(event)
        # Bury the target under newer sessions so a plain listing misses it.
        db.create_session(
            "target_an94", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("target_an94", "AN-94 Prestige Barrel Build #2")
        for i in range(12):
            sid = f"filler_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Filler {i}")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "AN-94 Prestige Barrel Build #2" in result
        assert "target_an94" in result
        assert "Filler" not in result
        db.close()


    @pytest.mark.asyncio
    async def test_sessions_search_does_not_leak_other_users_sessions(self, tmp_path):
        """Search results honor the same owner-scoping guard as listing —
        a matching title owned by a different user/chat must not surface."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions search an94")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "mine", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("mine", "AN-94 mine")
        db.create_session(
            "theirs", "telegram",
            session_key="agent:main:telegram:dm:55555",
            user_id="99999", chat_id="55555",
        )
        db.set_session_title("theirs", "AN-94 someone else's secret")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "AN-94 mine" in result
        assert "theirs" not in result
        assert "secret" not in result
        db.close()


    @pytest.mark.asyncio
    async def test_gateway_dispatches_sessions_command(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("tg_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("tg_session", "Telegram Work")

        event = _make_event(text="/sessions")
        runner = _make_runner(session_db=db, event=event)
        runner._handle_sessions_command = AsyncMock(return_value="sessions output")

        result = await runner._handle_message(event)

        assert result == "sessions output"
        runner._handle_sessions_command.assert_awaited_once_with(event)
        db.close()


class TestSameOriginChatGroupScoping:
    """Live group sessions are per-user by default (group_sessions_per_user=True),
    so a co-member must not be able to resume another member's live group session
    via the live-origin branch of _resume_target_allowed (IDOR)."""

    @staticmethod
    def _src(user_id, *, chat_type="group", chat_id="guild-123",
             platform=Platform.TELEGRAM, user_id_alt=None, thread_id=None):
        return SessionSource(platform=platform, chat_id=chat_id,
                             chat_type=chat_type, user_id=user_id,
                             user_id_alt=user_id_alt, thread_id=thread_id)


    def test_dm_cross_user_blocked_without_chat_id(self):
        # No-chat_id DM: build_session_key falls back to the participant id
        # (user_id_alt or user_id), so two different participants are different
        # origins and must not match. (With a chat_id present the DM key IS the
        # chat_id — see test_dm_same_chat_id_is_same_origin.)
        runner = _make_runner()
        a = self._src("alice", chat_type="dm", chat_id=None)
        b = self._src("bob", chat_type="dm", chat_id=None)
        assert runner._same_origin_chat(a, b) is False


    @pytest.mark.asyncio
    async def test_resume_target_allowed_blocks_cross_user_live_group(self):
        """End-to-end via the live-origin branch: Alice cannot resume Bob's
        active group session in the same chat."""
        runner = _make_runner()
        bob = self._src("bob")
        runner._gateway_session_origin_for_id = lambda sid: bob
        assert await runner._resume_target_allowed(
            self._src("alice"), "bobs_live_sid", allow_override=False
        ) is False

    # --- thread scoping: thread_id is part of the session key, so a session in
    # one thread must never match a caller in another thread of the same chat,
    # even when threads are shared among participants by default. ---


    def test_allows_same_thread_shared_participants(self):
        """Threads are shared by default (thread_sessions_per_user=False), so
        co-members in the SAME thread share the session."""
        runner = _make_runner()
        a = self._src("alice", thread_id="thread-A")
        b = self._src("bob", thread_id="thread-A")
        assert runner._same_origin_chat(a, b) is True


    def test_blocks_thread_vs_no_thread(self):
        """A threaded origin must not match a non-threaded caller in the same
        parent chat (and vice versa)."""
        runner = _make_runner()
        threaded = self._src("alice", thread_id="thread-A")
        parent = self._src("alice", thread_id=None)
        assert runner._same_origin_chat(parent, threaded) is False
        assert runner._same_origin_chat(threaded, parent) is False
