"""Session-owned phase of AIAgent initialization."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("run_agent")


def initialize_session(
    agent,
    *,
    session_id,
    checkpoints_enabled,
    checkpoint_max_snapshots,
    checkpoint_max_total_size_mb,
    checkpoint_max_file_size_mb,
    session_db,
    parent_session_id,
    reasoning_config,
    max_tokens,
):
    # Session logging setup - auto-save conversation trajectories for debugging
    agent.session_start = datetime.now()
    if session_id:
        # Use provided session ID (e.g., from CLI)
        agent.session_id = session_id
    else:
        # Generate a new session ID
        timestamp_str = agent.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        agent.session_id = f"{timestamp_str}_{short_uuid}"

    # Expose session ID to tools (terminal, execute_code) so agents can
    # reference their own session for --resume commands, cross-session
    # coordination, and logging. Keep the ContextVar and os.environ
    # fallback synchronized because different tool paths still read both.
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(agent.session_id)
    except Exception:
        # Preserve the root-agent legacy fallback, but never let delegated
        # construction publish a child ID process-wide even if the ContextVar
        # bridge itself failed to import.
        try:
            from agent.delegation_context import is_delegated_child_context

            delegated_child = is_delegated_child_context()
        except Exception:
            delegated_child = False
        if not delegated_child:
            os.environ["HERMES_SESSION_ID"] = agent.session_id

    # Session logs go into ~/.hermes/sessions/ alongside gateway sessions
    hermes_home = get_hermes_home()
    agent.logs_dir = hermes_home / "sessions"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    # Per-session JSON snapshot writer (~/.hermes/sessions/session_{sid}.json)
    # is opt-in via sessions.write_json_snapshots (default False).  state.db
    # is canonical — the snapshot is only useful for external tooling that
    # reads the JSON files directly.  See run_agent._save_session_log.
    agent._session_json_enabled = False
    try:
        from hermes_cli.config import load_config_readonly as _load_sess_cfg
        _sess_cfg = (_load_sess_cfg().get("sessions") or {})
        agent._session_json_enabled = bool(_sess_cfg.get("write_json_snapshots", False))
    except Exception:
        pass
    # logs_dir is retained unconditionally for request_dump_*.json (debug
    # breadcrumb path written by agent_runtime_helpers.dump_api_request_debug).

    # Track conversation messages for session logging
    agent._session_messages: List[Dict[str, Any]] = []
    # Responses encrypted reasoning replay state.  Some OpenAI-compatible
    # routes accept GPT-5 Responses requests but later reject replayed
    # encrypted reasoning blobs (HTTP 400 ``invalid_encrypted_content``).
    # When that happens we disable replay for the rest of the session and
    # fall back to stateless continuity.  See
    # agent/conversation_loop.py's invalid_encrypted_content retry branch.
    agent._codex_reasoning_replay_enabled = True
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"

    # Cached system prompt -- built once per session, only rebuilt on compression
    agent._cached_system_prompt: Optional[str] = None
    # Cross-session-stable prefix of the cached prompt. It remains separate
    # from the persisted string and is used only to place an early cache marker.
    agent._cached_system_prompt_static: Optional[str] = None

    # Filesystem checkpoint manager (transparent — not a tool)
    from tools.checkpoint_manager import CheckpointManager
    agent._checkpoint_mgr = CheckpointManager(
        enabled=checkpoints_enabled,
        max_snapshots=checkpoint_max_snapshots,
        max_total_size_mb=checkpoint_max_total_size_mb,
        max_file_size_mb=checkpoint_max_file_size_mb,
    )

    # SQLite session store (optional -- provided by CLI or gateway)
    agent._session_db = session_db
    # Whether close() must also close that handle. Default False: a
    # caller-supplied session_db is almost always the SHARED launch handle,
    # which outlives every agent and must never be closed here. Callers that
    # hand over a DEDICATED handle (the gateway's per-profile state.db opens)
    # set this True at the point ownership transfers, so teardown releases the
    # sqlite fds and the token-writer thread instead of leaking them for the
    # life of the process. Also set True on the lazy self-open in
    # _get_session_db_for_recall, where nothing else holds a reference.
    agent._owns_session_db = False
    agent._parent_session_id = parent_session_id
    # A close flush and the worker's turn-start flush can overlap. The durable
    # marker is attached to each in-memory message dict, so its test-and-append
    # sequence must be serialized per agent rather than relying on SQLite alone.
    agent._session_persist_lock = threading.RLock()
    # CLI retains its just-accepted user dict until turn setup can reuse it.
    # This preserves the message-local durable marker if close persistence wins
    # the race before the agent's normal early turn flush.
    agent._pending_cli_user_message = None
    agent._last_flushed_db_idx = 0  # tracks DB-write cursor to prevent duplicate writes
    agent._session_db_created = False  # DB row deferred to run_conversation()
    # Most agents own their session row and should finalize it on close().
    # Some temporary helper agents (manual compression / session-hygiene /
    # background-review forks) rotate or share the session forward to a
    # continuation row that must remain open after the helper is torn down;
    # those callers explicitly set this flag to False.
    agent._end_session_on_close = True
    # When True, this agent NEVER persists to the canonical session store
    # (state.db) or the JSON snapshot, regardless of session_id. Set on the
    # background skill/memory review fork so its harness turn can't leak into
    # the user's real session and hijack the next live turn. Default False.
    agent._persist_disabled = False
    agent._session_init_model_config = {
        "max_iterations": agent.max_iterations,
        "reasoning_config": reasoning_config,
        "max_tokens": max_tokens,
    }
    # Persist a process-scoped --yolo launch into the session row so a later
    # `hermes --resume <id>` can restore the bypass (CLI resume paths read
    # model_config.yolo_mode back via SessionDB.session_yolo_enabled).
    # Session-scoped /yolo toggles persist separately through
    # SessionDB.set_session_yolo at toggle time.
    try:
        from tools.approval import _YOLO_MODE_FROZEN
        if _YOLO_MODE_FROZEN:
            agent._session_init_model_config["yolo_mode"] = True
    except Exception:
        pass

    # In-memory todo list for task planning (one per agent/session)
    from tools.todo_tool import TodoStore
    agent._todo_store = TodoStore()

    # Load config once for memory, skills, and compression sections
    try:
        from hermes_cli.config import load_config_readonly as _load_agent_config
        _agent_cfg = _load_agent_config()
    except Exception:
        _agent_cfg = {}

    return _agent_cfg
