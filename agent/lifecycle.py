"""Responsibility-owned agent lifecycle behavior."""
import copy
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from agent.codex_responses_adapter import (
    _summarize_user_message_for_log,
)
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
)
from agent.memory_provider import is_trivial_prompt
from agent.message_sanitization import get_tool_call_name
from agent.session_activity import ActivityProvenance
from agent.tool_guardrails import (
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from model_tools import handle_function_call
from tools.browser_tool import cleanup_browser
from tools.interrupt import set_interrupt as _set_interrupt
from tools.terminal_tool import cleanup_vm, get_active_env


logger = logging.getLogger(__name__)

AGENT_RUNTIME_POST_HOOK_TOOL_NAMES = frozenset(
    {"todo", "session_search", "memory", "clarify", "delegate_task"}
)

def _cleanup_task_resources(self, task_id: str) -> None:
    """Forwarder — see ``agent.chat_completion_helpers.cleanup_task_resources``."""
    from agent.chat_completion_helpers import cleanup_task_resources
    return cleanup_task_resources(self, task_id)

def _summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
    notification_mode: str = "on",
) -> List[str]:
    """Forwarder — see ``agent.background_review.summarize_background_review_actions``."""
    from agent.background_review import summarize_background_review_actions
    return summarize_background_review_actions(
        review_messages,
        prior_snapshot,
        notification_mode=notification_mode,
    )

def _spawn_background_review(
    self,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
    focus: Optional[str] = None,
) -> None:
    """Spawn the background memory/skill review thread.

    Thin wrapper — the heavy lifting lives in
    ``agent.background_review.spawn_background_review_thread`` which
    returns the thread target.  ``threading.Thread`` is constructed
    here so existing tests that patch ``run_agent.threading.Thread``
    keep working.

    ``focus`` is optional user-supplied steering (from ``/refine``)
    appended to the review prompt — e.g. "save the deploy workflow as a
    skill". The automatic post-turn triggers never set it.
    """
    # A delegation subagent (``_delegate_depth > 0``) must not run the
    # automatic post-turn review. Subagents are ephemeral workers already
    # barred from writing shared MEMORY.md (``DELEGATE_BLOCKED_TOOLS``) and
    # are spawned with ``skip_memory=True``, so a review here has little to
    # persist — yet it inherits the subagent's (often premium) delegation
    # model and replays the whole conversation at premium rates, silently
    # inflating token cost (#85859). An explicit ``/refine`` (``focus`` set)
    # is a deliberate user request and still runs.
    if focus is None and getattr(self, "_delegate_depth", 0) > 0:
        return
    # Explicit off-switch for automatic post-turn forks
    # (``auxiliary.background_review.enabled: false``). Manual ``/refine``
    # still works — same contract as zeroing the nudge intervals (#87250).
    # Load the task block once here and pass it into the spawn path so
    # aux routing does not re-read config.
    task_cfg = None
    if focus is None:
        from agent.background_review import load_background_review_settings
        enabled, task_cfg = load_background_review_settings()
        if not enabled:
            return
    from agent.background_review import spawn_background_review_thread
    from tools.thread_context import propagate_context_to_thread
    target, _prompt = spawn_background_review_thread(
        self,
        messages_snapshot,
        review_memory=review_memory,
        review_skills=review_skills,
        focus=focus,
        task_cfg=task_cfg,
    )
    # Carry the active profile into the review thread so MEMORY.md / skill
    # review writes land in the right profile (#54937).
    t = threading.Thread(
        target=propagate_context_to_thread(target), daemon=True, name="bg-review"
    )
    t.start()

def _build_memory_write_metadata(
    self,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Forwarder — see ``agent.background_review.build_memory_write_metadata``."""
    from agent.background_review import build_memory_write_metadata
    return build_memory_write_metadata(
        self,
        write_origin=write_origin,
        execution_context=execution_context,
        task_id=task_id,
        tool_call_id=tool_call_id,
    )

def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
    """Rewrite the current-turn user message before persistence/return.

    Some call paths need an API-only user-message variant without letting
    that synthetic text leak into persisted transcripts or resumed session
    history. When an override is configured for the active turn, mutate the

    in-memory messages list in place so both persistence and returned
    history stay clean.  A paired timestamp override preserves the platform
    event time as message metadata, rather than embedding it in content.
    """
    idx = getattr(self, "_persist_user_message_idx", None)
    override = getattr(self, "_persist_user_message_override", None)
    timestamp = getattr(self, "_persist_user_message_timestamp", None)
    if idx is None or (override is None and timestamp is None):
        return
    if 0 <= idx < len(messages):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            # Text-only call paths may pass a synthetic API-facing prompt
            # and a cleaner transcript string separately. Before the API
            # call, a plain-text override must not replace native image/audio
            # blocks. A list override, however, is the original clean
            # multimodal payload (for example before a queued /model note)
            # and must replace the API-local list once the turn is final.
            # Preflight compaction can re-anchor this index at a message
            # whose content was MERGED with the compaction summary
            # (merge-summary-into-tail).  That is not an accident:
            # ``reanchor_current_turn_user_idx`` falls back to the last
            # user row precisely BECAUSE the merge rewrote the content and
            # the exact-match lookup misses.  Overwriting it with the clean
            # text would drop the summary from the continuation history the
            # next turn is built from — the same hazard the DB-write twin
            # below already refuses (see the sibling guard in
            # ``_flush_messages_to_session_db_unlocked``).
            if (
                override is not None
                and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                and (
                    not isinstance(msg.get("content"), list)
                    or isinstance(override, list)
                )
            ):
                msg["content"] = override
            if timestamp is not None:
                msg["timestamp"] = timestamp

def shutdown_memory_provider(self, messages: list = None) -> None:
    """Shut down the memory provider and context engine at session end.

    Idempotent: gateway cleanup and AgentState.close() may share this
    ownership boundary.
    """
    if getattr(self, "_memory_provider_shutdown", False):
        return
    self._memory_provider_shutdown = True
    if self._memory_manager:
        try:
            self._memory_manager.on_session_end(messages or [])
        except Exception as e:
            logger.warning("Memory provider on_session_end failed during shutdown: %s", e, exc_info=True)
        try:
            self._memory_manager.shutdown_all()
        except Exception:
            pass
    # Notify context engine of session end (flush DAG, close DBs, etc.)
    if hasattr(self, "context_compressor") and self.context_compressor:
        try:
            self.context_compressor.on_session_end(
                self.session_id or "",
                messages or [],
            )
        except Exception:
            pass

def commit_memory_session(self, messages: list = None) -> None:
    """Trigger end-of-session extraction without tearing providers down.
    Called when session_id rotates (e.g. /new, context compression);

    providers keep their state and continue running under the old
    session_id — they just flush pending extraction now."""
    if self._memory_manager:
        try:
            self._memory_manager.on_session_end(messages or [])
        except Exception:
            pass
    # Notify context engine of session end too — same lifecycle moment as
    # the memory manager's on_session_end. Without this, engines that
    # accumulate per-session state (DAGs, summaries) leak that state from
    # the rotated-out session into whatever comes next under the same
    # compressor instance. Mirrors the call in shutdown_memory_provider().
    # See issue #22394.
    if hasattr(self, "context_compressor") and self.context_compressor:
        try:
            self.context_compressor.on_session_end(
                self.session_id or "",
                messages or [],
            )
        except Exception:
            pass

def _sync_external_memory_for_turn(
    self,
    *,
    original_user_message: Any,
    final_response: Any,
    interrupted: bool,
    messages: list | None = None,
) -> None:
    """Mirror a completed turn into external memory providers.

    Called at the end of ``run_conversation`` with the cleaned user
    message (``original_user_message``) and the finalised assistant
    response.  The external memory backend gets both ``sync_all`` (to
    persist the exchange) and ``queue_prefetch_all`` (to start
    warming context for the next turn) in one shot.

    Uses ``original_user_message`` rather than ``user_message``
    because the latter may carry injected skill content that bloats
    or breaks provider queries.

    Interrupted turns are skipped entirely (#15218).  A partial
    assistant output, an aborted tool chain, or a mid-stream reset
    is not durable conversational truth — mirroring it into an
    external memory backend pollutes future recall with state the
    user never saw completed.  The prefetch is gated on the same
    flag: the user's next message is almost certainly a retry of
    the same intent, and a prefetch keyed on the interrupted turn
    would fire against stale context.

    Normal completed turns still sync as before.  The whole body is
    wrapped in ``try/except Exception`` because external memory
    providers are strictly best-effort — a misconfigured or offline
    backend must not block the user from seeing their response.
    """
    if interrupted:
        return
    if not (self._memory_manager and final_response and original_user_message):
        return
    # Multimodal turns carry content as a list of typed parts; providers
    # expect plain strings, so flatten to text first (newline-joined for
    # memory, vs the default space-join used for log/trajectory previews).
    user_text = _summarize_user_message_for_log(original_user_message, sep="\n")
    response_text = _summarize_user_message_for_log(final_response, sep="\n")
    if not (user_text and response_text):
        return
    try:
        sync_kwargs = {"session_id": self.session_id or ""}
        if messages is not None:
            sync_kwargs["messages"] = messages
        self._memory_manager.sync_all(
            user_text,
            response_text,
            **sync_kwargs,
        )
        # Sibling of the build_turn_context() prefetch gate: warming the
        # next turn's recall with a trivial prompt ("hi", "thanks") keys
        # provider searches on zero-signal text — skip it. The sync above
        # still runs so the turn itself is persisted.
        if not is_trivial_prompt(user_text):
            self._memory_manager.queue_prefetch_all(
                user_text,
                session_id=self.session_id or "",
            )
    except Exception:
        pass

def release_clients(self) -> None:
    """Release LLM client resources WITHOUT tearing down session tool state.

    Used by the gateway when evicting this agent from _agent_cache for
    memory-management reasons (LRU cap or idle TTL) — the session may
    resume at any time with a freshly-built AgentState that reuses the
    same task_id / session_id, so we must NOT kill:
      - process_registry entries for task_id (user's bg shells)
      - terminal sandbox for task_id (cwd, env, shell state)
      - browser daemon for task_id (open tabs, cookies)
      - computer-use backend for task_id (native target and browser refs)
      - memory provider (has its own lifecycle; keeps running)

    We DO close:
      - OpenAI/httpx client pool (big chunk of held memory + sockets;
        the rebuilt agent gets a fresh client anyway)
      - Active child subagents (per-turn artefacts; safe to drop)

    Safe to call multiple times.  Distinct from close() — which is the
    hard teardown for actual session boundaries (/new, /reset, session
    expiry).
    """
    import agent.provider_runtime as provider_runtime
    # Close active child agents (per-turn; no cross-turn persistence).
    try:
        with self._active_children_lock:
            children = list(self._active_children)
            self._active_children.clear()
        for child in children:
            try:
                release_clients(child)
            except Exception:
                # Fall back to full close on children; they're per-turn.
                try:
                    close(child)
                except Exception:
                    pass
    except Exception:
        pass

    # Retire the OpenAI/httpx client to release sockets immediately.
    # #70773: eviction runs on the gateway's memory-manager thread — a
    # cross-thread hard close of the shared client can release TLS FDs
    # under a still-unwinding worker (FD-recycle → SQLite corruption).
    # Retirement shuts the pooled sockets down (the memory/socket win we
    # want here) and lets GC release the FDs once no thread holds them.
    try:

        client = getattr(self, "client", None)
        if client is not None:
            provider_runtime._retire_shared_openai_client(self, client, reason="cache_evict")
            self.client = None
    except Exception:
        pass

    # Also drop the cached per-request wire client (reused across
    # sequential LLM calls) — same socket/memory rationale as above.
    try:
        provider_runtime._close_cached_request_openai_client(self, reason="cache_evict")
    except Exception:
        pass
    try:
        provider_runtime._close_cached_request_anthropic_client(self, reason="cache_evict")
    except Exception:
        pass

def close(self) -> None:
    """Release all resources held by this agent instance.

    Cleans up subprocess resources that would otherwise become orphans:
    - Background processes tracked in ProcessRegistry
    - Terminal sandbox environments
    - Browser daemon sessions
    - Computer-use backend sessions and target/ref state
    - Active child agents (subagent delegation)
    - OpenAI/httpx client connections

    Safe to call multiple times (idempotent).  Each cleanup step is
    independently guarded so a failure in one does not prevent the rest.
    """
    import agent.provider_runtime as provider_runtime
    # AgentState.close() is the hard owner boundary. Gateway cleanup may
    # call shutdown_memory_provider() first; its idempotence prevents
    # duplicate extraction while direct callers cannot skip provider close.
    try:
        session_messages = getattr(self, "_session_messages", None)
        shutdown_memory_provider(self,
            session_messages if isinstance(session_messages, list) else None
        )
    except Exception:
        pass

    task_id = getattr(self, "session_id", None) or ""

    # 1. Kill background processes for this task
    try:
        from tools.process_registry import process_registry
        process_registry.kill_all(task_id=task_id)
    except Exception:
        pass

    # 2. Clean terminal sandbox environments
    try:
        cleanup_vm(task_id)
    except Exception:
        pass

    # 3. Clean browser daemon sessions
    try:
        cleanup_browser(task_id)
    except Exception:
        pass

    # 4. Release the session-owned computer-use backend.  This ends the
    # exact cua-driver session, drops typed-browser refs/grants, and stops
    # a private embedded daemon when Hermes YOLO selected unrestricted
    # mode.  The import is lazy so sessions without computer_use retain
    # the narrow core footprint.
    try:
        from tools.computer_use import release_computer_use_session

        release_computer_use_session(task_id)
    except Exception:
        pass

    # 5. Close active child agents
    try:
        with self._active_children_lock:
            children = list(self._active_children)
            self._active_children.clear()
        for child in children:
            try:
                close(child)
            except Exception:
                pass
    except Exception:
        pass

    # 6. Close the OpenAI/httpx client
    try:
        client = getattr(self, "client", None)
        if client is not None:
            provider_runtime._close_openai_client(self, client, reason="agent_close", shared=True)
            self.client = None
    except Exception:
        pass

    # 6b. Close the cached per-request wire client (reused across
    # sequential LLM calls; see _create_request_openai_client).
    try:
        provider_runtime._close_cached_request_openai_client(self, reason="agent_close")
    except Exception:
        pass
    try:
        provider_runtime._close_cached_request_anthropic_client(self, reason="agent_close")
    except Exception:
        pass

    # 6c. Close the Codex app-server session. The runtime already drops
    # it on turn crash / retirement (agent/codex_runtime.py), but hard
    # teardown had no owner — a /new, /reset, or session expiry left the
    # app-server child process running until interpreter exit. Clear the
    # attribute BEFORE close() so a concurrent reader can't grab a
    # half-closed session, and so a raising close() can't strand a stale
    # reference behind.
    try:
        codex_session = getattr(self, "_codex_session", None)
        if codex_session is not None:
            self._codex_session = None
            codex_session.close()
    except Exception:
        pass

    # 7. Free conversation history.  Mirrors _release_evicted_agent_soft's
    # soft-eviction clear — close() is the hard teardown for true session
    # boundaries (/new, /reset, session expiry), so the message list won't
    # be reused.  Drops the reference proactively rather than waiting for
    # the agent object itself to be collected, which matters when a caller
    # still holds the closed agent (e.g. a draining background task).
    try:
        self._session_messages = []
    except Exception:
        pass

    # The references above are now gone; on Linux/glibc, return their free
    # heap pages immediately instead of retaining the process RSS high-water
    # mark until exit.  This helper is a safe no-op on other allocators.
    try:
        from hermes_cli.mem_trim import trim_memory
        trim_memory(force=True, reason="agent close")
    except Exception:
        pass

    # 8. Finalize the owned SQLite session row unless this agent is only a
    # temporary helper that deliberately handed session ownership forward
    # (manual compression helpers that rotate to a continuation session_id,
    # or background-review forks that share the live parent's session_id and
    # must leave it open). end_session() is first-reason-wins and no-ops on
    # an already-ended row, so this never clobbers a 'compression' /
    # 'cron_complete' / 'cli_close' reason set by an earlier terminal path.
    session_db = getattr(self, "_session_db", None)
    try:
        if getattr(self, "_end_session_on_close", True):
            session_id = getattr(self, "session_id", None)
            if session_db and session_id:
                session_db.end_session(session_id, "agent_close")
    except Exception:
        pass

    # 9. Close the SQLite handle itself, but ONLY when this agent owns it.
    # end_session() above finalizes the session ROW; it does not release the
    # connection. For the shared launch handle that is correct — it outlives
    # every agent — so _owns_session_db defaults False and this is a no-op.
    # A DEDICATED handle (the gateway's per-profile state.db opens, and the
    # lazy self-open in _get_session_db_for_recall) has no other owner: left
    # unclosed it keeps its db/-wal/-shm fds and its background token-writer
    # thread, and once that writer has started the instance pins ITSELF via
    # atexit.register(_drain_token_queue_at_exit) — which only close()
    # unregisters — so it survives for the life of the process.
    # Cleared first so the documented idempotency of close() holds.
    try:
        if getattr(self, "_owns_session_db", False) and session_db is not None:
            self._owns_session_db = False
            session_db.close()
    except Exception:
        pass

def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
    """
    Recover todo state from conversation history.

    The gateway creates a fresh AgentState per message, so the in-memory
    TodoStore is empty. We scan the history for the most recent todo
    tool response and replay it to reconstruct the state.

    Hydration is restricted to tool results that are paired with an
    earlier assistant ``todo`` tool call. The gateway/API server accepts
    caller-supplied ``conversation_history``, so a forged bare
    ``role: tool`` message carrying a ``todos`` array must not be able to
    seed the store without a matching canonical tool call
    (GHSA-5g4g-6jrg-mw3g).
    """
    import agent.status_output as status_output
    from tools.todo_tool import MAX_TODO_RESULT_CHARS

    # Walk history backwards to find the most recent todo tool response
    last_todo_response = None
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        # Only accept tool results paired with a prior assistant todo call.
        if not _tool_response_matches_todo_call(history, idx):
            continue
        if len(content) > MAX_TODO_RESULT_CHARS:
            logger.warning(
                "Skipping oversized todo tool response during hydration: "
                "session=%s chars=%d",
                self.session_id or "none",
                len(content),
            )
            continue
        # Quick check: todo responses contain "todos" key
        if '"todos"' not in content:
            continue
        try:
            data = json.loads(content)
            if "todos" in data and isinstance(data["todos"], list):
                last_todo_response = data["todos"]
                break
        except (json.JSONDecodeError, TypeError):
            continue

    if last_todo_response:
        # Replay the items into the store (replace mode)
        self._todo_store.write(last_todo_response, merge=False)
        if not self.quiet_mode:
            status_output._vprint(self, f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
    _set_interrupt(False)

def _tool_response_matches_todo_call(
    history: List[Dict[str, Any]],
    tool_index: int,
) -> bool:
    """Return True when a tool result belongs to a prior assistant todo call.

    Scans backwards from the tool result to the nearest assistant message
    and confirms it issued a ``todo`` tool call whose id matches this
    result's ``tool_call_id``. A ``user``/``system`` boundary (or a missing
    id) means the result is unpaired and must not hydrate the store.
    """
    if tool_index < 0 or tool_index >= len(history):
        return False
    tool_msg = history[tool_index]
    tool_call_id = tool_msg.get("tool_call_id")
    if not tool_call_id:
        return False

    for prior_idx in range(tool_index - 1, -1, -1):
        prior = history[prior_idx]
        role = prior.get("role")
        if role == "assistant":
            return _assistant_has_todo_tool_call(prior, tool_call_id)
        if role in {"user", "system"}:
            return False
    return False

def _assistant_has_todo_tool_call(
    assistant_msg: Dict[str, Any],
    tool_call_id: str,
) -> bool:
    """True when the assistant message issued a ``todo`` call with this id."""
    import agent.message_protocol as message_protocol
    tool_calls = assistant_msg.get("tool_calls")

    if not isinstance(tool_calls, list):
        return False

    for tool_call in tool_calls:
        if message_protocol._get_tool_call_id_static(tool_call) != tool_call_id:
            continue
        if get_tool_call_name(tool_call) == "todo":
            return True
    return False

def is_interrupted(agent) -> bool:
    """Check if an interrupt has been requested."""
    return agent._interrupt_requested

def _compress_context(
    self,
    messages: list,
    system_message: str,
    *,
    approx_tokens: int = None,
    task_id: str = "default",
    focus_topic: str = None,
    force: bool = False,
    defer_context_engine_notification: bool = False,
    commit_fence=None,
) -> tuple:
    """Forwarder — see ``agent.conversation_compression.compress_context``.

    ``force=True`` is passed by the manual ``/compress`` slash command
    so users can bypass the summary-failure cooldown after an
    auto-compress abort.  Auto-compress callers use the default
    ``force=False``.
    """
    import agent.message_protocol as message_protocol
    from agent.conversation_compression import (
        CompressionCommitFence,
        compress_context,
        resolve_context_compression_timeouts,
        run_compress_context_with_progress_timeout,
    )
    from agent.portal_tags import (
        get_conversation_context,
        reset_conversation_context,
        set_conversation_context,
    )
    # Out-of-turn compaction entry points — ``/compact`` (cli.py), the
    # gateway ``/compress`` command and its hygiene sweep (both of which
    # build a throwaway agent), and partial head compression — call this
    # forwarder directly, outside ``run_conversation``'s ambient scope.
    # With nothing ambient the summarizer's auxiliary call carries no
    # conversation tag and no Portal sticky key, so it routes independently
    # of the conversation it belongs to. Publish the root here as a
    # fallback; in-turn callers already have it set to the same value, so
    # this is a no-op for them.
    #
    # Note this does NOT keep the compaction turn's own prompt cache warm:
    # compaction replaces the history with a summary and rebuilds the
    # system prompt, so that request is a cold write on any endpoint. What
    # it buys is the turns AFTER compaction reading the cache it wrote.
    token = None
    if get_conversation_context() is None:
        root = _conversation_root_id(self)
        if root:
            token = set_conversation_context(root)
    # Every AgentState compression has a fence, including ordinary in-turn and
    # manual paths. interruption.hard_interrupt() uses this exact instance to serialize
    # cancel admission against begin_commit().
    active_fence = commit_fence or CompressionCommitFence()
    # A single agent can receive overlapping automatic/manual entrypoints.
    # Serialize fence publication so a waiter cannot replace the fence of
    # the attempt currently generating/committing a summary.
    fence_registration_lock = vars(self).setdefault(
        "_compression_commit_fence_lock", threading.RLock()
    )
    with fence_registration_lock:
        missing_fence = object()
        previous_fence = vars(self).get(
            "_active_compression_commit_fence", missing_fence
        )
        self._active_compression_commit_fence = active_fence
    try:
        def _run(fence=None, target_messages=None):
            return compress_context(
                self,

                target_messages if target_messages is not None else messages,
                system_message,
                approx_tokens=approx_tokens, task_id=task_id,
                focus_topic=focus_topic,
                force=force,
                defer_context_engine_notification=(
                    defer_context_engine_notification
                ),
                commit_fence=fence,
            )

        # Callers that already own a progress-aware wait (gateway session
        # hygiene) pass commit_fence and must not be double-wrapped.
        if commit_fence is not None:

            return _run(active_fence)

        idle_timeout, total_ceiling = resolve_context_compression_timeouts()
        if idle_timeout <= 0:
            return _run(active_fence)

        def _snapshot_worker(fence=None):
            # #76354 review F3: the pooled worker must NEVER share the
            # caller's live transcript. Plugin/legacy context engines are
            # allowed to mutate their input list in place; after a host
            # timeout the worker stays alive, so a shared list would let
            # a late engine rewrite the live conversation (roles,
            # ordering, persisted content) behind the caller's back.
            # Deep-snapshot here, on the worker thread, so the caller's
            # list object is never touched by pooled code. Results are
            # published to caller-visible state only via the returned
            # value of an ADMITTED commit (the host discards results on
            # timeout/cancel); durable SessionDB mutation is already
            # gated behind the commit fence inside compress_context.
            snapshot = copy.deepcopy(messages)
            result_msgs, result_prompt = _run(
                fence, target_messages=snapshot
            )
            if result_msgs is snapshot:
                # No-op/abort path returned the snapshot unchanged: hand
                # back the caller's ORIGINAL list so identity-based
                # semantics (len/identity no-op detection, flush dedup
                # by id()) keep working.
                return messages, result_prompt
            return result_msgs, result_prompt

        # Resolve the fallback prompt lazily on timeout only. Eager
        # rebuild here would raise before compress_context runs whenever
        # _cached_system_prompt is unset and _build_system_prompt fails
        # (lock-refresher / noop-exception tests rely on that path).
        def _fallback_prompt():
            cached = getattr(self, "_cached_system_prompt", None)
            if cached:
                return cached
            try:
                return message_protocol._build_system_prompt(self, system_message)
            except Exception:
                logger.debug(
                    "compress_context timeout fallback prompt rebuild "
                    "failed; using raw system_message",
                    exc_info=True,
                )
                return system_message or ""

        def _on_timeout(idle, waited, since_progress):
            logger.warning(
                "Context compression made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without "
                "compression",
                since_progress,
                waited,
                total_ceiling,
            )
            touch = getattr(self, "_touch_activity", None)
            if callable(touch):
                try:
                    touch(
                        "context compression timed out",
                        provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                    )
                except Exception:
                    logger.debug(
                        "compress_context timeout activity touch failed",
                        exc_info=True,
                    )
            # Same timeout cooldown ladder as summary-LLM timeouts
            # (#62452): avoid re-burning the full idle budget every turn.
            compressor = getattr(self, "context_compressor", None)
            if compressor is not None:
                record = getattr(compressor, "record_timeout_failure", None)
                if callable(record):
                    try:
                        record(
                            "host compress_context timeout "
                            "(no summary progress)"
                        )
                    except Exception:
                        logger.debug(
                            "failed to record compress_context timeout "
                            "cooldown",
                            exc_info=True,
                        )
            emit = getattr(self, "_emit_warning", None)
            if callable(emit):
                emit(
                    "⚠ Context compression timed out "
                    f"after {idle:.1f}s with no output from the summary "
                    "model. No messages were dropped — continuing without "
                    "compression. Run /compress to retry, /new for a clean "
                    "session, or check auxiliary.compression."
                )

        def _on_commit_overrun(waited, ceiling):
            # Commit-phase ceiling breach: the SessionDB mutation is in
            # flight and must complete (abandoning it mid-commit would
            # diverge live messages from durable session state), so this
            # only surfaces the overrun — it never cancels the commit.
            emit = getattr(self, "_emit_warning", None)
            if callable(emit):
                emit(
                    "⚠ Context compression commit is taking unusually "
                    f"long ({waited:.0f}s, ceiling {ceiling:.0f}s). "
                    "Waiting for it to finish safely — if this persists, "
                    "check SessionDB health (disk / lock contention)."
                )

        result = run_compress_context_with_progress_timeout(
            worker=_snapshot_worker,
            messages=messages,
            system_prompt_fallback=_fallback_prompt,
            idle_timeout_seconds=idle_timeout,
            total_ceiling_seconds=total_ceiling,
            on_timeout=_on_timeout,
            on_commit_overrun=_on_commit_overrun,
            fence=active_fence,
            telemetry_agent=self,
        )
        # compress_context ran on a daemon pool worker thread; the session
        # id rotation updated hermes_logging._session_context (a
        # threading.local) on the WORKER thread, not this one. Propagate
        # the current session_id back so subsequent log lines on this
        # thread carry the rotated id (#34089).
        try:
            from hermes_logging import set_session_context
            set_session_context(self.session_id)
        except Exception:
            pass
        # #76354 review F5: the worker thread also rebound the session
        # ContextVar inside its own (copied) context, which the caller
        # never sees — and get_session_env() prefers an already-bound
        # ContextVar over os.environ. Rebind in the CALLER's context so
        # post-compression tools/subprocesses on this thread resolve
        # HERMES_SESSION_ID to the child id after an out-of-place
        # rotation (idempotent when no rotation happened).
        try:
            from gateway.session_context import set_current_session_id
            if self.session_id:
                set_current_session_id(self.session_id)
        except Exception:
            logger.debug(
                "post-compression session ContextVar rebind failed",
                exc_info=True,
            )
        return result
    finally:
        with fence_registration_lock:
            if previous_fence is missing_fence:
                vars(self).pop("_active_compression_commit_fence", None)
            else:
                self._active_compression_commit_fence = previous_fence
        # Restore whatever the caller had, so a compaction never leaks its
        # tag into the surrounding scope.
        if token is not None:
            reset_conversation_context(token)

def _set_tool_guardrail_halt(self, decision: ToolGuardrailDecision) -> None:
    """Record the first guardrail decision that should stop this turn."""
    if decision.should_halt and self._tool_guardrail_halt_decision is None:
        self._tool_guardrail_halt_decision = decision

def _toolguard_controlled_halt_response(self, decision: ToolGuardrailDecision) -> str:
    tool = decision.tool_name or "a tool"
    return (
        f"I stopped retrying {tool} because it hit the tool-call guardrail "
        f"({decision.code}) after {decision.count} repeated non-progressing "
        "attempts. The last tool result explains the blocker; the next step is "
        "to change strategy instead of repeating the same call."
    )

def _append_guardrail_observation(
    self,
    tool_name: str,
    function_args: dict,
    function_result: str,
    *,
    failed: bool,
) -> str:
    decision = self._tool_guardrails.after_call(
        tool_name,
        function_args,
        function_result,
        failed=failed,
    )
    if decision.action in {"warn", "halt"}:
        function_result = append_toolguard_guidance(function_result, decision)
    if decision.should_halt:
        _set_tool_guardrail_halt(self, decision)
    return function_result

def _guardrail_block_result(self, decision: ToolGuardrailDecision) -> str:
    _set_tool_guardrail_halt(self, decision)
    return toolguard_synthetic_result(decision)

def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Execute tool calls from the assistant message and append results to messages.

    The segment planner splits the batch into maximal contiguous runs of
    parallel-safe calls (read-only tools, non-overlapping file targets,
    opted-in MCP tools) separated by sequential barriers (interactive,
    unsafe, or unrecognized tools). Homogeneous batches keep their
    original single-path dispatch; mixed batches execute segment by
    segment in emission order so safe subsets still run concurrently
    while side-effect ordering is preserved.
    """
    tool_calls = assistant_message.tool_calls

    # Allow _vprint during tool execution even with stream consumers
    self._executing_tools = True
    try:
        if len(tool_calls) <= 1:
            return _execute_tool_calls_sequential(self,
                assistant_message, messages, effective_task_id, api_call_count
            )

        from agent.tool_dispatch_helpers import _plan_tool_batch_segments
        _active_env = get_active_env(effective_task_id)
        _exec_cwd = Path(_active_env.cwd) if _active_env is not None and _active_env.cwd else None
        segments = _plan_tool_batch_segments(tool_calls, execution_cwd=_exec_cwd)

        if len(segments) == 1:
            kind = segments[0][0]
            if kind == "parallel":
                return _execute_tool_calls_concurrent(self,
                    assistant_message, messages, effective_task_id, api_call_count
                )
            return _execute_tool_calls_sequential(self,
                assistant_message, messages, effective_task_id, api_call_count
            )

        from agent.tool_executor import execute_tool_calls_segmented
        return execute_tool_calls_segmented(
            self, assistant_message, messages, effective_task_id, api_call_count,
            segments=segments,
        )
    finally:
        self._executing_tools = False

def _dispatch_delegate_task(self, function_args: dict) -> str:
    """Single call site for delegate_task dispatch.

    New DELEGATE_TASK_SCHEMA fields only need to be added here to reach all
    invocation paths (concurrent, sequential, inline).
    """
    from tools.delegate_tool import (
        _strip_model_hidden_task_fields,
        delegate_task as _delegate_task,
    )
    # Delegations from the top-level MODEL always run in the background —
    # the model does not get to choose. delegate_task returns immediately
    # with a handle (one per task) and each subagent's result re-enters the
    # conversation as a new message when it finishes. This applies to BOTH
    # a single task and a fan-out batch (each task becomes its own
    # independent background subagent). The one exception:
    #   - A delegation from an ORCHESTRATOR SUBAGENT (depth > 0) stays
    #     synchronous: the orchestrator needs its workers' results within
    #     its own turn to compose a summary, and a subagent doesn't own the
    #     gateway session the async result would route back to.
    # The schema-level `background` param is intentionally ignored here.
    _is_subagent = getattr(self, "_delegate_depth", 0) > 0
    return _delegate_task(
        goal=function_args.get("goal"),
        context=function_args.get("context"),
        tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
        max_iterations=function_args.get("max_iterations"),
        role=function_args.get("role"),
        background=(not _is_subagent),
        action=function_args.get("action"),
        subagent_id=function_args.get("subagent_id"),
        message=function_args.get("message"),
        parent_agent=self,
    )


def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
    """Word-wrap verbose tool output to fit the terminal width.

    Splits *text* on existing newlines and wraps each line individually,
    preserving intentional line breaks (e.g. pretty-printed JSON).
    Returns a ready-to-print string with *label* on the first line and
    continuation lines indented.
    """
    import shutil as _shutil
    import textwrap as _tw
    cols = _shutil.get_terminal_size((120, 24)).columns
    wrap_width = max(40, cols - len(indent))
    out_lines: list[str] = []
    for raw_line in text.split("\n"):
        if len(raw_line) <= wrap_width:
            out_lines.append(raw_line)
        else:
            wrapped = _tw.wrap(raw_line, width=wrap_width,
                               break_long_words=True,
                               break_on_hyphens=False)
            out_lines.extend(wrapped or [raw_line])
    body = ("\n" + indent).join(out_lines)
    return f"{indent}{label}{body}"

def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Forwarder — see ``agent.tool_executor.execute_tool_calls_concurrent``."""
    from agent.tool_executor import execute_tool_calls_concurrent
    return execute_tool_calls_concurrent(self, assistant_message, messages, effective_task_id, api_call_count)

def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
    """Forwarder — see ``agent.tool_executor.execute_tool_calls_sequential``."""
    from agent.tool_executor import execute_tool_calls_sequential
    return execute_tool_calls_sequential(self, assistant_message, messages, effective_task_id, api_call_count)

def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
    """Forwarder — see ``agent.chat_completion_helpers.handle_max_iterations``."""
    from agent.chat_completion_helpers import handle_max_iterations

    return handle_max_iterations(self, messages, api_call_count)

def _conversation_root_id(self) -> Optional[str]:
    """Resolve the stable conversation id for Portal usage attribution.

    Returns the session-lineage ROOT id rather than the current segment
    id, so one user-facing conversation keeps a single ``conversation=``
    tag across context-compression rotation (`/new` starts a genuinely
    new lineage). Delegate subagents resolve through their
    ``_parent_session_id`` so an entire delegation tree tags as the
    parent conversation.

    Best-effort: falls back to the raw session id when the session DB
    is unavailable or the lineage walk fails.
    """
    sid = getattr(self, "session_id", None)
    if not sid:
        return None
    # Subagents may not have a DB row yet on their first turn; walking
    # from the parent id still lands on the right root.
    start = getattr(self, "_parent_session_id", None) or sid
    db = getattr(self, "_session_db", None)
    if db is not None:
        try:
            root = db.get_conversation_root(start)
            if root:
                return root
        except Exception:
            logger.debug("Conversation root lineage walk failed", exc_info=True)
    return start

def run_conversation(
    self,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    moa_config: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forwarder — see ``agent.conversation_loop.run_conversation``."""
    import agent.interruption as interruption
    import agent.session_runtime as session_runtime
    import agent.status_output as status_output
    from agent.aux_accounting import (
        reset_accounting_context,
        set_accounting_context,
    )
    from agent import relay_runtime
    from agent.conversation_loop import run_conversation
    from agent.portal_tags import (
        reset_conversation_context,
        set_conversation_context,
    )
    from hermes_cli.observability.relay_shared_metrics import (
        finish_task_run,
        start_task_run,
    )
    from agent.subagent_lifecycle import bind_subagent_parent
    effective_task_id = task_id or str(uuid.uuid4())
    session_id = str(getattr(self, "session_id", None) or "")
    task_context = {
        "session_id": session_id,
        "task_id": effective_task_id,

        "platform": getattr(self, "platform", None) or "",
    }
    relay_turn_id = (
        f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    )
    self._relay_pending_turn_id = relay_turn_id
    relay_parent_session_id = (
        str(getattr(self, "_parent_session_id", None) or "")
        if task_context["platform"] == "subagent"
        else ""
    )
    relay_lease = None
    relay_turn = None
    durable_turn_lease = None
    durable_turn_lease_stop = None
    durable_turn_lease_thread = None
    durable_turn_lease_activity_lock = threading.Lock()
    durable_turn_lease_turn_active = False
    durable_turn_lease_interrupt_message = None
    token = None
    acct_token = None
    task_started = False
    task_finished = False
    relay_outcome = "failed"

    def _stop_durable_turn_lease_refresher() -> None:
        nonlocal durable_turn_lease_turn_active
        with durable_turn_lease_activity_lock:
            durable_turn_lease_turn_active = False
            if durable_turn_lease_stop is not None:
                durable_turn_lease_stop.set()

    def _clear_durable_turn_lease_interrupt() -> None:
        """Clear only the interrupt admitted by this turn's refresher."""
        message = durable_turn_lease_interrupt_message
        if not message:
            return

        def _clear_if_owned() -> None:
            if getattr(self, "_interrupt_message", None) != message:
                return
            self._interrupt_requested = False
            self._interrupt_message = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            self._interrupt_thread_signal_pending = False
            if self._execution_thread_id is not None:
                _set_interrupt(False, self._execution_thread_id)

        redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if redirect_lock is None:
            _clear_if_owned()
        else:
            with redirect_lock:
                _clear_if_owned()

    try:
        # Serialize the full load -> run -> flush region across Hermes
        # processes. Gateway's asyncio lease closes alias routing inside one
        # process; this durable lease covers Desktop, CLI resume, gateway,
        # and background delivery processes sharing state.db (#84234).
        _turn_db = getattr(self, "_session_db", None)
        _durable_session_exists = False
        if _turn_db is not None and session_id:
            try:
                _durable_session_exists = _turn_db.get_session(session_id) is not None
            except Exception:
                # A locked / non-WAL read is not proof the row is absent.
                # Treating probe failure as "fresh session" skipped the
                # lease this block exists to take and ran fail-open on
                # the exact contention point (#84234). Acquire (or fail
                # closed if acquire itself cannot) rather than start
                # load/run/flush unsynchronized. get_session returns
                # None — it does not raise — when the row is missing.
                logger.warning(
                    "Could not check durable session before turn lease; "
                    "will acquire rather than run without serialization",
                    exc_info=True,
                )
                _durable_session_exists = True
        if (
            _turn_db is not None
            and session_id
            and not getattr(self, "_persist_disabled", False)
            # A fresh session id is process-unique and has no durable
            # transcript to race over. More importantly, subagent/new-turn
            # callers may intentionally supply an in-memory seed before the
            # row exists; reloading an absent row would erase that seed.
            and _durable_session_exists
            # Test doubles and third-party DB shims may accept arbitrary
            # MagicMock attributes without implementing the protocol. Check
            # the concrete type so only real implementations opt in.
            and callable(
                getattr(type(_turn_db), "acquire_session_turn_lease", None)
            )
        ):
            # Resumed agents also defer their create check until the turn
            # prologue. We just proved this row exists, so suppress the
            # redundant create attempt after acquiring it.
            self._session_db_created = True
            _durable_holder = (
                f"pid={os.getpid()}:turn={relay_turn_id}:platform="
                f"{task_context['platform'] or 'unknown'}"
            )
            _lease_ttl = 300.0
            _lease_waited = False

            def _on_session_turn_lease_wait(elapsed: float) -> None:
                nonlocal _lease_waited
                _lease_waited = True
                if elapsed < 1.0:
                    status_output._emit_status(self,
                        "⏳ Another Hermes process is using this session; "
                        "waiting for it to finish before starting your turn..."
                    )
                else:
                    status_output._emit_status(self,
                        "⏳ Still waiting for the other Hermes process on "
                        f"this session ({int(elapsed)}s)..."
                    )

            if not _turn_db.acquire_session_turn_lease(
                session_id,
                _durable_holder,
                ttl_seconds=_lease_ttl,
                wait_seconds=1800.0,
                on_wait=_on_session_turn_lease_wait,
                should_abort=lambda: getattr(self, "_interrupt_requested", False),
            ):
                if getattr(self, "_interrupt_requested", False):
                    logger.info(
                        "session turn lease wait aborted by interrupt: %s",
                        session_id,
                    )
                    relay_outcome = "cancelled"
                    interrupt_msg = (
                        "Stopped waiting for another Hermes process on "
                        "this session. Your message was not processed."
                    )
                    interrupt_result = {
                        "final_response": interrupt_msg,
                        "messages": list(conversation_history or []),
                        "api_calls": 0,
                        "completed": False,
                        "interrupted": True,
                    }
                    interrupt_message = getattr(
                        self, "_interrupt_message", None
                    )
                    if interrupt_message:
                        interrupt_result["interrupt_message"] = (
                            interrupt_message
                        )
                    # Conversation-loop finalizer never runs on this
                    # early return. Clear so a cached agent cannot
                    # fail-close the next turn as interrupted.
                    try:
                        interruption.clear_interrupt(self)
                    except Exception:
                        self._interrupt_requested = False
                        self._interrupt_message = None
                    return interrupt_result
                # Fail closed like gateway TurnLeaseTimeoutError: do not
                # enter load/run/flush, and surface a resend notice instead
                # of a bare TimeoutError that looks like a hang.
                timeout_msg = (
                    "⏳ Another Hermes process kept this session busy too "
                    "long. Your message was not processed - wait for the "
                    "other process to finish, then send it again."
                )
                logger.error(
                    "session turn lease wait timed out for %s",
                    session_id,
                )
                try:
                    status_output._emit_warning(self, timeout_msg)
                except Exception:
                    logger.debug(
                        "Failed to emit session turn lease timeout warning",
                        exc_info=True,
                    )
                relay_outcome = "timed_out"
                return {
                    "final_response": timeout_msg,
                    "messages": list(conversation_history or []),
                    "api_calls": 0,
                    "completed": False,
                    "failed": True,
                    "error": f"session_turn_lease_timeout:{session_id}",
                }

            # Assign only after admission so finally release cannot target a
            # holder string that never owned the row. Persist paths read
            # the agent attr so a late flush after reclaim is fenced in
            # the same SQLite write transaction as the transcript insert.
            durable_turn_lease = _durable_holder
            self._active_session_turn_lease_holder = _durable_holder
            self._active_session_turn_lease_ttl_seconds = _lease_ttl
            if _lease_waited:
                status_output._emit_status(self,
                    "Session is free; loading the latest transcript..."
                )

            # The holder may have compressed and rotated the session while
            # this process waited. Resolve and reload only AFTER admission;
            # a caller-provided in-memory snapshot is necessarily stale.
            # Skip when acquisition was immediate — no other process held
            # the lease, so the in-memory history is current and reloading
            # would only cause an unnecessary prompt cache miss.
            if _lease_waited:
                latest_session_id = _turn_db.resolve_resume_session_id(session_id)
                if latest_session_id:
                    self.session_id = latest_session_id
                    task_context["session_id"] = latest_session_id
                conversation_history = _turn_db.get_messages_as_conversation(
                    self.session_id,
                    repair_alternation=True,
                    include_row_ids=True,
                )

            # Long model/tool/compression turns outlive a fixed TTL. Refresh
            # in a daemon thread; holder-qualified UPDATE and DELETE fence a
            # late refresher/release from a successor lease.
            durable_turn_lease_stop = threading.Event()
            _lease_refresh_interval = float(
                getattr(self, "_session_turn_lease_refresh_interval", 60.0)
            )

            def _refresh_durable_turn_lease() -> None:
                def _interrupt_turn(message: str) -> None:
                    nonlocal durable_turn_lease_interrupt_message
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return
                        durable_turn_lease_interrupt_message = message
                        try:
                            interruption.interrupt(self, message, hard_cancel=True)
                        except Exception:
                            self._interrupt_requested = True
                            self._interrupt_message = message

                while not durable_turn_lease_stop.wait(_lease_refresh_interval):
                    try:
                        if not _turn_db.refresh_session_turn_lease(
                            getattr(self, "session_id", None) or session_id,
                            durable_turn_lease,
                            ttl_seconds=_lease_ttl,
                        ):
                            # finally sets the stop event then releases.
                            # A late holder-fenced miss after that join
                            # timeout must not hard-interrupt the next turn.
                            if durable_turn_lease_stop.is_set():
                                return
                            logger.error(
                                "Lost session turn lease while turn is active: %s",
                                getattr(self, "session_id", None) or session_id,
                            )
                            _interrupt_turn(
                                "Session turn lease lost; stopping to protect "
                                "the transcript."
                            )
                            return
                    except Exception:
                        if durable_turn_lease_stop.is_set():
                            return
                        logger.warning(
                            "Failed to refresh session turn lease: %s",
                            getattr(self, "session_id", None) or session_id,
                            exc_info=True,
                        )
                        _interrupt_turn(
                            "Session turn lease could not be refreshed; "
                            "stopping to protect the transcript."
                        )
                        return

            durable_turn_lease_thread = threading.Thread(
                target=_refresh_durable_turn_lease,
                name="session-turn-lease-refresh",
                daemon=True,
            )



        relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(),
            session_id=task_context["session_id"],
            platform=task_context["platform"],
            parent_session_id=relay_parent_session_id,
            model=str(getattr(self, "model", None) or ""),
        )
        relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
            relay_lease,
            turn_id=relay_turn_id,
            task_id=effective_task_id,
        )
        # Keep existing tests and external relay-runtime shims that return
        # a minimal turn object compatible with the new opt-out flag.
        if getattr(relay_turn, "relay_enabled", True):
            start_task_run(
                **task_context,
                parent_session_id=getattr(self, "_parent_session_id", None) or "",
            )
            task_started = True
        # Publish the conversation id for ambient Nous Portal tagging. Every
        # LLM call made inside this turn — main loop, compression, vision,
        # web_extract, session_search, MoA slots, background-review forks
        # (which copy this Context into their thread) — inherits the
        # ``conversation=<root>`` tag with zero per-call-site plumbing.
        token = set_conversation_context(_conversation_root_id(self))
        # Publish the session accounting handles the same way so auxiliary
        # calls record their token usage into session_model_usage (task
        # dimension) — the fix for aux spend being invisible in analytics
        # (issue #23270).
        acct_token = set_accounting_context(
            getattr(self, "_session_db", None),
            getattr(self, "session_id", None),
        )
        from agent.auxiliary_client import scoped_runtime_main

        # The outer token restores the caller's Context even though turn setup
        # replaces the value with the live runtime after fallback restoration.
        # Keep the scope local instead of storing ContextVar tokens on the agent,
        # which may be observed from another thread.
        with bind_subagent_parent(self), scoped_runtime_main({}):
            try:
                if durable_turn_lease_thread is not None:
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_turn_active = True
                    durable_turn_lease_thread.start()
                result = run_conversation(
                    self,
                    user_message,
                    system_message,
                    conversation_history,
                    effective_task_id,
                    stream_callback,
                    persist_user_message,
                    persist_user_timestamp=persist_user_timestamp,
                    persist_user_display_kind=persist_user_display_kind,
                    persist_user_display_metadata=persist_user_display_metadata,
                    moa_config=moa_config,
                )
            finally:
                # The lease remains held through relay/task finalization, but
                # those post-loop steps must not receive a late refresh
                # interrupt that poisons the next turn on a cached agent.
                _stop_durable_turn_lease_refresher()
                # Interrupt clear is deferred to after thread join in the
                # outer finally: a refresher firing between stop and join
                # would otherwise set an interrupt that survives the clear.
        terminal = result if isinstance(result, dict) else {}
        if terminal.get("interrupted") is True:
            relay_outcome = "cancelled"
        elif terminal.get("failed") is True:
            relay_outcome = "failed"
        else:
            relay_outcome = "success"
        relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
            relay_turn,
            outcome=relay_outcome,
        )
        if task_started:
            task_finished = True
            finish_task_run(**task_context, result=result)
        return result
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
            type(exc).__name__ == "CancelledError"
        ):
            relay_outcome = "cancelled"
        elif isinstance(exc, TimeoutError):
            relay_outcome = "timed_out"
        if relay_turn is not None:
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn,
                outcome=relay_outcome,
            )
        if task_started and not task_finished:
            task_finished = True
            finish_task_run(**task_context, error=exc)
        raise
    finally:
        try:
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.end_turn(
                    relay_turn,
                    outcome=relay_outcome,
                )
        finally:
            try:
                if relay_lease is not None:
                    relay_runtime.SESSION_COORDINATOR.release_conversation(
                        relay_lease
                    )
            finally:
                _stop_durable_turn_lease_refresher()
                if (
                    durable_turn_lease_thread is not None

                    and durable_turn_lease_thread.is_alive()
                ):
                    durable_turn_lease_thread.join(timeout=1.0)
                # Clear any interrupt the refresher may have fired between
                # the inner stop and this join. Must run AFTER join so a
                # late interrupt does not survive into the next turn.
                _clear_durable_turn_lease_interrupt()
                if durable_turn_lease is not None:
                    try:
                        _turn_db.release_session_turn_lease(
                            session_id, durable_turn_lease
                        )
                    except Exception:
                        logger.error(
                            "Failed to release session turn lease: %s",
                            session_id,
                            exc_info=True,
                        )
                    if (
                        getattr(self, "_active_session_turn_lease_holder", None)
                        == durable_turn_lease
                    ):
                        self._active_session_turn_lease_holder = None
                        self._active_session_turn_lease_ttl_seconds = None
                # Always clear mid-turn labels when the turn exits — including
                # interrupted early returns that skip finalize_turn. Keep ts.
                try:
                    session_runtime._reset_activity_labels_after_turn(self)
                except Exception:
                    pass
                if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                    self._relay_pending_turn_id = None
                if acct_token is not None:
                    reset_accounting_context(acct_token)
                if token is not None:
                    reset_conversation_context(token)

def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
    """
    Simple chat interface that returns just the final response.

    Args:
        message (str): User message
        stream_callback: Optional callback invoked with each text delta during streaming.

    Returns:
        str: Final assistant response
    """
    result = run_conversation(self, message, stream_callback=stream_callback)
    return result["final_response"]

def _run_codex_app_server_turn(
    self,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Forwarder — see ``agent.codex_runtime.run_codex_app_server_turn``."""
    from agent.codex_runtime import run_codex_app_server_turn
    return run_codex_app_server_turn(self, user_message=user_message, original_user_message=original_user_message, messages=messages, effective_task_id=effective_task_id, should_review_memory=should_review_memory)

def agent_runtime_owns_post_tool_hook(agent: Any, function_name: str) -> bool:
    """Return True when an agent-level tool path emits its own post hook."""
    if function_name in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES:
        return True
    if getattr(agent, "_context_engine_tool_names", None) and function_name in agent._context_engine_tool_names:
        return True
    memory_manager = getattr(agent, "_memory_manager", None)
    return bool(memory_manager and memory_manager.has_tool(function_name))

def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False,
                 skip_tool_request_middleware: bool = False,
                 tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
                 skip_tool_execution_middleware: bool = False) -> str:
    """Invoke a single tool and return the result string. No display logic.

    Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
    tools. Used by the concurrent execution path; the sequential path retains
    its own inline invocation for backward-compatible display handling.
    """
    import agent.lifecycle as lifecycle
    import agent.session_runtime as session_runtime
    if not isinstance(function_args, dict):
        function_args = {}

    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    try:
        from hermes_cli.middleware import apply_tool_request_middleware

        if not skip_tool_request_middleware:
            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            )
            function_args = _tool_request_mw.payload
            _tool_middleware_trace = _tool_request_mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)

    # Check plugin hooks for a block or approval directive before executing.
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        try:
            from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
            block_message, modified_args = _dispatch_pre_tool_call_hooks(
                function_name, function_args, task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                middleware_trace=list(_tool_middleware_trace),
            )
            if modified_args is not None:
                function_args = modified_args
        except Exception:
            block_message = None
    if block_message is not None:
        result = json.dumps({"error": block_message}, ensure_ascii=False)
        try:
            from model_tools import _emit_post_tool_call_hook
            _emit_post_tool_call_hook(
                function_name=function_name,
                function_args=function_args,
                result=result,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                status="blocked",
                error_type="plugin_block",
                error_message=block_message,
                middleware_trace=list(_tool_middleware_trace),
            )
        except Exception:
            pass
        return result

    tool_start_time = time.monotonic()

    def _finish_agent_tool(result: Any, observed_args: Optional[dict] = None) -> Any:
        hook_args = observed_args if isinstance(observed_args, dict) else function_args
        try:
            from model_tools import _emit_post_tool_call_hook
            _emit_post_tool_call_hook(
                function_name=function_name,
                function_args=hook_args,
                result=result,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                duration_ms=int((time.monotonic() - tool_start_time) * 1000),
                middleware_trace=list(_tool_middleware_trace),
            )
        except Exception:
            pass
        return result

    if function_name == "todo":
        def _execute(next_args: dict) -> Any:
            from tools.todo_tool import todo_tool as _todo_tool
            return _finish_agent_tool(
                _todo_tool(
                    todos=next_args.get("todos"),
                    merge=next_args.get("merge", False),
                    store=agent._todo_store,
                ),
                next_args,
            )
    elif function_name == "session_search":
        def _execute(next_args: dict) -> Any:
            session_db = session_runtime._get_session_db_for_recall(agent)
            if not session_db:
                from hermes_state import format_session_db_unavailable
                return _finish_agent_tool(json.dumps({"success": False, "error": format_session_db_unavailable()}), next_args)
            from tools.session_search_tool import session_search as _session_search
            return _finish_agent_tool(
                _session_search(
                    query=next_args.get("query", ""),
                    role_filter=next_args.get("role_filter"),
                    limit=next_args.get("limit", 3),
                    session_id=next_args.get("session_id"),
                    around_message_id=next_args.get("around_message_id"),
                    window=next_args.get("window", 5),
                    sort=next_args.get("sort"),
                    detail=next_args.get("detail", "adaptive"),
                    db=session_db,
                    current_session_id=agent.session_id,
                ),
                next_args,
            )
    elif function_name == "memory":
        def _execute(next_args: dict) -> Any:
            target = next_args.get("target", "memory")
            operations = next_args.get("operations")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=next_args.get("action"),
                target=target,
                content=next_args.get("content"),
                old_text=next_args.get("old_text"),
                operations=operations,
                store=agent._memory_store,
            )
            # Mirror successful built-in memory writes to external providers.
            # All gating/op-expansion lives behind the manager interface
            # (MemoryManager.notify_memory_tool_write).
            if agent._memory_manager:
                agent._memory_manager.notify_memory_tool_write(
                    result,
                    next_args,
                    build_metadata=lambda: lifecycle._build_memory_write_metadata(agent,
                        task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                    ),
                )
            return _finish_agent_tool(result, next_args)
    elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
        def _execute(next_args: dict) -> Any:
            return _finish_agent_tool(agent._memory_manager.handle_tool_call(function_name, next_args), next_args)
    elif function_name == "clarify":
        def _execute(next_args: dict) -> Any:
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _finish_agent_tool(
                _clarify_tool(
                    question=next_args.get("question", ""),
                    choices=next_args.get("choices"),
                    multi_select=next_args.get("multi_select", False),
                    callback=agent.clarify_callback,
                ),
                next_args,
            )
    elif function_name == "delegate_task":
        def _execute(next_args: dict) -> Any:
            return _finish_agent_tool(lifecycle._dispatch_delegate_task(agent, next_args), next_args)
    else:
        def _execute(next_args: dict) -> Any:
            dispatch_kwargs = dict(
                tool_call_id=tool_call_id,
                session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                tool_request_middleware_trace=list(_tool_middleware_trace),
            )
            if skip_tool_execution_middleware:
                dispatch_kwargs["skip_tool_execution_middleware"] = True
            return handle_function_call(
                function_name,
                next_args,
                effective_task_id,
                **dispatch_kwargs,
            )

    if skip_tool_execution_middleware:
        return _execute(function_args)

    from hermes_cli.middleware import run_tool_execution_middleware

    return run_tool_execution_middleware(
        function_name,
        function_args,
        lambda next_args: _execute(next_args if isinstance(next_args, dict) else function_args),
        original_args=function_args,
        task_id=effective_task_id or "",
        session_id=getattr(agent, "session_id", "") or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )
