"""Responsibility-owned agent interruption behavior."""

import logging
import threading
from typing import Optional
from agent.interrupt_compat import request_hard_interrupt
from tools.interrupt import set_interrupt as _set_interrupt


logger = logging.getLogger(__name__)

def interrupt(self, message: Optional[str] = None, *, hard_cancel: bool = False) -> None:
    """
    Request the agent to interrupt its current tool-calling loop.

    Call this from another thread (e.g., input handler, message receiver)
    to gracefully stop the agent and process a new message.

    Also signals long-running tool executions (e.g. terminal commands)
    to terminate early, so the agent can respond immediately.

    Args:
        message: Optional new message that triggered the interrupt.
                 If provided, the agent will include this in its response context.
        hard_cancel: Mark this as an explicit stop rather than a redirect or
                     incoming-message interrupt. Compression may honor this
                     atomic signal even while ordinary interrupts are masked.

    Example (CLI):
        # In a separate input thread:
        if user_typed_something:
            interrupt(agent, user_input)

    Example (Messaging):
        # When new message arrives for active session:
        if session_has_running_agent:
            interrupt(running_agent, new_message.text)
    """
    # A hard stop and redirect share one lock so /stop cannot race with an
    # accepted correction and accidentally turn itself into a retry.
    def _admit_hard_cancel() -> None:
        event = getattr(self, "_hard_interrupt_requested", None)
        if event is None:
            return
        fence = vars(self).get("_active_compression_commit_fence")
        cancel_before_commit = getattr(
            type(fence), "cancel_before_commit", None
        )
        if callable(cancel_before_commit):
            try:
                # This sets the Event while holding the same lock used by
                # begin_commit(). If commit already won, it waits for that
                # tracked mutation to finish before publishing the stop.
                cancel_before_commit(fence, event)
                return
            except Exception:
                logger.debug(
                    "Compression hard-cancel fence admission failed",
                    exc_info=True,
                )
        event.set()

    _redirect_lock = getattr(self, "_pending_redirect_lock", None)
    if _redirect_lock is not None:
        with _redirect_lock:
            self._interrupt_requested = True
            self._interrupt_message = message
            if hard_cancel:
                _admit_hard_cancel()
            self._pending_redirect = None
    else:
        self._interrupt_requested = True
        self._interrupt_message = message
        if hard_cancel:
            _admit_hard_cancel()
        self._pending_redirect = None

    # Codex app-server owns its model/tool loop and watches a private
    # interrupt event rather than Hermes' per-thread flag.
    if getattr(self, "api_mode", None) == "codex_app_server":
        _codex_session = getattr(self, "_codex_session", None)
        _request_interrupt = getattr(_codex_session, "request_interrupt", None)
        if callable(_request_interrupt):
            try:
                _request_interrupt()
            except Exception:
                logger.debug(
                    "Failed to interrupt Codex app-server turn",
                    exc_info=True,
                )

    # A cron turn performs its API request on the conversation thread to
    # avoid the nested interrupt-worker deadlock.  Unlike the normal worker
    # path, its client is registered here so this cross-thread interrupt can
    # still shut down the active sockets promptly.
    _abort_active_request = getattr(self, "_active_request_abort", None)
    if callable(_abort_active_request):
        try:
            _abort_active_request("interrupt_abort")
        except Exception:
            logger.debug("Failed to abort active inline request", exc_info=True)
    # Signal all tools to abort any in-flight operations immediately.
    # Scope the interrupt to this agent's execution thread so other
    # agents running in the same process (gateway) are not affected.
    if self._execution_thread_id is not None:
        _set_interrupt(True, self._execution_thread_id)
        self._interrupt_thread_signal_pending = False
    else:
        # The interrupt arrived before lifecycle.run_conversation() finished
        # binding the agent to its execution thread. Defer the tool-level
        # interrupt signal until startup completes instead of targeting
        # the caller thread by mistake.
        self._interrupt_thread_signal_pending = True
    # Fan out to concurrent-tool worker threads.  Those workers run tools
    # on their own tids (ThreadPoolExecutor workers), so `lifecycle.is_interrupted()`
    # inside a tool only sees an interrupt when their specific tid is in
    # the `_interrupted_threads` set.  Without this propagation, an
    # already-running concurrent tool (e.g. a terminal command hung on
    # network I/O) never notices the interrupt and has to run to its own
    # timeout.  See `_run_tool` for the matching entry/exit bookkeeping.
    # `getattr` fallback covers test stubs that build AgentState via
    # object.__new__ and skip __init__.
    _tracker = getattr(self, "_tool_worker_threads", None)
    _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
    if _tracker is not None and _tracker_lock is not None:
        with _tracker_lock:
            _worker_tids = list(_tracker)
        for _wtid in _worker_tids:
            try:
                _set_interrupt(True, _wtid)
            except Exception:
                pass
    # Propagate interrupt to any running child agents (subagent delegation)
    with self._active_children_lock:
        children_copy = list(self._active_children)
    for child in children_copy:
        try:
            if hard_cancel:
                request_hard_interrupt(child, message)
            else:
                interrupt(child, message)
        except Exception as e:
            logger.debug("Failed to propagate interrupt to child agent: %s", e)
    if not self.quiet_mode:
        print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))

def hard_interrupt(self, message: Optional[str] = None) -> None:
    """Request an explicit stop while preserving ``interrupt()`` ABI.

    Frontends can feature-detect this method and fall back to the legacy
    ``interrupt()`` signature for synthetic or third-party agents.
    """
    # Deliberately bypass dynamic dispatch: subclasses written against the
    # legacy interrupt(message=None) ABI may override interrupt without the
    # newer keyword-only hard_cancel argument.
    interrupt(self, message, hard_cancel=True)

def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
    """Clear the interrupt request and per-thread tool signal.

    ``preserve_redirect`` is used only by the conversation loop after it
    intentionally cancels a model request to rebuild that same logical
    turn. Public hard-stop paths keep the default and clear everything.
    """
    _redirect_lock = getattr(self, "_pending_redirect_lock", None)
    if _redirect_lock is not None:
        with _redirect_lock:
            if preserve_redirect and not self._pending_redirect:
                return False
            self._interrupt_requested = False
            self._interrupt_message = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
    else:
        if preserve_redirect and not getattr(self, "_pending_redirect", None):
            return False
        self._interrupt_requested = False
        self._interrupt_message = None
        getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
        if not preserve_redirect:
            self._pending_redirect = None
    self._interrupt_thread_signal_pending = False
    if self._execution_thread_id is not None:
        _set_interrupt(False, self._execution_thread_id)
    # Also clear any concurrent-tool worker thread bits.  Tracked
    # workers normally clear their own bit on exit, but an explicit
    # clear here guarantees no stale interrupt can survive a turn
    # boundary and fire on a subsequent, unrelated tool call that
    # happens to get scheduled onto the same recycled worker tid.
    # `getattr` fallback covers test stubs that build AgentState via
    # object.__new__ and skip __init__.
    _tracker = getattr(self, "_tool_worker_threads", None)
    _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
    if _tracker is not None and _tracker_lock is not None:
        with _tracker_lock:
            _worker_tids = list(_tracker)
        for _wtid in _worker_tids:
            try:
                _set_interrupt(False, _wtid)
            except Exception:
                pass
    # A hard interrupt supersedes any pending /steer — the steer was
    # meant for the agent's next tool-call iteration, which will no
    # longer happen. Drop it instead of surprising the user with a
    # late injection on the post-interrupt turn.
    _steer_lock = getattr(self, "_pending_steer_lock", None)
    if _steer_lock is not None:
        with _steer_lock:
            self._pending_steer = None
    return True

def steer(self, text: str) -> bool:
    """
    Inject a user message into the next tool result without interrupting.

    Unlike interrupt(), this does NOT stop the current tool call. The
    text is stashed and the agent loop appends it to the LAST tool
    result's content once the current tool batch finishes. The model
    sees the steer as part of the tool output on its next iteration.

    Thread-safe: callable from gateway/CLI/TUI threads. Multiple calls
    before the drain point concatenate with newlines.

    Args:
        text: The user text to inject. Empty strings are ignored.

    Returns:
        True if the steer was accepted, False if the text was empty.
    """
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    _lock = getattr(self, "_pending_steer_lock", None)
    if _lock is None:
        # Test stubs that built AgentState via object.__new__ skip __init__.
        # Fall back to direct attribute set; no concurrent callers expected
        # in those stubs.
        existing = getattr(self, "_pending_steer", None)
        self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
        return True
    with _lock:
        if self._pending_steer:
            self._pending_steer = self._pending_steer + "\n" + cleaned
        else:
            self._pending_steer = cleaned
    return True

def redirect(self, text: str) -> bool:
    """Redirect the active turn without converting it into a new task.

    During a normal Hermes model request this cancels only that request;
    the conversation loop retains completed messages/tool results, records
    the displayed partial reasoning as plain assistant context, appends the
    correction as a real user message, and retries. During tool execution
    it degrades to ``steer()`` so the tool can finish at a safe boundary.
    Codex app-server has a native ``turn/steer`` operation and uses it
    directly instead of cancelling.

    Returns ``False`` when there is no live turn or the text is empty, so
    surfaces can fall back to their existing next-turn queue.
    """
    if not text or not text.strip():
        return False
    cleaned = text.strip()

    # Codex owns its internal reasoning/tool loop, so use its first-class
    # active-turn steering protocol rather than interrupting the subprocess.
    if getattr(self, "api_mode", None) == "codex_app_server":
        _codex_session = getattr(self, "_codex_session", None)
        _native_steer = getattr(_codex_session, "request_steer", None)
        if callable(_native_steer):
            _redirect_lock = getattr(self, "_pending_redirect_lock", None)
            if _redirect_lock is not None:
                with _redirect_lock:
                    if self._interrupt_requested:
                        return False
            elif self._interrupt_requested:
                return False
            try:
                return bool(_native_steer(cleaned))
            except Exception:
                logger.debug("Codex app-server turn/steer failed", exc_info=True)
                return False

    # Never kill a tool merely to deliver conversational guidance. The
    # existing steer drain puts it on the final tool result before the next
    # model decision, including delegate_task children.
    if getattr(self, "_executing_tools", False):
        return steer(self, cleaned)

    _model_active = getattr(self, "_model_request_active", None)
    _redirect_lock = getattr(self, "_pending_redirect_lock", None)
    if _redirect_lock is None:
        if _model_active is None or not _model_active.is_set():
            return False
        existing = getattr(self, "_pending_redirect", None)
        if self._interrupt_requested and not existing:
            return False
        self._pending_redirect = (
            f"{existing}\n\n[Additional user correction]\n{cleaned}"
            if existing
            else cleaned
        )
        self._interrupt_requested = True
        self._interrupt_message = None
    else:
        with _redirect_lock:
            if _model_active is None or not _model_active.is_set():
                # The response completed before we acquired the state lock.
                # Reject so the surface queues a new turn.
                return False
            if self._interrupt_requested and not self._pending_redirect:
                return False
            if self._pending_redirect:
                self._pending_redirect = (
                    f"{self._pending_redirect}\n\n"
                    f"[Additional user correction]\n{cleaned}"
                )
            else:
                self._pending_redirect = cleaned
            self._interrupt_requested = True
            self._interrupt_message = None

    # Interrupt only the model request. Do not fan out to tool workers or
    # child agents as interrupt() does.
    _execution_thread_id = getattr(self, "_execution_thread_id", None)
    if _execution_thread_id is not None:
        _set_interrupt(True, _execution_thread_id)
        self._interrupt_thread_signal_pending = False

    else:
        self._interrupt_thread_signal_pending = True
    _abort_active_request = getattr(self, "_active_request_abort", None)
    if callable(_abort_active_request):
        try:
            _abort_active_request("redirect_abort")
        except Exception:

            logger.debug("Failed to abort request for redirect", exc_info=True)
    return True

def _has_pending_redirect(self) -> bool:
    """Return whether an active-turn redirect is waiting to be applied."""
    _redirect_lock = getattr(self, "_pending_redirect_lock", None)
    if _redirect_lock is None:
        return bool(getattr(self, "_pending_redirect", None))
    with _redirect_lock:
        return bool(self._pending_redirect)

def _drain_pending_redirect(self) -> Optional[str]:
    """Return and clear pending active-turn correction text."""
    _redirect_lock = getattr(self, "_pending_redirect_lock", None)
    if _redirect_lock is None:
        text = getattr(self, "_pending_redirect", None)
        self._pending_redirect = None
        return text
    with _redirect_lock:
        text = self._pending_redirect
        self._pending_redirect = None
    return text

def _drain_pending_steer(self) -> Optional[str]:
    """Return the pending steer text (if any) and clear the slot.

    Safe to call from the agent execution thread after appending tool
    results. Returns None when no steer is pending.
    """
    _lock = getattr(self, "_pending_steer_lock", None)
    if _lock is None:
        text = getattr(self, "_pending_steer", None)
        self._pending_steer = None
        return text
    with _lock:
        text = self._pending_steer
        self._pending_steer = None
    return text
