"""Responsibility-owned agent stream runtime behavior."""

import logging
import re
import threading
from typing import List, Dict, Any
from agent.memory_manager import sanitize_context
from agent.redact import redact_sensitive_text
from agent.message_content import flatten_message_text


logger = logging.getLogger(__name__)

def _reset_stream_delivery_tracking(self) -> None:
    """Reset tracking for text delivered during the current model response."""
    # Flush any benign partial-tag tail held by the think scrubber
    # first (#17924): an innocent '<' at the end of the stream that
    # turned out not to be a tag prefix should reach the UI.  Then
    # flush the context scrubber.  Order matters — the think
    # scrubber's output feeds into the context scrubber's state.
    think_scrubber = getattr(self, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_tail = think_scrubber.flush()
        if think_tail:
            # Route the tail through the context scrubber too so a
            # memory-context span straddling the final boundary is
            # still caught.
            ctx_scrubber = getattr(self, "_stream_context_scrubber", None)
            if ctx_scrubber is not None:
                think_tail = ctx_scrubber.feed(think_tail)
            if think_tail:
                callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                for cb in callbacks:
                    try:
                        cb(think_tail)
                    except Exception:
                        pass
                _record_streamed_assistant_text(self, think_tail)
    # Flush any benign partial-tag tail held by the context scrubber so it
    # reaches the UI before we clear state for the next model call.  If
    # the scrubber is mid-span, flush() drops the orphaned content.
    scrubber = getattr(self, "_stream_context_scrubber", None)
    if scrubber is not None:
        tail = scrubber.flush()
        if tail:
            callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
            for cb in callbacks:
                try:
                    cb(tail)
                except Exception:
                    pass
            _record_streamed_assistant_text(self, tail)
    self._current_streamed_assistant_text = ""

def _record_streamed_assistant_text(self, text: str) -> None:
    """Accumulate visible assistant text emitted through stream callbacks."""
    # Single-writer guard (#65991): a superseded stream must not pollute the
    # turn's accumulated text (which also feeds the interim-visible-text
    # de-dup comparison), even when a caller reaches this directly (the
    # tool-suppressed content path) rather than through _fire_stream_delta.
    if _stream_writer_superseded(self):
        return
    if isinstance(text, str) and text:
        self._current_streamed_assistant_text = (
            getattr(self, "_current_streamed_assistant_text", "") + text
        )

def _normalize_interim_visible_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def _interim_content_was_streamed(self, content: str) -> bool:
    import agent.provider_runtime as provider_runtime
    visible_content = _normalize_interim_visible_text(
        message_protocol.strip_think_blocks(self, content or "")
    )
    if not visible_content:
        return False
    streamed = _normalize_interim_visible_text(
        message_protocol.strip_think_blocks(self, getattr(self, "_current_streamed_assistant_text", "") or "")
    )
    # Prefix match (not exact equality): the final response may be the
    # streamed text plus a trailing delta, or the stream may have been
    # partial when the verify nudge fired.  In both cases the streamed
    # content is a prefix of the final — that's enough to mark it
    # previewed (fails safe to a benign duplicate, never loses text).
    # The reverse direction (streamed longer than final) is NOT matched:
    # that could suppress a needed resend in the gateway path where
    # already_streamed=True calls on_segment_break() instead of
    # on_commentary() (#65919 review).
    return bool(streamed) and visible_content.startswith(streamed)

def _extract_codex_interim_visible_parts(
    self,
    assistant_msg: Dict[str, Any],
) -> List[str]:
    """Extract visible Codex commentary as one string per message item.

    Codex Responses can keep user-facing mid-turn narration as structured
    ``phase=commentary`` message items while final answer text remains in
    assistant ``content``.  Non-streaming gateway surfaces need that
    commentary through the interim assistant callback before tool calls run.
    ``phase=analysis`` remains hidden because it is provider scratchpad.
    """
    import agent.provider_runtime as provider_runtime
    if not getattr(self, "show_commentary", True):
        # display.show_commentary=false — commentary stays on the
        # reasoning channel (pre-commentary-channel behavior).
        return []
    items = assistant_msg.get("codex_message_items")
    if not isinstance(items, list):
        return []

    messages: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        phase = item.get("phase")
        if not isinstance(phase, str) or phase.strip().lower() != "commentary":
            continue
        content_parts = item.get("content")
        if not isinstance(content_parts, list):
            continue
        item_parts: List[str] = []
        for part in content_parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                item_parts.append(text)
        visible = "".join(item_parts).strip()
        if visible:
            visible = message_protocol.strip_think_blocks(self, visible).strip()
            visible = redact_sensitive_text(visible)
        if visible:
            messages.append(visible)
    return messages

def _extract_codex_interim_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
    """Extract all visible Codex commentary for comparison/fallback."""
    return "\n\n".join(
        _extract_codex_interim_visible_parts(self, assistant_msg)
    ).strip()

def _interim_assistant_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
    """Return the exact assistant text eligible for interim delivery.

    Prefer structured Codex commentary over top-level content. A Codex
    response can contain both commentary and a partial/final-answer message
    while tools are still pending; treating top-level content as progress
    in that shape leaks the answer before the tool call runs.

    Content may be a string or a structured parts list (e.g. after vision
    turns or context compaction), so flatten it before stripping reasoning.
    """
    import agent.provider_runtime as provider_runtime
    visible = _extract_codex_interim_visible_text(self, assistant_msg)
    if visible:
        return visible
    content = assistant_msg.get("content")
    return message_protocol.strip_think_blocks(self, flatten_message_text(content)).strip()

def _interim_text_was_delivered(self, text: str) -> bool:
    normalized = _normalize_interim_visible_text(text)
    if not normalized:
        return False
    return normalized in getattr(self, "_delivered_interim_texts", set())

def _record_delivered_interim_text(self, text: str) -> None:
    normalized = _normalize_interim_visible_text(text)
    if normalized:
        delivered = getattr(self, "_delivered_interim_texts", None)
        if not isinstance(delivered, set):
            delivered = set()
            self._delivered_interim_texts = delivered
        delivered.add(normalized)

def _fire_streamed_codex_commentary(self, text: str) -> None:
    """Deliver a completed live Codex commentary message immediately."""
    import agent.provider_runtime as provider_runtime
    cb = getattr(self, "interim_assistant_callback", None)
    if cb is None or not isinstance(text, str):
        return
    visible = message_protocol.strip_think_blocks(self, text).strip()
    if visible:
        visible = redact_sensitive_text(visible)
    if not visible or visible == "(empty)" or _interim_text_was_delivered(self, visible):
        return
    try:
        cb(visible, already_streamed=False)
        _record_delivered_interim_text(self, visible)
    except Exception:
        logger.debug("interim_assistant_callback error", exc_info=True)

def _emit_interim_assistant_message(
    self, assistant_msg: Dict[str, Any]
) -> None:
    """Surface a real mid-turn assistant commentary message to the UI layer.

    Does NOT set ``_response_was_previewed`` — that flag means "the final
    response was already shown to the user," but this helper is called for
    ordinary tool-call narration, intermediate acknowledgements, and
    verification candidates alike. Setting it here would cause the CLI to
    suppress a *different* final summary (e.g. from ``_handle_max_iterations``)
    when the only streamed text was unrelated mid-turn commentary. (#65919
    review: response-loss blocker)
    """
    if not isinstance(assistant_msg, dict):
        return
    commentary_parts = _extract_codex_interim_visible_parts(self, assistant_msg)
    undelivered_parts: List[str] = []
    pending_keys: set[str] = set()
    for part in commentary_parts:
        key = _normalize_interim_visible_text(part)
        if (
            not key
            or key in pending_keys
            or _interim_text_was_delivered(self, part)
        ):
            continue
        pending_keys.add(key)
        undelivered_parts.append(part)
    visible = (
        "\n\n".join(undelivered_parts).strip()
        if commentary_parts
        else _interim_assistant_visible_text(self, assistant_msg)
    )
    if (
        not visible
        or visible == "(empty)"
        or _interim_text_was_delivered(self, visible)
    ):
        return
    already_streamed = _interim_content_was_streamed(self, visible)
    try:
        from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

        enqueue_plugin_stream_hook(
            "on_interim_message",
            turn_id=getattr(self, "_current_turn_id", "") or "",
            iteration=int(getattr(self, "_api_call_count", 0) or 0),
            session_id=self.session_id or "",
            model=self.model or "",
            provider=self.provider or "",
            surface=self.platform or "cli",
            text=visible,
            already_streamed=already_streamed,
        )
    except Exception:
        logger.debug("on_interim_message plugin hook enqueue failed", exc_info=True)
    cb = getattr(self, "interim_assistant_callback", None)
    if cb is None:
        return
    try:
        cb(visible, already_streamed=already_streamed)
        if undelivered_parts:
            for part in undelivered_parts:
                _record_delivered_interim_text(self, part)
        else:
            _record_delivered_interim_text(self, visible)
    except Exception:
        logger.debug("interim_assistant_callback error", exc_info=True)

def _ensure_stream_writer_state(self) -> None:
    """Lazily create the single-writer guard fields (#65991).

    The fields are normally set in ``agent_init``, but agents constructed
    via ``AgentState.__new__`` (test doubles, legacy/partially-initialized
    instances) skip that path. Claiming/checking the writer must not crash
    those agents, so initialize the fields on first use.
    """
    if getattr(self, "_stream_writer_lock", None) is None:
        self._stream_writer_lock = threading.Lock()
    if not hasattr(self, "_stream_writer_token"):
        self._stream_writer_token = 0

    if getattr(self, "_stream_writer_tls", None) is None:
        self._stream_writer_tls = threading.local()
    if not hasattr(self, "_stream_writer_dropped"):
        self._stream_writer_dropped = 0

def _claim_stream_writer(self) -> int:
    """Claim exclusive ownership of the streaming delta sink for the calling
    stream attempt and return its monotonic writer token (#65991).

    Every streaming attempt (each provider path, each retry) calls this
    right before it begins consuming its stream. Claiming bumps the shared
    token, so any earlier attempt still alive on another thread is
    immediately superseded: its cached token no longer matches and the sink
    fences its late chunks out. The token is stored per-thread, so a thread
    that never claimed (a non-streaming caller) is never treated as a
    writer and can never be fenced.
    """
    _ensure_stream_writer_state(self)
    with self._stream_writer_lock:
        self._stream_writer_token += 1
        token = self._stream_writer_token
    self._stream_writer_tls.token = token
    return token

def _stream_writer_is_current(self, token: int) -> bool:
    """True when ``token`` (from a prior _claim_stream_writer) is still the
    active writer — i.e. no newer stream attempt has claimed the sink since
    (#65991). Lets a stream loop bail out the instant it is superseded."""
    return token == getattr(self, "_stream_writer_token", token)

def _stream_writer_superseded(self) -> bool:
    """True when the calling thread claimed the delta sink but a newer
    stream attempt has since claimed it — i.e. this thread is a stale
    writer whose chunks must be dropped (#65991).

    A thread that never claimed (``token is None``) is not a writer and is
    never reported as superseded, so non-streaming delta callers are
    unaffected.
    """
    tls = getattr(self, "_stream_writer_tls", None)
    token = getattr(tls, "token", None) if tls is not None else None
    if token is None:
        return False
    return token != getattr(self, "_stream_writer_token", token)

def _note_dropped_stream_writer(self, where: str) -> None:
    """Record + log that a superseded stream's delta was discarded."""
    try:
        self._stream_writer_dropped = int(getattr(self, "_stream_writer_dropped", 0)) + 1
    except Exception:
        self._stream_writer_dropped = 1
    # Log sparsely (first drop, then powers of two) so a chatty superseded
    # stream can't flood the log, but a real provider problem is still
    # visible. A silent discard would hide genuine failures.
    _n = self._stream_writer_dropped
    if _n == 1 or (_n & (_n - 1)) == 0:
        logger.warning(
            "Dropped delta from a superseded stream writer at %s "
            "(discarded=%d this turn) — a stale stream tried to write into "
            "the turn after a retry superseded it.",

            where, _n,
        )

def _stream_hook_base_payload(self) -> Dict[str, Any]:
    return {
        "turn_id": getattr(self, "_current_turn_id", "") or "",
        "iteration": int(getattr(self, "_api_call_count", 0) or 0),
        "session_id": self.session_id or "",
        "model": self.model or "",
        "provider": self.provider or "",
        "surface": self.platform or "cli",
    }

def _emit_stream_start(self) -> None:
    try:
        from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

        enqueue_plugin_stream_hook("on_stream_start", **_stream_hook_base_payload(self))
    except Exception:
        logger.debug("on_stream_start plugin hook enqueue failed", exc_info=True)

def _emit_stream_end(self, *, final_text: str, finished: bool, error: str | None) -> None:
    try:
        from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

        enqueue_plugin_stream_hook(
            "on_stream_end",
            **_stream_hook_base_payload(self),
            final_text=final_text,
            finished=finished,
            error=error,
        )
    except Exception:
        logger.debug("on_stream_end plugin hook enqueue failed", exc_info=True)

def _fire_stream_delta(self, text: str) -> None:
    """Fire all registered stream delta callbacks (display + TTS)."""
    import agent.provider_runtime as provider_runtime
    # Single-writer guard (#65991): a superseded stream must not interleave
    # its tokens into the turn alongside the retry that replaced it.
    if _stream_writer_superseded(self):
        _note_dropped_stream_writer(self, "_fire_stream_delta")
        return
    # If a tool iteration set the break flag, prepend a single paragraph
    # break before the first real text delta.  This prevents the original
    # problem (text concatenation across tool boundaries) without stacking
    # blank lines when multiple tool iterations run back-to-back.
    if getattr(self, "_stream_needs_break", False) and text and text.strip():
        self._stream_needs_break = False
        text = "\n\n" + text
        prepended_break = True
    else:
        prepended_break = False
    if isinstance(text, str):
        # Suppress reasoning/thinking blocks via the stateful
        # scrubber (#17924).  Earlier versions ran _strip_think_blocks
        # per-delta here, which destroyed downstream state machines
        # when a tag was split across deltas (e.g. MiniMax-M2.7
        # sends '<think>' and its content as separate deltas —
        # regex case 2 erased the first delta, so the CLI/gateway
        # state machine never saw the open tag and leaked the
        # reasoning content as regular response text).
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            text = think_scrubber.feed(text or "")
        else:
            # Defensive: legacy callers without the scrubber attribute.
            text = message_protocol.strip_think_blocks(self, text or "")
        # Then feed through the stateful context scrubber so memory-context
        # spans split across chunks cannot leak to the UI (#5719).
        scrubber = getattr(self, "_stream_context_scrubber", None)
        if scrubber is not None:
            text = scrubber.feed(text)
        else:
            # Defensive: legacy callers without the scrubber attribute.
            text = sanitize_context(text)
        # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
        if not prepended_break and not getattr(
            self, "_current_streamed_assistant_text", ""
        ):
            text = text.lstrip("\n")
    if not text:
        return
    callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
    delivered = False
    for cb in callbacks:
        try:
            cb(text)
            delivered = True
        except Exception:
            pass
    try:
        from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

        enqueue_plugin_stream_hook(
            "on_stream_delta",
            **_stream_hook_base_payload(self),
            delta=text,
            kind="text",
        )
    except Exception:
        logger.debug("on_stream_delta plugin hook enqueue failed", exc_info=True)
    if delivered:
        _record_streamed_assistant_text(self, text)

def _fire_reasoning_delta(self, text: str) -> None:
    """Fire reasoning callback if registered."""
    # Single-writer guard (#65991): fence out a superseded stream's
    # reasoning deltas the same way as content deltas.
    if _stream_writer_superseded(self):
        _note_dropped_stream_writer(self, "_fire_reasoning_delta")
        return
    cb = self.reasoning_callback
    if cb is not None:
        try:
            cb(text)
        except Exception:
            pass
    try:
        from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, stream_reasoning_deltas_enabled

        if stream_reasoning_deltas_enabled():
            enqueue_plugin_stream_hook(
                "on_stream_delta",
                **_stream_hook_base_payload(self),
                delta=text,
                kind="reasoning",
            )
    except Exception:
        logger.debug("reasoning on_stream_delta plugin hook enqueue failed", exc_info=True)

def _fire_tool_gen_started(self, tool_name: str) -> None:
    """Notify display layer that the model is generating tool call arguments.

    Fires once per tool name when the streaming response begins producing
    tool_call / tool_use tokens.  Gives the TUI a chance to show a spinner
    or status line so the user isn't staring at a frozen screen while a
    large tool payload (e.g. a 45 KB write_file) is being generated.
    """
    cb = self.tool_gen_callback
    if cb is not None:
        try:
            cb(tool_name)
        except Exception:
            pass

def _has_stream_consumers(self) -> bool:
    """Return True if any streaming consumer is registered."""
    try:
        from agent.plugin_stream_hooks import has_stream_observer_hooks

        if has_stream_observer_hooks():
            return True
    except Exception:
        logger.debug("plugin stream hook consumer check failed", exc_info=True)
    return (
        self.stream_delta_callback is not None
        or getattr(self, "_stream_callback", None) is not None
    )

def _interruptible_streaming_api_call(
    self, api_kwargs: dict, *, on_first_delta: callable = None
):
    """Forwarder — see ``agent.chat_completion_helpers.interruptible_streaming_api_call``."""
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    return interruptible_streaming_api_call(self, api_kwargs, on_first_delta=on_first_delta)
