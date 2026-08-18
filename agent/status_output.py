"""Responsibility-owned agent status output behavior."""

import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from agent.session_activity import ActivityProvenance
from utils import is_truthy_value


logger = logging.getLogger(__name__)

def _safe_print(self, *args, **kwargs):
    """Print that silently handles broken pipes / closed stdout.

    In headless environments (systemd, Docker, nohup) stdout may become
    unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
    can crash cron jobs and lose completed work.

    Internally routes through ``self._print_fn`` (default: builtin
    ``print``) so callers such as the CLI can inject a renderer that
    handles ANSI escape sequences properly (e.g. prompt_toolkit's
    ``print_formatted_text(ANSI(...))``) without touching this method.
    """
    try:
        fn = self._print_fn or print
        fn(*args, **kwargs)
    except (OSError, ValueError):
        pass

def _vprint(self, *args, force: bool = False, **kwargs):
    """Verbose print — suppressed when actively streaming tokens.

    Pass ``force=True`` for error/warning messages that should always be
    shown even during streaming playback (TTS or display).

    During tool execution (``_executing_tools`` is True), printing is
    allowed even with stream consumers registered because no tokens
    are being streamed at that point.

    After the main response has been delivered and the remaining tool
    calls are post-response housekeeping (``_mute_post_response``),
    all non-forced output is suppressed.

    ``suppress_status_output`` is a stricter CLI automation mode used by
    parseable single-query flows such as ``hermes chat -q``. In that mode,
    all status/diagnostic prints routed through ``_vprint`` are suppressed
    so stdout stays machine-readable.
    """
    import agent.stream_runtime as stream_runtime
    if getattr(self, "suppress_status_output", False):
        return
    if not force and getattr(self, "_mute_post_response", False):
        return
    if not force and stream_runtime._has_stream_consumers(self) and not self._executing_tools:
        return
    _safe_print(self, *args, **kwargs)

def _should_start_quiet_spinner(self) -> bool:
    """Return True when quiet-mode spinner output has a safe sink.

    In headless/stdio-protocol environments, a raw spinner with no custom
    ``_print_fn`` falls back to ``sys.stdout`` and can corrupt protocol
    streams such as ACP JSON-RPC. Allow quiet spinners only when either:
    - output is explicitly rerouted via ``_print_fn``; or
    - stdout is a real TTY.
    """
    if self._print_fn is not None:
        return True
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False

def _should_emit_quiet_tool_messages(self) -> bool:
    """Return True when quiet-mode tool summaries should print directly.

    Quiet mode is used by both the interactive CLI and embedded/library
    callers. The CLI may still want compact progress hints when no callback
    owns rendering. Embedded/library callers, on the other hand, expect
    quiet mode to be truly silent.
    """
    return (
        self.quiet_mode
        and not self.tool_progress_callback
        and getattr(self, "platform", "") == "cli"
    )

def _emit_status(self, message: str) -> None:
    """Emit a lifecycle status message to both CLI and gateway channels.

    CLI users see the message via ``_vprint(force=True)`` so it is always
    visible regardless of verbose/quiet mode.  Gateway consumers receive
    it through ``status_callback("lifecycle", ...)``.

    This helper never raises — exceptions are swallowed so it cannot
    interrupt the retry/fallback logic.
    """
    try:
        _vprint(self, f"{self.log_prefix}{message}", force=True)
    except Exception:
        pass
    if self.status_callback:
        try:
            self.status_callback("lifecycle", message)
        except Exception:
            logger.debug("status_callback error in _emit_status", exc_info=True)

def _emit_warning(self, message: str) -> None:
    """Emit a user-visible warning through the same status plumbing.

    Unlike debug logs, these warnings are meant for degraded side paths
    such as auxiliary compression or memory flushes where the main turn can
    continue but the user needs to know something important failed.
    """
    try:
        _vprint(self, f"{self.log_prefix}{message}", force=True)
    except Exception:
        pass
    if self.status_callback:
        try:
            self.status_callback("warn", message)
        except Exception:
            logger.debug("status_callback error in _emit_warning", exc_info=True)

def _warn_context_overflow_blocked(
    self, reason: str, preflight_tokens: int, threshold_tokens: int
) -> None:
    """Surface a deduped warning when the context is over the compression
    threshold but compression is blocked (summary-LLM cooldown or
    anti-thrashing).

    Without this signal the session keeps growing until the model silently
    stops answering — the conversation hits the hard provider token limit
    with no explanation. Centralised here so every caller that checks
    ``should_compress_info`` (turn-context preflight, conversation-loop
    guards) shares identical dedup/reset logic.

    Dedup is on the *kind* of block (``cooldown`` / ``ineffective``), not the
    exact countdown string, so a cooldown ticking down 30→29→… doesn't
    re-fire the warning every turn. The dedup key is cleared when the block
    clears (see ``_clear_context_overflow_warn``), so the warning can fire
    again on the next blocked-over-threshold turn.
    """
    import agent.session_runtime as session_runtime
    _warn_kind = (reason or "unknown").split(":", 1)[0]
    _warn_key = ("ctx_overflow_blocked", _warn_kind)
    if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
        self._last_ctx_overflow_warn = _warn_key
        from agent.conversation_compression import (
            CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
        )
        # cooldown + anti-thrash (ineffective) are both "compression blocked".
        if _warn_kind in ("cooldown", "ineffective"):
            session_runtime._touch_activity(self,
                f"compression blocked ({reason})",
                provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
            )
        _emit_warning(self,
            CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
                tokens=preflight_tokens,
                threshold=threshold_tokens,
                reason=reason,
            )
        )

def _clear_context_overflow_warn(self) -> None:
    """Reset the dedup state for the blocked-overflow warning.

    Call this whenever compression is no longer blocked while the context
    is over threshold (e.g. the cooldown elapsed, or compression ran
    successfully), so the warning can re-fire on the next blocked turn.
    """
    self._last_ctx_overflow_warn = None

def _emit_notice(self, notice) -> None:
    """Fire a structured ``AgentNotice`` to the active driver (TUI / CLI).


    Driver-agnostic: the bound ``notice_callback`` renders it however that
    driver does (TUI status-bar override, CLI console line). Swallows all
    callback errors — a notice must NEVER break the agent loop (D-D fail-open).
    """
    if self.notice_callback:
        try:
            self.notice_callback(notice)
        except Exception:
            logger.debug("notice_callback error in _emit_notice", exc_info=True)

def _emit_notice_clear(self, key: str) -> None:
    """Clear a previously-fired sticky notice by ``key`` (e.g. on recovery)."""
    if self.notice_clear_callback:
        try:
            self.notice_clear_callback(key)
        except Exception:
            logger.debug("notice_clear_callback error in _emit_notice_clear", exc_info=True)

def _emit_wait_notice(self, text: str) -> None:
    """Surface a live wait-state explanation on every driver.

    Long provider waits (slow/overloaded backend, no first byte, reasoning
    model thinking for minutes) used to leave the user staring at a generic
    "cogitating..." spinner with no hint of what the agent was waiting on.
    This helper rewrites the live status line with an explanation:

    - CLI: ``thinking_callback`` updates the prompt_toolkit spinner text.
    - TUI / Desktop: the same callback is bridged to the ``thinking.delta``
      event, which both render as the live spinner/status line.
    - Gateway: ``_touch_activity`` stores the text as the activity
      description, which the "⏳ Working — N min" heartbeat includes.

    Never raises — a wait notice must not break the API-call wait loop.
    """
    import agent.session_runtime as session_runtime
    session_runtime._touch_activity(self, text)
    _thinking_cb = getattr(self, "thinking_callback", None)
    if _thinking_cb:
        try:
            _thinking_cb(text)
        except Exception:
            logger.debug("thinking_callback error in _emit_wait_notice", exc_info=True)

def _buffer_status(self, message: str) -> None:
    """Buffer a retry/fallback status message.

    Stored as a (kind, text) tuple where ``kind`` is one of:
    - ``"status"``  -> replays via ``_emit_status``
    - ``"vprint"``  -> replays via ``_vprint(force=True)``
    - ``"warn"``    -> replays via ``_emit_warning``
    Used to defer noisy retry chatter until we know whether the
    turn ultimately recovered or failed.
    """
    try:
        buf = getattr(self, "_retry_status_buffer", None)
        if buf is None:
            buf = []
            self._retry_status_buffer = buf
        buf.append(("status", message))
    except Exception:
        # Never break the retry loop on a buffer hiccup.
        pass

def _buffer_vprint(self, message: str) -> None:
    """Buffer a vprint(force=True) retry/fallback line."""
    try:
        buf = getattr(self, "_retry_status_buffer", None)
        if buf is None:
            buf = []
            self._retry_status_buffer = buf
        buf.append(("vprint", message))
    except Exception:
        pass

def _clear_status_buffer(self) -> None:
    """Drop buffered retry messages — call on successful recovery."""
    try:
        buf = getattr(self, "_retry_status_buffer", None)
        if buf:
            buf.clear()
    except Exception:
        pass

def _emit_pending_fallback_notice(self) -> None:
    """Surface the one-shot fallback-switch notice on successful recovery.

    A provider/model switch is a durable state change operators must see,
    unlike transient retry chatter that ``_clear_status_buffer`` drops.
    ``try_activate_fallback`` records the switch in
    ``self._pending_fallback_notice``; this emits it exactly once via
    ``_emit_status`` and then clears it, so a successful fallback still
    produces one visible notice.  On terminal failure the buffered switch
    line is flushed instead (and this notice discarded) — see
    ``_flush_status_buffer`` — so the user always sees the switch once.
    """
    try:
        notice = getattr(self, "_pending_fallback_notice", None)
        if notice:
            # Clear before emitting so a (swallowed) callback error can't
            # leave the notice set for a stale re-emit on a later turn.
            self._pending_fallback_notice = None
            _emit_status(self, notice)
    except Exception:
        # Never break the conversation loop on a notice hiccup.
        pass

def _flush_status_buffer(self) -> None:
    """Emit buffered retry messages — call on terminal failure.

    Surfaces the full retry/fallback trace so the user can see what
    was tried before the turn gave up.
    """
    try:
        # The buffered trace already carries the fallback switch line, so
        # drop any one-shot fallback notice to avoid a stale duplicate
        # leaking into a later successful turn.
        self._pending_fallback_notice = None
        buf = getattr(self, "_retry_status_buffer", None)
        if not buf:
            return
        # Drain first so a callback exception doesn't double-emit.
        messages = list(buf)
        buf.clear()
        for kind, msg in messages:
            try:
                if kind == "status":
                    _emit_status(self, msg)
                elif kind == "warn":
                    _emit_warning(self, msg)
                else:
                    _vprint(self, f"{self.log_prefix}{msg}", force=True)
            except Exception:
                pass
    except Exception:
        pass

def _disable_codex_reasoning_replay(
    self,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Disable Responses encrypted reasoning replay and strip cached state.

    Called from the conversation_loop retry path when the provider
    rejects a replayed ``codex_reasoning_items`` blob with HTTP 400
    ``invalid_encrypted_content``.  Sets ``self._codex_reasoning_replay_enabled``
    to ``False`` (consumed by ``codex_responses_adapter._chat_messages_to_responses_input``
    and ``transports/codex.py`` to drop ``reasoning.encrypted_content``
    from subsequent requests) and pops ``codex_reasoning_items`` from
    every assistant message in ``messages`` so they cannot be replayed
    again later in the session.

    Returns a small stats dict ``{"messages": int, "items": int}``
    counting what was stripped — purely for diagnostic logging.
    """
    stripped_messages = 0

    stripped_items = 0
    target_messages = messages if isinstance(messages, list) else []

    for msg in target_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        items = msg.pop("codex_reasoning_items", None)
        if isinstance(items, list) and items:
            stripped_messages += 1
            stripped_items += len(items)

    self._codex_reasoning_replay_enabled = False
    return {"messages": stripped_messages, "items": stripped_items}

def _stream_diag_init() -> Dict[str, Any]:
    """Forwarder — see ``agent.stream_diag.stream_diag_init``."""
    from agent.stream_diag import stream_diag_init
    return stream_diag_init()

def _stream_diag_capture_response(
    self, diag: Dict[str, Any], http_response: Any
) -> None:
    """Forwarder — see ``agent.stream_diag.stream_diag_capture_response``."""
    from agent.stream_diag import stream_diag_capture_response
    stream_diag_capture_response(self, diag, http_response)

def _flatten_exception_chain(error: BaseException) -> str:
    """Forwarder — see ``agent.stream_diag.flatten_exception_chain``."""
    from agent.stream_diag import flatten_exception_chain
    return flatten_exception_chain(error)

def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
    """Return True for malformed provider streaming data from SDK parsers.

    Some Anthropic-compatible streaming providers can send a malformed
    event-stream frame.  The Anthropic SDK surfaces that as a plain
    ``ValueError`` such as ``expected ident at line 1 column 149``.  That
    is provider wire-format trouble, not local request validation, so it
    should follow the same retry path as a truncated JSON body.
    """
    if getattr(self, "api_mode", None) != "anthropic_messages":
        return False
    if not isinstance(error, ValueError):
        return False
    if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
        return False
    message = str(error).strip().lower()
    return "expected ident at line" in message

def _log_stream_retry(
    self,
    *,
    kind: str,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Forwarder — see ``agent.stream_diag.log_stream_retry``."""
    from agent.stream_diag import log_stream_retry
    log_stream_retry(
        self, kind=kind, error=error, attempt=attempt,
        max_attempts=max_attempts, mid_tool_call=mid_tool_call, diag=diag,
    )

def _emit_stream_drop(
    self,
    *,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Forwarder — see ``agent.stream_diag.emit_stream_drop``."""
    from agent.stream_diag import emit_stream_drop
    emit_stream_drop(
        self, error=error, attempt=attempt, max_attempts=max_attempts,
        mid_tool_call=mid_tool_call, diag=diag,
    )

def _emit_auxiliary_failure(self, task: str, exc: BaseException) -> None:
    """Surface a compact warning for failed auxiliary work."""
    import agent.error_reporting as error_reporting
    try:
        detail = error_reporting._summarize_api_error(exc)
    except Exception:
        detail = str(exc)
    detail = (detail or exc.__class__.__name__).strip()
    if len(detail) > 220:
        detail = detail[:217].rstrip() + "..."
    _emit_warning(self, f"⚠ Auxiliary {task} failed: {detail}")

def _current_main_runtime(self) -> Dict[str, str]:
    """Return the live main runtime for session-scoped auxiliary routing."""
    return {
        "model": getattr(self, "model", "") or "",
        "provider": getattr(self, "provider", "") or "",
        "base_url": getattr(self, "base_url", "") or "",
        "api_key": getattr(self, "api_key", "") or "",
        "api_mode": getattr(self, "api_mode", "") or "",
        "auth_mode": getattr(self, "auth_mode", "") or "",
    }

def _check_compression_model_feasibility(self) -> None:
    """Forwarder — see ``agent.conversation_compression.check_compression_model_feasibility``."""
    from agent.conversation_compression import check_compression_model_feasibility
    check_compression_model_feasibility(self)

def _replay_compression_warning(self) -> None:
    """Forwarder — see ``agent.conversation_compression.replay_compression_warning``."""
    from agent.conversation_compression import replay_compression_warning
    replay_compression_warning(self)

def _capture_rate_limits(self, http_response: Any) -> None:
    """Parse x-ratelimit-* headers from an HTTP response and cache the state.

    Called after each streaming API call.  The httpx Response object is
    available on the OpenAI SDK Stream via ``stream.response``.
    """
    if http_response is None:
        return
    headers = getattr(http_response, "headers", None)
    if not headers:
        return
    try:
        from agent.rate_limit_tracker import parse_rate_limit_headers
        state = parse_rate_limit_headers(headers, provider=self.provider)
        if state is not None:
            self._rate_limit_state = state
    except Exception:
        pass  # Never let header parsing break the agent loop

def get_rate_limit_state(self):
    """Return the last captured RateLimitState, or None."""
    return self._rate_limit_state

def _capture_anthropic_response_headers(self, http_response: Any) -> None:
    """Capture out-of-band state from Anthropic Messages response headers.

    The Anthropic SDK's aggregated ``Message`` drops HTTP headers. Portal
    (and other providers) put rate-limit and credits state there — the same
    families the OpenAI-wire streaming path captures via
    ``stream.response``. Fail-open: each capture swallows its own errors.
    """
    _capture_rate_limits(self, http_response)
    _capture_credits(self, http_response)

def _capture_credits(self, http_response: Any) -> None:
    """Parse x-nous-credits-* headers, cache CreditsState, fire threshold notices.

    Fail-open throughout — header issues never break the agent loop. The PARSE is
    swallowed (any error → treated as a miss → keep last-known). The notice
    EVALUATION/EMIT is a SEPARATE block that WARNS on failure (R1-M2): a bug in the
    depletion-notice path must not vanish silently under the parse swallow.
    """
    # Dev test fixture (HERMES_DEV_CREDITS_FIXTURE): inject a chosen notice state
    # each turn for repeatable testing, bypassing real headers. Throwaway scaffolding.
    try:
        from agent.credits_tracker import dev_fixture_credits_state
        _fixture = dev_fixture_credits_state()
    except Exception:
        _fixture = None
    if _fixture is not None:
        self._credits_state = _fixture
        if self._credits_session_start_micros is None:
            self._credits_session_start_micros = _fixture.remaining_micros
        _latch = getattr(self, "_credits_latch", None)
        if isinstance(_latch, dict):
            # Only seen_below_90 — never seen_grant_unspent (priming it would
            # fire grant_spent on a fixture's first observation, the exact
            # every-session nag the gate exists to prevent).
            _latch["seen_below_90"] = True  # let warn90 fire without a real crossing
        _used = _fixture.used_fraction
        logger.info(
            "credits ▸ [FIXTURE] remaining=%d (%s) · paid=%s · denom=%s · used=%s "
            "(real headers bypassed — `echo clear` / unset HERMES_DEV_CREDITS_FIXTURE to restore)",
            _fixture.remaining_micros,
            _fixture.remaining_usd or "?",
            _fixture.paid_access,
            _fixture.denominator_kind,
            ("%.0f%%" % (_used * 100)) if _used is not None else "n/a",
        )
        _emit_credits_notices(self)
        return
    if http_response is None:
        return
    headers = getattr(http_response, "headers", None)
    if not headers:
        return
    _dev = is_truthy_value(os.environ.get("HERMES_DEV_CREDITS"))

    # ── Parse (fail-open → miss; never overwrite good state with None) ──
    try:
        from agent.credits_tracker import parse_credits_headers
        state = parse_credits_headers(headers, provider=self.provider)
    except Exception:
        return  # parse error → treat as a miss, keep last-known
    if state is None:
        if _dev:
            logger.info(
                "credits ▸ response had no valid x-nous-credits-* headers "
                "(miss — producer off / non-Nous path / >TTL stale)"
            )
        return

    # retain-last-known: only overwrite on a fresh valid parse
    self._credits_state = state
    # Latch session-start remaining the first time we ever see a header
    if self._credits_session_start_micros is None:
        self._credits_session_start_micros = state.remaining_micros
    if _dev:
        # HERMES_DEV_CREDITS: stream each capture to agent.log — watch live with
        # `hermes logs -f` (grep 'credits ▸'). Dev-only; silent for normal users.
        spent = get_credits_spent_micros(self)
        used = state.used_fraction
        logger.info(
            "credits ▸ remaining=%d (%s) · paid=%s · denom=%s · used=%s "
            "· Δspent=%s · age=%s%s",
            state.remaining_micros,
            state.remaining_usd or "?",
            state.paid_access,
            state.denominator_kind,
            ("%.0f%%" % (used * 100)) if used is not None else "n/a",
            ("%.1f¢" % (spent / 10000)) if spent is not None else "n/a",
            ("%.0fs" % state.age_seconds) if state.age_seconds != float("inf") else "n/a",
            (" · disabled=%s" % state.disabled_reason) if state.disabled_reason else "",
        )

    # Threshold notices — shared with the cold-start seed (see _emit_credits_notices).
    _emit_credits_notices(self)

def _emit_credits_notices(self) -> None:
    """Run the threshold policy on the current credits state and emit notices.

    Shared by the warm path (_capture_credits) and the L3 cold-start seed, so a
    session that opens already depleted warns immediately — not only after the first
    inference header. Runs only when a notice consumer is bound (messaging binds none
    → state still cached for /usage, no policy). WARNS on failure rather than
    swallowing (R1-M2): a depletion-path bug must not vanish silently. Emits clears
    FIRST, then shows (so depleted lands last in a latest-wins slot).
    """
    if getattr(self, "notice_callback", None) is None and getattr(self, "notice_clear_callback", None) is None:
        return
    if not _credits_notices_enabled(self):
        return
    state = getattr(self, "_credits_state", None)
    if state is None:
        return
    try:
        from agent.credits_tracker import evaluate_credits_notices, is_free_tier_model, new_credits_latch
        latch = getattr(self, "_credits_latch", None)
        if latch is None:
            latch = self._credits_latch = new_credits_latch()
        # Free-model gate: a depleted account on a free model can still
        # inference, so the depleted error banner is suppressed. Local-data
        # only (":free" suffix + pricing-cache peek) — never a network call.
        model_is_free = is_free_tier_model(
            getattr(self, "model", "") or "",
            getattr(self, "base_url", "") or "",
        )
        to_show, to_clear = evaluate_credits_notices(state, latch, model_is_free=model_is_free)
        for key in to_clear:        # clears FIRST …
            _emit_notice_clear(self, key)
        for notice in to_show:      # … then shows (depleted lands last in a latest-wins slot)
            _emit_notice(self, notice)
    except Exception:
        logger.warning("credits notice evaluation/emit failed", exc_info=True)

def _credits_notices_enabled(self) -> bool:
    """Whether credits notices are enabled (config display.credits_notices).

    Read once per agent and cached — the policy runs after every API
    response, and the setting governs UI noise, not correctness, so a
    config flip applying on the next session is fine.  Fail-open True
    (preserve current behaviour) on any config error.
    """
    cached = getattr(self, "_credits_notices_enabled_cache", None)
    if cached is not None:
        return cached
    enabled = True
    try:
        from hermes_cli.config import load_config as _load_config
        _cfg = _load_config() or {}
        _display = _cfg.get("display") if isinstance(_cfg, dict) else None
        if isinstance(_display, dict) and "credits_notices" in _display:
            enabled = bool(_display.get("credits_notices"))
    except Exception:
        enabled = True
    self._credits_notices_enabled_cache = enabled
    return enabled

def get_credits_state(self):
    """Return the last captured CreditsState, or None."""
    return self._credits_state

def get_credits_spent_micros(self):
    """Session-cumulative micros spent = first_seen_remaining - current_remaining. None if no data."""
    if self._credits_session_start_micros is None or self._credits_state is None:
        return None
    return self._credits_session_start_micros - self._credits_state.remaining_micros

def _check_openrouter_cache_status(self, http_response: Any) -> None:
    """Read X-OpenRouter-Cache-Status from response headers and log it.

    Increments ``_or_cache_hits`` on HIT so callers can report savings.
    """
    if http_response is None:
        return
    headers = getattr(http_response, "headers", None)
    if not headers:
        return
    try:
        status = headers.get("x-openrouter-cache-status")
        if not status:
            return
        if status.upper() == "HIT":
            self._or_cache_hits += 1
            logger.info("OpenRouter response cache HIT (total: %d)", self._or_cache_hits)
        else:
            logger.debug("OpenRouter response cache %s", status.upper())
    except Exception:
        pass  # Never let header parsing break the agent loop

def get_activity_summary(self) -> dict:
    """Return a snapshot of the agent's current activity for diagnostics.

    Exposes the shared activity observation contract
    (``last_activity_at`` / ``last_activity_description`` /
    ``last_activity_provenance``) plus short aliases
    (``last_activity_ts`` / ``last_activity_desc`` / …) for existing
    gateway and delegate readers.
    """
    from agent.session_activity import (
        ActivityProvenance,
        build_activity_snapshot,
    )

    provenance = getattr(self, "_last_activity_provenance", None)
    if provenance is None:
        provenance = ActivityProvenance.UNKNOWN
    return build_activity_snapshot(
        last_activity_at=getattr(self, "_last_activity_ts", None),
        last_activity_description=getattr(self, "_last_activity_desc", None) or "",
        last_activity_provenance=provenance,
        extra={
        "current_tool": self._current_tool,
        "api_call_count": self._api_call_count,
        "max_iterations": self.max_iterations,
        "budget_used": self.iteration_budget.used,

        "budget_max": self.iteration_budget.max_total,
        },
    )
