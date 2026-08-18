"""Turn execution, post-turn delivery, media, and session-side effects."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import inspect
import json
import logging
import os
import queue
import re
import threading
import time
from contextvars import copy_context
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast

import agent.interruption as interruption
import agent.status_output as status_output
from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from gateway.config import Platform
from gateway.delivery import resolve_delivery_transport
from gateway.history import _float_env
from gateway.media import (
    _build_media_placeholder,
    _event_media_is_stt_input,
    _probe_audio_duration,
)
from gateway.message_router import (
    STOP_REQUEST_REASON as _INTERRUPT_REASON_STOP,
    _has_platform_display_override,
)
from gateway.notices import (
    _resolve_progress_thread_id,
    non_conversational_metadata as _non_conversational_metadata,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
    build_auto_tts_output_path,
    merge_pending_message_event,
)
from gateway.profile_routing import _multiplex_profile_homes, _profile_runtime_scope
from gateway.runtime_config import (
    _checkpoint_agent_kwargs,
    _credential_pool_for_provider,
    _current_max_iterations,
    _load_gateway_config,
    _platform_config_key,
    _resolve_gateway_model,
    _resolve_runtime_agent_kwargs,
    _resolve_runtime_agent_kwargs_for_provider,
)
from gateway.response_filters import _gateway_platform_value
from gateway.session import SessionContext, SessionSource, build_session_context_prompt
from gateway.session_state import AGENT_PENDING as _AGENT_PENDING_SENTINEL
from gateway.turn_context import TurnContext
from gateway.turn_processes import (
    _INTERRUPT_REASON_TIMEOUT,
    _abandon_timed_out_gateway_turn,
    _reap_gateway_turn_processes,
    _watch_gateway_turn_inactivity,
)
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home
from utils import base_url_hostname

logger = logging.getLogger("gateway.run")
_hermes_home = get_hermes_home()
_UNSET = object()
_INTERRUPT_REASON_RESET = "Session reset requested"
_INTERRUPT_REASON_SSE_DISCONNECT = "SSE client disconnected"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"
_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"
_CONTROL_INTERRUPT_MESSAGES = frozenset(
    {
        _INTERRUPT_REASON_STOP.lower(),
        _INTERRUPT_REASON_RESET.lower(),
        _INTERRUPT_REASON_TIMEOUT.lower(),
        _INTERRUPT_REASON_SSE_DISCONNECT.lower(),
        _INTERRUPT_REASON_GATEWAY_SHUTDOWN.lower(),
        _INTERRUPT_REASON_GATEWAY_RESTART.lower(),
    }
)

def _dequeue_pending_event(adapter, session_key: str) -> MessageEvent | None:
    """Consume and return the full pending event for a session.

    Queued follow-ups must preserve their media metadata so they can re-enter
    the normal image/STT/document preprocessing path instead of being reduced
    to a placeholder string.
    """
    return adapter.get_pending_message(session_key)

def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    normalized = " ".join(str(message).strip().split()).lower()
    return normalized in _CONTROL_INTERRUPT_MESSAGES

def _strip_response_attachments_for_direct_send(response: str, adapter) -> str:
    """Return the visible text portion of a response before direct send().

    Queued follow-up resends only replay explicit ``MEDIA:`` attachments in
    this path. Keep bare local paths and ordinary image URLs visible because
    the post-stream uploader intentionally ignores them (#20834).

    Do not apply a broad ``MEDIA:`` regex after ``extract_media()`` — the
    extractor deliberately preserves protected code/inline spans and
    unsupported or unvalidated tags in the cleaned text.
    """
    _, cleaned = adapter.extract_media(response)
    cleaned = cleaned.replace("[[audio_as_voice]]", "").strip()
    cleaned = cleaned.replace("[[as_document]]", "").strip()
    return cleaned.strip()

def _preserve_queued_followup_history_offset(
    current_result: dict,
    followup_result: dict,
) -> dict:
    """Carry the outer history offset through queued follow-up drains.

    ``_process_message_background()`` persists transcript rows only once, after the
    entire in-band queued-follow-up chain returns.  Each recursive ``_run_agent()``
    call advances ``history_offset`` to the history it received, so without
    correction the outermost persistence step sees only the *last* queued turn as
    "new" and silently drops earlier turns from the same drain chain.

    Preserve the earliest (outermost) history offset so the final transcript slice
    still includes every queued turn that ran during the chain.
    """
    if not isinstance(followup_result, dict):
        return followup_result
    if not isinstance(current_result, dict):
        return followup_result

    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if not isinstance(current_offset, int):
        return followup_result
    if isinstance(followup_offset, int) and followup_offset <= current_offset:
        return followup_result

    merged = dict(followup_result)
    merged["history_offset"] = current_offset
    return merged

class TurnExecution:
    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0
    _APPROVAL_TIMEOUT_SECONDS = 300
    _UPDATE_ALLOWED_PLATFORMS = frozenset(
        {Platform.TELEGRAM, Platform.MATTERMOST, Platform.LOCAL}
    )
    _MAX_INTERRUPT_DEPTH = 3
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"),
        ("model", "max_tokens"),
        ("compression", "enabled"),
        ("compression", "progress_notices"),
        ("compression", "threshold"),
        ("compression", "model_thresholds"),
        ("compression", "threshold_tokens"),
        ("compression", "codex_gpt55_autoraise"),
        ("compression", "codex_app_server_auto"),
        ("compression", "target_ratio"),
        ("compression", "protect_last_n"),
        ("compression", "proactive_prune_tokens"),
        ("compression", "proactive_prune_min_result_chars"),
        ("compression", "proactive_prune_min_reclaim_tokens"),
        ("compression", "min_tail_user_messages"),
        ("agent", "disabled_toolsets"),
        ("memory", "provider"),
        ("checkpoints", "enabled"),
        ("checkpoints", "max_snapshots"),
        ("checkpoints", "max_total_size_mb"),
        ("checkpoints", "max_file_size_mb"),
    )
    _HONCHO_CACHE_BUSTING_KEYS = (
        "honcho.peer_name",
        "honcho.ai_peer",
        "honcho.pin_peer_name",
        "honcho.runtime_peer_prefix",
        "honcho.user_peer_aliases",
    )
    _HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}

    def _reset_notice_session_info(self, source: SessionSource) -> str:
        """Session-info block for the auto-reset notice, profile-scoped.

        When multiplexing, resolve model/provider/context inside the profile
        serving ``source`` — otherwise the banner advertises the base config's
        model while the session actually runs on the profile's (#59003).
        Mirrors ``_run_agent``'s gating so single-profile gateways never
        enter the scope.

        Call via ``asyncio.to_thread`` from async handlers: under the scope,
        resolution can do blocking work (credential refresh, context-length
        HTTP probes) that must not run on the event loop. The scope is entered
        inside this method, so contextvars behave correctly in the worker
        thread.
        """
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return self._format_session_info()
        return self._format_session_info()

    def _format_session_info(self) -> str:
        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway
        users can immediately see if context detection went wrong (e.g.
        local models falling to the 128K default).
        """
        from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

        model = _resolve_gateway_model()
        config_context_length = None
        provider = None
        base_url = None
        api_key = None
        custom_provs = None
        data = None
        configured_model = None
        configured_provider = None
        configured_base_url = None

        try:
            data = _load_gateway_config()
            if data:
                model_cfg = data.get("model", {})
                if isinstance(model_cfg, dict):
                    configured_model = model_cfg.get("default") or model_cfg.get("model")
                    raw_ctx = model_cfg.get("context_length")
                    if raw_ctx is not None:
                        try:
                            config_context_length = int(raw_ctx)
                        except (TypeError, ValueError):
                            pass
                    provider = model_cfg.get("provider") or None
                    base_url = model_cfg.get("base_url") or None
                    configured_provider = provider
                    configured_base_url = base_url
                try:
                    from hermes_cli.config import get_compatible_custom_providers
                    custom_provs = get_compatible_custom_providers(data)
                except Exception:
                    custom_provs = data.get("custom_providers")
        except Exception:
            pass

        # Resolve runtime credentials for probing
        try:
            runtime = _resolve_runtime_agent_kwargs()
            provider = runtime.get("provider") or provider
            base_url = runtime.get("base_url") or base_url
            api_key = runtime.get("api_key")
        except Exception:
            pass

        if config_context_length is not None:
            try:
                from hermes_cli.route_identity import should_clear_context_pin

                if should_clear_context_pin(
                    configured_model,
                    model,
                    configured_base_url,
                    base_url,
                    configured_provider,
                    provider,
                ):
                    config_context_length = None
            except Exception:
                config_context_length = None

        if config_context_length is None and custom_provs and base_url:
            try:
                from hermes_cli.config import get_custom_provider_context_length

                custom_ctx = get_custom_provider_context_length(
                    model=model,
                    base_url=base_url,
                    custom_providers=custom_provs,
                )
                if custom_ctx:
                    config_context_length = custom_ctx
            except Exception:
                pass

        context_length = get_model_context_length(
            model,
            base_url=base_url or "",
            api_key=api_key or "",
            config_context_length=config_context_length,
            provider=provider or "",
            custom_providers=custom_provs,
        )

        # Format context source hint
        if config_context_length is not None:
            ctx_source = "config"
        elif context_length == DEFAULT_FALLBACK_CONTEXT:
            ctx_source = "default — set model.context_length in config to override"
        else:
            ctx_source = "detected"

        # Format context length for display
        if context_length >= 1_000_000:
            ctx_display = f"{context_length / 1_000_000:.1f}M"
        elif context_length >= 1_000:
            ctx_display = f"{context_length // 1_000}K"
        else:
            ctx_display = str(context_length)

        lines = [
            f"◆ Model: `{model}`",
            f"◆ Provider: {provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]

        # Show endpoint for local/custom setups
        if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
            lines.append(f"◆ Endpoint: {base_url}")

        return "\n".join(lines)

    def _check_slash_access(
        self, source: SessionSource, canonical_cmd: str
    ) -> Optional[str]:
        """Return a denial message if ``source`` cannot run ``canonical_cmd``,
        else None. Used by both the cold and running-agent dispatch paths
        in ``_handle_message`` so admin/user gating can't be bypassed by
        an in-flight agent.

        Backward-compat semantics live in
        :func:`gateway.slash_access.policy_for_source` — when the operator
        hasn't set ``allow_admin_from`` for the scope, the policy returns
        ``enabled=False`` and this method always returns None.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        if not canonical_cmd:
            return None
        policy = _policy_for_source(self.config, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd,
            source.platform.value if source.platform else "?",
            source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: "
                + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "")
                + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this "
                "platform. Ask an admin to add you to allow_admin_from "
                "or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"

    def _sibling_thread_run_keys(self, source: SessionSource, own_key: str) -> list:
        """Find running-agent keys for OTHER participants in the same thread.

        Only applies when the message originates in a thread.  In per-user
        thread mode (``thread_sessions_per_user=True``) each participant gets
        an isolated session key of the form
        ``agent:main:{platform}:{chat_type}:{chat_id}:{thread_id}:{user_id}``,
        so a run started by another user is invisible to the caller's own
        ``/stop``.  This returns the keys of any *actually running* agents
        (not the pending sentinel, not the caller's own key) whose key shares
        the caller's ``{chat_id}:{thread_id}`` prefix.

        Returns an empty list when the source is not in a thread, or when no
        sibling runs exist — callers must still gate on authorization.
        """
        thread_id = getattr(source, "thread_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not thread_id or not chat_id:
            return []
        platform = source.platform.value
        chat_type = getattr(source, "chat_type", None) or ""
        # Prefix that every per-user key in this thread shares, up to and
        # including the thread_id segment.  Matching either the exact
        # shared-thread key or any key with a further (user_id) segment
        # (prefix + ":") avoids cross-matching an unrelated thread whose id
        # merely starts with this one.
        prefix = ":".join(
            ["agent:main", platform, chat_type, str(chat_id), str(thread_id)]
        )
        matches = []
        for key, agent in self.sessions.running_items():
            if key == own_key:
                continue
            if agent is _AGENT_PENDING_SENTINEL or not agent:
                continue
            if key == prefix or key.startswith(prefix + ":"):
                matches.append(key)
        return matches

    def _is_stale_restart_redelivery(self, event: MessageEvent) -> bool:
        """Return True if this /restart is a Telegram re-delivery we already handled.

        The previous gateway wrote ``.restart_last_processed.json`` with the
        triggering platform + update_id when it processed the /restart.  If
        we now see a /restart on the same platform with an update_id <= that
        recorded value, it is a redelivery when this process booted from that
        restart. Otherwise the marker must still be recent (< 5 minutes).

        Only applies to Telegram today (the only platform that exposes a
        numeric cross-session update ordering); other platforms return False.
        """
        if event is None or event.source is None:
            return False
        if event.platform_update_id is None:
            return False
        if event.source.platform is None:
            return False
        # Only Telegram populates platform_update_id currently; be explicit
        # so future platforms aren't accidentally gated by this check.
        try:
            platform_value = event.source.platform.value
        except Exception:
            return False
        if platform_value != "telegram":
            return False

        try:
            marker_path = _hermes_home / ".restart_last_processed.json"
            if not marker_path.exists():
                # Belt-and-suspenders for when the dedup marker goes missing
                # (manually cleaned up, or the previous cycle's write failed).
                # Without a marker the update_id comparison below can't run, so
                # a redelivered /restart would sail through and re-restart the
                # gateway — an infinite loop (issue #18528).
                #
                # Suppress ONLY when we can independently confirm we just came
                # out of a restart cycle: this process booted from a
                # chat-originated /restart (_booted_from_restart) AND is still
                # within a short post-boot window. This never swallows a
                # genuine first /restart on a fresh boot (no restart marker on
                # boot → flag stays False). Consume the flag one-shot so a
                # legitimate /restart sent later in the same session is honored.
                if (
                    getattr(self, "_booted_from_restart", False)
                    and time.time() - getattr(self, "_startup_time", 0.0) < 60
                ):
                    self._booted_from_restart = False
                    return True
                return False
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        if data.get("platform") != platform_value:
            return False
        recorded_uid = data.get("update_id")
        if not isinstance(recorded_uid, int):
            return False
        if event.platform_update_id > recorded_uid:
            return False

        # A service-managed restart can legitimately take longer than the
        # marker's normal five-minute trust window while adapters, cron, and
        # in-flight deliveries drain. If this process booted from the recorded
        # chat restart, the first same-or-older update is still that restart's
        # redelivery regardless of elapsed wall time. Consume the boot signal
        # one-shot so a later genuine command is evaluated normally.
        if getattr(self, "_booted_from_restart", False):
            self._booted_from_restart = False
            return True

        # Staleness guard: ignore markers older than 5 minutes.  A legitimately
        # old marker (e.g. crash recovery where notify never fired) should not
        # swallow a fresh /restart from the user.
        requested_at = data.get("requested_at")
        if isinstance(requested_at, (int, float)):
            if time.time() - requested_at > 300:
                return False
        return True

    async def _handle_suggestions_command(self, event: MessageEvent) -> str:
        """Handle /suggestions in the gateway.

        Delegates to the shared handler so CLI and gateway never drift. The
        origin is built from the event source so an accepted suggestion's job
        delivers back to this chat/thread.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command

            return handle_suggestions_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("suggestions command failed: %s", e)
            return f"Suggestions command failed: {e}"

    async def _handle_blueprint_command(self, event: MessageEvent):
        """Handle /blueprint in the gateway.

        Delegates to the shared handler so CLI, TUI, and gateway never drift.
        Returns a BlueprintCommandResult: ``text`` is shown to the user, and if
        ``agent_seed`` is set the dispatch site rewrites ``event.text`` to the
        seed and falls through to the agent (the ``/steer`` pattern) so the
        agent gathers the slot values conversationally. Origin is built from the
        event source so a directly created blueprint job delivers back to this chat.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command

            return handle_blueprint_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("blueprint command failed: %s", e)
            from hermes_cli.blueprint_cmd import BlueprintCommandResult

            return BlueprintCommandResult(f"Cron blueprint command failed: {e}")

    def _goal_max_turns_from_config(self) -> int:
        """Resolve the configured /goal turn budget for gateway sessions.

        GatewayRunner.config is a GatewayConfig dataclass, not the full
        user config mapping. Top-level config blocks such as ``goals`` are
        therefore only available through hermes_cli.config.load_config().
        """
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    async def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return a GoalManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)`` or ``(None, None)`` if the
        goals module can't be loaded.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal manager unavailable: %s", exc)
            return None, None
        try:
            # Session lookups on behalf of an internal event must not advance
            # the user-activity clock that drives idle/daily reset policy
            # (same class as the wake fix in _handle_message_with_agent).
            session_entry = await self.async_session_store.get_or_create_session(
                event.source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("goal manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        max_turns = self._goal_max_turns_from_config()
        return GoalManager(session_id=sid, default_max_turns=max_turns), session_entry

    async def _get_heartbeat_manager_for_event(self, event: "MessageEvent"):
        """Return a HeartbeatManager bound to the session for this event.

        Returns ``(manager, session_entry)`` or ``(None, None)``.
        """
        try:
            from hermes_cli.heartbeat import HeartbeatManager
        except Exception as exc:
            logger.debug("heartbeat manager unavailable: %s", exc)
            return None, None
        try:
            # Same reset-policy contract as _get_goal_manager_for_event:
            # internal events look up the session without touching activity.
            session_entry = await self.async_session_store.get_or_create_session(
                event.source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("heartbeat manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        return HeartbeatManager(session_id=sid), session_entry

    def _register_heartbeat_watch(self, quick_key: str, source: Any, session_id: str) -> None:
        """Track a session with an active heartbeat and start the poller.

        The registry maps ``quick_key`` → ``(source, session_id)`` so the
        poller can rebuild a MessageEvent and enqueue via the adapter FIFO.
        In-memory by design: heartbeat STATE survives restarts in SessionDB,
        but firing resumes when the user touches /heartbeat again in the new
        gateway process (documented; durable schedules belong to cron).
        """
        watch = getattr(self, "_heartbeat_watch", None)
        if watch is None:
            watch = {}
            self._heartbeat_watch = watch
        watch[quick_key] = (source, session_id)
        self._start_heartbeat_poller()

    def _unregister_heartbeat_watch(self, quick_key: str) -> None:
        watch = getattr(self, "_heartbeat_watch", None)
        if watch:
            watch.pop(quick_key, None)

    def _start_heartbeat_poller(self) -> None:
        """Start the single gateway-wide heartbeat poll task (idempotent)."""
        existing = getattr(self, "_heartbeat_poll_task", None)
        if existing is not None and not existing.done():
            return

        from hermes_cli.heartbeat import POLL_SECONDS

        async def _poll_loop():
            while True:
                await asyncio.sleep(POLL_SECONDS)
                watch = getattr(self, "_heartbeat_watch", None)
                if not watch:
                    continue
                for quick_key, (source, session_id) in list(watch.items()):
                    try:
                        # Busy sessions coalesce their tick to the next idle poll.
                        if self.sessions.is_running(quick_key):
                            continue
                        from hermes_cli.heartbeat import HeartbeatManager

                        mgr = HeartbeatManager(session_id=session_id)
                        if not mgr.has_heartbeat():
                            watch.pop(quick_key, None)
                            continue
                        prompt = mgr.due_prompt()
                        if not prompt:
                            continue
                        adapter = self._adapter_for_source(source)
                        if adapter is None:
                            continue
                        hb_event = MessageEvent(
                            text=prompt,
                            message_type=MessageType.TEXT,
                            source=source,
                            message_id=None,
                            channel_prompt=None,
                        )
                        self.sessions.enqueue_fifo(quick_key, hb_event, adapter)
                    except Exception as exc:
                        logger.debug("heartbeat poll for %s failed: %s", quick_key, exc)

        try:
            task = asyncio.create_task(_poll_loop())
            self._heartbeat_poll_task = task
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                _bg.add(task)
                task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start heartbeat poller", exc_info=True)

    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None

        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s",
                getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The gateway message handler returns the agent response to the platform
        adapter, which sends it after this method's caller has returned.  For a
        natural Discord/Telegram reading order, goal status belongs after that
        send.  Platform adapters provide a one-shot post-delivery callback for
        exactly this boundary; when unavailable, fall back to direct awaited
        delivery rather than silently dropping the notice.
        """
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(
                    session_key,
                    _deliver,
                    generation=generation,
                )
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)

        await _deliver()

    async def _post_turn_goal_continuation(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn and, if still active,
        enqueue a continuation prompt for the same session.

        Called from ``_handle_message_with_agent`` at turn boundary, AFTER
        the response has been delivered. Safe when no goal is set.

        We use the adapter's pending-message / FIFO machinery so any real
        user message that arrives simultaneously is handled by the same
        queue and takes priority naturally.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal continuation: goals module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        max_turns = self._goal_max_turns_from_config()

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        if not mgr.is_active():
            return

        try:
            from hermes_cli.goals import gather_background_processes as _gather_bg
            _bg_procs = _gather_bg()
        except Exception:
            _bg_procs = None

        # evaluate_after_turn calls judge_goal() which makes a synchronous
        # HTTP request to the auxiliary LLM.  Running it on the event-loop
        # thread would block Discord heartbeats for 10-40 s and cause
        # connection flaps, so we offload it to a thread-pool executor.
        # _run_in_executor_with_context (not bare run_in_executor): the
        # profile secret scope and auxiliary runtime context are contextvars,
        # and a default-executor hop would drop them — aux-client provider
        # resolution would then read credentials unscoped and fail under
        # multiplexing (same pattern as compression in slash_commands.py).
        decision = await self._run_in_executor_with_context(
            lambda: mgr.evaluate_after_turn(
                final_response or "",
                user_initiated=True,
                background_processes=_bg_procs,
            ),
        )
        msg = decision.get("message") or ""

        # Defer the status line until after the adapter has delivered the
        # agent's visible final response. The judge runs after the response is
        # produced but before BasePlatformAdapter sends it, so sending here
        # would show "✓ Goal achieved" before the answer itself. Registering
        # an awaited post-delivery callback preserves delivery reliability
        # without reversing the user-visible ordering.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

        if not decision.get("should_continue"):
            return

        prompt = decision.get("continuation_prompt") or ""
        if not prompt or source is None:
            return

        # Enqueue via the adapter's FIFO so a user message already in
        # flight preempts the continuation naturally.
        try:
            adapter = self._adapter_for_source(source)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                cont_event = MessageEvent(
                    text=prompt,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=None,
                    channel_prompt=None,
                )
                self.sessions.enqueue_fifo(_quick_key, cont_event, adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)

    async def _run_post_turn_hooks(
        self,
        *,
        agent_result: Any,
        source: Any,
        is_internal: bool,
        event: Any = None,
    ) -> None:
        """Run goal and loop bookkeeping after an agent turn returns."""
        final_text = self._final_text_for_post_turn_hooks(agent_result, event)

        try:
            session_entry = await self.async_session_store.get_or_create_session(
                source,
                touch_activity=not is_internal,
            )
        except Exception as exc:
            logger.debug("post-turn session resolution failed: %s", exc)
            return

        # Empty interrupted/errored responses must not drive /goal, but an
        # in-flight /loop tick still needs to be released and rescheduled.
        if final_text.strip():
            try:
                await self._post_turn_goal_continuation(
                    session_entry=session_entry,
                    source=source,
                    final_response=final_text,
                )
            except Exception as exc:
                logger.debug("goal continuation hook failed: %s", exc)
        try:
            await self._post_turn_loop_completion(
                session_entry=session_entry,
                source=source,
                final_response=final_text,
            )
        except Exception as exc:
            logger.debug("loop completion hook failed: %s", exc)

    @staticmethod
    def _final_text_for_post_turn_hooks(agent_result, event=None) -> str:
        """Text for /goal and /loop after a gateway turn.

        Streamed turns return None from _handle_message_with_agent
        (already_sent). The delivered reply is stashed on the event so
        those hooks still see it.
        """
        text = ""
        if isinstance(agent_result, dict):
            text = str(agent_result.get("final_response") or "")
        elif isinstance(agent_result, str):
            text = agent_result
        if text.strip():
            return text
        streamed = getattr(event, "_streamed_final_response", None)
        if isinstance(streamed, str) and streamed.strip():
            return streamed
        return text

    async def _post_turn_loop_completion(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Complete a /loop wakeup tick after a gateway turn.

        No-op unless the session has a loop whose tick is in flight
        (``awaiting_response`` — set when the wakeup was injected). Applies
        the LOOP_COMPLETE marker / --until judge / caps and schedules the
        next tick; the idle wakeup watcher fires it when due.
        """
        try:
            from hermes_cli.loops import LoopManager
        except Exception as exc:
            logger.debug("loop completion: loops module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        mgr = LoopManager(session_id=sid)
        state = mgr.state
        if state is None or not state.awaiting_response:
            return

        # The --until judge is a sync aux-LLM call — keep it off the event loop.
        decision = await asyncio.get_running_loop().run_in_executor(
            None, mgr.complete_tick, final_response or ""
        )
        msg = decision.get("message") or ""
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

    async def _loop_wakeup_watcher(self, interval: float = 15.0) -> None:
        """Fire due /loop wakeups for idle gateway sessions.

        The gateway has no per-session scheduler thread, so a coarse ticker
        scans persisted loops (SessionDB ``loop:*`` rows) and injects the
        wakeup prompt into each due session's chat via the same synthetic-
        message path used by watch notifications. Deferrals:

        - session currently running an agent turn → skip (stays due; the
          adapter FIFO would race the live turn otherwise)
        - active non-parked /goal on the session → skip (goal owns the
          idle boundary)
        - no routing metadata on the loop → skip with a one-time warning
          (CLI/TUI loops carry no route and are driven by their own surfaces)
        """
        await asyncio.sleep(5)  # let platforms finish connecting
        warned_no_route: set = set()
        while self._running:
            try:
                from hermes_cli.loops import (
                    LoopManager,
                    goal_blocks_loop_tick,
                    list_active_loops,
                )

                now = time.time()
                for sid, state in list_active_loops():
                    if state.awaiting_response or now < state.next_due_at:
                        continue
                    route = state.route or {}
                    platform_name = route.get("platform", "")
                    chat_id = route.get("chat_id", "")
                    if not platform_name or not chat_id:
                        # CLI / TUI-owned loop — their own schedulers drive it.
                        continue
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter is None:
                        if sid not in warned_no_route:
                            warned_no_route.add(sid)
                            logger.debug(
                                "loop wakeup: no adapter for platform %r (session %s)",
                                platform_name, sid,
                            )
                        continue

                    # Build the source + session key to check business.
                    evt_stub = {
                        "session_key": "",
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "chat_type": route.get("chat_type", ""),
                        "thread_id": route.get("thread_id", ""),
                        "user_id": route.get("user_id", ""),
                        "user_name": route.get("user_name", ""),
                    }
                    source = self._build_process_event_source(evt_stub)
                    if source is None:
                        continue
                    try:
                        session_key = self._session_key_for_source(source)
                    except Exception:
                        session_key = None
                    if session_key and self.sessions.is_running(session_key):
                        continue  # busy — stays due, next scan retries
                    if goal_blocks_loop_tick(sid):
                        continue

                    mgr = LoopManager(session_id=sid)
                    if not mgr.is_due(now):
                        continue
                    wakeup = mgr.fire_tick()
                    if not wakeup:
                        continue
                    try:
                        synth_event = MessageEvent(
                            text=wakeup,
                            message_type=MessageType.TEXT,
                            source=source,
                            internal=True,
                        )
                        logger.info(
                            "loop wakeup #%s — injecting for %s chat=%s thread=%s",
                            mgr.state.ticks_fired if mgr.state else "?",
                            platform_name, source.chat_id, source.thread_id,
                        )
                        await adapter.handle_message(synth_event)
                        # Slash-command loops dispatch through the command
                        # path and never hit the post-turn completion hook —
                        # complete the tick immediately (caps + scheduling).
                        if wakeup.lstrip().startswith("/"):
                            mgr.complete_tick("")
                    except Exception as exc:
                        logger.warning("loop wakeup injection failed for %s: %s", sid, exc)
                        try:
                            mgr.abandon_tick()
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("loop wakeup watcher error: %s", exc)
            await asyncio.sleep(interval)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
        already_sent: bool = False,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          UNLESS streaming already consumed the response (already_sent=True),
          in which case the base adapter won't have text for auto-TTS so the
          runner must handle it.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_key = self._voice_key(event.source.platform, chat_id)
        voice_mode = self._voice_mode.get(voice_key)
        is_voice_input = (event.message_type == MessageType.VOICE)

        adapter = self.adapters.get(event.source.platform)
        adapter_auto_tts = False
        if adapter and hasattr(adapter, "_should_auto_tts_for_chat"):
            try:
                adapter_auto_tts = bool(adapter._should_auto_tts_for_chat(chat_id))
            except Exception:
                adapter_auto_tts = False

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
            # ``voice.auto_tts`` is synced into the adapter on gateway startup.
            # It is the fallback only when the chat has no explicit mode;
            # otherwise the chat-level all/voice_only/off choice takes precedence.
            or (voice_mode is None and adapter_auto_tts)
        )
        if not should:
            logger.debug(
                "Auto voice reply skipped: mode=%s adapter_auto_tts=%s chat=%s platform=%s",
                voice_mode, adapter_auto_tts, chat_id, event.source.platform.value,
            )
            return False

        # Dedup: agent already called TTS tool in THIS turn only
        last_user_idx = None
        for i, msg in enumerate(reversed(agent_messages)):
            if msg.get("role") == "user":
                last_user_idx = len(agent_messages) - 1 - i; break
        turn_messages = agent_messages[last_user_idx:] if last_user_idx is not None else agent_messages
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                (tc.get("function") or {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in turn_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        # When streaming already delivered the text (already_sent=True),
        # the base adapter will receive None and can't run auto-TTS,
        # so the runner must take over.
        if is_voice_input and not already_sent:
            return False

        return True

    def _should_echo_stt_transcripts(self) -> bool:
        """Return whether inbound voice/STT transcripts should be echoed to chat."""
        return bool(getattr(self.config, "stt_echo_transcripts", True))

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        audio_path = None
        actual_paths: List[str] = []
        try:
            from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text)
            if not tts_text:
                return

            # Telegram voice bubbles require Ogg/Opus; the TTS tool's central
            # container repair guarantees real Ogg/Opus bytes. Other retained
            # adapters keep their selected output format.
            audio_path = build_auto_tts_output_path(event.source.platform)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            try:
                result = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Auto voice reply TTS returned invalid JSON: %s", result_json[:200] if result_json else result_json)
                return

            # Final delivery may be one combined file or multiple separately
            # valid files when combination is unavailable or would exceed a
            # platform limit. Preserve legacy single-file results.
            actual_paths = result.get("file_paths") or [
                result.get("file_path", audio_path)
            ]
            actual_paths = [
                str(path) for path in actual_paths
                if path and os.path.isfile(path)
            ]
            if not result.get("success") or not actual_paths:
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self._adapter_for_source(event.source)

            send_voice = getattr(adapter, "send_voice", None)
            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
            if callable(send_voice):
                # Mark the auto voice reply as notify-worthy.  Mirrors the
                # final-text path in gateway/platforms/base.py which sets
                # ``notify=True`` so platform adapters that gate push
                # notifications (Telegram "important" mode) deliver the
                # final voice reply as a normal notification instead of a
                # silent message.  Clone first so we don't mutate metadata
                # shared with concurrent typing-indicator state.
                if thread_meta is not None:
                    thread_meta = dict(thread_meta)
                    thread_meta["notify"] = True
                else:
                    thread_meta = {"notify": True}
            for actual_path in actual_paths:
                if callable(send_voice):
                    send_voice_call = cast(Callable[..., Awaitable[Any]], send_voice)
                    send_kwargs: Dict[str, Any] = {
                        "chat_id": event.source.chat_id,
                        "audio_path": actual_path,
                        "reply_to": reply_anchor,
                        "metadata": thread_meta,
                    }
                    await send_voice_call(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in ({audio_path, *actual_paths} - {None}):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
        thread_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract explicit MEDIA: tags from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.

        Unlike the non-streaming path in ``gateway/platforms/base.py`` (which
        also auto-detects bare local paths via ``extract_local_files``), this
        post-stream rescan is EXPLICIT-ONLY. The visible reply has already
        been streamed verbatim, so a bare path string here was either (a)
        already shown to the user as text, or (b) stale tool/inspected
        content that was never part of the intended visible reply. Promoting
        such paths into uploads after the fact sent files the model never
        asked to deliver (#20834). Only ``MEDIA:`` directives — the explicit
        attachment contract — trigger post-stream uploads.
        """
        from pathlib import Path
        from urllib.parse import quote as _quote

        try:
            # Capture [[as_document]] before extract_media strips it, so the
            # dispatch partition below can route image-extension files
            # through send_document (preserving bytes) instead of
            # send_multiple_images (Telegram sendPhoto recompresses to ~1280px).
            force_document_attachments = "[[as_document]]" in response

            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            # Do NOT deduplicate explicit MEDIA tags against prior turns here
            # (#73771). This rescan is already EXPLICIT-ONLY (see docstring):
            # a MEDIA: directive in the final streamed reply is the model
            # deliberately attaching a file — including a user-requested
            # resend. Stale auto-appended tags are deduped upstream in
            # _collect_auto_append_media_tags with history_media_paths.
            # Mirrors the same filter removal on the non-streaming path in
            # gateway/platforms/base.py.
            # Strip image URLs from the cleaned text for parity with the
            # non-streaming chain, but do NOT run extract_local_files here:
            # post-stream delivery is explicit-only (#20834). Bare local paths
            # in an already-streamed reply are text the user has seen (or
            # stale inspected content), not an attachment request.
            adapter.extract_images(cleaned)

            _thread_meta = (
                dict(thread_metadata)
                if thread_metadata is not None
                else self._thread_metadata_for_source(
                    event.source,
                    self._reply_anchor_for_event(event),
                )
            )

            _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
            _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

            # Partition out images so they can be sent as a single batch
            # (e.g. Signal's multi-attachment RPC). When [[as_document]] was
            # set, image-extension files skip the photo path and route to
            # send_document below — preserving original bytes.
            image_paths: list = []
            non_image_media: list = []
            for media_path, is_voice in media_files:
                ext = Path(media_path).suffix.lower()
                if (ext in _IMAGE_EXTS
                        and not is_voice
                        and not force_document_attachments):
                    image_paths.append(media_path)
                else:
                    non_image_media.append((media_path, is_voice))

            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(
                        chat_id=event.source.chat_id,
                        images=images,
                        metadata=_thread_meta,
                    )
                except Exception as e:
                    logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)

            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=media_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=media_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

        except Exception as e:
            logger.warning("Post-stream media extraction failed: %s", e)

    async def _deliver_queued_first_response(
        self,
        response: str,
        source: SessionSource,
        adapter,
        metadata: Optional[Dict[str, Any]] = None,
        event_message_id: Optional[str] = None,
        text_already_delivered: bool = False,
        deliver_media: bool = True,
    ) -> None:
        """Deliver a queued response using the normal text+attachment split."""
        if not text_already_delivered:
            text_content = _strip_response_attachments_for_direct_send(response, adapter)
            if text_content:
                await adapter.send(
                    source.chat_id,
                    text_content,
                    metadata=metadata,
                )

        # Failed turns still deliver their (normalized failure) text above,
        # but must not upload attachments as if the turn succeeded — mirrors
        # the ``not agent_result.get("failed")`` guard on the completed-turn
        # delivery path.
        if not deliver_media:
            return

        synthetic_event = MessageEvent(
            text="",
            source=source,
            message_id=event_message_id,
        )
        await self._deliver_media_from_response(
            response,
            synthetic_event,
            adapter,
            thread_metadata=metadata,
        )

    async def _run_background_task(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Profile-scoping wrapper around the background agent task.

        When multiplexing is active, resolve the inbound source's profile and
        run the whole task inside ``_profile_runtime_scope`` so credentials
        resolve from that profile's secret scope. Mirrors the pattern in
        ``_run_agent``.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

    def _resolve_enabled_toolsets(
        self,
        user_config: dict,
        platform_key: str,
    ) -> list:
        """Resolve enabled toolsets for a retained gateway platform."""
        from hermes_cli.tools_config import _get_platform_tools
        return sorted(_get_platform_tools(user_config, platform_key))

    async def _run_background_task_inner(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        import agent.lifecycle as lifecycle
        from agent.agent_init import create_agent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            enabled_toolsets = self._resolve_enabled_toolsets(user_config, platform_key)
            agent_cfg = user_config.get("agent") or {}
            from agent.skill_utils import parse_config_string_list

            disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None

            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(
                source=source, model=model
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._resolve_session_service_tier(source=source)
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = create_agent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    **_checkpoint_agent_kwargs(user_config),
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    # Reload from disk — do not reuse the startup snapshot (#60955).
                    fallback_providers=self._refresh_fallback_providers(),
                )
                try:
                    return lifecycle.run_conversation(agent,
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )
                    except Exception:
                        pass

                # Send media files, routing each by type so a TTS clip
                # arrives as a voice bubble / a clip as a video rather than
                # a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass

    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        adapter = self._adapter_for_source(source)
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            if isinstance(me, dict):
                return me.get(name)
            return None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id:
            return

        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return

        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return

        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Return a Bot API-safe forum topic name from a generated session title."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        # Telegram forum topic names are short (currently 1-128 chars). Keep
        # extra room for multi-byte titles and avoid trailing ellipsis churn.
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    async def _rename_telegram_topic_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return

        # Operator can fully disable per-topic auto-rename via
        # extra.disable_topic_auto_rename. Useful when topics are managed
        # by the user (ad-hoc Threaded Mode) and auto-rename would
        # overwrite their chosen names every time the auto-title fires.
        if self._telegram_topic_auto_rename_disabled(source):
            return

        # Skip rename when the topic is operator-declared via
        # extra.dm_topics. Those topics have fixed names chosen by the
        # operator (plus optional skill binding); auto-renaming would
        # silently mutate operator config.
        #
        # Check the class, not the instance — getattr() on MagicMock
        # auto-creates attributes, so `hasattr(adapter, "_get_dm_topic_info")`
        # would return True for every test double.
        adapter = self._adapter_for_source(source)
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only treat dict-shaped returns as operator-declared; a
                # bare MagicMock or other sentinel shouldn't count.
                if isinstance(operator_topic, dict):
                    return

        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return

        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    name=topic_name,
                )
                return

            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) if bot is not None else None
            if edit_forum_topic is None:
                edit_forum_topic = getattr(bot, "editForumTopic", None) if bot is not None else None
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id),
                    message_thread_id=int(source.thread_id),
                    name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id,
                    message_thread_id=source.thread_id,
                    name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """Return True when operator disabled per-topic auto-rename for this Telegram chat.

        Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
        Default is False (auto-rename enabled, preserves prior behaviour).
        """
        platform_cfg = (
            self.config.platforms.get(source.platform)
            if getattr(self, "config", None) and getattr(self.config, "platforms", None)
            else None
        )
        if platform_cfg is None:
            return False
        extra = getattr(platform_cfg, "extra", None) or {}
        value = extra.get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _schedule_telegram_topic_title_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_telegram_topic_for_session_title(copied_source, session_id, title),
            loop,
            logger=logger,
            log_message="Telegram topic title rename failed to schedule",
        )
        if future is None:
            return
        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Telegram topic title rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot.

        If a user sends /topic repeatedly while Threads Settings are still
        off, we shouldn't keep re-uploading the screenshot every time.
        """
        if not hasattr(self, "_telegram_capability_hint_ts"):
            self._telegram_capability_hint_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_capability_hint_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S:
            return False
        self._telegram_capability_hint_ts[chat_id] = now
        return True

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        # No-op if never enabled.
        try:
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id,
                user_id=str(source.user_id or ""),
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-chat debounce state so the user doesn't see a stale
        # cooldown on the next activation.
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(chat_id, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )

    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []

        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                session_id = str(session.get("id") or "")
                title = str(session.get("title") or "Untitled session")
                preview = str(session.get("preview") or "").strip()
                line = f"- {title} — `{session_id}`"
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        session_id = await self._session_db.resolve_session_id(raw_session_id.strip())
        if not session_id:
            return f"Session not found: {raw_session_id.strip()}"

        session = await self._session_db.get_session(session_id)
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."

        linked = await self._session_db.is_telegram_session_linked_to_topic(session_id=session_id)
        current_binding = await self._session_db.get_telegram_topic_binding(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
        )
        if linked:
            if not current_binding or current_binding.get("session_id") != session_id:
                return "That session is already linked to another Telegram topic."

        session_key = self._session_key_for_source(source)
        try:
            await self._session_db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=session_key,
                session_id=session_id,
                managed_mode="restored",
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return "That session is already linked to another Telegram topic."
            raise

        title = await self._session_db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(await self._session_db.get_messages(session_id)):
                if message.get("role") == "assistant" and message.get("content"):
                    last_assistant = str(message.get("content"))
                    break
        except Exception:
            last_assistant = None

        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Actually disconnect, reconnect, and notify MCP tool changes.

        Split out from ``_handle_reload_mcp_command`` so the confirmation
        wrapper can invoke the same path whether the user confirmed via
        button, text reply, or has the confirm gate disabled.
        """
        loop = asyncio.get_running_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = [t("gateway.reload_mcp.header")]
            if reconnected:
                lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
            if added:
                lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
            if removed:
                lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            # Refresh cached agents so existing sessions see new MCP tools on
            # their next turn — without this, the user has to `/new` (which
            # discards conversation history) to pick up tools from a server
            # that was just added or reconnected. The user has already
            # consented to the prompt-cache invalidation via the slash-confirm
            # gate in _handle_reload_mcp_command before we reach this point.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools
                _cache = self.agent_cache.entries
                _cache_lock = self.agent_cache.lock
                if _cache_lock is not None and _cache:
                    with _cache_lock:
                        for _sess_key, _entry in list(_cache.items()):
                            _agent = _entry.agent
                            if _agent is None:
                                continue
                            # Preserve each cached agent's build-time toolset
                            # selection EXACTLY: a gateway session built with a
                            # restricted enabled_toolsets (e.g. ["safe"]) must
                            # NOT silently gain tools after a reload. This is the
                            # opposite of the interactive CLI/TUI /reload-mcp,
                            # which is a single user re-applying their own config
                            # edit; gateway agents are per-session and may be
                            # deliberately locked down. (Contract is asserted by
                            # test_reload_mcp_preserves_per_agent_toolset_overrides.)
                            refresh_agent_mcp_tools(_agent, quiet_mode=True)
            except Exception as _exc:
                logger.debug(
                    "Failed to update cached agent tools after MCP reload: %s",
                    _exc,
                )

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)

    async def _maybe_confirm_destructive_slash(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        detail: str,
        execute,
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute`` is an async callable ``execute() -> str | EphemeralReply``
        that performs the destructive action.  If the
        ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
        runs immediately (returning its result).  Otherwise this routes
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``
        """
        # Gate check.
        confirm_required = True
        try:
            cfg = self._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            if choice == "cancel":
                return f"🟡 /{command} cancelled. Conversation unchanged."
            persisted = False
            if choice == "always":
                try:
                    from cli import save_config_value
                    # save_config_value swallows its own errors and reports the
                    # outcome in the return value, so the try block alone says
                    # nothing about whether the write landed.
                    persisted = bool(
                        save_config_value("approvals.destructive_slash_confirm", False)
                    )
                    if persisted:
                        logger.info(
                            "User opted out of destructive slash confirm (session=%s)",
                            session_key,
                        )
                    else:
                        logger.warning(
                            "Could not persist destructive_slash_confirm=false "
                            "(session=%s); config.yaml is not writable",
                            session_key,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist destructive_slash_confirm=false: %s", exc,
                    )
            result = await execute()
            if choice == "always":
                if persisted:
                    note = (
                        "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                        "without confirmation. Re-enable via "
                        "`approvals.destructive_slash_confirm: true` in config.yaml."
                    )
                else:
                    # The user did approve this run, so the action still goes
                    # ahead, but the preference did not stick and the prompt
                    # will be back next time. Say so rather than promising an
                    # opt-out that was never written.
                    note = (
                        "\n\n⚠️ Could not save that preference (config.yaml is not "
                        "writable), so /clear, /new, /reset, and /undo will ask "
                        "again next time. To silence it permanently, set "
                        "`approvals.destructive_slash_confirm: false` in config.yaml."
                    )
                if isinstance(result, str):
                    return result + note
                # EphemeralReply or other: leave untouched, since the note would
                # mangle structured replies.
                return result
            return result

        _p = self._typed_command_prefix_for(event.source.platform)
        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            f"_Text fallback: reply `{_p}approve`, `{_p}always`, or `{_p}cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _request_slash_confirm(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        message: str,
        handler,
    ) -> Optional[str]:
        """Ask the user to confirm an expensive slash command.

        ``handler`` is an async callable ``handler(choice: str) -> str``
        where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
        The handler runs on the event loop when the user responds; its
        return value is sent back as a gateway message.

        Returns a short acknowledgment string to send immediately (before
        the user's response).  If buttons rendered successfully the ack
        is ``None`` (buttons are self-explanatory); if we fell back to
        text the message itself IS the ack.
        """
        from tools import slash_confirm as _slash_confirm_mod

        source = event.source
        session_key = self._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError.  Real runs always
        # have the instance attribute.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self._adapter_for_source(source)
        metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

        used_buttons = False
        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id,
                    title=title,
                    message=message,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    used_buttons = True
            except Exception as exc:
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message

    def _read_user_config(self) -> Dict[str, Any]:
        """Read the user's raw config.yaml (cached) for gate lookups.

        Used by slash-confirm gates that must reflect on-disk state changes
        (e.g. a prior "Always Approve" click) without a gateway restart.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _thread_metadata_for_source(
        self,
        source,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        metadata = self._thread_metadata_for_target(
            getattr(source, "platform", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None),
        )
        return metadata

    def _thread_metadata_for_target(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform,
            chat_id,
            thread_id,
            chat_type=chat_type,
            adapter=adapter,
        ):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # Telegram DM topic lanes need direct_messages_topic_id in metadata
            # so synthetic/queued messages (goal continuations, status notices)
            # route to the correct topic even when reply anchor is unavailable.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Inspect operator-declared DM topics via the adapter's lookup. Resolve
        # the method on the CLASS, not the instance: getattr() on a MagicMock
        # auto-creates a callable child for any attribute, so an instance-level
        # lookup would report a DM topic for every test double. Only a
        # dict-shaped return counts as an operator-declared topic — a bare
        # MagicMock or other sentinel must not. Mirrors the guard in
        # _rename_telegram_topic_for_session_title.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

    @staticmethod
    def _reply_anchor_for_event(event: MessageEvent) -> Optional[str]:
        """Return the platform-specific reply anchor for GatewayRunner sends."""
        return _reply_anchor_for_event(event)

    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return

        try:
            self._update_notification_task = asyncio.create_task(
                self._watch_update_progress()
            )
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    async def _watch_update_progress(
        self,
        poll_interval: float = 2.0,
        stream_interval: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the
        user periodically.  Detects ``.update_prompt.json`` (written by the
        update process when it needs user input) and forwards the prompt to
        the messenger.  The user's next message is intercepted by
        ``_handle_message`` and written to ``.update_response``.
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        prompt_path = _hermes_home / ".update_prompt.json"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # Resolve the adapter and chat_id for sending messages
        adapter = None
        chat_id = None
        session_key = None
        metadata = None
        for path in (claimed_path, pending_path):
            if path.exists():
                try:
                    pending = json.loads(path.read_text(encoding="utf-8"))
                    platform_str = pending.get("platform")
                    chat_id = pending.get("chat_id")
                    chat_type = pending.get("chat_type")
                    session_key = pending.get("session_key")
                    thread_id = pending.get("thread_id")
                    message_id = pending.get("message_id")
                    if platform_str and chat_id:
                        platform = Platform(platform_str)
                        adapter = self.adapters.get(platform)
                        metadata = self._thread_metadata_for_target(
                            platform,
                            chat_id,
                            thread_id,
                            chat_type=chat_type,
                            reply_to_message_id=message_id,
                            adapter=adapter,
                        )
                        # Fallback session key if not stored (old pending files)
                        if not session_key:
                            session_key = f"{platform_str}:{chat_id}"
                    break
                except Exception:
                    pass

        if not adapter or not chat_id:
            logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
            # Fall back to completion-only: wait for the exit code and send the
            # final notification. _send_update_notification re-resolves the
            # adapter on every call, so when the target platform is still
            # reconnecting it returns False and keeps the markers. Keep polling
            # until it actually delivers (returns True) instead of giving up
            # after the first completion check — otherwise a platform that
            # reconnects a few seconds after completion never gets notified.
            while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
                if exit_code_path.exists() and await self._send_update_notification():
                    return
                await asyncio.sleep(poll_interval)
            if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
                exit_code_path.write_text("124", encoding="utf-8")
                await self._send_update_notification()
            return

        def _strip_ansi(text: str) -> str:
            from tools.ansi_strip import strip_ansi
            return strip_ansi(text)

        def _read_output_since(path: Path, offset: int) -> tuple[str, int]:
            """Read update output defensively; logs may contain invalid UTF-8."""
            try:
                data = path.read_bytes()
            except OSError:
                return "", offset
            if len(data) <= offset:
                return "", len(data)
            return data[offset:].decode("utf-8", errors="replace"), len(data)

        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            """Send buffered output to the user."""
            nonlocal buffer, last_stream_time
            if not buffer.strip():
                buffer = ""
                return
            # Chunk to fit message limits (Telegram: 4096, others: generous)
            clean = _strip_ansi(buffer).strip()
            buffer = ""
            last_stream_time = loop.time()
            if not clean:
                return
            # Split into chunks if too long
            max_chunk = 3500
            chunks = [clean[i:i + max_chunk] for i in range(0, len(clean), max_chunk)]
            for chunk in chunks:
                try:
                    await adapter.send(
                        chat_id,
                        f"```\n{chunk}\n```",
                        metadata=_non_conversational_metadata(metadata, platform=platform),
                    )
                except Exception as e:
                    logger.debug("Update stream send failed: %s", e)

        while loop.time() < deadline:
            # Check for completion
            if exit_code_path.exists():
                # Read any remaining output
                if output_path.exists():
                    try:
                        chunk, bytes_sent = _read_output_since(output_path, bytes_sent)
                        if chunk:
                            buffer += chunk
                    except OSError:
                        pass
                await _flush_buffer()

                # Send final status
                try:
                    exit_code_raw = exit_code_path.read_text(encoding="utf-8").strip() or "1"
                    exit_code = int(exit_code_raw)
                    if exit_code == 0:
                        await adapter.send(
                            chat_id,
                            "✅ Hermes update finished.",
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    else:
                        await adapter.send(
                            chat_id,
                            "❌ Hermes update failed (exit code {}).".format(exit_code),
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                except Exception as e:
                    logger.warning("Update final notification failed: %s", e)

                # Cleanup
                for p in (pending_path, claimed_path, output_path,
                          exit_code_path, prompt_path):
                    p.unlink(missing_ok=True)
                (_hermes_home / ".update_response").unlink(missing_ok=True)
                _up_done = self.sessions.peek(session_key)
                if _up_done is not None:
                    _up_done.persistent.update_prompt_pending = False
                return

            # Check for new output
            if output_path.exists():
                try:
                    chunk, bytes_sent = _read_output_since(output_path, bytes_sent)
                    if chunk:
                        buffer += chunk
                except OSError:
                    pass

            # Flush buffer periodically
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()

            # Check for prompts — only forward if we haven't already sent
            # one that's still awaiting a response.  Without this guard the
            # watcher would re-read the same .update_prompt.json every poll
            # cycle and spam the user with duplicate prompt messages.
            _up_pending_state = (
                self.sessions.peek(session_key) if session_key else None
            )
            if (prompt_path.exists() and session_key
                    and not (
                        _up_pending_state is not None
                        and _up_pending_state.persistent.update_prompt_pending
                    )):
                try:
                    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
                    prompt_text = prompt_data.get("prompt", "")
                    default = prompt_data.get("default", "")
                    if prompt_text:
                        # Flush any buffered output first so the user sees
                        # context before the prompt
                        await _flush_buffer()
                        # Try platform-native buttons first (Discord, Telegram)
                        sent_buttons = False
                        if getattr(type(adapter), "send_update_prompt", None) is not None:
                            try:
                                await adapter.send_update_prompt(
                                    chat_id=chat_id,
                                    prompt=prompt_text,
                                    default=default,
                                    session_key=session_key,
                                    metadata=_non_conversational_metadata(metadata, platform=platform),
                                )
                                sent_buttons = True
                            except Exception as btn_err:
                                logger.debug("Button-based update prompt failed: %s", btn_err)
                        if not sent_buttons:
                            default_hint = f" (default: {default})" if default else ""
                            _p = getattr(adapter, "typed_command_prefix", "/")
                            await adapter.send(
                                chat_id,
                                f"⚕ **Update needs your input:**\n\n"
                                f"{prompt_text}{default_hint}\n\n"
                                f"Reply `{_p}approve` (yes) or `{_p}deny` (no), "
                                f"or type your answer directly.",
                                metadata=_non_conversational_metadata(metadata, platform=platform),
                            )
                        # Keep the prompt marker on disk until the user
                        # answers. If the gateway restarts mid-prompt, the
                        # next watcher can recover by re-forwarding it from
                        # disk. Duplicate sends in the same process are
                        # still suppressed by _update_prompt_pending.
                        self.sessions.state(
                            session_key
                        ).persistent.update_prompt_pending = True
                        # .update_response to continue — it doesn't re-check
                        logger.info("Forwarded update prompt to %s: %s", session_key, prompt_text[:80])
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)

            await asyncio.sleep(poll_interval)

        # Timeout
        if not exit_code_path.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            exit_code_path.write_text("124", encoding="utf-8")
            await _flush_buffer()
            try:
                await adapter.send(
                    chat_id,
                    "❌ Hermes update timed out after 30 minutes.",
                    metadata=_non_conversational_metadata(metadata, platform=platform),
                )
            except Exception:
                pass
            for p in (pending_path, claimed_path, output_path,
                      exit_code_path, prompt_path):
                p.unlink(missing_ok=True)
            (_hermes_home / ".update_response").unlink(missing_ok=True)
            _up_timeout_state = self.sessions.peek(session_key)
            if _up_timeout_state is not None:
                _up_timeout_state.persistent.update_prompt_pending = False

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        Returns False when the update is still running so a caller can retry
        later. Returns True after a definitive send/skip decision.

        This is the legacy notification path used when the streaming watcher
        cannot resolve the adapter (e.g. after a gateway restart where the
        platform hasn't reconnected yet).
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"

        if not pending_path.exists() and not claimed_path.exists():
            return False

        cleanup = True
        active_pending_path = claimed_path
        try:
            if pending_path.exists():
                try:
                    pending_path.replace(claimed_path)
                except FileNotFoundError:
                    if not claimed_path.exists():
                        return True
            elif not claimed_path.exists():
                return True

            pending = json.loads(claimed_path.read_text(encoding="utf-8"))
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")
            chat_type = pending.get("chat_type")
            thread_id = pending.get("thread_id")
            message_id = pending.get("message_id")

            if not exit_code_path.exists():
                logger.info("Update notification deferred: update still running")
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            exit_code_raw = exit_code_path.read_text(encoding="utf-8").strip() or "1"
            exit_code = int(exit_code_raw)

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_bytes().decode("utf-8", errors="replace")

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if not adapter and chat_id:
                # The update finished, but the target platform has not
                # reconnected yet (common right after the restart that
                # `hermes update` triggers). Treating "adapter missing" as a
                # definitive skip would delete the markers and silently lose the
                # completion notification — the user never learns whether the
                # update succeeded or timed out. Preserve the markers instead so
                # a later retry (the watcher poll loop, or the next gateway
                # startup) can deliver the result once the adapter is back.
                logger.info(
                    "Update notification deferred: %s adapter not connected yet",
                    platform_str,
                )
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            if adapter and chat_id:
                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=chat_type,
                    reply_to_message_id=message_id,
                    adapter=adapter,
                )
                # Strip ANSI escape codes for clean display
                from tools.ansi_strip import strip_ansi
                output = strip_ansi(output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    if exit_code == 0:
                        msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                    else:
                        msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
                elif exit_code == 0:
                    msg = "✅ Hermes update finished successfully."
                else:
                    msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                await adapter.send(
                    chat_id,
                    msg,
                    metadata=_non_conversational_metadata(metadata, platform=platform),
                )
                logger.info(
                    "Sent post-update notification to %s:%s (exit=%s)",
                    platform_str,
                    chat_id,
                    exit_code,
                )
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                active_pending_path.unlink(missing_ok=True)
                claimed_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                exit_code_path.unlink(missing_ok=True)

        return True

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None

        try:
            data = json.loads(notify_path.read_text(encoding="utf-8"))
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            chat_type = data.get("chat_type")
            thread_id = data.get("thread_id")
            message_id = data.get("message_id")

            if not platform_str or not chat_id:
                return None

            platform = Platform(platform_str)
            transport = resolve_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                logger.debug(
                    "Restart notification skipped: no live transport for %s",
                    platform_str,
                )
                return None

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false",
                    platform_str,
                )
                return None

            metadata = self._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=chat_type,
                reply_to_message_id=message_id,
                adapter=transport.adapter,
            )
            result = await transport.send(
                platform,
                str(chat_id),
                "♻ Gateway restarted successfully. Your session continues.",
                metadata=_non_conversational_metadata(metadata, platform=platform),
            )
            # adapter.send() catches provider errors (e.g. "Chat not found")
            # and returns SendResult(success=False) rather than raising, so
            # we must inspect the result before claiming success — otherwise
            # the log line is misleading and hides real delivery failures.
            if result is not None and getattr(result, "success", True) is False:
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                return None

            logger.info(
                "Sent restart notification to %s:%s",
                platform_str,
                chat_id,
            )
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)

    async def _send_home_channel_startup_notifications(
        self,
        *,
        skip_targets: Optional[set[tuple[str, str, Optional[str]]]] = None,
    ) -> set[tuple[str, str, Optional[str]]]:
        """Notify configured home channels that the gateway is back online.

        The notification is best-effort and sent once per connected platform
        home channel. ``skip_targets`` lets startup avoid duplicate messages
        when a more specific restart notification is queued for the same chat.
        """
        delivered: set[tuple[str, str, Optional[str]]] = set()
        skipped = skip_targets or set()
        message = "♻️ Gateway online — Hermes is back and ready."

        for platform, platform_cfg in self.config.platforms.items():
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue

            transport = resolve_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                continue

            if not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Home-channel startup notification suppressed: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            target = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if target in skipped or target in delivered:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=transport.adapter,
                )
                send_metadata = _non_conversational_metadata(metadata, platform=platform)
                result = await transport.send(
                    platform,
                    str(home.chat_id),
                    message,
                    metadata=send_metadata,
                )
                if result is not None and getattr(result, "success", True) is False:
                    logger.warning(
                        "Home-channel startup notification failed for %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                delivered.add(target)
                logger.info(
                    "Sent home-channel startup notification to %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as exc:
                logger.warning(
                    "Home-channel startup notification failed for %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    exc,
                )

        return delivered

    async def _send_session_db_warning_notifications(self) -> None:
        """Broadcast a state.db failure warning to all home channels (#88235).

        When SessionDB init fails at gateway startup, messages may flow but
        nothing is persisted — /resume, /history, and session_search all
        silently break.  This sends a one-time warning to each connected
        platform's home channel so the user knows to investigate before
        losing data.  Best-effort: failures are logged, not raised.
        """
        error = getattr(self, "_session_db_init_error", None)
        if not error:
            return

        from hermes_state import classify_persistence_error, format_session_db_unavailable

        cause = classify_persistence_error(error)
        hint = format_session_db_unavailable()
        if cause == "corrupt":
            message = (
                "⚠️ Session database corruption detected. Messages may not be "
                "persisted. Recovery options:\n"
                "1. Run `hermes doctor --fix`\n"
                "2. Salvage with: sqlite3 ~/.hermes/state.db \".recover\" "
                "(then replace state.db)\n"
                "3. Restore from a backup in ~/.hermes/backups/\n"
                f"Error: {error}"
            )
        else:
            message = (
                f"⚠️ Session database unavailable — messages may not be persisted. "
                f"{hint}\n"
                f"Run `hermes doctor` for diagnostics."
            )

        logger.warning(
            "Broadcasting state.db failure warning to home channels: %s", error
        )

        for platform, platform_cfg in self.config.platforms.items():
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue
            transport = resolve_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                continue
            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=transport.adapter,
                )
                send_metadata = _non_conversational_metadata(metadata, platform=platform)
                result = await transport.send(
                    platform,
                    str(home.chat_id),
                    message,
                    metadata=send_metadata,
                )
                if result is not None and getattr(result, "success", True) is False:
                    logger.warning(
                        "state.db warning notification failed for %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
            except Exception as exc:
                logger.warning(
                    "state.db warning notification failed for %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    exc,
                )

    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables for the current async task.

        Uses ``contextvars`` instead of ``os.environ`` so that concurrent
        gateway messages cannot overwrite each other's session state.

        Returns a list of reset tokens; pass them to ``_clear_session_env``
        in a ``finally`` block.
        """
        from gateway.session_context import set_session_vars
        # Propagate the adapter's async-delivery capability so async tools
        # (terminal notify_on_complete / watch_patterns, delegate_task
        # background=True) know whether this channel can wake a later turn.
        # Default True keeps CLI / unknown paths working; stateless adapters
        # (api_server) declare supports_async_delivery=False. Use getattr so
        # bare runners built via object.__new__ (tests) without self.adapters
        # don't blow up — they simply default to supported.
        _adapters = getattr(self, "adapters", None) or {}
        _adapter = _adapters.get(context.source.platform)
        _async_delivery = getattr(_adapter, "supports_async_delivery", True)
        return set_session_vars(
            platform=context.source.platform.value,
            chat_id=context.source.chat_id,
            chat_type=(
                str(context.source.chat_type) if context.source.chat_type else ""
            ),
            chat_name=context.source.chat_name or "",
            thread_id=str(context.source.thread_id) if context.source.thread_id else "",
            user_id=str(context.source.user_id) if context.source.user_id else "",
            user_id_alt=str(context.source.user_id_alt) if context.source.user_id_alt else "",
            user_name=str(context.source.user_name) if context.source.user_name else "",
            scope_id=str(getattr(context.source, "scope_id", "") or ""),
            session_key=context.session_key,
            message_id=str(context.source.message_id) if context.source.message_id else "",
            profile=getattr(context.source, "profile", "") or "",
            async_delivery=_async_delivery,
            cron_session="",
        )

    def _clear_session_env(self, tokens: list) -> None:
        """Restore session context variables to their pre-handler values."""
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)

    async def _run_in_executor_with_context(self, func, *args):
        """Run blocking work in the thread pool while preserving session contextvars."""
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(
            self._get_executor(),
            ctx.run,
            func,
            *args,
        )

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the gateway-owned executor for blocking agent work."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._executor_lock = lock

        with lock:
            if getattr(self, "_executor_closing", False):
                raise RuntimeError("Gateway is shutting down; executor unavailable")
            executor = getattr(self, "_executor", None)
            if executor is None or getattr(executor, "_shutdown", False):
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=10,
                    thread_name_prefix="hermes-gateway",
                )
                self._executor = executor
            return executor

    def _shutdown_executor(self) -> None:
        """Stop the gateway-owned executor without touching the loop default."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            return

        with lock:
            self._executor_closing = True
            executor = getattr(self, "_executor", None)
            self._executor = None

        if executor is None:
            return

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    def _decide_image_input_mode(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Resolve image-input routing for the effective model this turn.

        Returns ``"native"`` (attach pixels on the user turn) or ``"text"``
        (pre-analyze with vision_analyze and prepend the description). See
        agent/image_routing.py for the full decision table.

        Gateway sessions can have /model overrides that live outside
        config.yaml. Image preprocessing runs before create_agent sets the
        auxiliary_client runtime globals, so resolve the same per-session
        runtime bundle the upcoming agent turn will use instead of consulting
        only the persisted default model.
        """
        try:
            from agent.image_routing import decide_image_input_mode
            from agent.auxiliary_client import _read_main_model, _read_main_provider
            from hermes_cli.config import load_config

            cfg = user_config if isinstance(user_config, dict) else load_config()
            resolved_provider = (provider or "").strip()
            resolved_model = (model or "").strip()
            resolved_requested_provider = ""

            needs_session_runtime = not resolved_provider or not resolved_model
            has_session_identity = source is not None or session_key
            if needs_session_runtime and has_session_identity:
                try:
                    turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=cfg,
                    )
                    if not resolved_model and isinstance(turn_model, str):
                        resolved_model = turn_model.strip()
                    runtime_provider = runtime_kwargs.get("provider") if isinstance(runtime_kwargs, dict) else None
                    runtime_requested_provider = (
                        runtime_kwargs.get("requested_provider")
                        if isinstance(runtime_kwargs, dict)
                        else None
                    )
                    if not resolved_provider and isinstance(runtime_provider, str):
                        resolved_provider = runtime_provider.strip()
                    if isinstance(runtime_requested_provider, str):
                        resolved_requested_provider = runtime_requested_provider.strip()
                except Exception as exc:
                    logger.debug(
                        "image_routing: session runtime resolution failed, falling back to config — %s",
                        exc,
                    )

            if not resolved_provider:
                resolved_provider = _read_main_provider()
            if not resolved_model:
                resolved_model = _read_main_model()

            return decide_image_input_mode(
                resolved_provider,
                resolved_model,
                cfg,
                requested_provider=resolved_requested_provider,
            )
        except Exception as exc:
            logger.debug("image_routing: decision failed, falling back to text — %s", exc)
            return "text"

    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        from agent.memory_manager import sanitize_context

        analysis_prompt = (
            "Concisely describe this image in 2-4 sentences "
            "(~200 Chinese characters or ~150 English words). "
            "Cover the main subject, key visible text/data/code, and overall context. "
            "If it is a chart, diagram, or scientific figure, include the important "
            "labels, legend, and key values. Skip decorative details."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    description = sanitize_context(description)
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> tuple[str, List[str]]:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            A tuple of ``(enriched_text, successful_transcripts)``:
              - ``enriched_text``: the message string with transcription wrappers
                prepended (same as before).
              - ``successful_transcripts``: the raw transcript strings for audio
                clips that were successfully transcribed, in input order. Empty
                list if every clip failed or STT is disabled. Callers can use
                this to echo transcripts back to the user before the agent loop.
        """
        seen = set()
        audio_paths = [p for p in audio_paths if p not in seen and not seen.add(p)]
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                if duration_str:
                    notes.append(
                        f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                    )
                else:
                    notes.append(f"[The user sent a voice message: {abs_path}]")
            if not notes:
                return user_text, []
            prefix = "\n\n".join(notes)
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, []
            if user_text:
                return f"{prefix}\n\n{user_text}", []
            return prefix, []

        try:
            from tools.transcription_tools import (
                transcribe_audio,
                transcribe_audio_local_fallback,
            )
        except ModuleNotFoundError as e:
            logger.error("Transcription module unavailable: %s", e)
            unavailable_note = "[voice message could not be transcribed]"
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return unavailable_note, []
            if user_text:
                return f"{unavailable_note}\n\n{user_text}", []
            return unavailable_note, []

        enriched_parts = []
        successful_transcripts: List[str] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(
                    transcribe_audio, path, None, "gateway",
                )
                if not result.get("success"):
                    fallback = await asyncio.to_thread(
                        transcribe_audio_local_fallback,
                        path,
                    )
                    if fallback.get("success"):
                        logger.info(
                            "Configured STT failed for %s; recovered with local STT",
                            path,
                        )
                        result = fallback
                if result["success"]:
                    transcript = result["transcript"]
                    # Speech-to-text can return success=True with an empty or
                    # whitespace-only transcript on silence, cut-off, or
                    # inaudible audio. Emitting empty quotes ('""') makes the
                    # agent reply to nothing and can loop, so that case gets a
                    # clear sentinel note instead (#41603).
                    if not (transcript or "").strip():
                        enriched_parts.append(
                            "[The user sent a voice message but it came through "
                            "empty or inaudible — speech-to-text returned no "
                            "words. Do not guess at the content; ask the user "
                            "to resend or type it out.]"
                        )
                        continue
                    successful_transcripts.append(transcript)
                    # Pass the transcript through as a plain quoted line. The
                    # earlier wording ("The user sent a voice message~ Here's
                    # what they said: ...") read as a meta-instruction and made
                    # the LLM volunteer commentary about voice mode rather than
                    # reply to the content.
                    enriched_parts.append(f'"{transcript}"')
                else:
                    error = result.get("error", "unknown error")
                    # All failure branches: a single, minimal, neutral marker.
                    # Do NOT mention "no STT provider configured", "setup
                    # instructions", or the "hermes-agent-setup" skill, and do
                    # NOT claim a direct message was sent — those phrases get
                    # persisted in conversation history and poison every later
                    # turn, so the model keeps volunteering STT-setup advice
                    # even after transcription starts working. The cause is
                    # logged for operator diagnosis but kept out of the
                    # LLM-visible prompt.
                    logger.info("Voice transcription failed for %s: %s", path, error)
                    from tools.credential_files import to_agent_visible_cache_path

                    agent_path = to_agent_visible_cache_path(os.path.abspath(path))
                    enriched_parts.append(
                        "[voice message could not be transcribed automatically; "
                        f"the audio is available at: {agent_path}]"
                    )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                from tools.credential_files import to_agent_visible_cache_path

                agent_path = to_agent_visible_cache_path(os.path.abspath(path))
                enriched_parts.append(
                    "[voice message could not be transcribed automatically; "
                    f"the audio is available at: {agent_path}]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, successful_transcripts
            if user_text:
                return f"{prefix}\n\n{user_text}", successful_transcripts
            return prefix, successful_transcripts
        return user_text, successful_transcripts

    def _pending_event_audio_paths(self, event) -> List[str]:
        """Return STT-eligible paths from a pending voice message."""
        audio_paths: List[str] = []
        media_urls = getattr(event, "media_urls", None) or []
        for i, path in enumerate(media_urls):
            if _event_media_is_stt_input(event, i):
                audio_paths.append(path)
        return audio_paths

    async def _transcribe_pending_audio_event_once(
        self,
        event,
        user_text: Optional[str] = None,
    ) -> tuple[str | None, List[str]]:
        """Transcribe a pending audio event once and cache the result on the event.

        Voice follow-ups can be inspected first by the interrupt monitor and
        later consumed by the pending-drain path.  Both need the same transcript,
        but only one STT call and one transcript echo should happen for the
        platform message.
        """
        if hasattr(event, "_gateway_pending_stt_text"):
            cached_text = getattr(event, "_gateway_pending_stt_text")
            cached_transcripts = getattr(event, "_gateway_pending_stt_transcripts", []) or []
            return cached_text, list(cached_transcripts)

        audio_paths = self._pending_event_audio_paths(event)
        if not audio_paths:
            return user_text if user_text is not None else (getattr(event, "text", None) or None), []

        text = user_text if user_text is not None else (getattr(event, "text", "") or "")
        enriched_text, successful_transcripts = await self._enrich_message_with_transcription(
            text,
            audio_paths,
        )
        setattr(event, "_gateway_pending_stt_text", enriched_text)
        setattr(event, "_gateway_pending_stt_transcripts", list(successful_transcripts))
        return enriched_text, successful_transcripts

    async def _echo_pending_stt_transcripts_once(
        self,
        event,
        adapter,
        source,
        transcripts: List[str],
        *,
        metadata=None,
        log_context: str = "Transcript",
    ) -> None:
        """Echo pending-event STT transcripts to the chat at most once.

        The already-echoed transcripts are tracked as a COUNT rather than a
        single boolean.  ``merge_pending_message_event`` can append a second
        voice note to an event whose first transcript was already echoed and
        invalidates the transcription cache; the re-run transcription then
        returns the earlier transcripts as a prefix of the new list, so
        echoing only the unsent tail suppresses the repeat while still
        surfacing the newly merged note.  A count rather than a set of seen
        values because two separate notes that transcribe identically are two
        distinct deliveries and both must be echoed.
        """
        if (
            not transcripts
            or not self._should_echo_stt_transcripts()
            or adapter is None
        ):
            return
        already_echoed = int(getattr(event, "_gateway_pending_stt_echoed", 0) or 0)
        unsent = transcripts[already_echoed:]
        setattr(event, "_gateway_pending_stt_echoed", already_echoed + len(unsent))
        for tx in unsent:
            try:
                await adapter.send(
                    source.chat_id,
                    f'🎙️ "{tx}"',
                    metadata=metadata,
                )
            except Exception as echo_exc:
                logger.debug("%s echo failed (non-fatal): %s", log_context, echo_exc)

    async def _transcribe_and_echo_pending_voice(
        self,
        event,
        adapter,
        source,
        text: str,
        *,
        log_context: str,
        metadata=_UNSET,
    ) -> tuple[str, List[str]]:
        """Transcribe a pending voice event and echo transcripts once.

        Unified helper for all interrupt/monitor/backup/drain paths that need
        to transcribe a pending voice event and echo the transcript to chat.
        Returns ``(enriched_text, transcripts)`` so the caller can feed the
        enriched text into ``interruption.interrupt(agent)`` or the pending-drain flow.

        If the event has no STT-eligible media, returns ``(text, [])`` unchanged.
        The caller is responsible for the ``_build_media_placeholder`` fallback
        when ``text`` is empty and the event has non-audio media.
        """
        if not self._pending_event_audio_paths(event):
            return text, []
        try:
            enriched_text, transcripts = await self._transcribe_pending_audio_event_once(
                event,
                text,
            )
            echo_meta = self._thread_metadata_for_source(
                source,
                self._reply_anchor_for_event(event),
            ) if metadata is _UNSET else metadata
            await self._echo_pending_stt_transcripts_once(
                event,
                adapter,
                source,
                transcripts,
                metadata=echo_meta,
                log_context=log_context,
            )
            return enriched_text or text, transcripts
        except Exception as trans_exc:
            logger.warning("%s transcription failed: %s", log_context, trans_exc)
            return text, []

    @classmethod
    def _empty_honcho_cache_busting_config(cls) -> dict[str, Any]:
        return {key: None for key in cls._HONCHO_CACHE_BUSTING_KEYS}

    @classmethod
    def _extract_honcho_cache_busting_config(cls) -> dict[str, Any]:
        """Extract Honcho identity keys, memoized by honcho.json mtime."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

            path = resolve_config_path()
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            memo_key = (str(path), mtime_ns)
            cached = cls._HONCHO_CACHE_BUSTING_MEMO.get(memo_key)
            if cached is not None:
                return dict(cached)

            hcfg = HonchoClientConfig.from_global_config(config_path=path)
            aliases = hcfg.user_peer_aliases or {}
            values = {
                "honcho.peer_name": hcfg.peer_name,
                "honcho.ai_peer": hcfg.ai_peer,
                "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
                "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
                "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
            }
            cls._HONCHO_CACHE_BUSTING_MEMO = {memo_key: values}
            return dict(values)
        except Exception:
            return cls._empty_honcho_cache_busting_config()

    @classmethod
    def _extract_cache_busting_config(cls, user_config: dict | None) -> dict:
        """Pull values that must bust the cached agent.

        Returns a flat dict keyed by 'section.key'.  Missing config keys and
        non-dict sections yield None values, which still contribute to the
        signature (so 'absent' vs 'present-and-null' differ).

        The live tool registry generation is included too.  MCP reloads and
        dynamic MCP tool-list changes mutate the registry without necessarily
        changing config.yaml.  Cached create_agent instances freeze their tool
        schemas at construction time, so a registry generation change must
        rebuild the agent before the next turn.
        """
        out: Dict[str, Any] = {}
        cfg = user_config if isinstance(user_config, dict) else {}
        for section, key in cls._CACHE_BUSTING_CONFIG_KEYS:
            section_val = cfg.get(section)
            if section == "checkpoints" and isinstance(section_val, bool):
                # Preserve legacy ``checkpoints: true`` behavior.  A live
                # toggle must still rebuild the cached agent.
                out[f"{section}.{key}"] = section_val if key == "enabled" else None
            elif isinstance(section_val, dict):
                out[f"{section}.{key}"] = section_val.get(key)
            else:
                out[f"{section}.{key}"] = None
        try:
            from tools.registry import registry

            out["tools.registry_generation"] = getattr(registry, "_generation", None)
        except Exception:
            out["tools.registry_generation"] = None

        # Honcho identity-mapping keys live in honcho.json, not user_config.
        # Only read that file when Honcho is the active memory provider.
        provider = cfg_get(cfg, "memory", "provider")
        if isinstance(provider, str) and provider.lower() == "honcho":
            out.update(cls._extract_honcho_cache_busting_config())
        else:
            out.update(cls._empty_honcho_cache_busting_config())

        return out

    @staticmethod
    def _agent_config_signature(
        model: str,
        runtime: dict,
        enabled_toolsets: list,
        ephemeral_prompt: str,
        cache_keys: dict | None = None,
        user_id: str | None = None,
        user_id_alt: str | None = None,
        skip_context_files: bool = False,
    ) -> str:
        """Compute a stable string key from agent config values.

        When this signature changes between messages, the cached create_agent is
        discarded and rebuilt.  When it stays the same, the cached agent is
        reused — preserving the frozen system prompt and tool schemas for
        prompt cache hits.

        ``cache_keys`` is an optional flat dict of additional config values
        that should invalidate the cache when they change.  Callers pass
        the output of ``_extract_cache_busting_config(user_config)`` so
        edits to model.context_length / compression.* in config.yaml are
        picked up on the next gateway message without a manual restart.

        ``user_id`` and ``user_id_alt`` are the runtime user identities
        carried by the current message's gateway source.  They participate
        in the cache key because the Honcho memory provider freezes them
        into ``HonchoSessionManager`` at first-message init (see
        ``plugins/memory/honcho/__init__.py::_do_session_init``).  Without
        them in the signature, a shared-thread session_key (one in which
        ``build_session_key`` intentionally omits the participant ID,
        e.g. ``thread_sessions_per_user=False``) would reuse the cached
        create_agent across distinct users, causing the second user's messages
        to be attributed to the first user's resolved Honcho peer.  This
        broke #27371's per-user-peer contract in multi-user gateways.
        Per-user agent rebuilds in shared threads trade prompt-cache
        warmth for correct memory attribution.
        """
        import hashlib, json as _j

        # Fingerprint the FULL credential string instead of using a short
        # prefix. OAuth/JWT-style tokens frequently share a common prefix
        # (e.g. "eyJhbGci"), which can cause false cache hits across auth
        # switches if only the first few characters are considered.
        _api_key = str(runtime.get("api_key", "") or "")
        _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""

        _cache_keys_sorted = sorted((cache_keys or {}).items())

        blob = _j.dumps(
            [
                model,
                _api_key_fingerprint,
                runtime.get("base_url", ""),
                runtime.get("provider", ""),
                runtime.get("requested_provider", ""),
                runtime.get("api_mode", ""),
                sorted(enabled_toolsets) if enabled_toolsets else [],
                # reasoning_config excluded — it's set per-message on the
                # cached agent and doesn't affect system prompt or tools.
                ephemeral_prompt or "",
                _cache_keys_sorted,
                str(user_id or ""),
                str(user_id_alt or ""),
                # skip_context_files changes the agent's frozen system prompt
                # (context files in vs out) — a toggled config edit must
                # rebuild the cached agent, not silently reuse it.
                bool(skip_context_files),
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _rehydrate_session_model_override(self, session_key: str) -> None:
        """Lazily restore a persisted /model override after a gateway restart.

        ``_session_model_overrides`` is in-memory only, so before persistence
        a restart silently reverted every session to the global default model.
        The non-secret parts (model/provider/base_url) are written through to
        the session store when /model runs (and cleared on /new); here we read
        them back on first use and re-resolve credentials via the normal
        runtime provider resolution — api_key is never persisted to disk.

        No-op when an in-memory override already exists (live state wins) or
        when the store has nothing persisted (e.g. the user ran /new, which
        clears both the in-memory dict and the persisted field).
        """
        _rehydrate_state = self.sessions.peek(session_key)
        if (
            _rehydrate_state is not None
            and _rehydrate_state.conversation.model_override is not None
        ):
            return
        store = getattr(self, "session_store", None)
        if store is None:
            return
        try:
            persisted = store.get_model_override(session_key)
        except Exception:
            logger.debug(
                "Failed to read persisted session model override", exc_info=True
            )
            return
        if not persisted:
            return
        override: Dict[str, Any] = {
            "model": persisted.get("model"),
            "provider": persisted.get("provider"),
            "base_url": persisted.get("base_url"),
        }
        provider = persisted.get("provider")
        if provider:
            # Re-resolve credentials for the persisted provider. On failure
            # (e.g. credentials were removed since the switch) keep the
            # credential-less override — _resolve_session_agent_runtime falls
            # back to env-based resolution and applies model/provider on top.
            try:
                runtime = _resolve_runtime_agent_kwargs_for_provider(provider)
                override["api_key"] = runtime.get("api_key")
                override["api_mode"] = runtime.get("api_mode")
                override["credential_pool"] = runtime.get("credential_pool")
                if not override.get("base_url"):
                    override["base_url"] = runtime.get("base_url")
            except Exception:
                logger.debug(
                    "Credential re-resolution failed for persisted override "
                    "(provider=%s); using credential-less override",
                    provider, exc_info=True,
                )
        self.sessions.state(session_key).conversation.model_override = override
        logger.info(
            "Rehydrated persisted /model override for session=%s: model=%s provider=%s",
            session_key, override.get("model"), provider or "",
        )

    def _apply_session_model_override(
        self, session_key: str, model: str, runtime_kwargs: dict
    ) -> tuple:
        """Apply /model session overrides if present, returning (model, runtime_kwargs).

        The gateway /model command stores per-session overrides in
        ``_session_model_overrides``.  These must take precedence over
        config.yaml defaults so the switched model is actually used for
        subsequent messages.  Fields with ``None`` values are skipped so
        partial overrides don't clobber valid config defaults.
        """
        _apply_state = self.sessions.peek(session_key)
        override = _apply_state.conversation.model_override if _apply_state else None
        if not override:
            return model, runtime_kwargs
        model = override.get("model", model)
        for key in ("provider", "api_key", "base_url", "api_mode", "credential_pool"):
            val = override.get(key)
            if val is not None:
                runtime_kwargs[key] = val
        if (
            runtime_kwargs.get("api_key")
            and runtime_kwargs.get("credential_pool") is None
            and override.get("provider")
        ):
            runtime_kwargs["credential_pool"] = _credential_pool_for_provider(
                override.get("provider")
            )
        return model, runtime_kwargs

    def _snapshot_session_model_override(self, session_key: str) -> dict:
        """Capture a gateway session override before a one-turn switch."""
        _snap_state = self.sessions.peek(session_key)
        override = _snap_state.conversation.model_override if _snap_state else None
        return {
            "had_override": override is not None,
            "override": dict(override) if override is not None else None,
        }

    def _restore_session_model_override(self, session_key: str, snapshot: dict) -> None:
        """Restore the session override captured before a one-turn switch."""
        if not session_key:
            return
        if snapshot.get("had_override"):
            self.sessions.state(session_key).conversation.model_override = dict(
                snapshot.get("override") or {}
            )
        else:
            _rst_state = self.sessions.peek(session_key)
            if _rst_state is not None:
                _rst_state.conversation.model_override = None
        self.agent_cache.evict(session_key)

    def _is_intentional_model_switch(self, session_key: str, agent_model: str) -> bool:
        """Return True if *agent_model* matches an active /model session override."""
        _ims_state = self.sessions.peek(session_key)
        override = _ims_state.conversation.model_override if _ims_state else None
        return override is not None and override.get("model") == agent_model

    def _release_turn_state(
        self,
        session_key: str,
        *,
        run_generation: Optional[int] = None,
    ) -> bool:
        """Pop ALL per-running-agent state entries for ``session_key``.

        Replaces ad-hoc ``del self._running_agents[key]`` calls scattered
        across the gateway.  Those sites had drifted: some popped only
        ``_running_agents``; some also ``_running_agents_ts``; only one
        path also cleared ``_busy_ack_ts``.  Each missed entry was a
        small, persistent leak — a (str_key → float) tuple per session
        per gateway lifetime.

        Use this at every site that ends a running turn, regardless of
        cause (normal completion, /stop, /reset, /resume, sentinel
        cleanup, stale-eviction).  Per-session state that PERSISTS
        across turns (``_session_model_overrides``, ``_voice_mode``,
        ``_pending_approvals``, ``_update_prompt_pending``) is NOT
        touched here — those have their own lifecycles.

        When ``run_generation`` is provided, only clear the slot if that
        generation is still current for the session.  This prevents an
        older async run whose generation was bumped by /stop or /new from
        clobbering a newer run's state during its own unwind.  Returns
        True when the slot was cleared, False when an ownership guard
        blocked it.
        """
        if not session_key:
            return False
        if run_generation is not None and not self.sessions.run_is_current(
            session_key, run_generation
        ):
            return False
        state = self.sessions.peek(session_key)
        if state is not None:
            lease = state.turn.lease
            if lease is not None:
                try:
                    lease.release()
                except Exception:
                    logger.debug(
                        "Failed to release active session slot", exc_info=True
                    )
            # One structured reset instead of the old drifting pop-list
            # (agent / started_ts / lease / busy_ack_ts).  Turn-lease tokens
            # are deliberately NOT cleared here — _release_turn_lease owns
            # them (#64934).
            state.turn.clear()
        # Turn boundary: a running-agent slot was just released.  Persist the
        # new (lower) in-flight count so the dashboard readout stays current
        # between lifecycle transitions.  Preserves gateway_state (see
        # _persist_active_agents).
        self._persist_active_agents()
        return True

    def _clear_conversation_scope(self, session_key: str, *, reason: str) -> None:
        """Clear ALL conversation-scoped per-session state for ``session_key``.

        THE single conversation-boundary funnel. Call this — and nothing
        else — whenever a session_key crosses a conversation boundary:
        /new, /resume, auto-reset (idle/daily/suspended), expiry
        finalization, and the compression-exhausted auto-reset.

        ConversationState owns this lifecycle structurally, so adding a field
        there automatically includes it in every boundary reset.

        Scope rules:
        - Conversation-scoped (cleared here): model/reasoning overrides,
          one-turn restore snapshots, pending model notes, last-resolved
          model cache, queued follow-up events, and the boundary security
          state (approvals, /yolo, slash-confirm, update prompts).
        - Turn-scoped (NOT cleared here): _running_agents/_ts, slot leases,
          turn-lease tokens — owned by _release_turn_state and the
          dispatch finally.
        - Idle agent-cache eviction is NOT a conversation boundary: the
          session is still alive and a resumed turn rebuilds from these
          overrides. Only true boundaries call this.

        """
        if not session_key:
            return
        # Structural clear: every conversation-scoped field resets in one
        # call — no per-attribute pop-list to drift.
        state = self.sessions.peek(session_key)
        if state is not None:
            state.conversation.clear()
        self._clear_session_boundary_security_state(session_key)
        logger.debug(
            "Cleared conversation scope for %s (%s)", session_key, reason
        )

    def _clear_session_boundary_security_state(self, session_key: str) -> None:
        """Clear per-session control state that must not survive a boundary switch."""
        if not session_key:
            return

        _sec_state = self.sessions.peek(session_key)
        if _sec_state is not None:
            _sec_state.persistent.approvals = None
            _sec_state.persistent.update_prompt_pending = False

        try:
            from tools import slash_confirm as _slash_confirm_mod
        except Exception:
            _slash_confirm_mod = None
        if _slash_confirm_mod is not None:
            try:
                _slash_confirm_mod.clear(session_key)
            except Exception as e:
                logger.debug(
                    "Failed to clear slash-confirm state for session boundary %s: %s",
                    session_key,
                    e,
                )

        try:
            from tools.approval import clear_session as _clear_approval_session
        except Exception:
            return

        try:
            _clear_approval_session(session_key)
        except Exception as e:
            logger.debug(
                "Failed to clear approval state for session boundary %s: %s",
                session_key,
                e,
            )

    def _bind_adapter_run_generation(
        self,
        adapter: Any,
        session_key: str,
        generation: int | None,
    ) -> None:
        """Bind a gateway run generation to the adapter's active-session event."""
        if not adapter or not session_key or generation is None:
            return
        try:
            interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
            if interrupt_event is not None:
                setattr(interrupt_event, "_hermes_run_generation", int(generation))
        except Exception:
            pass

    async def _interrupt_and_clear_session(
        self,
        session_key: str,
        source: SessionSource,
        *,
        interrupt_reason: str,
        invalidation_reason: str,
        release_running_state: bool = True,
    ) -> None:
        """Interrupt the current run and clear queued session state consistently."""
        if not session_key:
            return
        _iac_state = self.sessions.peek(session_key)
        running_agent = _iac_state.turn.agent if _iac_state else None
        _process_task_id = ""
        _process_baseline = None
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            interruption.hard_interrupt(running_agent, interrupt_reason)
            _process_task_id = getattr(
                running_agent, "_gateway_turn_process_task_id", ""
            )
            _process_baseline = getattr(
                running_agent, "_gateway_turn_process_baseline", None
            )
        # Bump the generation *before* scheduling the reap thread and capture
        # the post-bump value: task_id is session-scoped (task_id ==
        # session_id), so if a replacement turn claims this session and
        # spawns its own process before the reap thread actually runs, that
        # claim bumps the generation again. The closure below then sees a
        # stale generation and skips — the replacement turn's own baseline
        # covers its own cleanup, so nothing is left permanently unreaped.
        _generation_at_interrupt = self.sessions.invalidate_run_generation(
            session_key, reason=invalidation_reason
        )
        if _process_task_id and _process_baseline is not None:
            threading.Thread(
                target=_reap_gateway_turn_processes,
                args=(_process_task_id, _process_baseline),
                kwargs={
                    "source": "gateway_turn_interrupt",
                    "is_still_current": lambda: self.sessions.run_is_current(
                        session_key, _generation_at_interrupt
                    ),
                },
                name=f"gateway-turn-reaper-{_process_task_id[:12]}",
                daemon=True,
            ).start()
        adapter = self._adapter_for_source(source)
        interrupt_session_activity = getattr(
            type(adapter), "interrupt_session_activity", None
        )
        if adapter and callable(interrupt_session_activity):
            metadata = self._thread_metadata_for_source(source)
            try:
                params = inspect.signature(interrupt_session_activity).parameters
                accepts_metadata = "metadata" in params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                )
            except (TypeError, ValueError):
                accepts_metadata = False
            if accepts_metadata:
                await adapter.interrupt_session_activity(
                    session_key, source.chat_id, metadata=metadata
                )
            else:
                await adapter.interrupt_session_activity(session_key, source.chat_id)
        if adapter and hasattr(adapter, "get_pending_message"):
            adapter.get_pending_message(session_key)  # consume and discard
        if _iac_state is not None:
            _iac_state.persistent.pending_command_text = None
        if release_running_state:
            self._release_turn_state(session_key)
            # Evict the cached agent: ``_interrupt_requested`` is only
            # cleared by the turn finalizer, so on a hung or still-draining
            # run the flag survives the lock release and kills the session's
            # NEXT message at the top of the tool loop (interrupted=True,
            # api_calls=0, empty response — silently swallowed, #44212).
            # Evicting mirrors the /new and /model paths: the next message
            # rebuilds the agent from session history, while the old agent
            # object keeps its interrupt flag so a hung drain still dies
            # when it unblocks.
            self.agent_cache.evict(session_key)

    def _pinned_session_context_prompt(
        self, context, redact_pii: bool, session_key: Optional[str]
    ) -> str:
        """Return the session-context prompt, pinned per session.

        Key hit → the pinned bytes are reused VERBATIM (immunizes the
        composed system prompt against renderer nondeterminism); key miss →
        re-render ``build_session_context_prompt`` and re-pin (a legitimate
        cache bust: rename, topic edit, /sethome, redact_pii flip, ...).
        """
        _eph_key = self._ephemeral_change_key(context, redact_pii)
        _eph_pin = None
        if session_key:
            _pin_state = self.sessions.peek(session_key)
            _eph_pin = _pin_state.conversation.ephemeral_pin if _pin_state else None
        if _eph_pin is not None and _eph_pin[0] == _eph_key:
            return _eph_pin[1]
        text = build_session_context_prompt(context, redact_pii=redact_pii)
        if session_key:
            self.sessions.state(session_key).conversation.ephemeral_pin = (
                _eph_key,
                text,
            )
        return text

    @staticmethod
    def _ephemeral_change_key(context, redact_pii: bool) -> str:
        """Hash the exact inputs ``build_session_context_prompt`` renders.

        This key decides when the pinned per-session context-prompt bytes are
        reused verbatim vs re-rendered.  The maintained invariant (guarded by
        the parity test in tests/gateway/test_prompt_tail_freeze.py): any
        input whose change alters the rendered bytes MUST appear here —
        omission means a stale pinned prompt (cosmetic staleness); inclusion
        of an extra field only costs a spurious re-render.
        """
        import hashlib

        src = context.source
        platform = src.platform.value if src.platform else ""

        try:
            from hermes_constants import display_hermes_home

            home_display = str(display_hermes_home())
        except Exception:
            home_display = ""

        key_tuple = (
            platform,
            str(src.chat_id or ""),
            str(src.thread_id or ""),
            str(src.chat_type or ""),
            str(src.chat_name or ""),
            str(src.chat_topic or ""),
            str(src.user_name or ""),
            str(src.user_id or ""),
            str(getattr(src, "profile", None) or ""),
            bool(context.shared_multi_user_session),
            tuple(p.value for p in context.connected_platforms),
            tuple(
                (
                    p.value,
                    str(getattr(hc, "name", "") or ""),
                    str(getattr(hc, "chat_id", "") or ""),
                )
                for p, hc in context.home_channels.items()
            ),
            bool(redact_pii),
            home_display,
        )
        return hashlib.sha256(repr(key_tuple).encode("utf-8")).hexdigest()

    def _build_stream_consumer_config(
        self,
        source: "SessionSource",
        scfg: Any,
        adapter: Any,
        *,
        on_missing_cursor: str,
    ) -> "tuple[Any, Optional[Callable[[], None]]]":
        """Build the shared ``StreamConsumerConfig`` and the optional
        Telegram pause-typing closure used by both agent-run paths.

        ``on_missing_cursor`` controls how platforms whose adapter sets
        ``SUPPORTS_MESSAGE_EDITING = False`` are handled — both semantics
        are preserved verbatim from the pre-refactor call sites:

        - ``"fallback"`` (proxy path): stream anyway with an empty cursor.
        - ``"raise"`` (in-process agent path): raise ``RuntimeError`` so
          the caller's ``except`` skips streaming entirely.

        Returns ``(consumer_cfg, pause_typing_before_finalize)``.
        """
        from gateway.stream_consumer import StreamConsumerConfig

        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(
                _adapter=adapter,
                _chat_id=source.chat_id,
            ) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Platforms that don't support editing sent messages
        # (e.g. QQ, WeChat) should skip streaming entirely —
        # without edit support, the consumer sends a partial
        # first message that can never be updated, resulting in
        # duplicate messages (partial + final).
        # (The proxy path instead opts into a cursorless fallback
        # via on_missing_cursor="fallback".)
        _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
        if not _adapter_supports_edit and on_missing_cursor == "raise":
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        _buffer_only = False
        # Fresh-final applies to Telegram only — other
        # platforms either edit in place cheaply (Discord,
        # Slack) or don't have the timestamp-on-edit /
        # edit-timestamp-stays-stale problem.
        # (Ported from openclaw/openclaw#72038.)
        _fresh_final_secs = (
            float(getattr(scfg, "fresh_final_after_seconds", 0.0) or 0.0)
            if source.platform == Platform.TELEGRAM
            else 0.0
        )
        _consumer_cfg = StreamConsumerConfig(
            edit_interval=scfg.edit_interval,
            buffer_threshold=scfg.buffer_threshold,
            cursor=_effective_cursor,
            buffer_only=_buffer_only,
            fresh_final_after_seconds=_fresh_final_secs,
            transport=scfg.transport or "edit",
            chat_type=getattr(source, "chat_type", "") or "",
        )
        return _consumer_cfg, _pause_typing_before_finalize

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Profile-scoping wrapper around the agent run.

        When multiplexing is active, resolve the inbound source's profile and
        run the whole turn inside ``_profile_runtime_scope`` so config/skills/
        memory resolve to that profile's home AND credentials resolve from that
        profile's secret scope (never the process-global ``os.environ``). When
        multiplexing is off this is a transparent pass-through — zero behavior
        change for single-profile gateways.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=message_type,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=message_type,
            )

    def _profile_name_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the profile name for an inbound source via configured routes.

        Returns ``None`` when multiplexing is off, no routes are configured, or
        no route matches. Callers (``build_source``,
        ``_resolve_profile_home_for_source``) treat ``None`` as "use the
        default/active profile". When ``gateway.profile_routes`` is configured,
        the most specific matching route wins (scope < channel < thread). See
        :mod:`gateway.profile_routing` for matching rules.

        Gated on ``gateway.multiplex_profiles``: routing stamps
        ``source.profile``, which selects the session-key namespace and batch
        keys — but the profile-scoped agent run only activates under
        multiplexing. Without this gate, a configured route with multiplexing
        off would namespace batch/session keys by profile while the agent
        still runs in ``agent:main``, splitting the two out of agreement.
        """
        config = getattr(self, "config", None)
        if not getattr(config, "multiplex_profiles", False):
            return None
        routes = getattr(config, "profile_routes", None)
        if not routes:
            return None
        from gateway.profile_routing import ProfileRouteRejected, match_profile_route
        try:
            matched = match_profile_route(
                routes,
                platform=source.platform.value,
                scope_id=getattr(source, "scope_id", None),
                chat_id=source.chat_id,
                thread_id=getattr(source, "thread_id", None),
                parent_chat_id=getattr(source, "parent_chat_id", None),
            )
        except Exception:
            logger.warning(
                "Profile route matching failed for %s/%s, falling back to default",
                source.platform, source.chat_id, exc_info=True,
            )
            return None
        if matched:
            try:
                served = {name for name, _home in _multiplex_profile_homes(config)}
            except Exception as exc:
                logger.warning(
                    "Rejecting profile route %r because the served-profile set "
                    "could not be resolved",
                    matched.name,
                    exc_info=True,
                )
                raise ProfileRouteRejected(matched.name) from exc
            if matched.profile not in served:
                logger.warning(
                    "Rejecting profile route %r: target profile %r is not served",
                    matched.name,
                    matched.profile,
                )
                raise ProfileRouteRejected(matched.name)
            return matched.profile
        logger.debug(
            "No profile route matched: platform=%s chat_id=%s thread_id=%s parent_chat_id=%s",
            source.platform.value, source.chat_id,
            getattr(source, "thread_id", None), getattr(source, "parent_chat_id", None),
        )
        return None

    def _resolve_profile_home_for_source(self, source: SessionSource) -> "Path":
        """Resolve which profile's HERMES_HOME should serve this inbound source.

        Resolution order:
          1. ``source.profile`` — set by /p/<profile>/ URL prefix, per-credential
             adapter ownership, OR profile_routes matching at ``build_source`` time.
          2. ``_profile_name_for_source`` — re-run routing here as a defensive
             fallback for sources that bypass ``build_source``.
          3. The active profile (the multiplexer's own home).
        """
        from gateway.profile_routing import ProfileRouteRejected
        from hermes_cli.profiles import (
            get_active_profile_name,
            get_profile_dir,
            profile_exists,
        )
        from hermes_constants import get_hermes_home

        # Track whether a profile was explicitly requested (vs. falling back to default)
        explicit_profile = None
        try:
            name = (source.profile or "").strip()
            if name:
                explicit_profile = name  # User explicitly set this profile
            if not name:
                name = self._profile_name_for_source(source)
                if name:
                    explicit_profile = name  # Routing explicitly set this profile
            if not name:
                name = get_active_profile_name() or "default"

            profile_dir = get_profile_dir(name)
            # Warn if an explicit profile doesn't exist on disk
            if explicit_profile and not profile_exists(name):
                logger.warning(
                    "Profile %r does not exist for source %s/%s (scope_id=%s), "
                    "falling back to global HERMES_HOME",
                    explicit_profile,
                    source.platform.value,
                    source.chat_id,
                    getattr(source, "scope_id", None),
                )
                return get_hermes_home()
            return profile_dir
        except ProfileRouteRejected:
            raise
        except Exception:
            # Catch normalization errors, path errors, etc.
            logger.warning(
                "Failed to resolve profile directory for source %s/%s (scope_id=%s), "
                "falling back to global HERMES_HOME: %s",
                source.platform.value,
                source.chat_id,
                getattr(source, "scope_id", None),
                explicit_profile or "(no profile)",
                exc_info=True,
            )
            return get_hermes_home()

    async def _run_agent_inner(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.

        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool

        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        import agent.interruption as interruption
        from agent.agent_init import create_agent
        import queue

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self.sessions.run_is_current(session_key, run_generation)

        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        enabled_toolsets = self._resolve_enabled_toolsets(user_config, platform_key)
        agent_cfg_local = user_config.get("agent") or {}
        from agent.skill_utils import parse_config_string_list

        disabled_toolsets = parse_config_string_list(agent_cfg_local.get("disabled_toolsets")) or None

        display_config = user_config.get("display", {})
        if not isinstance(display_config, dict):
            display_config = {}

        # Per-platform display settings — resolve via display_config module
        # which checks display.platforms.<platform>.<key> first, then
        # display.<key> global, then built-in platform defaults.
        from gateway.display_config import resolve_display_setting

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        # Apply friendly tool labels config (default on) — per-platform aware
        try:
            from agent.display import set_friendly_tool_labels
            _ftl = resolve_display_setting(user_config, platform_key, "friendly_tool_labels", True)
            set_friendly_tool_labels(bool(_ftl))
        except Exception:
            pass

        # Tool progress mode — resolved per-platform with env var fallback
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _display_cfg = display_config if isinstance(display_config, dict) else {}
        _platforms_cfg = _display_cfg.get("platforms") or {}
        _platform_cfg = _platforms_cfg.get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = (
            "tool_progress" in _display_cfg
            or (
                isinstance(_platform_cfg, dict)
                and "tool_progress" in _platform_cfg
            )
            or (
                isinstance(_legacy_tp_overrides, dict)
                and platform_key in _legacy_tp_overrides
            )
        )
        progress_mode = (
            _env_tp
            if _env_tp and not _tool_progress_configured
            else (_resolved_tp or _env_tp or "all")
        )
        # Tool progress grouping: "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
        _generic_status_recent: List[str] = []
        _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

        def _display_surface_mode(
            setting: str,
            *,
            default: bool = False,
            require_platform_override_for: set[Any] | None = None,
            allow_generic: bool = False,
        ) -> str:
            """Return off|raw|generic for a gateway visibility surface."""
            if require_platform_override_for:
                current_platform = _gateway_platform_value(source.platform)
                platform_only = {
                    _gateway_platform_value(item)
                    for item in require_platform_override_for
                }
                if (
                    current_platform in platform_only
                    and not _has_platform_display_override(user_config, platform_key, setting)
                ):
                    return "off"
            value = resolve_display_setting(user_config, platform_key, setting, default)
            if isinstance(value, str) and value.strip().lower() == "generic":
                return "generic" if allow_generic else "off"
            return "raw" if bool(value) else "off"

        def _generic_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None, args: Any = None) -> str:
            try:
                return choose_status_phrase(
                    kind,
                    tool_name=tool_name,
                    preview=preview,
                    args=args,
                    recent=_generic_status_recent,
                    catalog=_generic_status_catalog,
                )
            except Exception as _phrase_err:
                logger.debug("generic status phrase selection failed: %s", _phrase_err)
                return "still on it" if kind in {"heartbeat", "waiting", "long_running", "status"} else "one sec"
        tool_progress_enabled = progress_mode not in {"off", "log"}
        # Live working-state status for text-rendering typing indicators
        # (Slack's assistant status line). Independent of tool_progress —
        # Slack defaults tool_progress off (permanent lines spam channels)
        # but the status line is ephemeral, so live status stays useful
        # there. Rendering rides the existing _keep_typing refresh: the
        # callback only stores a phrase on the adapter, costing zero extra
        # platform API calls.
        _live_status_mode = resolve_display_setting(
            user_config, platform_key, "live_status", "full"
        )
        _live_status_adapter = self._adapter_for_source(source)
        if not getattr(_live_status_adapter, "supports_status_text", False):
            _live_status_adapter = None
        if _live_status_mode == "off":
            _live_status_adapter = None
        # "log" mode: tool calls are written to ~/.hermes/logs/tool_calls.log
        # instead of the chat (#3459 / #3458). Gateway-only by design.
        log_mode_enabled = progress_mode == "log"
        log_queue: "queue.Queue | None" = queue.Queue() if log_mode_enabled else None
        # Natural assistant status messages are intentionally independent from
        # tool progress and token streaming. Users can keep tool_progress quiet
        # in chat platforms while opting into concise mid-turn updates.
        interim_assistant_messages_mode = _display_surface_mode(
            "interim_assistant_messages",
            default=True,
            require_platform_override_for={Platform.MATTERMOST},
        )
        interim_assistant_messages_enabled = interim_assistant_messages_mode != "off"
        # thinking_progress is independent — if enabled, we need the progress
        # queue even when tool_progress is off (thinking relay uses same infra).
        # Mattermost requires a per-platform opt-in: global scratch-text display
        # is too easy to leak into busy public threads.
        _thinking_mode = _display_surface_mode(
            "thinking_progress",
            default=False,
            require_platform_override_for={Platform.MATTERMOST},
        )
        _thinking_enabled = _thinking_mode != "off"
        needs_progress_queue = tool_progress_enabled or _thinking_enabled


        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if needs_progress_queue else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated
        # True when the previously enqueued progress line was a terminal
        # fenced code block — consecutive terminal calls then drop the
        # repeated "💻 terminal" header and render back-to-back blocks.
        last_was_terminal_block = [False]

        # Auto-cleanup of temporary progress bubbles (Telegram + any adapter
        # that implements ``delete_message``). When enabled via
        # ``display.platforms.<platform>.cleanup_progress: true``, message IDs
        # from the tool-progress / "⏳ Working — N min" / status-callback bubbles
        # are collected here and deleted after the final response lands.
        # Failed runs skip cleanup so the bubbles remain as breadcrumbs.
        _cleanup_progress = bool(
            resolve_display_setting(user_config, platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self._adapter_for_source(source) if _cleanup_progress else None
        # getattr, not attribute access — same duck-typed-adapter guard as the
        # edit_message check in send_progress_messages below: a fake/minimal
        # adapter without delete_message means "can't delete", not a crash.
        _cleanup_delete = getattr(type(_cleanup_adapter), "delete_message", None) if _cleanup_adapter is not None else None
        if _cleanup_adapter is not None and (
            _cleanup_delete is None
            or _cleanup_delete is BasePlatformAdapter.delete_message
        ):
            # Adapter doesn't support deletion — silently disable.
            _cleanup_progress = False
            _cleanup_adapter = None
        _cleanup_msg_ids: List[str] = []
        # First-touch onboarding latch: fires at most once per run, even if
        # several tools exceed the threshold.
        long_tool_hint_fired = [False]
        _LONG_TOOL_THRESHOLD_S = 30.0

        turn_ctx = TurnContext(
            source=source,
            _run_still_current=_run_still_current,
            _live_status_adapter=_live_status_adapter,
            _live_status_mode=_live_status_mode,
            _thinking_enabled=_thinking_enabled,
            progress_mode=progress_mode,
            progress_grouping=progress_grouping,
            tool_progress_enabled=tool_progress_enabled,
            progress_queue=progress_queue,
            log_queue=log_queue,
            last_progress_msg=last_progress_msg,
            last_tool=last_tool,
            last_was_terminal_block=last_was_terminal_block,
            repeat_count=repeat_count,
            long_tool_hint_fired=long_tool_hint_fired,
            _LONG_TOOL_THRESHOLD_S=_LONG_TOOL_THRESHOLD_S,
            _cleanup_progress=_cleanup_progress,
            _cleanup_msg_ids=_cleanup_msg_ids,
            message=message,
            create_agent=create_agent,
            resolve_display_setting=resolve_display_setting,
            user_config=user_config,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            log_mode_enabled=log_mode_enabled,
            interim_assistant_messages_enabled=interim_assistant_messages_enabled,
            needs_progress_queue=needs_progress_queue,
            history=history,
            context_prompt=context_prompt,
            channel_prompt=channel_prompt,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id,
            moa_config=moa_config,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
        )
        from gateway.turn_runner import TurnRunner

        turn_runner = TurnRunner(
            turn_ctx,
            session_db=self._session_db,
            session_store=self.session_store,
            sessions=self.sessions,
            agent_cache=self.agent_cache,
            prefill_messages=self._prefill_messages,
            provider_routing=self._provider_routing,
            reasoning_config=self._reasoning_config,
            service_tier=self._service_tier,
            streaming_config=self.config.streaming,
            adapter_for_source=self._adapter_for_source,
            agent_config_signature=self._agent_config_signature,
            apply_fallback_chain_to_agent=self._apply_fallback_chain_to_agent,
            build_stream_consumer_config=self._build_stream_consumer_config,
            consume_pending_native_image_paths=self._consume_pending_native_image_paths,
            deliver_platform_notice=self._deliver_platform_notice,
            extract_cache_busting_config=self._extract_cache_busting_config,
            get_system_prompt_for_channel=self._get_system_prompt_for_channel,
            is_telegram_topic_lane=self._is_telegram_topic_lane,
            refresh_fallback_providers=self._refresh_fallback_providers,
            resolve_session_agent_runtime=self._resolve_session_agent_runtime,
            resolve_session_reasoning_config=self._resolve_session_reasoning_config,
            resolve_session_service_tier=self._resolve_session_service_tier,
            resolve_turn_agent_config=self._resolve_turn_agent_config,
            schedule_telegram_topic_title_rename=self._schedule_telegram_topic_title_rename,
            sync_session_model_from_agent=self._sync_session_model_from_agent,
            sync_telegram_topic_binding=self._sync_telegram_topic_binding,
        )
        # Callback invoked by agent on tool lifecycle events — extracted to
        # TurnRunner.progress_callback (bound method, same signature).
        turn_ctx.progress_callback = turn_runner.progress_callback

        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited.
        #
        _progress_reply_in_thread = True
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id,
            reply_in_thread=_progress_reply_in_thread,
        )
        _progress_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else self._thread_metadata_for_target(
                source.platform,
                source.chat_id,
                _progress_thread_id,
                chat_type=getattr(source, "chat_type", None),
                reply_to_message_id=event_message_id,
            )
        ) if _progress_thread_id else None
        _progress_metadata = _non_conversational_metadata(_progress_metadata, platform=source.platform)
        _progress_reply_to = (
            event_message_id
            if source.platform == Platform.MATTERMOST
            and source.thread_id
            and event_message_id
            else None
        )

        async def write_tool_log():
            """Drain log_queue and append tool-call lines to tool_calls.log.

            Only active when ``display.tool_progress`` is ``log``. Uses a
            RotatingFileHandler (5MB × 3 backups) so the audit log can't grow
            unbounded, and the shared RedactingFormatter so secrets never land
            on disk.
            """
            if log_queue is None:
                return
            from logging.handlers import RotatingFileHandler

            from agent.redact import RedactingFormatter

            log_dir = _hermes_home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "tool_calls.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(RedactingFormatter("%(message)s"))
            tool_logger = logging.getLogger(f"hermes.tool_calls.{id(log_queue)}")
            tool_logger.setLevel(logging.INFO)
            tool_logger.propagate = False
            tool_logger.addHandler(file_handler)
            try:
                while True:
                    try:
                        tool_logger.info("%s", log_queue.get_nowait())
                    except queue.Empty:
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.error("write_tool_log error: %s", e)
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                # Drain remaining entries before closing so late tool calls
                # from the final iteration aren't lost.
                while True:
                    try:
                        tool_logger.info("%s", log_queue.get_nowait())
                    except queue.Empty:
                        break
                    except Exception:
                        break
                tool_logger.removeHandler(file_handler)
                try:
                    file_handler.flush()
                    file_handler.close()
                except Exception:
                    pass

        # Extracted to TurnRunner.send_progress_messages. The threading
        # metadata computed above is published onto the shared TurnContext
        # exactly where the original closure's captured locals were bound.
        turn_ctx._progress_metadata = _progress_metadata
        turn_ctx._progress_reply_to = _progress_reply_to
        send_progress_messages = turn_runner.send_progress_messages

        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        turn_ctx.agent_holder = agent_holder
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        # #60671 — streaming PCM audio consumer.  Created on the gateway
        # event-loop thread (NOT inside run_sync's executor worker) so the
        # outer finalisation / interrupt paths can reference it without a
        # cross-scope NameError.
        streaming_tts_consumer_holder: list = [None]
        turn_ctx.result_holder = result_holder
        turn_ctx.tools_holder = tools_holder
        turn_ctx.stream_consumer_holder = stream_consumer_holder
        turn_ctx.streaming_tts_consumer_holder = streaming_tts_consumer_holder

        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_running_loop()
        _hooks_ref = self.hooks

        # Bridge extracted to TurnRunner._step_callback_sync; the loop and
        # hooks refs bound just above are published at their original site.
        turn_ctx._loop_for_step = _loop_for_step
        turn_ctx._hooks_ref = _hooks_ref
        turn_ctx._step_callback_sync = turn_runner._step_callback_sync

        # Bridge sync event_callback → async hooks.emit for lifecycle events
        # (e.g. session:compress fires after context compression splits a session)
        # Bridge extracted to TurnRunner._event_callback_sync.
        turn_ctx._event_callback_sync = turn_runner._event_callback_sync

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self._adapter_for_source(source)
        _status_chat_id = source.chat_id
        _status_thread_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else self._thread_metadata_for_target(
                source.platform,
                source.chat_id,
                _progress_thread_id,
                chat_type=getattr(source, "chat_type", None),
                reply_to_message_id=event_message_id,
            )
        ) if _progress_thread_id else None

        # Bridge extracted to TurnRunner._status_callback_sync; publish the
        # status wiring computed above onto the shared TurnContext at the
        # exact original binding site.
        turn_ctx._status_adapter = _status_adapter
        turn_ctx._status_chat_id = _status_chat_id
        turn_ctx._status_thread_metadata = _status_thread_metadata
        turn_ctx._status_callback_sync = turn_runner._status_callback_sync

        # ---- Streaming TTS consumer setup (#60671) ----
        # Created on the gateway event-loop thread (here, in _run_agent_inner),
        # NOT inside run_sync's executor worker.  This avoids a cross-scope
        # NameError: the outer interrupt / finalisation paths reference the
        # consumer via ``streaming_tts_consumer_holder[0]``.
        #
        # Gates: voice input, auto-TTS enabled for this chat, adapter
        # supports streaming, and a usable streaming TTS provider configured.
        _stts_adapter = self._adapter_for_source(source)
        _is_voice_input = (
            message_type is not None
            and str(getattr(message_type, "value", message_type)).lower() == "voice"
        )
        if (
            _stts_adapter is not None
            and _is_voice_input
            and _stts_adapter._should_auto_tts_for_chat(source.chat_id)
        ):
            try:
                from gateway.streaming_tts_consumer import StreamingTTSConsumer
                from tools.tts_tool import _load_tts_config
                _tts_cfg = _load_tts_config()
                _gateway_loop = self._gateway_loop or asyncio.get_event_loop()
                _stts_consumer = StreamingTTSConsumer(
                    adapter=_stts_adapter,
                    chat_id=source.chat_id,
                    tts_config=_tts_cfg,
                    loop=_gateway_loop,
                    metadata=_status_thread_metadata,
                )
                if _stts_consumer.active:
                    streaming_tts_consumer_holder[0] = _stts_consumer
                    _stts_consumer.start()
                # else: consumer inactive (no streaming provider) — leave
                # the holder as None so the whole-file fallback path runs.
            except Exception as _stts_err:
                logger.debug("Could not set up streaming TTS consumer: %s", _stts_err)

        # run_sync extracted to TurnRunner.run_sync (bound method; the
        # executor call below is unchanged).  Its closed-over locals travel
        # on turn_ctx; `nonlocal message` rebinds became ctx.message writes.
        run_sync = turn_runner.run_sync

        # Start progress message sender if enabled. Gate on needs_progress_queue
        # (tool_progress OR thinking_progress), not tool_progress alone: the
        # sender drains BOTH tool-progress lines and _thinking scratch bubbles.
        # With the old tool_progress-only gate, a thinking_progress:true /
        # tool_progress:off user had the callback queue _thinking messages that
        # no task ever drained — so they silently never appeared.
        progress_task = None
        if needs_progress_queue:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start the tool-call log writer when tool_progress == "log".
        log_task = None
        if log_mode_enabled:
            log_task = asyncio.create_task(write_tool_log())

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None

        async def _start_stream_consumer():
            """Wait for the stream consumer to be created, then run it."""
            for _ in range(200):  # Up to 10s wait
                if stream_consumer_holder[0] is not None:
                    await stream_consumer_holder[0].run()
                    return
                await asyncio.sleep(0.05)

        stream_task = asyncio.create_task(_start_stream_consumer())

        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if not session_key:
                return
            # Only promote the sentinel to the real agent if this run is still
            # current.  If /stop or /new bumped the generation while we were
            # spinning up, leave the newer run's slot alone — we'll be
            # discarded by the stale-result check in _handle_message_with_agent.
            if run_generation is not None and not self.sessions.run_is_current(
                session_key, run_generation
            ):
                logger.info(
                    "Skipping stale agent promotion for %s — generation %s is no longer current",
                    session_key or "",
                    run_generation,
                )
                return
            self.sessions.state(session_key).turn.agent = agent_holder[0]
            if self._draining:
                self._update_runtime_status("draining")

        tracking_task = asyncio.create_task(track_agent())

        # Monitor for interrupts from the adapter (new messages arriving).
        # This is the PRIMARY interrupt path for regular text messages —
        # Level 1 (base.py) catches them before _handle_message() is reached,
        # so the Level 2 interruption path never fires.
        # The inactivity poll loop below has a BACKUP check in case this
        # task dies (no error handling = silent death = lost interrupts).
        _interrupt_detected = asyncio.Event()  # shared with backup check

        async def monitor_for_interrupt():
            if not session_key:
                return

            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                try:
                    # Re-resolve adapter each iteration so reconnects don't
                    # leave us holding a stale reference.
                    _adapter = self._adapter_for_source(source)
                    if not _adapter:
                        continue
                    # Check if adapter has a pending interrupt for this session.
                    # Must use session_key (build_session_key output) — NOT
                    # source.chat_id — because the adapter stores interrupt events
                    # under the full session key.
                    if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                        agent = agent_holder[0]
                        if agent:
                            # Peek at the pending message text WITHOUT consuming it.
                            # The message must remain in _pending_messages so the
                            # post-run dequeue at _dequeue_pending_event() can
                            # retrieve the full MessageEvent (with media metadata).
                            # If we pop here, a race exists: the agent may finish
                            # before checking _interrupt_requested, and the message
                            # is lost — neither the interrupt path nor the dequeue
                            # path finds it.
                            _peek_event = _adapter._pending_messages.get(session_key)
                            pending_text = None
                            if _peek_event is not None:
                                pending_text = _peek_event.text or ""
                                # Transcribe audio media BEFORE signaling the
                                # agent, so voice messages interrupt with the
                                # real transcript instead of an empty string
                                # (or file-path placeholder). Matches the UX
                                # of fresh voice messages including the
                                # optional 🎙️ echo back to the user.
                                _media_urls = getattr(_peek_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_peek_event):
                                    pending_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _peek_event,
                                        _adapter,
                                        source,
                                        pending_text,
                                        log_context="Voice-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not pending_text and _media_urls:
                                    pending_text = _build_media_placeholder(_peek_event)
                            logger.debug("Interrupt detected from adapter, signaling agent...")
                            interruption.interrupt(agent, pending_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as _mon_err:
                    logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)

        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())

        # Periodic "still working" notifications for long-running tasks.
        # Fires every N seconds so the user knows the agent hasn't died.
        # Config: agent.gateway_notify_interval in config.yaml, or
        # HERMES_AGENT_NOTIFY_INTERVAL env var.  Default 180s (3 min).
        # 0 = disable notifications.
        _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
        _long_running_mode = _display_surface_mode(
            "long_running_notifications",
            default=True,
            allow_generic=True,
        )
        if _long_running_mode == "off":
            _NOTIFY_INTERVAL = None
        _notify_start = time.time()

        async def _notify_long_running():
            if _NOTIFY_INTERVAL is None:
                return  # Notifications disabled (gateway_notify_interval: 0)
            _notify_adapter = self._adapter_for_source(source)
            if not _notify_adapter:
                return
            # Track the heartbeat message id so we can edit-in-place on
            # platforms that support it (Telegram, Discord, Slack, etc.)
            # instead of spamming a new "Still working" bubble every
            # interval. Falls back to send-new when edit fails or isn't
            # supported by the adapter.
            _heartbeat_msg_id: Optional[str] = None
            while True:
                await asyncio.sleep(_NOTIFY_INTERVAL)
                # Stop heartbeating once this run no longer owns the session
                # slot or the executor has finished — otherwise a stale
                # "running: delegate_task" bubble can outlive the run that
                # spawned it (#12029). _executor_task is a closure var bound
                # just after this task is scheduled; tolerate the brief window
                # before then (the first wake is _NOTIFY_INTERVAL away anyway).
                try:
                    _exec_ref = _executor_task
                except NameError:
                    _exec_ref = None
                if not self._should_emit_long_running_notification(
                    session_key, agent_holder[0], _exec_ref
                ):
                    break
                _elapsed_mins = int((time.time() - _notify_start) // 60)
                # Include agent activity context if available. Default
                # heartbeat is terse: elapsed + current tool. Verbose
                # iteration counter is gated on busy_ack_detail so users
                # who want it can opt in per platform.
                _agent_ref = agent_holder[0]
                _status_detail = ""
                _want_iteration_detail = bool(
                    resolve_display_setting(
                        user_config,
                        platform_key,
                        "busy_ack_detail",
                        True,
                    )
                )
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _a = status_output.get_activity_summary(_agent_ref)
                        _parts = []
                        if _want_iteration_detail:
                            _parts.append(
                                f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                            )
                        _action = _a.get("current_tool") or _a.get("last_activity_desc")
                        if _action:
                            _parts.append(str(_action))
                        if _parts:
                            _status_detail = " — " + ", ".join(_parts)
                    except Exception:
                        pass
                _heartbeat_text = (
                    _generic_status_phrase("status")
                    if _long_running_mode == "generic"
                    else f"⏳ Working — {_elapsed_mins} min{_status_detail}"
                )
                try:
                    _notify_res = None
                    if _heartbeat_msg_id:
                        try:
                            _notify_res = await _notify_adapter.edit_message(
                                source.chat_id,
                                _heartbeat_msg_id,
                                _heartbeat_text,
                            )
                        except Exception as _ee:
                            logger.debug("Heartbeat edit failed: %s", _ee)
                            _notify_res = None
                    if not (_notify_res and getattr(_notify_res, "success", False)):
                        _notify_res = await _notify_adapter.send(
                            source.chat_id,
                            _heartbeat_text,
                            metadata=_non_conversational_metadata(_status_thread_metadata, platform=source.platform),
                        )
                        if getattr(_notify_res, "success", False) and getattr(
                            _notify_res, "message_id", None
                        ):
                            _heartbeat_msg_id = str(_notify_res.message_id)
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(_heartbeat_msg_id)
                except Exception as _ne:
                    logger.debug("Long-running notification error: %s", _ne)

        _notify_task = asyncio.create_task(_notify_long_running())

        def _stream_confirmed_final_delivery(
            consumer,
            final_text: str,
            *,
            previewed: bool = False,
        ) -> bool:
            """Return True only when the actual final reply reached the user."""
            if consumer is None:
                return False
            if getattr(consumer, "final_response_sent", False):
                # A successful finalize call is not proof the *content* was
                # final: the edit may have carried only the last preview
                # snapshot while the tail generated between that snapshot and
                # stream completion never reached any API call (#71643).
                # Reconcile the recorded turn-final payload against the
                # completed response; only a demonstrable mismatch (False)
                # overrides the flag — including payload-less multi-message
                # split delivery (#78541). None (no record on a non-split
                # legacy path) keeps the legacy trust so ambiguous-timeout
                # dedup is not regressed.
                matcher = getattr(consumer, "delivered_final_matches", None)
                if callable(matcher):
                    try:
                        if matcher(final_text) is False:
                            return False
                    except Exception:
                        pass
                return True
            if previewed:
                has_delivered_text = getattr(consumer, "has_delivered_text", None)
                if callable(has_delivered_text):
                    try:
                        return bool(has_delivered_text(final_text))
                    except Exception:
                        return False
            return False

        try:
            # Run in thread pool to not block.  Use an *inactivity*-based
            # timeout instead of a wall-clock limit: the agent can run for
            # hours if it's actively calling tools / receiving stream tokens,
            # but a hung API call or stuck tool with no activity for the
            # configured duration is caught and killed.  (#4815)
            #
            # Config: agent.gateway_timeout in config.yaml, or
            # HERMES_AGENT_TIMEOUT env var (env var takes precedence).
            # Default 1800s (30 min inactivity).  0 = unlimited.
            _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
            _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
            _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
            _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None
            _warning_fired = False

            # A background=true process intentionally survives a successful
            # turn, so capture existing IDs and reap only children created by
            # THIS turn if it times out. The daemon watchdog is independent of
            # asyncio: cgroup memory reclaim may starve the event loop that runs
            # the normal timeout poll, but it need not also postpone cleanup
            # until the loop recovers (#76115).
            from tools.process_registry import process_registry

            _turn_task_id = session_id or ""
            _turn_process_baseline = process_registry.snapshot_running_ids(_turn_task_id)
            turn_ctx.process_task_id = _turn_task_id
            turn_ctx.process_baseline = _turn_process_baseline
            _turn_worker_done = threading.Event()
            _turn_timeout_fired = threading.Event()
            _turn_cleanup_lock = threading.Lock()
            # task_id above is session-scoped, not turn-scoped (#76115
            # review): gate the eventual reap on this exact claim still
            # being current, so a replacement turn that starts on the same
            # session before the watchdog fires doesn't get its own fresh
            # process killed by this turn's stale baseline.
            _turn_run_generation = run_generation
            _turn_is_current = (
                (lambda: self.sessions.run_is_current(session_key, _turn_run_generation))
                if _turn_run_generation is not None
                else (lambda: True)
            )

            def _run_sync_with_timeout_lifecycle():
                try:
                    return run_sync()
                finally:
                    _turn_worker_done.set()
                    # `.turn.agent` on the session state is only reset to
                    # _AGENT_PENDING_SENTINEL when the *next* turn is
                    # claimed (see _session_state(...).turn.agent = ... at
                    # claim time), so a stale reference to this exact agent
                    # instance stays reachable from
                    # _interrupt_and_clear_session() until then. Clearing
                    # the ownership markers here — the instant this turn's
                    # own worker finishes — closes that window: an
                    # explicit /stop landing on the already-finished turn
                    # no longer reaps background work the turn deliberately
                    # left running (#76115).
                    _finished_agent = agent_holder[0] if agent_holder else None
                    if _finished_agent is not None:
                        _finished_agent._gateway_turn_process_task_id = ""
                        _finished_agent._gateway_turn_process_baseline = frozenset()

            if _agent_timeout is not None:
                threading.Thread(
                    target=_watch_gateway_turn_inactivity,
                    kwargs={
                        "agent_holder": agent_holder,
                        "task_id": _turn_task_id,
                        "process_baseline": _turn_process_baseline,
                        "timeout": _agent_timeout,
                        "worker_done": _turn_worker_done,
                        "timeout_fired": _turn_timeout_fired,
                        "cleanup_lock": _turn_cleanup_lock,
                        "poll_interval": 5.0,
                        "is_still_current": _turn_is_current,
                    },
                    name=f"gateway-turn-watchdog-{_turn_task_id[:12]}",
                    daemon=True,
                ).start()
            _executor_task = asyncio.ensure_future(
                self._run_in_executor_with_context(_run_sync_with_timeout_lifecycle)
            )

            _inactivity_timeout = False
            _POLL_INTERVAL = 5.0

            if _agent_timeout is None:
                # Unlimited — still poll periodically for backup interrupt
                # detection in case monitor_for_interrupt() silently died.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Backup interrupt check: if the monitor task died or
                    # missed the interrupt, catch it here.
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self._adapter_for_source(source)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            if _bp_event is not None:
                                _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_bp_event):
                                    _bp_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _bp_event,
                                        _backup_adapter,
                                        source,
                                        _bp_text or "",
                                        log_context="Voice-backup-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not _bp_text and _bp_media_urls:
                                    _bp_text = _build_media_placeholder(_bp_event)
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            interruption.interrupt(_backup_agent, _bp_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")

            else:
                # Poll loop: check the agent's built-in activity tracker
                # (updated by _touch_activity() on every tool call, API
                # call, and stream delta) every few seconds.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        # Prefer the real result when the worker finished,
                        # even if the watchdog fired in the same window: the
                        # completed run already persisted its reply to session
                        # history, so surfacing the "agent inactive" diagnostic
                        # here would contradict the stored transcript. This
                        # mirrors _abandon_timed_out_gateway_turn's own
                        # worker_done-wins tiebreak (under cleanup_lock).
                        response = _executor_task.result()
                        break
                    if _turn_timeout_fired.is_set():
                        _inactivity_timeout = True
                        break
                    # Agent still running — check inactivity.
                    _agent_ref = agent_holder[0]
                    _idle_secs = 0.0
                    if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                        try:
                            _act = status_output.get_activity_summary(_agent_ref)
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    # Staged warning: fire once before escalating to full timeout.
                    if (not _warning_fired and _agent_warning is not None
                            and _idle_secs >= _agent_warning):
                        _warning_fired = True
                        _warn_adapter = self._adapter_for_source(source)
                        if _warn_adapter:
                            _elapsed_warn = int(_agent_warning // 60) or 1
                            _remaining_mins = int((_agent_timeout - _agent_warning) // 60) or 1
                            try:
                                await _warn_adapter.send(
                                    source.chat_id,
                                    f"⚠️ No activity for {_elapsed_warn} min. "
                                    f"If the agent does not respond soon, it will "
                                    f"be timed out in {_remaining_mins} min. "
                                    f"You can continue waiting or use /reset.",
                                    metadata=_status_thread_metadata,
                                )
                            except Exception as _warn_err:
                                logger.debug("Inactivity warning send error: %s", _warn_err)
                    if _idle_secs >= _agent_timeout:
                        _inactivity_timeout = True
                        threading.Thread(
                            target=_abandon_timed_out_gateway_turn,
                            kwargs={
                                "agent_holder": agent_holder,
                                "task_id": _turn_task_id,
                                "process_baseline": _turn_process_baseline,
                                "worker_done": _turn_worker_done,
                                "timeout_fired": _turn_timeout_fired,
                                "cleanup_lock": _turn_cleanup_lock,
                                "is_still_current": _turn_is_current,
                            },
                            name=f"gateway-turn-reaper-{_turn_task_id[:12]}",
                            daemon=True,
                        ).start()
                        break
                    # Backup interrupt check (same as unlimited path).
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self._adapter_for_source(source)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            if _bp_event is not None:
                                _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                                if self._pending_event_audio_paths(_bp_event):
                                    _bp_text, _ = await self._transcribe_and_echo_pending_voice(
                                        _bp_event,
                                        _backup_adapter,
                                        source,
                                        _bp_text or "",
                                        log_context="Voice-backup-interrupt",
                                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                    )
                                elif not _bp_text and _bp_media_urls:
                                    _bp_text = _build_media_placeholder(_bp_event)
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            interruption.interrupt(_backup_agent, _bp_text)
                            _interrupt_detected.set()
                            # Abort streaming TTS on barge-in (#60671).
                            _stts = streaming_tts_consumer_holder[0]
                            if _stts is not None:
                                _stts.abort("barge-in")

            if _inactivity_timeout:
                # Build a diagnostic summary from the agent's activity tracker.
                _timed_out_agent = agent_holder[0]
                _activity = {}
                if _timed_out_agent:
                    try:
                        _activity = status_output.get_activity_summary(_timed_out_agent)
                    except Exception:
                        pass

                _last_desc = _activity.get("last_activity_desc", "unknown")
                _secs_ago = _activity.get("seconds_since_activity", 0)
                _cur_tool = _activity.get("current_tool")
                _iter_n = _activity.get("api_call_count", 0)
                _iter_max = _activity.get("max_iterations", 0)

                logger.error(
                    "Agent idle for %.0fs (timeout %.0fs) in session %s "
                    "| last_activity=%s | iteration=%s/%s | tool=%s",
                    _secs_ago, _agent_timeout, session_key,
                    _last_desc, _iter_n, _iter_max,
                    _cur_tool or "none",
                )

                # Interrupt the agent if it's still running so the thread
                # pool worker is freed.
                if _timed_out_agent:
                    interruption.hard_interrupt(_timed_out_agent, _INTERRUPT_REASON_TIMEOUT)

                _timeout_mins = int(_agent_timeout // 60) or 1

                # Construct a user-facing message with diagnostic context.
                _diag_lines = [
                    f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                    f"or API responses."
                ]
                if _cur_tool:
                    _diag_lines.append(
                        f"The agent appears stuck on tool `{_cur_tool}` "
                        f"({_secs_ago:.0f}s since last activity, "
                        f"iteration {_iter_n}/{_iter_max})."
                    )
                else:
                    _diag_lines.append(
                        f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                        f"iteration {_iter_n}/{_iter_max}). "
                        "The agent may have been waiting on an API response."
                    )
                _diag_lines.append(
                    "To increase the limit, set agent.gateway_timeout in config.yaml "
                    "(value in seconds, 0 = no limit) and restart the gateway.\n"
                    "Try again, or use /reset to start fresh."
                )

                response = {
                    "final_response": "\n".join(_diag_lines),
                    "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                    "api_calls": _iter_n,
                    "tools": tools_holder[0] or [],
                    "history_offset": 0,
                    "failed": True,
                }

            # Track fallback model state: if the agent switched to a
            # fallback model during this run, persist it so /model shows
            # the actually-active model instead of the config default.
            # Skip eviction when the run failed — evicting a failed agent
            # forces MCP reinit on the next message for no benefit (the
            # same error will recur).  This was the root cause of #7130:
            # a bad model ID triggered fallback → eviction → recreation →
            # MCP reinit → same 400 → loop, burning 91% CPU for hours.
            _agent = agent_holder[0]
            _result_for_fb = result_holder[0]
            _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
            if _agent is not None and hasattr(_agent, 'model') and not _run_failed:
                _cfg_model = _resolve_gateway_model()
                # Normalize _cfg_model the same way create_agent.__init__ does, so a
                # vendor-prefixed config value (e.g. "deepseek/deepseek-v4-pro")
                # matches the agent's stripped model ("deepseek-v4-pro") on
                # native providers. Without this, _agent.model != _cfg_model is
                # always true for vendor-prefixed config and the cached agent is
                # evicted on every successful turn — destroying prompt caching.
                # Aggregators (openrouter, etc.) keep the vendor/model slug, so
                # they're left untouched.
                try:
                    from hermes_cli.model_normalize import (
                        _AGGREGATOR_PROVIDERS,
                        normalize_model_for_provider,
                    )
                    _agent_provider = getattr(_agent, 'provider', '') or ''
                    if _agent_provider and _agent_provider not in _AGGREGATOR_PROVIDERS:
                        _cfg_model = normalize_model_for_provider(_cfg_model, _agent_provider)
                except Exception:
                    pass
                if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
                    # Fallback activated on a successful run — evict cached
                    # agent so the next message retries the primary model.
                    self.agent_cache.evict(session_key)

            # Check if we were interrupted OR have a queued message (/queue).
            result = result_holder[0]
            adapter = self._adapter_for_source(source)

            # Finalize the streaming-TTS consumer (#60671).
            #
            # finish() is called from the outer event-loop thread (not the
            # executor worker) so early returns from run_sync are also
            # finalised.  wait_complete() drains queued audio; on timeout
            # the consumer is aborted unconditionally — if audio was
            # audible, suppression is preserved so the gateway does not
            # replay from the beginning; if no audio was audible, the
            # whole-file fallback path is permitted.
            _stts = streaming_tts_consumer_holder[0]
            if _stts is not None:
                _stts.finish()
                try:
                    await _stts.wait_complete(timeout=10.0)
                except Exception as _stts_done_err:
                    logger.debug("streaming TTS wait_complete error: %s", _stts_done_err)
                if not _stts.done:
                    # Timeout before or after audible audio: abort to free
                    # the consumer task.  Audible streams retain suppression;
                    # silent streams remain eligible for whole-file fallback.
                    _stts.abort("streaming TTS finalisation timeout")
                    await _stts.wait_complete(timeout=2.0)
                if _stts.suppress_whole_file and adapter is not None:
                    _mark_turn = getattr(adapter, "_mark_streaming_tts_completed_turn", None)
                    if callable(_mark_turn):
                        _mark_turn(session_key, run_generation)

            # Get pending message from adapter.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending_event = None
            pending = None
            if result and adapter and session_key:
                pending_event = _dequeue_pending_event(adapter, session_key)
                # /queue overflow: after consuming the adapter's "next-up"
                # slot, promote the next queued event into it so the
                # recursive run's drain will see it.  This keeps the slot
                # occupied for the full FIFO chain, which (a) preserves
                # order, and (b) causes any mid-chain /queue to correctly
                # route to overflow rather than jumping the queue.
                pending_event = self.sessions.promote_queued_event(session_key, adapter, pending_event)
                if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                    interrupt_message = result.get("interrupt_message")
                    if _is_control_interrupt_message(interrupt_message):
                        logger.info(
                            "Ignoring control interrupt message for session %s: %s",
                            session_key or "?",
                            interrupt_message,
                        )
                    else:
                        pending = interrupt_message
                elif pending_event:
                    # Transcribe audio media on the dequeued event BEFORE it is
                    # handed back as the next user turn, so queued/interrupting
                    # voice messages drain with the real transcript instead of
                    # a file-path placeholder. When configured, echo each
                    # transcript back to the user in the same 🎙️ format as
                    # fresh voice messages.
                    _pending_text = pending_event.text or ""
                    _media_urls = getattr(pending_event, "media_urls", None) or []
                    if self._pending_event_audio_paths(pending_event):
                        pending, _ = await self._transcribe_and_echo_pending_voice(
                            pending_event,
                            adapter,
                            source,
                            _pending_text,
                            log_context="Voice-drain",
                            metadata={"thread_id": source.thread_id} if source.thread_id else None,
                        )
                        if not pending:
                            pending = _build_media_placeholder(pending_event)
                    else:
                        pending = _pending_text or _build_media_placeholder(pending_event)
                    if pending:
                        logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

            # Leftover /steer: if a steer arrived after the last tool batch
            # (e.g. during the final API call), the agent couldn't inject it
            # and returned it in result["pending_steer"]. Deliver it as the
            # next user turn so it isn't silently dropped.
            if result and not pending and not pending_event:
                _leftover_steer = result.get("pending_steer")
                if _leftover_steer:
                    pending = _leftover_steer
                    logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

            # Safety net: if the pending text is a slash command (e.g. "/stop",
            # "/new"), discard it — commands should never be passed to the agent
            # as user input.  The primary fix is in base.py (commands bypass the
            # active-session guard), but this catches edge cases where command
            # text leaks through the interrupt_message fallback.
            if pending and pending.strip().startswith("/"):
                _pending_parts = pending.strip().split(None, 1)
                _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
                if _pending_cmd_word:
                    try:
                        from hermes_cli.commands import resolve_command as _rc_pending
                        if _rc_pending(_pending_cmd_word):
                            logger.info(
                                "Discarding command '/%s' from pending queue — "
                                "commands must not be passed as agent input",
                                _pending_cmd_word,
                            )
                            pending_event = None
                            pending = None
                    except Exception:
                        pass

            if self._draining and (pending_event or pending):
                logger.info(
                    "Discarding pending follow-up for session %s during gateway %s",
                    session_key or "?",
                    self._status_action_label(),
                )
                pending_event = None
                pending = None

            if pending_event or pending:
                logger.debug("Processing pending message: '%s...'", pending[:40])

                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                    adapter._active_sessions[session_key].clear()

                # Cap recursion depth to prevent resource exhaustion when the
                # user sends multiple messages while the agent keeps failing. (#816)
                if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                    logger.warning(
                        "Interrupt recursion depth %d reached for session %s — "
                        "queueing message instead of recursing.",
                        _interrupt_depth, session_key,
                    )
                    adapter = self._adapter_for_source(source)
                    if adapter and pending_event:
                        merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
                    elif adapter and hasattr(adapter, 'queue_message'):
                        adapter.queue_message(session_key, pending)
                    return result_holder[0] or {"final_response": response, "messages": history}

                was_interrupted = result.get("interrupted")
                if not was_interrupted:
                    # Queued message after normal completion — deliver the first
                    # response before processing the queued follow-up.
                    # Skip if streaming already delivered it.
                    _sc = stream_consumer_holder[0]
                    if _sc and stream_task:
                        try:
                            await asyncio.wait_for(stream_task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            stream_task.cancel()
                            try:
                                await stream_task
                            except asyncio.CancelledError:
                                pass
                        except Exception as e:
                            logger.debug("Stream consumer wait before queued message failed: %s", e)
                    # The queued branch needs raw ``result`` for interruption,
                    # history, and recursion state, but delivery must use the
                    # finalized task result. The latter contains empty/failure
                    # normalization and any final response processing applied by
                    # _run_agent_task; sending the raw copy bypasses those steps.
                    _delivery_result = response if isinstance(response, dict) else (result or {})
                    _previewed = bool(_delivery_result.get("response_previewed"))
                    first_response = _delivery_result.get("final_response", "")
                    _already_streamed = _stream_confirmed_final_delivery(
                        _sc,
                        first_response,
                        previewed=_previewed,
                    )
                    # Apply the same predicate as the normal completed-turn path.
                    # This direct queued-send branch predates intentional-silence
                    # filtering, so without this check it leaks the literal marker.
                    try:
                        from gateway.response_filters import is_intentional_silence_agent_result
                        _intentional_silence = is_intentional_silence_agent_result(
                            _delivery_result, first_response,
                        )
                    except Exception:
                        _intentional_silence = False
                    if _intentional_silence:
                        logger.info(
                            "Queued follow-up for session %s: suppressing intentional silence marker before continuing.",
                            session_key or "?",
                        )
                    elif first_response:
                        try:
                            if _already_streamed:
                                logger.info(
                                    "Queued follow-up for session %s: final text delivery confirmed; delivering explicit media before continuing.",
                                    session_key or "?",
                                )
                            else:
                                logger.info(
                                    "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                                    session_key or "?",
                                )
                            await self._deliver_queued_first_response(
                                first_response,
                                source=source,
                                adapter=adapter,
                                metadata=_status_thread_metadata,
                                event_message_id=event_message_id,
                                text_already_delivered=_already_streamed,
                                deliver_media=not _delivery_result.get("failed"),
                            )
                        except Exception as e:
                            logger.warning("Failed to send first response before queued message: %s", e)
                    # Release deferred bg-review notifications now that the
                    # first response has been delivered.  Pop from the
                    # adapter's callback dict (prevents double-fire in
                    # base.py's finally block) and call it.
                    if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
                        _bg_cb = adapter.pop_post_delivery_callback(
                            session_key,
                            generation=run_generation,
                        )
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                    elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
                        _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                # else: interrupted — discard the interrupted response ("Operation
                # interrupted." is just noise; the user already knows they sent a
                # new message).

                updated_history = result.get("messages", history)
                next_source = source
                next_message = pending
                next_message_id = None
                next_channel_prompt = None
                next_session_key = session_key
                # #60671 — carry the pending event's message_type into the
                # recursive call so queued voice turns can stream TTS and
                # re-mark the generation for the final delivered turn.
                next_message_type = None
                if pending_event is not None:
                    next_source = getattr(pending_event, "source", None) or source
                    if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                        logger.info(
                            "Discarding stale goal continuation for session %s — goal is no longer active",
                            session_key or "?",
                        )
                        return result
                    # Resolve the follow-up's session key BEFORE preparing the
                    # inbound text: _prepare_inbound_message_text buffers native
                    # image paths under the key it is given, and the recursive
                    # _run_agent below consumes them under next_session_key.
                    # The write and consume keys must match or the images drop.
                    try:
                        next_session_key = self._session_key_for_source(next_source)
                    except Exception:
                        logger.debug(
                            "Queued follow-up session-key resolution failed; reusing %s",
                            session_key or "?",
                            exc_info=True,
                        )
                    next_message = await self._prepare_profile_scoped_inbound_message_text(
                        event=pending_event,
                        source=next_source,
                        history=updated_history,
                        session_key=next_session_key,
                    )
                    if next_message is None:
                        return result
                    next_message_id = self._reply_anchor_for_event(pending_event)
                    next_channel_prompt = getattr(pending_event, "channel_prompt", None)
                    next_message_type = getattr(pending_event, "message_type", None)

                # Clear the completed streaming marker from the prior logical
                # turn so the recursive turn's streaming TTS is not suppressed
                # by the prior turn's completion (#60671).
                _clear_adapter = self._adapter_for_source(source)
                if _clear_adapter is not None and session_key and run_generation is not None:
                    _completed_turns = getattr(_clear_adapter, "_streaming_tts_completed_turns", None)
                    if _completed_turns is not None:
                        _prior_key = getattr(_clear_adapter, "_streaming_tts_turn_key", None)
                        if callable(_prior_key):
                            _pk = _prior_key(session_key, run_generation)
                            if _pk:
                                _completed_turns.discard(_pk)

                # Restart typing indicator so the user sees activity while
                # the follow-up turn runs.  The outer _process_message_background
                # typing task is still alive but may be stale.
                _followup_adapter = self._adapter_for_source(source)
                if _followup_adapter:
                    try:
                        await _followup_adapter.send_typing(
                            source.chat_id,
                            metadata=_status_thread_metadata,
                        )
                    except Exception:
                        pass

                # Re-baseline the cached agent's message_count snapshot before
                # recursing into the in-band queued (/queue) follow-up turn.
                # The first turn has completed and flushed its own user +
                # assistant rows to the SessionDB, so the cross-process
                # coherence guard (#45966) — which this recursive _run_agent
                # call re-enters — would otherwise see the grown on-disk count
                # against the stale build-time snapshot and rebuild the agent
                # on THIS process's OWN writes, destroying the prompt-cache
                # prefix #46237 was merged to preserve.  The existing
                # re-baseline in _handle_message_with_agent only runs after the
                # whole _run_agent chain unwinds — too late for the in-band
                # follow-up.  Use the same (session_key, session_id) the
                # recursive call runs under so the snapshot matches exactly
                # what the follow-up's guard will consult.  Fail-safe in helper.
                await self.agent_cache.refresh_message_count(session_key, session_id)

                followup_result = await self._run_agent(
                    message=next_message,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=next_source,
                    session_id=session_id,
                    session_key=next_session_key,
                    run_generation=run_generation,
                    _interrupt_depth=_interrupt_depth + 1,
                    event_message_id=next_message_id,
                    channel_prompt=next_channel_prompt,
                    message_type=next_message_type,
                )
                return _preserve_queued_followup_history_offset(result, followup_result)
        finally:
            # Stop progress sender, interrupt monitor, and notification task
            if progress_task:
                progress_task.cancel()
            if log_task:
                log_task.cancel()
            interrupt_monitor.cancel()
            _notify_task.cancel()

            # Wait for stream consumer to finish its final edit
            if stream_task:
                # If the agent never created a stream consumer (e.g. non-
                # streaming code path, or a test stub returning synchronously)
                # there is nothing to flush — cancel immediately instead of
                # waiting out the 5s timeout on a task that's just polling for
                # a consumer that will never arrive.  This was a 5-second
                # cost per non-streaming test run.
                _has_stream_consumer = (
                    stream_consumer_holder
                    and stream_consumer_holder[0] is not None
                )
                if not _has_stream_consumer:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                else:
                    try:
                        await asyncio.wait_for(stream_task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass

            # Unconditional abort + bounded wait for the streaming-TTS
            # consumer (#60671 hardening).  Covers cancellation / exception
            # paths where the normal finalisation block was skipped.
            _stts_finally = streaming_tts_consumer_holder[0]
            if _stts_finally is not None and not _stts_finally.done:
                _stts_finally.abort("cleanup")
                try:
                    await _stts_finally.wait_complete(timeout=2.0)
                except Exception:
                    pass

            # Clean up tracking
            tracking_task.cancel()
            if session_key:
                # Only release the slot if this run's generation still owns
                # it.  A /stop or /new that bumped the generation while we
                # were unwinding has already installed its own state; this
                # guard prevents an old run from clobbering it on the way
                # out.
                self._release_turn_state(
                    session_key, run_generation=run_generation
                )
            if self._draining:
                self._update_runtime_status("draining")

            # Wait for cancelled tasks
            for task in [progress_task, log_task, interrupt_monitor, tracking_task, _notify_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # If streaming already delivered the response, mark it so the
        # caller's send() is skipped (avoiding duplicate messages).
        # BUT: never suppress delivery when the agent failed — the error
        # message is new content the user hasn't seen, and it must reach
        # them even if streaming had sent earlier partial output.
        #
        # Also never suppress when the final response is "(empty)" — this
        # means the model failed to produce content after tool calls (common
        # with mimo-v2-pro, GLM-5, etc.).  The stream consumer may have
        # sent intermediate text ("Let me search for that…") alongside the
        # tool call, setting already_sent=True, but that text is NOT the
        # final answer.  Suppressing delivery here leaves the user staring
        # at silence.  (#10xxx — "agent stops after web search")
        _sc = stream_consumer_holder[0]
        if isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            # response_previewed means the interim_assistant_callback already
            # saw the final text, but only suppress the normal send if that
            # exact final text was delivered. Unrelated commentary/progress
            # must not be mistaken for the final response (#14238).
            _previewed = bool(response.get("response_previewed"))
            _content_delivered = bool(
                _sc and getattr(_sc, "final_content_delivered", False)
            )
            # #71643: a *successful* finalize edit can still carry only the
            # last preview snapshot — deltas generated between that edit and
            # stream completion never reach any API call, and both suppression
            # flags are set from the call's success rather than its content.
            # Reconcile the consumer's recorded turn-final payload against the
            # completed response: on a demonstrable mismatch (False) neither
            # final_response_sent nor final_content_delivered may suppress the
            # normal final send. False also covers payload-less multi-message
            # split delivery (#78541). None (no record on a non-split legacy
            # path) keeps legacy trust; the failed-finalize family
            # (#51828 / #33793) is unaffected because those paths leave the
            # flags False or record the complete fallback payload.
            _stale_finalized = False
            if _content_delivered and not _is_empty_sentinel:
                _matcher = getattr(_sc, "delivered_final_matches", None)
                if callable(_matcher):
                    try:
                        _stale_finalized = _matcher(_final) is False
                    except Exception:
                        _stale_finalized = False
                if _stale_finalized:
                    _content_delivered = False
            # Plugin hooks (e.g. transform_llm_output) may have appended content
            # after streaming finished — when the response was transformed, always
            # send the final version so the appended content reaches the client.
            _transformed = bool(response.get("response_transformed"))
            # Only suppress the normal send when the actual final reply reached
            # the user: the stream consumer streamed it (final_response_sent /
            # final_content_delivered), or the interim preview delivered that
            # *exact* final text. Unrelated commentary/progress shown during a
            # compression/session split must not be mistaken for the final
            # response (#14238).
            _streamed = _stream_confirmed_final_delivery(
                _sc,
                _final,
                previewed=_previewed,
            )
            if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
                logger.info(
                    "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                )
                response["already_sent"] = True
            elif not _is_empty_sentinel and not _transformed and _stale_finalized and _sc is not None:
                # Stale finalize (#71643): the streamed message holds only the
                # last preview snapshot. Prefer editing it up to the complete
                # response (same shape as the transformed branch below) so the
                # user gets one corrected message; on edit failure fall through
                # with already_sent unset so the normal final send delivers the
                # complete text.
                #
                # Not valid for a multi-message split delivery: there
                # ``message_id`` is only the LAST chunk, so editing it with the
                # complete response would repeat every sealed head chunk's text
                # inside the tail message. Fall through to the normal final send
                # instead (#78541).
                _sc_msg_id = _sc.message_id
                _sc_adapter = getattr(_sc, "adapter", None)
                if getattr(_sc, "_turn_split_delivery", False):
                    logger.info(
                        "Stale streamed finalize detected for session %s on a multi-message split; skipping the in-place reconciliation edit and delivering the complete response via normal final send (#78541).",
                        session_key or "?",
                    )
                elif _sc_msg_id and _sc_msg_id != "__no_edit__" and _sc_adapter is not None:
                    try:
                        _reconcile_res = await _sc_adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=_final,
                            finalize=True,
                        )
                        if getattr(_reconcile_res, "success", True):
                            response["already_sent"] = True
                            logger.info(
                                "Reconciled stale streamed finalize for session %s: edited message %s with the complete response (#71643).",
                                session_key or "?", _sc_msg_id,
                            )
                        else:
                            logger.warning(
                                "Stale-finalize reconciliation edit failed for session %s (%s); sending complete response via normal final send.",
                                session_key or "?",
                                getattr(_reconcile_res, "error", None),
                            )
                    except Exception as _edit_err:
                        logger.warning(
                            "Stale-finalize reconciliation edit failed for session %s: %s; sending complete response via normal final send.",
                            session_key or "?", _edit_err,
                        )
                else:
                    logger.info(
                        "Stale streamed finalize detected for session %s with no editable message; delivering complete response via normal final send (#71643).",
                        session_key or "?",
                    )
            elif not _is_empty_sentinel and _transformed and _sc is not None:
                # Plugin hooks transformed the response after streaming — edit the
                # existing streamed message instead of sending a duplicate.
                _sc_msg_id = _sc.message_id
                if _sc_msg_id:
                    try:
                        await _sc.adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=response["final_response"],
                            finalize=True,
                        )
                        response["already_sent"] = True
                        logger.info(
                            "Edited streamed message %s for session %s to include plugin-transformed content.",
                            _sc_msg_id, session_key or "?",
                        )
                    except Exception as _edit_err:
                        logger.warning(
                            "Failed to edit streamed message for session %s: %s",
                            session_key or "?", _edit_err,
                        )

        # Schedule deletion of tracked temporary progress bubbles after the
        # final response lands. Failed runs skip this so bubbles remain as
        # breadcrumbs for the user to see what work happened. Only fires on
        # adapters that support ``delete_message`` (see init above); failures
        # are swallowed — deletion is best-effort.
        if (
            _cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            _ids_snapshot = list(_cleanup_msg_ids)
            _chat_id_snapshot = source.chat_id
            _adapter_snapshot = _cleanup_adapter
            _loop_snapshot = asyncio.get_running_loop()

            def _cleanup_temp_bubbles() -> None:
                async def _delete_all() -> None:
                    for _mid in _ids_snapshot:
                        try:
                            await _adapter_snapshot.delete_message(
                                _chat_id_snapshot, _mid
                            )
                        except Exception:
                            pass
                try:
                    safe_schedule_threadsafe(
                        _delete_all(), _loop_snapshot,
                        logger=logger,
                        log_message="Temp bubble cleanup scheduling error",
                    )
                except Exception:
                    pass

            try:
                _cleanup_adapter.register_post_delivery_callback(
                    session_key,
                    _cleanup_temp_bubbles,
                    generation=run_generation,
                )
            except Exception as _rpe:
                logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

        return response
