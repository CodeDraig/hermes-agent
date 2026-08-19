"""Gateway process lifecycle, draining, startup recovery, and watchdogs."""

from __future__ import annotations

import asyncio
import faulthandler
import json
import logging
import os
import shlex
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import agent.interruption as interruption
import agent.status_output as status_output
from gateway.config import Platform
from gateway.delivery import looks_like_telegram_private_chat_id, resolve_delivery_transport
from gateway.history import (
    _float_env,
    _startup_restore_drain_timeout_secs,
)
from gateway.media import _build_media_placeholder
from gateway.platform_runtime import _platform_has_bot_credential
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.process_notifications import _parse_session_key
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    parse_cron_drain_timeout,
    parse_restart_after_turn_timeout,
    parse_restart_drain_timeout,
    resolve_cron_drain_budget,
)
from gateway.runtime_config import (
    _get_channel_override,
    _load_gateway_config,
    _load_gateway_runtime_config,
    _platform_config_key,
    _resolve_gateway_model,
    _resolve_hermes_bin,
)
from gateway.session import SessionSource, auto_continue_freshness_window
from gateway.session_state import (
    AGENT_PENDING as _AGENT_PENDING_SENTINEL,
    SERVICE_TIER_UNSET as _SERVICE_TIER_UNSET,
)
from gateway.shutdown_watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    _arm_loop_floor_timer,
    arm_shutdown_watchdog,
    loop_heartbeat_forever,
    resolve_shutdown_watchdog_delay,
    start_loop_liveness_watchdog,
)
from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain
from hermes_constants import get_hermes_home
from utils import atomic_json_write, is_truthy_value

logger = logging.getLogger("gateway.run")
_hermes_home = get_hermes_home()
_STALL_NOTIFY_SEND_TIMEOUT_SECONDS = 15.0
_UNSET = object()


def _planned_restart_notification_path() -> Path:
    return _hermes_home / ".restart_pending.json"


def _shutdown_gateway_health_export(runner: Any) -> None:
    runtime = getattr(runner, "_gateway_health_export_runtime", None)
    if runtime is None:
        return
    runner._gateway_health_export_runtime = None
    try:
        runtime.shutdown()
    except Exception:
        logger.debug("gateway health OTLP export shutdown failed", exc_info=True)


_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"

def _restart_notification_pending() -> bool:
    """Return True when a /restart completion marker is waiting to be delivered."""
    return (_hermes_home / ".restart_notify.json").exists()

def _planned_restart_notification_pending() -> bool:
    """Return True when a non-chat planned restart should notify home channels."""
    return _planned_restart_notification_path().exists()

def _clear_planned_restart_notification() -> None:
    _planned_restart_notification_path().unlink(missing_ok=True)

class LifecycleRuntime:
    _BUSY_QUEUE_MAX_PENDING = 32
    _CLEANUP_TIMEOUT_S = 30.0
    _FINALIZE_TIMEOUT_S = 10.0
    _STUCK_LOOP_THRESHOLD = 3
    _STUCK_LOOP_FILE = ".restart_failure_counts"
    _AUTO_RESUME_REASONS = frozenset(
        {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
    )
    _MAX_SUPERVISED_RESTARTS = 5
    _SUPERVISED_HEALTHY_SECS = 300

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    def _active_work_count(self) -> int:
        """All agent work the gateway must expose and drain as one total."""
        return (
            self.sessions.running_count()
            + self._active_cron_job_count()
        )

    def _active_cron_job_count(self) -> int:
        """Count of cron jobs currently executing, from the cron scheduler's
        own in-flight tracking (``cron.scheduler._running_job_ids``).

        Cron jobs run through a standalone ``create_agent`` on the scheduler's own
        thread pool (``cron/scheduler.py::run_job``), entirely outside
        ``self._running_agents`` — the dict every OTHER active-work check on
        this class (``_running_agent_count``, ``_drain_active_agents``) reads.
        Without this, the shutdown drain is structurally blind to in-flight
        cron work: it can report ``active_at_start=0`` and proceed straight
        to killing tool subprocesses while a cron job's terminal command is
        still running (#60432). Best-effort: returns 0 if the cron module
        can't be imported (e.g. a minimal test double for this class).
        """
        try:
            from cron.scheduler import get_running_job_ids
            return len(get_running_job_ids())
        except Exception:
            return 0

    def _restart_loop_guard_config(self) -> tuple:
        """Return ``(max_restarts, window_seconds, max_gap_seconds)`` for the
        auto-resume restart-loop breaker (#30719, defense-3), read from
        ``gateway.restart_loop_guard`` in config.yaml with the module defaults
        as fallback. ``max_restarts <= 0`` disables the breaker.

        ``max_gap_seconds`` is the longest spacing between two consecutive
        restart-interrupted boots that still counts them as the same loop, so
        a crash cycle slower than ``window_seconds`` stays visible (#81642).
        """
        from gateway import restart_loop_guard as _rlg

        max_restarts = _rlg.DEFAULT_MAX_RESTARTS
        window_seconds = _rlg.DEFAULT_WINDOW_SECONDS
        max_gap_seconds = _rlg.DEFAULT_MAX_GAP_SECONDS
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            rlg = gw.get("restart_loop_guard") if isinstance(gw, dict) else None
            if isinstance(rlg, dict):
                if isinstance(rlg.get("max_restarts"), int):
                    max_restarts = rlg["max_restarts"]
                if isinstance(rlg.get("window_seconds"), int) and rlg["window_seconds"] > 0:
                    window_seconds = rlg["window_seconds"]
                if (
                    isinstance(rlg.get("max_gap_seconds"), int)
                    and rlg["max_gap_seconds"] > 0
                ):
                    max_gap_seconds = rlg["max_gap_seconds"]
        except Exception:  # noqa: BLE001
            pass
        return max_restarts, window_seconds, max_gap_seconds

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _queue_during_drain_enabled(
        self, busy_input_mode: Optional[str] = None
    ) -> bool:
        # Both "queue" and "steer" modes imply the user doesn't want messages
        # to be lost during restart — queue them for the newly-spawned gateway
        # process to pick up.  "interrupt" mode drops them (current behaviour).
        mode = busy_input_mode or self._busy_input_mode
        return self._restart_requested and mode in {"queue", "steer"}

    @staticmethod
    def _is_goal_continuation_event(event_or_text: Any) -> bool:
        """Return True for synthetic /goal continuation turns.

        Goal continuations are normal queued user-role events, so pause/clear
        must distinguish them from real user /queue messages before removing or
        suppressing them.
        """
        text = getattr(event_or_text, "text", event_or_text) or ""
        return str(text).startswith("[Continuing toward your standing goal]\nGoal:")

    def _clear_goal_pending_continuations(self, session_key: str, adapter: Any) -> int:
        """Remove queued synthetic /goal continuations for one session.

        User-issued /goal pause/clear can race with a continuation already
        queued by the judge.  Remove only synthetic goal continuations while
        preserving normal /queue and user follow-up events.
        """
        removed = 0
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if isinstance(pending_slot, dict):
            pending_event = pending_slot.get(session_key)
            if self._is_goal_continuation_event(pending_event):
                pending_slot.pop(session_key, None)
                removed += 1

        _q_state = self.sessions.peek(session_key)
        overflow = _q_state.conversation.queued_events if _q_state else []
        if overflow:
            kept = []
            for queued_event in overflow:
                if self._is_goal_continuation_event(queued_event):
                    removed += 1
                else:
                    kept.append(queued_event)
            _q_state.conversation.queued_events = kept
        return removed

    def _goal_still_active_for_session(self, session_id: str) -> bool:
        """Best-effort fresh DB check before running a queued continuation."""
        if not session_id:
            return False
        try:
            from hermes_cli.goals import GoalManager
            return GoalManager(session_id=session_id).is_active()
        except Exception as exc:
            logger.debug("goal continuation: active-state recheck failed: %s", exc)
            return False

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state=gateway_state,
                exit_reason=exit_reason,
                restart_requested=self._restart_requested,
                active_agents=self._active_work_count(),
            )
        except Exception:
            pass

    def _persist_active_agents(self) -> None:
        """Persist the live in-flight agent count to ``gateway_state.json``.

        Called at every turn boundary (a running-agent slot is claimed or
        released) so the dashboard ``/api/status`` readout reflects in-flight
        gateway turns in near-real-time.  Without this the file is only
        rewritten on lifecycle transitions, so any ``active_agents`` read
        between transitions is stale (a turn could start and finish without the
        file ever moving).

        Deliberately passes ONLY ``active_agents`` — ``gateway_state`` and the
        other fields stay ``_UNSET`` so ``write_runtime_status``'s
        read-merge-write preserves the current lifecycle state (``running`` /
        ``draining`` / …).  Passing ``gateway_state=None`` here would clobber it.
        Best-effort: a failed status write must never disrupt a turn.
        """
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(active_agents=self._active_work_count())
        except Exception:
            pass

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.

        Checks HERMES_PREFILL_MESSAGES_FILE first, then the top-level
        prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            cfg = _load_gateway_runtime_config()
            file_path = str(cfg.get("prefill_messages_file", "") or "")
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.

        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then
        ``display.personality`` / ``agent.system_prompt`` in config.yaml.
        """
        from hermes_cli.config import resolve_ephemeral_system_prompt_from_config

        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        cfg = _load_gateway_runtime_config()
        return resolve_ephemeral_system_prompt_from_config(cfg)

    def _resolve_model_for_channel(
        self,
        platform: Platform,
        chat_id: str,
        *,
        user_config: Optional[dict] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Resolve model for this channel: channel_overrides else global default.

        Delegates the precedence rule to
        :func:`hermes_cli.model_switch.resolve_effective_model` (session
        override > channel override > global default) — the single owner
        shared with the API server, so the two surfaces cannot diverge
        again (see 7dd00bb47d).  This call site has no session tier: session
        /model overrides are applied later by
        ``_apply_session_model_override`` on the resolved runtime.
        """
        from hermes_cli.model_switch import resolve_effective_model

        override = None
        config = getattr(self, "config", None)
        if config:
            override = _get_channel_override(
                config,
                platform,
                chat_id,
                thread_id=thread_id,
                parent_id=parent_id,
            )
        return resolve_effective_model(
            None,  # session tier applied downstream (_apply_session_model_override)
            override,
            _resolve_gateway_model(user_config),
        )

    def _get_system_prompt_for_channel(
        self,
        platform: Platform,
        chat_id: str,
        *,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Ephemeral system prompt for this channel/thread.

        Uses ``channel_overrides`` when set, else the global gateway prompt.
        Legacy ``channel_prompts`` are applied separately via ``event.channel_prompt``
        in ``run_sync`` (adapter ``resolve_channel_prompt``), so they are not
        duplicated here.
        """
        config = getattr(self, "config", None)
        if config:
            override = _get_channel_override(
                config,
                platform,
                chat_id,
                thread_id=thread_id,
                parent_id=parent_id,
            )
            if override and override.system_prompt:
                return (override.system_prompt or "").strip()
        return getattr(self, "_ephemeral_system_prompt", None) or ""

    @staticmethod
    def _load_reasoning_config(model: str = "") -> dict | None:
        """Load reasoning effort from config.yaml, respecting per-model overrides.

        Thin wrapper over the shared chokepoint
        :func:`hermes_constants.resolve_reasoning_config` (per-model override >
        global ``agent.reasoning_effort``; YAML boolean False = disabled).
        Closes #21256.

        Args:
            model: The effective model for the calling session. When empty,
                   the config's ``model.default`` is used.
        """
        from hermes_constants import resolve_reasoning_config
        cfg = _load_gateway_runtime_config()
        return resolve_reasoning_config(cfg, model)

    @staticmethod
    def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
        """Parse `/reasoning` args into `(value, persist_global)`.

        `/reasoning <level>` is session-scoped by default. `--global` may be
        supplied in any position to persist the change to config.yaml.
        """
        import shlex

        text = str(raw_args or "").strip().replace("—", "--")
        if not text:
            return "", False
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        persist_global = False
        value_tokens = []
        for token in tokens:
            if token == "--global":
                persist_global = True
            else:
                value_tokens.append(token)
        return " ".join(value_tokens).strip().lower(), persist_global

    def _resolve_session_reasoning_config(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        model: str = "",
    ) -> dict | None:
        """Resolve reasoning effort for a session, honoring session overrides.

        Priority: session-scoped ``/reasoning --session`` override >
        per-model override (``agent.reasoning_overrides``) > global
        ``agent.reasoning_effort``. ``model`` should be the session's
        *effective* model (session ``/model`` override included) so
        per-model overrides track what the session actually runs — when
        empty, the config's ``model.default`` is used.
        """
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        if resolved_session_key:
            _r_state = self.sessions.peek(resolved_session_key)
            if _r_state is not None and _r_state.conversation.reasoning_override is not None:
                return _r_state.conversation.reasoning_override
        return self._load_reasoning_config(model)

    def _set_session_reasoning_override(
        self,
        session_key: str,
        reasoning_config: Optional[dict],
    ) -> None:
        """Set or clear the session-scoped reasoning override."""
        if not session_key:
            return
        # Per-session field write — the old lazy ``self._session_reasoning_overrides
        # = {}`` init replaced the WHOLE dict, racing concurrent sessions'
        # overrides; a SessionState field reset cannot cross sessions.
        self.sessions.state(session_key).conversation.reasoning_override = (
            None if reasoning_config is None else dict(reasoning_config)
        )

    def _resolve_session_service_tier(
        self,
        source=None,
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the effective service tier for a session.

        A session-scoped /fast override wins over the config default. The
        override dict stores "priority" or None (explicit normal), so key
        presence — not value truthiness — decides whether it applies.
        """
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        if resolved_session_key:
            _t_state = self.sessions.peek(resolved_session_key)
            if (
                _t_state is not None
                and _t_state.conversation.service_tier_override
                is not _SERVICE_TIER_UNSET
            ):
                return _t_state.conversation.service_tier_override
        return self._load_service_tier()

    def _set_session_service_tier_override(
        self,
        session_key: str,
        service_tier,
        clear: bool = False,
    ) -> None:
        """Set or clear the session-scoped /fast override.

        ``service_tier`` is "priority" or None (explicit normal). Pass
        ``clear=True`` to remove the override entirely (fall back to config).
        """
        if not session_key:
            return
        # Presence-sensitive: "priority" or None (explicit normal) both count
        # as an override; the sentinel means "no override".  Old code
        # wholesale-replaced the dict on lazy init (cross-session race) —
        # per-session field writes eliminate that class of bug.
        self.sessions.state(session_key).conversation.service_tier_override = (
            _SERVICE_TIER_UNSET if clear else service_tier
        )

    @staticmethod
    def _load_service_tier() -> str | None:
        """Load Priority Processing setting from config.yaml.

        Reads agent.service_tier from config.yaml. Accepted values mirror the CLI:
        "fast"/"priority"/"on" => "priority", while "normal"/"off" disables it.
        Returns None when unset or unsupported.
        """
        cfg = _load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()

        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        cfg = _load_gateway_runtime_config()
        return is_truthy_value(
            cfg_get(cfg, "display", "show_reasoning"),
            default=False,
        )

    @staticmethod
    def _load_busy_input_mode() -> str:
        """Load gateway drain-time busy-input behavior from config/env."""
        mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
        if not mode:
            cfg = _load_gateway_runtime_config()
            mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
        if mode == "queue":
            return "queue"
        if mode == "steer":
            return "steer"
        return "interrupt"

    @staticmethod
    def _busy_mode_from_config(
        config: dict,
        *,
        fallback: str,
    ) -> str:
        """Resolve one profile's busy mode without consulting process env."""
        raw_input = str(
            cfg_get(config, "display", "busy_input_mode", default="") or ""
        ).strip().lower()
        return (
            raw_input
            if raw_input in {"interrupt", "queue", "steer"}
            else fallback
        )

    def _effective_busy_input_mode(self, source: SessionSource) -> str:
        """Return the process profile's busy input mode."""
        return getattr(self, "_busy_input_mode", "interrupt")

    @staticmethod
    def _load_restart_drain_timeout() -> float:
        """Load graceful gateway restart/stop drain timeout in seconds."""
        raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if not raw:
            cfg = _load_gateway_runtime_config()
            raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_restart_after_turn_timeout() -> float:
        """Load in-band restart wait-for-idle timeout in seconds (#77184)."""
        env_raw = os.getenv("HERMES_RESTART_AFTER_TURN_TIMEOUT")
        if env_raw is not None and str(env_raw).strip() != "":
            raw: object = env_raw
        else:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "agent", "restart_after_turn_timeout", default=None)
        value = parse_restart_after_turn_timeout(raw)
        # Warn only when the user supplied a non-empty value that failed to
        # parse (parser falls back to the default). ``0`` is valid.
        if raw is not None and str(raw).strip() != "":
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_after_turn_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_cron_drain_timeout() -> float:
        """Load the cron-only floor under the stop()/drain wait (#82161)."""
        env_raw = os.getenv("HERMES_CRON_DRAIN_TIMEOUT")
        if env_raw is not None and str(env_raw).strip() != "":
            raw: object = env_raw
        else:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "agent", "cron_drain_timeout", default=None)
        value = parse_cron_drain_timeout(raw)
        # Warn only when the user supplied a non-empty value that failed to
        # parse (parser falls back to the default). ``0`` is valid.
        if raw is not None and str(raw).strip() != "":
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid cron_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``concise`` — one-line status message on completion (default);
            failures append a short output tail
          - ``all``    — running-output updates *and* the final raw-output message
          - ``result`` — only the final raw-output completion message
          - ``error``  — only the final raw-output message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "display", "background_process_notifications")
            if raw is False:
                mode = "off"
            elif raw not in {None, ""}:
                mode = str(raw)
        mode = (mode or "concise").strip().lower()
        valid = {"concise", "all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'concise'",
                mode,
            )
            return "concise"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            # Canonical gateway loader (fail-open): managed overlay + ${VAR}
            # expansion now apply to provider_routing too.
            cfg = _load_gateway_runtime_config()
            return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_providers() -> list | None:
        """Load fallback provider chain from config.yaml.

        Returns the ordered ``fallback_providers`` chain.
        """
        try:
            # Canonical gateway loader (fail-open): managed overlay + ${VAR}
            # expansion now apply to the fallback chain too.
            cfg = _load_gateway_runtime_config()
            fb = get_fallback_chain(cfg)
            if fb:
                return fb
        except Exception:
            pass
        return None

    def _refresh_fallback_providers(self) -> list | None:
        """Re-read fallback_providers from disk for the next agent create/reuse.

        Cron already does this per job via ``get_fallback_chain``; the gateway
        previously froze ``self._fallback_providers`` at process start, so a chain
        configured (or changed) after ``hermes gateway`` was running never
        reached messaging sessions even though the same process's cron jobs
        fell back correctly. Fixes #60955.

        A TRANSIENT read/parse failure (user mid-edit of config.yaml with a
        non-atomic write) keeps the last known-good chain instead of wiping a
        cached agent's working fallback for that turn.  Only a successful read
        that genuinely lacks the key clears the chain.
        """
        try:
            from hermes_cli.config import read_user_config_raw
            cfg_path = _hermes_home / "config.yaml"
            if not cfg_path.exists():
                self._fallback_providers = None
                return self._fallback_providers
            # Raw primitive (raises on parse failure) is required here: the
            # canonical fail-open loader would return {} on a torn mid-edit
            # write and WIPE the last known-good chain. The overlay/expansion
            # below fixes the managed-scope/${VAR} drift without losing that.
            cfg = read_user_config_raw(cfg_path)
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            try:
                from hermes_cli.config import _expand_env_vars
                expanded = _expand_env_vars(cfg)
                if isinstance(expanded, dict):
                    cfg = expanded
            except Exception:
                pass
        except Exception:
            # Transient failure — keep last known-good chain.
            logger.debug(
                "fallback_providers refresh: config.yaml read failed; "
                "keeping last known-good chain", exc_info=True,
            )
            return self._fallback_providers
        self._fallback_providers = get_fallback_chain(cfg) or None
        return self._fallback_providers

    @staticmethod
    def _apply_fallback_chain_to_agent(agent: Any, chain: list | None) -> None:
        """Keep a cached agent's fallback chain aligned with current config.

        Skips rewrite while a cooldown is holding the agent on an already-
        activated fallback provider — ``restore_primary_runtime`` owns that
        turn-scoped lifecycle. When primary is active (or cooldown expired),
        replace the chain so mid-uptime ``fallback_providers`` edits take
        effect without requiring a gateway restart (#60955).
        """
        if agent is None:
            return
        new_chain = list(chain or [])
        rate_limited_until = getattr(agent, "_rate_limited_until", 0) or 0
        if (
            getattr(agent, "_fallback_activated", False)
            and rate_limited_until > time.monotonic()
        ):
            return
        old_chain = list(getattr(agent, "_fallback_chain", []) or [])
        agent._fallback_chain = new_chain
        if not getattr(agent, "_fallback_activated", False):
            agent._fallback_index = 0
        # A config edit signals the user changed something — drop the
        # session-scoped unavailability memo so re-configured entries
        # (e.g. credentials added mid-uptime for a previously-failing
        # provider) get retried instead of staying suppressed for the
        # cached agent's lifetime.  Only on actual content change, so
        # the per-message no-op refresh keeps the memo's rate-limiting
        # benefit (#60955).
        if new_chain != old_chain:
            unavailable = getattr(agent, "_unavailable_fallback_keys", None)
            if unavailable:
                unavailable.clear()

    def _snapshot_active_agents(self) -> Dict[str, Any]:
        return {
            session_key: agent
            for session_key, agent in self.sessions.running_items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    def _get_max_concurrent_sessions(self) -> Optional[int]:
        """Return the configured active chat session cap, if enabled."""
        try:
            from hermes_cli.active_sessions import resolve_max_concurrent_sessions

            return resolve_max_concurrent_sessions(getattr(self, "config", None))
        except Exception:
            return None

    def _active_session_limit_message(self, session_key: str) -> Optional[str]:
        """Return a user-facing rejection when starting a new session exceeds the cap."""
        max_sessions = self._get_max_concurrent_sessions()
        if max_sessions is None:
            return None
        if self.sessions.is_running(session_key):
            return None
        active_count = self.sessions.running_count()
        if active_count < max_sessions:
            return None
        from hermes_cli.active_sessions import active_session_limit_message

        return active_session_limit_message(active_count, max_sessions)

    def _claim_active_session_slot(
        self,
        session_key: str,
        source: SessionSource,
    ) -> tuple[Any, Optional[str]]:
        """Claim a cross-process active-session slot for a new gateway turn."""
        if self.sessions.is_running(session_key):
            return None, None
        local_limit_message = self._active_session_limit_message(session_key)
        if local_limit_message is not None:
            return None, local_limit_message
        try:
            from hermes_cli.active_sessions import try_acquire_active_session

            platform = source.platform.value if source and source.platform else "gateway"
            return try_acquire_active_session(
                session_id=session_key,
                surface=f"gateway:{platform}",
                config=getattr(self, "config", None),
                metadata={
                    "platform": platform,
                    "chat_id": getattr(source, "chat_id", "") or "",
                    "user_id": getattr(source, "user_id", "") or "",
                },
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return None, None

    @staticmethod
    def _agent_has_active_subagents(running_agent: Any) -> bool:
        """Return True when *running_agent* is currently driving subagents
        via the ``delegate_task`` tool.

        Background (#30170): ``create_agent.interrupt()`` cascades through the
        parent's ``_active_children`` list and calls ``interrupt()`` on
        every child synchronously, which aborts in-flight subagent work
        and produces a fallback cascade with no actionable signal.
        Demoting ``busy_input_mode='interrupt'`` to ``queue`` semantics
        whenever this helper returns True protects subagent work from
        conversational follow-ups while leaving the explicit ``/stop``
        path (which goes through ``_interrupt_and_clear_session``)
        untouched. Safe-by-default: returns False on any attribute or
        lock error so a missing/broken parent never blocks the existing
        interrupt path.
        """
        if running_agent is None or running_agent is _AGENT_PENDING_SENTINEL:
            return False
        children = getattr(running_agent, "_active_children", None)
        # create_agent always initialises this as a concrete list (see
        # agent/agent_init.py). Reject anything that isn't a real
        # collection — this guards against ``MagicMock()._active_children``
        # auto-creating a truthy stub in tests and triggering the demotion
        # against an agent that doesn't actually have subagents.
        if not isinstance(children, (list, tuple, set)):
            return False
        if not children:
            return False
        lock = getattr(running_agent, "_active_children_lock", None)
        try:
            if lock is not None:
                with lock:
                    return bool(children)
            return bool(children)
        except Exception:
            return False

    async def _session_has_compression_in_flight(self, session_key: str) -> bool:
        """Return True when a compression lock is held for this session's id.

        Context compression is interrupt-protected (#23975) but gateway
        ``interrupt`` busy-input mode can still start a follow-up turn against
        the pre-rotation parent while compression is mid-flight, producing
        orphaned compression siblings (#56391). Callers demote interrupt to
        queue when this returns True.

        Both blocking sources — the ``session_store`` lock + JSON load, and the
        SQLite ``get_compression_lock_holder`` SELECT — are offloaded to a
        worker thread so a large state.db never freezes the event loop (#5).
        """
        session_store = getattr(self, "session_store", None)
        if not session_key or session_store is None:
            return False
        try:
            session_id = await asyncio.to_thread(
                self._lookup_session_id_under_store_lock, session_store, session_key
            )
        except (AttributeError, TypeError):
            return False
        except Exception:
            logger.warning(
                "Compression in-flight check failed while reading session %s; "
                "treating compression as active to avoid interrupting a possible "
                "parent-session rotation",
                session_key,
                exc_info=True,
            )
            return True
        if not session_id:
            return False
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return False
        raw_db = getattr(session_db, "_db", session_db)
        try:
            holder = await asyncio.to_thread(
                raw_db.get_compression_lock_holder, str(session_id)
            )
            return bool(holder)
        except (AttributeError, TypeError):
            return False
        except Exception:
            logger.warning(
                "Compression in-flight check failed while reading lock holder "
                "for session %s; treating compression as active to avoid "
                "interrupting a possible parent-session rotation",
                session_id,
                exc_info=True,
            )
            return True

    @staticmethod
    def _lookup_session_id_under_store_lock(session_store, session_key: str):
        """Sync helper run in the thread pool: read session_id under the store lock."""
        # noqa: SLF001 — intentional private access; runs off the event loop.
        with session_store._lock:  # noqa: SLF001
            session_store._ensure_loaded_locked()  # noqa: SLF001
            entry = session_store._entries.get(session_key)  # noqa: SLF001
        return getattr(entry, "session_id", None) if entry is not None else None

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return
        # #28503 — Previously this called ``merge_pending_message_event``
        # with the default ``merge_text=False``, which silently OVERWROTE
        # the single pending slot when consecutive text messages arrived
        # in ``busy_input_mode: queue``. Route through the FIFO
        # infrastructure shared with ``/queue`` so each follow-up gets
        # its own turn in arrival order. Photo bursts still merge into
        # the head slot via ``merge_pending_message_event`` (album
        # semantics); everything else appends to the overflow tail.
        pending_slot = getattr(adapter, "_pending_messages", None)
        existing = pending_slot.get(session_key) if isinstance(pending_slot, dict) else None
        security_metadata_keys = (
            "hermes_plugin_id",
            "hermes_plugin_injection",
            "gateway_session_key",
            "gateway_session_id",
            "gateway_session_strict",
        )
        same_security_context = existing is not None and (
            getattr(existing, "internal", False) == getattr(event, "internal", False)
            and getattr(existing, "allow_gateway_control", True)
            == getattr(event, "allow_gateway_control", True)
            and all(
                (getattr(existing, "metadata", None) or {}).get(key)
                == (getattr(event, "metadata", None) or {}).get(key)
                for key in security_metadata_keys
            )
        )
        if same_security_context and (
            getattr(existing, "message_type", None) == MessageType.PHOTO
            or event.message_type == MessageType.PHOTO
            or bool(getattr(existing, "media_urls", None))
            or bool(getattr(event, "media_urls", None))
        ):
            # Preserve photo-burst / media-merge semantics for the head slot.
            merge_pending_message_event(
                adapter._pending_messages,
                session_key,
                event,
                merge_text=event.message_type == MessageType.TEXT,
            )
            return

        if self.sessions.queue_depth(session_key, adapter=adapter) >= self._BUSY_QUEUE_MAX_PENDING:
            logger.warning(
                "Dropping busy-mode follow-up for session %s — pending queue at cap (%d).",
                session_key,
                self._BUSY_QUEUE_MAX_PENDING,
            )
            return

        self.sessions.enqueue_fifo(session_key, event, adapter)

    async def _prepare_busy_steer_text(self, event: MessageEvent) -> str:
        """Return steerable text for a busy follow-up, transcribing voice first.

        Fresh and queued voice messages reach the normal inbound STT pipeline,
        but successful steer messages intentionally bypass that queue. Without
        preprocessing here, a media-only voice follow-up has an empty text
        payload and steer mode silently degrades to queue mode.

        Audio file attachments remain files; only voice-message media follows
        the automatic STT contract used by ``_prepare_inbound_message_text``.
        If transcription fails, preserve any caption and let the existing
        steer fallback handle an otherwise empty event without losing it.

        Routes through ``_transcribe_and_echo_pending_voice`` — the single
        out-of-band transcription choke point shared with the interrupt
        monitor and the pending-drain path — so the STT call is made at most
        once per platform message (cached on the event) and the transcript
        echo respects the count-based ledger.  If steering later falls back
        to queue mode, the drain path reuses the cached transcript instead of
        paying for a second STT call or re-echoing the same line.
        """
        text = (event.text or "").strip()
        if not self._pending_event_audio_paths(event):
            return text

        adapter = self._adapter_for_source(event.source)
        enriched_text, successful_transcripts = await self._transcribe_and_echo_pending_voice(
            event,
            adapter,
            event.source,
            text,
            log_context="Busy-steer",
        )
        if not successful_transcripts:
            return text
        return (enriched_text or text).strip()

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        # --- Authorization gate (#17775) ---
        # The cold path (_handle_message) checks _is_user_authorized before
        # creating a session.  The busy path must enforce the same check;
        # otherwise unauthorized users in shared threads (Slack/Telegram/Discord)
        # can inject messages into an active session they don't own.
        if not self._is_user_authorized(event.source):
            logger.warning(
                "Dropping message from unauthorized user in active session: "
                "user=%s (%s), platform=%s, session=%s",
                event.source.user_id,
                event.source.user_name,
                event.source.platform.value if event.source.platform else "unknown",
                session_key,
            )
            return True  # handled (silently dropped); do not fall through

        effective_mode = self._effective_busy_input_mode(event.source)

        # --- Draining case (gateway restarting/stopping) ---
        if self._draining:
            adapter = self._adapter_for_source(event.source)
            if not adapter:
                return True

            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
            if self._queue_during_drain_enabled(effective_mode):
                self._queue_or_replace_pending_event(session_key, event)
                message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
            else:
                message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."

            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
            return True

        # --- Approval response routing (#46866) ---
        # When the agent is blocked waiting for a dangerous-command approval,
        # plain-text responses like "yes" or "approve" must be routed to the
        # approval handler instead of being steered/queued/interrupted.
        # Otherwise approval via messaging platforms never succeeds — the
        # reply is queued behind a turn that can't start until the approval
        # resolves, so the approval times out and auto-denies (a deadlock).
        #
        # Slash forms (/approve, /deny) already bypass to the runner at the
        # base-adapter guard.  This handles the bare-word forms (Signal/SMS
        # users naturally type "yes" rather than "/approve").  Gating on
        # has_blocking_approval(session_key) is the disambiguator that keeps
        # a conversational "yes" from triggering a dangerous command when no
        # approval is actually pending (design intent — see run.py "Pending
        # exec approvals are handled by /approve and /deny" note).
        #
        # We reuse the canonical /approve and /deny handlers rather than
        # re-deriving the resolution + i18n messaging: they resolve the
        # waiting thread, resume typing, AND return a localized confirmation
        # string.  The busy-handler path does not auto-send that return, so
        # we deliver it ourselves (mirroring the draining-case send above).
        try:
            from tools.approval import has_blocking_approval
            if event.allow_gateway_control and has_blocking_approval(session_key):
                _raw_text = (event.text or "").strip().lower()
                _approve_words = {"approve", "yes", "ok", "okay", "confirm", "y", "👍"}
                _deny_words = {"deny", "no", "reject", "cancel", "n", "👎"}
                _approval_handler = None
                _normalized_args = ""
                if _raw_text in _approve_words:
                    _approval_handler = self._handle_approve_command
                elif _raw_text in _deny_words:
                    _approval_handler = self._handle_deny_command
                elif _raw_text in {"always", "approve always", "always approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "always"
                elif _raw_text in {"session", "approve session", "session approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "session"
                if _approval_handler is not None:
                    # Synthesize the canonical "/approve [args]" / "/deny"
                    # command text so the slash handlers parse modifiers via
                    # event.get_command_args().  Always use a literal "/" —
                    # MessageEvent.is_command()/get_command_args() only
                    # recognize the "/" prefix, not the per-platform display
                    # prefix ("!" on Slack/Matrix).
                    _verb = "approve" if _approval_handler is self._handle_approve_command else "deny"
                    _synth = f"/{_verb}"
                    if _normalized_args:
                        _synth = f"{_synth} {_normalized_args}"
                    event.text = _synth
                    _reply = await _approval_handler(event)
                    logger.info(
                        "Approval response via plain text: session=%s verb=%s args=%r",
                        session_key, _verb, _normalized_args,
                    )
                    _adapter = self._adapter_for_source(event.source)
                    if _adapter and _reply:
                        _text, _eph_ttl = _adapter._unwrap_ephemeral(_reply)
                        if _text:
                            _anchor = self._reply_anchor_for_event(event)
                            await _adapter._send_with_retry(
                                chat_id=event.source.chat_id,
                                content=_text,
                                reply_to=_anchor,
                                metadata=self._thread_metadata_for_source(event.source, _anchor),
                            )
                    return True
        except Exception:
            logger.warning(
                "Plain-text approval routing failed for session %s; "
                "falling through to busy handling",
                session_key, exc_info=True,
            )

        # Normal busy case (agent actively running a task)
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return False  # let default path handle it

        # --- Internal synthetic events must never interrupt/steer ---
        # Async-delegation completions (delegate_task(background=true)) and
        # background-process completions (terminal notify_on_complete) re-enter
        # the originating session as internal MessageEvents. When the session
        # is busy, treating them like a user TEXT message means interrupt-mode
        # (the default busy_input_mode) aborts the active turn AND sends a "⚡
        # Interrupting current task" ack — exactly the opposite of the design
        # invariant that a completion surfaces as a NEW turn only when idle and
        # never splices into a running turn. Plugin events carry untrusted
        # payload text, so queue those through the gateway FIFO to keep their
        # security metadata separate from pending user input.
        if getattr(event, "internal", False) and not event.allow_gateway_control:
            self._queue_or_replace_pending_event(session_key, event)
            return True
        if getattr(event, "internal", False):
            return False

        _busy_state = self.sessions.peek(session_key)
        running_agent = _busy_state.turn.agent if _busy_state else None

        busy_input_mode = self._effective_busy_input_mode(event.source)
        if (
            event.message_type == MessageType.TEXT
            and busy_input_mode == "queue"
            and effective_mode != "steer"
        ):
            return False

        # Steer mode: inject mid-run via interruption.steer() instead of
        # queueing + interrupting.  If the agent isn't running yet
        # (sentinel) or lacks steer(), or the payload is empty, fall back
        # to queue semantics so nothing is lost.
        # #30170 — Subagent protection. ``create_agent.interrupt()`` cascades
        # to every entry in the parent's ``_active_children`` list and
        # aborts in-flight ``delegate_task`` work. Demote ``interrupt``
        # to ``queue`` when the parent is currently driving subagents so
        # a conversational follow-up doesn't destroy minutes of subagent
        # work. Explicit ``/stop`` and ``/new`` slash commands go through
        # ``_interrupt_and_clear_session`` and are unaffected — the
        # operator still has a way to force-cancel everything.
        demoted_for_subagents = (
            effective_mode == "interrupt"
            and self._agent_has_active_subagents(running_agent)
        )
        if demoted_for_subagents:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because the running agent has active subagents (#30170)",
                session_key,
            )
            effective_mode = "queue"
        demoted_for_compression = (
            effective_mode == "interrupt"
            and await self._session_has_compression_in_flight(session_key)
        )
        if demoted_for_compression:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because context compression is in flight (#56391)",
                session_key,
            )
            effective_mode = "queue"
        steered = False
        redirected = False
        if effective_mode == "steer":
            steer_text = await self._prepare_busy_steer_text(event)
            # A follow-up qualifies for steering when it is plain text, OR
            # when every attachment is STT-eligible voice media whose
            # transcript was just folded into steer_text — otherwise a voice
            # note in steer mode silently degrades to queue mode (#58780).
            _steer_media_urls = getattr(event, "media_urls", None) or []
            _steer_all_voice = bool(_steer_media_urls) and (
                len(self._pending_event_audio_paths(event)) == len(_steer_media_urls)
            )
            can_steer = (
                steer_text
                and (
                    (
                        event.message_type == MessageType.TEXT
                        and not event.media_urls
                        and not event.media_types
                    )
                    or _steer_all_voice
                )
                and running_agent is not None
                and running_agent is not _AGENT_PENDING_SENTINEL
            )
            if can_steer:
                try:
                    steered = bool(interruption.steer(running_agent, steer_text))
                except Exception as exc:
                    logger.warning("Gateway steer failed for session %s: %s", session_key, exc)
                    steered = False
            if not steered:
                # Fall back to queue (merge into pending messages, no interrupt)
                effective_mode = "queue"
        elif (
            effective_mode == "interrupt"
            and event.message_type == MessageType.TEXT
            and not event.media_urls
            and not event.media_types
            and running_agent is not None
            and running_agent is not _AGENT_PENDING_SENTINEL
            and getattr(running_agent, "_supports_active_turn_redirect", False) is True
        ):
            try:
                redirected = bool(interruption.redirect(running_agent, (event.text or "").strip()))
            except Exception as exc:
                logger.warning("Gateway redirect failed for session %s: %s", session_key, exc)
                redirected = False

        # Store the message so it's processed as the next turn after the
        # current run finishes (or is interrupted).  Skip this for a
        # successful steer — the text already landed inside the run and
        # must NOT also be replayed as a next-turn user message.
        #
        # Route through _queue_or_replace_pending_event (the same FIFO
        # infrastructure used by busy queue-mode and /queue) rather than a
        # raw merge_pending_message_event(merge_text=True). The raw merge
        # newline-joins consecutive TEXT follow-ups into a SINGLE pending
        # turn, destroying message boundaries — so two separate user
        # messages sent while the agent was busy (interrupt mode, or a
        # steer that fell back to queue) arrived as one mashed-together
        # turn (#43066 sub-bug 2). The FIFO path gives each text its own
        # turn in arrival order while still preserving photo-burst / album
        # merge semantics for media.
        if not steered and not redirected:
            self._queue_or_replace_pending_event(session_key, event)

        is_queue_mode = effective_mode == "queue"
        is_steer_mode = effective_mode == "steer"
        is_redirect_mode = effective_mode == "interrupt" and redirected

        # If not in queue/steer mode, interrupt the running agent immediately.
        # This aborts in-flight tool calls and causes the agent loop to exit
        # at the next check point.
        if (
            effective_mode == "interrupt"
            and not redirected
            and running_agent
            and running_agent is not _AGENT_PENDING_SENTINEL
        ):
            try:
                _interrupt_text = event.text
                _media_urls = getattr(event, "media_urls", None) or []
                if self._pending_event_audio_paths(event):
                    _interrupt_text, _ = await self._transcribe_and_echo_pending_voice(
                        event,
                        adapter,
                        event.source,
                        event.text or "",
                        log_context="Voice-busy-interrupt",
                    )
                elif not _interrupt_text and _media_urls:
                    _interrupt_text = _build_media_placeholder(event)
                interruption.interrupt(running_agent, _interrupt_text)
            except Exception:
                pass  # don't let interrupt failure block the ack

        # Check if busy ack is disabled — skip sending but still process the input.
        # Placed before debounce so we don't stamp a "last ack" timestamp that was
        # never actually delivered.
        busy_ack_enabled = os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() == "true"
        if not busy_ack_enabled:
            logger.debug("Busy ack suppressed for session %s", session_key)
            return True  # input still processed, just no ack sent

        # Debounce before consulting config-heavy display settings. Rapid
        # follow-ups should be processed but should not trigger another config
        # read just to discover that no ack will be sent.
        _BUSY_ACK_COOLDOWN = 30
        now = time.time()
        last_ack = _busy_state.turn.busy_ack_ts if _busy_state else 0
        if now - last_ack < _BUSY_ACK_COOLDOWN:
            return True  # interrupt sent (if not queue), ack already delivered recently

        from gateway.display_config import resolve_display_setting
        platform_key = _platform_config_key(event.source.platform)

        # In steer mode the user's text has already been injected into the
        # active run. Some mobile chat setups want that steering to be silent,
        # like STT transcript echo suppression: keep the behavior, drop only
        # the confirmation bubble.
        if is_steer_mode:
            steer_ack_env = os.environ.get("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED")
            if steer_ack_env is not None:
                steer_ack_enabled = steer_ack_env.strip().lower() in {"1", "true", "yes", "on"}
            else:
                steer_ack_enabled = bool(
                    resolve_display_setting(
                        _load_gateway_config(),
                        platform_key,
                        "busy_steer_ack_enabled",
                        True,
                    )
                )
            if not steer_ack_enabled:
                logger.debug("Busy steer ack suppressed for session %s", session_key)
                return True

        self.sessions.state(session_key).turn.busy_ack_ts = now

        # Build a status-rich acknowledgment. Mobile chat defaults keep this
        # terse; detailed iteration/tool state is still available in logs and
        # can be opted in per platform via display.platforms.<platform>.busy_ack_detail.
        status_parts = []
        busy_ack_detail_enabled = bool(
            resolve_display_setting(
                _load_gateway_config(),
                _platform_config_key(event.source.platform),
                "busy_ack_detail",
                True,
            )
        )

        if busy_ack_detail_enabled and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                summary = status_output.get_activity_summary(running_agent)
                iteration = summary.get("api_call_count", 0)
                max_iter = summary.get("max_iterations", 0)
                current_tool = summary.get("current_tool")
                start_ts = _busy_state.turn.started_ts if _busy_state else 0
                if start_ts:
                    elapsed_min = int((now - start_ts) / 60)
                    if elapsed_min > 0:
                        status_parts.append(f"{elapsed_min} min elapsed")
                if max_iter:
                    status_parts.append(f"iteration {iteration}/{max_iter}")
                if current_tool:
                    status_parts.append(f"running: {current_tool}")
            except Exception:
                pass

        status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
        if is_steer_mode:
            message = (
                f"⏩ Steered into current run{status_detail}. "
                f"Your message arrives after the next tool call."
            )
        elif is_redirect_mode:
            message = (
                f"↪ Redirected current run{status_detail}. "
                f"I'll adjust using your correction."
            )
        elif is_queue_mode and demoted_for_subagents:
            # #30170 — explain the demotion so the user knows their
            # follow-up didn't accidentally kill the subagent and
            # discovers `/stop` as the explicit escape hatch.
            message = (
                f"⏳ Subagent working{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode and demoted_for_compression:
            message = (
                f"⏳ Compressing context{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode:
            message = (
                f"⏳ Queued for the next turn{status_detail}. "
                f"I'll respond once the current task finishes."
            )
        else:
            message = (
                f"⚡ Interrupting current task{status_detail}. "
                f"I'll respond to your message shortly."
            )

        # First-touch onboarding: the very first time a user sends a message
        # while the agent is busy, append a one-time hint explaining the
        # queue/interrupt knob.  Flag is persisted to config.yaml so it never
        # fires again on this install.
        try:
            from agent.onboarding import (
                BUSY_INPUT_FLAG,
                busy_input_hint_gateway,
                is_seen,
                mark_seen,
            )
            _user_cfg = _load_gateway_config()
            if not is_seen(_user_cfg, BUSY_INPUT_FLAG):
                if is_steer_mode:
                    _hint_mode = "steer"
                elif is_queue_mode:
                    _hint_mode = "queue"
                elif is_redirect_mode:
                    _hint_mode = "redirect"
                else:
                    _hint_mode = "interrupt"
                message = (
                    f"{message}\n\n"
                    f"{busy_input_hint_gateway(_hint_mode)}"
                )
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
        except Exception as _onb_err:
            logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)

        reply_anchor = self._reply_anchor_for_event(event)
        thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
        try:
            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
        except Exception as e:
            logger.debug("Failed to send busy-ack: %s", e)

        return True

    async def _drain_active_agents(
        self, timeout: float, cron_timeout: Optional[float] = None
    ) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_active_agents()
        last_active_count = self.sessions.running_count()
        last_cron_count = self._active_cron_job_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_cron_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self.sessions.running_count()
            cron_count = self._active_cron_job_count()
            if (
                force
                or active_count != last_active_count
                or cron_count != last_cron_count
                or (now - last_status_at) >= 1.0
            ):
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_cron_count = cron_count
                last_status_at = now

        # Cron jobs run on the scheduler's own thread pool, outside
        # ``self._running_agents`` — fold their in-flight count into the
        # same wait/timeout this method already applies to chat sessions,
        # or a cron job's tool work gets killed with zero warning the
        # instant it's the only active thing running (#60432).
        if self.sessions.running_count() == 0 and last_cron_count == 0:
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)

        # Cron work drains on its own deadline. ``timeout``
        # (``restart_drain_timeout``) defaults to 0 because interrupting a
        # chat turn is announced and resumable; a cron run killed mid-flight
        # is recorded in jobs.json as a permanent failure nobody is waiting
        # on. Sharing one budget meant the default config could report
        # ``timed_out=True`` after 0.00s with a cron job in flight and kill
        # it — the drain never even entered this loop (#82161).
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + timeout
        cron_deadline = started + (timeout if cron_timeout is None else cron_timeout)

        def _still_draining() -> bool:
            now = loop.time()
            if self.sessions.running_count() and now < deadline:
                return True
            return bool(self._active_cron_job_count()) and now < cron_deadline

        # Both budgets at 0 leave this loop unentered, which is the legacy
        # "interrupt immediately" behaviour — expressed as an expired
        # deadline rather than a special case, so the timed_out value below
        # is always computed from real state instead of asserted up front.
        while _still_draining():
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = (
            bool(self.sessions.running_count())
            or bool(self._active_cron_job_count())
        )
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_active_agents(self, reason: str) -> None:
        for session_key, agent in self.sessions.running_items():
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                interruption.hard_interrupt(agent, reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)

    async def _notify_interrupted_cron_jobs(self, job_ids) -> int:
        """Tell the owner of each just-interrupted cron job that its run died.

        The cron worker cannot do this itself. Its thread reaches
        ``_deliver_result`` asynchronously, and by then
        ``_bounded_adapter_teardown`` has closed the transport — so the notice
        never leaves the process, and ``_consume_interrupted_flag`` discards
        the resulting ``delivery_error`` along with it. The run's only trace is
        a line in jobs.json nobody reads (#82232).

        Must therefore be called from the post-interrupt phase, while adapters
        are still connected — the same window
        ``_notify_active_sessions_of_shutdown`` relies on for chat sessions,
        which is blind to cron work because cron runs on the scheduler's own
        thread pool rather than ``self._running_agents`` (#60432).

        Best-effort by construction: every failure is swallowed so a wedged
        adapter can never extend shutdown. Returns the number of notices sent.
        """
        if not job_ids:
            return 0
        try:
            from cron.jobs import get_job
            from cron.scheduler import _resolve_delivery_targets
        except Exception as e:
            logger.debug("Cron interrupt notification unavailable: %s", e)
            return 0

        action = "restarting" if self._restart_requested else "shutting down"
        notified: set = set()
        for job_id in job_ids:
            try:
                job = get_job(job_id)
                if not job:
                    continue
                # deliver=local jobs — and deliver=origin jobs with no
                # resolvable origin (#43014) — resolve to zero targets and
                # must stay silent rather than fall back to a home channel.
                targets = _resolve_delivery_targets(job)
            except Exception as e:
                logger.debug("Cron interrupt targets unresolved for %s: %s", job_id, e)
                continue
            if not targets:
                continue

            msg = (
                f"⚠️ Cron job '{job.get('name') or job_id}' was interrupted — "
                f"the gateway is {action} and killed the run before it "
                "finished. No result was produced for this run."
            )
            for target in targets:
                try:
                    platform = Platform(str(target.get("platform", "")).lower())
                except Exception:
                    continue
                adapter = self.adapters.get(platform)
                if adapter is None:
                    continue
                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    continue

                chat_id = str(target.get("chat_id"))
                thread_id = target.get("thread_id")
                dedup_key = (
                    job_id,
                    platform.value,
                    chat_id,
                    str(thread_id) if thread_id else None,
                )
                if dedup_key in notified:
                    continue
                try:
                    metadata = self._thread_metadata_for_target(
                        platform, chat_id, thread_id, adapter=adapter
                    )
                    result = await adapter.send(chat_id, msg, metadata=metadata)
                    if result is not None and getattr(result, "success", True) is False:
                        logger.debug(
                            "Cron interrupt notice to %s:%s failed: %s",
                            platform.value, chat_id,
                            getattr(result, "error", "send returned success=False"),
                        )
                        continue
                    notified.add(dedup_key)
                except Exception as e:
                    logger.debug(
                        "Cron interrupt notice to %s:%s raised: %s",
                        platform.value, chat_id, e,
                    )
        if notified:
            logger.info(
                "Shutdown: delivered %d interrupted-cron-job notice(s)",
                len(notified),
            )
        return len(notified)

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the very start of stop() — adapters are still connected so
        messages can be delivered. Best-effort: individual send failures are
        logged and swallowed so they never block the shutdown sequence.
        """
        active = self._snapshot_active_agents()
        restart_source = self._restart_command_source if self._restart_requested else None

        action = "restarting" if self._restart_requested else "shutting down"
        hint = (
            "Your current task will be interrupted. "
            "Send any message after restart and I'll try to resume where you left off."
            if self._restart_requested
            else "Your current task will be interrupted."
        )
        msg = f"⚠️ Gateway {action} — {hint}"

        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in active:
            source = None
            try:
                if getattr(self, "session_store", None) is not None:
                    await self.async_session_store._ensure_loaded()
                    entry = self.session_store._entries.get(session_key)
                    source = getattr(entry, "origin", None) if entry else None
            except Exception as e:
                logger.debug(
                    "Failed to load session origin for shutdown notification %s: %s",
                    session_key,
                    e,
                )

            if source is None:
                source = self._get_cached_session_source(session_key)

            if source is not None:
                platform_str = source.platform.value
                chat_id = str(source.chat_id)
                thread_id = source.thread_id
            else:
                # Fall back to parsing the session key when no persisted
                # origin is available (legacy sessions/tests).
                _parsed = _parse_session_key(session_key)
                if not _parsed:
                    continue
                platform_str = _parsed["platform"]
                chat_id = _parsed["chat_id"]
                thread_id = _parsed.get("thread_id")

            # Deduplicate only identical delivery targets. Thread/topic-aware
            # platforms can share a parent chat while still routing to distinct
            # destinations via metadata.
            dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
            if dedup_key in notified:
                continue

            try:
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue

                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    logger.info(
                        "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                        platform_str,
                    )
                    continue

                reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
                if reply_to_message_id is None and restart_source is not None:
                    try:
                        restart_platform = restart_source.platform.value
                        restart_chat_id = str(restart_source.chat_id)
                        restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                        if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                            reply_to_message_id = getattr(restart_source, "message_id", None)
                    except Exception:
                        pass

                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=getattr(source, "chat_type", None) if source is not None else None,
                    reply_to_message_id=reply_to_message_id,
                    adapter=adapter,
                )

                result = await adapter.send(chat_id, msg, metadata=metadata)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to %s:%s: %s",
                        platform_str,
                        chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to active chat %s:%s",
                    platform_str, chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str, chat_id, e,
                )

        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return

        # Snapshot adapters up front: adapter.send() can hit a fatal error
        # path that pops the adapter from self.adapters (see _handle_fatal
        # elsewhere), which would otherwise trigger
        # ``RuntimeError: dictionary changed size during iteration`` —
        # observed in a user report during gateway shutdown.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if dedup_key in notified:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
                else:
                    result = await adapter.send(str(home.chat_id), msg)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to home channel %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to home channel %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    e,
                )

    async def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            # Persist any in-flight transcript to the SQLite session store
            # before teardown (#13121).  An agent forcibly interrupted by the
            # drain-timeout escalation may never reach
            # ``turn_finalizer.finalize_turn`` (the only place that flushes the
            # turn to state.db) — e.g. it was blocked in a tool call that did
            # not abort within the post-interrupt grace window.  Its in-flight
            # tool rounds live only in the in-memory ``_session_messages``
            # (refreshed per tool round in ``conversation_loop`` but never
            # written to SQLite mid-turn), so the immediate pre-restart turn is
            # silently dropped from ``load_transcript()`` on resume.  Flushing
            # here closes that gap; the resume_pending / fresh-tool-tail
            # branches in ``_handle_message_with_agent`` already expect a
            # transcript whose tail may be a pending tool result.  The flush is
            # idempotent (identity-tracked in ``_flush_messages_to_session_db``),
            # so agents that DID finish gracefully re-flush nothing.
            try:
                _flush = getattr(agent, "_flush_messages_to_session_db", None)
                _session_messages = getattr(agent, "_session_messages", None)
                if callable(_flush) and isinstance(_session_messages, list) and _session_messages:
                    # Strip private empty-response retry scaffolding from the
                    # tail first, mirroring the graceful ``_persist_session``
                    # path, so a resumed turn doesn't replay synthetic recovery
                    # nudges.
                    _strip = getattr(
                        agent, "_drop_trailing_empty_response_scaffolding", None
                    )
                    if callable(_strip):
                        try:
                            _strip(_session_messages)
                        except Exception:
                            pass
                    try:
                        _flush(_session_messages)
                    except Exception as _flush_err:
                        # The in-memory transcript could not be persisted
                        # (e.g. FTS/SQLite index corruption — #72680). A plain
                        # debug log loses the conversation permanently when the
                        # process exits. Dump the live agent history to an
                        # external JSON recovery snapshot so an operator can
                        # salvage it after repairing state.db. The flush is
                        # non-fatal; shutdown must never block on a best-effort
                        # backup.
                        logger.warning(
                            "Shutdown transcript flush failed (%s); preserving "
                            "%d in-memory message(s) to recovery snapshot",
                            _flush_err,
                            len(_session_messages),
                        )
                        from gateway.shutdown_flush import flush_agent_history_to_file
                        flush_agent_history_to_file(
                            getattr(agent, "session_id", None),
                            _session_messages,
                        )
            except Exception as _e:
                logger.debug("Shutdown transcript flush failed: %s", _e)
            # Off-loop + bounded: finalize_session fans out to plugin
            # on_session_finalize hooks that can do arbitrary synchronous
            # work (e.g. an observability plugin serializing a full-session
            # trace export). Running it inline on the event loop blocked the
            # entire shutdown sequence past systemd's TimeoutStopSec on a
            # multi-day 4.7G session — heartbeats froze and the process was
            # SIGKILLed mid-export. Same class as the memory-provider hang
            # below (#53175).
            await self._finalize_session_off_loop(
                session_id=getattr(agent, "session_id", None),
                platform="gateway",
                reason="shutdown",
            )
            # Off-loop + bounded: a wedged memory provider here used to hang
            # the whole shutdown so SIGTERM never completed (#53175).
            await self._cleanup_agent_resources_off_loop(
                agent, context="shutdown finalize"
            )

    def _should_emit_long_running_notification(
        self,
        session_key: Optional[str],
        agent: Any,
        executor_task: Optional[Any],
    ) -> bool:
        """Only emit the heartbeat while this task still owns the live run.

        Guards against a stale ``running: delegate_task`` heartbeat outliving the
        run that started it: stop once the executor finishes, the agent is gone,
        or the session key has been rebound to a different live agent (e.g. the
        user sent ``/new`` and a fresh agent took the slot mid-run, #12029).
        """
        if agent is None:
            return False
        if executor_task is not None and executor_task.done():
            return False
        if session_key:
            _hb_state = self.sessions.peek(session_key)
            if (_hb_state.turn.agent if _hb_state else None) is not agent:
                return False
        return True

    def _defer_agent_cleanup_until_future_done(
        self,
        future: asyncio.Future,
        agent: Any,
        *,
        context: str,
    ) -> None:
        """Clean up ``agent`` only after its executor future has finished.

        A timed-out executor call keeps running in its worker thread. Closing
        the agent before that thread exits can tear down clients or providers
        it is still using. Keep a strong task reference and wait for the real
        future before invoking the normal bounded, off-loop cleanup path.
        """

        async def _cleanup_when_done() -> None:
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                # Loop shutdown can cancel this waiter while the executor still
                # runs. Never turn that cancellation into premature cleanup.
                return
            except Exception as exc:
                logger.debug(
                    "Deferred agent worker%s finished with an error: %s",
                    f" ({context})" if context else "",
                    exc,
                )
            await self._cleanup_agent_resources_off_loop(agent, context=context)

        task = asyncio.create_task(_cleanup_when_done())
        tasks = getattr(self, "_deferred_agent_cleanup_tasks", None)
        if tasks is None:
            tasks = set()
            self._deferred_agent_cleanup_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _finalize_session_off_loop(
        self,
        *,
        session_id: Any,
        platform: str,
        reason: str,
        **extra: Any,
    ) -> None:
        """Run hermes_cli.lifecycle.finalize_session off the event loop, bounded.

        finalize_session() invokes plugin ``on_session_finalize`` hooks
        synchronously; a hook doing heavy blocking work (observability trace
        export, network flush) on the event loop freezes heartbeats, adapters,
        and the shutdown drain itself. Off-loop + ``wait_for`` keeps the loop
        live; on timeout the worker thread is left to finish (or leak) on its
        own and the caller proceeds — mirroring
        ``_cleanup_agent_resources_off_loop`` (#53175).
        """

        def _call() -> None:
            from hermes_cli.lifecycle import finalize_session

            finalize_session(
                session_id=session_id,
                platform=platform,
                reason=reason,
                **extra,
            )

        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(_call),
                timeout=self._FINALIZE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Session finalize hooks (%s, reason=%s) exceeded %ss; "
                "proceeding without blocking the event loop (the worker "
                "thread is left to finish on its own).",
                session_id,
                reason,
                self._FINALIZE_TIMEOUT_S,
            )
        except Exception as finalize_exc:
            logger.debug(
                "Session finalize hooks (%s, reason=%s) failed: %s",
                session_id,
                reason,
                finalize_exc,
            )

    async def _cleanup_agent_resources_off_loop(
        self, agent: Any, *, context: str = ""
    ) -> None:
        """Run _cleanup_agent_resources in a worker thread with a bounded wait.

        Safe to await from coroutines on the gateway event loop: a slow or
        wedged teardown (memory provider IO, subprocess close) can no longer
        block message processing. On timeout the await is cancelled and the
        worker thread is left to finish (or leak) on its own — the caller
        proceeds regardless, exactly as the /new reset path does (#35994).
        """
        if agent is None:
            return
        if context.startswith("shutdown") or context == "session expiry":
            try:
                agent._end_session_on_close = False
            except Exception:
                pass
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(
                    self._cleanup_agent_resources, agent
                ),
                timeout=self._CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup%s exceeded %ss; proceeding without "
                "blocking the event loop (the worker thread is left to finish "
                "on its own). (#53175)",
                f" ({context})" if context else "",
                self._CLEANUP_TIMEOUT_S,
            )
        except Exception as cleanup_exc:
            logger.warning(
                "Agent resource cleanup%s failed: %s (#53175)",
                f" ({context})" if context else "",
                cleanup_exc,
            )

    def _cleanup_agent_resources(self, agent: Any) -> None:
        """Best-effort cleanup for temporary or cached agent instances."""
        import agent.lifecycle as lifecycle
        if agent is None:
            return
        try:
            # Drain queued memory writes BEFORE tearing the provider down.
            # The memory manager persists per-turn sync and end-of-session
            # extraction on a single serialized background worker.
            # shutdown_memory_provider() -> shutdown_all() only gives that
            # worker a ~5s bounded drain and abandons (cancels) anything
            # still queued past it, so a /reset — or any gateway session
            # rotation that reaches this cleanup path — could silently drop
            # writes the session had already handed off. The next session
            # then loads stale memory (#73297). Give pending work a bounded
            # head start through the manager's own barrier first, mirroring
            # the CLI exit path (cli.py). Best-effort: a flush failure must
            # never block teardown.
            _mm = getattr(agent, "_memory_manager", None)
            if _mm is not None and hasattr(_mm, "flush_pending"):
                try:
                    _mm.flush_pending(timeout=10)
                except Exception:
                    pass
            # Pass the agent's own conversation transcript so memory
            # providers' ``on_session_end`` hooks see the real messages
            # instead of the empty default (#15165). ``_session_messages``
            # is set on ``create_agent`` (run_agent.py:1518) and refreshed at
            # the end of every ``run_conversation`` turn via
            # ``_persist_session``; on a partially initialized agent the
            # attribute may be absent, so preserve the no-argument fallback.
            session_messages = getattr(agent, "_session_messages", None)
            if isinstance(session_messages, list):
                lifecycle.shutdown_memory_provider(agent, session_messages)
            else:
                lifecycle.shutdown_memory_provider(agent)
        except Exception:
            pass
        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) to prevent zombie
        # process accumulation.
        try:
            lifecycle.close(agent)
        except Exception:
            pass
        # Auxiliary async clients (session_search/web/vision/etc.) live in a
        # process-global cache and are created inside worker threads. Clean up
        # any entries whose event loop is now dead so their httpx transports do
        # not accumulate across gateway turns.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

    def _increment_restart_failure_counts(self, active_session_keys: set) -> None:
        """Increment restart-failure counters for sessions active at shutdown.

        Persists to a JSON file so counters survive across restarts.
        Sessions NOT in active_session_keys are removed (they completed
        successfully, so the loop is broken).
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        try:
            counts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            counts = {}

        # Increment active sessions, remove inactive ones (loop broken)
        new_counts = {}
        for key in active_session_keys:
            new_counts[key] = counts.get(key, 0) + 1
        # Keep any entries that are still above 0 even if not active now
        # (they might become active again next restart)

        try:
            atomic_json_write(path, new_counts, indent=None)
        except Exception:
            pass

    def _suspend_stuck_loop_sessions(self) -> int:
        """Suspend sessions that have been active across too many restarts.

        Returns the number of sessions suspended.  Called on gateway startup
        AFTER suspend_recently_active() to catch the stuck-loop pattern:
        session loads → agent gets stuck → gateway restarts → repeat.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return 0

        try:
            counts = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        suspended = 0
        stuck_keys = [k for k, v in counts.items() if v >= self._STUCK_LOOP_THRESHOLD]

        for session_key in stuck_keys:
            try:
                entry = self.session_store._entries.get(session_key)
                if entry and not entry.suspended:
                    entry.suspended = True
                    suspended += 1
                    logger.warning(
                        "Auto-suspended stuck session %s (active across %d "
                        "consecutive restarts — likely a stuck loop)",
                        session_key, counts[session_key],
                    )
            except Exception:
                pass

        if suspended:
            try:
                self.session_store._save()
            except Exception:
                pass

        # Clear the file — counters start fresh after suspension
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

        return suspended

    async def _clear_restart_failure_count(self, session_key: str) -> None:
        """Clear the restart-failure counter for a session that completed OK.

        Called after a successful agent turn to signal the loop is broken.
        Offloaded to a thread because the caller (_handle_message_with_agent)
        runs on the event loop and atomic_json_write calls os.fsync.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return
        try:
            counts = json.loads(path.read_text(encoding="utf-8"))
            if session_key in counts:
                del counts[session_key]
                if counts:
                    await asyncio.to_thread(atomic_json_write, path, counts, indent=None)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    async def _launch_detached_restart_command(self) -> None:
        import shutil
        import subprocess

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return
        if self._detached_restart_helper_started:
            return
        self._detached_restart_helper_started = True

        current_pid = os.getpid()
        restart_after_s = max(float(getattr(self, "_restart_drain_timeout", 0.0) or 0.0) + 5.0, 5.0)

        # On Windows there's no bash/setsid chain — spawn a tiny Python
        # watcher directly via sys.executable instead.  The watcher polls
        # current_pid, waits for our exit, then runs `hermes gateway
        # restart` with detach flags so the respawn survives the CLI
        # that triggered the /restart command closing its console.
        if sys.platform == "win32":
            import textwrap
            from hermes_cli._subprocess_compat import (
                windows_detach_flags_without_breakaway,
                windows_detach_popen_kwargs,
            )

            cmd_argv = [*hermes_cmd, "gateway", "restart"]
            watcher = textwrap.dedent(
                """
                import os, subprocess, sys, time
                from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
                pid = int(sys.argv[1])
                restart_after_s = float(sys.argv[2])
                cmd = sys.argv[3:]
                deadline = time.monotonic() + restart_after_s

                def _alive(p):
                    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
                    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
                    # Win32 handle-based existence check instead.
                    if os.name == 'nt':
                        import ctypes
                        k32 = ctypes.windll.kernel32
                        k32.OpenProcess.restype = ctypes.c_void_p
                        k32.WaitForSingleObject.restype = ctypes.c_uint
                        k32.GetLastError.restype = ctypes.c_uint
                        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
                        if not h:
                            return k32.GetLastError() != 87
                        try:
                            return k32.WaitForSingleObject(h, 0) == 0x102
                        finally:
                            k32.CloseHandle(h)
                    try:
                        os.kill(int(p), 0)
                        return True
                    except ProcessLookupError:
                        return False
                    except PermissionError:
                        return True
                    except OSError:
                        return False

                while time.monotonic() < deadline:
                    if not _alive(pid):
                        break
                    time.sleep(0.2)
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=windows_detach_flags_without_breakaway(),
                )
                """
            ).strip()
            from tools.environments.local import build_subprocess_env
            watcher_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
            # This watcher is intentionally outside the running gateway. If it
            # inherits the gateway marker, `hermes gateway restart` refuses to
            # run as a self-restart loop guard and the gateway stays stopped.
            watcher_env.pop("_HERMES_GATEWAY", None)
            project_root = Path(__file__).resolve().parent.parent
            # The watcher runs sys.executable (console python) under the
            # CREATE_NO_WINDOW detach kwargs below: it owns one hidden
            # console, inherited by the `hermes gateway restart` child, so
            # nothing flashes. Do NOT swap in GUI-subsystem pythonw.exe —
            # a console-less watcher forces every console-subsystem
            # descendant to allocate a visible conhost (#54220/#56747).
            watcher_python = sys.executable
            venv_dir = Path(watcher_env.get("VIRTUAL_ENV") or project_root / "venv")
            site_packages = venv_dir / "Lib" / "site-packages"
            if site_packages.exists():
                watcher_env["VIRTUAL_ENV"] = str(venv_dir)
                pythonpath = [str(project_root), str(site_packages)]
                if watcher_env.get("PYTHONPATH"):
                    pythonpath.append(watcher_env["PYTHONPATH"])
                watcher_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
            watcher_argv = [
                watcher_python,
                "-c",
                watcher,
                str(current_pid),
                str(restart_after_s),
                *cmd_argv,
            ]
            # The watcher process must itself break away from any job object the
            # parent CLI lives in (Windows Terminal, scheduled tasks, or another
            # process supervisor); otherwise it is reaped when the CLI
            # exits and the gateway never respawns.  windows_detach_popen_kwargs()
            # carries CREATE_BREAKAWAY_FROM_JOB, but a restrictive job object
            # (no JOB_OBJECT_LIMIT_BREAKAWAY_OK) rejects that bit with
            # ERROR_ACCESS_DENIED, surfaced as OSError.  Retry once without the
            # breakaway bit, preserving argv and the scrubbed watcher_env.
            # Mirrors the canonical fallback in
            # hermes_cli/gateway_windows.py::_spawn_detached.
            try:
                subprocess.Popen(
                    watcher_argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=watcher_env,
                    **windows_detach_popen_kwargs(),
                )
            except OSError:
                try:
                    subprocess.Popen(
                        watcher_argv,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=watcher_env,
                        creationflags=windows_detach_flags_without_breakaway(),
                    )
                except OSError as exc:
                    # Both spawn attempts failed (a breakaway-denying job object
                    # is the common cause, but OSError covers others too).
                    # Record a minimal, path-safe diagnostic and return without
                    # crashing the caller: state plainly that no watcher was
                    # started, and log only the interpreter basename and a
                    # numeric error code — never argv, env, watcher source, or
                    # str(exc) (which can carry a full interpreter path for a
                    # FileNotFoundError).
                    winerror = getattr(exc, "winerror", None)
                    error_code = winerror if winerror is not None else exc.errno
                    error_field = "winerror" if winerror is not None else "errno"
                    logger.warning(
                        "Detached restart watcher was not started after the "
                        "no-breakaway retry (%s; %s=%r). The gateway will not "
                        "be respawned by this restart attempt.",
                        os.path.basename(watcher_python),
                        error_field,
                        error_code,
                    )
            return

        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"deadline=$(( $(date +%s) + {int(restart_after_s)} )); "
            f"while kill -0 {current_pid} 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        # Same marker scrub as the Windows watcher above: this watcher runs
        # `hermes gateway restart` from outside the gateway, but it inherits
        # _HERMES_GATEWAY=1 from us, and the CLI's self-restart loop guard
        # refuses to run when that marker is set — silently (DEVNULL), so the
        # gateway stops and never comes back.
        from tools.environments.local import build_subprocess_env
        watcher_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
        watcher_env.pop("_HERMES_GATEWAY", None)
        setsid_bin = shutil.which("setsid")
        if setsid_bin:
            subprocess.Popen(
                [setsid_bin, "bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )

    def _launch_systemd_restart_shortcut(self) -> None:
        """Best-effort helper to bypass systemd's automatic restart delay.

        For planned in-chat restarts, the gateway exits cleanly so systemd does
        not record a failure.  However, units with RestartSteps still count
        automatic restarts and can delay repeated /restart tests.  A transient
        user service survives our cgroup teardown and explicitly starts the
        gateway as soon as this PID exits, while the unit keeps its normal
        backoff for real crash loops.
        """
        if sys.platform != "linux" or not os.environ.get("INVOCATION_ID"):
            return

        try:
            import shutil
            import subprocess

            systemd_run = shutil.which("systemd-run")
            systemctl = shutil.which("systemctl")
            if not systemd_run or not systemctl:
                return

            try:
                from hermes_cli.gateway import get_service_name

                service_name = get_service_name()
            except Exception:
                service_name = "hermes-gateway"

            current_pid = os.getpid()

            # Detect whether the gateway unit is registered as a system or
            # user service.  Daemon-style deployments are typically system
            # units (e.g. /etc/systemd/system/hermes-gateway.service), while
            # `hermes setup` under a non-root account may register a user
            # unit.  Hard-coding ``--user`` broke system-unit deployments:
            # systemctl returned an empty MainPID, the PID-equality check
            # below failed, and the planned-restart helper was never
            # launched — leaving the gateway dead until a manual reboot.
            def _query_pid(scope_flags):
                try:
                    out = subprocess.run(
                        [systemctl, *scope_flags, "show", service_name,
                         "--property=MainPID", "--value"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2,
                    )
                    return (out.stdout or "").strip()
                except Exception:
                    return ""

            system_pid = _query_pid([])
            user_pid = _query_pid(["--user"])
            if str(current_pid) == system_pid:
                scope_flags = []
                systemctl_scope = "systemctl"
            elif str(current_pid) == user_pid:
                scope_flags = ["--user"]
                systemctl_scope = "systemctl --user"
            else:
                # MainPID does not match in either scope — likely invoked
                # outside of systemd or the unit was renamed.  Bail out
                # rather than restart the wrong unit.
                return

            service_arg = shlex.quote(service_name)
            shell_cmd = (
                f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
                f"{systemctl_scope} reset-failed {service_arg}; "
                f"{systemctl_scope} restart {service_arg}"
            )
            unit_name = f"{service_name}-planned-restart-{current_pid}".replace(".", "-")
            subprocess.Popen(
                [
                    systemd_run,
                    *scope_flags,
                    "--collect",
                    "--unit",
                    unit_name,
                    "/bin/sh",
                    "-lc",
                    shell_cmd,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info(
                "Launched systemd planned-restart helper for %s (pid=%s, scope=%s)",
                service_name,
                current_pid,
                "user" if scope_flags else "system",
            )
        except Exception as e:
            logger.debug("Failed to launch systemd planned-restart helper: %s", e)

    def _wedged_agent_count(self) -> int:
        """Count running chat agents already past the inactivity timeout.

        A turn whose agent has recorded no activity (no API bytes, no tool
        progress) for longer than ``agent.gateway_timeout`` is wedged — the
        same threshold at which the turn reaper gives up on it. The restart
        after-turn wait must not treat such turns as work worth waiting for:
        a wedged agent pinned ``hermes update`` in "draining" for the full
        ``restart_after_turn_timeout`` cap because the drain counted it as
        active while its own inactivity watchdog had already declared it dead
        (Aug 2026, WhatsApp turn idle 30+ min, drain waited on it anyway).

        Returns 0 when the inactivity timeout is disabled (``gateway_timeout``
        0/unset ⇒ the operator opted into unbounded turns; the after-turn cap
        still bounds the wait). Cron/API-server work has no per-turn activity
        clock and is never counted as wedged. Pending sentinels are brand-new
        turns, never wedged. Fail-open per agent: an unreadable activity
        summary means "not wedged".
        """
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        if timeout <= 0:
            return 0
        wedged = 0
        for _, agent in self.sessions.running_items():
            if agent is None or agent is _AGENT_PENDING_SENTINEL:
                continue
            summary_fn = getattr(agent, "get_activity_summary", None)
            if not callable(summary_fn):
                continue
            try:
                summary = summary_fn()
                if not isinstance(summary, dict):
                    continue
                idle = float(summary.get("seconds_since_activity", 0.0))
            except Exception:
                continue
            if idle >= timeout:
                wedged += 1
        return wedged

    def _awaitable_work_count(self) -> int:
        """Active work minus wedged turns — what the restart wait waits on."""
        return max(0, self._active_work_count() - self._wedged_agent_count())

    async def _await_active_work_before_restart(self) -> bool:
        """Wait for in-flight work to finish before entering ``stop()``.

        In-band restart used to call ``stop()`` immediately, which folded the
        requesting turn into the drain wait set and force-interrupted it at
        ``restart_drain_timeout`` (#77184). Instead we refuse new turns and
        wait here for active agents/cron/api work to reach zero, then let
        ``stop()`` run against an idle gateway (drain is instant).

        Turns already past the inactivity timeout are excluded from the wait
        (``_wedged_agent_count``): restart is usually the *remedy* for a
        wedged turn, so deferring it behind one inverts the point of the
        graceful path. ``stop()``'s drain interrupts them under
        ``restart_drain_timeout`` instead.

        Returns True when work drained to zero, False when the safety cap
        elapsed with work still active — or when only wedged work remains —
        (caller proceeds to ``stop()``, which may then interrupt remaining
        runs under ``restart_drain_timeout``).
        """
        active = self._active_work_count()
        if active <= 0:
            return True

        awaitable = self._awaitable_work_count()
        if awaitable <= 0:
            logger.warning(
                "Restart requested with %d active work unit(s), all wedged "
                "past the inactivity timeout; skipping the after-turn wait "
                "and proceeding to stop()/drain which will interrupt them",
                active,
            )
            return False

        timeout = float(getattr(self, "_restart_after_turn_timeout", 0.0) or 0.0)
        if timeout <= 0:
            logger.info(
                "Restart requested with %d active work unit(s); "
                "restart_after_turn_timeout=0 — entering stop()/drain immediately",
                active,
            )
            return False

        logger.info(
            "Restart requested with %d active work unit(s); "
            "deferring stop() until they finish (cap=%.0fs) so in-flight "
            "turns are not amputated (#77184)",
            active,
            timeout,
        )
        try:
            self._update_runtime_status("draining")
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_status_at = 0.0
        while self._awaitable_work_count() > 0:
            now = loop.time()
            if now >= deadline:
                logger.warning(
                    "Restart after-turn wait timed out after %.0fs with %d "
                    "still active; proceeding to stop()/drain which may "
                    "interrupt remaining work (#77184)",
                    timeout,
                    self._active_work_count(),
                )
                return False
            if (now - last_status_at) >= 30.0:
                logger.info(
                    "Restart deferred: waiting on %d active work unit(s) "
                    "(%d wedged and excluded; %.0fs remaining before force drain)",
                    self._awaitable_work_count(),
                    self._wedged_agent_count(),
                    deadline - now,
                )
                try:
                    self._update_runtime_status("draining")
                except Exception:
                    pass
                last_status_at = now
            await asyncio.sleep(0.1)

        if self._active_work_count() > 0:
            logger.warning(
                "Restart deferred wait: %d wedged work unit(s) remain; "
                "proceeding to stop()/drain which will interrupt them",
                self._active_work_count(),
            )
            return False

        logger.info(
            "Restart deferred wait complete — active work drained; "
            "proceeding to stop()"
        )
        return True

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True
        # Refuse new turns immediately while in-flight work finishes.
        # Keep ``_running`` True so adapters stay connected and the active
        # turn can still deliver its final response (#77184).
        self._draining = True

        async def _run_restart() -> None:
            await self._await_active_work_before_restart()
            # Launch the detached helper only AFTER the after-turn wait.
            # Its deadline is drain_timeout+5 and covers stop() teardown —
            # launching earlier would fire `hermes gateway restart` while
            # the requesting turn was still running.
            if detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart helper: %s", e)
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        # _run_restart is a short-lived self-terminating task (calls stop()
        # then returns).  Don't add it to _background_tasks — _stop_impl
        # cancels all entries in that set, which would cancel _run_restart
        # while it's awaiting _stop_task, propagating CancelledError into
        # _stop_impl and preventing _shutdown_event.set() / _exit_code = 75.
        # See #12875.
        #
        # We still hold a strong reference in self._restart_task: a bare
        # asyncio.create_task() keeps only a weak reference, so the event
        # loop may garbage-collect a still-pending task mid-flight.  The
        # cancel loop in _stop_impl explicitly skips _restart_task for the
        # same reason it skips _stop_task.
        self._restart_task = asyncio.create_task(_run_restart())
        return True

    async def _run_startup_resume_event(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
    ) -> None:
        """Dispatch one synthetic startup resume and wait for its agent turn.

        ``BasePlatformAdapter.handle_message()`` returns after it installs the
        adapter-level guard and spawns the background processing task.  Startup
        restore needs a stronger boundary: inbound messages must stay queued
        until the resumed agent turn itself has finished, otherwise a user
        message can race the restore turn immediately after ``handle_message``
        returns.
        """
        try:
            await adapter.handle_message(event)
            session_tasks = getattr(adapter, "_session_tasks", {})
            task = session_tasks.get(session_key) if isinstance(session_tasks, dict) else None
            if task is not None:
                await asyncio.shield(task)
        finally:
            # _schedule_resume_pending_sessions pre-claims the runner slot
            # before spawning this task.  If adapter.handle_message raises
            # before _handle_message takes ownership, release that pre-claim;
            # otherwise the real run's normal cleanup owns the slot.
            _pre_state = self.sessions.peek(session_key)
            if (_pre_state.turn.agent if _pre_state else None) is _AGENT_PENDING_SENTINEL:
                self._release_turn_state(session_key)

    def _queue_startup_restore_event(self, event: MessageEvent) -> None:
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            queue = []
            self._startup_restore_queue = queue
        queue.append(event)
        try:
            source = event.source
            logger.info(
                "Queued inbound message during gateway startup restore: platform=%s chat=%s",
                source.platform.value if source and source.platform else "unknown",
                source.chat_id if source else "unknown",
            )
        except Exception:
            pass

    async def _drain_startup_restore_queue(self) -> int:
        """Replay inbound messages queued while startup auto-resume ran."""
        drained = 0
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            return 0
        while queue:
            event = queue.pop(0)
            source = getattr(event, "source", None)
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Dropping startup-restore queued message: adapter unavailable for %s",
                    getattr(getattr(source, "platform", None), "value", None),
                )
                continue
            # Mark this replay so _handle_message does not queue it again while
            # the restore gate remains closed for any fresh inbound arrivals.
            try:
                setattr(event, "_hermes_startup_restore_replay", True)
            except Exception:
                pass
            await adapter.handle_message(event)
            drained += 1
        return drained

    async def _finish_startup_restore(self) -> None:
        """Wait (BOUNDED) for startup auto-resume, then release + drain inbound.

        The wait is bounded by ``_startup_restore_drain_timeout_secs`` so that
        a single pathologically long boot-resume turn cannot hold the inbound
        gate shut for every channel.  On timeout we release the gate and let
        the still-running resume turn(s) finish in the background — they are
        NOT cancelled.  This is safe because duplicate-agent protection does
        not depend on the wait: ``_schedule_resume_pending_sessions`` claims
        each session's ``_running_agents`` slot SYNCHRONOUSLY before this gate
        runs, so any inbound message drained while a resume turn is still in
        flight queues behind that slot instead of spawning a second agent.
        """
        tasks = list(getattr(self, "_startup_restore_tasks", []) or [])
        if tasks:
            timeout = _startup_restore_drain_timeout_secs()
            if timeout > 0:
                # asyncio.wait (unlike wait_for / gather+timeout) does NOT
                # cancel the pending tasks on timeout — the slow resume turn
                # keeps running in the background instead of being killed.
                done, pending = await asyncio.wait(tasks, timeout=timeout)
                if pending:
                    logger.warning(
                        "Startup-restore gate released after %.0fs with %d boot "
                        "auto-resume turn(s) still running; draining inbound "
                        "queue now (resume slots already claimed, so no "
                        "duplicate agents). Slow turn(s) continue in the "
                        "background.",
                        timeout,
                        len(pending),
                    )
                    # These tasks outlive the gate.  Their normal done-callback
                    # only discards them from _background_tasks, so a LATER
                    # failure would be silently swallowed.  Attach a logging
                    # callback so a background resume turn that fails after the
                    # timeout is still recorded.
                    for task in pending:
                        task.add_done_callback(self._log_background_resume_result)
            else:
                # Non-positive timeout => opt out of the bound (historical
                # "wait forever" behaviour).
                await asyncio.gather(*tasks, return_exceptions=True)
                done = set(tasks)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.debug(
                        "startup auto-resume task failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
        self._startup_restore_tasks = []
        drained = await self._drain_startup_restore_queue()
        self._startup_restore_in_progress = False
        if drained:
            logger.info("Drained %d inbound message(s) queued during startup restore", drained)

    @staticmethod
    def _log_background_resume_result(task: "asyncio.Task") -> None:
        """Done-callback for a boot-resume turn that outlived the
        startup-restore gate.  Logs a late failure that would otherwise be
        swallowed once the task is discarded from ``_background_tasks``.
        Cancellation is expected (shutdown) and is not an error."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug(
                "background startup auto-resume task failed after gate release",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _redeliver_pending_obligations(self) -> int:
        """Redeliver final responses recorded in the delivery ledger by a
        previous (now dead) gateway process.

        Runs at startup BEFORE ``_schedule_resume_pending_sessions``. A
        session with a recoverable obligation already produced its answer —
        the turn completed and only delivery is owed — so this method sends
        the stored text and clears ``resume_pending`` for that session,
        preventing the resume path from re-running (and re-paying for) a
        turn whose output we hold.

        Crash-ambiguity contract (see gateway/delivery_ledger.py):
        rows that were mid-send or previously rejected carry a visible
        recovered-reply marker so a possible duplicate is labeled, never
        silent. Returns the number of redeliveries attempted.
        """
        try:
            from gateway.delivery_ledger import (
                RECOVERED_MARKER,
                ledger_enabled,
                mark_delivered,
                mark_failed,
                sweep_recoverable,
            )

            if not await asyncio.to_thread(ledger_enabled):
                return 0
            # Only claim rows we can actually send this boot: self.adapters
            # holds a platform only after its connect() succeeded, and each
            # claim spends one of the row's three redelivery attempts.
            _deliverable = {
                getattr(p, "value", str(p)) for p in self.adapters
            }
            claimed = await asyncio.to_thread(
                sweep_recoverable, None, deliverable_platforms=_deliverable
            )
        except Exception:
            logger.debug("delivery ledger sweep failed", exc_info=True)
            return 0
        if not claimed:
            return 0

        redelivered = 0
        for row in claimed:
            try:
                platform = Platform(row["platform"])
            except Exception:
                logger.debug(
                    "obligation %s: unknown platform %r",
                    row["obligation_id"], row.get("platform"),
                )
                continue
            adapter = self.adapters.get(platform)
            if adapter is None:
                # Platform not connected this boot — leave the row claimed;
                # attempts cap + stale cutoff bound the retries on later boots.
                continue
            content = row["content"]
            if row.get("needs_marker"):
                content = RECOVERED_MARKER + content
            metadata = (
                {"thread_id": row["thread_id"]} if row.get("thread_id") else None
            )
            try:
                result = await adapter.send(
                    chat_id=row["chat_id"],
                    content=content,
                    metadata=metadata,
                )
            except Exception as send_err:
                logger.warning(
                    "obligation %s: redelivery send raised: %s",
                    row["obligation_id"], send_err,
                )
                result = None
            try:
                if result is not None and getattr(result, "success", False):
                    await asyncio.to_thread(mark_delivered, row["obligation_id"])
                    redelivered += 1
                    logger.info(
                        "Redelivered recovered final response to %s:%s "
                        "(obligation %s, attempt %d)",
                        row["platform"], row["chat_id"],
                        row["obligation_id"], row["attempts"],
                    )
                else:
                    await asyncio.to_thread(
                        mark_failed,
                        row["obligation_id"],
                        str(getattr(result, "error", "") or "send failed"),
                    )
            except Exception:
                logger.debug("delivery ledger update failed", exc_info=True)

            # The answer reached (or was owed to) this session — don't ALSO
            # re-run the turn via the resume path.
            session_key = row.get("session_key") or ""
            if session_key:
                try:
                    await self.async_session_store.clear_resume_pending(session_key)
                except Exception:
                    logger.debug(
                        "clear_resume_pending failed for %s", session_key,
                        exc_info=True,
                    )
        return redelivered

    def _schedule_resume_pending_sessions(self, platform=None) -> int:
        """Auto-continue fresh restart-interrupted sessions after startup.

        ``resume_pending`` already preserves the transcript AND the existing
        ``_is_resume_pending`` branch in ``_handle_message_with_agent``
        injects a reason-aware recovery system note on the next turn.  This
        method closes the UX gap by synthesizing that next turn once
        adapters are back online — the event text is empty so the existing
        injection path owns the wording and we never double up.

        Adapters that are not yet ready (adapter missing from
        ``self.adapters``) are skipped silently; their sessions stay
        ``resume_pending`` and will auto-resume on the next real user
        message, or when the platform reconnects — the reconnect watcher
        calls this again scoped to that ``platform``.

        ``platform`` (a ``Platform``) restricts the pass to sessions that
        originated on that platform.  The reconnect path passes it so a
        platform coming back online retries only its own sessions and never
        re-touches another platform's in-flight recoveries.  Sessions whose
        agent is already running are skipped regardless, so a session
        scheduled at startup is never resumed a second time.
        """
        window = auto_continue_freshness_window()
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                    and (platform is None or entry.origin.platform == platform)
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return 0

        # Defense-3 (#30719): break the SIGTERM-respawn loop. Only count this
        # boot when there are restart-interrupted sessions to resume — a clean
        # boot must not accrue toward the breaker. If too many such boots have
        # happened in the configured window, skip auto-resume for THIS boot:
        # the gateway still comes up and serves real inbound messages, it just
        # stops replaying the session that keeps killing it. The session stays
        # resume_pending, so a real user message can still continue it (a human
        # is now in the loop). Defenses 1-2 cover the cron/CLI/terminal paths;
        # this catches every other SIGTERM source (e.g. a raw `terminal(
        # "launchctl kickstart ai.hermes.gateway")`).
        if candidates:
            try:
                from gateway import restart_loop_guard as _rlg

                _max_restarts, _window, _max_gap = self._restart_loop_guard_config()
                if _rlg.check_and_record(
                    _max_restarts, _window, max_gap_seconds=_max_gap
                ):
                    return 0
            except Exception as exc:  # noqa: BLE001 — breaker must fail OPEN
                logger.debug("Restart-loop guard check skipped: %s", exc)

        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue

            # Already being resumed (e.g. scheduled at startup and still
            # in-flight) — don't synthesize a second continuation turn.
            if self.sessions.is_running(entry.session_key):
                continue

            source = entry.origin
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s",
                    entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue

            # Validate the session owner against the current allowlist
            # before auto-resuming. A session created before
            # TELEGRAM_ALLOWED_USERS (or equivalent) was configured, or
            # before the owner was removed from it, must not silently
            # receive a full agent response on gateway restart just
            # because it has a resume-pending marker (issue #23778).
            try:
                if not self._is_user_authorized(source):
                    logger.warning(
                        "Skipping auto-resume for %s: session owner is no "
                        "longer authorized under the current allowlist",
                        entry.session_key,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "Skipping auto-resume for %s: authorization check failed: %s",
                    entry.session_key, exc,
                )
                continue

            # Claim the session slot *before* spawning the task so that an
            # inbound message arriving between task creation and the task's
            # first await (where _process_message_background sets the real
            # sentinel) sees the slot as occupied and queues behind it
            # instead of spinning up a duplicate create_agent (#45456).
            _resume_state = self.sessions.state(entry.session_key)
            _resume_state.turn.agent = _AGENT_PENDING_SENTINEL
            _resume_state.turn.started_ts = time.time()
            self._persist_active_agents()

            # Empty-text internal event — the _is_resume_pending branch in
            # _handle_message_with_agent prepends the proper reason-aware
            # system note before the turn runs.
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            task = asyncio.create_task(
                self._run_startup_resume_event(adapter, event, entry.session_key)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if getattr(self, "_startup_restore_in_progress", False):
                tasks = getattr(self, "_startup_restore_tasks", None)
                if tasks is None:
                    tasks = []
                    self._startup_restore_tasks = tasks
                tasks.append(task)
            scheduled += 1
        if scheduled:
            logger.info(
                "Scheduled auto-resume for %d restart-interrupted session(s)",
                scheduled,
            )
        return scheduled

    def _startup_should_abort(self) -> bool:
        return (
            self._restart_requested
            or self._draining
            or self._shutdown_event.is_set()
        )

    async def _abort_startup_if_shutdown_requested(
        self,
        adapter: Optional[BasePlatformAdapter] = None,
        platform: Optional[Platform] = None,
    ) -> bool:
        """Clean up and exit startup when restart/shutdown begins mid-startup."""
        if not self._startup_should_abort():
            return False
        if adapter is not None and platform is not None:
            try:
                await adapter.cancel_background_tasks()
            except Exception as e:
                logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
            await self._safe_adapter_disconnect(adapter, platform)
        stop_task = self._stop_task
        current_task = asyncio.current_task()
        if stop_task is not None and stop_task is not current_task:
            await stop_task
        elif not self._shutdown_event.is_set():
            await self.stop(
                restart=self._restart_requested,
                detached_restart=self._restart_detached,
                service_restart=self._restart_via_service,
            )
        return True

    def _start_loop_liveness_guards(self, loop: asyncio.AbstractEventLoop) -> None:
        """Arm the selector floor and out-of-loop watchdog before adapters.

        Disabled entirely with ``gateway.loop_watchdog: false`` in config.yaml
        (no env override — config-only knob, #69089).
        """
        config = getattr(self, "config", None)
        if config is not None and not getattr(config, "loop_watchdog", True):
            return
        if getattr(self, "_loop_floor_timer_handle", None) is None:
            try:
                self._loop_floor_timer_handle = _arm_loop_floor_timer(loop)
            except Exception:
                logger.debug("Failed to arm gateway loop floor timer", exc_info=True)

        watchdog = getattr(self, "_loop_liveness_watchdog", None)
        if watchdog is None or not watchdog.is_alive():
            try:
                self._loop_liveness_watchdog = start_loop_liveness_watchdog(loop)
            except Exception:
                logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)

    def _stop_loop_liveness_guards(self) -> None:
        """Disarm lifetime liveness guards before shutdown can load the loop."""
        watchdog = getattr(self, "_loop_liveness_watchdog", None)
        self._loop_liveness_watchdog = None
        if watchdog is not None:
            try:
                watchdog.stop()
            except Exception:
                logger.debug("Failed to stop gateway loop liveness watchdog", exc_info=True)

        floor_timer = getattr(self, "_loop_floor_timer_handle", None)
        self._loop_floor_timer_handle = None
        if floor_timer is not None:
            try:
                floor_timer.cancel()
            except Exception:
                logger.debug("Failed to cancel gateway loop floor timer", exc_info=True)

    async def _consume_clean_shutdown_marker(self, marker_path) -> int:
        """Discard orphan turn markers before consuming a clean-exit receipt.

        If either persistence or marker removal fails, startup must fail closed.
        Continuing with the old receipt would let a later unclean exit masquerade
        as clean and discard genuinely interrupted turns.
        """
        discarded = await self.async_session_store.discard_active_turn_markers()
        marker_path.unlink()
        return discarded

    async def _recover_unclean_sessions(self) -> tuple[int, int]:
        """Recover exact active turns, then run the legacy recency fallback."""
        exact = 0
        fallback = 0
        try:
            agent_timeout = max(1.0, _float_env("HERMES_AGENT_TIMEOUT", 1800))
            marker_max_age = max(60 * 60, int(agent_timeout * 2))
            exact = await self.async_session_store.recover_interrupted_turns(
                max_age_seconds=marker_max_age
            )
        except Exception as exc:
            logger.warning("Exact active-turn recovery on startup failed: %s", exc)
        try:
            fallback = await self.async_session_store.suspend_recently_active(
                max_age_seconds=120
            )
        except Exception as exc:
            logger.warning("Legacy session recovery on startup failed: %s", exc)
        return exact, fallback

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.

        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        # Enable faulthandler for stack dumps on freezes/crashes (#70344).
        # Falls back to a log file when sys.stderr is None (Windows VBS /
        # pythonw / detached service) — otherwise the gateway would die
        # here and take every adapter offline. See #71671.
        try:
            faulthandler.enable()
        except (RuntimeError, ValueError, OSError):
            try:
                _fh_log_dir = getattr(self.config, "log_dir", None) or os.path.join(
                    str(get_hermes_home()),
                    "logs",
                )
                os.makedirs(_fh_log_dir, exist_ok=True)
                _fh_enable_path = os.path.join(_fh_log_dir, "gateway_faulthandler.log")
                _fh_enable_file = open(_fh_enable_path, "a", encoding="utf-8")
                faulthandler.enable(file=_fh_enable_file, all_threads=True)
            except Exception:
                logger.debug("faulthandler.enable() unavailable", exc_info=True)
        # Also dump stacks to a rotating file for off-line analysis when
        # the gateway is running under a service manager that doesn't
        # capture stderr.
        # faulthandler.register() and SIGUSR2 are POSIX-only; skip the
        # signal-triggered file dump on Windows (faulthandler.enable()
        # above still covers fatal-error dumps there).
        _sigusr2 = getattr(signal, "SIGUSR2", None)
        if _sigusr2 is not None and hasattr(faulthandler, "register"):
            try:
                _log_dir = getattr(self.config, "log_dir", None) or os.path.join(
                    str(get_hermes_home()),
                    "logs",
                )
                _faulthandler_path = os.path.join(_log_dir, "gateway_faulthandler.log")
                os.makedirs(_log_dir, exist_ok=True)
                _fh = open(_faulthandler_path, "a", encoding="utf-8")
                faulthandler.register(
                    _sigusr2,
                    file=_fh,
                    all_threads=True,
                    chain=True,
                )
            except Exception:
                logger.debug("Could not set up faulthandler file logging", exc_info=True)

        try:
            self._gateway_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._gateway_loop = None
        if self._gateway_loop is not None:
            self._start_loop_liveness_guards(self._gateway_loop)
        logger.info("Session storage: %s", self.config.sessions_dir)

        # Sanity-check that systemd's TimeoutStopSec covers our drain
        # window.  When the user upgraded hermes-agent without re-running
        # ``hermes setup``, their unit file may still encode the old
        # default — in which case SIGKILL hits mid-drain and looks like
        # a phantom kill in the journal.  Best-effort, never raises.
        try:
            from gateway.shutdown_forensics import check_systemd_timing_alignment
            _alignment = check_systemd_timing_alignment(self._restart_drain_timeout)
            if _alignment is not None and _alignment.get("mismatch"):
                logger.warning(
                    "Stale systemd unit detected: %s has TimeoutStopSec=%.0fs but "
                    "drain_timeout=%.0fs (expected >=%.0fs). systemd may SIGKILL the "
                    "gateway mid-drain. Run `hermes gateway install --force` "
                    "to regenerate the unit, or shorten agent.restart_drain_timeout.",
                    _alignment.get("unit", "(unknown)"),
                    _alignment["timeout_stop_sec"],
                    _alignment["drain_timeout"],
                    _alignment["expected_min"],
                )
        except Exception as _e:
            logger.debug("check_systemd_timing_alignment failed: %s", _e)
        # Log the resolved max_iterations budget so operators can verify the
        # config.yaml → env bridge did the right thing at a glance (instead
        # of silently running at a stale .env value for weeks).
        try:
            _effective_max_iter = int(os.getenv("HERMES_MAX_ITERATIONS", "500"))
            logger.info(
                "Agent budget: max_iterations=%d (agent.max_turns from config.yaml, "
                "or HERMES_MAX_ITERATIONS from .env, or default 500)",
                _effective_max_iter,
            )
        except Exception:
            pass
        # Redaction status: ON by default (#17691). Surface a prominent
        # warning if an operator has explicitly opted out so they don't
        # forget the downgrade is active — the redactor snapshots its
        # state at import time, so this log line is the source of truth
        # for this process's lifetime.
        try:
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            _redact_on = _redact_raw.lower() in {"1", "true", "yes", "on"}
            if _redact_on:
                logger.info(
                    "Secret redaction: ENABLED (tool output, logs, and chat "
                    "responses are scrubbed before delivery)"
                )
            else:
                logger.warning(
                    "Secret redaction: DISABLED (HERMES_REDACT_SECRETS=%s). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set security.redact_secrets: true "
                    "in config.yaml to re-enable.",
                    _redact_raw,
                )
        except Exception:
            pass
        try:
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        except Exception:
            pass
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state="starting",
                exit_reason=None,
                clear_profile_platforms=True,
            )
        except Exception:
            pass
        try:
            from hermes_cli.config import load_config
            from agent.monitoring.gateway_health_export import start_gateway_health_export
            self._gateway_health_export_runtime = start_gateway_health_export(load_config())
            if getattr(self._gateway_health_export_runtime, "enabled", False):
                logger.info("Gateway health OTLP export: enabled")
        except Exception:
            logger.debug("gateway health OTLP export startup failed", exc_info=True)

        # Log any active supply-chain security advisories. Operators see this
        # in gateway.log and `hermes status` surfaces it; we do NOT block
        # startup or surface it inline to user messages, since the gateway
        # operator is the one who can act on it (uninstall the package,
        # rotate credentials).  See hermes_cli/security_advisories.py.
        try:
            from hermes_cli.security_advisories import (
                detect_compromised,
                gateway_log_message,
            )
            _adv_hits = detect_compromised()
            _adv_msg = gateway_log_message(_adv_hits)
            if _adv_msg:
                logger.warning("%s", _adv_msg)
                logger.warning(
                    "Run `hermes doctor` on the gateway host for full "
                    "remediation steps."
                )
        except Exception:
            logger.debug(
                "security advisory check failed at gateway startup",
                exc_info=True,
            )
        if await self._abort_startup_if_shutdown_requested():
            return True

        # Warn if no user allowlists are configured and open access is not opted in
        _builtin_allowed_vars = (
            "TELEGRAM_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_CHATS",
            "MATTERMOST_ALLOWED_USERS",
            "GATEWAY_ALLOWED_USERS",
        )
        _builtin_allow_all_vars = (
            "TELEGRAM_ALLOW_ALL_USERS",
            "MATTERMOST_ALLOW_ALL_USERS",
        )
        _any_allowlist = any(os.getenv(v) for v in _builtin_allowed_vars)
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
            os.getenv(v, "").lower() in {"true", "1", "yes"}
            for v in _builtin_allow_all_vars
        )
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No env user allowlists configured. Messaging platforms default to "
                "pairing/allowlist policies and will deny unknown senders unless you "
                "configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id) "
                "or explicitly opt in with GATEWAY_ALLOW_ALL_USERS=true plus "
                "dm_policy/group_policy: open on the platform."
            )

        # Discover Python plugins before shell hooks so plugin block
        # decisions take precedence in tie cases.  The CLI startup path
        # does this via an explicit call in hermes_cli/main.py; the
        # gateway lazily imports run_agent inside per-request handlers,
        # so the discover_plugins() side-effect in model_tools.py is NOT
        # guaranteed to have run by the time we reach this point.
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
        except Exception:
            logger.warning(
                "plugin discovery failed at gateway startup", exc_info=True,
            )

        # Register declarative shell hooks from cli-config.yaml.  Gateway
        # has no TTY, so consent has to come from one of the three opt-in
        # channels (--accept-hooks on launch, HERMES_ACCEPT_HOOKS env var,
        # or hooks_auto_accept: true in config.yaml).  We pass
        # accept_hooks=False here and let register_from_config resolve
        # the effective value from env + config itself — the CLI-side
        # registration already honored --accept-hooks, and re-reading
        # hooks_auto_accept here would just duplicate that lookup.
        # Failures are logged but must never block gateway startup.
        try:
            from hermes_cli.config import load_config
            from agent.shell_hooks import register_from_config
            _hooks_cfg = load_config()
            register_from_config(_hooks_cfg, accept_hooks=False)
        except Exception:
            logger.debug(
                "shell-hook registration failed at gateway startup",
                exc_info=True,
            )

        # Discover and load event hooks
        self.hooks.discover_and_load()


        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)

        # Recover sessions that were active when the gateway last exited.
        # Exact durable turn markers cover long-running work; the 120-second
        # recency heuristic remains as an upgrade fallback for turns started by
        # older Hermes versions that did not write exact markers.
        #
        # SKIP suspension after a clean (graceful) shutdown — the previous
        # process already drained active agents, so sessions aren't stuck.
        # This prevents unwanted auto-resets after `hermes update`,
        # `hermes gateway restart`, or `/restart`.
        _clean_marker = _hermes_home / ".clean_shutdown"
        if _clean_marker.exists():
            logger.info("Previous gateway exited cleanly — skipping session suspension")
            try:
                discarded = await self._consume_clean_shutdown_marker(_clean_marker)
            except Exception as exc:
                logger.error(
                    "Clean-start marker cleanup failed; refusing startup so the "
                    "clean-exit receipt cannot mask a later unclean exit: %s",
                    exc,
                )
                raise RuntimeError("clean-start recovery cleanup failed") from exc
            if discarded:
                logger.info(
                    "Discarded %d orphan active-turn marker(s) after clean shutdown",
                    discarded,
                )
        else:
            exact, fallback = await self._recover_unclean_sessions()
            recovered = exact + fallback
            if recovered:
                logger.info(
                    "Marked %d in-flight session(s) as resumable from previous run "
                    "(%d exact, %d legacy)",
                    recovered,
                    exact,
                    fallback,
                )

        # Stuck-loop detection (#7536): if a session has been active across
        # 3+ consecutive restarts, it's probably stuck in a loop (the same
        # history keeps causing the agent to hang).  Auto-suspend it so the
        # user gets a clean slate on the next message.
        try:
            stuck = self._suspend_stuck_loop_sessions()
            if stuck:
                logger.warning("Auto-suspended %d stuck-loop session(s)", stuck)
        except Exception as e:
            logger.debug("Stuck-loop detection failed: %s", e)

        # Serialize startup restore against inbound dispatch.  Platform
        # adapters can begin receiving messages as soon as they connect, but
        # restart-interrupted sessions are not auto-resumed until all startup
        # wiring below completes.  Queue inbound messages until the resume
        # pass runs and every synthetic resume turn has finished.
        self._startup_restore_in_progress = True
        self._startup_restore_queue = []
        self._startup_restore_tasks = []

        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        # Initialize and connect each configured platform.
        #
        # Parallel startup connect (#83791): the original code ran a serial for-loop,
        # so every platform's connect() (with its own timeout) had to finish before
        # the next began. A single slow/failing platform (e.g. Telegram behind a dead
        # proxy) therefore delayed every other platform's connect by a full timeout
        # window, cascading one platform's failure onto WeChat/QQ/etc. We now launch
        # all platform connects concurrently and let each resolve on its own timeline;
        # per-platform timeouts and error handling are unchanged.
        # The serial pre-filter (cheap checks, adapter creation, handler wiring) stays
        # sequential -- only the (slow) connect() calls run in parallel.
        _pending_connects = []  # (platform, platform_config, adapter)
        for platform, platform_config in self.config.platforms.items():
            if await self._abort_startup_if_shutdown_requested():
                return True
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1

            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue

            # Set up message + fatal error handlers. Under multiplexing the
            # default profile needs the same whole-handler runtime scope as a
            # secondary profile: authorization and prompt rendering both run
            # before the narrower agent-turn scope is installed.
            adapter.set_message_handler(self._primary_message_handler())
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            _set_reaction = getattr(adapter, "set_reaction_handler", None)
            if callable(_set_reaction):
                _set_reaction(self._handle_reaction_event)
            adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
            adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
            adapter.set_platform_event_handler(self._primary_platform_event_handler())
            adapter._busy_input_mode = self._busy_input_mode
            _pending_connects.append((platform, platform_config, adapter))

        if await self._abort_startup_if_shutdown_requested():
            return True

        async def _connect_one_startup(p, p_cfg, adp):
            """Connect a single platform; never let one block the others (#83791)."""
            if await self._abort_startup_if_shutdown_requested(adp, p):
                return (p, adp, p_cfg, "aborted", None)
            logger.info("Connecting to %s...", p.value)
            self._update_platform_runtime_status(
                p.value, platform_state="connecting", error_code=None, error_message=None,
            )
            try:
                ok = await self._connect_initial_adapter_with_timeout(adp, p)
            except Exception as _exc:  # noqa: BLE001 - surfaced below as a retryable error
                return (p, adp, p_cfg, "exception", _exc)
            return (p, adp, p_cfg, "ok" if ok else "failed", None)

        if _pending_connects:
            # Abort-aware concurrent wait (parity with the serial loop's
            # between-platforms abort check): a restart/shutdown requested
            # while connects are in flight must cancel the still-pending
            # connects — no later platform may finish connecting — clean up
            # the ones that already completed, and abort startup.
            _task_map: dict = {}
            for (p, c, a) in _pending_connects:
                _t = asyncio.ensure_future(_connect_one_startup(p, c, a))
                _task_map[_t] = (p, c, a)
            _pending_tasks = set(_task_map)
            _abort_mid_connect = False
            while _pending_tasks:
                _done, _pending_tasks = await asyncio.wait(
                    _pending_tasks, timeout=0.05
                )
                if _pending_tasks and self._startup_should_abort():
                    _abort_mid_connect = True
                    break
            if _abort_mid_connect:
                # Cancel and fully settle the in-flight connects FIRST, so a
                # completed adapter's disconnect cannot unblock a sibling's
                # connect() before the sibling is cancelled.
                for _t in _pending_tasks:
                    _t.cancel()
                await asyncio.gather(*_pending_tasks, return_exceptions=True)
                for _t in _pending_tasks:
                    _p, _c, _a = _task_map[_t]
                    try:
                        await _a.cancel_background_tasks()
                    except Exception as e:
                        logger.debug(
                            "✗ %s background-task cancel error: %s", _p.value, e
                        )
                    await self._safe_adapter_disconnect(_a, _p)
                # Tear down adapters whose connect already succeeded — they
                # were never registered, so stop() won't reach them.
                for _t, (_p, _c, _a) in _task_map.items():
                    if _t in _pending_tasks or _t.cancelled():
                        continue
                    _res = _t.exception() is None and _t.result() or None
                    if _res and _res[3] == "ok":
                        try:
                            await _a.cancel_background_tasks()
                        except Exception as e:
                            logger.debug(
                                "✗ %s background-task cancel error: %s",
                                _p.value, e,
                            )
                        await self._safe_adapter_disconnect(_a, _p)
                await self._abort_startup_if_shutdown_requested()
                return True
            _raw = [
                _t.exception() or _t.result() for _t in _task_map
            ]
        else:
            _raw = []

        # Aggregate results single-threaded so shared state (self.adapters,
        # self._failed_platforms, the error lists, connected_count) is mutated
        # exactly as the original serial loop did -- only the connect() wall-clock
        # overlap changed.
        for _item in _raw:
            if isinstance(_item, Exception):
                # Unexpected escape from _connect_one_startup (shouldn't happen);
                # log and skip rather than aborting the whole startup.
                logger.error("Unexpected startup connect error: %s", _item)
                continue
            platform, adapter, platform_config, outcome, exc = _item
            if outcome == "aborted":
                continue
            if outcome == "exception":
                logger.error("\u2717 %s error: %s", platform.value, exc)
                # Same defensive cleanup path for exceptions -- an adapter that
                # raised mid-connect may still have a live aiohttp.ClientSession or
                # child subprocess.
                await self._safe_adapter_disconnect(adapter, platform)
                self._update_platform_runtime_status(
                    platform.value, platform_state="retrying", error_code=None, error_message=str(exc),
                )
                startup_retryable_errors.append(f"{platform.value}: {exc}")
                # Unexpected exceptions are typically transient -- queue for retry
                self._failed_platforms[platform] = {
                    "config": platform_config,
                    "attempts": 1,
                    "next_retry": time.monotonic() + 30,
                    "queued_at": time.monotonic(),
                    "credential_claim": self._adapter_credential_claim(platform, adapter),
                }
                continue
            if outcome == "ok":
                self.adapters[platform] = adapter
                connected_count += 1
                self._update_platform_runtime_status(
                    platform.value, platform_state="connected", error_code=None, error_message=None,
                )
                logger.info("\u2713 %s connected", platform.value)
            else:  # outcome == "failed"
                logger.warning("\u2717 %s failed to connect", platform.value)
                # Defensive cleanup: a failed connect() may have allocated resources
                # (aiohttp.ClientSession, poll tasks, bridge subprocesses) before
                # giving up. Without this call, those resources are orphaned and
                # Python logs "Unclosed client session" at process exit.
                await self._safe_adapter_disconnect(adapter, platform)
                if adapter.has_fatal_error:
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
                        error_code=adapter.fatal_error_code,
                        error_message=adapter.fatal_error_message,
                    )
                    target = (
                        startup_retryable_errors
                        if adapter.fatal_error_retryable
                        else startup_nonretryable_errors
                    )
                    target.append(f"{platform.value}: {adapter.fatal_error_message}")
                    # Queue for reconnection if the error is retryable
                    if adapter.fatal_error_retryable:
                        self._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                            "credential_claim": self._adapter_credential_claim(platform, adapter),
                        }
                else:
                    self._update_platform_runtime_status(
                        platform.value, platform_state="retrying", error_code=None, error_message="failed to connect",
                    )
                    startup_retryable_errors.append(f"{platform.value}: failed to connect")
                    # No fatal error info means likely a transient issue -- queue for retry
                    self._failed_platforms[platform] = {
                        "config": platform_config,
                        "attempts": 1,
                        "next_retry": time.monotonic() + 30,
                        "queued_at": time.monotonic(),
                        "credential_claim": self._adapter_credential_claim(platform, adapter),
                    }

        if await self._abort_startup_if_shutdown_requested():
            return True
        self._platform_lock_takeover_on_start = False

        if connected_count == 0:
            if startup_nonretryable_errors and not startup_retryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception:
                    pass
                self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
                self._request_clean_exit(reason)
                self._startup_restore_in_progress = False
                return True
            if startup_nonretryable_errors:
                # Mixed failure mode (NS-609): some platforms are fatally
                # misconfigured (e.g. WhatsApp enabled but never paired) while
                # others hit merely transient errors (e.g. Telegram TimedOut
                # during polling startup).  Exiting with
                # GATEWAY_FATAL_CONFIG_EXIT_CODE here is wrong in both
                # supervision worlds: under supervisors that honor the
                # exit-78 contract (systemd RestartPreventExitStatus, s6
                # finish→125 since #51228) the gateway goes PERMANENTLY down
                # over a network blip; under anything else it crash-loops.
                # Either way the retryable platforms never get their retry.
                # Log the fatal side loudly, then fall through to the
                # degraded/retry path below: the reconnect watcher recovers
                # the retryable platforms; the non-retryable ones remain
                # fatal-parked and visible in runtime status.
                logger.error(
                    "%d platform(s) fatally misconfigured and parked: %s. "
                    "Staying alive so retryable platforms can recover.",
                    len(startup_nonretryable_errors),
                    "; ".join(startup_nonretryable_errors),
                )
            if enabled_platform_count > 0:
                if startup_retryable_errors:
                    # All enabled platforms hit retryable failures (network
                    # blip, proxy timeout, or credential service outage).
                    # Keep the gateway alive so:
                    #   • cron jobs still run
                    #   • the reconnect watcher gets a chance to recover the
                    #     failing platforms once the underlying problem is
                    #     fixed (for example, a proxy or credential is corrected).
                    # Exiting here used to convert a single misconfigured
                    # platform into an infinite systemd restart loop.
                    reason = "; ".join(startup_retryable_errors)
                    logger.warning(
                        "Gateway started with no connected platforms — "
                        "%d platform(s) queued for retry: %s",
                        len(self._failed_platforms), reason,
                    )
                    try:
                        from gateway.status import write_runtime_status
                        write_runtime_status(
                            gateway_state="degraded",
                            exit_reason=None,
                        )
                    except Exception:
                        pass
                    # Fall through to the normal "running" state — reconnect
                    # watcher takes it from here.
                # All enabled platforms had no adapter (missing library or credentials).
                # In fleet deployments the same config.yaml is shared across nodes that
                # may only have credentials for a subset of platforms.  Rather than
                # failing hard, degrade gracefully and allow cron jobs to run (#5196).
                logger.warning(
                    "No adapter could be created for any of the %d configured platform(s). "
                    "Check that required dependencies are installed and credentials are set. "
                    "Gateway will continue for cron job execution.",
                    enabled_platform_count,
                )
            else:
                logger.warning("No messaging platforms enabled.")
                logger.info("Gateway will continue running for cron job execution.")

        # Update delivery router with adapters
        if await self._abort_startup_if_shutdown_requested():
            return True
        self.delivery_router.adapters = self.adapters

        self._running = True
        self._install_plugin_message_injector()
        self._update_runtime_status("running")

        # Loop-liveness heartbeat (#66892): an asyncio task so a frozen loop
        # stops refreshing ``state/gateway.heartbeat``. Cancelled with the
        # other background tasks during stop(). Best-effort — a liveness probe
        # must never be able to abort startup.
        try:
            _existing_hb = getattr(self, "_loop_heartbeat_task", None)
            if _existing_hb is None or _existing_hb.done():
                self._loop_heartbeat_task = asyncio.create_task(
                    loop_heartbeat_forever(
                        interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
                        start_time=getattr(self, "_gateway_started_at", 0.0),
                    )
                )
                _bg = getattr(self, "_background_tasks", None)
                if _bg is not None:
                    _bg.add(self._loop_heartbeat_task)
                    self._loop_heartbeat_task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start gateway loop heartbeat", exc_info=True)

        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })

        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)

        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = await build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)

        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Give freshly connected platform adapters a brief moment to settle
        # before sending restart/startup lifecycle messages. In practice this
        # helps Discord thread deliveries right after reconnect.
        if connected_count > 0:
            await asyncio.sleep(1.0)

        # Notify the chat that initiated /restart that the gateway is back.
        chat_restart_notification_pending = _restart_notification_pending()
        planned_restart_notification_pending = _planned_restart_notification_pending()
        # Capture, before _send_restart_notification() unlinks the marker,
        # whether this process booted from a chat-originated /restart. Used as
        # a one-shot signal by the /restart redelivery guard so a missing
        # dedup marker only suppresses a /restart when we KNOW we just came out
        # of a restart cycle (see _is_stale_restart_redelivery).
        if chat_restart_notification_pending:
            self._booted_from_restart = True
        await self._send_restart_notification()

        # Broadcast a lightweight "gateway is back" message to configured home
        # channels only for non-chat planned restarts (terminal/SIGUSR1/service
        # paths). Chat-originated /restart already has a precise reply target
        # in .restart_notify.json, so keep that lifecycle in the originating
        # chat/topic instead of also leaking it to the configured home channel.
        if planned_restart_notification_pending:
            try:
                await self._send_home_channel_startup_notifications(
                    skip_targets=None,
                )
            finally:
                _clear_planned_restart_notification()

        # Automatically continue fresh sessions that were interrupted by the
        # previous gateway restart/shutdown.  The resume_pending flag is cleared
        # by the normal successful-turn path, so a failed auto-resume remains
        # visible for manual recovery on the next user message.
        #
        # Delivery-obligation redelivery runs FIRST: a session whose final
        # response was generated but never confirmed-delivered has its answer
        # in the ledger — redelivering it (and clearing resume_pending for
        # that session) is strictly cheaper and more correct than re-running
        # the whole turn.
        await self._redeliver_pending_obligations()
        self._schedule_resume_pending_sessions()
        await self._finish_startup_restore()

        # Surface state.db init failures to the user's messaging platforms
        # so they know persistence is broken before losing data (#88235).
        await self._send_session_db_warning_notifications()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            # Detach the current batch atomically: reassigning to a fresh list
            # takes ownership of exactly the watchers present now, so any watcher
            # appended concurrently during the yield below isn't silently dropped
            # by a clear() on the shared list.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            # Process in batches of 100 with event-loop yield points to avoid
            # O(n^2) event-loop blocking when recovering thousands of watchers.
            for i, watcher in enumerate(watchers):
                self._spawn_supervised(
                    lambda w=watcher: self._run_process_watcher(w),
                    f"process_watcher:{watcher.get('session_id')}",
                    restart=False,
                )
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher to finalize expired sessions
        self._spawn_supervised(self._session_expiry_watcher, "session_expiry_watcher")

        # Stall watchdog: pending inbound + stale agent activity → warn user
        # to /new (does not kill the turn; see agent.session_stall_timeout).
        self._spawn_supervised(self._session_stall_watcher, "session_stall_watcher")

        # Start background kanban notifier — each gateway delivers events for
        # subscriptions owned by the profiles whose adapters it hosts, even
        # when another gateway owns the single dispatcher.
        self._spawn_supervised(self._kanban_notifier_watcher, "kanban_notifier_watcher")

        # Start background kanban dispatcher — spawns workers for ready
        # tasks. Gated by `kanban.dispatch_in_gateway` (default True).
        # When false, users run `hermes kanban daemon` externally or
        # simply don't use kanban; this loop becomes a no-op.
        self._spawn_supervised(self._kanban_dispatcher_watcher, "kanban_dispatcher_watcher")

        # Start background reconnection watcher for platforms that failed at startup
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        # Track the reconnect watcher task so _ensure_reconnect_watcher_running
        # can detect if it dies and respawn it (#70344). Spawned via
        # _spawn_supervised (not a bare asyncio.create_task) so an exception
        # escaping the watcher's OUTER while-loop -- not just the per-platform
        # inner try/except -- is caught, logged, and auto-restarted with
        # backoff instead of silently killing the watcher forever. Without
        # this, a platform already queued in _failed_platforms when the
        # watcher dies stays stranded indefinitely: _ensure_reconnect_watcher_running()
        # only gets called from a NEW fatal-error arrival, so if no other
        # platform ever fails afterward, nothing ever notices the watcher is
        # dead (#71758 -- reported as 17.5h of silent downtime for a platform
        # whose transient upstream outage had long since recovered).
        # ``on_spawn`` keeps ``_reconnect_watcher_task`` pointed at the CURRENT
        # live task even when _spawn_supervised's own backoff respawns it — so
        # _ensure_reconnect_watcher_running never mistakes a superseded handle
        # for a dead watcher and spawns a duplicate.
        self._reconnect_watcher_task = self._spawn_supervised(
            self._platform_reconnect_watcher,
            "platform_reconnect_watcher",
            on_spawn=lambda t: setattr(self, "_reconnect_watcher_task", t),
        )

        # Start background handoff watcher — picks up CLI sessions marked
        # handoff_state='pending' in state.db and re-binds them to the
        # destination platform's home channel, then forges a synthetic user
        # turn so the agent kicks off the new chat.
        self._spawn_supervised(self._handoff_watcher, "handoff_watcher")

        # Start background async-delegation watcher — drains completion events
        # from delegate_task(background=true) subagents and injects each
        # result back into its originating session as a new turn, covering the
        # idle case where the subagent finishes with no agent turn running.
        self._spawn_supervised(self._async_delegation_watcher, "async_delegation_watcher")

        # Start background /loop wakeup watcher — scans persisted loops
        # (SessionDB loop:* rows) and injects due wakeup prompts into their
        # originating chats while the session is idle.
        self._spawn_supervised(self._loop_wakeup_watcher, "loop_wakeup_watcher")

        logger.info("Press Ctrl+C to stop")

        return True

    def _spawn_supervised(self, coro_factory, name, *, restart=True, _attempt=0, on_spawn=None):
        """Launch a long-lived background task with task-level supervision.

        Complements upstream's per-iteration inner-loop try/except (which only
        guards a single loop-body) by covering what that CANNOT: an exception
        raised in the watcher's OUTER ``while self._running:`` loop or its
        pre-try setup region, plus task-level death generally. A bare
        ``asyncio.create_task`` drops such an exception on the floor — no log,
        no restart, the watcher silently gone. This retains the handle in
        ``self._background_tasks``, logs any crash, and restarts with capped
        exponential backoff up to ``_MAX_SUPERVISED_RESTARTS`` failures in rapid
        succession (each within ``_SUPERVISED_HEALTHY_SECS`` of its restart).
        The counter resets after any run that stayed healthy for at least
        ``_SUPERVISED_HEALTHY_SECS`` — so a long-lived daemon that crashes
        occasionally over days is never permanently abandoned.

        ``on_spawn`` (optional) is invoked with the freshly-created task on
        every spawn, INCLUDING internal backoff respawns. Callers that also
        track the live handle elsewhere (e.g. ``self._reconnect_watcher_task``
        for ``_ensure_reconnect_watcher_running``) MUST pass it — otherwise the
        supervisor's own respawn creates a new task without updating that
        external handle, so ``_ensure_...`` later sees the stale/done handle
        and spawns a SECOND concurrent watcher (double reconnect attempts).
        """
        if getattr(self, "_background_tasks", None) is None:
            self._background_tasks = set()

        # Monotonic spawn timestamp captured per spawn: the ``_done`` callback
        # uses it to distinguish a rapid crash-loop from a healthy-run-then-crash.
        _started = time.monotonic()

        # Deliberately do NOT pass name= to create_task — some test doubles mock
        # create_task with a signature that rejects the name kwarg.
        task = asyncio.create_task(coro_factory())
        # Mark this as a PERMANENT supervised watcher, not transient background
        # work. Session-expiry, kanban, and reconnect watchers live for the
        # whole process and must remain distinguishable from finite work.
        task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
        self._background_tasks.add(task)
        if on_spawn is not None:
            # Record the live handle NOW so an external tracker (e.g.
            # _reconnect_watcher_task) always points at the current task, not a
            # dead one left behind by a prior supervised respawn.
            try:
                on_spawn(task)
            except Exception:  # pragma: no cover - defensive; a tracker must never kill the spawn
                logger.debug("on_spawn callback for %s raised", name, exc_info=True)

        def _done(t):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is None:
                # Clean return == deliberate shutdown or a self-disabling watcher
                # (e.g. a gated no-op that returns synchronously). Respawning here
                # would busy-spin such a watcher — so NEVER restart on clean exit.
                return
            logger.error("Supervised task %s died: %r", name, exc, exc_info=exc)
            if restart and self._running:
                ran_for = time.monotonic() - _started
                if ran_for >= self._SUPERVISED_HEALTHY_SECS:
                    # Ran healthily for a while before crashing — this is a
                    # FRESH failure, not part of a rapid crash-loop. Reset the
                    # consecutive counter so a daemon that crashes a handful of
                    # times over days is never permanently abandoned.
                    effective_attempt = 0
                else:
                    effective_attempt = _attempt
                if effective_attempt >= self._MAX_SUPERVISED_RESTARTS:
                    logger.error(
                        "Supervised task %s died %d times in rapid succession "
                        "(each within %ds of restart) — giving up restarts",
                        name,
                        effective_attempt,
                        self._SUPERVISED_HEALTHY_SECS,
                    )
                    return
                backoff = min(60, 2 ** min(effective_attempt, 6))

                async def _respawn():
                    await asyncio.sleep(backoff)
                    if self._running:
                        self._spawn_supervised(
                            coro_factory,
                            name,
                            restart=restart,
                            _attempt=effective_attempt + 1,
                            on_spawn=on_spawn,
                        )

                respawn_task = asyncio.create_task(_respawn())
                self._background_tasks.add(respawn_task)
                respawn_task.add_done_callback(self._background_tasks.discard)

        task.add_done_callback(_done)
        return task

    async def _handoff_watcher(self, interval: float = 2.0) -> None:
        """Background task that processes pending CLI→gateway session handoffs.

        Polls ``state.db`` for sessions in ``handoff_state='pending'`` and,
        for each one:

        1. Atomically claims it (pending → running).
        2. Resolves the destination platform's configured home channel.
        3. Re-binds the gateway's session_key for that home channel to the
           CLI's existing session_id via ``session_store.switch_session`` so
           the full role-aware transcript replays on the next agent turn.
        4. Forges a synthetic ``MessageEvent`` (``internal=True``) with a
           handoff-notice text and dispatches through the normal gateway
           message pipeline so the agent runs and replies on the platform.
        5. Marks the row ``completed`` (or ``failed`` with ``handoff_error``).

        The CLI process is poll-blocked on the row's terminal state and
        prints the result to the user.
        """
        # Initial delay so the gateway is fully connected to its platforms
        # before we try to dispatch handoffs through them.
        await asyncio.sleep(5)
        while self._running:
            try:
                if self._session_db is None:
                    await asyncio.sleep(interval)
                    continue
                pending = await self._session_db.list_pending_handoffs()
                for row in pending:
                    session_id = row.get("id")
                    if not session_id:
                        continue
                    if not await self._session_db.claim_handoff(session_id):
                        # Another tick or another gateway already claimed it.
                        continue
                    try:
                        await self._process_handoff(row)
                        await self._session_db.complete_handoff(session_id)
                    except Exception as exc:
                        logger.warning(
                            "Handoff for session %s failed: %s",
                            session_id, exc, exc_info=True,
                        )
                        await self._session_db.fail_handoff(session_id, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    async def _process_handoff(self, row: Dict[str, Any]) -> None:
        """Execute one handoff row. Raises on failure (caller marks failed)."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        from gateway.platforms.base import MessageEvent

        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")

        # Resolve platform enum
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")

        # The retained platform adapter must be live.
        transport = resolve_delivery_transport(platform, self.config, self.adapters)
        if not transport:
            raise RuntimeError(
                f"platform '{platform_name}' is not active in this gateway"
            )
        adapter = transport.adapter

        # Home channel must be configured
        home = self.config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; "
                f"run /sethome on the desired chat first"
            )

        cli_title = row.get("title") or cli_session_id[:8]

        # Try to create a fresh thread on the destination so the handoff
        # has its own scrollback. Adapter returns None if threading isn't
        # supported (Matrix/WhatsApp/Signal/SMS) or if creation failed
        # (no permission, topics-mode off, parent is a DM, etc.). When
        # None we fall through to using the home channel directly — the
        # synthetic turn still lands; just without thread isolation.
        thread_name = f"Hermes — {cli_title}"
        try:
            new_thread_id = await adapter.create_handoff_thread(
                str(home.chat_id), thread_name,
            )
        except Exception as exc:
            logger.debug(
                "Handoff: create_handoff_thread raised on %s: %s",
                platform_name, exc, exc_info=True,
            )
            new_thread_id = None

        # Use the new thread if the adapter created one; otherwise fall
        # back to whatever thread (if any) the home channel was configured
        # with.
        effective_thread_id = new_thread_id or (
            str(home.thread_id) if home.thread_id else None
        )

        # Determine chat_type/user_id for the destination source.
        #
        # Telegram private-chat DM topics are represented differently from
        # group/forum threads by the inbound adapter. A handoff-created topic
        # in a positive Telegram chat_id must therefore use the same DM-topic
        # source shape as the user's next real message; otherwise the synthetic
        # handoff turn binds a generic `thread` session key while real replies
        # arrive on a `dm` session key.
        home_chat_id = str(home.chat_id)
        is_telegram_private_chat = (
            platform == Platform.TELEGRAM
            and looks_like_telegram_private_chat_id(home_chat_id)
        )

        if new_thread_id and not is_telegram_private_chat:
            dest_chat_type = "thread"
            dest_user_id = "system:handoff"
        else:
            # No thread — assume DM-style for the home channel. For Telegram
            # private-chat topics, use the real user id (same as chat_id) so
            # topic-mode checks and binding persistence see the same identity as
            # subsequent inbound user messages.
            dest_chat_type = "dm"
            dest_user_id = home_chat_id if is_telegram_private_chat else "system:handoff"

        # Discord thread destinations must key on the thread's OWN id, not the
        # parent channel's, because the Discord adapter builds organic in-thread
        # messages with ``chat_id == thread id`` — so ``build_session_key``
        # yields ``…:thread:{thread}:{thread}``. If the handoff keys on the
        # parent channel (``…:thread:{parent}:{thread}``) the next real user
        # reply in the thread resolves to a DIFFERENT session_key and spawns a
        # fresh session instead of continuing the handed-off one.
        #
        # This is Discord-specific: Slack and Telegram adapters key organic
        # thread messages with ``chat_id == parent_channel`` and the thread
        #/topic id only in ``thread_id``, so for those platforms the parent
        # channel is correct (and the deeper chat_type normalization — handoff
        # uses "thread" but Slack organic uses "group" — is a separate issue).
        dest_chat_id = home_chat_id
        dest_source = SessionSource(
            platform=platform,
            chat_id=dest_chat_id,
            chat_name=home.name,
            chat_type=dest_chat_type,
            user_id=dest_user_id,
            user_name="Handoff",
            thread_id=effective_thread_id,
        )

        # Compute the gateway's session_key for that destination using the
        # same rules its adapters use, so switch_session targets the right
        # entry. For thread destinations build_session_key keys without
        # user_id (thread_sessions_per_user defaults to False) — so the
        # next real user message in the thread shares this same session.
        platform_cfg = self.config.platforms.get(platform)
        extra = platform_cfg.extra if platform_cfg else {}
        session_key = build_session_key(
            dest_source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
        )

        # Make sure there's an entry in the session_store for this key. If
        # the home channel has never been used, get_or_create_session
        # creates one; switch_session then re-points it.
        await self.async_session_store.get_or_create_session(dest_source)

        # Re-bind the destination key to the CLI session_id. switch_session
        # ends the prior session in SQLite and reopens the CLI session under
        # the new key. The CLI's transcript becomes the active one for the
        # gateway from this moment on.
        switched = await self.async_session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(
                f"could not switch session key {session_key} → {cli_session_id}"
            )

        # Evict any cached create_agent for this session_key so the next dispatch
        # rebuilds it against the CLI session_id (mirrors /resume / /branch).
        self.agent_cache.evict(session_key)

        # Cancel any in-flight running-agent state for the destination key
        # so the synthetic turn isn't queued behind a stale running flag.
        self._release_turn_state(session_key)

        synthetic_text = (
            f"[Session was just handed off from CLI (\"{cli_title}\") to this "
            f"channel. The full prior conversation history is loaded above. "
            f"Briefly confirm you're working here and summarize what we were "
            f"working on, so the user can continue from this device.]"
        )

        synthetic_event = MessageEvent(
            text=synthetic_text,
            source=dest_source,
            internal=True,
        )

        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, platform_name, home.chat_id, effective_thread_id,
            session_key,
        )

        # Dispatch through the runner directly. Going through
        # adapter.handle_message would spawn a background task and we'd
        # lose synchronous error visibility; calling _handle_message inline
        # keeps the success/failure path observable for the watcher.
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have already delivered the response inline.
            # Either way, agent ran without raising — count as success.
            return

        # Send the agent's reply to the destination. Route to the new
        # thread if we created one; otherwise the configured home channel
        # (which may itself carry a thread_id). Send through the resolved
        # transport so profile-aware routing stays centralized.
        send_metadata: Dict[str, Any] = {}
        if effective_thread_id:
            send_metadata["thread_id"] = effective_thread_id
        try:
            result = await transport.send(
                platform,
                str(home.chat_id),
                response_text,
                send_metadata or None,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc

        if not getattr(result, "success", True):
            err = getattr(result, "error", "send returned success=False")
            raise RuntimeError(f"adapter.send failed: {err}")

    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that finalizes expired sessions.

        Runs every ``interval`` seconds (default 5 min).  For each session
        whose reset policy has expired, invokes ``on_session_finalize``
        hooks, cleans up the cached create_agent's tool resources, evicts the
        cache entry so it can be garbage-collected, and marks the session
        so it won't be finalized again.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _finalize_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FINALIZE_RETRIES = 3
        while self._running:
            try:
                await self.async_session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.session_store._entries.items()):
                    if entry.expiry_finalized:
                        continue
                    if not await self.async_session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to finalize (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        try:
                            _parts = key.split(":")
                            _platform = _parts[2] if len(_parts) > 2 else ""
                            # Off-loop + bounded: plugin finalize hooks can
                            # block arbitrarily (see _finalize_session_off_loop)
                            # and this watcher runs on the gateway event loop.
                            await self._finalize_session_off_loop(
                                session_id=entry.session_id,
                                platform=_platform,
                                reason="session_expired",
                            )
                        except Exception:
                            pass
                        # Shut down memory provider and close tool resources
                        # on the cached agent.  Idle agents live in
                        # _agent_cache (not _running_agents), so look there.
                        _cached_agent = None
                        _cache_lock = self.agent_cache.lock
                        if _cache_lock is not None:
                            with _cache_lock:
                                _cached = self.agent_cache.entries.get(key)
                                _cached_agent = _cached.agent if _cached else None
                        # Fall back to _running_agents in case the agent is
                        # still mid-turn when the expiry fires.
                        if _cached_agent is None:
                            _exp_state = self.sessions.peek(key)
                            _cached_agent = _exp_state.turn.agent if _exp_state else None
                        if _cached_agent and _cached_agent is not _AGENT_PENDING_SENTINEL:
                            await self._cleanup_agent_resources_off_loop(
                                _cached_agent, context="session expiry"
                            )
                        # Drop the cache entry so the create_agent (and its LLM
                        # clients, tool schemas, memory provider refs) can
                        # be garbage-collected.  Otherwise the cache grows
                        # unbounded across the gateway's lifetime.
                        self.agent_cache.evict(key)
                        # Permanently finalizing this session — one funnel
                        # call drops every conversation-scoped dict AND the
                        # boundary security state (approvals, update
                        # prompts, slash-confirm) so the dicts don't grow
                        # unbounded across the gateway's lifetime. (Idle
                        # agent-cache eviction must NOT do this: the
                        # session is still alive and a resumed turn rebuilds
                        # its agent from these overrides. Only true session
                        # finalization, /new, and /reset clear them.) See
                        # _CONVERSATION_SCOPED_STATE.
                        self._clear_conversation_scope(
                            key, reason="expiry_finalized"
                        )
                        # Persist the finalized flag to sessions.json AND
                        # state.db (single write-path, #9006) — also drops
                        # the persisted /model override, since finalization
                        # is a conversation boundary.
                        await self.async_session_store.set_expiry_finalized(entry)
                        logger.debug(
                            "Session expiry finalized for %s",
                            entry.session_id,
                        )
                        _finalize_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _finalize_failures.get(entry.session_id, 0) + 1
                        _finalize_failures[entry.session_id] = failures
                        if failures >= _MAX_FINALIZE_RETRIES:
                            logger.warning(
                                "Session finalize gave up after %d attempts for %s: %s. "
                                "Marking as finalized to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            await self.async_session_store.set_expiry_finalized(
                                entry, clear_model_override=False
                            )
                            _finalize_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Session finalize failed (%d/%d) for %s: %s",
                                failures, _MAX_FINALIZE_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _done = sum(
                        1 for _, e in _expired_entries if e.expiry_finalized
                    )
                    _failed = len(_expired_entries) - _done
                    if _failed:
                        logger.info(
                            "Session expiry done: %d finalized, %d pending retry",
                            _done, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d finalized", _done,
                        )

                # Sweep agents that have been idle beyond the TTL regardless
                # of session reset policy.  This catches sessions with very
                # long / "never" reset windows, whose cached agents would
                # otherwise pin memory for the gateway's entire lifetime.
                try:
                    _idle_evicted = self.agent_cache.sweep_idle()
                    if _idle_evicted:
                        logger.info(
                            "Agent cache idle sweep: evicted %d agent(s)",
                            _idle_evicted,
                        )
                except Exception as _e:
                    logger.debug("Idle agent sweep failed: %s", _e)

                # Neither the LRU cap nor the idle TTL is aware of how much
                # memory a cached transcript costs, so a busy gateway keeps
                # every warm session's tool output resident until RSS hits the
                # cgroup limit (#80764). Shed LRU transcripts once the heap is
                # over budget; they reload from the persisted session on the
                # next turn.
                try:
                    self.agent_cache.sweep_under_pressure()
                except Exception as _e:
                    logger.debug("Agent cache pressure sweep failed: %s", _e)

                # Periodically prune stale SessionStore entries.  The
                # in-memory dict (and sessions.json) would otherwise grow
                # unbounded in gateways serving many rotating chats /
                # threads / users over long time windows.  Pruning is
                # invisible to users — a resumed session just gets a
                # fresh session_id, exactly as if the reset policy fired.
                _last_prune_ts = getattr(self, "_last_session_store_prune_ts", 0.0)
                _prune_interval = 3600.0  # once per hour
                if time.time() - _last_prune_ts > _prune_interval:
                    try:
                        _max_age = int(
                            getattr(self.config, "session_store_max_age_days", 0) or 0
                        )
                        if _max_age > 0:
                            _pruned = await self.async_session_store.prune_old_entries(_max_age)
                            if _pruned:
                                logger.info(
                                    "SessionStore prune: dropped %d stale entries",
                                    _pruned,
                                )
                    except Exception as _e:
                        logger.debug("SessionStore prune failed: %s", _e)
                    self._last_session_store_prune_ts = time.time()
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    def _session_stall_timeout_seconds(self) -> float:
        """Return configured stall timeout (seconds); 0 disables the watchdog."""
        return _float_env("HERMES_SESSION_STALL_TIMEOUT", 300)

    def _iter_gateway_adapters(self):
        """Yield every live platform adapter."""
        seen: set[int] = set()
        for adapter in list(getattr(self, "adapters", {}).values()):
            if adapter is None:
                continue
            aid = id(adapter)
            if aid in seen:
                continue
            seen.add(aid)
            yield adapter

    def _session_activity_for_stall(self, session_key: str) -> Optional[dict]:
        """Return the shared activity snapshot for stall progress (#72039).

        Single progress source: ``create_agent.get_activity_summary()`` /
        ``agent.session_activity``. No turn-start or pending-inbound clocks.
        """
        import agent.status_output as status_output
        state = self.sessions.peek(session_key)
        agent = state.turn.agent if state is not None else None
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return None
        if not hasattr(agent, "get_activity_summary"):
            return None
        try:
            summary = status_output.get_activity_summary(agent)
        except Exception:
            return None
        return summary if isinstance(summary, dict) else None

    async def _check_session_stalls(self, timeout_seconds: float) -> int:
        """Scan pending inbound sessions and notify once per stall episode.

        Returns the number of notifications sent this pass (for tests).
        """
        from gateway.session_stall import (
            format_session_stall_notification,
            resolve_session_idle_seconds_from_activity,
            should_clear_session_stall_notification,
            should_emit_session_stall_notification,
        )

        sent = 0
        now = time.time()
        candidates: Dict[str, tuple[Any, Any]] = {}

        for adapter in self._iter_gateway_adapters():
            pending_slot = getattr(adapter, "_pending_messages", None) or {}
            for session_key, event in list(pending_slot.items()):
                if session_key and session_key not in candidates and event is not None:
                    candidates[session_key] = (adapter, event)

        for session_key, state in list(self.sessions.states.items()):
            overflow = state.conversation.queued_events
            if not session_key or session_key in candidates or not overflow:
                continue
            event = overflow[0]
            source = getattr(event, "source", None)
            adapter = (
                self._adapter_for_source(source) if source is not None else None
            )
            if adapter is None:
                continue
            candidates[session_key] = (adapter, event)

        for session_key, (adapter, pending_event) in list(candidates.items()):
            has_pending = pending_event is not None
            activity = (
                self._session_activity_for_stall(session_key) if has_pending else None
            )
            idle_seconds = (
                resolve_session_idle_seconds_from_activity(activity, now=now)
                if has_pending
                else None
            )
            conversation = self.sessions.state(session_key).conversation
            already = conversation.stall_notified
            if should_clear_session_stall_notification(
                timeout_seconds=timeout_seconds,
                idle_seconds=idle_seconds,
                has_pending_inbound=has_pending,
            ):
                conversation.stall_notified = False
                already = False
            if not should_emit_session_stall_notification(
                timeout_seconds=timeout_seconds,
                idle_seconds=idle_seconds,
                has_pending_inbound=has_pending,
                already_notified=already,
            ):
                continue

            if idle_seconds is None:
                continue
            mins = max(1, int(idle_seconds // 60))
            activity = activity or {}
            logger.warning(
                "Session stall detected: session=%s idle=%.0fs "
                "(timeout=%.0fs, ~%d min); pending inbound present "
                "| last_activity=%s | provenance=%s "
                "(agent.session_stall_timeout)",
                session_key,
                idle_seconds,
                timeout_seconds,
                mins,
                activity.get("last_activity_desc")
                or activity.get("last_activity_description")
                or "unknown",
                activity.get("provenance")
                or activity.get("last_activity_provenance")
                or "unknown",
            )
            source = getattr(pending_event, "source", None)
            chat_id = getattr(source, "chat_id", None) if source is not None else None
            if not chat_id:
                logger.warning(
                    "Session stall notify skipped (no chat_id): session=%s",
                    session_key,
                )
                # Cannot deliver; latch to avoid log spam every tick.
                conversation.stall_notified = True
                continue
            # #76354 review S2: re-read pending state + activity timestamp
            # IMMEDIATELY before delivery. The snapshot above ages while
            # earlier candidates in this pass await their sends; an agent
            # that made progress (or drained its queue) in that window must
            # not receive a false stall notice. Abort and leave the latch
            # un-set so the next tick re-evaluates from scratch.
            still_pending = (
                (getattr(adapter, "_pending_messages", None) or {}).get(
                    session_key
                )
                is not None
                or bool(
                    self.sessions.state(session_key).conversation.queued_events
                )
            )
            fresh_idle = resolve_session_idle_seconds_from_activity(
                self._session_activity_for_stall(session_key),
                now=time.time(),
            )
            if not still_pending or (
                fresh_idle is not None and fresh_idle < timeout_seconds
            ):
                logger.info(
                    "Session stall notify aborted (no longer stale): "
                    "session=%s pending=%s fresh_idle=%s",
                    session_key,
                    still_pending,
                    fresh_idle,
                )
                # Re-arm: drop any stale latch so a FUTURE genuine stall
                # episode notifies again.
                conversation.stall_notified = False
                continue
            try:
                metadata = (
                    self._thread_metadata_for_source(source)
                    if source is not None and hasattr(self, "_thread_metadata_for_source")
                    else None
                )
                # Round-2 #2: bound the send. A wedged adapter transport
                # (network hang, dead websocket) must not block the whole
                # watcher pass — sibling candidates in this loop would never
                # be evaluated and the watcher itself would stop ticking.
                try:
                    result = await asyncio.wait_for(
                        adapter.send(
                            str(chat_id),
                            format_session_stall_notification(idle_seconds),
                            metadata=metadata,
                        ),
                        timeout=_STALL_NOTIFY_SEND_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Session stall notify send timed out after %.0fs "
                        "for %s; will retry next tick",
                        _STALL_NOTIFY_SEND_TIMEOUT_SECONDS,
                        session_key,
                    )
                    continue  # do not latch; retry next tick
                # Adapters often return SendResult(success=False) instead of raising.
                if result is not None and getattr(result, "success", True) is False:
                    logger.warning(
                        "Session stall notify failed for %s: %s",
                        session_key,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue  # do not latch; retry next tick
                sent += 1
                conversation.stall_notified = True
            except Exception as exc:
                logger.warning(
                    "Session stall notify failed for %s: %s",
                    session_key,
                    exc,
                )
                # Do not latch — retry next watcher tick until delivery or episode clear.

        # Drop latches for sessions that no longer appear in any pending map.
        for key, state in self.sessions.states.items():
            if key not in candidates:
                state.conversation.stall_notified = False

        return sent

    async def _session_stall_watcher(self, interval: float = 30.0):
        """Periodic pending-inbound + stale-activity stall watchdog (#72016).

        Progress comes only from ``get_activity_summary()`` (#72039).
        Pending inbound is a notify policy gate, not a progress clock.
        Notify-only: does not kill the turn (contrast ``gateway_timeout`` /
        ``shutdown_watchdog``).
        """
        # Short initial delay so startup reconnect noise does not false-fire.
        await asyncio.sleep(min(30.0, max(1.0, float(interval))))
        while self._running:
            try:
                timeout = self._session_stall_timeout_seconds()
                if timeout > 0:
                    await self._check_session_stalls(timeout)
            except Exception as exc:
                logger.debug("Session stall watcher error: %s", exc)
            # Interruptible sleep
            steps = max(1, int(float(interval)))
            for _ in range(steps):
                if not self._running:
                    break
                await asyncio.sleep(1)

    def _start_systemd_watchdog(self) -> bool:
        """Start sd_notify only after a configured gateway is truly running."""
        if not self._running or self.config.systemd_watchdog_seconds <= 0:
            return False
        if self._systemd_watchdog is not None:
            return True

        from gateway.systemd_notify import SystemdWatchdog

        watchdog = SystemdWatchdog(config_enabled=True)
        if not watchdog.start():
            return False
        self._systemd_watchdog = watchdog
        watchdog.ready("Hermes Gateway running")
        return True

    async def _stop_systemd_watchdog(self) -> None:
        """Stop heartbeats before any potentially long shutdown drain."""
        watchdog = self._systemd_watchdog
        if watchdog is None:
            return
        self._systemd_watchdog = None
        await watchdog.stop()

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        # getattr-guard: shutdown-path tests build bare runners via
        # object.__new__ that lack the liveness-guard machinery.
        _stop_guards = getattr(self, "_stop_loop_liveness_guards", None)
        if callable(_stop_guards):
            _stop_guards()
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            def _kill_tool_subprocesses(phase: str) -> list:
                """Kill tool subprocesses + tear down terminal envs + browsers.

                Returns the cron job IDs this phase marked interrupted, so the
                caller can notify their owners while adapters are still up
                (#82232). Empty list when no cron work was in flight.

                Called twice in the shutdown path: once eagerly after a
                drain timeout forces agent interrupt (so we reclaim bash/
                sleep children before systemd TimeoutStopSec escalates to
                SIGKILL on the cgroup — #8202), and once as a final
                catch-all at the end of _stop_impl() for the graceful
                path or anything respawned mid-teardown.

                All steps are best-effort; exceptions are swallowed so
                one subsystem's failure doesn't block the rest.
                """
                try:
                    from tools.process_registry import process_registry
                    _killed = process_registry.kill_all()
                    if _killed:
                        logger.info(
                            "Shutdown (%s): killed %d tool subprocess(es)",
                            phase, _killed,
                        )
                except Exception as _e:
                    logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
                _marked_cron_jobs: list = []
                try:
                    # Any cron job still dispatched at this instant just had
                    # its tool subprocess killed above (kill_all() has no
                    # per-job-ID targeting — it's a global sweep). Its agent
                    # thread is still alive in this process and may go on to
                    # produce a plausible-looking final response from the
                    # now-truncated tool output; mark the run interrupted so
                    # the scheduler can never report that as success (#60432).
                    # No-op when no cron job is in flight.
                    from cron.scheduler import mark_running_jobs_interrupted
                    _interrupted = _marked_cron_jobs = mark_running_jobs_interrupted(
                        f"Gateway shutdown ({phase}) killed the job's tool "
                        "subprocess before the run finished."
                    )
                    if _interrupted:
                        logger.warning(
                            "Shutdown (%s): marked %d in-flight cron job(s) interrupted: %s",
                            phase, len(_interrupted), ", ".join(_interrupted),
                        )
                except Exception as _e:
                    logger.debug("mark_running_jobs_interrupted (%s) error: %s", phase, _e)
                try:
                    from tools.async_delegation import interrupt_all as _interrupt_async
                    _async_n = _interrupt_async(reason=f"gateway shutdown ({phase})")
                    if _async_n:
                        logger.info(
                            "Shutdown (%s): interrupted %d background delegation(s)",
                            phase, _async_n,
                        )
                except Exception as _e:
                    logger.debug("async interrupt_all (%s) error: %s", phase, _e)
                try:
                    from tools.terminal_tool import cleanup_all_environments
                    cleanup_all_environments()
                except Exception as _e:
                    logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception as _e:
                    logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)
                return _marked_cron_jobs

            # Thread-based shutdown watchdog (#66892): asyncio timeouts cannot
            # recover a frozen loop. Arm a plain OS thread at the start of
            # stop(); if teardown never finishes within drain+grace it dumps
            # faulthandler stacks and os._exit so KeepAlive/systemd can revive.
            # Skip under pytest so stop()-driving unit tests don't get a
            # delayed hard-exit in the worker.
            _watchdog_done = threading.Event()
            self._shutdown_watchdog_done = _watchdog_done
            _stop_started_at_box: dict[str, float] = {}

            def _shutdown_watchdog_snapshot() -> dict:
                started = _stop_started_at_box.get("t")
                return {
                    "restart_requested": bool(self._restart_requested),
                    "draining": bool(self._draining),
                    "running": bool(self._running),
                    "active_agents": self.sessions.running_count(),
                    "active_cron_jobs": self._active_cron_job_count(),
                    "restart_drain_timeout": self._restart_drain_timeout,
                    "watchdog_delay_s": resolve_shutdown_watchdog_delay(
                        self._restart_drain_timeout
                    ),
                    "phase_elapsed_s": (
                        time.monotonic() - started if started is not None else None
                    ),
                }

            if not os.environ.get("PYTEST_CURRENT_TEST"):
                arm_shutdown_watchdog(
                    resolve_shutdown_watchdog_delay(self._restart_drain_timeout),
                    done_event=_watchdog_done,
                    snapshot_fn=_shutdown_watchdog_snapshot,
                    exit_code=1,
                )

            try:
                await _stop_impl_body(
                    _kill_tool_subprocesses,
                    _stop_started_at_box,
                )
            finally:
                _watchdog_done.set()

        async def _stop_impl_body(_kill_tool_subprocesses, _stop_started_at_box) -> None:
            logger.info(
                "Stopping gateway%s...",
                " for restart" if self._restart_requested else "",
            )
            _stop_started_at = time.monotonic()
            _stop_started_at_box["t"] = _stop_started_at

            def _phase_elapsed() -> float:
                return time.monotonic() - _stop_started_at

            self._running = False
            self._clear_plugin_message_injector()
            self._draining = True

            stop_watchdog = getattr(self, "_stop_systemd_watchdog", None)
            if callable(stop_watchdog):
                await stop_watchdog()

            # Notify all chats with active agents BEFORE draining.
            # Adapters are still connected here, so messages can be sent.
            await self._notify_active_sessions_of_shutdown()
            logger.info(
                "Shutdown phase: notify_active_sessions done at +%.2fs",
                _phase_elapsed(),
            )

            timeout = self._restart_drain_timeout

            # Pre-mark sessions as resume_pending BEFORE the drain wait.
            # If the process is killed by the service manager during the
            # drain, the durable marker is already written so the next
            # gateway boot can recover in-flight sessions (#27856).
            _pre_drain_keys: list[str] = []
            for _sk, _agent in self.sessions.running_items():
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    await self.async_session_store.mark_resume_pending(
                        _sk,
                        "restart_timeout" if self._restart_requested else "shutdown_timeout",
                    )
                    _pre_drain_keys.append(_sk)
                except Exception as _e:
                    logger.debug("pre-drain mark_resume_pending failed for %s: %s", _sk, _e)

            _cron_at_start = self._active_cron_job_count()
            # In-flight cron work gets its own floor, clamped to the watchdog
            # leash we're already running under so the extra wait can never
            # cost us the post-drain cleanup window (#82161).
            # getattr-guard: shutdown-path tests drive _stop_impl_body from
            # bare doubles that aren't GatewayRunner instances, so they don't
            # pick up the class-level default.
            _cron_drain_cfg = getattr(
                self, "_cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
            )
            _cron_timeout = resolve_cron_drain_budget(
                timeout,
                _cron_drain_cfg,
                watchdog_delay=resolve_shutdown_watchdog_delay(timeout),
                elapsed=_phase_elapsed(),
            )
            if _cron_at_start and _cron_timeout > timeout:
                logger.info(
                    "Shutdown drain: %d in-flight cron job(s) — waiting up to "
                    "%.0fs for them (cron_drain_timeout=%.0fs, "
                    "restart_drain_timeout=%.0fs)",
                    _cron_at_start,
                    _cron_timeout,
                    _cron_drain_cfg,
                    timeout,
                )
            _drain_started_at = time.monotonic()
            active_agents, timed_out = await self._drain_active_agents(
                timeout, _cron_timeout
            )
            _drain_elapsed = time.monotonic() - _drain_started_at
            logger.info(
                "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
                "timed_out=%s, active_at_start=%d, active_now=%d, "
                "cron_at_start=%d, cron_now=%d)",
                _phase_elapsed(),
                _drain_elapsed,
                timed_out,
                len(active_agents),
                self.sessions.running_count(),
                _cron_at_start,
                self._active_cron_job_count(),
            )

            if not timed_out:
                # Drain completed gracefully — all running sessions finished.
                # Clear the pre-drain resume_pending markers so sessions that
                # completed during the drain window don't carry a stale flag.
                for _sk in _pre_drain_keys:
                    if not self.sessions.is_running(_sk):
                        try:
                            await self.async_session_store.clear_resume_pending(_sk)
                        except Exception as _e:
                            logger.debug(
                                "clear_resume_pending after drain failed for %s: %s",
                                _sk, _e,
                            )

            if timed_out:
                logger.warning(
                    "Gateway drain timed out after %.1fs with %d active agent(s), "
                    "%d in-flight cron job(s); "
                    "interrupting remaining work.",
                    _drain_elapsed,
                    self.sessions.running_count(),
                    self._active_cron_job_count(),
                )
                # Mark forcibly-interrupted sessions as resume_pending BEFORE
                # interrupting the agents.  This preserves each session's
                # session_id + transcript so the next message on the same
                # session_key auto-resumes from the existing conversation
                # instead of getting routed through suspend_recently_active()
                # and converted into a fresh session.  Terminal escalation
                # for genuinely stuck sessions still flows through the
                # existing ``.restart_failure_counts`` stuck-loop counter
                # (incremented below, threshold 3), which sets
                # ``suspended=True`` and overrides resume_pending.
                #
                # Iterate self._running_agents (current) rather than the
                # drain-start ``active_agents`` snapshot — the snapshot
                # may include sessions that finished gracefully during
                # the drain window, and marking those falsely would give
                # them a stray restart-interruption system note on their
                # next turn even though their previous turn completed
                # cleanly.  Skip pending sentinels for the same reason
                # _interrupt_active_agents() does: their agent hasn't
                # started yet, there's nothing to interrupt, and the
                # session shouldn't carry a misleading resume flag.
                _resume_reason = (
                    "restart_timeout" if self._restart_requested else "shutdown_timeout"
                )
                for _sk, _agent in self.sessions.running_items():
                    if _agent is _AGENT_PENDING_SENTINEL:
                        continue
                    try:
                        await self.async_session_store.mark_resume_pending(_sk, _resume_reason)
                    except Exception as _e:
                        logger.debug(
                            "mark_resume_pending failed for %s: %s",
                            _sk, _e,
                        )
                self._interrupt_active_agents(
                    _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                )
                interrupt_deadline = asyncio.get_running_loop().time() + 5.0
                while self.sessions.running_count() and asyncio.get_running_loop().time() < interrupt_deadline:
                    self._update_runtime_status("draining")
                    await asyncio.sleep(0.1)

                # A pending entry can be promoted to a real agent after the
                # first interrupt. Re-signal any work still live at the end of
                # the cooperative settle window.
                if self.sessions.running_count():
                    self._interrupt_active_agents(
                        _INTERRUPT_REASON_GATEWAY_RESTART
                        if self._restart_requested
                        else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                    )
                    logger.debug(
                        "Re-signaled interrupt for work still live at settle-window exit"
                    )

                # Kill lingering tool subprocesses NOW, before we spend more
                # budget on adapter disconnect / session DB close.  Under
                # systemd (TimeoutStopSec bounded by drain_timeout+headroom),
                # deferring this to the end of stop() risks systemd escalating
                # to SIGKILL on the cgroup first — at which point bash/sleep
                # children left behind by an interrupted terminal tool get
                # killed by systemd instead of us (issue #8202).  The final
                # catch-all cleanup below still runs for the graceful path.
                _interrupted_cron_jobs = _kill_tool_subprocesses("post-interrupt")
                logger.info(
                    "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                    _phase_elapsed(),
                )
                # Last window where the transport is still up. The cron worker
                # whose run we just killed will try to deliver its own
                # "interrupted" notice, but it gets there after the adapter
                # teardown below and the message is lost (#82232).
                try:
                    await self._notify_interrupted_cron_jobs(_interrupted_cron_jobs)
                except Exception as _e:
                    logger.debug("Cron interrupt notification failed: %s", _e)
                logger.info(
                    "Shutdown phase: cron interrupt notices done at +%.2fs",
                    _phase_elapsed(),
                )

            if self._restart_requested and self._restart_detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart: %s", e)

            await self._finalize_shutdown_agents(active_agents)

            # Also shut down memory providers on idle cached agents.
            # _finalize_shutdown_agents only handles agents that were
            # mid-turn at drain time; the _agent_cache may still hold
            # idle agents whose MemoryProviders never received
            # on_session_end().
            _cache_lock = self.agent_cache.lock
            _cache = self.agent_cache.entries
            if _cache_lock is not None and _cache is not None:
                with _cache_lock:
                    _idle_agents = list(_cache.values())
                    _cache.clear()
                for _entry in _idle_agents:
                    _agent = _entry.agent
                    # Bounded + off-loop so a wedged memory provider on one
                    # idle agent can't hang shutdown indefinitely — that path
                    # is why SIGTERM failed to kill the process (#53175).
                    await self._cleanup_agent_resources_off_loop(
                        _agent, context="shutdown idle-cache"
                    )

            # Completion flush tasks can be sleeping in their fan-in window or
            # blocked in adapter delivery.  Cancel and await them while adapters
            # are still alive so every watcher receives a retryable result
            # before platform teardown begins.
            cancel_completion_batches = getattr(
                self, "_cancel_process_completion_batch_tasks", None
            )
            if cancel_completion_batches is not None:
                await cancel_completion_batches()

            for platform, adapter in list(self.adapters.items()):
                await self._bounded_adapter_teardown(adapter, platform)

            logger.info(
                "Shutdown phase: all adapters disconnected at +%.2fs",
                _phase_elapsed(),
            )

            for _task in list(self._background_tasks):
                if _task is self._stop_task:
                    continue
                if _task is self._restart_task:
                    # The restart orchestration task is awaiting _stop_task
                    # right now; cancelling it would propagate CancelledError
                    # into this _stop_impl and skip _shutdown_event.set() /
                    # _exit_code = 75 (#12875).  It self-terminates anyway.
                    continue
                _task.cancel()
            self._background_tasks.clear()

            self.adapters.clear()
            for _session_key, _agent in self.sessions.running_items():
                self._release_turn_state(_session_key)
            # Flush pending messages to disk before clearing (#72680).
            # When FTS5 corruption prevents message persistence, the
            # in-memory pending text is the only surviving copy.  Clearing
            # without flushing causes permanent data loss.
            try:
                from gateway.shutdown_flush import flush_pending_to_file
                pending_commands = {
                    key: state.persistent.pending_command_text
                    for key, state in self.sessions.states.items()
                    if state.persistent.pending_command_text is not None
                }
                flush_pending_to_file(pending_commands, reason="shutdown")
            except Exception:
                pass
            # On the real runner these are live SessionState views whose
            # clear() resets one field per session — never a wholesale dict
            # swap, so a concurrent writer on another session can't lose its
            # entry.  Test fakes borrowing _stop_impl keep plain dicts.
            for state in self.sessions.states.values():
                state.turn.clear()
                state.persistent.pending_command_text = None
                state.persistent.approvals = None
            self._shutdown_event.set()

            # Global cleanup: kill any remaining tool subprocesses not tied
            # to a specific agent (catch-all for zombie prevention). On the
            # drain-timeout path we already did this earlier after agent
            # interrupt — this second call catches (a) the graceful path
            # where drain succeeded without interrupt, and (b) anything
            # that got respawned between the earlier call and adapter
            # disconnect (defense in depth; safe to call repeatedly).
            _kill_tool_subprocesses("final-cleanup")
            logger.info(
                "Shutdown phase: final-cleanup tool kill done at +%.2fs",
                _phase_elapsed(),
            )

            # Reap the process-global auxiliary-client cache once at the very
            # end of teardown.  Per-turn cleanup runs in _cleanup_agent_resources
            # for each active agent, but clients bound to worker-thread loops
            # that died with their ThreadPoolExecutor (notably cron ticks) only
            # get swept here.  Without this, long-running gateways accumulate
            # async httpx transports until they hit EMFILE on macOS's default
            # RLIMIT_NOFILE=256.  See #14210.
            try:
                from agent.auxiliary_client import shutdown_cached_clients
                shutdown_cached_clients()
            except Exception as _e:
                logger.debug("shutdown_cached_clients error: %s", _e)

            # Close SQLite session DBs so the WAL write lock is released.
            # Without this, --replace and similar restart flows leave the
            # old gateway's connection holding the WAL lock until Python
            # actually exits — causing 'database is locked' errors when
            # the new gateway tries to open the same file.
            # ``self`` holds the DB at ``_session_db`` (an AsyncSessionDB facade);
            # unwrap to the sync handle. ``session_store`` holds it at ``_db``.
            _self_db = getattr(self, "_session_db", None)
            _self_db = getattr(_self_db, "_db", _self_db)
            for _db in (_self_db, getattr(getattr(self, "session_store", None), "_db", None)):
                if _db is None or not hasattr(_db, "close"):
                    continue
                try:
                    _db.close()
                except Exception as _e:
                    logger.debug("SessionDB close error: %s", _e)
            self._shutdown_executor()
            logger.info(
                "Shutdown phase: SessionDB close done at +%.2fs",
                _phase_elapsed(),
            )

            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()

            # Write a clean-shutdown marker so the next startup knows this
            # wasn't a crash.  suspend_recently_active() only needs to run
            # after unexpected exits.  However, if the drain timed out and
            # agents were force-interrupted, their sessions may be in an
            # incomplete state (trailing tool response, no final assistant
            # message).  Skip the marker in that case so the next startup
            # suspends those sessions — giving users a clean slate instead
            # of resuming a half-finished tool loop.
            if not timed_out:
                try:
                    (_hermes_home / ".clean_shutdown").touch()
                except Exception:
                    pass
            else:
                logger.info(
                    "Skipping .clean_shutdown marker — drain timed out with "
                    "interrupted agents; next startup will suspend recently "
                    "active sessions."
                )

            # Track sessions that were active at shutdown for stuck-loop
            # detection (#7536).  On each restart, the counter increments
            # for sessions that were running.  If a session hits the
            # threshold (3 consecutive restarts while active), the next
            # startup auto-suspends it — breaking the loop.
            if active_agents:
                self._increment_restart_failure_counts(set(active_agents.keys()))

            if self._restart_requested and self._restart_command_source is None:
                try:
                    atomic_json_write(
                        _planned_restart_notification_path(),
                        {
                            "requested_at": time.time(),
                            "via_service": bool(self._restart_via_service),
                            "detached": bool(self._restart_detached),
                        },
                        indent=None,
                    )
                except Exception as e:
                    logger.debug("Failed to write planned restart notification marker: %s", e)

            if self._restart_requested and self._restart_via_service:
                self._launch_systemd_restart_shortcut()
                # Always exit with TEMPFAIL (75) on service-managed
                # restarts.  The shortcut helper above is best-effort and
                # commonly fails on real deployments: non-root gateway
                # units hit Polkit denials when invoking ``systemd-run
                # --system``, headless boxes have no user bus for
                # ``--user``, and operator-managed unit files may use
                # ``Restart=on-failure`` rather than ``Restart=always``.
                # Exit 75 paired with ``RestartForceExitStatus=75`` makes
                # systemd treat the planned restart as a controlled
                # failure and revive the unit via ``Restart=on-failure``,
                # regardless of whether the helper survived.  Without
                # this, a clean exit (0) on Linux left the gateway dead
                # until someone rebooted the host.  Only the planned code
                # (75) is whitelisted via ``RestartForceExitStatus``; a
                # genuine crash exits non-zero-but-not-75, so real crash
                # loops are still governed by the unit's normal
                # ``Restart=``/``RestartSec`` (and any StartLimit the
                # operator sets) rather than force-restarted here.
                self._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
                self._exit_reason = self._exit_reason or "Gateway restart requested"

            self._draining = False
            self._update_runtime_status("stopped", self._exit_reason)
            _shutdown_gateway_health_export(self)
            logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task
