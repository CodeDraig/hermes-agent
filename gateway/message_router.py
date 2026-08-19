"""Gateway inbound-message routing policy."""

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import agent.interruption as interruption
import agent.status_output as status_output
from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from gateway.config import Platform
from gateway.history import (
    _float_env,
    _message_timestamps_enabled,
    _stamp_hygiene_compression_provenance,
)
from gateway.hygiene import (
    _GATEWAY_HYGIENE_PLATFORM,
    _hygiene_cooldown_for_failure,
    _record_hygiene_cooldown,
    _reset_hygiene_failure_streak,
    _seed_hygiene_system_prompt,
    hygiene_compaction_recovered,
)
from gateway.media import (
    _build_document_context_note,
    _build_media_placeholder,
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_stt_input,
    _event_media_is_video,
)
from gateway.platforms.base import (
    EphemeralReply,
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.response_filters import (
    _gateway_platform_value,
    _is_gateway_hidden_reasoning_incomplete_turn,
    _normalize_empty_agent_response,
    _sanitize_gateway_final_response,
    _should_clear_resume_pending_after_turn,
)
from gateway.runtime_config import (
    _credential_pool_for_provider,
    _get_channel_override,
    _home_target_env_var,
    _load_gateway_config,
    _platform_config_key,
    _resolve_gateway_model,
    _resolve_runtime_agent_kwargs,
    _resolve_runtime_agent_kwargs_for_provider,
)
from gateway.session import (
    AsyncSessionStore,
    SessionEntry,
    SessionSource,
    build_session_context,
    build_session_key,
    is_shared_multi_user_session,
    neutralize_untrusted_inline_text,
)
from gateway.session_state import AGENT_PENDING as _AGENT_PENDING_SENTINEL
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_constants import get_hermes_home

STOP_REQUEST_REASON = "Stop requested"
_INTERRUPT_REASON_STOP = STOP_REQUEST_REASON
_INTERRUPT_REASON_RESET = "Session reset requested"
USER_BOUNDARY_END_REASONS = (
    "session_reset",
    "user_exit",
    "session_switch",
    "new_session",
)

_USER_BOUNDARY_END_REASONS = USER_BOUNDARY_END_REASONS
logger = logging.getLogger("gateway.run")
_hermes_home = get_hermes_home()


def _skill_slug_from_frontmatter(skill_md: Path) -> tuple[str | None, str | None]:
    """Derive the command slug and declared name from skill frontmatter."""
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    content = content.lstrip("\ufeff")
    if not content.startswith("---"):
        return None, None
    end = content.find("\n---", 3)
    if end < 0:
        return None, None
    declared_name = None
    for line in content[3:end].splitlines():
        line = line.strip()
        if line.startswith("name:"):
            raw = line.split(":", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            declared_name = raw.strip()
            break
    if not declared_name:
        return None, None
    slug = declared_name.lower().replace(" ", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return (slug or None), declared_name


def _has_platform_display_override(
    user_config: dict, platform_key: str, setting: str
) -> bool:
    """Return whether a platform explicitly sets one display option."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg

_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")

def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid command names.

    Telegram Bot API command names allow only lowercase letters, digits, and
    underscores.  Keep other platform renderings unchanged, but normalize
    Telegram help text so command mentions remain clickable/valid there.
    """
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)

def _resolve_gateway_display_bool(
    user_config: dict,
    platform_key: str,
    setting: str,
    *,
    default: bool = False,
    platform: Any = None,
    require_platform_override_for: set[Any] | None = None,
) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Some display features expose assistant scratch text rather than deliberate
    user-facing output.  For high-noise threaded chat surfaces such as
    Mattermost, a global opt-in is too broad: they must be enabled with an
    explicit display.platforms.<platform>.<setting> override.
    """
    current_platform = _gateway_platform_value(platform or platform_key)
    platform_only = {
        _gateway_platform_value(candidate)
        for candidate in (require_platform_override_for or set())
    }
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)
    ):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)

def _check_unavailable_skill(command_name: str) -> str | None:
    """Check if a command matches a known-but-inactive skill.

    Returns a helpful message if the skill exists but is disabled. Returns
    None if no match is found.

    The slug for each on-disk skill is derived from its frontmatter ``name:``
    (via :func:`_skill_slug_from_frontmatter`), NOT from its containing
    directory name — because the two can differ (e.g. directory
    ``stable-diffusion`` + frontmatter ``Stable Diffusion Image Generation``
    yields slug ``stable-diffusion-image-generation``). Matching on
    directory name would miss that slug entirely and fall through to the
    generic "unknown command" path.
    """
    # Normalize: command uses hyphens, skill names may use hyphens or underscores
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
        disabled = _get_disabled_skill_names()

        # Check disabled skills across all dirs (local + external)
        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, declared_name = _skill_slug_from_frontmatter(skill_md)
                if not slug or not declared_name:
                    continue
                # disabled is keyed by the declared frontmatter name (what
                # skills.disabled / skills.platform_disabled store).
                if slug == normalized and declared_name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`"
                    )

    except Exception:
        pass
    return None

class MessageRouter:
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})
    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0
    _BUSY_REJECT_TEXT: Dict[str, str] = {
        "model": "Agent is running — wait or /stop first, then switch models.",
        "codex-runtime": (
            "Agent is running — wait or /stop first, then change runtime."
        ),
        "moa": "Agent is running — wait or /stop first, then run /moa.",
    }

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        return build_session_key(
            source,
            group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
        )

    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return False
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only honor a real True from the SessionDB. Any other value
        # (including MagicMock instances from test fixtures that didn't
        # opt into topic mode) means topic mode is off for this chat.
        return raw is True

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        return tid in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        if not tid or tid in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return False
        return True

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one message per cooldown window.

        A user who forgets multi-session mode is enabled and types several
        prompts in the root DM would otherwise get a reminder for every
        message. Cap it so the first one lands and the rest stay quiet.
        """
        if not hasattr(self, "_telegram_lobby_reminder_ts"):
            self._telegram_lobby_reminder_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_lobby_reminder_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S:
            return False
        self._telegram_lobby_reminder_ts[chat_id] = now
        return True

    def _telegram_topic_root_lobby_message(self) -> str:
        return (
            "This main chat is reserved for system commands.\n\n"
            "To start a new Hermes chat, open the All Messages topic at the top "
            "of this bot interface and send any message there. Telegram will "
            "create a new topic for that message; each topic works as an "
            "independent Hermes session."
        )

    def _telegram_topic_root_new_message(self) -> str:
        return (
            "To start a new parallel Hermes chat, open the All Messages topic "
            "at the top of this bot interface and send any message there. "
            "Telegram will create a new topic for it.\n\n"
            "Each topic is an independent Hermes session. Use /new inside an "
            "existing topic only if you want to replace that topic's current session."
        )

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        if not self._is_telegram_topic_lane(source):
            return None
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        )

    def _record_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
    ) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
            user_id=str(source.user_id or ""),
            session_key=session_entry.session_key,
            session_id=session_entry.session_id,
        )

    def _sync_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
        *,
        reason: str,
    ) -> None:
        """Update the topic binding to point at ``session_entry.session_id``.

        Telegram topic lanes persist a (chat_id, thread_id) -> session_id row
        so reopening a topic in a fresh process resumes the right Hermes
        session. When compression rotates ``session_entry.session_id`` mid-turn,
        the binding goes stale and the next inbound message in that topic
        reloads the oversized parent transcript instead of the compressed
        child, retriggering preflight compression — sometimes in a loop
        (#20470, #29712, #33414).
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug(
                "telegram topic binding refresh failed (%s)", reason, exc_info=True,
            )

    def _recover_telegram_topic_thread_id(
        self,
        source: SessionSource,
    ) -> Optional[str]:
        """Pin DM-topic routing to the user's last-active topic.

        Telegram can omit ``message_thread_id`` or surface General (``1``)
        for some topic-mode DM replies. In those lobby-shaped cases, keep the
        conversation attached to the user's most-recent bound topic.

        Do not rewrite a non-lobby, previously-unbound thread id: a newly
        created Telegram DM topic is also "unknown" until the first inbound
        message is recorded, and rewriting it would send that brand-new topic's
        answer into an older lane. Returns None to leave the source alone.
        """
        if (
            source.platform != Platform.TELEGRAM
            or source.chat_type != "dm"
            or not source.chat_id
            or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        is_lobby = not inbound or inbound in self._TELEGRAM_GENERAL_TOPIC_IDS
        if not is_lobby:
            # A non-lobby, unknown thread_id is most likely the first message in
            # a brand-new Telegram DM topic. Preserve it so it can be recorded
            # as a new independent lane below instead of hijacking the latest
            # existing topic binding.
            return None
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id),
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        if not bindings:
            return None
        user_id = str(source.user_id)
        for b in bindings:  # newest-first
            if str(b.get("user_id") or "") == user_id:
                recovered = str(b.get("thread_id") or "")
                if recovered and recovered != inbound:
                    return recovered
                return None
        return None

    def _normalize_source_for_session_key(
        self,
        source: SessionSource,
    ) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes.

        ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
        ``_recover_telegram_topic_thread_id`` *before* deriving the session
        key for a normal message turn (a lobby/stripped reply gets pinned to
        the user's last-active topic).  Session-scoped command handlers like
        ``/model`` and ``/reasoning`` derive their override key from the raw
        inbound ``event.source``, which skips that recovery — so the override
        is stored under a different key than the next message turn reads,
        and the override is silently dropped on Telegram forum topics and
        after compression session splits (#30479).

        Returns a recovery-normalized copy when a rewrite applies, otherwise
        the original source unchanged.  Always derive the override storage key
        from the result so storage and read use an identical key.
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        if recovered is None:
            return source
        return dataclasses.replace(source, thread_id=recovered)

    def _resolve_session_agent_runtime(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session.

        Priority (highest first): session ``/model`` → ``channel_overrides`` →
        global config/env (``_resolve_gateway_model(user_config)`` and default
        provider resolution).
        """
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        model = _resolve_gateway_model(user_config)
        if resolved_session_key:
            self._rehydrate_session_model_override(resolved_session_key)
        _override_state = (
            self.sessions.peek(resolved_session_key)
            if resolved_session_key
            else None
        )
        override = (
            _override_state.conversation.model_override if _override_state else None
        )
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                "provider": override.get("provider"),
                "api_key": override.get("api_key"),
                "base_url": override.get("base_url"),
                "api_mode": override.get("api_mode"),
                "max_tokens": override.get("max_tokens"),
                "credential_pool": override.get("credential_pool"),
            }
            if override_runtime.get("api_key"):
                if override_runtime.get("credential_pool") is None:
                    override_runtime["credential_pool"] = _credential_pool_for_provider(
                        override.get("provider")
                    )
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    resolved_session_key or "", model, override_model,
                    override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # Override exists but has no api_key — fall through to env-based
            # resolution and apply model/provider from the override on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                resolved_session_key or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                resolved_session_key or "", model,
                [
                    _key
                    for _key, _st in list(self.sessions.states.items())
                    if _st.conversation.model_override is not None
                ][:5] or "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info(
                "Runtime provider supplied explicit model override: %s -> %s",
                model,
                runtime_model,
            )
            model = runtime_model

        cfg = getattr(self, "config", None)
        if cfg and source is not None:
            chat_id = str(source.chat_id) if source.chat_id else ""
            thread_id = (
                str(source.thread_id) if getattr(source, "thread_id", None) else None
            )
            parent_id = (
                str(source.parent_chat_id)
                if getattr(source, "parent_chat_id", None)
                else None
            )
            ch = _get_channel_override(
                cfg,
                source.platform,
                chat_id,
                thread_id=thread_id,
                parent_id=parent_id,
            )
            if ch:
                if ch.model:
                    model = ch.model
                if ch.provider:
                    runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(
                        ch.provider
                    )
                    ch_runtime_model = runtime_kwargs.pop("model", None)
                    # Only adopt the provider's bundled model when the override
                    # did not specify an explicit model.
                    if ch_runtime_model and not ch.model:
                        model = ch_runtime_model

        if override and resolved_session_key:
            model, runtime_kwargs = self._apply_session_model_override(
                resolved_session_key, model, runtime_kwargs
            )

        # When the config has no model.default but a provider was resolved
        # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
        # fall back to the provider's first catalog model so the API call
        # doesn't fail with "model must be a non-empty string".
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception:
                pass

        # Final safety net (#35314): if resolution still produced an empty
        # model — e.g. a transient config-cache miss during a post-interrupt
        # recovery turn returned an empty user_config — reuse the last model we
        # successfully resolved for this session (or, failing that, the most
        # recent one resolved process-wide). Building an agent with model=""
        # makes every API call fail HTTP 400 "No models provided" and the
        # session goes silent until the user manually re-sends. ``getattr``
        # guards against bare test runners built via ``object.__new__``.
        if not model:
            _lr_state = (
                self.sessions.peek(resolved_session_key)
                if resolved_session_key
                else None
            )
            _lr_star = self.sessions.peek("*")
            _recovered = (
                (_lr_state.conversation.last_resolved_model if _lr_state else "")
                or (_lr_star.conversation.last_resolved_model if _lr_star else "")
            )
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)",
                    resolved_session_key or "", _recovered,
                )
                model = _recovered
        elif model:
            # Cache the good resolution for future recovery turns.
            if resolved_session_key:
                self.sessions.state(
                    resolved_session_key
                ).conversation.last_resolved_model = model
            self.sessions.state("*").conversation.last_resolved_model = model

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Build the effective model/runtime config for a single turn.

        Always uses the session's primary model/provider.  If `/fast` is
        enabled and the model supports Priority Processing / Anthropic fast
        mode, attach `request_overrides` so the API call is marked
        accordingly.
        """
        from hermes_cli.models import resolve_fast_mode_overrides

        runtime = {
            "api_key": runtime_kwargs.get("api_key"),
            "base_url": runtime_kwargs.get("base_url"),
            "provider": runtime_kwargs.get("provider"),
            "requested_provider": runtime_kwargs.get("requested_provider"),
            "api_mode": runtime_kwargs.get("api_mode"),
            "command": runtime_kwargs.get("command"),
            "args": list(runtime_kwargs.get("args") or []),
            "credential_pool": runtime_kwargs.get("credential_pool"),
            "max_tokens": runtime_kwargs.get("max_tokens"),
        }
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model,
                runtime["provider"],
                runtime["requested_provider"],
                runtime["base_url"],
                runtime["api_mode"],
                runtime["command"],
                tuple(runtime["args"]),
            ),
        }

        service_tier = getattr(self, "_service_tier", None)
        if not service_tier:
            route["request_overrides"] = {}
            return route

        try:
            overrides = resolve_fast_mode_overrides(route["model"])
        except Exception:
            overrides = None
        route["request_overrides"] = overrides or {}
        return route

    def _sync_session_model_from_agent(self, session_id: str, agent: Any) -> None:
        """Persist the runtime model/provider actually used by a gateway turn.

        Provider fallback can switch ``agent.model``/``agent.provider`` after the
        session row was created. Keep the session DB metadata in sync so session
        lists, desktop/dashboard details, and follow-up session tooling report the
        backend that actually answered the latest turn.

        Called from the ``run_sync`` closure, which executes off the event loop
        in the executor thread — so the synchronous ``SessionDB`` (``_db``) is
        used directly rather than awaiting the AsyncSessionDB forwarder.
        """
        if not session_id or agent is None or self._session_db is None:
            return
        model = getattr(agent, "model", None)
        if not model:
            return
        runtime = {
            "provider": getattr(agent, "provider", None),
            "base_url": getattr(agent, "base_url", None),
            "api_mode": getattr(agent, "api_mode", None),
            "fallback_active": bool(getattr(agent, "_fallback_activated", False)),
        }
        runtime = {k: v for k, v in runtime.items() if v not in (None, "")}

        try:
            db = self._session_db._db
            row = db.get_session(session_id)
            if not row:
                return
            current_model = row.get("model")
            raw_config = row.get("model_config")
            try:
                config = json.loads(raw_config) if raw_config else {}
            except Exception:
                config = {}
            if not isinstance(config, dict):
                config = {}
            gateway_runtime = dict(config.get("gateway_runtime") or {})
            if current_model == model and all(
                gateway_runtime.get(k) == v for k, v in runtime.items()
            ):
                return
            config["gateway_runtime"] = runtime
            db.update_session_meta(session_id, json.dumps(config), model=model)
        except Exception:
            logger.debug("Failed to sync gateway session model metadata", exc_info=True)

    async def _deliver_platform_notice(self, source, content: str) -> None:
        """Deliver a setup/operational notice using platform-specific privacy rules."""
        adapter = self._adapter_for_source(source)
        if not adapter:
            return

        config = getattr(self, "config", None)
        notice_delivery = "public"
        if config and hasattr(config, "get_notice_delivery"):
            notice_delivery = config.get_notice_delivery(source.platform)

        metadata = self._thread_metadata_for_source(source)
        if notice_delivery == "private" and getattr(source, "user_id", None):
            try:
                result = await adapter.send_private_notice(
                    source.chat_id,
                    source.user_id,
                    content,
                    metadata=metadata,
                )
                if getattr(result, "success", False):
                    return
            except Exception:
                logger.debug(
                    "[%s] send_private_notice failed, falling back to public",
                    getattr(source, "platform", "?"),
                    exc_info=True,
                )

        await adapter.send(source.chat_id, content, metadata=metadata)

    async def _resolve_async_delegation_session(
        self,
        session_entry: SessionEntry,
        pinned_session_id: str,
    ) -> Optional[SessionEntry]:
        """Resolve an async completion to its verified owning gateway session.

        A compression rotation ends the physical parent row while continuing
        the same logical conversation in a child.  Follow that lineage, but
        never let a late completion override an unrelated /new or restored
        route.  Unknown ownership remains fail-closed; the result is still
        available in the delegation records.
        """
        session_db = cast(Any, self._session_db)
        if session_db is None:
            logger.warning(
                "Async-delegation completion has no session database; "
                "dropping injection (#55578 fail-closed)."
            )
            return None

        pinned_row = None
        try:
            pinned_row = await session_db.get_session(pinned_session_id)
        except Exception:
            logger.debug(
                "Async-delegation parent lookup failed for %s",
                pinned_session_id,
                exc_info=True,
            )

        if pinned_row is None:
            logger.warning(
                "Async-delegation completion has unknown spawning session %s; "
                "dropping injection (#55578 fail-closed).",
                pinned_session_id,
            )
            return None

        target_session_id = pinned_session_id
        follows_compression = False
        if pinned_row.get("ended_at"):
            _end_reason = str(pinned_row.get("end_reason") or "")
            if _end_reason in _USER_BOUNDARY_END_REASONS:
                logger.warning(
                    "Async-delegation completion pinned to user-closed session %s "
                    "(end_reason=%r); dropping injection instead of resurrecting it "
                    "(#55578 fail-closed).",
                    pinned_session_id,
                    _end_reason,
                )
                return None
            if _end_reason != "compression":
                # Idle/timeout/lifecycle end: the chat
                # route remains valid and ``session_entry`` IS the routing
                # key's current session for this same chat, so deliver the
                # finished work there instead of dropping it. This is the
                # delivery leg _classify_completion_target promises when it
                # returns "deliver" for non-boundary ends — without it the
                # pre-flight verdict and this resolver disagree, and the
                # durable row is acked at adapter acceptance then silently
                # dropped here (falsely-acknowledged permanent loss;
                # staging incident 2026-08-09 defect #2).
                logger.info(
                    "Async-delegation completion pinned to %s-ended session %s; "
                    "retargeting to the chat's current session %s.",
                    _end_reason or "idle",
                    pinned_session_id,
                    session_entry.session_id,
                )
                return session_entry

            follows_compression = True
            try:
                target_session_id = await session_db.get_compression_tip(
                    pinned_session_id
                )
            except Exception:
                logger.debug(
                    "Async-delegation compression-tip lookup failed for %s",
                    pinned_session_id,
                    exc_info=True,
                )
                target_session_id = None

            if not target_session_id or target_session_id == pinned_session_id:
                logger.warning(
                    "Async-delegation completion pinned to compressed session %s "
                    "without a continuation; dropping injection.",
                    pinned_session_id,
                )
                return None

            try:
                tip_row = await session_db.get_session(target_session_id)
            except Exception:
                tip_row = None
            if tip_row is None or tip_row.get("ended_at"):
                logger.warning(
                    "Async-delegation compression continuation %s is %s; "
                    "dropping injection.",
                    target_session_id,
                    "unknown" if tip_row is None else "ended",
                )
                return None

            route_owns_lineage = session_entry.session_id in {
                pinned_session_id,
                target_session_id,
            }
            if not route_owns_lineage:
                # A long-running delegation may survive multiple compression
                # rotations.  Accept an intermediate stale route only when its
                # own verified compression tip is the same live target.
                try:
                    route_row = await session_db.get_session(session_entry.session_id)
                    route_tip = (
                        await session_db.get_compression_tip(session_entry.session_id)
                        if route_row is not None
                        and route_row.get("ended_at")
                        and route_row.get("end_reason") == "compression"
                        else None
                    )
                except Exception:
                    route_tip = None
                route_owns_lineage = route_tip == target_session_id

            if not route_owns_lineage:
                logger.warning(
                    "Async-delegation completion for compression lineage %s -> %s "
                    "does not own current route %s; dropping injection.",
                    pinned_session_id,
                    target_session_id,
                    session_entry.session_id,
                )
                return None

        if target_session_id == session_entry.session_id:
            return session_entry

        prior_session_id = session_entry.session_id
        if follows_compression:
            switched = await self.async_session_store.advance_compression_session(
                session_entry.session_key,
                prior_session_id,
                target_session_id,
            )
        else:
            switched = await self.async_session_store.switch_session(
                session_entry.session_key,
                target_session_id,
            )
        if switched is None:
            logger.warning(
                "Async-delegation completion could not bind routing key %s to "
                "owning session %s; dropping injection.",
                session_entry.session_key,
                target_session_id,
            )
            return None

        logger.info(
            "Pinned async-delegation completion to owning session %s "
            "(was %s) for routing key %s (#57498)",
            target_session_id,
            prior_session_id,
            session_entry.session_key,
        )
        return switched

    async def _dispatch_busy_slash_command(
        self, event: MessageEvent, cmd_def, quick_key: str, source,
    ):
        """Dispatch a recognized slash command while an agent is running.

        Resolution order:
          1. ``busy_handler`` — special mid-run variant (e.g. /goal's
             control-verb whitelist, /queue's FIFO enqueue, /model's
             custom reject text).
          2. ``busy_policy == "dispatch"`` — the command's normal handler.
          3. Catch-all busy-reject text. Rejecting is required rather than
             falling through to interrupt + discard: commands like /model,
             /reasoning, /voice, /insights, /title, /resume, /retry,
             /undo, /compress, /usage, /reload-mcp, /sethome, /reset (all
             registered as Discord slash commands) would interrupt the
             agent AND get silently discarded by the slash-command safety
             net, producing a zero-char response. See #5057, #6252, #10370.
        """
        name = cmd_def.name
        policy = getattr(cmd_def, "busy_policy", "reject")
        handler_key = getattr(cmd_def, "busy_handler", None)

        if handler_key:
            special = {
                "start": self._busy_start_command,
                "stop": self._busy_stop_command,
                "new": self._busy_new_command,
                "queue": self._busy_queue_command,
                "steer": self._busy_steer_command,
                "egress": self._busy_egress_command,
                "goal": self._busy_goal_command,
                "loop": self._busy_loop_command,
            }.get(handler_key)
            if special is not None:
                return await special(event, quick_key, source)
            reject_text = self._BUSY_REJECT_TEXT.get(handler_key)
            if reject_text is not None:
                return reject_text

        if policy in ("dispatch", "interrupt_then_dispatch"):
            plain = {
                "status": self._handle_status_command,
                "context": self._handle_context_command,
                "restart": self._handle_restart_command,
                "approve": self._handle_approve_command,
                "deny": self._handle_deny_command,
                "pause": self._handle_pause_command,
                "agents": self._handle_agents_command,
                "background": self._handle_background_command,
                "kanban": self._handle_kanban_command,
                "subgoal": self._handle_subgoal_command,
                "heartbeat": self._handle_heartbeat_command,
                "yolo": self._handle_yolo_command,
                "verbose": self._handle_verbose_command,
                "footer": self._handle_footer_command,
                "help": self._handle_help_command,
                "commands": self._handle_commands_command,
                "profile": self._handle_profile_command,
                "update": self._handle_update_command,
                "version": self._handle_version_command,
            }.get(name)
            if plain is not None:
                return await plain(event)
            logger.warning(
                "busy_policy=%s for /%s has no mid-run handler — "
                "falling back to busy-reject", policy, name,
            )

        # Catch-all: any other recognized slash command reached the
        # running-agent guard. Reject gracefully rather than falling
        # through to interrupt + discard.
        return (
            f"⏳ Agent is running — `/{name}` can't run "
            f"mid-turn. Wait for the current response or `/stop` first."
        )

    async def _handle_pause_command(self, event: MessageEvent):
        """`/pause [reason]` engages the global emergency stop; `/pause off`
        (aliases: resume/stop) lifts it.

        This is the in-band resume path for messaging-only operators — the
        estop gate above deliberately lets recognized slash commands through
        while paused so a user without host-shell access is never locked out.
        """
        from agent import estop

        args = (event.get_command_args() or "").strip()
        if args.lower() in {"off", "resume", "stop", "disengage"}:
            if estop.disengage():
                return "▶️ Resumed — new work is accepted again."
            return "Hermes wasn't paused."
        state = estop.get_state()
        if state is not None and not args:
            reason = state.get("reason")
            suffix = f" (reason: {reason})" if reason else ""
            return (
                f"⏸️ Hermes is already paused{suffix}. "
                "Use `/pause off` to resume."
            )
        estop.engage(reason=args or None)
        suffix = f" (reason: {args})" if args else ""
        return (
            f"⏸️ Paused{suffix}. New cron/kanban/gateway work is on hold; "
            "in-flight work finishes normally. Use `/pause off` to resume."
        )

    async def _busy_start_command(self, event: MessageEvent, quick_key: str, source):
        # Telegram sends /start for bot launches/deep-links. Treat it as a
        # platform ping, not a user command: no help dump, no agent
        # interrupt, no queued text.
        logger.info("Ignoring /start platform ping for active session %s", quick_key)
        return ""

    async def _busy_egress_command(self, event: MessageEvent, quick_key: str, source):
        from hermes_cli.proxy_cli import format_status_text

        return format_status_text()

    async def _busy_stop_command(self, event: MessageEvent, quick_key: str, source):
        # /stop must hard-kill the session when an agent is running.
        # A soft interrupt (interruption.interrupt(agent)) doesn't help when the agent
        # is truly hung — the executor thread is blocked and never checks
        # _interrupt_requested.  Force-clean _running_agents so the session
        # is unlocked and subsequent messages are processed normally.
        await self._interrupt_and_clear_session(
            quick_key,
            source,
            interrupt_reason=_INTERRUPT_REASON_STOP,
            invalidation_reason="stop_command",
        )
        logger.info("STOP for session %s — agent interrupted, session lock released", quick_key)
        return EphemeralReply(t("gateway.stop.stopped"))

    async def _busy_new_command(self, event: MessageEvent, quick_key: str, source):
        # /reset and /new must bypass the running-agent guard so they
        # actually dispatch as commands instead of being queued as user
        # text (which would be fed back to the agent with the same
        # broken history — #2170).  Interrupt the agent first, then
        # clear the adapter's pending queue so the stale "/reset" text
        # doesn't get re-processed as a user message after the
        # interrupt completes.
        # Clear any pending messages so the old text doesn't replay
        await self._interrupt_and_clear_session(
            quick_key,
            source,
            interrupt_reason=_INTERRUPT_REASON_RESET,
            invalidation_reason="new_command",
        )
        # Clean up the running agent entry so the reset handler
        # doesn't think an agent is still active.
        return await self._handle_reset_command(event)

    async def _busy_queue_command(self, event: MessageEvent, quick_key: str, source):
        # /queue <prompt> — queue without interrupting.
        # Semantics: each /queue invocation produces its own full agent
        # turn, processed in FIFO order after the current run (and any
        # earlier /queue items) finishes.  Messages are NOT merged.
        queued_text = event.get_command_args().strip()
        # Preserve media/reply payloads: a /queue carrying a photo,
        # document, or reply context is valid even with no prompt text
        # (e.g. "/queue" as the caption of an image). Dropping these
        # fields silently lost the attachment when the queued turn ran.
        has_media = bool(getattr(event, "media_urls", None))
        if not queued_text and not has_media:
            return "Usage: /queue <prompt>"
        adapter = self._adapter_for_source(source)
        if adapter:
            queued_event = MessageEvent(
                text=queued_text,
                message_type=event.message_type if has_media else MessageType.TEXT,
                source=event.source,
                raw_message=event.raw_message,
                message_id=event.message_id,
                media_urls=list(getattr(event, "media_urls", []) or []),
                media_types=list(getattr(event, "media_types", []) or []),
                reply_to_message_id=event.reply_to_message_id,
                reply_to_text=event.reply_to_text,
                reply_to_author_id=event.reply_to_author_id,
                reply_to_author_name=event.reply_to_author_name,
                reply_to_is_own_message=event.reply_to_is_own_message,
                auto_skill=event.auto_skill,
                channel_prompt=event.channel_prompt,
                channel_context=event.channel_context,
                internal=event.internal,
                timestamp=event.timestamp,
            )
            self.sessions.enqueue_fifo(quick_key, queued_event, adapter)
        depth = self.sessions.queue_depth(quick_key, adapter=self._adapter_for_source(source))
        if depth <= 1:
            return "Queued for the next turn."
        return f"Queued for the next turn. ({depth} queued)"

    async def _busy_steer_command(self, event: MessageEvent, quick_key: str, source):
        # /steer <prompt> — inject mid-run after the next tool call.
        # Unlike /queue (turn boundary), /steer lands BETWEEN tool-call
        # iterations inside the same agent run, by appending to the
        # last tool result's content. No interrupt, no new user turn,
        # no role-alternation violation.
        steer_text = event.get_command_args().strip()
        if not steer_text:
            return "Usage: /steer <prompt>"
        _steer_state = self.sessions.peek(quick_key)
        running_agent = _steer_state.turn.agent if _steer_state else None
        if running_agent is _AGENT_PENDING_SENTINEL:
            # Agent hasn't started yet — queue as turn-boundary fallback.
            adapter = self._adapter_for_source(source)
            if adapter:
                queued_event = MessageEvent(
                    text=steer_text,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                    channel_context=event.channel_context,
                )
                self.sessions.enqueue_fifo(quick_key, queued_event, adapter)
            return "Agent still starting — /steer queued for the next turn."
        if running_agent:
            try:
                accepted = interruption.steer(running_agent, steer_text)
            except Exception as exc:
                logger.warning("Steer failed for session %s: %s", quick_key, exc)
                return f"⚠️ Steer failed: {exc}"
            if accepted:
                preview = steer_text[:60] + ("..." if len(steer_text) > 60 else "")
                return f"⏩ Steer queued — arrives after the next tool call: '{preview}'"
            return "Steer rejected (empty payload)."
        # Running agent is missing or lacks steer() — fall back to queue.
        adapter = self._adapter_for_source(source)
        if adapter:
            queued_event = MessageEvent(
                text=steer_text,
                message_type=MessageType.TEXT,
                source=event.source,
                message_id=event.message_id,
                channel_prompt=event.channel_prompt,
                channel_context=event.channel_context,
            )
            self.sessions.enqueue_fifo(quick_key, queued_event, adapter)
        return "No active agent — /steer queued for the next turn."

    async def _busy_goal_command(self, event: MessageEvent, quick_key: str, source):
        # /goal is safe mid-run for status/pause/clear/wait (inspection
        # and control-plane only — doesn't interrupt the running turn).
        # Setting a new goal text mid-run is rejected with the same
        # "wait or /stop" message as /model so we don't race a second
        # continuation prompt against the current turn.
        _goal_arg = (event.get_command_args() or "").strip().lower()
        _goal_verb = _goal_arg.split(None, 1)[0] if _goal_arg else ""
        # Exact-match control verbs (unchanged semantics), plus the
        # wait/unwait barrier verbs which take a pid argument and the
        # gate management verb (inspection/mutation of the gate list only —
        # gates run at turn boundary, so editing them mid-run is safe).
        _is_control = (
            not _goal_arg
            or _goal_arg in {"status", "pause", "resume", "clear", "stop", "done", "unwait"}
            or _goal_verb in {"wait", "gate"}
        )
        if _is_control:
            return await self._handle_goal_command(event)
        return "Agent is running — use /goal status / pause / clear / wait mid-run, or /stop before setting a new goal."

    async def _busy_loop_command(self, event: MessageEvent, quick_key: str, source):
        # /loop mirrors /goal: control verbs are safe mid-run (state
        # only — read at the next idle boundary); setting a new loop
        # mid-run is rejected so we don't race the current turn.
        _loop_arg = (event.get_command_args() or "").strip().lower()
        if not _loop_arg or _loop_arg in {"status", "pause", "resume", "stop", "clear", "cancel", "help", "--help", "-h"}:
            return await self._handle_loop_command(event)
        return "Agent is running — use /loop status / pause / stop mid-run, or /stop before setting a new loop."

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.

        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source

        # 🔴 Cross-session leak guard. This handler runs inside a per-message
        # asyncio task created via create_task(), which snapshots the spawning
        # context with copy_context(). If a *concurrent* message had already
        # bound its session via set_session_vars() when this task was created,
        # we inherited ITS HERMES_SESSION_* ContextVars. Until we bind our own
        # (a few steps down, in _set_session_env), any subprocess spawned here
        # would read the foreign session's identity via the subprocess-env
        # bridge — the _UNSET-strip guard there can't help because the vars are
        # set-to-foreign, not _UNSET. Reset to _UNSET now so that window strips
        # safe (no session) instead of leaking the sibling's. See
        # gateway/session_context.reset_session_vars + the inheritance test.
        try:
            from gateway.session_context import reset_session_vars
            reset_session_vars()
        except Exception:
            logger.debug("reset_session_vars failed at handler entry", exc_info=True)

        # Internal events (e.g. background-process completion notifications)
        # are system-generated and must skip user authorization.
        is_internal = bool(getattr(event, "internal", False))

        if (
            getattr(self, "_startup_restore_in_progress", False)
            and not is_internal
            and not getattr(event, "_hermes_startup_restore_replay", False)
        ):
            self._queue_startup_restore_event(event)
            return None

        # Fire pre_gateway_dispatch plugin hook for user-originated messages.
        # Plugins receive the MessageEvent and may return a dict influencing flow:
        #   {"action": "skip",    "reason": ...}    -> drop (no reply, plugin handled)
        #   {"action": "rewrite", "text":  ...}     -> replace event.text, continue
        #   {"action": "allow"}   /   None          -> normal dispatch
        # Hook runs BEFORE auth so plugins can handle unauthorized senders
        # (e.g. customer handover ingest) without triggering the pairing flow.
        if not is_internal:
            try:
                from hermes_cli.lifecycle import invoke_hook as _invoke_hook
                _hook_results = _invoke_hook(
                    "pre_gateway_dispatch",
                    event=event,
                    gateway=self,
                    # getattr: bare-runner tests build GatewayRunner via
                    # object.__new__ without __init__ (pitfall #17), and the
                    # hook must not fail dispatch over a missing attribute.
                    session_store=getattr(self, "session_store", None),
                )
            except Exception as _hook_exc:
                logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
                _hook_results = []

            for _result in _hook_results:
                if not isinstance(_result, dict):
                    continue
                _action = _result.get("action")
                if _action == "skip":
                    logger.info(
                        "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                        _result.get("reason"),
                        source.platform.value if source.platform else "unknown",
                        source.chat_id or "unknown",
                    )
                    return None
                if _action == "rewrite":
                    _new_text = _result.get("text")
                    if isinstance(_new_text, str):
                        event = dataclasses.replace(event, text=_new_text)
                        source = event.source
                    break
                if _action == "allow":
                    break

        if is_internal:
            pass
        elif source.user_id is None:
            # Messages with no user identity (Telegram service messages,
            # channel forwards, anonymous admin posts, sender_chat) can't
            # be paired, but they can still be authorized via a
            # chat-scoped allowlist (e.g. TELEGRAM_GROUP_ALLOWED_CHATS
            # authorizes every member of the listed chat regardless of
            # sender). Defer to _is_user_authorized so that path runs.
            if not self._is_user_authorized(source):
                logger.debug("Ignoring message with no user_id from %s", source.platform.value)
                return None
        elif not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if (
                source.chat_type == "dm"
                and self._get_unauthorized_dm_behavior(
                    source.platform,
                    profile=source.profile,
                )
                == "pair"
            ):
                platform_name = source.platform.value if source.platform else "unknown"
                pairing_store = self._pairing_store_for(source)
                if pairing_store is None:
                    logger.error(
                        "Cannot offer pairing code on %s: no pairing store",
                        platform_name,
                    )
                    return None
                # Rate-limit ALL pairing responses (code or rejection) to
                # prevent spamming the user with repeated messages when
                # multiple DMs arrive in quick succession.
                if pairing_store._is_rate_limited(platform_name, source.user_id):
                    return None
                code = pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        store_profile = getattr(pairing_store, "profile", None)
                        profile_arg = (
                            f"-p {store_profile} "
                            if isinstance(store_profile, str)
                            and store_profile
                            and store_profile != "default"
                            else ""
                        )
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes {profile_arg}pairing approve "
                            f"{platform_name} {code}`"
                        )
                else:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
                    # Record rate limit so subsequent messages are silently ignored
                    pairing_store._record_rate_limit(platform_name, source.user_id)
            return None

        # Global emergency stop (`hermes pause`): give new turns a brief
        # paused notice instead of starting an agent run. Internal events
        # (background-process completions from IN-FLIGHT work) bypass the
        # gate — pause stops NEW work, it never kills or orphans running
        # work. Placed after auth so unauthorized senders keep the normal
        # silent/pairing behavior and can't probe pause state.
        #
        # Passthroughs (pause blocks new AGENT turns, not control traffic):
        #   * recognized slash commands — /status, /help, /new, /approve and
        #     friends must keep working while paused, and /pause off is the
        #     in-band resume path for messaging-only users;
        #   * replies owned by IN-FLIGHT work — a pending detached-update
        #     prompt, clarify, slash-confirm, or dangerous-command approval,
        #     plus any message steering a session whose agent is already
        #     running. Swallowing those would stall work the pause promised
        #     not to touch.
        if not is_internal:
            try:
                from agent.estop import paused_reply as _estop_paused_reply
                _paused_notice = _estop_paused_reply()
            except ImportError:
                _paused_notice = None
            if _paused_notice is not None:
                _estop_allow = False
                _estop_cmd = None
                try:
                    _estop_cmd = event.get_command()
                except Exception:
                    _estop_cmd = None
                if _estop_cmd:
                    try:
                        from hermes_cli.commands import (
                            resolve_command as _resolve_estop_cmd,
                        )
                        _estop_allow = _resolve_estop_cmd(_estop_cmd) is not None
                    except Exception:
                        _estop_allow = False
                if not _estop_allow:
                    try:
                        _estop_key = self._session_key_for_source(source)
                        _estop_state = self.sessions.peek(_estop_key)
                        if (
                            _estop_state is not None
                            and _estop_state.persistent.update_prompt_pending
                        ):
                            _estop_allow = True
                        if not _estop_allow and self.sessions.is_running(_estop_key):
                            # Steering / interrupting in-flight work (which
                            # also covers pending clarify + tool approvals
                            # held by the running agent).
                            _estop_allow = True
                        if not _estop_allow:
                            from tools import slash_confirm as _estop_confirm_mod
                            if _estop_confirm_mod.get_pending(_estop_key):
                                _estop_allow = True
                        if not _estop_allow:
                            from tools.approval import (
                                has_blocking_approval as _estop_has_approval,
                            )
                            if _estop_has_approval(_estop_key):
                                _estop_allow = True
                    except Exception:
                        pass
                if not _estop_allow:
                    logger.info(
                        "Gateway turn paused by global emergency stop (platform=%s chat=%s)",
                        getattr(getattr(source, "platform", None), "value", "unknown"),
                        getattr(source, "chat_id", None) or "unknown",
                    )
                    return _paused_notice

        # Intercept messages that are responses to a pending /update prompt.
        # The update process (detached) wrote .update_prompt.json; the watcher
        # forwarded it to the user; now the user's reply goes back via
        # .update_response so the update process can continue.
        #
        # IMPORTANT: recognized slash commands must bypass this interception.
        # Otherwise control/session commands like /new or /help get silently
        # consumed as update answers instead of being dispatched normally.
        _quick_key = self._session_key_for_source(source)
        allow_gateway_control = event.allow_gateway_control
        _up_state = self.sessions.peek(_quick_key)
        if (
            allow_gateway_control
            and _up_state is not None
            and _up_state.persistent.update_prompt_pending
        ):
            raw = (event.text or "").strip()
            # Accept /approve and /deny as shorthand for yes/no
            cmd = event.get_command()
            if cmd in {"approve", "yes"}:
                response_text = "y"
            elif cmd in {"deny", "no"}:
                response_text = "n"
            else:
                _recognized_cmd = None
                if cmd:
                    try:
                        from hermes_cli.commands import resolve_command as _resolve_update_cmd
                    except Exception:
                        _resolve_update_cmd = None
                    if _resolve_update_cmd is not None:
                        try:
                            _cmd_def = _resolve_update_cmd(cmd)
                            _recognized_cmd = _cmd_def.name if _cmd_def else None
                        except Exception:
                            _recognized_cmd = None
                if _recognized_cmd:
                    response_text = ""
                else:
                    response_text = raw
            if response_text:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text(response_text, encoding="utf-8")
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("Failed to write update response: %s", e)
                    return f"✗ Failed to send response to update process: {e}"
                _up_state.persistent.update_prompt_pending = False
                label = response_text if len(response_text) <= 20 else response_text[:20] + "…"
                return f"✓ Sent `{label}` to the update process."
            # Recognized slash command during a pending update prompt:
            # unblock the detached update subprocess by writing a blank
            # response so ``_gateway_prompt`` returns the prompt's default
            # (typically a safe "n" / skip) and exits cleanly instead of
            # blocking on stdin until the 30-minute watcher timeout.
            # The slash command then falls through to normal dispatch.
            if _recognized_cmd:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text("", encoding="utf-8")
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                    logger.info(
                        "Recognized /%s during pending update prompt for %s; "
                        "cancelled prompt with default and dispatching command",
                        _recognized_cmd,
                        _quick_key,
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to write cancel response for pending update prompt: %s",
                        e,
                    )
                _up_state.persistent.update_prompt_pending = False

        # Intercept messages that are responses to a pending clarify.
        # Open-ended prompts and "Other" responses are captured as free text;
        # direct replies to multi-choice prompts are accepted too ("2" maps
        # to the second option). Slash
        # commands still bypass this path so /stop and friends keep working.
        _clarify_mod = None
        try:
            from tools import clarify_gateway as _clarify_mod
            _pending_clarify = _clarify_mod.get_pending_for_session(
                _quick_key, include_choice_prompts=True,
            )
        except Exception:
            _pending_clarify = None
        if (
            allow_gateway_control
            and _pending_clarify is not None
            and _clarify_mod is not None
        ):
            _clarify_has_audio = bool(self._pending_event_audio_paths(event))
            _raw_clarify_reply = await self._prepare_clarify_reply_text(event)
            if _clarify_has_audio and not _raw_clarify_reply:
                logger.info(
                    "Gateway retained pending clarify after voice transcription "
                    "produced no usable text (session=%s, id=%s)",
                    _quick_key,
                    _pending_clarify.clarify_id,
                )
                return ""
            # Skip slash commands — the user clearly wanted to issue a
            # command, not answer the clarify.  Leave the clarify pending
            # so the user can retry; if it times out, the agent unblocks
            # with an empty response.
            if _raw_clarify_reply and not _raw_clarify_reply.startswith("/"):
                _text_outcome = _clarify_mod.attempt_text_response_for_session(
                    _quick_key, _raw_clarify_reply,
                )
                if _text_outcome == _clarify_mod.TEXT_RESOLVED:
                    logger.info(
                        "Gateway intercepted clarify text response (session=%s, id=%s)",
                        _quick_key, _pending_clarify.clarify_id,
                    )
                    # The clarify callback pauses the platform typing/status
                    # indicator while waiting so Slack users can type their
                    # answer. The active agent resumes as soon as this reply
                    # resolves the wait, so re-enable its indicator here too.
                    # Without this, Slack stays silent until the independent
                    # long-running heartbeat fires (three minutes by default).
                    _clarify_adapter = self._adapter_for_source(source)
                    if _clarify_adapter:
                        try:
                            _clarify_adapter.resume_typing_for_chat(source.chat_id)
                        except Exception:
                            logger.debug(
                                "Failed to resume typing after clarify response",
                                exc_info=True,
                            )
                    # Acknowledge with empty string so adapters that emit
                    # the agent's response don't double-post.  The agent
                    # itself will produce the next user-facing message.
                    return ""
                if _text_outcome == _clarify_mod.TEXT_REJECTED_SELECTION:
                    # Selection-shaped but invalid (out-of-range number,
                    # unrecognised comma-list). Keep the clarify armed so
                    # the user can retry — do not cancel and do not treat
                    # this as an unrelated follow-up turn.
                    logger.info(
                        "Gateway retained pending clarify after invalid "
                        "selection attempt (session=%s, id=%s)",
                        _quick_key, _pending_clarify.clarify_id,
                    )
                    return ""
                if _text_outcome == _clarify_mod.TEXT_REJECTED_PROSE:
                    # Native-choice prompts deliberately reject unmatched
                    # prose so it can continue through normal busy-message
                    # routing. Release this clarify first: redirect()
                    # degrades to steer() while tools are executing, and
                    # that steer cannot drain until the clarify tool returns.
                    _clarify_mod.resolve_gateway_clarify(
                        _pending_clarify.clarify_id,
                        "",
                    )

        # Intercept messages that are responses to a pending /reload-mcp
        # (or future) slash-confirm prompt.  Recognized confirm replies are
        # /approve, /always, /cancel (plus short aliases).  Anything else
        # falls through to normal dispatch — a stale pending confirm does
        # NOT block other commands.
        #
        # Important: if a dangerous-command approval is ALSO pending (agent
        # blocked inside tools/approval.py), the tool approval takes
        # precedence — /approve there unblocks the waiting tool thread.
        # Slash-confirm only catches /approve when no tool approval is live.
        from tools import slash_confirm as _slash_confirm_mod
        _pending_confirm = _slash_confirm_mod.get_pending(_quick_key)
        _tool_approval_live = False
        try:
            from tools.approval import has_blocking_approval
            _tool_approval_live = has_blocking_approval(_quick_key)
        except Exception:
            _tool_approval_live = False
        if allow_gateway_control and _pending_confirm and not _tool_approval_live:
            _raw_reply = (event.text or "").strip()
            # Accept bang-prefixed replies (`!always`, `!cancel`) verbatim.
            # Slack/Matrix instruction text shows the `!` prefix (typed `/`
            # is blocked in Slack threads), but the adapters only rewrite
            # `!<known-command>` — `always`/`cancel` are confirm keywords,
            # not registered commands, so the `!` survives to here.
            _norm_reply = _raw_reply.lstrip("!/").lower()
            _cmd_reply = event.get_command()
            _confirm_choice = None
            if _cmd_reply in {"approve", "yes", "ok", "confirm"}:
                _confirm_choice = "once"
            elif _cmd_reply in {"always", "remember"}:
                _confirm_choice = "always"
            elif _cmd_reply in {"cancel", "no", "deny", "nevermind"}:
                _confirm_choice = "cancel"
            elif _norm_reply in {"approve", "approve once", "once"}:
                _confirm_choice = "once"
            elif _norm_reply in {"always", "always approve"}:
                _confirm_choice = "always"
            elif _norm_reply in {"cancel", "nevermind", "no"}:
                _confirm_choice = "cancel"
            if _confirm_choice is not None:
                _resolved = await _slash_confirm_mod.resolve(
                    _quick_key, _pending_confirm.get("confirm_id"), _confirm_choice,
                )
                return _resolved or ""
            # Stale pending + unrelated command: drop the pending state so
            # the confirm doesn't block normal usage indefinitely.  The user
            # clearly moved on.
            _slash_confirm_mod.clear_if_stale(_quick_key)

        # PRIORITY handling when an agent is already running for this session.
        # Default behavior is to interrupt immediately so user text/stop messages
        # are handled with minimal latency.
        #
        # Special case: Telegram/photo bursts often arrive as multiple near-
        # simultaneous updates. Do NOT interrupt for photo-only follow-ups here;
        # let the adapter-level batching/queueing logic absorb them.

        # Staleness eviction: detect leaked locks from hung/crashed handlers.
        # With inactivity-based timeout, active tasks can run for hours, so
        # wall-clock age alone isn't sufficient.  Evict only when the agent
        # has been *idle* beyond the inactivity threshold (or when the agent
        # object has no activity tracker and wall-clock age is extreme).
        _raw_stale_timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _quick_state = self.sessions.peek(_quick_key)
        _stale_ts = _quick_state.turn.started_ts if _quick_state else 0
        if _quick_state is not None and _quick_state.turn.agent is not None and _stale_ts:
            _stale_age = time.time() - _stale_ts
            _stale_agent = _quick_state.turn.agent
            # Never evict the pending sentinel — it was just placed moments
            # ago during the async setup phase before the real agent is
            # created. Sentinels have no activity state, so the
            # idle check below would always evaluate to inf >= timeout and
            # immediately evict them, racing with the setup path.
            _stale_idle = float("inf")  # assume idle if we can't check
            _stale_detail = ""
            if _stale_agent and _stale_agent is not _AGENT_PENDING_SENTINEL:
                try:
                    _sa = status_output.get_activity_summary(_stale_agent)
                    _stale_idle = _sa.get("seconds_since_activity", float("inf"))
                    _stale_detail = (
                        f" | last_activity={_sa.get('last_activity_desc', 'unknown')} "
                        f"({_stale_idle:.0f}s ago) "
                        f"| iteration={_sa.get('api_call_count', 0)}/{_sa.get('max_iterations', 0)}"
                    )
                except Exception:
                    pass
            # Evict if: agent is idle beyond timeout, OR wall-clock age is
            # extreme (10x timeout or 2h, whichever is larger — catches
            # cases where the agent object was garbage-collected).
            _wall_ttl = max(_raw_stale_timeout * 10, 7200) if _raw_stale_timeout > 0 else float("inf")
            _should_evict = (
                _stale_agent is not _AGENT_PENDING_SENTINEL
                and (
                    (_raw_stale_timeout > 0 and _stale_idle >= _raw_stale_timeout)
                    or _stale_age > _wall_ttl
                )
            )
            if _should_evict:
                logger.warning(
                    "Evicting stale _running_agents entry for %s "
                    "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
                    _quick_key, _stale_age, _stale_idle,
                    _raw_stale_timeout, _stale_detail,
                )
                self.sessions.invalidate_run_generation(
                    _quick_key,
                    reason="stale_running_agent_eviction",
                )
                self._release_turn_state(_quick_key)

        if self.sessions.is_running(_quick_key):
            # Resolve the command once; every command's mid-run behavior is
            # declared on its CommandDef (busy_policy / busy_handler in
            # hermes_cli/commands.py) and dispatched through the single
            # resolver _dispatch_busy_slash_command below — no per-command
            # if-chain here.
            from hermes_cli.commands import resolve_command as _resolve_cmd_inner
            _evt_cmd = event.get_command()
            _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

            # /status and /context are intentionally pre-gate so users
            # always see session state.
            if _cmd_def_inner and _cmd_def_inner.name == "status":
                return await self._handle_status_command(event)
            if _cmd_def_inner and _cmd_def_inner.name == "context":
                return await self._handle_context_command(event)

            # Slash command access control on the running-agent fast-path.
            # Mirrors the cold-path gate further below so non-admin users
            # can't bypass gating just because an agent happens to be busy.
            # /status above is intentionally pre-gate so users always see
            # session state. /help and /whoami fall under the always-allowed
            # floor inside _check_slash_access.
            if _evt_cmd and _cmd_def_inner is not None:
                _denied = self._check_slash_access(source, _cmd_def_inner.name)
                if _denied is not None:
                    return _denied

            # Any recognized slash command: dispatch according to its
            # declared busy_policy (dispatch / interrupt_then_dispatch /
            # reject). Unrecognized commands and plain text fall through
            # to the interrupt/queue logic below.
            if _cmd_def_inner:
                return await self._dispatch_busy_slash_command(
                    event, _cmd_def_inner, _quick_key, source,
                )

            if event.message_type == MessageType.PHOTO:
                logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key)
                adapter = self._adapter_for_source(source)
                if adapter:
                    merge_pending_message_event(adapter._pending_messages, _quick_key, event)
                return None

            effective_busy_input_mode = self._effective_busy_input_mode(source)
            _telegram_followup_grace = float(
                os.getenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
            )
            _grace_state = self.sessions.peek(_quick_key)
            _started_at = _grace_state.turn.started_ts if _grace_state else 0
            if (
                source.platform == Platform.TELEGRAM
                and event.message_type == MessageType.TEXT
                and _telegram_followup_grace > 0
                and _started_at
                and (time.time() - _started_at) <= _telegram_followup_grace
            ):
                logger.debug(
                    "Telegram follow-up arrived %.2fs after run start for %s — queueing without interrupt",
                    time.time() - _started_at,
                    _quick_key,
                )
                adapter = self._adapter_for_source(source)
                if adapter:
                    if effective_busy_input_mode == "queue":
                        self.sessions.enqueue_fifo(_quick_key, event, adapter)
                    else:
                        merge_pending_message_event(
                            adapter._pending_messages,
                            _quick_key,
                            event,
                            merge_text=True,
                        )
                return None

            _ra_state = self.sessions.peek(_quick_key)
            running_agent = _ra_state.turn.agent if _ra_state else None
            if running_agent is _AGENT_PENDING_SENTINEL:
                # Agent is being set up but not ready yet.
                if event.get_command() == "stop":
                    # Force-clean the sentinel so the session is unlocked.
                    self._release_turn_state(_quick_key)
                    logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key)
                    return EphemeralReply("⚡ Force-stopped. The agent was still starting — session unlocked.")
                # Queue the message so it will be picked up after the
                # agent starts.
                adapter = self._adapter_for_source(source)
                if adapter:
                    merge_pending_message_event(
                        adapter._pending_messages,
                        _quick_key,
                        event,
                        merge_text=True,
                    )
                return None
            if self._draining:
                queue_during_drain = self._queue_during_drain_enabled(
                    effective_busy_input_mode
                )
                if queue_during_drain:
                    self._queue_or_replace_pending_event(_quick_key, event)
                return (
                    f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
                    if queue_during_drain
                    else f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
                )
            if effective_busy_input_mode == "queue":
                logger.debug("PRIORITY queue follow-up for session %s", _quick_key)
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            if effective_busy_input_mode == "steer":
                # Steer mode: inject text into the running agent mid-run via
                # interruption.steer(agent).  Falls back to queue semantics if the payload
                # is empty, the agent lacks steer(), or steer() rejects.
                steer_text = (event.text or "").strip()
                steered = False
                if (
                    event.message_type == MessageType.TEXT
                    and not event.media_urls
                    and not event.media_types
                    and steer_text
                ):
                    try:
                        steered = bool(interruption.steer(running_agent, steer_text))
                    except Exception as exc:
                        logger.warning("PRIORITY steer failed for session %s: %s", _quick_key, exc)
                        steered = False
                if steered:
                    logger.debug("PRIORITY steer for session %s", _quick_key)
                    return None
                logger.debug("PRIORITY steer-fallback-to-queue for session %s", _quick_key)
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            # #30170 — Subagent protection (PRIORITY path). Same rationale
            # as ``_handle_active_session_busy_message``: an interrupt
            # cascades through ``_active_children`` and aborts in-flight
            # delegate_task work. Demote to queue semantics when the
            # parent is currently driving subagents so a conversational
            # follow-up doesn't destroy minutes of subagent progress.
            # /stop reaches its dedicated handler above, so the operator
            # still has a clean escape hatch.
            if self._agent_has_active_subagents(running_agent):
                logger.info(
                    "PRIORITY interrupt demoted to queue for session %s "
                    "because the running agent has active subagents (#30170)",
                    _quick_key,
                )
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            # #56391 — Compression protection (PRIORITY path). Same
            # rationale as ``_handle_active_session_busy_message``: context
            # compression is interrupt-protected (#23975), but an interrupt
            # here starts a new turn against the pre-rotation parent
            # session while the still-running compression later rotates
            # the id out from under it, forking orphaned compression
            # siblings. Demote to queue semantics so the follow-up waits
            # for the in-flight compression + rotation to land.
            if await self._session_has_compression_in_flight(_quick_key):
                logger.info(
                    "PRIORITY interrupt demoted to queue for session %s "
                    "because context compression is in flight (#56391)",
                    _quick_key,
                )
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            # Text-only corrections redirect the live turn (preserving
            # displayed context) when the runtime supports it; media/voice and
            # older runtimes fall back to the proven interrupt path below.
            if (
                event.message_type == MessageType.TEXT
                and not event.media_urls
                and not event.media_types
                and getattr(running_agent, "_supports_active_turn_redirect", False)
                is True
            ):
                try:
                    if interruption.redirect(running_agent, (event.text or "").strip()):
                        logger.debug("PRIORITY redirect for session %s", _quick_key)
                        return None
                except Exception as exc:
                    logger.warning(
                        "PRIORITY redirect failed for session %s: %s",
                        _quick_key,
                        exc,
                    )
            logger.debug("PRIORITY interrupt for session %s", _quick_key)
            _interrupt_text = event.text
            _media_urls = getattr(event, "media_urls", None) or []
            if self._pending_event_audio_paths(event):
                _interrupt_text, _ = await self._prepare_pending_audio_event_once(
                    event,
                    self._adapter_for_source(source),
                    source,
                    event.text or "",
                    log_context="Voice-priority-interrupt",
                )
            elif not _interrupt_text and _media_urls:
                _interrupt_text = _build_media_placeholder(event)
            interruption.interrupt(running_agent, _interrupt_text)
            # NOTE: self._pending_messages was write-only (never consumed).
            # The actual interrupt message is delivered via adapter._pending_messages
            # which is read by _run_agent. Removed to prevent unbounded growth.
            return None

        # Check for commands
        command = event.get_command()

        from hermes_cli.commands import (
            GATEWAY_KNOWN_COMMANDS,
            is_gateway_known_command,
            resolve_command as _resolve_cmd,
        )

        # Resolve aliases to canonical name so dispatch and hook names
        # don't depend on the exact alias the user typed.
        _cmd_def = _resolve_cmd(command) if command else None
        canonical = _cmd_def.name if _cmd_def else command

        # Expand alias quick commands before built-in dispatch so targets like
        # /model openai/gpt-5.5 --provider openrouter reach the /model handler.
        # Preserve built-in precedence; aliases only need early handling when
        # the typed command is not already known.
        if command and _cmd_def is None:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if isinstance(quick_commands, dict) and command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "alias":
                    target = (qcmd.get("target") or "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        _cmd_def = _resolve_cmd(command) if command else None
                        canonical = _cmd_def.name if _cmd_def else command

        # Per-platform slash command access control. Only kicks in when the
        # operator has set ``allow_admin_from`` for the source's scope (DM
        # vs group). When unset → backward-compat: every allowed user can
        # run every command. When set → non-admins can run only commands in
        # ``user_allowed_commands`` (plus the always-allowed floor: /help,
        # /whoami). Plain chat is unaffected — only slash commands gate.
        if command and canonical and is_gateway_known_command(canonical):
            _denied = self._check_slash_access(source, canonical)
            if _denied is not None:
                return _denied

        # pre_command observer hook (#64204): fires for every recognized
        # slash command BEFORE core handling, mirroring the CLI fire-site in
        # cli.py process_command. Observer-only in v1 (returns ignored).
        #
        # Placement matters: this cold-path dispatch is only reached when NO
        # agent is running for the session. The running-agent intercept path
        # above (/stop, /approve, busy_policy dispatch via
        # _dispatch_busy_slash_command) deliberately does NOT fire this hook —
        # those are control-plane operations on an in-flight run, and giving
        # plugins an observation (and eventually veto) point there would let
        # a slow or hostile plugin interfere with the operator's escape
        # hatches for a live agent.
        if command and is_gateway_known_command(canonical):
            try:
                from hermes_cli.plugins import fire_pre_command_hook
                fire_pre_command_hook(
                    surface="gateway",
                    command=str(canonical),
                    alias_used=str(command),
                    args_raw=event.get_command_args().strip(),
                    session_key=_quick_key,
                    platform=source.platform.value if source.platform else "",
                )
            except Exception as _pre_cmd_err:
                logger.debug(
                    "pre_command hook dispatch failed (non-fatal): %s",
                    _pre_cmd_err,
                )

        # Fire the ``command:<canonical>`` hook for any recognized slash
        # command — built-in OR plugin-registered. Handlers can return a
        # dict with ``{"decision": "deny" | "handled" | "rewrite", ...}``
        # to intercept dispatch before core handling runs. This replaces
        # the previous fire-and-forget emit(): return values are now
        # honored, but handlers that return nothing behave exactly as
        # before (telemetry-style hooks keep working).
        if command and is_gateway_known_command(canonical):
            raw_args = event.get_command_args().strip()
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": canonical,
                "raw_command": command,
                "args": raw_args,
                "raw_args": raw_args,
            }
            try:
                hook_results = await self.hooks.emit_collect(
                    f"command:{canonical}", hook_ctx
                )
            except Exception as _hook_err:
                logger.debug(
                    "command:%s hook dispatch failed (non-fatal): %s",
                    canonical, _hook_err,
                )
                hook_results = []

            for hook_result in hook_results:
                if not isinstance(hook_result, dict):
                    continue
                decision = str(hook_result.get("decision", "")).strip().lower()
                if not decision or decision == "allow":
                    continue
                if decision == "deny":
                    message = hook_result.get("message")
                    if isinstance(message, str) and message:
                        return message
                    return f"Command `/{command}` was blocked by a hook."
                if decision == "handled":
                    message = hook_result.get("message")
                    return message if isinstance(message, str) and message else None
                if decision == "rewrite":
                    new_command = str(
                        hook_result.get("command_name", "")
                    ).strip().lstrip("/")
                    if not new_command:
                        continue
                    new_args = str(hook_result.get("raw_args", "")).strip()
                    event.text = f"/{new_command} {new_args}".strip()
                    command = event.get_command()
                    _cmd_def = _resolve_cmd(command) if command else None
                    canonical = _cmd_def.name if _cmd_def else command
                    break

        if canonical == "pause":
            return await self._handle_pause_command(event)

        if canonical == "new":
            if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
                return self._telegram_topic_root_new_message()
            async def _do_reset():
                return await self._handle_reset_command(event)
            return await self._maybe_confirm_destructive_slash(
                event=event,
                command="new",
                title="/new",
                detail=(
                    "This starts a fresh session and discards the current "
                    "conversation history."
                ),
                execute=_do_reset,
            )

        if canonical == "topic":
            return await self._handle_topic_command(event)

        if canonical == "help":
            return await self._handle_help_command(event)

        if canonical == "start":
            logger.info("Ignoring /start platform ping for session %s", _quick_key)
            return ""

        if canonical == "commands":
            return await self._handle_commands_command(event)

        if canonical == "profile":
            return await self._handle_profile_command(event)

        if canonical == "whoami":
            return await self._handle_whoami_command(event)

        if canonical == "status":
            return await self._handle_status_command(event)

        if canonical == "egress":
            from hermes_cli.proxy_cli import format_status_text

            return format_status_text()

        if canonical == "context":
            return await self._handle_context_command(event)

        if canonical == "agents":
            return await self._handle_agents_command(event)

        if canonical == "platform":
            return await self._handle_platform_command(event)

        if canonical == "restart":
            return await self._handle_restart_command(event)

        if canonical == "stop":
            return await self._handle_stop_command(event)

        if canonical == "reasoning":
            return await self._handle_reasoning_command(event)

        if canonical == "memory":
            return await self._handle_memory_command(event)

        if canonical == "skills":
            return await self._handle_skills_command(event)

        if canonical == "learn":
            # Open-ended: rewrite the turn to a standards-guided prompt and fall
            # through to normal agent processing. The live agent gathers the
            # sources the user described (dirs via read_file, URLs via
            # web_extract, this conversation, pasted text) and authors the skill
            # via skill_manage. Mirrors the /blueprint fall-through so role
            # alternation is preserved. No engine, works on any backend.
            from agent.learn_prompt import build_learn_prompt

            _learn_req = event.get_command_args().strip()
            _ack = (
                "Learning a skill from what you described…"
                if _learn_req
                else "Learning a skill from this conversation…"
            )
            try:
                adapter = self._adapter_for_source(source)
                if adapter:
                    _ack_meta = self._thread_metadata_for_source(source)
                    await adapter.send(str(source.chat_id), _ack, metadata=_ack_meta)
            except Exception:
                logger.debug("learn ack send failed", exc_info=True)
            try:
                event.text = build_learn_prompt(_learn_req)
                # fall through to agent processing
            except Exception:
                return "Could not start /learn — please try again."

        if canonical == "init":
            # /init: rewrite the turn to a guidance-laden prompt and fall
            # through to normal agent processing (same fall-through as /learn
            # so role alternation is preserved). The live agent scans the
            # project with its own read-only tools and writes/updates
            # AGENTS.md via write_file. No engine, works on any backend.
            from hermes_cli.init_command import build_init_prompt_for_cwd

            _init_notes = event.get_command_args().strip()
            try:
                _init_prompt = build_init_prompt_for_cwd(extra=_init_notes)
            except Exception:
                return "Could not start /init — please try again."
            _ack = (
                "Updating AGENTS.md from a project scan…"
                if "UPDATE the existing AGENTS.md" in _init_prompt
                else "Generating AGENTS.md from a project scan…"
            )
            try:
                adapter = self._adapter_for_source(source)
                if adapter:
                    _ack_meta = self._thread_metadata_for_source(source)
                    await adapter.send(str(source.chat_id), _ack, metadata=_ack_meta)
            except Exception:
                logger.debug("init ack send failed", exc_info=True)
            event.text = _init_prompt
            # fall through to agent processing

        if canonical == "fast":
            return await self._handle_fast_command(event)

        if canonical == "verbose":
            return await self._handle_verbose_command(event)

        if canonical == "footer":
            return await self._handle_footer_command(event)

        if canonical == "yolo":
            return await self._handle_yolo_command(event)

        if canonical == "approvals":
            return await self._handle_approvals_command(event)

        if canonical == "model":
            return await self._handle_model_command(event)

        if canonical == "codex-runtime":
            return await self._handle_codex_runtime_command(event)

        if canonical == "personality":
            return await self._handle_personality_command(event)

        if canonical == "kanban":
            return await self._handle_kanban_command(event)

        if canonical == "suggestions":
            return await self._handle_suggestions_command(event)

        if canonical == "blueprint":
            _blueprint_result = await self._handle_blueprint_command(event)
            _blueprint_seed = getattr(_blueprint_result, "agent_seed", None)
            if _blueprint_seed:
                # Blueprint matched — rewrite the turn to the seed and fall
                # through to _handle_message_with_agent so the agent asks the
                # user for each slot value conversationally and then calls the
                # cronjob tool (the /steer fall-through pattern). The seed
                # enters as a normal user turn, preserving role alternation.
                # Send the "Setting up X…" ack first so the user gets the same
                # immediate feedback CLI users see, instead of silence until
                # the agent's first question.
                _ack = getattr(_blueprint_result, "text", "") or ""
                if _ack:
                    try:
                        adapter = self._adapter_for_source(source)
                        if adapter:
                            _ack_meta = self._thread_metadata_for_source(source)
                            await adapter.send(str(source.chat_id), _ack, metadata=_ack_meta)
                    except Exception:
                        logger.debug("blueprint ack send failed", exc_info=True)
                try:
                    event.text = _blueprint_seed
                except Exception:
                    return getattr(_blueprint_result, "text", "") or None
            else:
                return getattr(_blueprint_result, "text", "") or None

        if canonical == "save":
            return await self._handle_save_command(event)

        if canonical == "retry":
            return await self._handle_retry_command(event)

        if canonical == "undo":
            async def _do_undo():
                return await self._handle_undo_command(event)
            _undo_n = 1
            _undo_raw = event.get_command_args().strip()
            if _undo_raw:
                try:
                    _undo_n = max(1, int(_undo_raw.split()[0]))
                except (ValueError, IndexError):
                    _undo_n = 1
            _undo_detail = (
                "This removes the last user/assistant exchange from history."
                if _undo_n == 1
                else f"This removes the last {_undo_n} user turns from history."
            )
            return await self._maybe_confirm_destructive_slash(
                event=event,
                command="undo",
                title="/undo",
                detail=_undo_detail,
                execute=_do_undo,
            )

        if canonical == "sethome":
            return await self._handle_set_home_command(event)

        if canonical == "compress":
            return await self._handle_compress_command(event)

        if canonical == "usage":
            return await self._handle_usage_command(event)

        if canonical == "insights":
            return await self._handle_insights_command(event)

        if canonical == "reload-mcp":
            return await self._handle_reload_mcp_command(event)

        if canonical == "reload-skills":
            return await self._handle_reload_skills_command(event)

        if canonical == "bundles":
            return await self._handle_bundles_command(event)

        if canonical == "approve":
            return await self._handle_approve_command(event)

        if canonical == "deny":
            return await self._handle_deny_command(event)

        if canonical == "update":
            return await self._handle_update_command(event)

        if canonical == "version":
            return await self._handle_version_command(event)

        if canonical == "debug":
            return await self._handle_debug_command(event)

        if canonical == "title":
            return await self._handle_title_command(event)

        if canonical == "resume":
            return await self._handle_resume_command(event)

        if canonical == "sessions":
            return await self._handle_sessions_command(event)

        if canonical == "branch":
            return await self._handle_branch_command(event)

        if canonical == "rollback":
            return await self._handle_rollback_command(event)

        if canonical == "diff":
            return await self._handle_diff_command(event)

        if canonical == "background":
            return await self._handle_background_command(event)

        if canonical == "queue":
            queue_payload = event.get_command_args().strip()
            if not queue_payload:
                return "Usage: /queue <prompt>"
            try:
                event.text = queue_payload
            except Exception:
                pass

        if canonical == "steer":
            # No active agent — /steer has no tool call to inject into.
            # Strip the prefix so downstream treats it as a normal user
            # message. If the payload is empty, surface the usage hint.
            steer_payload = event.get_command_args().strip()
            if not steer_payload:
                return "Usage: /steer <prompt>  (no agent is running; sending as a normal message)"
            try:
                event.text = steer_payload
            except Exception:
                pass
            # Do NOT return — fall through to _handle_message_with_agent
            # at the end of this function so the rewritten text is sent
            # to the agent as a regular user turn.

        if canonical == "goal":
            return await self._handle_goal_command(event)

        if canonical == "loop":
            return await self._handle_loop_command(event)

        if canonical == "heartbeat":
            return await self._handle_heartbeat_command(event)
        if canonical == "refine":
            return await self._handle_refine_command(event)

        if canonical == "moa":
            # /moa is one-shot sugar only: run a single prompt through the
            # default MoA preset, then restore the prior model. To *switch* to a
            # MoA preset for the session, pick it from the model picker (MoA
            # presets surface as a virtual "Mixture of Agents" provider).
            from hermes_cli.moa_config import (
                moa_usage,
                normalize_moa_config,
            )
            from hermes_cli.config import load_config

            moa_payload = event.get_command_args().strip()
            if not moa_payload:
                return moa_usage()
            try:
                cfg = load_config()
                moa_cfg = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
            except Exception:
                moa_cfg = normalize_moa_config({})
            preset = moa_cfg["default_preset"]
            try:
                event.text = moa_payload
                _moa_state = self.sessions.state(_quick_key)
                event._moa_restore_override = _moa_state.conversation.model_override
                _moa_state.conversation.model_override = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
                self.agent_cache.evict(_quick_key)
                event._moa_disable_after_turn = True
            except Exception:
                return "Failed to prepare MoA turn."

        if canonical == "subgoal":
            return await self._handle_subgoal_command(event)

        if self._draining:
            return f"⏳ Gateway is {self._status_action_gerund()} and is not accepting new work right now."

        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if not isinstance(quick_commands, dict):
                quick_commands = {}
            if command in quick_commands:
                # Quick commands are slash capabilities too — and type:exec
                # ones run a shell command in the gateway process. The early
                # gate above only fires for registry-known commands, so quick
                # commands (never in the registry) would otherwise reach this
                # dispatch sink unchecked. Apply the same admin/user policy to
                # the raw typed name here so non-admins can't invoke admin-only
                # quick commands. (#44727)
                _denied = self._check_slash_access(source, command)
                if _denied is not None:
                    return _denied
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            # Sanitize env to prevent credential leakage —
                            # quick commands run in the gateway process which
                            # has all API keys in os.environ.
                            from tools.environments.local import build_subprocess_env
                            sanitized_env = build_subprocess_env()
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=sanitized_env,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            # Redact any remaining sensitive patterns in output
                            if output:
                                from agent.redact import redact_sensitive_text
                                output = redact_sensitive_text(output)
                            return output if output else "Command returned no output."
                        except asyncio.TimeoutError:
                            return "Quick command timed out (30s)."
                        except Exception as e:
                            return f"Quick command error: {e}"
                    else:
                        return f"Quick command '/{command}' has no command defined."
                elif qcmd.get("type") == "alias":
                    target = (qcmd.get("target") or "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        # Fall through to normal command dispatch below
                    else:
                        return f"Quick command '/{command}' has no target defined."
                else:
                    return f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias')."

        # Plugin-registered slash commands
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                # Normalize underscores to hyphens so Telegram's underscored
                # autocomplete form matches plugin commands registered with
                # hyphens. See hermes_cli/commands.py:_build_telegram_menu.
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    user_args = event.get_command_args().strip()
                    result = plugin_handler(user_args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result) if result else None
            except Exception as e:
                logger.warning("Plugin command dispatch failed: %s", e)

        # Skill slash commands: /skill-name loads the skill and sends to agent.
        # resolve_skill_command_key() handles the Telegram underscore/hyphen
        # round-trip so /claude_code from Telegram autocomplete still resolves
        # to the claude-code skill.
        if command:
            # Skill bundles take precedence over individual skill commands —
            # /<bundle> loads multiple skills at once. Mirrors CLI dispatch.
            _bundle_handled = False
            try:
                from agent.skill_bundles import (
                    build_bundle_invocation_message,
                    resolve_bundle_command_key,
                )
                bundle_key = resolve_bundle_command_key(command)
                if bundle_key is not None:
                    user_instruction = event.get_command_args().strip()
                    # Pass the platform explicitly: bundle skill loading
                    # bypasses get_skill_commands()' scan-time disabled
                    # filter, and the gateway serves multiple platforms in
                    # one process, so env-var platform resolution can't be
                    # trusted here. Mirrors the stacked-skill gate (#58888).
                    _bundle_plat = source.platform.value if source.platform else None
                    bundle_result = build_bundle_invocation_message(
                        bundle_key, user_instruction, task_id=_quick_key,
                        platform=_bundle_plat,
                    )
                    if bundle_result:
                        msg, _loaded, missing = bundle_result
                        event.text = msg
                        _bundle_handled = True
                        if missing:
                            logger.info(
                                "Bundle %s skipped missing skills: %s",
                                bundle_key, ", ".join(missing),
                            )
                        # Fall through to normal message processing with bundle content
            except Exception as exc:
                logger.warning("Bundle dispatch failed: %s", exc)

        if command and not locals().get("_bundle_handled", False):
            try:
                from agent.skill_commands import (
                    get_skill_commands,
                    build_skill_invocation_message,
                    resolve_skill_command_key,
                )
                skill_cmds = get_skill_commands()
                cmd_key = resolve_skill_command_key(command)
                if cmd_key is not None:
                    # Check per-platform disabled status before executing.
                    # get_skill_commands() only applies the *global* disabled
                    # list at scan time; per-platform overrides need checking
                    # here because the cache is process-global across platforms.
                    _skill_name = skill_cmds[cmd_key].get("name", "")
                    _plat = source.platform.value if source.platform else None
                    if _plat and _skill_name:
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        if _skill_name in _get_plat_disabled(platform=_plat):
                            return (
                                f"The **{_skill_name}** skill is disabled for {_plat}.\n"
                                f"Enable it with: `hermes skills config`"
                            )
                    user_instruction = event.get_command_args().strip()
                    # Stacked slash-skill invocations: `/skill-a /skill-b do
                    # XYZ` loads every leading skill (up to 5), not just the
                    # first. Inspired by Claude Code v2.1.199. Mirrors CLI.
                    try:
                        from agent.skill_commands import (
                            build_stacked_skill_invocation_message as _build_stacked,
                            split_stacked_skill_commands,
                        )
                        extra_keys, stacked_instruction = (
                            split_stacked_skill_commands(user_instruction)
                        )
                    except Exception:
                        _build_stacked = None
                        extra_keys, stacked_instruction = [], user_instruction
                    if extra_keys and _plat:
                        # split_stacked_skill_commands() only resolves that
                        # each extra token is a KNOWN skill command — like
                        # get_skill_commands() itself, it has no per-platform
                        # view. Re-check every stacked skill (not just the
                        # leading one above) against the same disabled list,
                        # or a skill an operator disabled for this platform
                        # still gets its full content loaded via the stack.
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        _plat_disabled = _get_plat_disabled(platform=_plat)
                        _disabled_extra = [
                            skill_cmds.get(k, {}).get("name", "")
                            for k in extra_keys
                            if skill_cmds.get(k, {}).get("name", "") in _plat_disabled
                        ]
                        if _disabled_extra:
                            return (
                                f"The **{', '.join(_disabled_extra)}** skill(s) in this "
                                f"stacked invocation are disabled for {_plat}.\n"
                                f"Enable them with: `hermes skills config`"
                            )
                    if extra_keys and _build_stacked is not None:
                        stacked_result = _build_stacked(
                            [cmd_key, *extra_keys],
                            stacked_instruction,
                            task_id=_quick_key,
                        )
                        if stacked_result:
                            msg, _loaded, _missing = stacked_result
                            event.text = msg
                            # Fall through to normal message processing
                        else:
                            return f"Failed to load stacked skills for /{command}."
                    else:
                        msg = build_skill_invocation_message(
                            cmd_key, user_instruction, task_id=_quick_key
                        )
                        if msg:
                            event.text = msg
                            # Fall through to normal message processing with skill content
                else:
                    # Not an active skill — check if it's a known-but-disabled or
                    # uninstalled skill and give actionable guidance.
                    _unavail_msg = _check_unavailable_skill(command)
                    if _unavail_msg:
                        return _unavail_msg
                    # Genuinely unrecognized /command: not a built-in, not a
                    # plugin, not a skill, not a known-inactive skill. Warn
                    # the user instead of silently forwarding it to the LLM
                    # as free text (which leads to silent-failure behavior
                    # like the model inventing a delegate_task call).
                    # Normalize to hyphenated form before checking known
                    # built-ins (command may be an alias target set by the
                    # quick-command block above, so _cmd_def can be stale).
                    if command.replace("_", "-") not in GATEWAY_KNOWN_COMMANDS:
                        logger.warning(
                            "Unrecognized slash command /%s from %s — "
                            "replying with unknown-command notice",
                            command,
                            source.platform.value if source.platform else "?",
                        )
                        return (
                            f"Unknown command `/{command}`. "
                            f"Type /commands to see what's available, "
                            f"or resend without the leading slash to send "
                            f"as a regular message."
                        )
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)

        # Pending exec approvals are handled by /approve and /deny commands above.
        # No bare text matching — "yes" in normal conversation must not trigger
        # execution of a dangerous command.

        if not is_internal and await asyncio.to_thread(
            self._is_telegram_topic_root_lobby, source
        ):
            # Debounce the lobby reminder so a user who forgets about
            # topic mode and fires ten prompts doesn't get ten copies.
            if self._should_send_telegram_lobby_reminder(source):
                return self._telegram_topic_root_lobby_message()
            return None

        # ── Claim this session before any await ───────────────────────
        # Between here and _run_agent registering the real create_agent, there
        # are numerous await points (hooks, vision enrichment, STT,
        # session hygiene compression).  Without this sentinel a second
        # message arriving during any of those yields would pass the
        # "already running" guard and spin up a duplicate agent for the
        # same session — corrupting the transcript.
        _active_session_lease, _limit_message = self._claim_active_session_slot(
            _quick_key,
            source,
        )
        if _limit_message is not None:
            logger.info(
                "Rejecting new active session %s: max_concurrent_sessions reached",
                _quick_key,
            )
            return _limit_message
        _claim_state = self.sessions.state(_quick_key)
        if _active_session_lease is not None:
            _claim_state.turn.lease = _active_session_lease
        _claim_state.turn.agent = _AGENT_PENDING_SENTINEL
        _claim_state.turn.started_ts = time.time()
        self._persist_active_agents()
        _run_generation = self.sessions.begin_run_generation(_quick_key)

        try:
            try:
                _agent_result = await self._handle_message_with_agent(
                    event, source, _quick_key, _run_generation
                )
            except TurnLeaseTimeoutError as exc:
                # This is a rejected message, not a completed agent turn. Return
                # before the /goal judge below so it cannot consume the resend
                # notice and enqueue a synthetic continuation loop.
                logger.error(
                    "Rejecting turn for routing key %s on session %s after "
                    "turn-lease timeout; transcript load was not started and "
                    "the user must resend",
                    _quick_key,
                    exc.session_id,
                )
                return (
                    "⏳ Another turn is still running on this session. To "
                    "protect the transcript, this message was not processed. "
                    "Wait for the active turn to finish, then resend it."
                )
            try:
                await self._run_post_turn_hooks(
                    agent_result=_agent_result,
                    source=source,
                    is_internal=is_internal,
                    event=event,
                )
            except Exception as _goal_exc:
                logger.debug("post-turn hook failed: %s", _goal_exc)
            return _agent_result
        finally:
            # MoA one-shot restore must run on EVERY exit path, not just
            # success. The restore data lives on the per-turn event object
            # (_moa_restore_override), which is discarded once the event goes
            # out of scope — so if _handle_message_with_agent raises, a restore
            # in the try block would be skipped and the MoA override would leak
            # permanently (every later message silently fans out through MoA).
            # Putting it in finally guarantees the revert on success, exception,
            # and interrupt alike.
            self._restore_moa_one_shot(event, _quick_key)
            self._restore_pending_one_turn_model_override(_quick_key)
            # Normal completion/exception/interrupt owns and clears this exact
            # durable marker.  SIGKILL/OOM skips finally, leaving the marker for
            # the next unclean startup's recovery pass.
            await self._clear_durable_active_turn(event)
            # Unconditional release covers every exit path. _release_turn_state
            # is idempotent (pop-on-absent is harmless) and, called without a
            # run_generation guard, always clears the slot regardless of which
            # generation it holds. This evicts the zombie left when session_reset
            # bumps the generation (N -> N+1) mid-flight: gen-N's guarded release
            # inside _run_agent returns False, and the old sentinel-only check here
            # missed the leftover real agent — locking the session out forever (#28686).
            self._release_turn_state(_quick_key)
            # Turn lease (#64934): release THIS turn's lease token — keyed by
            # (routing key, run generation) so this unwind can only ever free
            # the lease its own turn acquired, never a newer turn's.
            self.sessions.release_turn_lease(_quick_key, _run_generation)

    def _restore_moa_one_shot(self, event: "MessageEvent", quick_key: str) -> None:
        """Revert a ``/moa <prompt>`` one-shot model override after its turn.

        Called from the ``finally`` of the message-handling path so the revert
        fires whether the turn succeeded, raised, or was interrupted. A no-op
        unless ``event._moa_disable_after_turn`` is set. ``_moa_restore_override``
        carries the prior per-session override (``None`` means the user had no
        override, so the MoA override is cleared outright).
        """
        if not getattr(event, "_moa_disable_after_turn", False):
            return
        try:
            _restore = getattr(event, "_moa_restore_override", None)
            self.sessions.state(quick_key).conversation.model_override = _restore
            self.agent_cache.evict(quick_key)
        except Exception:
            pass

    def _restore_pending_one_turn_model_override(self, session_key: str) -> None:
        """Restore a per-session model override after ``/model --once`` runs."""
        if not session_key:
            return
        try:
            _otr_state = self.sessions.peek(session_key)
            snapshot = _otr_state.conversation.one_turn_restore if _otr_state else None
            if _otr_state is not None:
                _otr_state.conversation.one_turn_restore = None
            if not snapshot:
                return
            self._restore_session_model_override(session_key, snapshot)
        except Exception:
            logger.debug("Failed to restore one-turn model override", exc_info=True)

    async def _prepare_inbound_message_text(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare inbound event text for the agent.

        Keep the normal inbound path and the queued follow-up path on the same
        preprocessing pipeline so sender attribution, image enrichment,
        document notes, reply context, and @ references all behave the same.

        Side effect: buffers per-session native image paths when the active
        model supports native vision AND the user has images attached. The
        caller consumes and clears that session-scoped buffer at the
        ``run_conversation`` site to build a multimodal user turn. When the
        list is empty, the ``_enrich_message_with_vision`` text path has
        already run and images are represented in-text.
        """
        history = history or []
        _pending_audio_prepared = hasattr(event, "_gateway_pending_audio_text")
        message_text = (
            getattr(event, "_gateway_pending_audio_text", None)
            if _pending_audio_prepared
            else event.text
        ) or ""
        _group_sessions_per_user = getattr(self.config, "group_sessions_per_user", True)
        _thread_sessions_per_user = getattr(self.config, "thread_sessions_per_user", False)
        # Prefer the already resolved session key from the caller so this write
        # key matches the consume key at the run_conversation site. Fall back
        # to deriving it here for tests and legacy standalone callers.
        session_key = session_key or self._session_key_for_source(source)
        # Reset only this session's per-call buffer; other sessions may be
        # concurrently preparing multimodal turns on the same runner.
        self._consume_pending_native_image_paths(session_key)

        _is_shared_multi_user = is_shared_multi_user_session(
            source,
            group_sessions_per_user=_group_sessions_per_user,
            thread_sessions_per_user=_thread_sessions_per_user,
        )
        if _is_shared_multi_user and source.user_name:
            # source.user_name is the platform display name — attacker-
            # influenceable on any platform that lets participants set their
            # own name. Neutralize embedded newlines/control chars before
            # interpolating it into every message in the shared session, or
            # a hostile name can masquerade as a fake markdown section
            # (mirrors the same field's treatment in
            # build_session_context_prompt via _format_untrusted_prompt_value).
            _safe_user_name = neutralize_untrusted_inline_text(source.user_name)
            message_text = f"[{_safe_user_name}] {message_text}"

        # Prepend channel context from history backfill (if any).  This
        # happens after sender-prefix so the prefix only applies to the
        # trigger message, not the backfill block.
        if getattr(event, "channel_context", None):
            message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"

        # Declare at outer scope so the audio-file-paths handling block below
        # remains safe when ``event.media_urls`` is empty (no inner block runs).
        video_paths: list[str] = []

        if event.media_urls:
            image_paths = []
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                # Classify images per-attachment: trust this attachment's own
                # MIME, and only honour the message-level PHOTO type when the
                # per-attachment MIME is unknown. Otherwise a document (or any
                # non-image) sent alongside an image in the same message gets
                # mis-routed here as an image and the provider 400s.
                if _event_media_is_image(event, i):
                    image_paths.append(path)
                if not _pending_audio_prepared and (
                    event.message_type == MessageType.AUDIO
                    or _event_media_is_stt_input(event, i)
                ):
                    audio_paths.append(path)
                if mtype.startswith("video/") or (not mtype and event.message_type == MessageType.VIDEO):
                    video_paths.append(path)

            if image_paths:
                # Decide routing: native (attach pixels) vs text (vision_analyze
                # pre-run + prepend description).  See agent/image_routing.py.
                # Offload to a worker thread: the decision does blocking network
                # I/O — a models.dev fetch on cache miss, and the Ollama
                # ``/api/show`` capability probe for local servers — whose
                # request timeout would otherwise stall the whole gateway event
                # loop (every session) while a single image is routed.
                _img_mode = await asyncio.to_thread(
                    self._decide_image_input_mode,
                    source=source,
                    session_key=session_key,
                )
                if _img_mode == "native":
                    # Defer attachment to the run_conversation call site.
                    self.sessions.state(
                        session_key
                    ).persistent.native_image_paths = list(image_paths)
                    logger.info(
                        "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                        len(image_paths),
                    )
                else:
                    logger.info(
                        "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
                        _img_mode, len(image_paths),
                    )
                    # Vision enrichment runs before create_agent.run_conversation(),
                    # so bind this session's resolved runtime explicitly rather
                    # than consulting process-global compatibility mirrors.
                    vision_runtime = None
                    try:
                        turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                            source=source,
                            session_key=session_key,
                        )
                        vision_runtime = dict(runtime_kwargs or {})
                        vision_runtime["model"] = turn_model
                    except Exception:
                        logger.debug(
                            "vision enrichment: session runtime resolution failed",
                            exc_info=True,
                        )

                    from agent.auxiliary_client import scoped_runtime_main

                    with scoped_runtime_main(vision_runtime):
                        message_text = await self._enrich_message_with_vision(
                            message_text,
                            image_paths,
                        )

            if audio_paths:
                message_text, _ = await self._enrich_message_with_audio_paths(
                    message_text,
                    audio_paths,
                )

        if video_paths:
            from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
            for _vpath in video_paths:
                _basename = os.path.basename(_vpath)
                _parts = _basename.split("_", 2)
                _display = _parts[2] if len(_parts) >= 3 else _basename
                _display = re.sub(r'[^\w.\- ]', '_', _display)
                _agent_path = _to_agent_path(_vpath)
                _note = (
                    f"[The user sent a video attachment: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the video contains, inspect or process it yourself — for "
                    f"example by passing the path to a video analysis or media tool — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                )
                message_text = f"{_note}\n\n{message_text}"

        if event.media_urls:
            import mimetypes as _mimetypes
            from tools.credential_files import to_agent_visible_cache_path

            _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
            for i, path in enumerate(event.media_urls):
                # Per-attachment document handling. Skip anything already routed
                # as image / audio / video by the buckets above — only genuine
                # non-media files get a path-pointing context note. This makes a
                # document mixed into a PHOTO/VOICE message (whole-message type
                # != DOCUMENT) still reach the agent as a readable cached file,
                # instead of being silently dropped because the message-level
                # type wasn't DOCUMENT.
                if (
                    _event_media_is_image(event, i)
                    or _event_media_is_audio(event, i)
                    or _event_media_is_video(event, i)
                ):
                    continue
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if mtype in {"", "application/octet-stream"}:
                    _ext = os.path.splitext(path)[1].lower()
                    if _ext in _TEXT_EXTENSIONS:
                        mtype = "text/plain"
                    else:
                        guessed, _ = _mimetypes.guess_type(path)
                        if guessed:
                            mtype = guessed
                        else:
                            mtype = "application/octet-stream"
                # Any accepted file gets a path-pointing context note — we accept
                # all file types now, so a non-text/non-application MIME (font/*,
                # model/*, etc.) must still tell the agent the file exists.

                basename = os.path.basename(path)
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                display_name = re.sub(r'[^\w.\- ]', '_', display_name)

                # Translate host cache path to in-container path if running under Docker backend.
                # This ensures the agent receives a path it can open inside its sandbox, as the
                # cache directories are auto-mounted at /root/.hermes/cache/* by get_cache_directory_mounts().
                agent_path = to_agent_visible_cache_path(path)

                context_note = _build_document_context_note(display_name, agent_path, mtype)
                message_text = f"{context_note}\n\n{message_text}"

        if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
            # Always inject the reply-to pointer — even when the quoted text
            # already appears in history. The prefix isn't deduplication, it's
            # disambiguation: it tells the agent *which* prior message the user
            # is referencing. History can contain the same or similar text
            # multiple times, and without an explicit pointer the agent has to
            # guess (or answer for both subjects). Token overhead is minimal.
            reply_snippet = event.reply_to_text[:500]
            if getattr(event, "reply_to_is_own_message", False):
                message_text = (
                    f'[Replying to your previous message: "{reply_snippet}"]\n\n'
                    f"{message_text}"
                )
            else:
                message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'

        if "@" in message_text:
            try:
                from agent.context_references import preprocess_context_references_async
                from agent.model_metadata import get_model_context_length_async

                _msg_cwd = os.environ.get("TERMINAL_CWD", os.path.expanduser("~"))
                _msg_config_ctx = None
                _msg_cfg = None
                _msg_model_cfg = {}
                _msg_custom_providers = []
                try:
                    _msg_cfg = _load_gateway_config()
                    _msg_model_cfg = _msg_cfg.get("model", {})
                    if isinstance(_msg_model_cfg, dict):
                        _msg_raw_ctx = _msg_model_cfg.get("context_length")
                        if _msg_raw_ctx is not None:
                            _msg_config_ctx = int(_msg_raw_ctx)
                    try:
                        from hermes_cli.config import get_compatible_custom_providers

                        _msg_custom_providers = get_compatible_custom_providers(_msg_cfg)
                    except Exception:
                        _msg_custom_providers = _msg_cfg.get("custom_providers") or []
                except Exception:
                    pass
                # Resolve the session's actual model/provider/base_url the
                # same way the hygiene compression block does (~11080).
                # GatewayRunner has no self._model/self._base_url attrs
                # (that was copy-pasted from HermesCLI, which does carry
                # self.model/self.base_url), so using them here always raised
                # AttributeError, silently caught below, meaning this feature
                # never ran.
                _msg_model, _msg_runtime = self._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                    user_config=_msg_cfg,
                )
                _msg_base_url = _msg_runtime.get("base_url") or ""
                # A global model.context_length belongs to the configured
                # model, not a session /model or channel override. Prefer a
                # matching per-custom-provider model limit when available.
                _msg_configured_model = (
                    _msg_model_cfg.get("default") or _msg_model_cfg.get("model")
                    if isinstance(_msg_model_cfg, dict)
                    else _msg_model_cfg
                )
                if _msg_model != _msg_configured_model:
                    _msg_config_ctx = None
                if _msg_config_ctx is not None and isinstance(_msg_model_cfg, dict):
                    try:
                        from hermes_cli.route_identity import should_clear_context_pin_async

                        if await should_clear_context_pin_async(
                            None,  # model match already checked above
                            None,
                            _msg_model_cfg.get("base_url"),
                            _msg_base_url,
                            _msg_model_cfg.get("provider"),
                            _msg_runtime.get("provider"),
                        ):
                            _msg_config_ctx = None
                    except Exception:
                        _msg_config_ctx = None
                if _msg_custom_providers and _msg_base_url:
                    try:
                        from hermes_cli.config import get_custom_provider_context_length

                        _msg_custom_ctx = get_custom_provider_context_length(
                            model=_msg_model,
                            base_url=_msg_base_url,
                            custom_providers=_msg_custom_providers,
                        )
                        if _msg_custom_ctx:
                            _msg_config_ctx = _msg_custom_ctx
                    except Exception:
                        pass
                _msg_ctx_len = await get_model_context_length_async(
                    _msg_model,
                    base_url=_msg_base_url,
                    api_key=_msg_runtime.get("api_key") or "",
                    config_context_length=_msg_config_ctx,
                    provider=_msg_runtime.get("provider") or "",
                    custom_providers=_msg_custom_providers,
                )
                _ctx_result = await preprocess_context_references_async(
                    message_text,
                    cwd=_msg_cwd,
                    context_length=_msg_ctx_len,
                    allowed_root=_msg_cwd,
                )
                if _ctx_result.blocked:
                    _adapter = self._adapter_for_source(source)
                    if _adapter:
                        await _adapter.send(
                            source.chat_id,
                            "\n".join(_ctx_result.warnings) or "Context injection refused.",
                        )
                    return None
                if _ctx_result.expanded:
                    message_text = _ctx_result.message
            except Exception as exc:
                logger.warning("@ context reference expansion failed: %s", exc)
                logger.debug("@ context reference expansion failure detail", exc_info=True)

        return message_text

    async def _prepare_inbound_message_for_turn(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Run inbound preprocessing for the process profile."""
        return await self._prepare_inbound_message_text(
            event=event,
            source=source,
            history=history,
            session_key=session_key,
        )

    async def _prepare_clarify_reply_text(self, event) -> str:
        """Return raw text or successful voice transcripts for a clarify reply."""
        if not self._pending_event_audio_paths(event):
            return (event.text or "").strip()

        return (event.text or "").strip()

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        state = self.sessions.peek(session_key)
        if state is None or not state.persistent.native_image_paths:
            return []
        paths = list(state.persistent.native_image_paths)
        state.persistent.native_image_paths = []
        return paths

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        # LRU: mark as most-recently-used and trim to max size.
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    @property
    def async_session_store(self) -> AsyncSessionStore:
        """Return the single async facade for this runner's SessionStore."""
        facade = getattr(self, "_async_session_store", None)
        if facade is None or facade._store is not self.session_store:
            facade = AsyncSessionStore(self.session_store)
            self._async_session_store = facade
        return facade

    async def _mark_durable_active_turn(
        self,
        event: "MessageEvent",
        session_key: str,
    ) -> bool:
        """Persist the exact resolved routing key for this running turn."""
        try:
            token = await self.async_session_store.mark_turn_active(session_key)
        except Exception as exc:
            logger.warning(
                "Could not persist active-turn marker for %s: %s",
                session_key,
                exc,
            )
            return False
        if not token:
            return False
        # Private event attributes are process-local ownership state.  Keep the
        # token out of public metadata, transcripts, and platform payloads.
        setattr(event, "_gateway_active_turn_session_key", session_key)
        setattr(event, "_gateway_active_turn_token", token)
        return True

    async def _clear_durable_active_turn(self, event: "MessageEvent") -> bool:
        """Best-effort CAS clear of the marker owned by *event*."""
        session_key = getattr(event, "_gateway_active_turn_session_key", None)
        token = getattr(event, "_gateway_active_turn_token", None)
        try:
            if not session_key or not token:
                return False
            last_error: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    return bool(
                        await self.async_session_store.clear_turn_active(
                            session_key, token
                        )
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        logger.debug(
                            "Retrying active-turn marker cleanup for %s (%d/3): %s",
                            session_key,
                            attempt,
                            exc,
                        )
            # Never let marker cleanup block in-memory agent/lease release.  A
            # stale marker is bounded by the configured agent timeout and the
            # clean-start orphan-marker discard path.
            logger.warning(
                "Could not clear active-turn marker for %s after 3 attempts: %s",
                session_key,
                last_error,
            )
            return False
        finally:
            for attr in (
                "_gateway_active_turn_session_key",
                "_gateway_active_turn_token",
            ):
                try:
                    delattr(event, attr)
                except AttributeError:
                    pass

    def _install_plugin_message_injector(self) -> None:
        """Publish this live gateway's plugin message scheduler."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().set_gateway_message_injector(
            self,
            self._schedule_plugin_message_injection,
        )

    def _clear_plugin_message_injector(self) -> None:
        """Remove this runner's scheduler without clobbering a newer owner."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().clear_gateway_message_injector(self)

    def _schedule_plugin_message_injection(
        self,
        *,
        session_key: str,
        content: str,
        plugin_id: str,
    ) -> bool:
        """Schedule a plugin-triggered turn on the live gateway loop."""
        loop = getattr(self, "_gateway_loop", None)
        if not getattr(self, "_running", False) or loop is None or loop.is_closed():
            return False

        coro = self._dispatch_plugin_message_injection(
            session_key=session_key,
            content=content,
            plugin_id=plugin_id,
        )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            try:
                future = loop.create_task(coro)
            except Exception:
                coro.close()
                logger.warning(
                    "Plugin message injection scheduling failed",
                    exc_info=True,
                )
                return False
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)
        else:
            future = safe_schedule_threadsafe(
                coro,
                loop,
                logger=logger,
                log_message="Plugin message injection scheduling failed",
                log_level=logging.WARNING,
            )
            if future is None:
                return False

        def _log_result(completed) -> None:
            try:
                accepted = completed.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            except Exception:
                logger.warning(
                    "Plugin message injection failed: plugin=%s session=%s",
                    plugin_id,
                    session_key,
                    exc_info=True,
                )
                return
            if not accepted:
                logger.warning(
                    "Plugin message injection was not routed: plugin=%s session=%s",
                    plugin_id,
                    session_key,
                )

        future.add_done_callback(_log_result)
        return True

    async def _dispatch_plugin_message_injection(
        self,
        *,
        session_key: str,
        content: str,
        plugin_id: str,
    ) -> bool:
        """Route a plugin-triggered turn through the session's live adapter."""
        if not getattr(self, "_running", False) or getattr(self, "_draining", False):
            return False

        entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None or entry.origin is None:
            return False
        if not getattr(self, "_running", False) or getattr(self, "_draining", False):
            return False

        source = dataclasses.replace(entry.origin)
        try:
            if not self._is_user_authorized(
                source,
                allow_adapter_delegation=False,
            ):
                logger.warning(
                    "Plugin message injection denied by current gateway authorization: "
                    "plugin=%s session=%s",
                    plugin_id,
                    session_key,
                )
                return False
        except Exception:
            logger.warning(
                "Plugin message injection authorization check failed: "
                "plugin=%s session=%s",
                plugin_id,
                session_key,
                exc_info=True,
            )
            return False

        adapter = self._adapter_for_source(source)
        if adapter is None:
            return False

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            allow_gateway_control=False,
            metadata={
                "hermes_plugin_id": plugin_id,
                "hermes_plugin_injection": True,
                "gateway_session_key": session_key,
                "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
            },
        )
        await adapter.handle_message(event)
        logger.info(
            "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
            plugin_id,
            session_key,
            entry.session_id,
        )
        return True

    def _get_cached_session_source(self, session_key: str):
        if not session_key:
            return None
        cached_sources = getattr(self, "_session_sources", None)
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            try:
                cached_sources.move_to_end(session_key)
            except Exception:
                pass
        return source

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        import agent.lifecycle as lifecycle
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _msg_preview = (event.text or "")[:80].replace("\n", " ")
        _reply_id = getattr(event, "reply_to_message_id", None)
        _reply_txt = (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " ")
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", _msg_preview, _reply_id, _reply_txt,
        )

        # Get or create session
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's
        # last-active topic so a cross-topic Reply or stripped plain reply
        # doesn't fragment the conversation across sessions.
        recovered = await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            try:
                event.source = source
            except Exception:
                pass

        event_metadata = getattr(event, "metadata", None) or {}
        expected_session_key = str(
            event_metadata.get("gateway_session_key") or ""
        ).strip()
        if expected_session_key:
            derived_session_key = self._session_key_for_source(source)
            if derived_session_key != expected_session_key:
                logger.warning(
                    "Dropping internally routed event after route recovery: "
                    "expected session=%s derived=%s",
                    expected_session_key,
                    derived_session_key,
                )
                return

        strict_session = bool(event_metadata.get("gateway_session_strict"))
        pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
        if strict_session:
            session_entry = await self.async_session_store.lookup_by_session_key(
                expected_session_key
            )
            if (
                session_entry is None
                or not pinned_session_id
                or session_entry.session_id != pinned_session_id
            ):
                logger.warning(
                    "Dropping internally routed event: expected session id=%s is no "
                    "longer current for key=%s",
                    pinned_session_id or "missing",
                    expected_session_key or "missing",
                )
                return
        else:
            # Internal wakes must observe reset policy without becoming user
            # activity themselves. Otherwise periodic Kanban/process
            # notifications keep the stable routing key alive across every
            # daily/idle boundary.
            session_entry = await self.async_session_store.get_or_create_session(
                source,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        session_key = session_entry.session_key
        if not strict_session and pinned_session_id:
            resolved_entry = await self._resolve_async_delegation_session(
                session_entry,
                pinned_session_id,
            )
            if resolved_entry is None:
                return
            session_entry = resolved_entry
        self._cache_session_source(session_key, source)
        if await asyncio.to_thread(self._is_telegram_topic_lane, source):
            try:
                binding = (await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )) if self._session_db else None
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                bound_session_id = str(binding.get("session_id") or "")
                # Heal bindings that point at a pre-compression parent: walk
                # the compression-continuation chain forward to its tip so the
                # next message resumes the compressed child instead of
                # reloading the oversized parent transcript (#20470/#29712/
                # #33414). Returns the input unchanged when the session isn't
                # a compression parent, so this is cheap and safe.
                if bound_session_id and self._session_db is not None:
                    try:
                        canonical_session_id = await self._session_db.get_compression_tip(
                            bound_session_id,
                        )
                    except Exception:
                        logger.debug(
                            "compression-tip lookup failed for %s",
                            bound_session_id, exc_info=True,
                        )
                        canonical_session_id = bound_session_id
                    if (
                        canonical_session_id
                        and canonical_session_id != bound_session_id
                    ):
                        bound_session_id = canonical_session_id
                if bound_session_id and bound_session_id != session_entry.session_id:
                    # Route the override through SessionStore so the session_key
                    # → session_id mapping is persisted to disk and the previous
                    # lane session is ended cleanly. Mutating session_entry in
                    # place here created a split-brain state where the JSON
                    # index pointed at one id but code downstream used another.
                    switched = await self.async_session_store.switch_session(session_key, bound_session_id)
                    if switched is not None:
                        session_entry = switched
                # If the stored binding pointed at a parent, rewrite it to the
                # canonical descendant now that we've followed the chain.
                if (
                    bound_session_id
                    and bound_session_id != str(binding.get("session_id") or "")
                ):
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-tip-walk",
                    )
            else:
                try:
                    await asyncio.to_thread(self._record_telegram_topic_binding, source, session_entry)
                except Exception:
                    logger.debug("Failed to record Telegram topic binding", exc_info=True)
        # Capture and immediately consume was_auto_reset so it does not
        # re-fire on subsequent messages — preventing the cleanup from
        # wiping model/reasoning overrides set between turns (Closes #48031).
        _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
        if _was_auto_reset:
            # Treat auto-reset as a full conversation boundary — clear every
            # conversation-scoped per-session dict in one funnel call so the
            # fresh session does not inherit the previous conversation's
            # model/reasoning overrides, a queued "/model switched" note, or
            # a stale resolved-model cache (#48031, #58403). See
            # _CONVERSATION_SCOPED_STATE.
            self._clear_conversation_scope(session_key, reason="auto_reset")
            # Evict the cached agent so the fresh session does not inherit the
            # previous conversation's context_compressor._previous_summary —
            # the cache is keyed on the stable session_key, so an auto-reset
            # otherwise reuses the old agent and leaks prior history into new
            # compaction summaries. Mirrors /reset and the compression-exhausted
            # path (#9893). Covers daily/idle/suspended auto-reset.
            self.agent_cache.evict(session_key)
            session_entry.was_auto_reset = False

        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or _was_auto_reset
            or getattr(session_entry, "is_fresh_reset", False)
        )
        # Consume the is_fresh_reset flag immediately so it doesn't leak
        # onto subsequent messages in the same session (issue #6508).
        if getattr(session_entry, "is_fresh_reset", False):
            session_entry.is_fresh_reset = False
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })

        # Build session context
        context = build_session_context(source, self.config, session_entry)

        # Set session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)

        # Read privacy.redact_pii from config (re-read per message)
        _redact_pii = False
        persist_user_message = None
        persist_user_timestamp = None
        # Synthetic self-injected turns (async-delegation batch completions,
        # background watch notifications, resume wake-ups) arrive as
        # MessageEvent(internal=True). Persist their user row typed with
        # display_kind="internal_notification" so transcripts/UIs can render
        # them as timeline notices instead of user bubbles (#82888). Role and
        # content are untouched — display_kind is a DB-only sidecar stripped
        # from every provider-bound payload (see conversation_loop's
        # api_msg.pop("display_kind")).
        persist_user_display_kind = (
            "internal_notification" if getattr(event, "internal", False) else None
        )
        try:
            _pcfg = _load_gateway_config()
            _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # Build the context prompt to inject.  The render is pinned per
        # session, keyed by a hash of the exact renderer inputs
        # (_ephemeral_change_key).  A key hit reuses the pinned bytes verbatim
        # so the composed system prompt cannot drift turn-over-turn; a key
        # miss (thread rename, /sethome, redact_pii flip, ...) re-renders
        # once — the only legitimate cache busts.
        context_prompt = self._pinned_session_context_prompt(
            context, _redact_pii, session_key
        )

        # Per-turn must-deliver notes.  These used to be appended to
        # context_prompt (the ephemeral system prompt), which guaranteed a
        # turn1→turn2 system-prompt diff and a full agent rebuild.  They now
        # ride the current user message via the api_content sidecar instead
        # (staged below, consumed in run_sync → build_turn_context).
        turn_sidecar_notes: List[str] = []

        # If the previous session expired and was auto-reset, deliver a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if _was_auto_reset:
            reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
            if reset_reason == "suspended":
                context_note = "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]"
            elif reset_reason == "daily":
                context_note = "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]"
            elif reset_reason == "resume_pending_expired":
                context_note = "[System note: The previous gateway session could not be recovered after a restart (API recovery timed out). This is a fresh conversation — use /resume to restore history if needed.]"
            else:
                context_note = "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]"
            turn_sidecar_notes.append(context_note)

            # Send a user-facing notification explaining the reset, unless:
            # - notifications are disabled in config
            # - the platform is excluded
            # - the expired session had no activity (nothing was cleared)
            try:
                policy = self.session_store.config.get_reset_policy(
                    platform=source.platform,
                    session_type=getattr(source, 'chat_type', 'dm'),
                )
                platform_name = source.platform.value if source.platform else ""
                had_activity = getattr(session_entry, 'reset_had_activity', False)
                # Suspended and restart-recovery-expired sessions always notify
                # regardless of policy.notify — the user had an active session
                # that was silently replaced, so they need to know they can
                # /resume it.  Idle/daily resets respect the policy flag.
                should_notify = reset_reason in {"suspended", "resume_pending_expired"} or (
                    policy.notify
                    and had_activity
                    and platform_name not in policy.notify_exclude_platforms
                )
                if should_notify:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        if reset_reason == "suspended":
                            reason_text = "previous session was stopped or interrupted"
                        elif reset_reason == "resume_pending_expired":
                            reason_text = "gateway restart recovery timed out"
                        elif reset_reason == "daily":
                            reason_text = f"daily schedule at {policy.at_hour}:00"
                        else:
                            hours = policy.idle_minutes // 60
                            mins = policy.idle_minutes % 60
                            duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
                            reason_text = f"inactive for {duration}"
                        notice = (
                            f"◐ Session automatically reset ({reason_text}). "
                            f"Conversation history cleared.\n"
                            f"Use /resume to browse and restore a previous session.\n"
                            f"Adjust reset timing in config.yaml under session_reset."
                        )
                        try:
                            session_info = await asyncio.to_thread(
                                self._reset_notice_session_info, source
                            )
                            if session_info:
                                notice = f"{notice}\n\n{session_info}"
                        except Exception:
                            pass
                        await adapter.send(
                            source.chat_id, notice,
                            metadata=self._thread_metadata_for_source(source),
                        )
            except Exception as e:
                logger.debug("Auto-reset notification failed (non-fatal): %s", e)

            # was_auto_reset is already consumed in the cleanup block above
            # (single source of truth); only the reset reason needs clearing here.
            session_entry.auto_reset_reason = None

        # Auto-load skill(s) for Telegram topic or Mattermost channel bindings.
        # Supports a single name or ordered list.
        # Only inject on NEW sessions — ongoing conversations already have the
        # skill content in their conversation history from the first message.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
            try:
                from agent.skill_commands import _load_skill_payload, _build_skill_message
                _combined_parts: list[str] = []
                _loaded_names: list[str] = []
                for _sname in _skill_names:
                    _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                    if _loaded:
                        _loaded_skill, _skill_dir, _display_name = _loaded
                        _note = (
                            f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                            f"Follow its instructions for this session.]"
                        )
                        _part = _build_skill_message(_loaded_skill, _skill_dir, _note)
                        if _part:
                            _combined_parts.append(_part)
                            _loaded_names.append(_sname)
                    else:
                        logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                if _combined_parts:
                    # Append the user's original text after all skill payloads
                    _combined_parts.append(event.text)
                    event.text = "\n\n".join(_combined_parts)
                    logger.info(
                        "[Gateway] Auto-loaded skill(s) %s for session %s",
                        _loaded_names, session_key,
                    )
            except Exception as e:
                logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

        # ── Turn lease (#64934) ────────────────────────────────────────
        # Session resolution is FINAL here (get_or_create → async-delegation
        # pinning → topic tip-walk switch_session are all above). Serialize
        # the [load history → run → flush] region per resolved SESSION_ID:
        # when a second routing key is mapped to this same session_id, its
        # turn waits here for the previous turn's flush instead of loading a
        # stale history base and interleaving transcript writes. Same-key
        # messages never reach this point mid-turn (adapter + runner guards
        # hold them), so the lock is uncontended outside the alias-key route.
        # Fail-closed on timeout: never enter the transcript region without a
        # lease. Outer dispatch returns a bounded rejection/resend notice rather
        # than recreating the exact concurrent-turn corruption this lease exists
        # to prevent. Released in _handle_message's finally via
        # _release_turn_lease — granted per (routing key, run generation) so a
        # stale unwind can't release a newer turn's lease.
        _lease_registry = self.sessions.turn_leases
        if _lease_registry is not None:
            try:
                _lease_token = await _lease_registry.acquire(
                    session_entry.session_id,
                    owner_key=_quick_key,
                    generation=run_generation,
                    timeout=_float_env(
                        "HERMES_TURN_LEASE_TIMEOUT", DEFAULT_LEASE_WAIT
                    ),
                )
            except TurnLeaseTimeoutError:
                # The broad session-context cleanup finally starts later in this
                # method. Restore the tokens here before propagating the rejection
                # to outer dispatch, or this early exit leaks task-local identity.
                self._clear_session_env(_session_env_tokens)
                raise
            if _lease_token is not None:
                _lease_state = self.sessions.state(_quick_key).turn
                _lease_state.lease_token = _lease_token
                _lease_state.lease_generation = run_generation

        # A turn only becomes durable recovery work after it owns (or has
        # explicitly degraded past) the per-session lease.  Marking before the
        # await above would falsely recover an alias-routed message that never
        # began processing if the gateway died while it was still waiting.
        await self._mark_durable_active_turn(event, session_entry.session_key)

        # Load conversation history from transcript
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        # -----------------------------------------------------------------
        # Session hygiene: auto-compress pathologically large transcripts
        #
        # Long-lived gateway sessions can accumulate enough history that
        # every new message rehydrates an oversized transcript, causing
        # repeated truncation/context failures.  Detect this early and
        # compress proactively — before the agent even starts.  (#628)
        #
        # Token source priority:
        # 1. Actual API-reported prompt_tokens from the last turn
        #    (stored in session_entry.last_prompt_tokens)
        # 2. Rough char-based estimate (str(msg)//4). Overestimates
        #    by 30-50% on code/JSON-heavy sessions, but that just
        #    means hygiene fires a bit early — safe and harmless.
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length_async,
            )

            # Read model + compression config from config.yaml.
            # NOTE: hygiene threshold is intentionally HIGHER than the agent's
            # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
            # sessions that grew too large between turns — it fires pre-agent
            # to prevent API failures.  The agent's own compressor handles
            # normal context management during its tool loop with accurate
            # real token counts.  Having hygiene at 0.50 caused premature
            # compression on every turn in long gateway sessions.
            _hyg_model = "anthropic/claude-sonnet-4.6"
            _hyg_threshold_pct = 0.85
            _hyg_compression_enabled = True
            _hyg_hard_msg_limit = 5000
            _hyg_timeout_seconds = 30.0
            _hyg_total_ceiling_seconds = 600.0
            _hyg_failure_cooldown_seconds = 300.0
            _hyg_config_context_length = None
            _hyg_provider = None
            _hyg_base_url = None
            _hyg_api_key = None
            _hyg_configured_model = None
            _hyg_configured_provider = None
            _hyg_configured_base_url = None
            _hyg_data = {}
            try:
                _hyg_data = _load_gateway_config()
                if _hyg_data:
                    # Resolve model name (same logic as run_sync)
                    _model_cfg = _hyg_data.get("model", {})
                    if isinstance(_model_cfg, str):
                        _hyg_model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        _hyg_model = _model_cfg.get("default") or _model_cfg.get("model") or _hyg_model
                        # Read explicit context_length override from model config
                        # (same as run_agent.py lines 995-1005)
                        _raw_ctx = _model_cfg.get("context_length")
                        if _raw_ctx is not None:
                            try:
                                _hyg_config_context_length = int(_raw_ctx)
                            except (TypeError, ValueError):
                                pass
                        # Read provider for accurate context detection
                        _hyg_provider = _model_cfg.get("provider") or None
                        _hyg_base_url = _model_cfg.get("base_url") or None

                    # Read compression settings — only use enabled flag.
                    # The threshold is intentionally separate from the agent's
                    # compression.threshold (hygiene runs higher).
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in {"true", "1", "yes"}
                        _raw_hard_limit = _comp_cfg.get("hygiene_hard_message_limit")
                        if _raw_hard_limit is not None:
                            try:
                                _parsed = int(_raw_hard_limit)
                                if _parsed > 0:
                                    _hyg_hard_msg_limit = _parsed
                            except (TypeError, ValueError):
                                pass
                        _raw_timeout = _comp_cfg.get("hygiene_timeout_seconds")
                        if _raw_timeout is not None:
                            try:
                                _parsed = float(_raw_timeout)
                                if _parsed > 0:
                                    _hyg_timeout_seconds = _parsed
                            except (TypeError, ValueError):
                                pass
                        _raw_ceiling = _comp_cfg.get("hygiene_total_ceiling_seconds")
                        if _raw_ceiling is not None:
                            try:
                                _parsed = float(_raw_ceiling)
                                if _parsed > 0:
                                    _hyg_total_ceiling_seconds = _parsed
                            except (TypeError, ValueError):
                                pass
                        # The ceiling can never be tighter than one idle
                        # window, or the extension loop would be dead code.
                        _hyg_total_ceiling_seconds = max(
                            _hyg_total_ceiling_seconds, _hyg_timeout_seconds,
                        )
                        _raw_cooldown = _comp_cfg.get("hygiene_failure_cooldown_seconds")
                        if _raw_cooldown is not None:
                            try:
                                _parsed = float(_raw_cooldown)
                                if _parsed >= 0:
                                    _hyg_failure_cooldown_seconds = _parsed
                            except (TypeError, ValueError):
                                pass

                _hyg_configured_model = _hyg_model
                _hyg_configured_provider = _hyg_provider
                _hyg_configured_base_url = _hyg_base_url

                try:
                    _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                    )
                    _hyg_provider = _hyg_runtime.get("provider") or _hyg_provider
                    _hyg_base_url = _hyg_runtime.get("base_url") or _hyg_base_url
                    _hyg_api_key = _hyg_runtime.get("api_key") or _hyg_api_key
                except Exception:
                    pass

                if _hyg_config_context_length is not None:
                    try:
                        from hermes_cli.route_identity import should_clear_context_pin_async

                        if await should_clear_context_pin_async(
                            _hyg_configured_model,
                            _hyg_model,
                            _hyg_configured_base_url,
                            _hyg_base_url,
                            _hyg_configured_provider,
                            _hyg_provider,
                        ):
                            _hyg_config_context_length = None
                    except Exception:
                        _hyg_config_context_length = None

                # Check custom_providers per-model context_length
                # (same fallback as run_agent.py lines 1171-1189).
                # Must run after runtime resolution so _hyg_base_url is set.
                if _hyg_config_context_length is None and _hyg_base_url:
                    try:
                        try:
                            from hermes_cli.config import (
                                get_compatible_custom_providers as _gw_gcp,
                                get_custom_provider_context_length as _gw_gccl,
                            )
                            _hyg_custom_providers = _gw_gcp(_hyg_data)
                        except Exception:
                            _hyg_custom_providers = _hyg_data.get("custom_providers")
                            if not isinstance(_hyg_custom_providers, list):
                                _hyg_custom_providers = []
                        _hyg_custom_ctx = _gw_gccl(
                            model=_hyg_model,
                            base_url=_hyg_base_url,
                            custom_providers=_hyg_custom_providers,
                        )
                        if _hyg_custom_ctx:
                            _hyg_config_context_length = int(_hyg_custom_ctx)
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

            if _hyg_compression_enabled:
                _hyg_context_length = await get_model_context_length_async(
                    _hyg_model,
                    base_url=_hyg_base_url or "",
                    api_key=_hyg_api_key or "",
                    config_context_length=_hyg_config_context_length,
                    provider=_hyg_provider or "",
                )
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)

                # Prefer actual API-reported tokens from the last turn
                # (stored in session entry) over the rough char-based estimate.
                _stored_tokens = session_entry.last_prompt_tokens
                if _stored_tokens > 0:
                    _approx_tokens = _stored_tokens
                    _token_source = "actual"
                else:
                    _approx_tokens = estimate_messages_tokens_rough(history)
                    _token_source = "estimated"
                    # Note: rough estimates overestimate by 30-50% for code/JSON-heavy
                    # sessions, but that just means hygiene fires a bit early — which
                    # is safe and harmless.  The 85% threshold already provides ample
                    # headroom (agent's own compressor runs at 50%).  A previous 1.4x
                    # multiplier tried to compensate by inflating the threshold, but
                    # 85% * 1.4 = 119% of context — which exceeds the model's limit
                    # and prevented hygiene from ever firing for ~200K models (GLM-5).

                # Hard safety valve: force compression if message count is
                # extreme, regardless of token estimates.  This breaks the
                # death spiral where API disconnects prevent token data
                # collection, which prevents compression, which causes more
                # disconnects.  5000 messages is far above any normal session
                # but catches truly runaway growth before it becomes
                # unrecoverable.  Set well clear of legitimate large-context
                # (1M+) sessions doing thousands of short turns — those
                # compress on the token threshold, not this count-based floor.
                # Threshold is configurable via
                # compression.hygiene_hard_message_limit.
                # (#2153)
                _HARD_MSG_LIMIT = _hyg_hard_msg_limit
                _needs_compress = (
                    _approx_tokens >= _compress_token_threshold
                    or _msg_count >= _HARD_MSG_LIMIT
                )

                if _needs_compress:
                    # Use the persistent DB-backed cooldown (same as the
                    # in-conversation compression path in context_compressor.py)
                    # so the cooldown survives gateway restarts. The in-memory
                    # dict was reset on every restart, re-triggering the same
                    # failing compression and wedging session storage (#74136).
                    _session_db = getattr(self, "_session_db", None)
                    if _session_db is not None:
                        _session_db = getattr(_session_db, "_db", _session_db)
                        _getter = getattr(_session_db, "get_compression_failure_cooldown", None)
                        if _getter is not None:
                            try:
                                _cooldown_state = _getter(session_entry.session_id)
                            except Exception:
                                _cooldown_state = None
                            if _cooldown_state and _cooldown_state.get("remaining_seconds", 0) > 0:
                                logger.info(
                                    "Session hygiene: skipping compression for %s; "
                                    "previous failure cooldown active for %.1fs",
                                    session_entry.session_id,
                                    _cooldown_state["remaining_seconds"],
                                )
                                _needs_compress = False

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}", _token_source,
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

                    try:
                        from agent.conversation_compression import CompressionCommitFence
                        from agent.agent_init import create_agent

                        _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                            source=source,
                            session_key=session_key,
                            user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                        )
                        if _hyg_runtime.get("api_key"):
                            # Pass the FULL transcript (tool results included).
                            # Filtering to user/assistant-only starved the
                            # compressor: tool results are usually the bulk of
                            # the context, _prune_old_tool_results never saw
                            # them, and short filtered histories tripped the
                            # protect-first/last early-return so nothing was
                            # compressed at all (#3854). The agent loop passes
                            # its full message list to _compress_context — the
                            # gateway now matches.
                            _hyg_msgs = [
                                m for m in history
                                if m.get("role") in {"user", "assistant", "tool"}
                            ]

                            if len(_hyg_msgs) >= 4:
                                try:
                                    _hyg_session_row = await self._session_db.get_session(
                                        session_entry.session_id
                                    )
                                except Exception as exc:
                                    _hyg_session_row = None
                                    logger.warning(
                                        "Session hygiene could not restore the system "
                                        "prompt for session %s: %s. Preserving an empty "
                                        "prompt so the live turn rebuilds it with its "
                                        "configured providers.",
                                        session_entry.session_id,
                                        exc,
                                        exc_info=True,
                                    )
                                _hyg_session_db = getattr(self._session_db, "_db", self._session_db)
                                _hyg_agent = create_agent(
                                    **_hyg_runtime,
                                    model=_hyg_model,
                                    max_iterations=4,
                                    quiet_mode=True,
                                    skip_memory=True,
                                    enabled_toolsets=["memory"],
                                    session_id=session_entry.session_id,
                                    session_db=_hyg_session_db,
                                )
                                _seed_hygiene_system_prompt(
                                    _hyg_agent,
                                    _hyg_session_row,
                                )
                                # If compression must rebuild instead of retaining
                                # the cached prompt, make the persisted result
                                # deliberately stale for every real gateway surface.
                                _hyg_agent.platform = _GATEWAY_HYGIENE_PLATFORM
                                _hyg_cleanup_deferred = False
                                try:
                                    # Gateway hygiene runs before the user turn
                                    # starts and already owns the session binding.
                                    # Prefer in-place compaction here: it archives
                                    # old rows under the same session id instead of
                                    # minting a continuation child that then has to
                                    # be published back to SessionStore/topic
                                    # bindings.  If no SessionDB is available,
                                    # compress_context leaves this flag false and
                                    # the guard below preserves the transcript.
                                    _hyg_agent.compression_in_place = True
                                    _bind_hyg_state = getattr(
                                        getattr(_hyg_agent, "context_compressor", None),
                                        "bind_session_state",
                                        None,
                                    )
                                    if callable(_bind_hyg_state):
                                        _bind_hyg_state(
                                            _hyg_session_db,
                                            session_entry.session_id,
                                        )
                                    # It must never finalize on close() — close()
                                    # would end the live gateway session row.
                                    _hyg_agent._end_session_on_close = False
                                    _hyg_agent._print_fn = lambda *a, **kw: None

                                    loop = asyncio.get_running_loop()
                                    _hyg_commit_fence = CompressionCommitFence()
                                    _hyg_future = loop.run_in_executor(
                                        None,
                                        lambda: lifecycle._compress_context(_hyg_agent,
                                            _hyg_msgs, "",
                                            approx_tokens=_approx_tokens,
                                            commit_fence=_hyg_commit_fence,
                                        ),
                                    )
                                    try:
                                        # Progress-aware wait: the timeout is an
                                        # INACTIVITY budget, not a total one. The
                                        # compression worker streams its summary
                                        # call and ticks the fence per token
                                        # (CompressionCommitFence.touch_progress),
                                        # so a slow reasoning model that is still
                                        # generating keeps extending the deadline;
                                        # only a genuinely silent worker times out.
                                        # A hard ceiling bounds the total wait so
                                        # a degenerate trickle stream can't hold
                                        # the turn forever.
                                        _hyg_wait_started = time.monotonic()
                                        while True:
                                            # #76354 S3: charge the idle budget
                                            # from the LAST PROGRESS event, not
                                            # from the start of this wait slice —
                                            # otherwise silence can approach 2x
                                            # the configured timeout.
                                            _slice = max(
                                                _hyg_timeout_seconds
                                                - _hyg_commit_fence.seconds_since_progress(),
                                                0.005,
                                            )
                                            try:
                                                _compressed, _ = await asyncio.wait_for(
                                                    asyncio.shield(_hyg_future),
                                                    timeout=_slice,
                                                )
                                                break
                                            except asyncio.TimeoutError:
                                                _hyg_waited = time.monotonic() - _hyg_wait_started
                                                _idle = _hyg_commit_fence.seconds_since_progress()
                                                if (
                                                    _idle < _hyg_timeout_seconds
                                                    and _hyg_waited < _hyg_total_ceiling_seconds
                                                ):
                                                    logger.info(
                                                        "Session hygiene compression for "
                                                        "session %s still streaming after "
                                                        "%.0fs (last progress %.1fs ago) — "
                                                        "extending wait (ceiling %.0fs)",
                                                        session_entry.session_id,
                                                        _hyg_waited, _idle,
                                                        _hyg_total_ceiling_seconds,
                                                    )
                                                    continue
                                                raise
                                    except asyncio.TimeoutError:
                                        _cancelled = None
                                        while _cancelled is None:
                                            # #76354 F1: a hung commit retains the
                                            # fence lock; the lock-free phase
                                            # marker keeps this loop from spinning
                                            # forever while the commit blocks.
                                            if _hyg_commit_fence.commit_in_flight:
                                                _cancelled = False
                                                break
                                            _cancelled = (
                                                _hyg_commit_fence.try_cancel_before_commit()
                                            )
                                            if _cancelled is None:
                                                # Round-2 #5: transient
                                                # lock-setup windows ride
                                                # write patience for seconds;
                                                # 25ms keeps sub-tick latency
                                                # without 1kHz spin.
                                                await asyncio.sleep(0.025)
                                        if not _cancelled:
                                            # The worker crossed the commit boundary just
                                            # before the timeout. The fence poll waited for
                                            # that boundary to finish, so consume the
                                            # completed result instead of treating a
                                            # successful compaction as a timeout.
                                            _compressed, _ = await _hyg_future
                                        else:
                                            # #76354 F4: release the timed-out
                                            # worker's durable lease via the
                                            # holder-qualified hook so the next
                                            # compressor can acquire the lock
                                            # immediately (no ABA against a new
                                            # holder — release is holder-scoped).
                                            _hyg_commit_fence.release_cancelled_compression_lock()
                                            self._defer_agent_cleanup_until_future_done(
                                                _hyg_future,
                                                _hyg_agent,
                                                context="session hygiene timeout",
                                            )
                                            _hyg_cleanup_deferred = True
                                            if _hyg_failure_cooldown_seconds >= 0:
                                                _hyg_cooldown = await asyncio.to_thread(
                                                    _hygiene_cooldown_for_failure,
                                                    self,
                                                    session_key,
                                                    _hyg_failure_cooldown_seconds,
                                                )
                                                _record_hygiene_cooldown(
                                                    self, session_entry.session_id,
                                                    _hyg_cooldown,
                                                    "session hygiene compression "
                                                    "timed out with no output from "
                                                    "the summary model",
                                                )
                                            from agent.session_activity import (
                                                ActivityProvenance,
                                            )
                                            _stamp_hygiene_compression_provenance(
                                                _hyg_agent,
                                                "session hygiene compression timed out",
                                                ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
                                                "hygiene compression timeout "
                                                "activity stamp failed",
                                            )
                                            logger.warning(
                                                "Session hygiene compression for session %s "
                                                "made no progress for %.1fs "
                                                "(total wait %.1fs, ceiling %.1fs); "
                                                "continuing without compression",
                                                session_entry.session_id,
                                                _hyg_commit_fence.seconds_since_progress(),
                                                time.monotonic() - _hyg_wait_started,
                                                _hyg_total_ceiling_seconds,
                                            )
                                            _timeout_msg = (
                                                "⚠️ Context compression timed out "
                                                f"after {_hyg_timeout_seconds:.1f}s "
                                                "with no output from the summary model. "
                                                "No messages were dropped — continuing without "
                                                "compression. Run /compress to retry, /reset for "
                                                "a clean session, or check your "
                                                "auxiliary.compression model configuration."
                                            )
                                            try:
                                                _adapter = self._adapter_for_source(source)
                                                if _adapter and source.chat_id:
                                                    await _adapter.send(
                                                        source.chat_id,
                                                        _timeout_msg,
                                                        metadata=_hyg_meta,
                                                    )
                                            except Exception as _werr:
                                                logger.warning(
                                                    "Failed to deliver compression-timeout "
                                                    "warning to user: %s",
                                                    _werr,
                                                )
                                            raise
                                    except BaseException:
                                        # #76354 F2: non-timeout unwind while the
                                        # detached hygiene worker may still run —
                                        # KeyboardInterrupt, task cancellation, or
                                        # any unexpected error. Revoke commit
                                        # admission (and release the worker's
                                        # durable lease via the holder-qualified
                                        # hook) BEFORE the host unwinds so the
                                        # worker can never commit later.
                                        _hyg_commit_fence.revoke_commit_admission()
                                        if not _hyg_cleanup_deferred:
                                            self._defer_agent_cleanup_until_future_done(
                                                _hyg_future,
                                                _hyg_agent,
                                                context="session hygiene unwind",
                                            )
                                            _hyg_cleanup_deferred = True
                                        raise

                                    # _compress_context ends the old session and creates
                                    # a new session_id.  Write compressed messages into
                                    # the NEW session so the old transcript stays intact
                                    # and searchable via session_search.
                                    _hyg_new_sid = _hyg_agent.session_id
                                    _hyg_rotated = _hyg_new_sid != session_entry.session_id
                                    _hyg_in_place = bool(
                                        getattr(_hyg_agent, "_last_compaction_in_place", False)
                                    )
                                    # Anti-growth guard: refuse a compression
                                    # that did not shrink the transcript
                                    # (observed: 427K -> 598K). Compare
                                    # like-for-like rough estimates.
                                    _hyg_in_toks = estimate_messages_tokens_rough(history)
                                    _hyg_out_toks = estimate_messages_tokens_rough(_compressed)
                                    if _hyg_rotated and _hyg_out_toks > _hyg_in_toks:
                                        logger.warning(
                                            "Gateway hygiene compression for session %s "
                                            "would grow transcript (~%s -> ~%s tokens); "
                                            "keeping the original transcript unchanged",
                                            session_entry.session_id,
                                            f"{_hyg_in_toks:,}",
                                            f"{_hyg_out_toks:,}",
                                        )
                                        _hyg_rotated = False
                                        _compressed = history
                                    # Only rewrite the transcript when rotation produced
                                    # a NEW session id.  In-place compaction does NOT
                                    # need a rewrite: archive_and_compact() has already
                                    # soft-archived the previous active rows and inserted
                                    # the compacted messages as the new active set inside
                                    # _compress_context().  Calling rewrite_transcript()
                                    # after in-place compaction would invoke
                                    # replace_messages(active_only=False) which DELETEs
                                    # ALL rows — including the archived turns that
                                    # archive_and_compact() deliberately preserved
                                    # (silent data loss, #61145).
                                    #
                                    # The danger this guards against (mirrors the
                                    # /compress fix #44794/#39704): if _compress_context
                                    # returns a summary but neither rotates nor completes
                                    # archive_and_compact(), the session_id is unchanged
                                    # for a FAILURE reason, and an unconditional
                                    # rewrite_transcript() would DELETE the original
                                    # messages and replace them with only the compressed
                                    # summary (permanent data loss, #21301).
                                    #
                                    # Write-before-repoint (mirrors manual /compress):
                                    # if we repointed session_entry onto the child SID
                                    # and rewrite_transcript then failed (lock/ENOSPC),
                                    # the live entry would already reference a brand-new
                                    # empty session while the turn continues — the
                                    # conversation silently vanishes. Persist the child
                                    # transcript first; only then rebind the live entry.
                                    if _hyg_rotated:
                                        if not await self.async_session_store.rewrite_transcript(
                                            _hyg_new_sid, _compressed
                                        ):
                                            logger.error(
                                                "Session hygiene: failed to persist "
                                                "compressed transcript for rotated "
                                                "session %s → %s; keeping the live "
                                                "entry on the original session so the "
                                                "conversation is not dropped",
                                                session_entry.session_id,
                                                _hyg_new_sid,
                                            )
                                            # Fail closed: treat like no rotation.
                                            _hyg_rotated = False
                                            _hyg_in_place = False
                                        else:
                                            session_entry.session_id = _hyg_new_sid
                                            # The held turn lease follows the
                                            # rotation so an alias key resolving
                                            # the fresh child still serializes
                                            # against this turn (#64934).
                                            self.sessions.rebind_turn_lease(
                                                _quick_key, run_generation, _hyg_new_sid
                                            )
                                            await self.async_session_store._save()
                                            await asyncio.to_thread(
                                                self._sync_telegram_topic_binding,
                                                source, session_entry,
                                                reason="hygiene-compression",
                                            )

                                    if _hyg_rotated:
                                        # Reset stored token count — transcript rewritten
                                        session_entry.last_prompt_tokens = 0
                                        history = _compressed
                                        _new_count = len(_compressed)
                                        _new_tokens = estimate_messages_tokens_rough(
                                            _compressed
                                        )
                                    elif _hyg_in_place:
                                        # archive_and_compact() already persisted the
                                        # compacted transcript inside _compress_context.
                                        # Reset counts to match the new active set.
                                        session_entry.last_prompt_tokens = 0
                                        history = _compressed
                                        _new_count = len(_compressed)
                                        _new_tokens = estimate_messages_tokens_rough(
                                            _compressed
                                        )
                                    else:
                                        # No rewrite happened — transcript preserved
                                        # unchanged, so the post-compression counts equal
                                        # the pre-compression ones.
                                        _new_count = _msg_count
                                        _new_tokens = _approx_tokens
                                        logger.warning(
                                            "Gateway hygiene compression for session %s "
                                            "did not rotate or compact in place "
                                            "(no session_db on the hygiene agent) — "
                                            "preserving the original transcript instead "
                                            "of overwriting it with the summary (#21301).",
                                            session_entry.session_id,
                                        )

                                    logger.info(
                                        "Session hygiene: compressed %s → %s msgs, "
                                        "~%s → ~%s tokens",
                                        _msg_count, _new_count,
                                        f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                    )

                                    if _new_tokens >= _warn_token_threshold:
                                        logger.warning(
                                            "Session hygiene: still ~%s tokens after "
                                            "compression",
                                            f"{_new_tokens:,}",
                                        )

                                    # If summary generation failed, the
                                    # compressor aborts entirely and returns
                                    # messages unchanged — nothing is dropped.
                                    # Surface a visible warning to the gateway
                                    # user — agent.log alone is invisible on
                                    # TG/Discord/etc. — so they know the chat
                                    # is "frozen" at the current size and can
                                    # /compress to retry or /reset to start
                                    # fresh.
                                    _comp = getattr(_hyg_agent, "context_compressor", None)
                                    _hyg_aborted = _comp is not None and getattr(
                                        _comp, "_last_compress_aborted", False
                                    )
                                    if not _hyg_aborted:
                                        # Recovery decision lives in the
                                        # extracted, unit-tested predicate — the
                                        # degenerate "did not rotate or compact
                                        # in place" path (#21301) sets both flags
                                        # False and reuses the pre-compression
                                        # counts, so a numbers-only check would
                                        # read a no-op as success and clear the
                                        # streak on every wedged run (#79624).
                                        if hygiene_compaction_recovered(
                                            aborted=_hyg_aborted,
                                            rotated=_hyg_rotated,
                                            in_place=_hyg_in_place,
                                            msg_count=_msg_count,
                                            new_count=_new_count,
                                            approx_tokens=_approx_tokens,
                                            new_tokens=_new_tokens,
                                        ):
                                            await asyncio.to_thread(
                                                _reset_hygiene_failure_streak,
                                                self,
                                                session_key,
                                            )
                                    if _hyg_aborted:
                                        if _hyg_failure_cooldown_seconds >= 0:
                                            _hyg_cooldown = await asyncio.to_thread(
                                                _hygiene_cooldown_for_failure,
                                                self,
                                                session_key,
                                                _hyg_failure_cooldown_seconds,
                                            )
                                            _record_hygiene_cooldown(
                                                self, session_entry.session_id,
                                                _hyg_cooldown,
                                                getattr(
                                                    _comp, "_last_summary_error", None
                                                ),
                                            )
                                        from agent.session_activity import (
                                            ActivityProvenance,
                                        )
                                        _stamp_hygiene_compression_provenance(
                                            _hyg_agent,
                                            "session hygiene compression aborted",
                                            ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
                                            "hygiene compression abort "
                                            "activity stamp failed",
                                        )
                                        _err = getattr(_comp, "_last_summary_error", None) or "unknown error"
                                        # Force-redact: provider exception text
                                        # may contain credentials; this message
                                        # reaches gateway users directly.
                                        from agent.redact import redact_sensitive_text
                                        _err = redact_sensitive_text(_err, force=True)
                                        _warn_msg = (
                                            "⚠️ Context compression aborted "
                                            f"({_err}). No messages were dropped — "
                                            "conversation is unchanged. Run /compress "
                                            "to retry, /reset for a clean session, or "
                                            "check your auxiliary.compression model "
                                            "configuration."
                                        )
                                        try:
                                            _adapter = self._adapter_for_source(source)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _warn_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver compression-failure warning to user: %s",
                                                _werr,
                                            )
                                    # Separately: if the user's CONFIGURED aux
                                    # model failed and we recovered by falling
                                    # back to the main model, tell them — a
                                    # misconfigured auxiliary.compression.model
                                    # is something only they can fix, and
                                    # silent recovery would hide it.
                                    elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
                                        _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
                                        _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
                                        _aux_msg = (
                                            f"ℹ️ Configured compression model `{_aux_model}` "
                                            f"failed ({_aux_err}). Recovered using your main "
                                            "model — context is intact — but you may want to "
                                            "check `auxiliary.compression.model` in config.yaml."
                                        )
                                        try:
                                            _adapter = self._adapter_for_source(source)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _aux_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver aux-model-fallback notice to user: %s",
                                                _werr,
                                            )
                                finally:
                                    # Evict the cached agent so the next turn
                                    # rebuilds its system prompt from current
                                    # SOUL.md, memory, and skills.
                                    self.agent_cache.evict(session_key)
                                    if not _hyg_cleanup_deferred:
                                        await self._cleanup_agent_resources_off_loop(
                                            _hyg_agent, context="session hygiene"
                                        )

                    except Exception as e:
                        logger.warning(
                            "Session hygiene auto-compress failed: %s", e
                        )

        # First-message onboarding -- only on the very first interaction ever.
        # Delivered on the current user message (sidecar), NOT the ephemeral
        # system prompt: present-on-turn-1/absent-on-turn-2 was a guaranteed
        # system-prompt diff and agent rebuild.
        if not history and not await self.async_session_store.has_any_sessions():
            # Default first-contact note: a brief self-introduction.
            _intro_note = (
                "[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
            # Opt-in structured profile-build path. When enabled (default
            # "ask") and not yet offered on this install, swap the plain intro
            # for a consent-gated directive that offers to build a user
            # profile and persists confirmed facts via memory(target="user").
            # The offer fires at most once (onboarding.seen flag); set
            # onboarding.profile_build: off in config.yaml to disable.
            try:
                from agent.onboarding import (
                    PROFILE_BUILD_FLAG,
                    is_seen,
                    mark_seen,
                    profile_build_directive,
                    profile_build_mode,
                )
                _onb_cfg = _load_gateway_config()
                if (
                    profile_build_mode(_onb_cfg) == "ask"
                    and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG)
                ):
                    turn_sidecar_notes.append(profile_build_directive().strip())
                    mark_seen(_hermes_home / "config.yaml", PROFILE_BUILD_FLAG)
                else:
                    turn_sidecar_notes.append(_intro_note)
            except Exception as _pb_err:
                logger.debug(
                    "Profile-build onboarding directive failed, using plain intro: %s",
                    _pb_err,
                )
                turn_sidecar_notes.append(_intro_note)

        # One-time prompt if no home channel is set for this platform
        if not history and source.platform and source.platform != Platform.LOCAL:
            platform_name = source.platform.value
            env_key = _home_target_env_var(platform_name)
            # Multiplex: home channel may live only in the profile secret
            # scope / PlatformConfig, not process os.environ.
            home_env = ""
            try:
                from agent.secret_scope import get_secret

                home_env = (get_secret(env_key) or "").strip() if env_key else ""
            except Exception:
                home_env = ""
            if not home_env:
                home_env = (os.getenv(env_key) or "").strip() if env_key else ""
            # Also honor in-memory / yaml home_channel on this platform.
            try:
                if not home_env and self.config.get_home_channel(source.platform):
                    home_env = "set"
            except Exception:
                pass
            # Secondary-profile platforms (e.g. Slack on yolo) may only exist
            # under that profile's loaded config — check after scope install.
            if not home_env:
                try:
                    from gateway.config import load_gateway_config as _lgc
                    prof = (getattr(source, "profile", None) or "").strip()
                    if prof and prof != "default":
                        # Already inside profile scope for secondary handlers;
                        # re-read live config for home_channel.
                        _pcfg = _lgc()
                        if _pcfg.get_home_channel(source.platform):
                            home_env = "set"
                except Exception:
                    pass
            if not home_env:
                sethome_cmd = "/sethome"
                notice = (
                    f"📬 No home channel is set for {platform_name.title()}. "
                    f"A home channel is where Hermes delivers cron job results "
                    f"and cross-platform messages.\n\n"
                    f"Type {sethome_cmd} to make this chat your home channel, "
                    f"or ignore to skip."
                )
                await self._deliver_platform_notice(source, notice)

        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        message_text = await self._prepare_inbound_message_for_turn(
            event=event,
            source=source,
            history=history,
            session_key=session_key,
        )
        if message_text is None:
            return

        # Capture the platform event time as message metadata and keep the
        # persisted transcript clean (strip any leading timestamp prefix).
        # This runs regardless of the toggle so storage stays clean and the
        # send-time is preserved. Only the in-context RENDER (prepending the
        # human-readable prefix the model sees) is gated behind
        # gateway.message_timestamps.enabled — default OFF.
        try:
            from hermes_time import get_timezone as _get_evt_tz
            from gateway.message_timestamps import (
                coerce_message_timestamp as _coerce_msg_ts,
                render_user_content_with_timestamp as _render_msg_ts,
                strip_leading_message_timestamps as _strip_msg_ts,
            )
            _evt_tz = _get_evt_tz()
            _evt_ts = getattr(event, "timestamp", None)
            if message_text and isinstance(message_text, str):
                _clean_message_text, _embedded_ts = _strip_msg_ts(
                    message_text, tz=_evt_tz)
                persist_user_message = _clean_message_text
                _event_epoch = _coerce_msg_ts(_evt_ts, tz=_evt_tz)
                persist_user_timestamp = (
                    _event_epoch if _event_epoch is not None else _embedded_ts
                )
                if _message_timestamps_enabled(_load_gateway_config()):
                    message_text = _render_msg_ts(
                        _clean_message_text,
                        persist_user_timestamp,
                        tz=_evt_tz,
                    )
                else:
                    # Toggle off: model sees the clean message; the timestamp
                    # is still stored as metadata for later opt-in.
                    message_text = _clean_message_text
        except Exception as _ts_err:
            logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)

        # Stage the collected must-deliver notes for this turn's agent run
        # (one-shot; consumed in run_sync).  Staged AFTER the message_text
        # early-out above so an aborted turn cannot leak its notes into the
        # next turn's user message.
        if turn_sidecar_notes and session_key:
            self.sessions.set_sidecar_notes(session_key, turn_sidecar_notes)

        # Bind this gateway run generation to the adapter's active-session
        # event so deferred post-delivery callbacks can be released by the
        # same run that registered them.
        self._bind_adapter_run_generation(
            self._adapter_for_source(source),
            session_key,
            run_generation,
        )

        try:
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(getattr(source, "thread_id", None)) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Run the agent. Capture the session id that this run was launched
            # against so post-run compression publication can be identity-guarded
            # below; a /new or another lifecycle transition may move
            # session_entry.session_id while the old run is still unwinding.
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=_run_start_session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=self._reply_anchor_for_event(event),
                channel_prompt=event.channel_prompt,
                moa_config=getattr(event, "_moa_config", None),
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                message_type=event.message_type,
            )
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            # Stop persistent typing indicator now that the agent is done.
            # Slack AI status is scoped to a thread/workspace, so preserve the
            # same routing metadata used by the response delivery path.
            try:
                _typing_adapter = self._adapter_for_source(source)
                _stop_with_metadata = getattr(
                    type(_typing_adapter), "_stop_typing_with_metadata", None
                )
                _stop_typing = getattr(type(_typing_adapter), "stop_typing", None)
                if _typing_adapter and callable(_stop_with_metadata):
                    await _typing_adapter._stop_typing_with_metadata(
                        source.chat_id,
                        self._thread_metadata_for_source(
                            source, self._reply_anchor_for_event(event)
                        ),
                    )
                elif _typing_adapter and callable(_stop_typing):
                    await _typing_adapter.stop_typing(source.chat_id)
            except Exception:
                pass

            if not self.sessions.run_is_current(_quick_key, run_generation):
                logger.info(
                    "Discarding stale agent result for %s — generation %d is no longer current",
                    _quick_key or "?",
                    run_generation,
                )
                _stale_adapter = self._adapter_for_source(source)
                if getattr(type(_stale_adapter), "pop_post_delivery_callback", None) is not None:
                    _stale_adapter.pop_post_delivery_callback(
                        _quick_key,
                        generation=run_generation,
                    )
                elif _stale_adapter and hasattr(_stale_adapter, "_post_delivery_callbacks"):
                    _stale_adapter._post_delivery_callbacks.pop(_quick_key, None)
                return None

            response = agent_result.get("final_response") or ""
            # Hidden-reasoning-only retry exhaustion: the loop's sentinel text
            # ("Codex response remained incomplete after 3 continuation
            # attempts") doubles as final_response, so it would be delivered
            # verbatim into the channel — where peer agents can ingest it as a
            # completed assistant turn (#51628). Blank it here so the normal
            # empty-response handling (and the suppression below) applies.
            if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
                response = ""
            try:
                from gateway.response_filters import is_intentional_silence_agent_result
                _intentional_silence = is_intentional_silence_agent_result(
                    agent_result, response,
                )
            except Exception:
                _intentional_silence = False

            # Convert the agent's internal "(empty)" sentinel into a
            # user-friendly message.  "(empty)" means the model failed to
            # produce visible content after exhausting all retries (nudge,
            # prefill, empty-retry, fallback).  Sending the raw sentinel
            # looks like a bug; a short explanation is more helpful.
            if response == "(empty)" and not _intentional_silence:
                response = (
                    "⚠️ The model returned no response after processing tool "
                    "results. This can happen with some models — try again or "
                    "rephrase your question."
                )
            agent_messages = agent_result.get("messages", [])
            _response_time = time.time() - _msg_start_time
            _api_calls = agent_result.get("api_calls", 0)
            _resp_len = len(response)
            logger.info(
                "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
                _platform_name, source.chat_id or "unknown",
                _response_time, _api_calls, _resp_len,
            )

            # NOTE: the cross-process cache-coherence re-baseline
            # (_refresh_agent_cache_message_count) is intentionally deferred
            # until AFTER this turn's transcript persistence block below — it
            # must include the first-turn `session_meta` marker row and the
            # compression session_id swap, both of which happen later.  See
            # the call site after the `update_session(...)` write.

            # Successful turn — clear any stuck-loop counter for this session.
            # This ensures the counter only accumulates across CONSECUTIVE
            # restarts where the session was active (never completed).
            #
            # Also clear the resume_pending flag (set by drain-timeout
            # shutdown) — the turn ran to completion, so recovery
            # succeeded and subsequent messages should no longer receive
            # the restart-interruption system note.
            if session_key and _should_clear_resume_pending_after_turn(agent_result):
                await self._clear_restart_failure_count(session_key)
                try:
                    await self.async_session_store.clear_resume_pending(session_key)
                except Exception as _e:
                    logger.debug(
                        "clear_resume_pending failed for %s: %s",
                        session_key, _e,
                    )

            # Normalize empty responses: surface errors, partial failures, and
            # the case where agent did work but returned no text. Fix for #18765.
            if not _intentional_silence:
                response = _normalize_empty_agent_response(
                    agent_result, response, history_len=len(history),
                )
                response = _sanitize_gateway_final_response(source.platform, response)

            # Ordering contract: the agent thread already updated the contextvar
            # in conversation_compression.py; propagate to SessionEntry + _save().
            # If the agent's session_id changed during compression, update
            # session_entry so transcript writes below go to the right session.
            if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
                if session_entry.session_id == _run_start_session_id:
                    session_entry.session_id = agent_result["session_id"]
                    # The held turn lease follows the rotation: the transcript
                    # persistence below writes to the NEW id, so the
                    # serialization boundary must move with it or an alias
                    # key resolving the fresh child could interleave (#64934).
                    self.sessions.rebind_turn_lease(
                        _quick_key, run_generation, session_entry.session_id
                    )
                    await self.async_session_store._save()
                    await self.async_session_store._record_gateway_session_peer(
                        session_entry.session_id,
                        session_key,
                        source,
                    )
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="agent-result-compression",
                    )
                else:
                    logger.info(
                        "Skipping agent-result session split sync for %s because "
                        "the session binding moved from %s to %s before "
                        "compression finished",
                        session_key or "?",
                        _run_start_session_id,
                        session_entry.session_id,
                    )

            # Prepend reasoning/thinking if display is enabled (per-platform).
            # Mattermost requires explicit per-platform opt-in because this is
            # scratch text, not ordinary final-answer content.
            try:
                _show_reasoning_effective = _resolve_gateway_display_bool(
                    _load_gateway_config(),
                    _platform_config_key(source.platform),
                    "show_reasoning",
                    default=bool(getattr(self, "_show_reasoning", False)),
                    platform=source.platform,
                    require_platform_override_for={Platform.MATTERMOST},
                )
            except Exception:
                _show_reasoning_effective = (
                    False
                    if source.platform == Platform.MATTERMOST
                    else getattr(self, "_show_reasoning", False)
                )
            if _show_reasoning_effective and response and not _intentional_silence:
                last_reasoning = agent_result.get("last_reasoning")
                if last_reasoning:
                    from gateway.stream_consumer import escape_code_fences_for_display
                    # Collapse long reasoning to keep messages readable
                    lines = last_reasoning.strip().splitlines()
                    if len(lines) > 15:
                        display_reasoning = "\n".join(lines[:15])
                        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
                    else:
                        display_reasoning = last_reasoning.strip()
                    # Render style is per-platform: Discord defaults to "-# "
                    # subtext (native small grey metadata text); other
                    # platforms keep the fenced code block.
                    try:
                        from gateway.display_config import resolve_display_setting
                        _reasoning_style = resolve_display_setting(
                            _load_gateway_config(),
                            _platform_config_key(source.platform),
                            "reasoning_style",
                            "code",
                        )
                    except Exception:
                        _reasoning_style = "code"
                    if _reasoning_style == "subtext":
                        _quoted = "\n".join(
                            f"-# {ln}" if ln else "-#" for ln in display_reasoning.splitlines()
                        )
                        response = f"-# 💭 Reasoning\n{_quoted}\n\n{response}"
                    elif _reasoning_style == "blockquote":
                        _quoted = "\n".join(
                            f"> {ln}" if ln else ">" for ln in display_reasoning.splitlines()
                        )
                        response = f"> 💭 **Reasoning:**\n{_quoted}\n\n{response}"
                    else:
                        # Escape ``` inside reasoning so inner fences don't
                        # break the outer code block used to render it.
                        display_reasoning = escape_code_fences_for_display(display_reasoning)
                        response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

            # Runtime-metadata footer — only on the FINAL message of the turn.
            # Off by default (display.runtime_footer.enabled=false).  When
            # streaming already delivered the body, we can't mutate the sent
            # text, so we fire a separate trailing send below.
            _footer_line = ""
            try:
                from gateway.runtime_footer import build_footer_line as _bfl
                _footer_line = _bfl(
                    user_config=_load_gateway_config(),
                    platform_key=_platform_config_key(source.platform),
                    model=agent_result.get("model"),
                    context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                    context_length=agent_result.get("context_length") or None,
                    cwd=os.environ.get("TERMINAL_CWD", ""),
                    turn_seconds=_turn_seconds,
                )
            except Exception as _footer_err:
                logger.debug("runtime_footer build failed: %s", _footer_err)
                _footer_line = ""
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"

            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
                "model": agent_result.get("model", ""),
                "provider": agent_result.get("provider", ""),
            })

            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                # Detach the current batch atomically (see crash-recovery drain
                # above): reassign to a fresh list so a watcher appended by a
                # concurrent session during the yield isn't dropped by clear().
                watchers = process_registry.pending_watchers
                process_registry.pending_watchers = []
                for i, watcher in enumerate(watchers):
                    asyncio.create_task(self._run_process_watcher(watcher))
                    if i % 100 == 99:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Drain watch pattern notifications that arrived during the agent run.
            # Watch events and completions share the same queue; process
            # completions are already handled by the per-process watcher task
            # above, so we only inject watch-type events here.
            #
            # Async-delegation completions ALSO ride this shared queue but are
            # owned by the dedicated _async_delegation_watcher (started at
            # boot), which covers both the idle and post-turn cases with a
            # single consumer — so we leave them on the queue here.
            try:
                from tools.process_registry import process_registry as _pr
                await self._drain_watch_notifications(_pr.completion_queue)
            except Exception as e:
                logger.debug("Watch queue drain error: %s", e)

            # NOTE: Dangerous command approvals are now handled inline by the
            # blocking gateway approval mechanism in tools/approval.py.  The agent
            # thread blocks until the user responds with /approve or /deny, so by
            # the time we reach here the approval has already been resolved.  The
            # old post-loop pop_pending + approval_hint code was removed in favour
            # of the blocking approach that mirrors CLI's synchronous input().

            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            #
            # IMPORTANT: For context-overflow failures (compression exhausted,
            # generic 400 on large sessions) we must NOT persist the user's
            # message — doing so would grow the session further and cause the
            # same failure on the next attempt, an infinite loop. (#1630, #9893)
            #
            # Transient failures (429, timeout, connection error, provider 5xx)
            # are different: the session is not oversized, and silently dropping
            # the user message causes severe context loss on retry — the agent
            # forgets what was just asked.  Persist the user turn so the
            # conversation is preserved. (#7100)
            agent_failed_early = bool(agent_result.get("failed"))
            hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(
                agent_result
            )
            _err_str_for_classify = str(agent_result.get("error", "")).lower()
            # Use specific multi-word phrases (not bare "exceed" or "token")
            # to avoid false positives on transient errors like "rate limit
            # exceeded" or "invalid auth token". Matches run_agent.py's
            # own context-length classifier.
            is_context_overflow_failure = agent_failed_early and (
                bool(agent_result.get("compression_exhausted"))
                or any(p in _err_str_for_classify for p in (
                    "context length", "context size", "context window",
                    "maximum context", "token limit", "too many tokens",
                    "reduce the length", "exceeds the limit",
                    "request entity too large", "prompt is too long",
                    "payload too large", "input is too long",
                ))
                or ("400" in _err_str_for_classify and len(history) > 50)
            )
            if is_context_overflow_failure:
                logger.info(
                    "Skipping transcript persistence for context-overflow "
                    "failure in session %s to prevent session growth loop.",
                    session_entry.session_id,
                )
            elif agent_failed_early:
                logger.info(
                    "Transient agent failure in session %s — persisting user "
                    "message so conversation context is preserved on retry.",
                    session_entry.session_id,
                )
            elif hidden_reasoning_incomplete:
                logger.warning(
                    "Suppressing hidden-reasoning-only incomplete gateway turn "
                    "for session %s: %s",
                    session_entry.session_id,
                    agent_result.get("error", "processing incomplete"),
                )

            # When compression is exhausted, the session is permanently too
            # large to process.  Auto-reset it so the next message starts
            # fresh instead of replaying the same oversized context in an
            # infinite fail loop.  (#9893)
            #
            # A lock-contended defer is the OPPOSITE case: the session is
            # temporarily uncompressible only because a concurrent path holds
            # the compression lock and is actively shrinking it. Never wipe
            # the session for that — retry-next-message semantics apply
            # (#69870 lock-skip consumer; salvaged from #49874).
            if agent_result.get("compression_deferred"):
                logger.info(
                    "Compression deferred for session %s — the compression "
                    "lock is held by a concurrent compressor. Keeping the "
                    "session intact; the next message retries normally.",
                    session_entry.session_id if session_entry else "?",
                )
            elif agent_result.get("compression_exhausted") and session_entry and session_key:
                logger.info(
                    "Auto-resetting session %s after compression exhaustion.",
                    session_entry.session_id,
                )
                new_entry = await self.async_session_store.reset_session(session_key)
                self.agent_cache.evict(session_key)
                # Conversation boundary: one funnel call clears every
                # conversation-scoped per-session dict (#58403 and siblings).
                # See _CONVERSATION_SCOPED_STATE.
                self._clear_conversation_scope(
                    session_key, reason="compression_exhausted_reset"
                )
                if new_entry is not None:
                    # Drop the stale reference to the bloated compressed child and
                    # re-point the Telegram topic binding at the fresh session.
                    # Compression rotated session_entry.session_id to the oversized
                    # compressed child earlier this turn (the agent-result sync
                    # above), and that _sync also rewrote the (chat_id, thread_id)
                    # -> bloated-child binding. reset_session swaps in a clean,
                    # parentless session, but without re-syncing the binding the
                    # next inbound message in this topic gets switch_session'd back
                    # onto the bloated child by the binding-heal walk, reloads the
                    # oversized transcript, and re-triggers compression exhaustion
                    # forever (#35809 — regression of the #9893/#10063 auto-reset).
                    # No-op on non-topic lanes.
                    session_entry = new_entry
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-exhausted-reset",
                    )
                response = (response or "") + (
                    "\n\n🔄 Session auto-reset — the conversation exceeded the "
                    "maximum context size and could not be compressed further. "
                    "Your next message will start a fresh session."
                )

            ts = time.time()  # Unix epoch float — consistent with DB storage

            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if is_context_overflow_failure:
                pass  # Skip all transcript writes — don't grow a broken session
            elif not history:
                tool_defs = agent_result.get("tools", [])
                await self.async_session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": _resolve_gateway_model(),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )

            # The agent already persisted these messages to SQLite via
            # _flush_messages_to_session_db(), so skip the DB write here
            # to prevent the duplicate-write bug (#860 / #42039). This holds
            # for the codex app-server runtime too: although it early-returns
            # and bypasses conversation_loop's per-step flushes, it flushes its
            # own projected assistant/tool messages before returning and
            # reports agent_persisted=True (see agent/codex_runtime.py). Reading
            # the flag (default = self._session_db is not None) keeps the
            # persistence contract explicit and lets any future non-persisting
            # runtime opt into a gateway-side write by returning False.
            agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)

            # Find only the NEW messages from this turn (skip history we loaded).
            # Use the filtered history length (history_offset) that was actually
            # passed to the agent, not len(history) which includes session_meta
            # entries that were stripped before the agent saw them.
            if is_context_overflow_failure:
                pass  # handled above — skip all transcript writes
            elif agent_failed_early or hidden_reasoning_incomplete:
                # Transient failure (429/timeout/5xx): persist only the user
                # message so the next message can load a transcript that
                # reflects what was said.  Skip the assistant error text since
                # it's a gateway-generated hint, not model output. Hidden-
                # reasoning-only incomplete turns follow the same persistence
                # rule so peer-agent channels don't ingest them as completed
                # assistant turns. (#7100, #51628)
                _user_entry = {
                    "role": "user",
                    "content": (
                        persist_user_message
                        if persist_user_message is not None
                        else message_text
                    ),
                    "timestamp": (
                        persist_user_timestamp
                        if persist_user_timestamp is not None
                        else ts
                    ),
                }
                if persist_user_display_kind:
                    _user_entry["display_kind"] = persist_user_display_kind
                if event.message_id:
                    _user_entry["message_id"] = str(event.message_id)
                # Dedupe: skip if this platform message_id is already in the
                # transcript (prevents duplicate user turns on Telegram retries
                # after transient failures). #47237
                _skip_persist = (
                    event.message_id
                    and await self.async_session_store.has_platform_message_id(
                        session_entry.session_id, str(event.message_id)
                    )
                )
                if _skip_persist:
                    logger.info(
                        "Skipping duplicate user turn "
                        "(message_id=%s) in session %s",
                        event.message_id, session_entry.session_id,
                    )
                else:
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
            else:
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []

                # If no new messages found (edge case), fall back to simple user/assistant
                if not new_messages:
                    _user_entry = {
                        "role": "user",
                        "content": (
                            persist_user_message
                            if persist_user_message is not None
                            else message_text
                        ),
                        "timestamp": (
                            persist_user_timestamp
                            if persist_user_timestamp is not None
                            else ts
                        ),
                    }
                    if persist_user_display_kind:
                        _user_entry["display_kind"] = persist_user_display_kind
                    if event.message_id:
                        _user_entry["message_id"] = str(event.message_id)
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
                    if response:
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id,
                            {"role": "assistant", "content": response, "timestamp": ts},
                            skip_db=agent_persisted,
                        )
                else:
                    # Attach the inbound platform message_id to the first user
                    # entry written this turn so platform-level quote-resolution
                    # (e.g. Yuanbao QuoteContextMiddleware's transcript fallback)
                    # can find earlier @bot messages by their original message_id.
                    _user_msg_id_attached = False
                    for msg in new_messages:
                        # Skip system messages (they're rebuilt each run)
                        if msg.get("role") == "system":
                            continue
                        # Add timestamp to each message for debugging
                        entry = {**msg, "timestamp": ts}
                        if (
                            not _user_msg_id_attached
                            and msg.get("role") == "user"
                            and event.message_id
                            and "message_id" not in entry
                        ):
                            entry["message_id"] = str(event.message_id)
                            _user_msg_id_attached = True
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id, entry,
                            skip_db=agent_persisted,
                        )

            # Token counts and model are now persisted by the agent directly.
            # Keep only last_prompt_tokens here for context-window tracking and
            # compression decisions.
            await self.async_session_store.update_session(
                session_entry.session_key,
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
                touch_activity=not bool(getattr(event, "internal", False)),
            )

            # Re-baseline the cached agent's message_count snapshot now that
            # ALL of this turn's transcript writes are done — the agent's
            # flushed user/assistant/tool rows AND the first-turn `session_meta`
            # marker appended above.  The cross-process coherence guard (#45966)
            # snapshots the count at agent-BUILD time (before this turn's own
            # writes) and never refreshes it on reuse, so without this the
            # process's own turn grows message_count and the next turn sees a
            # mismatch and rebuilds the agent — destroying prompt caching.
            #
            # This MUST run after the `session_meta` append: that row also
            # increments message_count, so re-baselining before it (the old
            # position) left the snapshot one short and the guard mis-fired on
            # turn 2 of EVERY fresh gateway conversation, rebuilding the cached
            # agent and busting the prompt cache.  Running here also uses the
            # compaction-updated session_id (the agent_result session_id swap
            # above), matching this function's documented contract.  Refreshing
            # here makes the guard fire only on a DIFFERENT process's writes.
            # Fail-safe inside the helper.
            await self.agent_cache.refresh_message_count(
                session_key, session_entry.session_id
            )

            # Intentional silence is a delivery decision, not a transcript
            # mutation.  The agent's [SILENT]/NO_REPLY assistant turn above is
            # still persisted in session history so later turns keep normal
            # user/assistant alternation; only the outbound chat delivery is
            # suppressed.
            if _intentional_silence:
                logger.info(
                    "Suppressing intentional silence marker for session %s",
                    session_entry.session_id,
                )
                response = ""

            # If streaming already delivered the response, extract and
            # deliver any MEDIA: files before returning None.  Streaming
            # sends raw text chunks that include MEDIA: tags — the normal
            # post-processing in _process_message_background is skipped
            # when already_sent is True, so media files would never be
            # delivered without this.
            #
            # Never skip when the agent failed — the error message is new
            # content the user hasn't seen (streaming only sent earlier
            # partial output before the failure).  Without this guard,
            # users see the agent "stop responding without explanation."
            if agent_result.get("already_sent") and not agent_result.get("failed"):
                if response:
                    _media_adapter = self._adapter_for_source(source)
                    if _media_adapter:
                        await self._deliver_media_from_response(
                            response, event, _media_adapter,
                        )
                # Streaming already delivered the body text, but the footer was
                # intentionally held back (see the `not already_sent` gate above).
                # Send it now as a small trailing message so Telegram/Discord/etc.
                # still surface the runtime metadata on the final reply.
                if _footer_line:
                    try:
                        _foot_adapter = self._adapter_for_source(source)
                        if _foot_adapter:
                            await _foot_adapter.send(
                                source.chat_id,
                                _footer_line,
                                metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
                            )
                    except Exception as _e:
                        logger.debug("trailing footer send failed: %s", _e)
                # This branch returns None so the adapter does not send the
                # body twice. /loop and /goal hooks in _handle_message read
                # the return value, so stash the delivered text on the event
                # or those hooks never run and a /loop tick stays awaiting.
                try:
                    event._streamed_final_response = str(response or "")
                except Exception:
                    pass
                return None

            return response

        except Exception as e:
            # Stop typing indicator on error too, retaining Slack thread/workspace
            # routing so a failed turn cannot leave its status visible.
            try:
                _err_adapter = self._adapter_for_source(source)
                _stop_with_metadata = getattr(
                    type(_err_adapter), "_stop_typing_with_metadata", None
                )
                _stop_typing = getattr(type(_err_adapter), "stop_typing", None)
                if _err_adapter and callable(_stop_with_metadata):
                    await _err_adapter._stop_typing_with_metadata(
                        source.chat_id,
                        self._thread_metadata_for_source(
                            source, self._reply_anchor_for_event(event)
                        ),
                    )
                elif _err_adapter and callable(_stop_typing):
                    await _err_adapter.stop_typing(source.chat_id)
            except Exception:
                pass
            logger.exception("Agent error in session %s", session_key)
            # Crash-resilience for failures that happen before create_agent enters
            # run_conversation() (for example: provider/httpx client init
            # failures). In that path the agent cannot persist the current
            # inbound turn itself, so append the user message here once. If the
            # agent already reached its early turn-start persistence, the latest
            # transcript user row will match and we skip the duplicate.
            try:
                if 'message_text' in locals() and message_text is not None and session_entry is not None:
                    _already_persisted = False
                    try:
                        _recent_transcript = await self.async_session_store.load_transcript(session_entry.session_id)
                    except Exception:
                        _recent_transcript = []
                    for _msg in reversed(_recent_transcript[-10:]):
                        if _msg.get("role") == "user":
                            _expected_user_content = (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            )
                            _already_persisted = (_msg.get("content") == _expected_user_content)
                            break
                    if not _already_persisted:
                        _user_entry = {
                            "role": "user",
                            "content": (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            ),
                            "timestamp": (
                                persist_user_timestamp
                                if persist_user_timestamp is not None
                                else time.time()
                            ),
                        }
                        if 'persist_user_display_kind' in locals() and persist_user_display_kind:
                            _user_entry["display_kind"] = persist_user_display_kind
                        if getattr(event, "message_id", None):
                            _user_entry["message_id"] = str(event.message_id)
                        await self.async_session_store.append_to_transcript(
                            session_entry.session_id,
                            _user_entry,
                        )
            except Exception:
                logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
            # Log full details server-side only; never expose raw exception
            # types or messages to end users (info-leakage risk).
            status_hint = ""
            status_code = getattr(e, "status_code", None)
            _hist_len = len(history) if 'history' in locals() else 0
            if status_code == 401:
                status_hint = " Check your API key or run `claude /login` to refresh OAuth credentials."
            elif status_code == 402:
                status_hint = " Your API balance or quota is exhausted. Check your provider dashboard."
            elif status_code == 429:
                # Check if this is a plan usage limit (resets on a schedule) vs a transient rate limit
                _err_body = getattr(e, "response", None)
                _err_json = {}
                try:
                    if _err_body is not None:
                        _err_json = _err_body.json().get("error", {})
                        if not isinstance(_err_json, dict):
                            _err_json = {}
                except Exception:
                    pass
                if _err_json.get("type") == "usage_limit_reached":
                    _resets_in = _err_json.get("resets_in_seconds")
                    if _resets_in and _resets_in > 0:
                        import math
                        _hours = math.ceil(_resets_in / 3600)
                        status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
                    else:
                        status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
                else:
                    status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif status_code == 529:
                status_hint = " The API is temporarily overloaded. Please try again shortly."
            elif status_code in {400, 500}:
                # 400 with a large session is context overflow.
                # 500 with a large session often means the payload is too large
                # for the API to process — treat it the same way.
                if _hist_len > 50:
                    return (
                        "⚠️ Session too large for the model's context window.\n"
                        "Use /compact to compress the conversation, or "
                        "/reset to start fresh."
                    )
                elif status_code == 400:
                    status_hint = " The request was rejected by the API."
            return (
                f"Sorry, I encountered an unexpected error.{status_hint}\n"
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)
