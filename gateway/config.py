"""
Gateway configuration management.

Handles loading and validating configuration for:
- Telegram, Mattermost, and the OpenAI-compatible API server
- Home channels for each platform
- Session reset policies
- Delivery preferences
"""

import logging
import os
import json
from pathlib import Path
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from hermes_cli.config import get_hermes_home
from agent.secret_scope import current_secret_scope, get_secret as _get_secret
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return is_truthy_value(value, default=default)




def _normalize_transport_token(value: Any) -> str:
    """Normalize a streaming transport/mode value to a canonical token.

    Handles the YAML 1.1 boolean quirk where bare ``on`` / ``off`` parse to
    Python ``True`` / ``False`` (see ``gateway/display_config.py`` ``_normalise``).
    Without this, ``mode: off`` arrives as boolean ``False`` and stringifying it
    yields ``"false"`` instead of the advertised ``"off"``, so streaming would be
    enabled instead of disabled. Booleans map to ``"auto"`` (True) / ``"off"``
    (False); anything else is lower-cased, defaulting to ``"auto"``.
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower() or "auto"


def _coerce_float(value: Any, default: float) -> float:
    """Coerce numeric config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce integer config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_positive_int(value: Any, key: str) -> Optional[int]:
    """Coerce an optional positive integer config value.

    ``None``/0/negative disable the setting. Malformed values are ignored with
    a warning so a typo never prevents the gateway from starting.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(value)
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip(), 10)
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    if parsed <= 0:
        return None
    return parsed


_SYSTEMD_WATCHDOG_MAX_SECONDS = 2_147_483_647


def coerce_systemd_watchdog_seconds(
    value: Any, key: str = "gateway.systemd_watchdog_seconds"
) -> int:
    """Return a bounded positive watchdog interval or zero when disabled.

    Runtime and service generation share this normalization so a value can
    never enable ``Type=notify`` while disabling application heartbeats.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        logger.warning("Ignoring invalid %s (expected a positive integer)", key)
        return 0
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw or not raw.isascii() or not raw.isdecimal():
            logger.warning("Ignoring invalid %s (expected a positive integer)", key)
            return 0
        try:
            parsed = int(raw, 10)
        except (TypeError, ValueError, OverflowError):
            logger.warning("Ignoring invalid %s (expected a positive integer)", key)
            return 0
    else:
        logger.warning("Ignoring invalid %s (expected a positive integer)", key)
        return 0
    if parsed == 0:
        return 0
    if not 0 < parsed <= _SYSTEMD_WATCHDOG_MAX_SECONDS:
        logger.warning(
            "Ignoring invalid %s (expected an integer from 1 to %d)",
            key,
            _SYSTEMD_WATCHDOG_MAX_SECONDS,
        )
        return 0
    return parsed


def _coerce_dict(value: Any) -> Dict[str, Any]:
    """Return *value* when it is a mapping, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _normalize_unauthorized_dm_behavior(value: Any, default: str = "pair") -> str:
    """Normalize unauthorized DM behavior to a supported value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pair", "ignore"}:
            return normalized
    return default


def _normalize_notice_delivery(value: Any, default: str = "public") -> str:
    """Normalize notice delivery mode to a supported value."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"public", "private"}:
            return normalized
    return default


def _ensure_platform_extra_dict(platforms_data: dict, name: str) -> tuple[dict, dict]:
    """Get-or-create ``platforms_data[name]`` and its nested ``extra`` dict.

    Both slots are coerced to ``{}`` if a non-dict value is encountered, so
    callers can safely write keys without type-checking.  Returns
    ``(plat_data, extra)`` for in-place mutation.
    """
    plat_data = platforms_data.setdefault(name, {})
    if not isinstance(plat_data, dict):
        plat_data = {}
        platforms_data[name] = plat_data
    extra = plat_data.setdefault("extra", {})
    if not isinstance(extra, dict):
        extra = {}
        plat_data["extra"] = extra
    return plat_data, extra


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read env vars through the active profile secret scope when present.

    ``load_gateway_config()`` runs in many contexts, including multiplexed
    profile startup where ``_profile_runtime_scope`` installs per-profile
    secrets. In that scope we must prefer the scoped value; outside it we keep
    legacy ``os.getenv`` behavior for single-profile callers and unscoped
    gateway reads.
    """
    if current_secret_scope() is not None:
        scope_val = _get_secret(name, None)
        return scope_val if scope_val is not None else default
    env_val = os.environ.get(name)
    if env_val is not None:
        return env_val
    return default


def _getenv_str(name: str, default: str = "") -> str:
    val = _getenv(name, default)
    return val if val is not None else default


def _getenv_int(name: str, default: int) -> int:
    raw = _getenv(name, None)
    if raw is None:
        return default
    try:
        return int(str(raw).strip(), 10)
    except (TypeError, ValueError):
        return default


class Platform(Enum):
    """Gateway origins and retained transports."""

    LOCAL = "local"
    TELEGRAM = "telegram"
    MATTERMOST = "mattermost"


# Platforms that bind a host TCP port. In a profile
# multiplexer the default profile owns the single shared listener and serves
# every profile through the /p/<profile>/ URL prefix, so a SECONDARY profile
# enabling one of these is always a misconfiguration: it would try to bind a
# port already held by the default's listener. Single source of truth for
# the gateway's fail-fast startup validation. Stored as platform .value
# strings so configuration checks share the same policy.
PORT_BINDING_PLATFORM_VALUES = frozenset()

def platform_binds_port(platform_value: str, extra: Optional[dict] = None) -> bool:
    """Return True when *platform_value* owns a listener port."""
    return platform_value in PORT_BINDING_PLATFORM_VALUES


@dataclass
class HomeChannel:
    """
    Default destination for a platform.
    
    When a cron job specifies deliver="telegram" without a specific chat ID,
    messages are sent to this home channel. Thread-aware platforms may also
    store a thread/topic ID so the bare platform target routes to the exact
    conversation where /sethome was run.
    """
    platform: Platform
    chat_id: str
    name: str  # Human-readable name for display
    thread_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "name": self.name,
        }
        if self.thread_id:
            result["thread_id"] = self.thread_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomeChannel":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            name=data.get("name", "Home"),
            thread_id=str(data["thread_id"]) if data.get("thread_id") else None,
        )


def persist_home_channel(home: HomeChannel, *, enabled_if_new: bool = False) -> None:
    """Persist a platform home channel."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    platform_config = platforms.setdefault(home.platform.value, {})
    if not isinstance(platform_config, dict):
        platform_config = {}
        platforms[home.platform.value] = platform_config
    if enabled_if_new:
        platform_config.setdefault("enabled", True)
    platform_config["home_channel"] = home.to_dict()
    save_config(config)


@dataclass
class SessionResetPolicy:
    """
    Controls when sessions reset (lose context).
    
    Modes:
    - "daily": Reset at a specific hour each day
    - "idle": Reset after N minutes of inactivity
    - "both": Whichever triggers first (daily boundary OR idle timeout)
    - "none": Never auto-reset (context managed only by compression)

    Default is "none" — sessions never auto-reset unless the user opts in
    via the `session_reset` section in config.yaml (or gateway.json
    overrides). Changed July 2026 from "both" (24h idle + daily 4am), which
    surprised users who expected their conversations to persist.
    """
    mode: str = "none"  # "daily", "idle", "both", or "none"
    at_hour: int = 4  # Hour for daily reset (0-23, local time)
    idle_minutes: int = 1440  # Minutes of inactivity before reset (24 hours)
    notify: bool = True  # Send a notification to the user when auto-reset occurs
    notify_exclude_platforms: tuple = ()
    # A background process this many hours old (or older) no longer blocks
    # session idle/daily reset. A forgotten preview server should not keep a
    # session alive forever (#29177). The process is NOT killed — only ignored
    # by the reset guard. Raise this if you run legitimate multi-day jobs whose
    # liveness should pin the conversation open.
    bg_process_max_age_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "at_hour": self.at_hour,
            "idle_minutes": self.idle_minutes,
            "notify": self.notify,
            "notify_exclude_platforms": list(self.notify_exclude_platforms),
            "bg_process_max_age_hours": self.bg_process_max_age_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        data = _coerce_dict(data)
        # Handle both missing keys and explicit null values (YAML null → None)
        mode = data.get("mode")
        at_hour = data.get("at_hour")
        idle_minutes = data.get("idle_minutes")
        notify = data.get("notify")
        exclude = data.get("notify_exclude_platforms")
        bg_max_age = data.get("bg_process_max_age_hours")
        return cls(
            mode=mode if mode is not None else "none",
            at_hour=at_hour if at_hour is not None else 4,
            idle_minutes=idle_minutes if idle_minutes is not None else 1440,
            notify=_coerce_bool(notify, True),
            notify_exclude_platforms=tuple(exclude) if exclude is not None else (),
            bg_process_max_age_hours=bg_max_age if bg_max_age is not None else 24,
        )


@dataclass
class ChannelOverride:
    """
    Per-channel override for model, provider, and system prompt.

    Used in config under platforms.<name>.channel_overrides[channel_id].
    Enables different channels (e.g. Mattermost town-square vs developers) to use different
    models and personas without running separate gateway instances.
    """
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.model is not None:
            out["model"] = self.model
        if self.provider is not None:
            out["provider"] = self.provider
        if self.system_prompt is not None:
            out["system_prompt"] = self.system_prompt
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelOverride":
        if not data:
            return cls()
        return cls(
            model=data.get("model"),
            provider=data.get("provider"),
            system_prompt=data.get("system_prompt"),
        )


# Canonical map of platforms whose primary credential is ``PlatformConfig.token``
# and the env var it loads from. Used for empty-token warnings at config
# validation and by the multiplex primary-startup credential gate in
# ``gateway.run`` (#64674). The API server authenticates with an API key and
# must never be skipped for a missing bot token.
PLATFORM_TOKEN_ENV_NAMES: dict["Platform", str] = {
    Platform.TELEGRAM: "TELEGRAM_BOT_TOKEN",
    Platform.MATTERMOST: "MATTERMOST_TOKEN",
}


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None  # Bot token (Telegram, Mattermost)
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None

    # Reply threading mode for retained messaging adapters.
    # - "off": Never thread replies to original message
    # - "first": Only first chunk threads to user's message (default)
    # - "all": All chunks in multi-part replies thread to user's message
    reply_to_mode: str = "first"

    # Whether the gateway is allowed to send "♻️ Gateway online" /
    # "♻ Gateway restarted" lifecycle notifications on this platform.
    # Set False when operator-flavored restart pings are unwanted.
    gateway_restart_notification: bool = True

    # Whether the gateway shows a "typing…" / "is thinking…" status indicator
    # while the agent processes a message on this platform. Default True
    # preserves prior behavior. Set False when the indicator is unwanted.
    # Drives the per-message _keep_typing refresh loop in
    # gateway/platforms/base.py.
    typing_indicator: bool = True

    # Optional custom text for adapters whose typing indicator renders a
    # visible status line. Adapters with textless indicators ignore it.
    typing_status_text: Optional[str] = None

    # Per-channel model/provider/system_prompt overrides (channel_id -> ChannelOverride)
    channel_overrides: Dict[str, ChannelOverride] = field(default_factory=dict)

    # Platform-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "extra": self.extra,
            "reply_to_mode": self.reply_to_mode,
            "gateway_restart_notification": self.gateway_restart_notification,
            "typing_indicator": self.typing_indicator,
        }
        if self.typing_status_text is not None:
            result["typing_status_text"] = self.typing_status_text
        if self.token:
            result["token"] = self.token
        if self.api_key:
            result["api_key"] = self.api_key
        if self.home_channel:
            result["home_channel"] = self.home_channel.to_dict()
        if self.channel_overrides:
            result["channel_overrides"] = {
                cid: ov.to_dict() for cid, ov in self.channel_overrides.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        data = _coerce_dict(data)
        home_channel = None
        if isinstance(data.get("home_channel"), dict):
            home_channel = HomeChannel.from_dict(data["home_channel"])

        # gateway_restart_notification may be bridged into extra via the
        # shared-key loop in load_gateway_config(); check both top-level
        # and extra so YAML ``telegram: gateway_restart_notification: false``
        # works without needing a separate platforms: block.
        extra = _coerce_dict(data.get("extra", {}))
        _grn = data.get("gateway_restart_notification")
        if _grn is None:
            _grn = extra.get("gateway_restart_notification")

        # typing_indicator mirrors gateway_restart_notification: it may arrive
        # top-level or bridged into extra by the shared-key loop in
        # load_gateway_config(), so check both.
        _typing = data.get("typing_indicator")
        if _typing is None:
            _typing = extra.get("typing_indicator")

        # typing_status_text takes the same two routes (top-level or bridged
        # into extra); string passthrough, no coercion.
        _typing_text = data.get("typing_status_text")
        if _typing_text is None:
            _typing_text = extra.get("typing_status_text")

        channel_overrides: Dict[str, ChannelOverride] = {}
        raw_overrides = data.get("channel_overrides") or {}
        if isinstance(raw_overrides, dict):
            for cid, ov_data in raw_overrides.items():
                if isinstance(ov_data, dict):
                    channel_overrides[str(cid)] = ChannelOverride.from_dict(ov_data)

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=home_channel,
            reply_to_mode=data.get("reply_to_mode", "first"),
            gateway_restart_notification=_coerce_bool(_grn, True),
            typing_indicator=_coerce_bool(_typing, True),
            typing_status_text=_typing_text,
            channel_overrides=channel_overrides,
            extra=extra,
        )


# Streaming defaults — single source of truth so both StreamingConfig and
# StreamConsumerConfig agree on the out-of-the-box edit rhythm.  Tuned for
# Telegram's ~1 edit/s flood envelope: a touch under 1s lets the cadence
# breathe without bumping into rate limits, and a smaller buffer threshold
# makes short replies feel near-instant in DMs.
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24
DEFAULT_STREAMING_CURSOR: str = " ▉"


@dataclass
class StreamingConfig:
    """Configuration for real-time token streaming to messaging platforms."""
    enabled: bool = False
    # Transport selection:
    #   "auto"  — prefer native streaming-draft updates when the platform
    #             supports them (Telegram sendMessageDraft, Bot API 9.5+);
    #             fall back to edit-based when not.
    #   "draft" — explicitly request native drafts; falls back to edit when
    #             the platform/chat doesn't support them.
    #   "edit"  — progressive editMessageText only (legacy behaviour).
    #   "off"   — disable streaming entirely.
    #
    # Default is "auto": prefer native draft streaming on platforms that
    # support it (Telegram DMs via sendMessageDraft, Bot API 9.5+) and fall
    # back to edit-based streaming everywhere else. Adapters without draft
    # support report supports_draft_streaming() == False and use the edit path.
    transport: str = "auto"
    edit_interval: float = DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = DEFAULT_STREAMING_CURSOR
    # Ported from openclaw/openclaw#72038.  When >0, the final edit for
    # a long-running streamed response is delivered as a fresh message
    # if the original preview has been visible for at least this many
    # seconds, so the platform's visible timestamp reflects completion
    # time instead of the preview creation time.  Currently applied to
    # Telegram only (other platforms ignore the setting).  Default 0 disables
    # the fresh-message replacement path; set >0 to opt in.
    fresh_final_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "transport": self.transport,
            "edit_interval": self.edit_interval,
            "buffer_threshold": self.buffer_threshold,
            "cursor": self.cursor,
            "fresh_final_after_seconds": self.fresh_final_after_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        if not isinstance(data, dict) or not data:
            return cls()

        # ``mode`` is an ergonomic alias for the transport that ALSO implies
        # ``enabled``.  A config like ``streaming: {mode: auto}`` reads as
        # "turn streaming on, transport=auto" — matching the natural intent
        # of someone enabling streaming without also spelling out
        # ``enabled: true``.  Without this, ``mode`` was silently ignored and
        # streaming stayed disabled (``enabled`` defaults to False), which is
        # a surprising footgun: the whole reply buffers and sends at once.
        # ``mode: off`` disables streaming; an explicit ``enabled`` key always
        # wins so callers can force either state.
        #
        # ``transport`` alone does NOT imply ``enabled``: ``streaming.enabled``
        # is the documented master switch (see README.md
        # configuration.md), so a bare ``transport`` only selects HOW to stream
        # once streaming is on. Only the ``mode`` alias flips ``enabled``.
        raw_transport = data.get("transport")
        raw_mode = data.get("mode")
        # Normalize both through the same helper so YAML's bare ``off``/``on``
        # (parsed as bool False/True) become canonical tokens rather than
        # ``"false"``/``"true"``.
        picked = raw_transport if raw_transport is not None else raw_mode
        transport = _normalize_transport_token(picked)

        if "enabled" in data:
            enabled = _coerce_bool(data.get("enabled"), False)
        elif raw_mode is not None:
            # The ``mode`` alias (and only ``mode``) infers enabled:
            # ``off`` disables, anything else enables.
            enabled = _normalize_transport_token(raw_mode) != "off"
        else:
            enabled = False

        return cls(
            enabled=enabled,
            transport=transport,
            edit_interval=_coerce_float(
                data.get("edit_interval"), DEFAULT_STREAMING_EDIT_INTERVAL,
            ),
            buffer_threshold=_coerce_int(
                data.get("buffer_threshold"), DEFAULT_STREAMING_BUFFER_THRESHOLD,
            ),
            cursor=data.get("cursor", DEFAULT_STREAMING_CURSOR),
            fresh_final_after_seconds=_coerce_float(
                data.get("fresh_final_after_seconds"), 0.0
            ),
        )


# -----------------------------------------------------------------------------
# Built-in platform connection checkers
# -----------------------------------------------------------------------------
# Each callable receives a ``PlatformConfig`` and returns ``True`` when the
# platform is sufficiently configured to be considered "connected". Telegram
# and Mattermost use the generic ``token or api_key`` check.
_PLATFORM_CONNECTED_CHECKERS: dict[Platform, Callable[[PlatformConfig], bool]] = {}


@dataclass
class GatewayConfig:
    """
    Main gateway configuration.
    
    Manages all platform connections, session policies, and delivery settings.
    """
    # Platform configurations
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    
    # Session reset policies by type
    default_reset_policy: SessionResetPolicy = field(default_factory=SessionResetPolicy)
    reset_by_type: Dict[str, SessionResetPolicy] = field(default_factory=dict)
    reset_by_platform: Dict[Platform, SessionResetPolicy] = field(default_factory=dict)
    
    # Reset trigger commands
    reset_triggers: List[str] = field(default_factory=lambda: ["/new", "/reset"])

    # User-defined quick commands (slash commands that bypass the agent loop)
    quick_commands: Dict[str, Any] = field(default_factory=dict)
    
    # Storage paths
    sessions_dir: Path = field(default_factory=lambda: get_hermes_home() / "sessions")

    # Delivery settings
    always_log_local: bool = True  # Always save cron outputs to local files
    # Drop outbound "silence narration" messages (e.g. *(silent)*, 🔇, a bare
    # ".") pre-send. These are model hallucinations emitted when a persona has
    # nothing actionable to say; in bot-to-bot channels they mirror back and
    # forth, burning tokens and crashing models. Substrate-level guard that
    # survives SOUL.md/prompt drift across providers. Opt out with False for
    # raw passthrough.
    filter_silence_narration: bool = True

    # Session isolation in shared chats
    group_sessions_per_user: bool = True  # Isolate group/channel sessions per participant when user IDs are available
    thread_sessions_per_user: bool = False  # When False (default), threads are shared across all participants
    max_concurrent_sessions: Optional[int] = None  # Positive int caps simultaneous active chat sessions

    # Opt-in systemd event-loop watchdog. Zero preserves Type=simple and
    # disables sd_notify at runtime.
    systemd_watchdog_seconds: int = 0

    # In-process event-loop liveness watchdog (#69089). A daemon OS thread
    # probes the gateway loop with call_soon_threadsafe; after consecutive
    # missed probes it dumps all-thread stacks and hard-exits with the
    # service-restart code so the supervisor can revive the process. On by
    # default; set gateway.loop_watchdog: false in config.yaml to disable.
    loop_watchdog: bool = True

    # Unauthorized DM policy
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"

    # Streaming configuration
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Session store pruning: drop SessionEntry records older than this many
    # days from the in-memory dict and sessions.json.  Keeps the store from
    # growing unbounded in gateways serving many chats/threads/users over
    # months.  Pruning is invisible to users — if they resume, they get a
    # fresh session exactly as if the reset policy had fired.  0 = disabled.
    session_store_max_age_days: int = 90

    def __post_init__(self) -> None:
        self.systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            self.systemd_watchdog_seconds
        )

    def get_connected_platforms(self) -> List[Platform]:
        """Return list of platforms that are enabled and configured.

        Sorted by platform value so the rendered "Connected Platforms" list
        is byte-stable across gateway restarts.
        """
        connected = []
        for platform, config in self.platforms.items():
            if not config.enabled:
                continue
            if self._is_platform_connected(platform, config):
                connected.append(platform)
        return sorted(connected, key=lambda p: str(p.value))

    def _is_platform_connected(self, platform: Platform, config: PlatformConfig) -> bool:
        """Check whether a single platform is sufficiently configured."""
        # Telegram and Mattermost use token credentials.
        if config.token or config.api_key:
            return True

        # Platform-specific check
        checker = _PLATFORM_CONNECTED_CHECKERS.get(platform)
        if checker is not None:
            return checker(config)

        return False
    
    def get_home_channel(self, platform: Platform) -> Optional[HomeChannel]:
        """Get the home channel for a platform."""
        config = self.platforms.get(platform)
        if config:
            return config.home_channel
        return None
    
    def get_reset_policy(
        self, 
        platform: Optional[Platform] = None,
        session_type: Optional[str] = None
    ) -> SessionResetPolicy:
        """
        Get the appropriate reset policy for a session.
        
        Priority: platform override > type override > default
        """
        # Platform-specific override takes precedence
        if platform and platform in self.reset_by_platform:
            return self.reset_by_platform[platform]
        
        # Type-specific override (dm, group, thread)
        if session_type and session_type in self.reset_by_type:
            return self.reset_by_type[session_type]
        
        return self.default_reset_policy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {
                p.value: c.to_dict() for p, c in self.platforms.items()
            },
            "default_reset_policy": self.default_reset_policy.to_dict(),
            "reset_by_type": {
                k: v.to_dict() for k, v in self.reset_by_type.items()
            },
            "reset_by_platform": {
                p.value: v.to_dict() for p, v in self.reset_by_platform.items()
            },
            "reset_triggers": self.reset_triggers,
            "quick_commands": self.quick_commands,
            "sessions_dir": str(self.sessions_dir),
            "always_log_local": self.always_log_local,
            "filter_silence_narration": self.filter_silence_narration,
            "group_sessions_per_user": self.group_sessions_per_user,
            "thread_sessions_per_user": self.thread_sessions_per_user,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "systemd_watchdog_seconds": self.systemd_watchdog_seconds,
            "loop_watchdog": self.loop_watchdog,
            "unauthorized_dm_behavior": self.unauthorized_dm_behavior,
            "streaming": self.streaming.to_dict(),
            "session_store_max_age_days": self.session_store_max_age_days,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        data = _coerce_dict(data)
        platforms = {}
        platforms_data = _coerce_dict(data.get("platforms", {}))
        for platform_name, platform_data in platforms_data.items():
            if not isinstance(platform_data, dict):
                continue
            try:
                platform = Platform(platform_name)
                platforms[platform] = PlatformConfig.from_dict(platform_data)
            except ValueError:
                pass  # Skip unknown platforms
        
        reset_by_type = {}
        for type_name, policy_data in _coerce_dict(data.get("reset_by_type", {})).items():
            reset_by_type[type_name] = SessionResetPolicy.from_dict(policy_data)
        
        reset_by_platform = {}
        for platform_name, policy_data in _coerce_dict(data.get("reset_by_platform", {})).items():
            try:
                platform = Platform(platform_name)
                reset_by_platform[platform] = SessionResetPolicy.from_dict(policy_data)
            except ValueError:
                pass
        
        default_policy = SessionResetPolicy()
        if "default_reset_policy" in data:
            default_policy = SessionResetPolicy.from_dict(data["default_reset_policy"])
        
        sessions_dir = get_hermes_home() / "sessions"
        if "sessions_dir" in data:
            sessions_dir = Path(data["sessions_dir"])
        
        quick_commands = data.get("quick_commands", {})
        if not isinstance(quick_commands, dict):
            quick_commands = {}

        group_sessions_per_user = data.get("group_sessions_per_user")
        thread_sessions_per_user = data.get("thread_sessions_per_user")
        raw_gateway = data.get("gateway")
        nested_gateway = raw_gateway if isinstance(raw_gateway, dict) else {}
        if "systemd_watchdog_seconds" in data:
            systemd_watchdog_raw = data.get("systemd_watchdog_seconds")
            systemd_watchdog_key = "systemd_watchdog_seconds"
        else:
            systemd_watchdog_raw = nested_gateway.get("systemd_watchdog_seconds")
            systemd_watchdog_key = "gateway.systemd_watchdog_seconds"
        systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            systemd_watchdog_raw, systemd_watchdog_key
        )
        if "loop_watchdog" in data:
            loop_watchdog_raw = data.get("loop_watchdog")
        else:
            loop_watchdog_raw = nested_gateway.get("loop_watchdog")
        loop_watchdog = _coerce_bool(loop_watchdog_raw, True)
        if "max_concurrent_sessions" in data:
            max_concurrent_raw = data.get("max_concurrent_sessions")
            max_concurrent_key = "max_concurrent_sessions"
        else:
            max_concurrent_raw = nested_gateway.get("max_concurrent_sessions")
            max_concurrent_key = "gateway.max_concurrent_sessions"
        max_concurrent_sessions = _coerce_optional_positive_int(
            max_concurrent_raw,
            max_concurrent_key,
        )
        unauthorized_dm_behavior = _normalize_unauthorized_dm_behavior(
            data.get("unauthorized_dm_behavior"),
            "pair",
        )

        try:
            session_store_max_age_days = int(data.get("session_store_max_age_days", 90))
            session_store_max_age_days = max(session_store_max_age_days, 0)
        except (TypeError, ValueError):
            session_store_max_age_days = 90

        return cls(
            platforms=platforms,
            default_reset_policy=default_policy,
            reset_by_type=reset_by_type,
            reset_by_platform=reset_by_platform,
            reset_triggers=data.get("reset_triggers", ["/new", "/reset"]),
            quick_commands=quick_commands,
            sessions_dir=sessions_dir,
            always_log_local=_coerce_bool(data.get("always_log_local"), True),
            filter_silence_narration=_coerce_bool(
                data.get("filter_silence_narration"), True
            ),
            group_sessions_per_user=_coerce_bool(group_sessions_per_user, True),
            thread_sessions_per_user=_coerce_bool(thread_sessions_per_user, False),
            systemd_watchdog_seconds=systemd_watchdog_seconds,
            loop_watchdog=loop_watchdog,
            max_concurrent_sessions=max_concurrent_sessions,
            unauthorized_dm_behavior=unauthorized_dm_behavior,
            streaming=StreamingConfig.from_dict(data.get("streaming", {})),
            session_store_max_age_days=session_store_max_age_days,
        )

    def get_unauthorized_dm_behavior(self, platform: Optional[Platform] = None) -> str:
        """Return the effective unauthorized-DM behavior for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "unauthorized_dm_behavior" in platform_cfg.extra:
                return _normalize_unauthorized_dm_behavior(
                    platform_cfg.extra.get("unauthorized_dm_behavior"),
                    self.unauthorized_dm_behavior,
                )
        return self.unauthorized_dm_behavior

    def get_notice_delivery(self, platform: Optional[Platform] = None) -> str:
        """Return the effective notice-delivery mode for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "notice_delivery" in platform_cfg.extra:
                return _normalize_notice_delivery(
                    platform_cfg.extra.get("notice_delivery"),
                    "public",
                )
        return "public"


def load_gateway_config() -> GatewayConfig:
    """
    Load gateway configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables
    2. ~/.hermes/config.yaml (primary user-facing config)
    3. ~/.hermes/gateway.json (legacy — provides defaults under config.yaml)
    4. Built-in defaults
    """
    _home = get_hermes_home()
    gw_data: dict = {}

    # Legacy fallback: gateway.json provides the base layer.
    # config.yaml keys always win when both specify the same setting.
    gateway_json_path = _home / "gateway.json"
    if gateway_json_path.exists():
        try:
            with open(gateway_json_path, "r", encoding="utf-8") as f:
                gw_data = json.load(f) or {}
            logger.info(
                "Loaded legacy %s — consider moving settings to config.yaml",
                gateway_json_path,
            )
        except Exception as e:
            logger.warning("Failed to load %s: %s", gateway_json_path, e)

    # Primary source: config.yaml
    try:
        import yaml
        config_yaml_path = _home / "config.yaml"
        if config_yaml_path.exists():
            with open(config_yaml_path, encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}

            # Managed scope: overlay administrator-pinned values so the gateway
            # honors them too. This loader builds its own dict instead of going
            # through hermes_cli.config.load_config, so without this a managed
            # session_reset / quick_commands / stt / model would be ignored by
            # the messaging gateway. Fail-open via the shared helper.
            from hermes_cli import managed_scope
            yaml_cfg = managed_scope.apply_managed_overlay(yaml_cfg)

            # Shared nested-fallback source: settings meant to be top-level
            # keys are also accepted when a user nests them under `gateway:`
            # (e.g. via `hermes config set gateway.<key> ...`, which naturally
            # produces that shape). Every key below mirrors the precedent
            # already established for gateway.multiplex_profiles/streaming/
            # write_sessions_json: top-level wins, nested gateway.* falls back.
            gateway_section = yaml_cfg.get("gateway")

            # Map config.yaml keys → GatewayConfig.from_dict() schema.
            # Each key overwrites whatever gateway.json may have set.
            # Precedence contract: key-presence at the TOP LEVEL wins; the
            # nested gateway.* form is consulted only when the top-level key
            # is absent (not merely falsy/mistyped), so a present-but-empty
            # top-level value is never silently replaced by the nested one.
            sr = yaml_cfg.get("session_reset")
            if "session_reset" not in yaml_cfg and isinstance(gateway_section, dict):
                sr = gateway_section.get("session_reset")
            if sr and isinstance(sr, dict):
                gw_data["default_reset_policy"] = sr

            qc = yaml_cfg.get("quick_commands")
            if qc is None and isinstance(gateway_section, dict):
                qc = gateway_section.get("quick_commands")
            if qc is not None:
                if isinstance(qc, dict):
                    gw_data["quick_commands"] = qc
                else:
                    logger.warning(
                        "Ignoring invalid quick_commands in config.yaml "
                        "(expected mapping, got %s)",
                        type(qc).__name__,
                    )

            gateway_cfg = yaml_cfg.get("gateway")

            if "group_sessions_per_user" in yaml_cfg:
                gw_data["group_sessions_per_user"] = yaml_cfg["group_sessions_per_user"]
            elif isinstance(gateway_section, dict) and "group_sessions_per_user" in gateway_section:
                gw_data["group_sessions_per_user"] = gateway_section["group_sessions_per_user"]

            if "thread_sessions_per_user" in yaml_cfg:
                gw_data["thread_sessions_per_user"] = yaml_cfg["thread_sessions_per_user"]
            elif isinstance(gateway_section, dict) and "thread_sessions_per_user" in gateway_section:
                gw_data["thread_sessions_per_user"] = gateway_section["thread_sessions_per_user"]

            if isinstance(gateway_section, dict):
                if "max_concurrent_sessions" in gateway_section:
                    gw_data["max_concurrent_sessions"] = gateway_section["max_concurrent_sessions"]
                if "systemd_watchdog_seconds" in gateway_section:
                    gw_data["systemd_watchdog_seconds"] = gateway_section[
                        "systemd_watchdog_seconds"
                    ]

            if "max_concurrent_sessions" in yaml_cfg:
                gw_data["max_concurrent_sessions"] = yaml_cfg["max_concurrent_sessions"]

            streaming_cfg = yaml_cfg.get("streaming")
            if not isinstance(streaming_cfg, dict) and isinstance(gateway_section, dict):
                # Fall back to nested gateway.streaming written by
                # ``hermes config set gateway.streaming.*``
                streaming_cfg = gateway_section.get("streaming")
            if isinstance(streaming_cfg, dict):
                gw_data["streaming"] = streaming_cfg

            if "reset_triggers" in yaml_cfg:
                gw_data["reset_triggers"] = yaml_cfg["reset_triggers"]
            elif isinstance(gateway_section, dict) and "reset_triggers" in gateway_section:
                gw_data["reset_triggers"] = gateway_section["reset_triggers"]

            if "always_log_local" in yaml_cfg:
                gw_data["always_log_local"] = yaml_cfg["always_log_local"]
            elif isinstance(gateway_section, dict) and "always_log_local" in gateway_section:
                gw_data["always_log_local"] = gateway_section["always_log_local"]

            if "filter_silence_narration" in yaml_cfg:
                gw_data["filter_silence_narration"] = yaml_cfg[
                    "filter_silence_narration"
                ]
            elif isinstance(gateway_section, dict) and "filter_silence_narration" in gateway_section:
                gw_data["filter_silence_narration"] = gateway_section[
                    "filter_silence_narration"
                ]

            if "unauthorized_dm_behavior" in yaml_cfg:
                gw_data["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                    yaml_cfg.get("unauthorized_dm_behavior"),
                    "pair",
                )
            elif isinstance(gateway_section, dict) and "unauthorized_dm_behavior" in gateway_section:
                gw_data["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                    gateway_section.get("unauthorized_dm_behavior"),
                    "pair",
                )

            # Merge platform config into gw_data so runtime-only settings under
            # ``gateway.platforms`` are loaded the same way as top-level
            # ``platforms``. Merge nested first so top-level config keeps
            # precedence, matching the existing gateway.streaming fallback.
            gateway_platforms = gateway_cfg.get("platforms") if isinstance(gateway_cfg, dict) else None
            platforms_data = gw_data.setdefault("platforms", {})
            if not isinstance(platforms_data, dict):
                platforms_data = {}
                gw_data["platforms"] = platforms_data

            def _merge_platform_map(source_platforms: Any) -> None:
                if not isinstance(source_platforms, dict):
                    return
                for plat_name, plat_block in source_platforms.items():
                    if not isinstance(plat_block, dict):
                        continue
                    existing = platforms_data.get(plat_name, {})
                    if not isinstance(existing, dict):
                        existing = {}
                    # Deep-merge extra dicts so gateway.json defaults survive
                    merged_extra = {**existing.get("extra", {}), **plat_block.get("extra", {})}
                    if "enabled" in plat_block:
                        merged_extra["_enabled_explicit"] = True
                    merged = {**existing, **plat_block}
                    if merged_extra:
                        merged["extra"] = merged_extra
                    platforms_data[plat_name] = merged

            _merge_platform_map(gateway_platforms)
            _merge_platform_map(yaml_cfg.get("platforms"))

            # Also merge platform configs placed directly under ``gateway.*``
            # so subsections are discovered the
            # same way ``gateway.streaming`` is handled elsewhere.  Iterate
            # all ``gateway:*`` keys and merge only those that match a known
            # platform value, skipping reserved keys like ``platforms``.
            if isinstance(gateway_cfg, dict):
                _nested_platforms: dict = {}
                for _k, _v in gateway_cfg.items():
                    if _k == "platforms":
                        continue
                    try:
                        Platform(_k)
                    except (ValueError, AttributeError):
                        continue
                    if isinstance(_v, dict):
                        _nested_platforms[_k] = _v
                if _nested_platforms:
                    _merge_platform_map(_nested_platforms)

            if platforms_data:
                gw_data["platforms"] = platforms_data
            for plat in Platform:
                if plat == Platform.LOCAL:
                    continue
                platform_cfg = yaml_cfg.get(plat.value)
                _cfg_toplevel = isinstance(platform_cfg, dict)
                # Fall back to the platform's block under ``platforms`` /
                # ``gateway.platforms`` so shared-key bridging (allow_from,
                # require_mention, free_response_channels, …) still runs when
                # the user configured the platform only under those nested paths
                # and not via a top-level block.  Mirrors the identical fallback
                # already applied to the apply_yaml_config_fn dispatch below
                # (#44f3e51).
                # Note: ``enabled`` is only written to plat_data from a
                # top-level block (``_cfg_toplevel``); for nested-only configs
                # ``_merge_platform_map`` already merged it with the correct
                # precedence, so re-applying it here would overwrite that.
                if not _cfg_toplevel:
                    for _src in (gateway_platforms, yaml_cfg.get("platforms")):
                        if isinstance(_src, dict):
                            _candidate = _src.get(plat.value)
                            if isinstance(_candidate, dict):
                                platform_cfg = _candidate
                                break
                if not isinstance(platform_cfg, dict):
                    continue
                # Collect bridgeable keys from this platform section
                bridged = {}
                if "unauthorized_dm_behavior" in platform_cfg:
                    bridged["unauthorized_dm_behavior"] = _normalize_unauthorized_dm_behavior(
                        platform_cfg.get("unauthorized_dm_behavior"),
                        gw_data.get("unauthorized_dm_behavior", "pair"),
                    )
                if "notice_delivery" in platform_cfg:
                    bridged["notice_delivery"] = _normalize_notice_delivery(
                        platform_cfg.get("notice_delivery"),
                        "public",
                    )
                if "reply_prefix" in platform_cfg:
                    bridged["reply_prefix"] = platform_cfg["reply_prefix"]
                if "reply_in_thread" in platform_cfg:
                    bridged["reply_in_thread"] = platform_cfg["reply_in_thread"]
                if "cron_continuable_surface" in platform_cfg:
                    bridged["cron_continuable_surface"] = platform_cfg["cron_continuable_surface"]
                if "require_mention" in platform_cfg:
                    bridged["require_mention"] = platform_cfg["require_mention"]
                if "send_read_receipts" in platform_cfg:
                    bridged["send_read_receipts"] = platform_cfg["send_read_receipts"]
                if plat == Platform.TELEGRAM and "allowed_chats" in platform_cfg:
                    bridged["allowed_chats"] = platform_cfg["allowed_chats"]
                if plat == Platform.TELEGRAM and "group_allowed_chats" in platform_cfg:
                    bridged["group_allowed_chats"] = platform_cfg["group_allowed_chats"]
                if plat == Platform.TELEGRAM and "allowed_topics" in platform_cfg:
                    bridged["allowed_topics"] = platform_cfg["allowed_topics"]
                if "free_response_channels" in platform_cfg:
                    bridged["free_response_channels"] = platform_cfg["free_response_channels"]
                if "mention_patterns" in platform_cfg:
                    bridged["mention_patterns"] = platform_cfg["mention_patterns"]
                if "exclusive_bot_mentions" in platform_cfg:
                    bridged["exclusive_bot_mentions"] = platform_cfg["exclusive_bot_mentions"]
                if plat == Platform.TELEGRAM and "observe_unmentioned_group_messages" in platform_cfg:
                    bridged["observe_unmentioned_group_messages"] = platform_cfg["observe_unmentioned_group_messages"]
                if "dm_policy" in platform_cfg:
                    bridged["dm_policy"] = platform_cfg["dm_policy"]
                if "allow_from" in platform_cfg:
                    bridged["allow_from"] = platform_cfg["allow_from"]
                if "allow_admin_from" in platform_cfg:
                    bridged["allow_admin_from"] = platform_cfg["allow_admin_from"]
                if "user_allowed_commands" in platform_cfg:
                    bridged["user_allowed_commands"] = platform_cfg["user_allowed_commands"]
                if "group_policy" in platform_cfg:
                    bridged["group_policy"] = platform_cfg["group_policy"]
                if "group_allow_from" in platform_cfg:
                    bridged["group_allow_from"] = platform_cfg["group_allow_from"]
                if "group_allow_admin_from" in platform_cfg:
                    bridged["group_allow_admin_from"] = platform_cfg["group_allow_admin_from"]
                if "group_user_allowed_commands" in platform_cfg:
                    bridged["group_user_allowed_commands"] = platform_cfg["group_user_allowed_commands"]
                if "channel_prompts" in platform_cfg:
                    channel_prompts = platform_cfg["channel_prompts"]
                    if isinstance(channel_prompts, dict):
                        bridged["channel_prompts"] = {str(k): v for k, v in channel_prompts.items()}
                    else:
                        bridged["channel_prompts"] = channel_prompts
                if "gateway_restart_notification" in platform_cfg:
                    bridged["gateway_restart_notification"] = platform_cfg["gateway_restart_notification"]
                if "typing_indicator" in platform_cfg:
                    bridged["typing_indicator"] = platform_cfg["typing_indicator"]
                if "typing_status_text" in platform_cfg:
                    bridged["typing_status_text"] = platform_cfg["typing_status_text"]
                has_channel_overrides = "channel_overrides" in platform_cfg
                if has_channel_overrides:
                    raw_overrides = platform_cfg.get("channel_overrides")
                    if isinstance(raw_overrides, dict):
                        plat_data, _extra = _ensure_platform_extra_dict(
                            platforms_data, plat.value
                        )
                        plat_data["channel_overrides"] = {
                            str(cid): ov_data
                            for cid, ov_data in raw_overrides.items()
                            if isinstance(ov_data, dict)
                        }
                enabled_was_explicit = _cfg_toplevel and "enabled" in platform_cfg
                if not bridged and not enabled_was_explicit and not has_channel_overrides:
                    continue
                plat_data, extra = _ensure_platform_extra_dict(platforms_data, plat.value)
                if enabled_was_explicit:
                    plat_data["enabled"] = platform_cfg["enabled"]
                    # Preserve explicit disable against environment enablement.
                    extra["_enabled_explicit"] = True
                extra.update(bridged)

            from gateway.platforms.mattermost import apply_yaml_config as apply_mattermost_yaml
            from gateway.platforms.telegram import apply_yaml_config as apply_telegram_yaml

            for platform_name, bridge in (
                ("telegram", apply_telegram_yaml),
                ("mattermost", apply_mattermost_yaml),
            ):
                platform_cfg = yaml_cfg.get(platform_name)
                if not isinstance(platform_cfg, dict):
                    for source in (gateway_platforms, yaml_cfg.get("platforms")):
                        candidate = source.get(platform_name) if isinstance(source, dict) else None
                        if isinstance(candidate, dict):
                            platform_cfg = candidate
                            break
                if not isinstance(platform_cfg, dict):
                    continue
                seeded = bridge(yaml_cfg, platform_cfg)
                if isinstance(seeded, dict) and seeded:
                    _, extra = _ensure_platform_extra_dict(platforms_data, platform_name)
                    extra.update(seeded)

            # Bridge top-level require_mention to Telegram when the telegram: section
            # does not already provide one.  Users often write "require_mention: true"
            # at the top level alongside group_sessions_per_user, expecting it to work
            # the same way (#3979).
            _tl_require_mention = yaml_cfg.get("require_mention")
            if _tl_require_mention is not None:
                _tg_section = yaml_cfg.get("telegram") or {}
                if "require_mention" not in _tg_section:
                    _tg_plat = platforms_data.setdefault(Platform.TELEGRAM.value, {})
                    _tg_extra = _tg_plat.setdefault("extra", {})
                    _tg_extra.setdefault("require_mention", _tl_require_mention)
                    # Also bridge to the TELEGRAM_REQUIRE_MENTION env var that the
                    # adapter reads at runtime.  This used to live in the telegram_cfg
                    # block in core; it stays in core because it keys off the TOP-LEVEL
                    # require_mention (not a telegram: block), so the telegram plugin's
                    # apply_yaml_config_fn hook — which only runs when a telegram config
                    # block exists — can't cover the no-telegram-block case (#3979).
                    if not os.getenv("TELEGRAM_REQUIRE_MENTION"):
                        os.environ["TELEGRAM_REQUIRE_MENTION"] = str(_tl_require_mention).lower()

    except Exception as e:
        logger.warning(
            "Failed to process config.yaml — falling back to .env / gateway.json values. "
            "Check %s for syntax errors. Error: %s",
            _home / "config.yaml",
            e,
        )

    config = GatewayConfig.from_dict(gw_data)

    # Override with environment variables
    _apply_env_overrides(config)
    
    # --- Validate loaded values ---
    _validate_gateway_config(config)

    return config


def _validate_gateway_config(config: "GatewayConfig") -> None:
    """Validate and sanitize a loaded GatewayConfig in place.

    Called by ``load_gateway_config()`` after all config sources are merged.
    Extracted as a separate function for testability.
    """
    policy = config.default_reset_policy

    if not (0 <= policy.at_hour <= 23):
        logger.warning(
            "Invalid at_hour=%s (must be 0-23). Using default 4.", policy.at_hour
        )
        policy.at_hour = 4

    if policy.idle_minutes is None or policy.idle_minutes <= 0:
        logger.warning(
            "Invalid idle_minutes=%s (must be positive). Using default 1440.",
            policy.idle_minutes,
        )
        policy.idle_minutes = 1440

    # Warn about empty bot tokens — platforms that loaded an empty string
    # won't connect and the cause can be confusing without a log line.
    _token_env_names = PLATFORM_TOKEN_ENV_NAMES
    for platform, pconfig in config.platforms.items():
        if not pconfig.enabled:
            continue
        env_name = _token_env_names.get(platform)
        if env_name and pconfig.token is not None and not pconfig.token.strip():
            logger.warning(
                "%s is enabled but %s is empty. "
                "The adapter will likely fail to connect.",
                platform.value, env_name,
            )

    # Reject known-weak placeholder tokens.
    # Ported from openclaw/openclaw#64586: users who copy .env.example
    # without changing placeholder values get a clear startup error instead
    # of a confusing "auth failed" from the platform API.
    try:
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        has_usable_secret = None  # type: ignore[assignment]

    if has_usable_secret is not None:
        for platform, pconfig in config.platforms.items():
            if not pconfig.enabled:
                continue
            env_name = _token_env_names.get(platform)
            if not env_name:
                continue
            token = pconfig.token
            if token and token.strip() and not has_usable_secret(token, min_length=4):
                logger.error(
                    "%s is enabled but %s is set to a placeholder value ('%s'). "
                    "Set a real bot token before starting the gateway. "
                    "The adapter will NOT be started.",
                    platform.value, env_name, token.strip()[:6] + "...",
                )
                pconfig.enabled = False


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply retained gateway environment overrides."""
    getenv = _getenv_str

    def _enable_from_env(platform: Platform) -> PlatformConfig:
        platform_config = config.platforms.setdefault(platform, PlatformConfig())
        if not platform_config.enabled and not platform_config.extra.get("_enabled_explicit", False):
            platform_config.enabled = True
        return platform_config

    telegram_token = getenv("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        telegram_config = _enable_from_env(Platform.TELEGRAM)
        telegram_config.token = telegram_token

    telegram_reply_mode = getenv("TELEGRAM_REPLY_TO_MODE", "").lower()
    if telegram_reply_mode in {"off", "first", "all"}:
        telegram_config = config.platforms.setdefault(Platform.TELEGRAM, PlatformConfig())
        telegram_config.reply_to_mode = telegram_reply_mode

    telegram_fallback_ips = getenv("TELEGRAM_FALLBACK_IPS", "")
    if telegram_fallback_ips:
        telegram_config = config.platforms.setdefault(Platform.TELEGRAM, PlatformConfig())
        telegram_config.extra["fallback_ips"] = [
            ip.strip() for ip in telegram_fallback_ips.split(",") if ip.strip()
        ]

    telegram_home = getenv("TELEGRAM_HOME_CHANNEL")
    if telegram_home and Platform.TELEGRAM in config.platforms:
        config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id=telegram_home,
            name=getenv("TELEGRAM_HOME_CHANNEL_NAME", "Home"),
            thread_id=getenv("TELEGRAM_HOME_CHANNEL_THREAD_ID") or None,
        )

    mattermost_token = getenv("MATTERMOST_TOKEN")
    if mattermost_token:
        mattermost_url = getenv("MATTERMOST_URL", "")
        if not mattermost_url:
            logger.warning("MATTERMOST_TOKEN set but MATTERMOST_URL is missing")
        mattermost_config = _enable_from_env(Platform.MATTERMOST)
        mattermost_config.token = mattermost_token
        mattermost_config.extra["url"] = mattermost_url

    mattermost_home = getenv("MATTERMOST_HOME_CHANNEL")
    if mattermost_home and Platform.MATTERMOST in config.platforms:
        config.platforms[Platform.MATTERMOST].home_channel = HomeChannel(
            platform=Platform.MATTERMOST,
            chat_id=mattermost_home,
            name=getenv("MATTERMOST_HOME_CHANNEL_NAME", "Home"),
            thread_id=getenv("MATTERMOST_HOME_CHANNEL_THREAD_ID") or None,
        )

    idle_minutes = getenv("SESSION_IDLE_MINUTES")
    if idle_minutes:
        try:
            config.default_reset_policy.idle_minutes = int(idle_minutes)
        except ValueError:
            pass

    reset_hour = getenv("SESSION_RESET_HOUR")
    if reset_hour:
        try:
            config.default_reset_policy.at_hour = int(reset_hour)
        except ValueError:
            pass

    for platform_config in config.platforms.values():
        platform_config.extra.pop("_enabled_explicit", None)
