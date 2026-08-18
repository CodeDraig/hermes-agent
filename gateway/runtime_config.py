"""Live gateway configuration and provider-runtime resolution."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from gateway.config import ChannelOverride, GatewayConfig, Platform
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.fallback_config import get_fallback_chain
from hermes_constants import get_hermes_home, get_hermes_home_override

logger = logging.getLogger("gateway.run")

def _reload_runtime_env_preserving_config_authority() -> None:
    """Reload .env for fresh credentials without letting stale .env override config.

    Gateway processes are long-lived, so per-turn code reloads ~/.hermes/.env to
    pick up rotated API keys. config.yaml remains authoritative for agent budget
    settings such as agent.max_turns; otherwise a stale HERMES_MAX_ITERATIONS in
    .env can replace the startup bridge on later turns.

    In multiplex mode this is a NO-OP for the credential reload: secrets come
    from the per-turn ``set_secret_scope`` (installed by ``_profile_runtime_scope``)
    which loads the routed profile's ``.env`` into an isolated mapping. Mutating
    the process-global ``os.environ`` here would defeat that isolation and leak
    the default profile's keys to every profile's turns and subprocesses.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        # Credentials are resolved from the active profile's secret scope, not
        # os.environ. Still honor config.yaml's agent.max_turns bridge below
        # using the scoped home, but never reload .env into global env.
        _bridge_max_turns_from_config(get_hermes_home())
        return

    load_hermes_dotenv(
        hermes_home=get_hermes_home(),
        project_env=Path(__file__).resolve().parents[1] / '.env',
    )
    _bridge_max_turns_from_config(get_hermes_home())

def _bridge_max_turns_from_config(home: "Path") -> None:
    """Bridge config.yaml agent.max_turns into HERMES_MAX_ITERATIONS (a global)."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        # Presence-sensitive env bridge: raw read is deliberate (only keys the
        # user actually wrote get bridged); overlay + expansion applied below.
        cfg = read_user_config_raw(config_path)
        cfg = _expand_env_vars(cfg)
        if not isinstance(cfg, dict):
            cfg = {}
        # Managed scope: keep administrator-pinned values authoritative on every
        # turn too. This per-turn reload re-bridges config→env, so without the
        # overlay a managed agent.max_turns / timezone / redact_secrets would be
        # replaced by the user's value after the first turn. Fail-open.
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
    except Exception:
        return

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict) and "max_turns" in agent_cfg:
        os.environ["HERMES_MAX_ITERATIONS"] = str(agent_cfg["max_turns"])
    # config-authoritative knobs for the session-search index (config.yaml
    # sessions.* wins over stale env; env stays the cross-process carrier).
    sessions_cfg = cfg.get("sessions", {})
    if isinstance(sessions_cfg, dict):
        if "cjk_fts" in sessions_cfg:
            os.environ["HERMES_CJK_FTS"] = str(sessions_cfg["cjk_fts"])
        if "search_slow_ms" in sessions_cfg:
            os.environ["HERMES_SEARCH_SLOW_MS"] = str(sessions_cfg["search_slow_ms"])

def _current_max_iterations() -> int:
    """Return the current per-turn iteration budget after runtime env refresh."""
    _reload_runtime_env_preserving_config_authority()
    try:
        return int(os.getenv("HERMES_MAX_ITERATIONS", "500"))
    except (TypeError, ValueError):
        return 500

def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created create_agent instances.

    Provider is read from ``config.yaml`` ``model.provider`` (the single
    source of truth). ``resolve_runtime_provider()`` falls through to env
    var lookups internally for legacy compatibility, but the gateway does
    not consult environment variables for behavioral config — config.yaml
    is authoritative.

    If the primary provider fails with an authentication error, attempt to
    resolve credentials using the fallback provider chain from config.yaml
    before giving up.
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
        _get_model_config,
    )
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    try:
        runtime = resolve_runtime_provider()
    except AuthError as auth_exc:
        # Distinguish a transient rate-limit/quota cap (credentials are fine,
        # re-auth cannot help) from a genuine auth failure (expired/revoked
        # token). Both fall through to the fallback chain, but the log message
        # must not mislabel a quota exhaustion as an auth failure (#32790).
        if is_rate_limited_auth_error(auth_exc):
            logger.warning("Primary provider rate-limited (429): %s — trying fallback", auth_exc)
        else:
            logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    model_cfg = _get_model_config()
    max_tokens = None
    _env_mt = os.environ.get("HERMES_MAX_TOKENS")
    if _env_mt:
        try:
            max_tokens = int(_env_mt)
        except (ValueError, TypeError):
            max_tokens = None
    elif isinstance(model_cfg, dict):
        mt = model_cfg.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt
    # Fall back to a per-provider output cap (custom_providers max_output_tokens)
    # only when the documented global model.max_tokens isn't set, so the global
    # key always wins.
    if max_tokens is None:
        _runtime_mot = runtime.get("max_output_tokens")
        if isinstance(_runtime_mot, int) and _runtime_mot > 0:
            max_tokens = _runtime_mot

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "requested_provider": runtime.get("requested_provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "max_tokens": max_tokens,
    }

def _resolve_runtime_agent_kwargs_for_provider(provider: str) -> dict:
    """Resolve runtime credentials for a specific provider (e.g. from channel override)."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )
    try:
        runtime = resolve_runtime_provider(requested=provider)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "requested_provider": runtime.get("requested_provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }

def _credential_pool_for_provider(provider: Optional[str]):
    """Return the live credential pool for a provider id (e.g. ``custom:hyper``)."""
    if not provider or not str(provider).strip():
        return None
    try:
        return _resolve_runtime_agent_kwargs_for_provider(str(provider).strip()).get(
            "credential_pool"
        )
    except Exception:
        logger.debug(
            "Failed to resolve credential pool for provider=%s",
            provider,
            exc_info=True,
        )
        return None

def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_providers config."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        # Canonical gateway loader: managed overlay + ${VAR} expansion +
        # root-model normalization now reach the fallback chain too (a raw
        # read here used to miss administrator-pinned fallback_providers).
        cfg = _load_gateway_runtime_config()
        fb_list = get_fallback_chain(cfg)
        if not fb_list:
            return None
        for entry in fb_list:
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key

                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"),
                    explicit_base_url=entry.get("base_url"),
                    explicit_api_key=resolve_entry_api_key(entry),
                )
                # Log the literal `provider` key from config, not the resolved
                # runtime category — an Ollama fallback resolves through the
                # OpenAI-compatible path and would otherwise be logged as
                # "openrouter", contradicting the operator's config (#32790).
                logger.info(
                    "Fallback provider resolved: %s model=%s",
                    entry.get("provider") or runtime.get("provider"),
                    entry.get("model"),
                )
                return {
                    "api_key": runtime.get("api_key"),
                    "base_url": runtime.get("base_url"),
                    "provider": runtime.get("provider"),
                    "requested_provider": runtime.get("requested_provider"),
                    "api_mode": runtime.get("api_mode"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                    "credential_pool": runtime.get("credential_pool"),
                    "model": entry.get("model"),
                }
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
    except Exception:
        pass
    return None

def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value

def _gateway_config_home() -> Path:
    """Return the Hermes home that gateway config reads should use."""
    override = get_hermes_home_override()
    if override:
        return Path(override)
    return get_hermes_home()

def _load_gateway_config(config_path: "Path | None" = None) -> dict:
    """Load and parse a gateway config.yaml, returning {} on any error.

    Defaults to the active gateway home (so tests that monkeypatch
    ``get_hermes_home()`` still see their fixture). Callers handling multiplexed
    profile routes may pass that profile's explicit config path. The canonical
    path shares the mtime-keyed raw-yaml cache from
    ``hermes_cli.config.read_raw_config``.

    Managed scope is overlaid on the result (via the shared helper) so the
    gateway honors administrator-pinned values — neither read_raw_config nor a
    direct yaml.safe_load carries the managed merge on its own. Fail-open.
    """
    if config_path is None:
        config_path = _gateway_config_home() / 'config.yaml'
    raw: dict = {}
    used_canonical = False
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        # Fast path: if get_hermes_home() agrees with the canonical config
        # location, reuse the shared cache. Otherwise fall through to a
        # direct read (keeps test fixtures with a monkeypatched
        # get_hermes_home() working).
        if config_path == get_config_path():
            raw = read_raw_config()
            used_canonical = True
    except Exception:
        pass

    if not used_canonical:
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load gateway config from %s", config_path)
            raw = {}

    # Overlay managed scope. read_raw_config() returns the user's raw YAML
    # WITHOUT the managed merge (that lives in load_config/_load_config_impl),
    # so the overlay is required on both paths for the gateway to honor pinned
    # values. Helper is fail-open and a no-op when no managed scope exists.
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    if not isinstance(raw, dict):
        return {}
    # Canonicalize model-id aliases (model.name / model.model → model.default)
    # and migrate stale root-level provider/base_url into the model section.
    # The gateway bypasses load_config() (it reads raw YAML for speed), so the
    # normalization that load_config() applies must be replayed here or the
    # gateway would resolve an empty model for ``model: {name: <id>}`` configs
    # while the CLI resolves it correctly. See issue #34500. Fail-open.
    try:
        from hermes_cli.config import _normalize_root_model_keys
        raw = _normalize_root_model_keys(raw)
    except Exception:
        pass
    return raw

def _checkpoint_agent_kwargs(config: dict | None) -> dict:
    """Translate gateway checkpoint config into ``create_agent`` constructor args.

    The gateway reads raw YAML instead of ``load_config()``, so checkpoint
    defaults must be supplied here. The checkpoints section is mapping-only.
    """
    cp_cfg = config.get("checkpoints", {}) if isinstance(config, dict) else {}
    if not isinstance(cp_cfg, dict):
        raise TypeError("checkpoints must be a mapping")

    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["checkpoints"]
    return {
        "checkpoints_enabled": cp_cfg.get("enabled", defaults["enabled"]),
        "checkpoint_max_snapshots": cp_cfg.get(
            "max_snapshots", defaults["max_snapshots"],
        ),
        "checkpoint_max_total_size_mb": cp_cfg.get(
            "max_total_size_mb", defaults["max_total_size_mb"],
        ),
        "checkpoint_max_file_size_mb": cp_cfg.get(
            "max_file_size_mb", defaults["max_file_size_mb"],
        ),
    }

def _load_gateway_runtime_config() -> dict:
    """Load gateway config for runtime reads, expanding supported ``${VAR}`` refs.

    Runtime helpers should honor the same env-template expansion documented for
    ``config.yaml`` while still respecting tests that monkeypatch
    ``gateway.run.get_hermes_home()``. Build on ``_load_gateway_config()`` rather
    than calling the canonical loader directly so both behaviors stay aligned.

    Expansion failures are intentionally NOT swallowed — silently returning
    the unexpanded dict would mask the very bug this helper exists to fix.
    """
    cfg = _load_gateway_config()
    if not isinstance(cfg, dict) or not cfg:
        return {}
    from hermes_cli.config import _expand_env_vars

    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else {}

def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth.

    Without this, temporary create_agent instances (e.g. /compress) fall
    back to the hardcoded default which fails when the active provider is
    openai-codex.
    """
    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""

def _channel_override_lookup_keys(
    chat_id: str,
    *,
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> list[str]:
    """Ordered, de-duplicated keys for ``channel_overrides`` lookup.

    Matches ``resolve_channel_prompt`` semantics: exact thread/channel id first,
    then parent channel/forum id (Discord threads inherit parent overrides).
    """
    keys: list[str] = []
    seen: set[str] = set()
    for key in (chat_id, thread_id, parent_id):
        if not key:
            continue
        sk = str(key)
        if sk in seen:
            continue
        seen.add(sk)
        keys.append(sk)
    return keys

def _get_channel_override(
    config: GatewayConfig,
    platform: Platform,
    chat_id: str,
    *,
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Optional[ChannelOverride]:
    """Return per-channel override for this platform/chat_id, or None.

    Looks up ``channel_overrides`` by ``chat_id``, then ``thread_id``, then
    ``parent_id`` (forum threads / child channels inherit the parent entry).
    """
    platforms = getattr(config, "platforms", None)
    if not platforms:
        return None
    platform_config = platforms.get(platform)
    if not platform_config or not platform_config.channel_overrides:
        return None
    overrides = platform_config.channel_overrides
    for key in _channel_override_lookup_keys(
        chat_id, thread_id=thread_id, parent_id=parent_id
    ):
        ov = overrides.get(key)
        if ov is not None:
            return ov
    return None

def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None

def _home_target_env_var(platform_name: str) -> str:
    """Return the configured home-target env var for a platform.

    Consults built-in ``_HOME_TARGET_ENV_VARS`` first, then the plugin
    registry via ``cron.scheduler._resolve_home_env_var``, then falls back
    to ``<PLATFORM>_HOME_CHANNEL`` for unknown names.
    """
    from cron.scheduler import _resolve_home_env_var

    resolved = _resolve_home_env_var(platform_name)
    if resolved:
        return resolved
    return f"{platform_name.upper()}_HOME_CHANNEL"

def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"
