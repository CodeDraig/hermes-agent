"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run

    # Or from CLI
    python cli.py --gateway
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import site
import sys
import signal
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Any, List

from agent.async_utils import safe_schedule_threadsafe
from hermes_cli.config import _is_ssh_remote_tilde_cwd, cfg_get












def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Some Windows restart paths run the gateway under uv's base ``pythonw.exe``
    to avoid the venv launcher respawning a visible console interpreter.  That
    mode can import the source tree via cwd/PYTHONPATH but still miss optional
    packages installed only in ``venv/Lib/site-packages`` (notably the MCP SDK).
    Patch the live process before MCP discovery so tool injection does not
    depend on every launcher preserving PYTHONPATH perfectly.
    """
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(project_root / "venv")

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue

        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitepackages() semantics matter here: pywin32, used by the MCP
        # SDK on Windows, relies on .pth processing to expose pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        pythonpath = [project_entry, site_entry]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        return










def _gateway_loop_exception_handler(
    loop: "asyncio.AbstractEventLoop", context: Dict[str, Any]
) -> None:
    """Loop-level safety net for transient network errors.

    Installed once during :func:`start_gateway`. Catches the
    ``telegram.error.TimedOut`` crash class (issues #31066 / #31110)
    and any peer transient network error before it can kill the
    gateway process. Logs at WARNING with full traceback so the
    originating call site stays diagnosable; non-transient errors
    are forwarded to the default loop handler so real bugs still
    surface.
    """
    exc = context.get("exception")
    if exc is not None and _is_transient_network_error(exc):
        task = context.get("future") or context.get("task")
        task_name = ""
        if task is not None:
            try:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
            except Exception:
                task_name = repr(task)
        logger.warning(
            "Gateway swallowed transient network error from %s: %s: %s",
            task_name or "<unknown task>",
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    # Fall back to the default handler for anything we don't recognise.
    loop.default_exception_handler(context)





























from gateway.history import (
    _float_env,
)
from gateway.message_router import (
    MessageRouter,
)
from gateway.process_notifications import ProcessNotifications
from gateway.platform_runtime import PlatformRuntime
from gateway.lifecycle import LifecycleRuntime, _shutdown_gateway_health_export
from gateway.turn_execution import TurnExecution
from gateway.response_filters import (
    _is_transient_network_error,
)
from gateway.runtime_config import (
    _load_gateway_config,
)

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run before aiohttp or another HTTP client is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python.

    Windows startup paths (Desktop, Scheduled Tasks, installer children) can
    occasionally inherit a stale SSL_CERT_FILE. Returning just because the
    variable is present makes every later httpx/OpenAI client construction fail
    with FileNotFoundError from ssl.load_verify_locations(). Treat a missing
    path as unset and fall back to certifi instead.
    """
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist",
            configured_cert,
        )
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return













# Mark this process as a gateway so cli.py's module-level load_cli_config()
# knows not to clobber TERMINAL_CWD if lazily imported.
os.environ["_HERMES_GATEWAY"] = "1"

_ensure_ssl_certs()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
from hermes_constants import get_hermes_home
_hermes_home = get_hermes_home()

# Load environment variables from ~/.hermes/.env first.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')









# Platforms that bind a host TCP port (HTTP/webhook listeners). In a profile
# multiplexer the default profile owns the single shared listener and serves
# every profile through the /p/<profile>/ URL prefix, so a SECONDARY profile
# enabling one of these is always a misconfiguration. We skip that secondary
# profile (SecondaryPortBindingConfigError) so a single bad profile cannot
# take down the whole multiplexer. The set lives in gateway.config so the
# dashboard's pre-write validation enforces the same policy.










_DOCKER_VOLUME_SPEC_RE = re.compile(r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$")
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}

# This env var is internal bridge plumbing, not a user-facing configuration
# source. Initialize it from the canonical config default after dotenv loading
# so an ambient process/.env value can never control lease safety on its own.
from hermes_cli.config_defaults import DEFAULT_CONFIG as _DEFAULT_CONFIG

os.environ["HERMES_TURN_LEASE_TIMEOUT"] = str(
    _DEFAULT_CONFIG["agent"]["gateway_turn_lease_timeout"]
)

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        # Presence-sensitive env bridge: raw read is deliberate — only keys the
        # user actually wrote may be bridged (a defaults merge would export the
        # whole DEFAULT_CONFIG into the env). Overlay + expansion applied below.
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        _cfg = read_user_config_raw(_config_path)
        # Expand ${ENV_VAR} references before bridging to env vars.
        _cfg = _expand_env_vars(_cfg)
        if not isinstance(_cfg, dict):
            _cfg = {}
        # Managed scope: overlay administrator-pinned values BEFORE bridging to
        # env vars, so a managed timezone / redact_secrets / max_turns / terminal
        # setting wins over the user's value at the env layer too. This bridge
        # reads config.yaml directly (not via load_config), so without the
        # overlay every HERMES_*/TERMINAL_* env var below would carry the user's
        # value even when an administrator pinned it. Fail-open via the helper.
        try:
            from hermes_cli import managed_scope
            _cfg = managed_scope.apply_managed_overlay(_cfg)
        except Exception:
            pass
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_backend = str(
                _terminal_cfg.get("backend") or os.environ.get("TERMINAL_ENV") or ""
            ).strip().lower()
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "degraded_mode": "TERMINAL_DEGRADED_MODE",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "home_mode": "TERMINAL_HOME_MODE",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "docker_env": "TERMINAL_DOCKER_ENV",
                "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
                "docker_shm_size": "TERMINAL_DOCKER_SHM_SIZE",
                "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
                "docker_network": "TERMINAL_DOCKER_NETWORK",
                "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
                "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
                "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
                "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    # Skip cwd placeholder values (".", "auto", "cwd") — the
                    # gateway resolves these to Path.home() later (line ~255).
                    # Writing the raw placeholder here would just be noise.
                    # Only bridge explicit absolute paths from config.yaml.
                    if _cfg_key == "cwd" and str(_val) in {".", "auto", "cwd"}:
                        continue
                    # Expand shell tilde in local/container cwd so subprocess.Popen
                    # never receives a literal "~/" which the kernel rejects.
                    # SSH cwd is interpreted by the remote shell, so preserve
                    # "~" / "~/..." for the SSH backend instead of expanding it
                    # to the Hermes host/container HOME (often /opt/data). Shared
                    # predicate with terminal_tool so the two sites can't drift.
                    if _cfg_key == "cwd" and isinstance(_val, str):
                        if not _is_ssh_remote_tilde_cwd(_terminal_backend, _val.strip()):
                            _val = os.path.expanduser(_val)
                    if isinstance(_val, (list, dict)):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        # Compression config is read directly from config.yaml by run_agent.py
        # and auxiliary_client.py — no env var bridging needed.
        # Auxiliary model/direct-endpoint overrides (vision, web_extract,
        # approval, plus any plugin-registered auxiliary tasks).
        # Each task has provider/model/base_url/api_key; bridge non-default
        # values to env vars named AUXILIARY_<KEY_UPPER>_*. The legacy
        # hard-coded list (vision/web_extract/approval) is replaced by a
        # dynamic loop so plugin-registered tasks benefit from the same
        # config→env bridging without core knowing about each one.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            # Built-in tasks that previously had explicit env-var bridging.
            # Kept here as the canonical bridged set; plugin tasks are added
            # below via the plugin auxiliary registry.
            _aux_bridged_keys = {"vision", "web_extract", "approval"}
            try:
                from hermes_cli.plugins import get_plugin_auxiliary_tasks
                for _entry in get_plugin_auxiliary_tasks():
                    _aux_bridged_keys.add(_entry["key"])
            except Exception:
                # Plugin discovery failure must not break gateway startup;
                # built-in bridging stays intact.
                pass

            for _task_key in _aux_bridged_keys:
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                _base_url = str(_task_cfg.get("base_url", "")).strip()
                _api_key = str(_task_cfg.get("api_key", "")).strip()
                _upper = _task_key.upper()
                if _prov and _prov != "auto":
                    os.environ[f"AUXILIARY_{_upper}_PROVIDER"] = _prov
                if _model:
                    os.environ[f"AUXILIARY_{_upper}_MODEL"] = _model
                if _base_url:
                    os.environ[f"AUXILIARY_{_upper}_BASE_URL"] = _base_url
                if _api_key:
                    os.environ[f"AUXILIARY_{_upper}_API_KEY"] = _api_key
        # config.yaml is the documented, authoritative source for these
        # settings — it unconditionally wins over .env values. Previously
        # the guards below read `if X not in os.environ` and let stale
        # .env entries (e.g. HERMES_MAX_ITERATIONS=60 written by an old
        # `hermes setup` run) silently shadow the user's current config.
        # See PR #18413 / the 60-vs-500 max_turns incident.
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
            if "gateway_timeout" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT"] = str(_agent_cfg["gateway_timeout"])
            if "gateway_turn_lease_timeout" in _agent_cfg:
                os.environ["HERMES_TURN_LEASE_TIMEOUT"] = str(
                    _agent_cfg["gateway_turn_lease_timeout"]
                )
            if "gateway_timeout_warning" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT_WARNING"] = str(_agent_cfg["gateway_timeout_warning"])
            if "gateway_notify_interval" in _agent_cfg:
                os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] = str(_agent_cfg["gateway_notify_interval"])
            if "session_stall_timeout" in _agent_cfg:
                os.environ["HERMES_SESSION_STALL_TIMEOUT"] = str(
                    _agent_cfg["session_stall_timeout"]
                )
            if "reconnect_attention_after" in _agent_cfg:
                # Internal bridge only — config.yaml (agent.reconnect_attention_after)
                # is the documented, user-facing setting.
                os.environ["HERMES_RECONNECT_ATTENTION_AFTER_SECONDS"] = str(
                    _agent_cfg["reconnect_attention_after"]
                )
            if "restart_drain_timeout" in _agent_cfg:
                os.environ["HERMES_RESTART_DRAIN_TIMEOUT"] = str(_agent_cfg["restart_drain_timeout"])
            if "cron_drain_timeout" in _agent_cfg:
                os.environ["HERMES_CRON_DRAIN_TIMEOUT"] = str(_agent_cfg["cron_drain_timeout"])
            if "gateway_auto_continue_freshness" in _agent_cfg:
                os.environ["HERMES_AUTO_CONTINUE_FRESHNESS"] = str(
                    _agent_cfg["gateway_auto_continue_freshness"]
                )
            if "gateway_startup_restore_drain_timeout" in _agent_cfg:
                os.environ["HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT"] = str(
                    _agent_cfg["gateway_startup_restore_drain_timeout"]
                )
        # config-authoritative knobs for the session-search index; same
        # bridge semantics as the agent settings above.
        _sessions_cfg = _cfg.get("sessions", {})
        if _sessions_cfg and isinstance(_sessions_cfg, dict):
            if "cjk_fts" in _sessions_cfg:
                os.environ["HERMES_CJK_FTS"] = str(_sessions_cfg["cjk_fts"])
            if "search_slow_ms" in _sessions_cfg:
                os.environ["HERMES_SEARCH_SLOW_MS"] = str(
                    _sessions_cfg["search_slow_ms"]
                )
        _display_cfg = _cfg.get("display", {})
        if _display_cfg and isinstance(_display_cfg, dict):
            if "busy_input_mode" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = str(_display_cfg["busy_input_mode"])
            if "busy_ack_enabled" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_ACK_ENABLED"] = str(_display_cfg["busy_ack_enabled"])
            # This process-level env var is documented as an override for
            # service managers, so preserve it when already set. Other display
            # bridges stay config-authoritative for backwards compatibility.
            if (
                "busy_steer_ack_enabled" in _display_cfg
                and "HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED" not in os.environ
            ):
                os.environ["HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED"] = str(
                    _display_cfg["busy_steer_ack_enabled"]
                )
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str):
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
        # Gateway settings (media delivery allowlist + recency trust + strict mode)
        _gateway_cfg = _cfg.get("gateway", {})
        if isinstance(_gateway_cfg, dict):
            _strict = _gateway_cfg.get("strict")
            if _strict is not None:
                os.environ["HERMES_MEDIA_DELIVERY_STRICT"] = (
                    "1" if _strict else "0"
                )
            _allow_dirs = _gateway_cfg.get("media_delivery_allow_dirs")
            if _allow_dirs:
                if isinstance(_allow_dirs, str):
                    _allow_dirs_str = _allow_dirs
                elif isinstance(_allow_dirs, (list, tuple)):
                    _allow_dirs_str = os.pathsep.join(str(p) for p in _allow_dirs if p)
                else:
                    _allow_dirs_str = ""
                if _allow_dirs_str:
                    os.environ["HERMES_MEDIA_ALLOW_DIRS"] = _allow_dirs_str
            _trust_recent = _gateway_cfg.get("trust_recent_files")
            if _trust_recent is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_FILES"] = (
                    "1" if _trust_recent else "0"
                )
            _trust_recent_seconds = _gateway_cfg.get("trust_recent_files_seconds")
            if _trust_recent_seconds is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_SECONDS"] = str(_trust_recent_seconds)
            # Bridge gateway.platform_connect_timeout → the internal env var the
            # connect path + Discord adapter ready-wait both read (#19776).
            # Unlike the agent.*/display.* bridges above (config-authoritative),
            # this env var is the manual-override escape hatch, so it WINS if
            # already set explicitly; otherwise config.yaml supplies the value.
            if (
                "platform_connect_timeout" in _gateway_cfg
                and not os.environ.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
            ):
                os.environ["HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT"] = str(
                    _gateway_cfg["platform_connect_timeout"]
                )
    except Exception as _bridge_err:
        # Previously this was silent (`except Exception: pass`), which
        # hid partial bridge failures and let .env defaults shadow
        # config.yaml values — users observed max_turns=500 in config
        # but a 60-iteration cap in practice. Surface the failure to
        # stderr so operators see it even though `logger` is not yet
        # initialized at module-import time (logger is defined further
        # down this module).
        print(
            f"  Warning: config.yaml → env bridge failed: "
            f"{type(_bridge_err).__name__}: {_bridge_err}",
            file=sys.stderr,
        )
        print(
            "  Gateway will fall back to .env values, which may not match "
            "your current config.yaml. Run `hermes doctor` to investigate.",
            file=sys.stderr,
        )

# Apply IPv4 preference if configured (before any HTTP clients are created).
try:
    from hermes_constants import apply_ipv4_preference
    _network_cfg = (_cfg if '_cfg' in dir() else {}).get("network", {})
    if isinstance(_network_cfg, dict) and _network_cfg.get("force_ipv4"):
        apply_ipv4_preference(force=True)
except Exception as _bootstrap_exc:
    print(f"  Warning: IPv4 preference application failed: {_bootstrap_exc}", file=sys.stderr)

# Validate config structure early — log warnings so gateway operators see problems
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception as _bootstrap_exc:
    print(f"  Warning: config validation failed: {_bootstrap_exc}", file=sys.stderr)

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# HERMES_EXEC_ASK is set in start_gateway(), not at import time. Importing this
# module from CLI tools (e.g. send_message → _gateway_runner_ref) must not flip
# interactive CLI sessions into ask-mode, or Dangerous Command prompts become
# silent pending_approval with no Approve/Deny UI.

# Set terminal working directory for messaging platforms.
# config.yaml terminal.cwd is the canonical source (bridged to TERMINAL_CWD
# by the config bridge above). Placeholder values are resolved per backend.
from gateway.cwd_placeholder import CWD_PLACEHOLDERS, resolve_placeholder_terminal_cwd

_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in CWD_PLACEHOLDERS:
    _resolved_cwd = resolve_placeholder_terminal_cwd(
        configured_cwd=_configured_cwd,
        terminal_backend=os.environ.get("TERMINAL_ENV", ""),
        home_fallback=str(Path.home()),
    )
    if _resolved_cwd is None:
        os.environ.pop("TERMINAL_CWD", None)
    else:
        os.environ["TERMINAL_CWD"] = _resolved_cwd

from gateway.config import (
    Platform,
    GatewayConfig,
)
from gateway.session import (
    AsyncSessionStore,
    SessionStore,
    SessionSource,
)
from gateway.delivery import (
    DeliveryRouter,
)
from gateway.session_runtime import GatewaySessionRuntime
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.agent_cache import GatewayAgentCache
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.turn_processes import (
    _INTERRUPT_REASON_TIMEOUT,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
)


logger = logging.getLogger(__name__)


# Sentinel for "caller did not pass metadata" vs "caller passed None".
_UNSET = object()






























from gateway.message_router import (
    STOP_REQUEST_REASON as _INTERRUPT_REASON_STOP,
)
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
















































# Max seconds between platform reconnect retries (primary watcher and
# secondary-profile reconnects share this policy — tune in one place).
_RECONNECT_BACKOFF_CAP = 300

# Seconds a platform may sit continuously in the reconnect queue before the
# watcher flags it NEEDS_ATTENTION in runtime status. Retrying never stops
# (auto-pause was deliberately removed — a transient outage must self-heal
# without operator action); this only makes a *long-lived* retry loop loud so
# owners and fleet monitoring can distinguish hour one from week three.
# A dead bot token, a revoked Discord intent, or a deterministically crashing
# sidecar all present as "retrying" forever without this signal.
# User-facing setting: agent.reconnect_attention_after in config.yaml
# (bridged to this env var above). 0 disables.
_RECONNECT_ATTENTION_AFTER_SECONDS = _float_env(
    "HERMES_RECONNECT_ATTENTION_AFTER_SECONDS", 7200
)









class GatewayRunner(
    GatewayAuthorizationMixin,
    GatewayKanbanWatchersMixin,
    GatewaySlashCommandsMixin,
    PlatformRuntime,
    ProcessNotifications,
    LifecycleRuntime,
    MessageRouter,
    TurnExecution,
):
    """
    Main gateway controller.

    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config if config is not None else _load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        # When non-None, SessionDB init failed — the gateway broadcasts a
        # one-time warning to the home channel(s) after connecting, so the
        # user knows persistence is broken instead of discovering it later
        # via a missing /resume or empty history (#88235).
        self._session_db_init_error: Optional[str] = None
        self._warn_if_docker_media_delivery_is_risky()
        from gateway.runtime_registry import register_runner

        register_runner(self)

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._restart_after_turn_timeout = self._load_restart_after_turn_timeout()
        self._cron_drain_timeout = self._load_cron_drain_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_providers = self._load_fallback_providers()

        # Wire process registry into session store for reset protection.
        # A background process older than the configured threshold (default 24h,
        # session_reset.bg_process_max_age_hours) is treated as stale and no
        # longer blocks session idle / daily reset — see #29177. The process is
        # NOT killed, only ignored by the reset guard.
        from tools.process_registry import process_registry
        _bg_max_age_hours = getattr(
            self.config.default_reset_policy, "bg_process_max_age_hours", 24
        )
        _bg_max_age_seconds = (
            _bg_max_age_hours * 3600 if _bg_max_age_hours and _bg_max_age_hours > 0 else None
        )
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(
                key, max_active_age=_bg_max_age_seconds,
            ),
        )
        # One enforced loop-side boundary for the synchronous SessionStore.
        # Sync helpers keep using ``session_store`` directly; async gateway
        # handlers call this facade and await every operation.
        self._async_session_store = AsyncSessionStore(self.session_store)
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_cleanly = False
        self._exit_with_failure = False
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._draining = False
        self._systemd_watchdog = None
        self._restart_requested = False
        self._restart_task_started = False
        self._restart_detached = False
        self._restart_via_service = False
        self._detached_restart_helper_started = False
        self._restart_command_source: Optional[SessionSource] = None
        # Monotonic-ish wall clock of when this GatewayRunner was constructed.
        # Used by the /restart redelivery guard to bound the window in which a
        # missing dedup marker is treated as a stale redelivery.
        self._startup_time: float = time.time()
        # Set True at startup when this process booted as the result of a
        # chat-originated /restart (i.e. .restart_notify.json existed on boot).
        # A one-shot signal consumed by _is_stale_restart_redelivery so the
        # marker-missing fallback only suppresses a /restart when we KNOW we
        # just came out of a restart cycle — never on a genuine fresh boot.
        self._booted_from_restart: bool = False
        self._stop_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._reconnect_watcher_task: Optional[asyncio.Task] = None
        self._shutdown_watchdog_done: Optional[threading.Event] = None
        self._executor_lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set on gateway stop so the recreate-on-shutdown path can't resurrect
        # the pool during a real shutdown.
        self._executor_closing = False
        # ALL per-session state (turn / conversation / persistent scopes)
        # lives in one container — see gateway/session_state.py.  Access via
        # self.sessions.state(key) (get-or-create) or
        # self.sessions.peek(key) (read-only).
        self.sessions = GatewaySessionRuntime()
        # Per-SESSION_ID turn lease (#64934): serializes the
        # [load history → run → flush] region when two ROUTING KEYS resolve
        # to one session_id (switch_session's many-to-one mapping). The
        # routing-key guards above cannot see that overlap. Acquired in
        # _handle_message_with_agent after session resolution is final,
        # released via _release_turn_lease in the same method's finally.
        # Tokens for held turn leases, keyed by (routing key, run generation)
        # so release is granted per-turn and a stale unwind can never free a
        # newer turn's lease (#28686 ownership lesson).
        # Held turn-lease tokens live on SessionState.turn.lease_token /
        # .lease_generation (the old dict was keyed (routing key, generation)
        # so a stale unwind could never free a newer turn's lease — the
        # generation field preserves that ownership check, #28686).
        # Runner-level queued interrupt text lives on
        # SessionState.persistent.pending_command_text (NOTE: distinct from
        # the adapter-level _pending_messages Dict[str, MessageEvent] in
        # gateway/platforms/base.py, which shares the legacy name).
        # Last successfully-resolved (non-empty) model, keyed by session. Used
        # as a fallback when a fresh config read transiently returns an empty
        # model (e.g. an mtime-keyed config-cache miss during a post-interrupt
        # recovery turn). Without this, the agent is built with model="" and
        # every API call fails HTTP 400 "No models provided" — the session goes
        # silent until the user manually re-sends. See #35314. The ``"*"``
        # session entry holds a process-wide last-known-good for sessions
        # seen for the first time.  Lives on
        # SessionState.conversation.last_resolved_model.
        # Overflow buffer for explicit /queue commands.  The adapter-level
        # _pending_messages dict is a single slot per session (designed for
        # "next-turn" follow-ups where repeated sends collapse into one
        # event).  /queue has different semantics: each invocation must
        # produce its own full agent turn, in FIFO order, with no merging.
        # When the slot is occupied, additional /queue items land here and
        # are promoted one-at-a-time after each run's drain.  Cleared on
        # /new and /reset.  /model and other mid-session operations
        # preserve the queue.  Lives on SessionState.conversation.queued_events;
        # native image paths, busy-ack debounce timestamps and the monotonic
        # run-generation counter (#28686, NEVER reset) live on SessionState too.
        # Session keys that already received a stall notification for the
        # current stall episode (cleared when pending clears / activity resumes
        # / conversation boundary). See gateway.session_stall.
        # Startup restore gate: while restart-interrupted sessions are being
        # auto-resumed, real inbound messages are queued instead of competing
        # with the synthetic resume turns for the same session.  The queued
        # events drain only after all startup resume tasks have finished.
        self._startup_restore_in_progress = False
        # Set by start_gateway() only for an explicit ``--replace`` launch.
        # _connect_initial_adapter_with_timeout scopes it to each adapter's
        # cold-start connect and removes it before any reconnect can run.
        self._platform_lock_takeover_on_start = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # LRU cache of live SessionSources keyed by session_key. Used by
        # fallback routing paths (shutdown notifications, synthetic
        # background-process events) when the persisted origin is missing
        # and _parse_session_key can't recover thread_id. Capped so it
        # cannot grow unbounded over a long-running gateway lifetime.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512
        # Completion delivery is intentionally lifecycle-scoped. This closes
        # duplicate queue/watcher races inside one gateway without pretending
        # the adapter call and a persistence write can be exactly-once across
        # a process crash. Any durable async-delegation replay state remains
        # owned by tools.async_delegation, not a parallel gateway ledger.
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight: set[tuple[str, str, object]] = set()
        self._completion_deliveries_delivered: "OrderedDict[tuple[str, str, object], None]" = OrderedDict()
        self._completion_delivery_retention = 2048
        # Agent-triggered terminal completions from one conversation often land
        # in the same scheduler tick.  Hold them briefly so the agent receives
        # one synthetic turn instead of one turn per process (#70300).
        self._completion_notification_batches: dict[tuple[str, ...], list[tuple[str, dict, asyncio.Future]]] = {}
        self._completion_notification_batch_tasks: dict[tuple[str, ...], asyncio.Task] = {}
        self._completion_notification_batch_flush_tasks: set[asyncio.Task] = set()
        self._completion_notification_batch_window = 0.1
        self._completion_notification_batches_stopping = False

        # Cache create_agent instances per session to preserve prompt caching.
        # Without this, a new create_agent is created per message, rebuilding the
        # system prompt (including memory) every turn — breaking prefix cache
        # and costing ~10x more on providers with prompt caching (Anthropic).
        self.agent_cache = GatewayAgentCache(
            sessions=self.sessions,
            session_store=self.session_store,
            session_db=lambda: self._session_db,
            load_config=_load_gateway_config,
        )

        # Conversation-scoped per-session state (/model, /model --once,
        # /reasoning, /fast overrides; per-turn sidecar notes; ephemeral
        # context pin; last-delivered voice-channel context) lives on
        # SessionState.conversation — see gateway/session_state.py.
        self._kanban_notifier_profile = self._active_profile_name()
        # Pending exec approvals live on SessionState.persistent.approvals.

        # Track platforms that failed to connect for background reconnection.
        # Key: Platform enum, Value: {"config": platform_config, "attempts": int, "next_retry": float}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}

        # Strong refs to detached fatal-error handler tasks (see
        # _handle_adapter_fatal_error) so the event loop can't GC them mid-run.
        self._fatal_handler_tasks: set = set()

        # Pending /update prompt flags live on
        # SessionState.persistent.update_prompt_pending.

        # Slash-confirm state lives in tools.slash_confirm (module-level),
        # so platform adapters can resolve callbacks without a backref to
        # this runner.  Keep a local counter for confirm_id generation so
        # IDs stay compact (button callback_data has a 64-byte cap on
        # some platforms).
        import itertools as _itertools
        self._slash_confirm_counter = _itertools.count(1)

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message create_agent instances.



        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable

        # Startup heads-up (#30882): a gateway in manual approval mode with no
        # automated risk assessor (tirith disabled AND no auxiliary.approval
        # model) can only gate dangerous commands / execute_code scripts via
        # live in-chat approval. With approval routing fixed, those actions now
        # fail closed (block) rather than silently auto-running — surface that
        # so operators knowingly enable tirith or configure auxiliary.approval
        # for unattended gateways.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation."
                )
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)

        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import AsyncSessionDB, SessionDB
            self._session_db = AsyncSessionDB(SessionDB())
        except Exception as e:
            # WARNING (not DEBUG) so the failure appears in errors.log — matches
            # cli.py's handling of the same init path.  Users hitting NFS-mounted
            # HERMES_HOME silently lost /resume, /title, /history, /branch, and
            # session search without this.  The underlying cause (usually
            # "locking protocol" from NFS) is now also captured by
            # hermes_state.get_last_init_error() for slash-command error strings.
            logger.warning("SQLite session store not available: %s", e)
            # Surface the failure to the user via their home channel(s) once
            # the gateway connects.  Without this, state.db corruption or
            # NFS/SMB lock failures silently degrade the entire gateway —
            # messages may flow but nothing is persisted, and the user has
            # no indication until they try /resume and find nothing (#88235).
            self._session_db_init_error = str(e)

        # Opportunistic state.db maintenance: prune ended sessions inactive
        # for sessions.retention_days + optional VACUUM. Tracks last-run
        # in state_meta so it only actually executes once per
        # sessions.min_interval_hours.  Gateway is long-lived so blocking
        # a few seconds once per day is acceptable; failures are logged
        # but never raised.
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                # Non-destructive stale-session archive, independent of prune.
                if _sess_cfg.get("auto_archive", False):
                    self._session_db._db.maybe_auto_archive(
                        idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                    )
                if _sess_cfg.get("auto_prune", False):
                    # Construction-time, before the loop serves traffic; sync DB is fine.
                    self._session_db._db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        min_vacuum_interval_days=int(
                            _sess_cfg.get("min_vacuum_interval_days", 30)
                        ),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir,
                    )
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Opportunistic shadow-repo cleanup — deletes stale checkpoint repos
        # under ~/.hermes/checkpoints/.  Opt-in via checkpoints.auto_prune,
        # idempotent via .last_prune marker.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                # delete_orphans is intentionally never honoured here: a
                # missing workdir at startup is ambiguous (deleted project
                # vs. an unmounted external volume / network share / VPN
                # not yet up) and this sweep runs unattended. Orphan cleanup
                # is only ever done via the explicit `hermes checkpoints
                # prune` command, which the user has to invoke.
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=False,
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)),
                )
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)

        # DM pairing store for code-based user authorization.
        # ``pairing_store`` stays as the global/default store for the
        # ``hermes pairing`` CLI and any caller without a profile context.
        # ``pairing_stores`` is the per-profile map used by
        # ``authz_mixin._is_user_authorized`` to route checks to the right
        # whitelist (one per profile in multiplex mode).
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        self.pairing_stores: Dict[str, "PairingStore"] = {}

        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

        # Track background tasks to prevent garbage collection mid-execution
        self._background_tasks: set = set()

        # Event-loop liveness heartbeat (#66892): rewritten every 30s while
        # the loop is dispatching. External supervisors use the file mtime /
        # updated_at to distinguish "process alive" from "loop frozen".
        self._gateway_started_at: float = time.time()
        self._loop_heartbeat_task: Optional[asyncio.Task] = None
        self._loop_floor_timer_handle = None
        self._loop_liveness_watchdog = None

































def _run_planned_stop_watcher(
    stop_event: threading.Event,
    runner,
    loop: asyncio.AbstractEventLoop,
    shutdown_handler,
    *,
    poll_interval: float = 0.5,
) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown.

    On Windows, ``asyncio.add_signal_handler`` raises NotImplementedError
    for SIGTERM/SIGINT, so the standard signal-driven shutdown path
    never runs when ``hermes gateway stop`` signals the gateway. The
    consequence is that the drain loop is skipped — in-flight agent
    sessions are killed mid-turn and ``resume_pending`` is never set,
    so the next gateway boot has no idea those sessions need to be
    auto-resumed (issue #33778, v0.13.0 session-resume feature broken
    on native Windows).

    This watcher runs on every platform (cheap, defensive) and bridges
    the gap on Windows by translating a filesystem marker into the
    same shutdown-handler invocation a real SIGTERM would have produced
    on POSIX. The CLI's ``hermes_cli.gateway_windows.stop()`` writes
    the marker via ``write_planned_stop_marker(pid)`` and then waits
    for the gateway PID to exit; this watcher is what makes that
    exit happen cleanly.

    On POSIX this is a no-op safety net — the signal handler always
    races us to consuming the marker file because it fires synchronously
    from the kernel's signal delivery.

    Args:
        stop_event: cleared by start_gateway() during normal shutdown
            to tell the watcher to exit.
        runner: the GatewayRunner instance; we check ``_running`` and
            ``_draining`` to avoid triggering shutdown if the gateway
            is already in one of those states.
        loop: the asyncio event loop the shutdown handler must run on.
        shutdown_handler: same callable that's wired to SIGTERM —
            tolerates a ``None`` signal argument (planned stop case)
            and consumes the marker via
            ``consume_planned_stop_marker_for_self()``.
        poll_interval: seconds between marker checks. 0.5s gives a
            responsive shutdown without burning CPU.
    """
    from gateway.status import (
        _get_planned_stop_marker_path,
        planned_stop_marker_targets_self,
    )
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)
            ):
                # A marker existing is NOT sufficient — it may have been
                # written for a PREVIOUS gateway instance (different PID)
                # and left behind because that process exited before the
                # CLI's stop() could clean it up. Firing the handler on a
                # stale/foreign marker drives the gateway into shutdown,
                # then consume_planned_stop_marker_for_self() correctly
                # reports a PID mismatch — but by then we're already
                # stopping, so it's logged as an unexpected "UNKNOWN" exit
                # and the watchdog crash-loops the gateway (issue #34597,
                # a regression from PR #33798 which added this watcher
                # without the PID check).
                #
                # Only fire when the marker actually targets us. The probe
                # is non-destructive on a match (the handler does the
                # authoritative consume on the loop thread) and self-heals
                # by unlinking stale/malformed markers so they cannot wedge
                # a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Drive the same path as a real signal handler.
                # Pass signal=None — the handler tolerates that and consumes
                # the marker via consume_planned_stop_marker_for_self,
                # which also validates target_pid + start_time match us.
                loop.call_soon_threadsafe(shutdown_handler, None)
                # Done — the handler will set _draining; we exit on next tick.
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)


def _start_gateway_housekeeping(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """Background thread for gateway-only periodic chores (NOT cron).

    Split out of the historical ``_start_cron_ticker`` so the cron *trigger*
    can live behind the ``CronScheduler`` provider (built-in or external) while
    these gateway-specific chores keep running independently of which provider
    fires cron. This housekeeping still wants its hourly cadence independently,
    so it owns its own loop.

    Refreshes the channel directory every 5 minutes and prunes the
    image/audio/video/document/screenshot caches + expired ``hermes debug
    share`` pastes once per hour, and polls the curator hourly (its inner
    gate enforces the real weekly cadence).
    """
    from gateway.platforms.base import (
        cleanup_audio_cache,
        cleanup_document_cache,
        cleanup_image_cache,
        cleanup_screenshot_cache,
        cleanup_video_cache,
    )
    from hermes_cli.debug import _sweep_expired_pastes

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes
    PASTE_SWEEP_EVERY = 60   # ticks — once per hour
    CURATOR_EVERY = 60       # ticks — poll hourly (inner gate handles the real cadence)
    AUTO_ARCHIVE_EVERY = 60  # ticks — poll hourly (state_meta gate owns the real cadence)
    MEMORY_TRIM_EVERY = 1    # shared helper cooldown bounds actual allocator work

    # Every platform media cache prunes on the same hourly cadence — one loop
    # over (name, cleanup_fn), not a copy-pasted try/except per cache.
    MEDIA_CACHE_CLEANUPS = (
        ("Image", cleanup_image_cache),
        ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache),
        ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache),
    )

    logger.info("Gateway housekeeping started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                if loop is not None:
                    # build_channel_directory is async (Slack web calls), and
                    # this runs in a background thread. Schedule onto the
                    # gateway event loop and wait briefly for completion so
                    # refresh failures are still logged via the except.
                    fut = safe_schedule_threadsafe(
                        build_channel_directory(adapters), loop,
                        logger=logger,
                        log_message="Channel directory refresh scheduling error",
                    )
                    if fut is not None:
                        fut.result(timeout=30)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            for cache_name, cleanup_fn in MEDIA_CACHE_CLEANUPS:
                try:
                    removed = cleanup_fn(max_age_hours=24)
                    if removed:
                        logger.info("%s cache cleanup: removed %d stale file(s)", cache_name, removed)
                except Exception as e:
                    logger.debug("%s cache cleanup error: %s", cache_name, e)

        if tick_count % PASTE_SWEEP_EVERY == 0:
            try:
                deleted, remaining = _sweep_expired_pastes()
                if deleted:
                    logger.info(
                        "Paste sweep: deleted %d expired paste(s), %d pending",
                        deleted, remaining,
                    )
            except Exception as e:
                logger.debug("Paste sweep error: %s", e)

        # Curator — piggy-back on the housekeeping loop so long-running
        # gateways get weekly skill maintenance without needing restarts.
        # maybe_run_curator() is internally gated by config.interval_hours
        # (7 days by default), so CURATOR_EVERY is just the poll rate — the
        # real work only fires once per config interval.
        if tick_count % CURATOR_EVERY == 0:
            try:
                from agent.curator import maybe_run_curator
                maybe_run_curator(
                    idle_for_seconds=float("inf"),
                    on_summary=lambda msg: logger.info("curator: %s", msg),
                )
            except Exception as e:
                logger.debug("Curator tick error: %s", e)

        # Stale-session auto-archive — a live timer, so gateways that stay up
        # for weeks keep sweeping on schedule (the startup hook fires once).
        # maybe_auto_archive() is gated by sessions.min_interval_hours in
        # state_meta; this is just the poll rate. Opens its own SessionDB —
        # SQLite connections are thread-bound and this runs off-loop.
        if tick_count % AUTO_ARCHIVE_EVERY == 0:
            try:
                from hermes_cli.config import load_config as _load_full_config
                from hermes_state import SessionDB
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_archive", False):
                    _adb = SessionDB()
                    try:
                        _adb.maybe_auto_archive(
                            idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                            min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        )
                    finally:
                        _adb.close()
            except Exception as e:
                logger.debug("Auto-archive tick error: %s", e)

        # This is the long-lived messaging-gateway counterpart to the TUI idle
        # reaper. The helper is config-gated and rate-limited, so calling it on
        # the 60s housekeeping cadence does not create a trim storm.
        if tick_count % MEMORY_TRIM_EVERY == 0:
            try:
                from hermes_cli.mem_trim import trim_memory

                trim_memory(reason="messaging gateway housekeeping")
            except Exception as exc:
                # debug, not warning: sibling housekeeping branches all log
                # failures at debug, and a persistent failure (e.g. broken
                # import after a partial update) would otherwise warn every
                # 60s forever.
                logger.debug(
                    "gateway housekeeping memory trim failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        stop_event.wait(timeout=interval)
    logger.info("Gateway housekeeping stopped")


def _stop_cron_provider(provider) -> None:
    """Stop a cron provider without letting it choose the gateway exit code."""
    try:
        provider.stop()
    except SystemExit as exc:
        logger.warning(
            "Cron provider stop() attempted to exit the gateway with code %s; ignoring",
            exc.code,
        )
    except Exception as exc:
        logger.debug("Cron provider stop() error: %s", exc)


# Upper bound for cooperatively draining the cron ticker on shutdown. The cron
# thread delivers via ``safe_schedule_threadsafe`` and blocks on
# ``future.result(timeout=60)`` (see cron/scheduler.py::_deliver_result), so a
# single in-flight delivery unblocks within ~60s. The extra margin covers the
# hop back through run_one_job's bookkeeping.
_CRON_SHUTDOWN_DRAIN_TIMEOUT = 65.0

# Upper bound for cooperatively draining the housekeeping ticker on shutdown.
# Housekeeping periodically refreshes the channel directory via
# ``safe_schedule_threadsafe(build_channel_directory(...), loop)`` and blocks on
# ``fut.result(timeout=30)`` (see ``_start_gateway_housekeeping``) — the same
# loop-scheduled-future pattern as cron. So the cooperative bound must cover
# that 30s future (plus margin) rather than the old 5s join, otherwise a
# channel-directory refresh in flight at shutdown gets abandoned mid-resolve.
# Unlike a dropped cron delivery this is not user-facing (it self-heals on the
# next tick), but bounding it correctly keeps the drain honest.
_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT = 35.0


async def _await_thread_exit(
    thread: Optional[threading.Thread], timeout: float, poll: float = 0.1
) -> bool:
    """Wait for a daemon thread to exit WITHOUT blocking the event loop.

    A synchronous ``thread.join()`` here would freeze the event loop — fatal
    for the cron ticker, whose in-flight delivery is a coroutine scheduled onto
    *this* loop via ``safe_schedule_threadsafe``. Blocking the loop deadlocks
    that delivery (the loop can never run it), so ``join(timeout=5)`` always
    times out and the message is silently dropped on restart (#58818).

    Polling ``is_alive()`` with ``await asyncio.sleep`` keeps the loop running
    so the pending delivery completes, then the ticker sees ``stop_event`` and
    exits. Returns True if the thread exited within ``timeout``.
    """
    if thread is None:
        return True
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
    while thread.is_alive() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(poll)
    return not thread.is_alive()




def _gateway_stderr_formatter() -> logging.Formatter:
    """Return the redacting formatter used by the gateway stderr stream."""
    from agent.redact import RedactingFormatter

    return RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """
    Start the gateway and run until interrupted.

    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.

    Args:
        config: Optional gateway configuration override.
        replace: If True, kill any existing gateway instance before starting.
                 Useful for systemd services to avoid restart-loop deadlocks
                 when the previous process hasn't fully exited yet.
    """
    # Enable interactive exec approval for dangerous commands on messaging
    # platforms. Set here (not at module import) so incidental imports of
    # gateway.run from CLI/tool code do not poison HERMES_EXEC_ASK.
    os.environ["HERMES_EXEC_ASK"] = "1"

    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    # Snapshot the checkout revision now, while sys.modules still matches disk,
    # so a later `git pull` under this long-lived process can be detected (and
    # risky work like model switching refused) instead of crashing on a stale
    # in-memory module.
    from gateway.code_skew import record_boot_fingerprint
    record_boot_fingerprint()

    # ── Duplicate-instance guard ──────────────────────────────────────
    # Prevent two gateways from running under the same HERMES_HOME.
    # The PID file is scoped to HERMES_HOME, so future multi-profile
    # setups (each profile using a distinct HERMES_HOME) will naturally
    # allow concurrent instances without tripping this guard.
    from gateway.status import (
        acquire_gateway_runtime_lock,
        get_running_pid,
        get_process_start_time,
        release_gateway_runtime_lock,
        remove_pid_file,
        terminate_pid,
    )
    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        if replace:
            existing_start_time = get_process_start_time(existing_pid)
            logger.info(
                "Replacing existing gateway instance (PID %d) with --replace.",
                existing_pid,
            )
            # Record a takeover marker so the target's shutdown handler
            # recognises its SIGTERM as a planned takeover and exits 0
            # (rather than exit 1, which would trigger systemd's
            # Restart=on-failure and start a flap loop against us).
            # Best-effort — proceed even if the write fails.
            try:
                from gateway.status import write_takeover_marker
                write_takeover_marker(existing_pid)
            except Exception as e:
                logger.debug("Could not write takeover marker: %s", e)
            # Snapshot the old gateway's child processes BEFORE signalling it:
            # once it exits, orphans are reparented and can no longer be found
            # by a parent walk. On POSIX, adapter subprocesses that outlive
            # the gateway keep holding scoped token locks and block the
            # replacement (Windows terminate_pid(force=True) already
            # tree-kills via taskkill /T). Best-effort — [] on any failure.
            try:
                from gateway.status import _snapshot_gateway_children
                _old_gateway_children = _snapshot_gateway_children(existing_pid)
            except Exception:
                _old_gateway_children = []
            try:
                terminate_pid(existing_pid, force=False)
            except ProcessLookupError:
                pass  # Already gone
            except (PermissionError, OSError):
                logger.error(
                    "Permission denied killing PID %d. Cannot replace.",
                    existing_pid,
                )
                # Marker is scoped to a specific target; clean it up on
                # give-up so it doesn't grief an unrelated future shutdown.
                try:
                    from gateway.status import clear_takeover_marker
                    clear_takeover_marker()
                except Exception:
                    pass
                return False
            # Wait up to 10 seconds for the old process to exit.
            # ``os.kill(pid, 0)`` on Windows is NOT a no-op — use the
            # handle-based existence check instead.
            from gateway.status import _pid_exists
            old_gateway_exited = False
            for _ in range(20):
                if not _pid_exists(existing_pid):
                    old_gateway_exited = True
                    break  # Process is gone
                time.sleep(0.5)
            else:
                # Still alive after 10s — force kill
                logger.warning(
                    "Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.",
                    existing_pid,
                )
                try:
                    terminate_pid(existing_pid, force=True)
                except ProcessLookupError:
                    old_gateway_exited = True
                except (PermissionError, OSError):
                    pass
                # Confirm the force-kill actually reaped the process before we
                # clear its PID file / scoped locks. SIGKILL can fail to take
                # (e.g. an uninterruptible-sleep or zombie-reaping parent), and
                # if we blindly clear the metadata and start a fresh instance
                # we end up with two live gateways fighting over the same
                # token — the duplicate-gateway failure in #19471.
                if not old_gateway_exited:
                    for _ in range(20):
                        if not _pid_exists(existing_pid):
                            old_gateway_exited = True
                            break
                        time.sleep(0.25)
                if not old_gateway_exited:
                    logger.error(
                        "Old gateway (PID %d) still appears alive after SIGKILL; "
                        "aborting replacement to avoid a duplicate gateway.",
                        existing_pid,
                    )
                    try:
                        from gateway.status import clear_takeover_marker
                        clear_takeover_marker()
                    except Exception:
                        pass
                    return False
            # Old gateway confirmed dead — reap any orphaned child processes
            # it left behind (POSIX; mirrors Windows taskkill /T tree-kill).
            # Orphaned adapter subprocesses would otherwise keep holding
            # scoped token locks against us. Best-effort, never raises.
            try:
                from gateway.status import reap_gateway_children
                reap_gateway_children(
                    _old_gateway_children, parent_pid=existing_pid
                )
            except Exception:
                logger.debug(
                    "Child reap for replaced gateway PID %d failed",
                    existing_pid,
                    exc_info=True,
                )
            remove_pid_file()
            # remove_pid_file() is a no-op when the PID doesn't match.
            # Force-unlink to cover the old-process-crashed case.
            try:
                (get_hermes_home() / "gateway.pid").unlink(missing_ok=True)
            except Exception:
                pass
            # Clean up any takeover marker the old process didn't consume
            # (e.g. SIGKILL'd before its shutdown handler could read it).
            try:
                from gateway.status import clear_takeover_marker
                clear_takeover_marker()
            except Exception:
                pass
            # Also release all scoped locks left by the old process.
            # Stopped (Ctrl+Z) processes don't release locks on exit,
            # leaving stale lock files that block the new gateway from starting.
            try:
                from gateway.status import release_all_scoped_locks
                _released = release_all_scoped_locks(
                    owner_pid=existing_pid,
                    owner_start_time=existing_start_time,
                )
                if _released:
                    logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
            except Exception:
                pass
        else:
            hermes_home = str(get_hermes_home())
            logger.error(
                "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
                "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
                existing_pid, hermes_home,
            )
            print(
                f"\n❌ Gateway already running (PID {existing_pid}).\n"
                f"   Use 'hermes gateway restart' to replace it,\n"
                f"   or 'hermes gateway stop' to kill it first.\n"
                f"   Or use 'hermes gateway run --replace' to auto-replace.\n"
            )
            return False

    # Sync bundled skills on gateway start (fast -- skips unchanged)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # Centralized logging — agent.log (INFO+), errors.log (WARNING+),
    # and gateway.log (INFO+, gateway-component records only).
    # Idempotent, so repeated calls from create_agent.__init__ won't duplicate.
    from hermes_logging import setup_logging, _safe_stderr
    setup_logging(hermes_home=_hermes_home, mode="gateway")

    # Startup security posture audit — warn-on-load, never blocks. Surfaces
    # root / weak-SSH / ephemeral-container / unauthenticated-listener posture
    # so operators get the "you're exposed" signal the June 2026 MCP-config
    # persistence campaign victims never had.
    try:
        from hermes_cli.security_audit_startup import log_startup_security_warnings

        _audit_cfg = None
        try:
            from hermes_cli.config import read_raw_config

            _audit_cfg = read_raw_config()
        except Exception:
            _audit_cfg = None
        log_startup_security_warnings(hermes_home=_hermes_home, config=_audit_cfg)
    except Exception as _audit_exc:
        logger.debug("Startup security audit failed (non-fatal): %s", _audit_exc)

    # Optional stderr handler — level driven by -v/-q flags on the CLI.
    # verbosity=None (-q/--quiet): no stderr output
    # verbosity=0    (default):    WARNING and above
    # verbosity=1    (-v):         INFO and above
    # verbosity=2+   (-vv/-vvv):   DEBUG
    if verbosity is not None:
        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler(_safe_stderr())
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(_gateway_stderr_formatter())
        logging.getLogger().addHandler(_stderr_handler)
        # Lower root logger level if needed so DEBUG records can reach the handler
        if _stderr_level < logging.getLogger().level:
            logging.getLogger().setLevel(_stderr_level)

    runner = GatewayRunner(config)
    # ``--replace`` is explicit startup authority, not a durable reconnect
    # policy. GatewayRunner scopes this bit to cold adapter connects and clears
    # it before the background reconnect watcher starts.
    runner._platform_lock_takeover_on_start = bool(replace)

    # Track whether an unexpected signal initiated the shutdown. When an
    # unexpected SIGTERM kills the gateway, we exit non-zero so service
    # managers can revive the process. Planned stop paths write a marker
    # before signalling us so they can exit cleanly instead.
    _signal_initiated_shutdown = False

    # Set up signal handlers
    def shutdown_signal_handler(received_signal=None):
        nonlocal _signal_initiated_shutdown
        # Planned --replace takeover check: when a sibling gateway is
        # taking over via --replace, it wrote a marker naming this PID
        # before sending SIGTERM. If present, treat the signal as a
        # planned shutdown and exit 0 so systemd's Restart=on-failure
        # doesn't revive us (which would flap-fight the replacer when
        # both services are enabled, e.g. hermes.service + hermes-
        # gateway.service from pre-rename installs).
        planned_takeover = False
        try:
            from gateway.status import consume_takeover_marker_for_self
            planned_takeover = consume_takeover_marker_for_self()
        except Exception as e:
            logger.debug("Takeover marker check failed: %s", e)

        # Planned stop check: service managers and `hermes gateway stop`
        # also send SIGTERM, which is indistinguishable from an unexpected
        # external kill unless the CLI marks it first. SIGINT comes from an
        # interactive Ctrl+C and is likewise an intentional foreground stop.
        planned_stop = False
        if received_signal == signal.SIGINT:
            planned_stop = True
        elif not planned_takeover:
            try:
                from gateway.status import consume_planned_stop_marker_for_self
                planned_stop = consume_planned_stop_marker_for_self()
            except Exception as e:
                logger.debug("Planned stop marker check failed: %s", e)

        # Fast (<10ms) snapshot of who's asking us to shut down — runs
        # synchronously inside the asyncio signal handler, so we keep it
        # purely stdlib + /proc reads, no subprocesses.  See PR #15826
        # (May 2026): the previous implementation called `ps aux` here
        # synchronously, blocking the event loop for up to 3s while
        # adapter teardown couldn't begin.
        try:
            from gateway.shutdown_forensics import (
                format_context_for_log,
                snapshot_shutdown_context,
                spawn_async_diagnostic,
            )
            _shutdown_ctx = snapshot_shutdown_context(received_signal)
        except Exception as _e:
            _shutdown_ctx = None
            logger.debug("snapshot_shutdown_context failed: %s", _e)

        if planned_takeover:
            logger.info(
                "Received %s as a planned --replace takeover — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM",
            )
        elif planned_stop:
            logger.info(
                "Received %s as a planned gateway stop — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT",
            )
        else:
            _signal_initiated_shutdown = True
            logger.info(
                "Received %s — initiating shutdown",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT",
            )

        # Always log who/what triggered the signal — most useful single
        # line when diagnosing "the gateway keeps dying" tickets.  Format
        # is one line, key=value, parent_cmdline last (often long).
        if _shutdown_ctx is not None:
            try:
                logger.warning(
                    "Shutdown context: %s", format_context_for_log(_shutdown_ctx)
                )
            except Exception as _e:
                logger.debug("format_context_for_log failed: %s", _e)

            # Spawn the heavyweight diagnostic (ps auxf, pstree, dmesg) in
            # a detached subprocess so it can finish writing to disk even
            # if our cgroup is being torn down.  Bounded by an internal
            # timeout; never blocks the event loop here.
            try:
                _diag_log = _hermes_home / "logs" / "gateway-shutdown-diag.log"
                spawn_async_diagnostic(
                    _diag_log, _shutdown_ctx["signal"], timeout_seconds=5.0
                )
            except Exception as _e:
                logger.debug("spawn_async_diagnostic failed: %s", _e)
        asyncio.create_task(runner.stop())

    def restart_signal_handler():
        runner.request_restart(detached=False, via_service=True)

    loop = asyncio.get_running_loop()

    # Install a loop-level exception handler that swallows transient
    # network errors from background tasks. Issues #31066 / #31110:
    # an unhandled ``telegram.error.TimedOut`` (or peer NetworkError /
    # httpx connection error) in any awaited coroutine would propagate
    # to the loop and kill the gateway process, taking down every
    # profile attached to the same runner. systemd then restarts the
    # service after ~5s but the active conversation turn is lost.
    #
    # The fix is intentionally narrow: only well-known transient
    # network errors are swallowed (and logged with full traceback so
    # the originating call site is still discoverable). Anything else
    # is forwarded to the default handler so real bugs still surface.
    loop.set_exception_handler(_gateway_loop_exception_handler)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_signal_handler, sig)  # windows-footgun: ok — wrapped in try/except NotImplementedError for Windows
            except NotImplementedError:
                pass
        if hasattr(signal, "SIGUSR1"):
            try:
                loop.add_signal_handler(signal.SIGUSR1, restart_signal_handler)  # windows-footgun: ok — POSIX signal, guarded by hasattr above + try/except NotImplementedError
            except NotImplementedError:
                pass
    else:
        logger.info("Skipping signal handlers (not running in main thread).")

    # Windows fallback: asyncio.add_signal_handler raises NotImplementedError
    # on Windows, so `hermes gateway stop`'s SIGTERM (which Python maps to
    # TerminateProcess on Windows) never invokes shutdown_signal_handler.
    # That means the drain loop never runs, mark_resume_pending never fires,
    # and sessions are silently lost across restarts (issue #33778).
    #
    # The fix is a marker-polling thread: `hermes gateway stop` writes the
    # planned-stop marker BEFORE killing, and this thread notices it and
    # drives the same shutdown path the signal handler would have.  Runs
    # on every platform (cheap, defensive) so non-signal-bearing
    # environments (Windows native, sandboxed CI runners that mask
    # SIGTERM) still get a clean drain.
    _planned_stop_watcher_stop = threading.Event()
    _planned_stop_watcher_thread = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(_planned_stop_watcher_stop, runner, loop, shutdown_signal_handler),
        daemon=True,
        name="planned-stop-watcher",
    )
    _planned_stop_watcher_thread.start()

    # Claim the PID file BEFORE bringing up any platform adapters.
    # This closes the --replace race window: two concurrent `gateway run
    # --replace` invocations both pass the termination-wait above, but
    # only the winner of the O_CREAT|O_EXCL race below will ever open
    # Telegram polling, Discord gateway sockets, etc. The loser exits
    # cleanly before touching any external service.
    import atexit
    from gateway.status import write_pid_file, remove_pid_file, get_running_pid
    _current_pid = get_running_pid()
    if _current_pid is not None and _current_pid != os.getpid():
        logger.error(
            "Another gateway instance (PID %d) started during our startup. "
            "Exiting to avoid double-running.", _current_pid
        )
        return False
    if not acquire_gateway_runtime_lock():
        logger.error(
            "Gateway runtime lock is already held by another instance. Exiting."
        )
        return False
    try:
        write_pid_file()
    except FileExistsError:
        release_gateway_runtime_lock()
        logger.error(
            "PID file race lost to another gateway instance. Exiting."
        )
        return False
    atexit.register(remove_pid_file)
    atexit.register(release_gateway_runtime_lock)

    # Lifecycle ledger (NS-608): report if the previous gateway life died
    # uncleanly (SIGKILL / OOM / VM death — no exit path ran), then claim
    # the sentinel for this life. Placed after the PID-file/lock claim so
    # only the authoritative gateway for this HERMES_HOME touches the
    # sentinel — a --replace loser exiting above must not clobber it.
    try:
        from gateway.lifecycle_ledger import record_startup as _lifecycle_record_startup
        _lifecycle_record_startup()
    except Exception as _lc_exc:
        logger.debug("Lifecycle ledger startup record failed: %s", _lc_exc)

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        logger.debug("Nous auth keepalive did not start: %s", exc)

    _ensure_windows_gateway_venv_imports()

    # MCP tool discovery — run in an executor so the asyncio event loop
    # stays responsive even when a configured MCP server is slow or
    # unreachable.  discover_mcp_tools() uses a blocking 120s wait
    # internally; calling it from the loop thread would freeze platform
    # heartbeats (Discord shard, Telegram polling) until it returned.
    # See #16856.
    try:
        from tools.mcp_tool import discover_mcp_tools
        _loop = asyncio.get_running_loop()
        await _loop.run_in_executor(None, discover_mcp_tools)
    except Exception as e:
        logger.debug("MCP tool discovery failed: %s", e)

    # Start the gateway
    try:
        success = await runner.start()
    except BaseException:
        _shutdown_gateway_health_export(runner)
        raise
    if not success:
        _shutdown_gateway_health_export(runner)
        return False
    # Recover any pending messages flushed during a previous shutdown (#72680).
    try:
        from gateway.shutdown_flush import recover_pending_to_db
        recovered = recover_pending_to_db()
        if recovered:
            logger.info(
                "Recovered %d pending message(s) from shutdown flush", recovered,
            )
    except Exception:
        pass
    if runner.should_exit_cleanly:
        _shutdown_gateway_health_export(runner)
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        # A clean exit that carries an explicit exit code (e.g. a fatal
        # config error stamped with GATEWAY_FATAL_CONFIG_EXIT_CODE) must
        # propagate that code to the process so the s6 finish script can
        # translate it (78 → 125) and stop the supervisor restart loop.
        # Without this, the early `return True` below makes main() exit 0,
        # the finish script's `[ "$1" = "78" ]` check never matches, and
        # s6 crash-loops the gateway anyway (#51228).
        if runner.exit_code is not None:
            raise SystemExit(runner.exit_code)
        return True
    if not runner._running:
        # Startup was intentionally aborted by restart/shutdown before entering
        # running mode; preserve that lifecycle path without starting cron.
        try:
            await runner.wait_for_shutdown()
            if runner.should_exit_with_failure:
                if runner.exit_reason:
                    logger.error("Gateway exiting with failure: %s", runner.exit_reason)
                return False
            try:
                from tools.mcp_tool import shutdown_mcp_servers
                shutdown_mcp_servers()
            except Exception:
                pass
            if runner.exit_code is not None:
                raise SystemExit(runner.exit_code)
            return True
        finally:
            _shutdown_gateway_health_export(runner)

    # Start the background cron scheduler via the resolved provider so
    # scheduled jobs fire automatically. The built-in provider is the
    # historical in-process 60s ticker; an external provider
    # may arm a schedule and return. Pass the event loop so cron delivery can
    # use live adapters (E2EE support).
    from cron.scheduler_provider import (
        InProcessCronScheduler,
        resolve_cron_scheduler,
    )
    cron_stop = threading.Event()
    cron_provider = resolve_cron_scheduler()
    cron_start_kwargs: Dict[str, Any] = {"adapters": runner.adapters, "loop": asyncio.get_running_loop()}

    if isinstance(cron_provider, InProcessCronScheduler):
        cron_start_kwargs["can_dispatch"] = lambda: not runner._draining
    cron_thread = threading.Thread(
        target=cron_provider.start,
        args=(cron_stop,),
        kwargs=cron_start_kwargs,
        daemon=True,
        name="cron-scheduler",
    )
    cron_thread.start()

    # Gateway-only periodic housekeeping (channel dir, cache cleanup, paste
    # sweep, curator) — runs independently of which cron provider is active.
    # Shares cron_stop as the shutdown signal.
    housekeeping_thread = threading.Thread(
        target=_start_gateway_housekeeping,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop()},
        daemon=True,
        name="gateway-housekeeping",
    )
    housekeeping_thread.start()

    # READY is emitted only after adapters, cron, and housekeeping have all
    # reached their running boundary. Missing config/systemd runtime state
    # leaves the watchdog disabled without changing gateway behavior.
    start_watchdog = getattr(runner, "_start_systemd_watchdog", None)
    if callable(start_watchdog):
        start_watchdog()

    # Wait for shutdown
    await runner.wait_for_shutdown()

    try:
        from hermes_cli.nous_auth_keepalive import stop_nous_auth_keepalive

        stop_nous_auth_keepalive()
    except Exception:
        pass

    if runner.should_exit_with_failure:
        if runner.exit_reason:
            logger.error("Gateway exiting with failure: %s", runner.exit_reason)
        return False

    # Stop cron scheduler + housekeeping cleanly.
    #
    # These MUST be awaited cooperatively, not join()ed. A cron delivery in
    # flight when the gateway restarts is a coroutine scheduled onto THIS event
    # loop (safe_schedule_threadsafe); the ticker thread is blocked on its
    # future.result(). A synchronous cron_thread.join() would block the loop,
    # so that delivery could never run — it timed out and the message was
    # silently dropped (#58818). Awaiting keeps the loop alive so the in-flight
    # delivery finishes before we tear down.
    cron_stop.set()
    _stop_cron_provider(cron_provider)
    if not await _await_thread_exit(cron_thread, timeout=_CRON_SHUTDOWN_DRAIN_TIMEOUT):
        logger.warning(
            "Cron ticker did not exit within %.0fs of shutdown — an in-flight "
            "delivery may have been dropped.", _CRON_SHUTDOWN_DRAIN_TIMEOUT,
        )
    await _await_thread_exit(
        housekeeping_thread, timeout=_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT
    )

    # Stop the planned-stop watcher (daemon=True so this is belt-and-suspenders).
    _planned_stop_watcher_stop.set()
    _planned_stop_watcher_thread.join(timeout=2)

    # Close MCP server connections
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass

    if runner.exit_code is not None:
        raise SystemExit(runner.exit_code)

    # When an unexpected SIGTERM caused the shutdown and it wasn't a planned
    # restart (/restart, /update, SIGUSR1), exit non-zero so systemd's
    # Restart=on-failure revives the process.  This covers:
    #   - hermes update killing the gateway mid-work
    #   - External kill commands
    #   - WSL2/container runtime sending unexpected signals
    # `hermes gateway stop` and interactive Ctrl+C are handled above as
    # planned stops and should not trigger service-manager revival.
    if _signal_initiated_shutdown and not runner._restart_requested:
        logger.info(
            "Exiting with code 1 (signal-initiated shutdown without restart "
            "request) so systemd Restart=on-failure can revive the gateway."
        )
        return False  # → sys.exit(1) in the caller

    # Older restart paths may reach here without ``runner.exit_code`` set.
    # Keep the historical non-zero fallback for service-managed restarts.
    if runner._restart_via_service:
        logger.info(
            "Exiting with code 75 (service-restart requested) so the service "
            "manager relaunches the gateway."
        )
        raise SystemExit(75)

    return True


def main():
    """CLI entry point for the gateway."""
    # Advertise the agent harness to child processes (AI_AGENT is the
    # cross-agent standard; HERMES_AGENT the Hermes-specific marker — see
    # _advertise_agent_env in hermes_cli/main.py, kept inline here to avoid
    # importing that module's startup side effects). The value must equal our
    # public agent-harness registry id (``hermes-agent``) — standard-var
    # matching is exact. setdefault so an outer harness is never clobbered.
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")

    # Force UTF-8 stdio on Windows — gateway logs and startup banner would
    # otherwise UnicodeEncodeError on cp1252 consoles.  No-op on POSIX.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    import argparse

    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)

    # start_gateway() performs the full graceful teardown (adapters
    # disconnected, sessions saved + flushed, SQLite closed, cron/MCP stopped,
    # PID file + runtime lock released) before it returns OR raises SystemExit
    # with an explicit code. Force-exit afterwards so a wedged non-daemon worker
    # thread (e.g. a ThreadPoolExecutor tool/LLM call blocked with no timeout)
    # cannot block interpreter finalization (Py_FinalizeEx joins all non-daemon
    # threads, incl. concurrent.futures' _python_exit) and strand the gateway
    # half-shut down with the supervisor unable to restart it (#53107).
    #
    # SystemExit is caught explicitly: start_gateway raises it on the
    # clean-fatal-config (#51228), planned-restart, and service-restart paths,
    # all of which complete teardown first. Routing those codes through the
    # same os._exit backstop means EVERY exit path is wedge-proof, not just the
    # boolean-return ones.
    try:
        success = asyncio.run(start_gateway(config))
        exit_code = 0 if success else 1
    except SystemExit as e:
        # e.code may be None (→ 0), an int, or a str (→ 1, like CPython).
        if e.code is None:
            exit_code = 0
        elif isinstance(e.code, int):
            exit_code = e.code
        else:
            exit_code = 1
    _exit_after_graceful_shutdown(exit_code)


def _exit_after_graceful_shutdown(exit_code: int) -> None:
    """Flush stdio, release the PID file + runtime lock, then hard-exit.

    Graceful teardown is already complete by the time this runs, so there is
    nothing left that needs a clean interpreter shutdown. We deliberately use
    ``os._exit`` (not ``sys.exit``): ``sys.exit`` raises ``SystemExit``, which
    triggers ``Py_FinalizeEx`` → ``wait_for_thread_shutdown`` and joins every
    non-daemon thread — exactly the hang (#53107) a wedged tool-worker causes.

    ``os._exit`` bypasses ``atexit`` handlers, so we cannot rely on the
    ``atexit``-registered ``remove_pid_file`` / ``release_gateway_runtime_lock``
    (registered in ``start_gateway``) to run. The full-shutdown path releases
    both explicitly in ``_stop_impl``, but the EARLY exit paths —
    clean-fatal-config (#51228) and startup-aborted-before-running — raise
    ``SystemExit`` right after ``runner.start()`` without going through
    ``_stop_impl``, so on those paths ``atexit`` was the only thing releasing
    them. Now that those paths are routed through this backstop (#53107),
    release both here explicitly. Both calls are idempotent —
    ``remove_pid_file`` only unlinks a PID file that belongs to this process,
    and ``release_gateway_runtime_lock`` no-ops when the lock is already
    released — so this is a no-op on the normal shutdown path and the actual
    cleanup on the early-exit paths.

    Logging IS drained here: the rotating file handlers are driven by an
    async ``QueueListener`` on a dedicated thread (see
    ``hermes_logging._register_queued_handler``), so records emitted right
    before shutdown may still be sitting in the in-memory queue. ``os._exit``
    below bypasses ``atexit``, so the ``atexit``-registered listener drain
    never runs on this path — we drain explicitly (bounded, via
    ``drain_log_queue``) or lose the last log lines (including the shutdown
    reason on the early-exit paths). Stdio is flushed too.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    # Release PID + runtime lock BEFORE the log drain: the drain is bounded but
    # could still take up to its timeout on a wedged disk, and these locks must
    # never be stranded. os._exit skips atexit, and the early SystemExit exit
    # paths never run _stop_impl, so release here (idempotent).
    try:
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()
    except Exception:
        pass
    # Mark this life cleanly exited in the lifecycle sentinel (NS-608). This
    # is the single funnel every graceful exit passes through, so the next
    # boot's unclean-death detector only fires for genuine SIGKILL/OOM/VM
    # deaths. Ownership-guarded internally: a --replace old life won't
    # clobber the replacement's freshly claimed "running" sentinel.
    try:
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason="graceful_shutdown")
    except Exception:
        pass
    # Drain the async log queue: os._exit bypasses atexit, so the listener's
    # atexit drain won't fire. Use drain_log_queue() (bounded, no restart), NOT
    # flush_log_queue(): if the listener is wedged on the rotation lock — the
    # exact failure this async-logging change survives — an unbounded stop()
    # join would re-freeze the shutdown. drain_log_queue() no-ops when logging
    # never initialized a queue (very early aborts), so this is always safe.
    try:
        from hermes_logging import drain_log_queue
        drain_log_queue(timeout=1.0)
    except Exception:
        pass
    os._exit(exit_code)


if __name__ == "__main__":
    main()
