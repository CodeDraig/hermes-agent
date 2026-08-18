"""Configuration, memory, compression, and context-engine initialization."""

from __future__ import annotations



import copy
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from agent.context_compressor import ContextCompressor
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, is_local_endpoint, query_ollama_num_ctx
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home
from utils import is_truthy_value

from agent.init_runtime import (
    _context_route_mismatch,
    _custom_provider_runtime_ids,
    _merge_custom_provider_extra_body,
    _normalize_custom_provider_name,
    _normalize_route_base_url,
)

logger = logging.getLogger("run_agent")


# Memory providers we've already warned are unavailable. Deduped because the
# gateway builds a fresh create_agent per message, so an un-deduped warning would
# fire on every turn.
_warned_unavailable_providers: set[str] = set()


def _warn_memory_provider_unavailable(name: str, reason: str = "") -> None:
    """Warn (once per provider) when a configured memory provider is unavailable.

    ``is_available()`` is a fast, side-effect-free hot-path check, so it can't
    log for itself. Without this warning a provider whose credentials/config are
    missing is silently dropped — the user has ``memory.provider`` set but gets
    no memory and no diagnostic. A common trigger is systemd/gateway services
    not inheriting ``~/.hermes/.env``. See NousResearch/hermes-agent#2765.

    ``reason`` is the provider's ``unavailable_reason()`` — a provider-specific,
    actionable hint (e.g. which package to install). Because an unavailable
    provider is never initialized, this is the only place such a hint can reach
    the user, so it is appended to the warning when present (#7718).
    """
    if name in _warned_unavailable_providers:
        return
    _warned_unavailable_providers.add(name)
    logger.warning(
        "Memory provider %r is selected but reports unavailable — external memory "
        "is disabled for this session (built-in memory still works). Check the "
        "provider's credentials/config with 'hermes memory status'. Note: "
        "systemd/gateway services do not inherit ~/.hermes/.env automatically; set "
        "any required variables in the service environment.%s",
        name,
        f" {reason}" if reason else "",
    )

def _build_codex_gpt5_autoraise_notice(
    autoraise: Dict[str, Any], context_length: Optional[int] = None
) -> str:
    """Build the one-time notice shown when Codex gpt-5.x raises compaction.

    ``autoraise`` is ``{"model": <slug>, "from": <old_ratio>, "to": <new_ratio>}``.
    ``context_length`` is the live-resolved window from the context compressor
    (Codex's /models catalog is authoritative and can change server-side, e.g.
    the gpt-5.6 family's 272K → 372K → 272K shifts in July 2026), so the banner
    reports what this session actually got rather than a hardcoded cap. The
    same text is printed inline for CLI users and replayed via
    ``status_callback`` for gateway users, so it must be self-contained and
    include the exact opt-back-out command.
    """
    model = str(autoraise.get("model") or "gpt-5.4/5.5").strip().lower().rsplit("/", 1)[-1]
    if isinstance(context_length, int) and context_length > 0:
        cap = f"{round(context_length / 1000)}K"
    else:
        # Static fallback when the resolved window isn't available:
        # gpt-5.3-codex-spark has a native 128K window; the gpt-5.4/5.5/5.6
        # family is capped at 272K by the Codex OAuth backend.
        cap = "128K" if model.startswith("gpt-5.3-codex-spark") else "272K"
    from_pct = int(round(autoraise["from"] * 100))
    to_pct = int(round(autoraise["to"] * 100))
    return (
        f"ℹ Codex {model} caps context at {cap}, so auto-compaction was raised "
        f"to {to_pct}% (from {from_pct}%) to use more of the window before "
        f"summarizing.\n"
        f"  Opt back out: hermes config set compression.codex_gpt55_autoraise false"
    )

def _resolve_compression_threshold(
    global_threshold: float,
    model_cthresh: Optional[float],
    *,
    model: Optional[str] = None,
    is_codex_autoraise: bool,
) -> tuple[float, Optional[Dict[str, Any]]]:
    """Combine the user's global compaction threshold with a per-model override.

    Returns ``(effective_threshold, autoraise_notice)``. ``autoraise_notice`` is
    ``{"model": <slug>, "from": <old>, "to": <new>}`` only when a Codex
    autoraise (gpt-5.4/5.5 272K family or gpt-5.3-codex-spark) actually raises
    the threshold, otherwise ``None``.

    The Codex overrides are *autoraises*: they must never LOWER a higher
    user-configured threshold. A user who already set ``compression.threshold``
    above the raised value deliberately keeps more raw context, and silently
    dropping them would both waste usable window and contradict the feature's
    purpose (use more of the window). Other overrides (e.g. Arcee Trinity)
    keep their existing unconditional behaviour.
    """
    if model_cthresh is None:
        return global_threshold, None
    if is_codex_autoraise:
        if model_cthresh <= global_threshold + 1e-9:
            # Autoraise never lowers; keep the user's higher/equal threshold.
            return global_threshold, None
        return model_cthresh, {
            "model": model,
            "from": global_threshold,
            "to": model_cthresh,
        }
    return model_cthresh, None

def _codex_gpt55_autoraise_notice_marker():
    """Path to the per-profile marker recording that the autoraise notice ran.

    Lives under ``$HERMES_HOME`` (which is profile-scoped) alongside the other
    internal markers like ``.container-mode`` — so it is not a user-facing config
    key, and every profile tracks its own notice state independently.
    """
    return get_hermes_home() / ".codex_gpt55_autoraise_notice"

def _codex_gpt55_autoraise_notice_state(autoraise: Dict[str, Any]) -> str:
    """Stable identity for one autoraise notice, keyed on what it displays.

    Uses the model slug plus the same from→to percentages the notice text
    shows, so an unchanged threshold stays silent across restarts while a
    later change (the user edits their global ``threshold``, or switches to a
    different autoraised Codex model) re-notifies once.
    """
    model = str(autoraise.get("model") or "").strip().lower().rsplit("/", 1)[-1]
    from_pct = int(round(float(autoraise["from"]) * 100))
    to_pct = int(round(float(autoraise["to"]) * 100))
    return f"{model}:{from_pct}:{to_pct}"

def _codex_gpt55_autoraise_notice_seen(autoraise: Dict[str, Any]) -> bool:
    """True if this exact autoraise notice was already shown for this profile.

    A missing/unreadable marker (or one recording a different threshold) reads
    as unseen, so the notice shows.
    """
    try:
        current = _codex_gpt55_autoraise_notice_state(autoraise)
        return _codex_gpt55_autoraise_notice_marker().read_text(
            encoding="utf-8"
        ).strip() == current
    except (OSError, KeyError, TypeError, ValueError):
        return False

def _record_codex_gpt55_autoraise_notice(autoraise: Dict[str, Any]) -> None:
    """Persist that the autoraise notice was shown for this profile/config state.

    Best-effort: a read-only or missing ``$HERMES_HOME`` just means the notice
    may show again next init, which is preferable to breaking agent init.
    """
    try:
        marker = _codex_gpt55_autoraise_notice_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            _codex_gpt55_autoraise_notice_state(autoraise), encoding="utf-8"
        )
    except (OSError, KeyError, TypeError, ValueError):
        pass


def initialize_context(
    agent,
    *,
    config,
    base_url,
    platform,
    skip_memory,
    session_db,
):
    import agent.provider_runtime as provider_runtime
    import agent.status_output as status_output
    _agent_cfg = config
    # Codex commentary visibility (display.show_commentary, default true).
    # When true, completed Codex phase=commentary messages are delivered as
    # visible mid-turn updates through the interim message path. When false,
    # commentary falls back to the reasoning channel (visible only with
    # show_reasoning enabled).
    agent.show_commentary = True
    try:
        _display_section = _agent_cfg.get("display", {})
        if isinstance(_display_section, dict):
            agent.show_commentary = bool(_display_section.get("show_commentary", True))
    except Exception:
        agent.show_commentary = True

    # LM Studio can either be explicitly preloaded through LM Studio's
    # management API (the historical Hermes behavior) or left to LM Studio's
    # just-in-time / Auto-Evict chat-completions path.  Keep the default
    # explicit for backward compatibility; users with LM Studio Auto-Evict can
    # opt into JIT via ``model.lmstudio_load_mode: jit``.
    agent.lmstudio_load_mode = "explicit"
    try:
        _model_section = _agent_cfg.get("model", {})
        if isinstance(_model_section, dict):
            _load_mode = str(_model_section.get("lmstudio_load_mode", "explicit") or "explicit").strip().lower()
            if _load_mode in {"explicit", "jit"}:
                agent.lmstudio_load_mode = _load_mode
            else:
                logger.warning(
                    "Invalid model.lmstudio_load_mode=%r; expected 'explicit' or 'jit'. Using explicit.",
                    _model_section.get("lmstudio_load_mode"),
                )
    except Exception:
        agent.lmstudio_load_mode = "explicit"

    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {})
            )
        )
    except Exception as _tlg_err:
        logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)
    # Cache only the derived auxiliary compression context override that is
    # needed later by the startup feasibility check.  Avoid exposing a
    # broad pseudo-public config object on the agent instance.
    agent._aux_compression_context_length_config = None

    # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    # A flush/background agent may pass skip_memory=True to avoid spinning up an
    # external memory *provider*, but if the caller also explicitly enables the
    # "memory" toolset it still needs the built-in file-backed store — otherwise
    # the memory tool dispatches with store=None and every call fails (#65429).
    # So the built-in store is created unless memory is globally disabled, while
    # the external-provider block below stays gated on skip_memory.
    _memory_toolset_requested = "memory" in (agent.enabled_toolsets or [])
    if not skip_memory or _memory_toolset_requested:
        try:
            mem_config = _agent_cfg.get("memory", {})
            agent._memory_enabled = mem_config.get("memory_enabled", False)
            agent._user_profile_enabled = mem_config.get("user_profile_enabled", False)
            agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
            if agent._memory_enabled or agent._user_profile_enabled:
                from tools.memory_tool import MemoryStore
                agent._memory_store = MemoryStore(
                    memory_char_limit=mem_config.get("memory_char_limit", 2200),
                    user_char_limit=mem_config.get("user_char_limit", 1375),
                )
                agent._memory_store.load_from_disk()
        except Exception:
            pass  # Memory is optional -- don't break agent init



    # Memory provider plugin (external — one at a time, alongside built-in)
    # Reads memory.provider from config to select which plugin to activate.
    agent._memory_manager = None
    if not skip_memory:
        try:
            _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

            if _mem_provider_name and _mem_provider_name.strip():
                from agent.memory_manager import MemoryManager as _MemoryManager
                from plugins.memory import load_memory_provider as _load_mem
                agent._memory_manager = _MemoryManager()
                _mp = _load_mem(_mem_provider_name)
                if _mp and _mp.is_available():
                    agent._memory_manager.add_provider(_mp)
                elif _mp is not None:
                    # Skip the (potentially expensive) unavailable_reason() call
                    # if we've already warned for this provider — the gateway
                    # builds a fresh create_agent per message, so without this guard
                    # unavailable_reason() (which reads config from disk and may
                    # probe importlib) runs on every turn.
                    if _mem_provider_name not in _warned_unavailable_providers:
                        try:
                            _unavailable_reason = _mp.unavailable_reason()
                        except Exception:
                            _unavailable_reason = ""
                        _warn_memory_provider_unavailable(_mem_provider_name, _unavailable_reason)
                if agent._memory_manager.providers:
                    _init_kwargs = {
                        "session_id": agent.session_id,
                        "platform": platform or "cli",
                        "hermes_home": str(get_hermes_home()),
                        "agent_context": "primary",
                    }
                    if _init_kwargs["platform"] == "cli":
                        _init_kwargs["warning_callback"] = (
                            lambda message: status_output._emit_warning(agent, message)
                        )
                        _init_kwargs["status_callback"] = (
                            lambda message: status_output._emit_status(agent, message)
                        )
                    # Thread session title for memory provider scoping
                    # (e.g. honcho uses this to derive chat-scoped session keys)
                    if agent._session_db:
                        try:
                            _st = agent._session_db.get_session_title(agent.session_id)
                            if _st:
                                _init_kwargs["session_title"] = _st
                        except Exception:
                            pass
                    # Thread gateway user identity for per-user memory scoping
                    if agent._user_id:
                        _init_kwargs["user_id"] = agent._user_id
                    if agent._user_id_alt:
                        _init_kwargs["user_id_alt"] = agent._user_id_alt
                    if agent._user_name:
                        _init_kwargs["user_name"] = agent._user_name
                    if agent._chat_id:
                        _init_kwargs["chat_id"] = agent._chat_id
                    if agent._chat_name:
                        _init_kwargs["chat_name"] = agent._chat_name
                    if agent._chat_type:
                        _init_kwargs["chat_type"] = agent._chat_type
                    if agent._thread_id:
                        _init_kwargs["thread_id"] = agent._thread_id
                    # Thread gateway session key for stable per-chat Honcho session isolation
                    if agent._gateway_session_key:
                        _init_kwargs["gateway_session_key"] = agent._gateway_session_key
                    # Profile identity for per-profile provider scoping
                    try:
                        from hermes_cli.profiles import get_active_profile_name
                        _profile = get_active_profile_name()
                        _init_kwargs["agent_identity"] = _profile
                        _init_kwargs["agent_workspace"] = "hermes"
                    except Exception:
                        pass
                    # NOTE: status_callback (for the deterministic retain
                    # indicator) is wired above, CLI-only — gateway status is
                    # delivered on a different path (see the platform=="cli"
                    # block), and the indicator no-ops when it's absent.
                    agent._memory_manager.initialize_all(**_init_kwargs)
                    logger.info("Memory provider '%s' activated", _mem_provider_name)
                else:
                    logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                    agent._memory_manager = None
        except Exception as _mpe:
            logger.warning("Memory provider plugin init failed: %s", _mpe)
            agent._memory_manager = None

    from agent.memory_manager import inject_memory_provider_tools as _inject_memory_provider_tools
    _inject_memory_provider_tools(agent)

    # Skills config: nudge interval for skill creation reminders
    agent._skill_nudge_interval = 10
    try:
        skills_config = _agent_cfg.get("skills", {})
        agent._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
    except Exception:
        pass

    # Tool-use enforcement config: "auto" (default — matches hardcoded
    # model list), true (always), false (never), or list of substrings.
    _agent_section = _agent_cfg.get("agent", {})
    if not isinstance(_agent_section, dict):
        _agent_section = {}
    agent._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

    # Empty-response retry guard config (NS-503): additive
    # ``agent.empty_response_guard`` subsection. Resolution is tolerant —
    # a malformed section falls back to the schema defaults (guard on,
    # $0.25 threshold), matching the guard's overall fail-open posture.
    from agent.empty_response_guard import resolve_guard_settings
    (
        agent._empty_guard_enabled,
        agent._empty_guard_cost_threshold_usd,
    ) = resolve_guard_settings(_agent_section.get("empty_response_guard"))

    # Intent-ack continuation config: "auto" (default — codex_responses only,
    # the historical gate), true (all api_modes), false (never), or a list of
    # model-name substrings.  Resolved against the active api_mode/model in the
    # conversation loop's intent-ack block.
    agent._intent_ack_continuation = _agent_section.get("intent_ack_continuation", "auto")

    # Universal task-completion guidance toggle.  Default True.  Surfaced
    # as a separate flag from tool_use_enforcement because the guidance
    # applies to ALL models, not just the model families enforcement
    # targets.
    agent._task_completion_guidance = bool(_agent_section.get("task_completion_guidance", True))

    # Universal parallel-tool-call guidance toggle.  Default True.  Separate
    # flag from task_completion_guidance because a user may want one but not
    # the other.  Steers the model to batch independent tool calls into a
    # single turn; the runtime already executes such batches concurrently.
    agent._parallel_tool_call_guidance = bool(_agent_section.get("parallel_tool_call_guidance", True))

    # Local Python toolchain probe toggle.  Default True.  When False,
    # the probe is skipped entirely (no subprocess calls, no system-prompt
    # line).  Useful for users on exotic setups where the probe heuristics
    # are noisy.
    agent._environment_probe = bool(_agent_section.get("environment_probe", True))
    # Warm the probe off-thread: it shells out to python3/pip (~0.5s of
    # subprocess round-trips) and its result lands in the FIRST system
    # prompt build, which sits on the time-to-first-token critical path.
    # The warm runs during agent init (network/credential setup dominates),
    # so by the time the first prompt is built the line is already cached.
    if agent._environment_probe:
        try:
            from tools.env_probe import warm_environment_probe_async
            warm_environment_probe_async()
        except Exception:
            pass

    # Per-platform prompt-hint overrides (config.yaml → platform_hints).
    # Lets an enterprise admin append to or replace Hermes' built-in
    # platform hint for a single messaging platform (e.g. WhatsApp) without
    # affecting other platforms. Shape:
    #   platform_hints:
    #     whatsapp:
    #       append: "When tabular output would help, invoke the ... skill."
    #     slack:
    #       replace: "Custom Slack hint that fully replaces the default."
    # Stored verbatim; resolution happens in agent/system_prompt.py against
    # the active platform. Invalid shapes are ignored defensively so a bad
    # config entry can never break prompt assembly.
    _platform_hints_cfg = _agent_cfg.get("platform_hints", {})
    if not isinstance(_platform_hints_cfg, dict):
        _platform_hints_cfg = {}
    agent._platform_hint_overrides = _platform_hints_cfg

    # App-level API retry count (wraps each model API call).  Default 3,
    # overridable via agent.api_max_retries in config.yaml.  See #11616.
    try:
        _raw_api_retries = _agent_section.get("api_max_retries", 3)
        _api_retries = int(_raw_api_retries)
        _api_retries = max(_api_retries, 1)  # 1 = no retry (single attempt)
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries

    # Initialize context compressor for automatic context management
    # Compresses conversation when approaching model's context limit
    # Configuration via config.yaml (compression section)
    _compression_cfg = _agent_cfg.get("compression", {})
    if not isinstance(_compression_cfg, dict):
        _compression_cfg = {}
    compression_threshold = float(_compression_cfg.get("threshold", 0.50))
    # Per-model/route compaction-threshold override. Codex gpt-5.4 / gpt-5.5
    # raise to 85% (the Codex backend caps both families at 272K, so the
    # default 50% would compact at ~136K — half the usable context). Gated by
    # an opt-out config flag so the user can fall back to the global threshold;
    # when the override fires we stash a one-time notification (replayed on the
    # first turn) that tells the user what changed and how to revert. The
    # notice has its own display gate so users can keep the threshold
    # autoraise without getting the banner on gateway turns.
    _codex_gpt55_autoraise = str(
        _compression_cfg.get("codex_gpt55_autoraise", True)
    ).lower() in {"true", "1", "yes"}
    _codex_gpt55_autoraise_notice = str(
        _compression_cfg.get("codex_gpt55_autoraise_notice", True)
    ).lower() in {"true", "1", "yes"}
    agent._compression_threshold_autoraised = None
    try:
        from agent.auxiliary_client import (
            _compression_threshold_for_model as _cthresh_fn,
            _is_codex_gpt54_or_gpt55 as _is_codex_gpt54_or_gpt55_fn,
            _is_codex_spark as _is_codex_spark_fn,
        )
        _model_cthresh = _cthresh_fn(
            agent.model,
            agent.provider,
            allow_codex_gpt55_autoraise=_codex_gpt55_autoraise,
        )
        # The Codex autoraises (gpt-5.4/5.5 272K family and gpt-5.3-codex-spark)
        # apply only when they RAISE (never lower a user's higher global
        # threshold). The notice is populated only when it actually fires, and
        # carries the model slug so the banner names the right family. Arcee
        # Trinity keeps its long-standing unconditional behaviour.
        compression_threshold, agent._compression_threshold_autoraised = (
            _resolve_compression_threshold(
                compression_threshold,
                _model_cthresh,
                model=agent.model,
                is_codex_autoraise=(
                    _is_codex_gpt54_or_gpt55_fn(agent.model, agent.provider)
                    or _is_codex_spark_fn(agent.model, agent.provider)
                ),
            )
        )
    except Exception:
        pass
    compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}
    compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
    compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))
    # Tail retention mode (compression.tail_mode). "legacy" (default) keeps
    # the 0.20*window verbatim tail; "lean" switches to the clamped
    # 2.5%/10K-25K tail with recovery-pointer machinery (#87326). Unknown
    # values fall back to legacy inside the compressor.
    compression_tail_mode = str(_compression_cfg.get("tail_mode", "legacy")).strip().lower()
    # Minimum REAL (actionable) user messages guaranteed to survive in the
    # uncompressed tail (compression.min_tail_user_messages).  Default 1
    # preserves current behavior exactly — the existing single-user tail
    # anchor.  Values > 1 extend the guarantee to the last N actionable
    # user turns.  Booleans rejected (bool subclasses int), non-int-like
    # values fall back to 1, floor at 1.
    _raw_min_tail_users = _compression_cfg.get("min_tail_user_messages", 1)
    if isinstance(_raw_min_tail_users, bool):
        compression_min_tail_users = 1
    elif isinstance(_raw_min_tail_users, int):
        compression_min_tail_users = _raw_min_tail_users
    elif isinstance(_raw_min_tail_users, float):
        compression_min_tail_users = (
            int(_raw_min_tail_users) if _raw_min_tail_users.is_integer() else 1
        )
    else:
        try:
            compression_min_tail_users = int(str(_raw_min_tail_users).strip())
        except (TypeError, ValueError):
            compression_min_tail_users = 1
    if compression_min_tail_users < 1:
        compression_min_tail_users = 1
    # Cap on compression retry rounds before a turn gives up with "max
    # compression attempts reached" (compression.max_attempts).  Hardcoding 3
    # strands sessions that legitimately need more rounds — e.g. a restart
    # history reload whose incompressible tool schemas keep the request
    # estimate above the threshold even though the messages compress fine
    # (the #62605 failure class).  Default 3 preserves current behavior, so
    # an unset key is behavior-neutral; validated >= 1, hard-capped at 10,
    # and any non-int-like value falls back to 3.  Booleans are rejected
    # (bool subclasses int, so int(True) would silently become 1) and
    # fractional floats are rejected rather than truncated — "4.7 attempts"
    # is a config mistake, not a request for 4.
    _raw_max_attempts = _compression_cfg.get("max_attempts", 3)
    if isinstance(_raw_max_attempts, bool):
        compression_max_attempts = 3
    elif isinstance(_raw_max_attempts, int):
        compression_max_attempts = _raw_max_attempts
    elif isinstance(_raw_max_attempts, float):
        compression_max_attempts = (
            int(_raw_max_attempts) if _raw_max_attempts.is_integer() else 3
        )
    else:
        try:
            compression_max_attempts = int(str(_raw_max_attempts).strip())
        except (TypeError, ValueError):
            compression_max_attempts = 3
    if compression_max_attempts < 1:
        compression_max_attempts = 3
    compression_max_attempts = min(compression_max_attempts, 10)

    def _parse_prune_int(raw, default):
        # Same parser semantics as compression.max_attempts above: reject
        # booleans (bool subclasses int — YAML `true` would coerce to 1),
        # reject fractional floats rather than truncating them, accept
        # integral floats and numeric strings, fall back to the default on
        # anything else.
        if isinstance(raw, bool):
            return default
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw) if raw.is_integer() else default
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return default

    # Opt-in proactive tool-result prune trigger (0 = disabled — the
    # default, so an unset key is behavior-neutral).  Negative values are
    # treated as disabled rather than erroring.
    compression_proactive_prune_tokens = max(
        0, _parse_prune_int(_compression_cfg.get("proactive_prune_tokens", 0), 0)
    )
    compression_proactive_prune_min_chars = _parse_prune_int(
        _compression_cfg.get("proactive_prune_min_result_chars", 8000), 8000
    )
    compression_proactive_prune_min_reclaim = max(
        0,
        _parse_prune_int(
            _compression_cfg.get("proactive_prune_min_reclaim_tokens", 4096), 4096
        ),
    )
    # protect_first_n is the number of non-system messages to protect at
    # the head, in addition to the system prompt (which is always
    # implicitly protected by the compressor).  Floor at 0 — a value of
    # 0 means "preserve only the system prompt + summary + tail", which
    # is a legitimate (and common) configuration for long-running
    # rolling-compaction sessions.
    compression_protect_first = max(
        0, int(_compression_cfg.get("protect_first_n", 3))
    )
    compression_abort_on_summary_failure = str(
        _compression_cfg.get("abort_on_summary_failure", False)
    ).lower() in {"true", "1", "yes"}
    # Per-model threshold overrides: keys are substring-matched against the
    # model name (longest match wins). Empty dict = use the global threshold
    # for all models (backward compatible).
    _raw_model_thresholds = _compression_cfg.get("model_thresholds", {})
    if isinstance(_raw_model_thresholds, dict):
        compression_model_thresholds = {
            str(k): float(v) for k, v in _raw_model_thresholds.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    else:
        compression_model_thresholds = {}
    # Absolute token cap: when set, compression triggers at the lower of
    # the ratio-based threshold and this absolute count. Clamped to the
    # model's context length at apply-time so a cap above the window is
    # a no-op (ratio-based threshold wins).
    compression_threshold_tokens = _compression_cfg.get("threshold_tokens")
    if compression_threshold_tokens is not None:
        try:
            compression_threshold_tokens = int(compression_threshold_tokens)
            if compression_threshold_tokens <= 0:
                compression_threshold_tokens = None
        except (TypeError, ValueError):
            compression_threshold_tokens = None
    # In-place compaction: when True, compress_context() rewrites the message
    # list + rebuilds the system prompt WITHOUT rotating the session id (no
    # parent_session_id chain, no `name #N` renumber). See #38763 and
    # agent/conversation_compression.py. Consumed by compress_context(), not the
    # compressor, so it rides on the agent.
    # Default True must match DEFAULT_CONFIG["compression"]["in_place"]
    # (#38763). default=False here previously flipped agents into rotation
    # mode whenever the merged config omitted the key (partial configs,
    # load_config failure → {}), re-arming the pre-lease drift abort.
    compression_in_place = is_truthy_value(
        _compression_cfg.get("in_place"), default=True
    )
    # Opt-in (default False): a micro-compaction pass rewrites already-sent
    # history every turn, which breaks the provider prompt-cache prefix on a
    # per-turn cadence rather than at an episodic boundary. That is the cost
    # `proactive_prune_min_reclaim_tokens` exists to amortize, so the feature
    # stays off until an operator opts in and accepts the tradeoff.
    compression_micro_compact = is_truthy_value(
        _compression_cfg.get("micro_compact"), default=False
    )
    # How often a pass runs, in completed turns. Each pass rewrites
    # already-sent history and costs one prompt-cache break, so this is the
    # dial for how often that cost is paid: 1 = every turn (most aggressive
    # reclaim), 5 = one break per five turns. Clamped to >= 1.
    compression_micro_compact_every_n_turns = max(
        1,
        _parse_prune_int(_compression_cfg.get("micro_compact_every_n_turns", 1), 1),
    )
    # Rolling-summary defrag threshold, in tokens. Lived on the compressor as
    # a hardcoded attribute with no path from config until now.
    compression_micro_compact_defrag_tokens = max(
        1,
        _parse_prune_int(
            _compression_cfg.get("micro_compact_defrag_threshold_tokens", 2000),
            2000,
        ),
    )
    codex_app_server_auto_compaction = str(
        _compression_cfg.get("codex_app_server_auto", "native") or "native"
    ).lower()
    if codex_app_server_auto_compaction not in {"native", "hermes", "off"}:
        logger.warning(
            "Invalid compression.codex_app_server_auto=%r; using 'native'. "
            "Valid values are: native, hermes, off.",
            codex_app_server_auto_compaction,
        )
        codex_app_server_auto_compaction = "native"
    # Native OpenAI Responses server-side compaction (opt-in). Only ever
    # engages for gpt-5.6-family models on api.openai.com or the ChatGPT
    # Codex backend — the per-request gate lives in agent/native_compaction.py.
    # Shared truthy coercion: "false"/"off"/"no" strings stay disabled
    # (bool("false") is True — #82777).
    from utils import is_truthy_value as _is_truthy

    codex_responses_native_compaction = _is_truthy(
        _compression_cfg.get("codex_responses_native", False)
    )
    _native_threshold_raw = _compression_cfg.get(
        "codex_responses_compact_threshold", 200_000
    )
    try:
        if isinstance(_native_threshold_raw, bool):
            raise ValueError
        codex_responses_compact_threshold = int(_native_threshold_raw)
        if codex_responses_compact_threshold <= 0:
            raise ValueError
    except (TypeError, ValueError):
        logger.warning(
            "Invalid compression.codex_responses_compact_threshold=%r; using 200000.",
            _native_threshold_raw,
        )
        codex_responses_compact_threshold = 200_000
    # Opt-in idle compaction: compact a session up front when it resumes after
    # this many seconds of inactivity (0 = disabled). Time-based, so it
    # complements the size-based threshold above. Consumed by build_turn_context().
    compression_idle_compact_after_seconds = max(
        0, int(_compression_cfg.get("idle_compact_after_seconds", 0))
    )

    # Read optional explicit context_length override for the auxiliary
    # compression model. Custom endpoints often cannot report this via
    # /models, so the startup feasibility check needs the config hint.
    try:
        _aux_cfg = cfg_get(_agent_cfg, "auxiliary", "compression", default={})
    except Exception:
        _aux_cfg = {}
    if isinstance(_aux_cfg, dict):
        _aux_context_config = _aux_cfg.get("context_length")
    else:
        _aux_context_config = None
    if _aux_context_config is not None:
        try:
            _aux_context_config = int(_aux_context_config)
        except (TypeError, ValueError):
            _aux_context_config = None
    agent._aux_compression_context_length_config = _aux_context_config

    # Read explicit model output-token override from config when the
    # caller did not pass one directly.
    _model_cfg = _agent_cfg.get("model", {})
    if agent.max_tokens is None and isinstance(_model_cfg, dict):
        _config_max_tokens = _model_cfg.get("max_tokens")
        if _config_max_tokens is not None:
            try:
                if isinstance(_config_max_tokens, bool):
                    raise ValueError
                _parsed_max_tokens = int(_config_max_tokens)
                if _parsed_max_tokens <= 0:
                    raise ValueError
                agent.max_tokens = _parsed_max_tokens
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid model.max_tokens in config.yaml: %r — "
                    "must be a positive integer (e.g. 4096). "
                    "Falling back to provider default.",
                    _config_max_tokens,
                )
                print(
                    f"\n⚠ Invalid model.max_tokens in config.yaml: {_config_max_tokens!r}\n"
                    f"  Must be a positive integer (e.g. 4096).\n"
                    f"  Falling back to provider default.\n",
                    file=sys.stderr,
                )
    agent._session_init_model_config["max_tokens"] = agent.max_tokens

    # Read explicit context_length override from model config
    if isinstance(_model_cfg, dict):
        _config_context_length = _model_cfg.get("context_length")
    else:
        _config_context_length = None
    if _config_context_length is not None:
        try:
            _config_context_length = int(_config_context_length)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid model.context_length in config.yaml: %r — "
                "must be a plain integer (e.g. 256000, not '256K'). "
                "Falling back to auto-detection.",
                _config_context_length,
            )
            print(
                f"\n⚠ Invalid model.context_length in config.yaml: {_config_context_length!r}\n"
                f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                f"  Falling back to auto-detected context window.\n",
                file=sys.stderr,
            )
            _config_context_length = None

    # Resolve custom_providers once before route-scoping a global context pin:
    # a named custom provider may keep its base URL only in this list rather
    # than repeating it under ``model``.
    try:
        from hermes_cli.config import get_compatible_custom_providers
        _custom_providers = get_compatible_custom_providers(_agent_cfg)
    except Exception:
        _custom_providers = _agent_cfg.get("custom_providers")
        if not isinstance(_custom_providers, list):
            _custom_providers = []

    # ``model.context_length`` describes the configured default model. A
    # process launched directly with ``--model`` / ``-m`` has already replaced
    # ``agent.model`` before this initializer loads config, so carrying the
    # default model's explicit window into that different runtime is stale. The
    # live switch/fallback paths already clear this override; keep direct-start
    # overrides consistent with them and let provider metadata resolve the
    # active model's window instead.
    if _config_context_length is not None and isinstance(_model_cfg, dict):
        _default = _model_cfg.get("default")
        if isinstance(_default, dict):
            from hermes_cli.config import split_model_config_default
            _default, _ = split_model_config_default(_default)
        _configured_default_model = str(_default or "").strip()
        _configured_default_runtime_model = _configured_default_model
        _active_runtime_model = agent.model
        if _configured_default_model:
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                _configured_default_runtime_model = normalize_model_for_provider(
                    _configured_default_model, agent.provider
                )
                _active_runtime_model = normalize_model_for_provider(
                    agent.model, agent.provider
                )
            except Exception:
                pass
        _configured_provider = str(_model_cfg.get("provider") or "").strip()
        _configured_base_url = _normalize_route_base_url(
            _model_cfg.get("base_url")
        )
        _configured_provider_norm = _normalize_custom_provider_name(
            _configured_provider
        )
        _custom_provider_candidate = bool(_configured_provider_norm)
        _runtime_first_provider_ids = {
            "auto",
            "moa",
            "vertex",
            "google-vertex",
            "vertex-ai",
            "gcp-vertex",
            "vertexai",
        }
        if _configured_provider_norm in _runtime_first_provider_ids:
            _custom_provider_candidate = False
        elif (
            _custom_provider_candidate
            and _configured_provider_norm != "custom"
            and not _configured_provider_norm.startswith("custom:")
        ):
            try:
                from hermes_cli.auth import resolve_provider as resolve_auth_provider

                _resolved_auth_provider = resolve_auth_provider(
                    _configured_provider_norm
                )
                _custom_provider_candidate = (
                    str(_resolved_auth_provider or "").strip().lower()
                    != _configured_provider_norm
                )
            except Exception:
                pass
        if not _configured_base_url and _custom_provider_candidate:
            _configured_custom_provider = _normalize_custom_provider_name(
                _configured_provider
            )
            _user_providers = _agent_cfg.get("providers")
            _disabled_custom_provider_ids: set[str] = set()
            if isinstance(_user_providers, dict):
                from hermes_cli.config import is_provider_enabled

                for _provider_key, _provider_entry in _user_providers.items():
                    if not isinstance(_provider_entry, dict):
                        continue
                    _entry_name = str(
                        _provider_entry.get("name") or ""
                    ).strip()
                    _entry_provider_ids = _custom_provider_runtime_ids(
                        _provider_key
                    ) | _custom_provider_runtime_ids(_entry_name)
                    if not is_provider_enabled(_provider_entry):
                        _disabled_custom_provider_ids.update(
                            provider_id
                            for provider_id in _entry_provider_ids
                            if provider_id
                        )
                        continue
                    if _configured_custom_provider not in _entry_provider_ids:
                        continue
                    _configured_base_url = _normalize_route_base_url(
                        _provider_entry.get("api")
                        or _provider_entry.get("url")
                        or _provider_entry.get("base_url")
                    )
                    if _configured_base_url:
                        break
            if not _configured_base_url:
                for _provider_entry in _custom_providers:
                    if not isinstance(_provider_entry, dict):
                        continue
                    _entry_name = str(
                        _provider_entry.get("name") or ""
                    ).strip()
                    _entry_provider_key = str(
                        _provider_entry.get("provider_key") or ""
                    ).strip().lower()
                    _entry_provider_ids = _custom_provider_runtime_ids(
                        _entry_name
                    ) | _custom_provider_runtime_ids(_entry_provider_key)
                    if (
                        _entry_provider_key
                        and _custom_provider_runtime_ids(_entry_provider_key)
                        & _disabled_custom_provider_ids
                    ):
                        continue
                    if _configured_custom_provider not in _entry_provider_ids:
                        continue
                    _configured_base_url = _normalize_route_base_url(
                        _provider_entry.get("base_url")
                    )
                    if _configured_base_url:
                        break
        _active_route_url = str(agent.base_url or "")
        _requested_route_url = str(base_url or "")
        if "?" in _requested_route_url.split("#", 1)[0]:
            try:
                _requested_parts = urlparse(_requested_route_url)
                _requested_without_query = urlunparse(
                    _requested_parts._replace(query="")
                )
                if _normalize_route_base_url(
                    _requested_without_query
                ) == _normalize_route_base_url(_active_route_url):
                    _active_route_url = _requested_route_url
            except (TypeError, ValueError):
                pass
        _active_base_url = _normalize_route_base_url(_active_route_url)
        _route_mismatch = _context_route_mismatch(
            _configured_base_url,
            _active_base_url,
            _configured_provider,
            agent.provider,
            already_normalized=True,
        )
        _model_mismatch = bool(
            _configured_default_runtime_model
            and _configured_default_runtime_model != _active_runtime_model
        )
        if _model_mismatch or _route_mismatch:
            logger.debug(
                "Ignoring model.context_length=%s for startup runtime %s at %s "
                "(configured default is %s at %s)",
                _config_context_length,
                agent.model,
                _active_base_url or agent.provider,
                _configured_default_model,
                _configured_base_url or _model_cfg.get("provider"),
            )
            _config_context_length = None

    # Store for reuse by _check_compression_model_feasibility (auxiliary
    # compression model context-length detection needs the same list).
    agent._custom_providers = _custom_providers
    _merge_custom_provider_extra_body(agent, _custom_providers)

    # Check custom_providers per-model context_length
    if _config_context_length is None and _custom_providers:
        try:
            from hermes_cli.config import get_custom_provider_context_length
            _cp_ctx_resolved = get_custom_provider_context_length(
                model=agent.model,
                base_url=agent.base_url,
                custom_providers=_custom_providers,
            )
            if _cp_ctx_resolved:
                _config_context_length = int(_cp_ctx_resolved)
        except Exception:
            _cp_ctx_resolved = None

        # Surface a clear warning if the user set a context_length but it
        # wasn't a valid positive int — the helper silently skips those.
        if _config_context_length is None:
            _target = _normalize_route_base_url(agent.base_url)
            for _cp_entry in _custom_providers:
                if not isinstance(_cp_entry, dict):
                    continue
                _cp_url = _normalize_route_base_url(_cp_entry.get("base_url"))
                if _target and _cp_url == _target:
                    _cp_models = _cp_entry.get("models", {})
                    if isinstance(_cp_models, dict):
                        _cp_model_cfg = _cp_models.get(agent.model, {})
                        if isinstance(_cp_model_cfg, dict):
                            _cp_ctx = _cp_model_cfg.get("context_length")
                            if _cp_ctx is not None:
                                try:
                                    _parsed = int(_cp_ctx)
                                    if _parsed <= 0:
                                        raise ValueError
                                except (TypeError, ValueError):
                                    logger.warning(
                                        "Invalid context_length for model %r in "
                                        "custom_providers: %r — must be a positive "
                                        "integer (e.g. 256000, not '256K'). "
                                        "Falling back to auto-detection.",
                                        agent.model, _cp_ctx,
                                    )
                                    print(
                                        f"\n⚠ Invalid context_length for model {agent.model!r} in custom_providers: {_cp_ctx!r}\n"
                                        f"  Must be a positive integer (e.g. 256000, not '256K').\n"
                                        f"  Falling back to auto-detected context window.\n",
                                        file=sys.stderr,
                                    )
                    break

    # Persist for reuse on switch_model / fallback activation. Must come
    # AFTER the custom_providers branch so per-model overrides aren't lost.
    agent._config_context_length = _config_context_length

    _lmstudio_runtime_context_length = provider_runtime._ensure_lmstudio_runtime_loaded(agent,
        _config_context_length
    )
    if provider_runtime._lmstudio_load_was_unverified(_lmstudio_runtime_context_length):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length; falling back to configured context"
        )
    _effective_context_length = provider_runtime._effective_lmstudio_context_length(
        _config_context_length,
        _lmstudio_runtime_context_length,
    )



    # Select context engine: config-driven (like memory providers).
    # 1. Check config.yaml context.engine setting
    # 2. Check plugins/context_engine/<name>/ directory (repo-shipped)
    # 3. Check general plugin system (user-installed plugins)
    # 4. Fall back to built-in ContextCompressor
    _selected_engine = None
    _copy_failed = False
    _engine_name = "compressor"  # default
    try:
        _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
        _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
    except Exception:
        pass

    if _engine_name != "compressor":
        # Try loading from plugins/context_engine/<name>/
        try:
            from plugins.context_engine import load_context_engine
            _selected_engine = load_context_engine(_engine_name)
        except Exception as _ce_load_err:
            logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

        # Try general plugin system as fallback
        if _selected_engine is None:
            _candidate = None
            try:
                from hermes_cli.plugins import get_plugin_context_engine
                _candidate = get_plugin_context_engine()
            except Exception:
                _candidate = None
            if _candidate is not None and _candidate.name == _engine_name:
                # Deep-copy the shared plugin singleton so a child agent's
                # update_model() can't mutate the parent's compressor (#42449).
                # Copy can fail for engines holding uncopyable state (locks, DB
                # connections, clients); in that case fall back to the built-in
                # compressor with an ACCURATE message rather than silently
                # mislabelling it "not found".
                import copy
                try:
                    _selected_engine = copy.deepcopy(_candidate)
                except Exception as _copy_err:
                    _copy_failed = True
                    logger.warning(
                        "Context engine '%s' could not be safely copied for this "
                        "agent (%s) — falling back to built-in compressor. Plugin "
                        "engines that hold uncopyable state (locks, DB connections) "
                        "should implement __deepcopy__ to copy only mutable budget "
                        "state.",
                        _engine_name, _copy_err,
                    )
                    _selected_engine = None

        if _selected_engine is None and not _copy_failed:
            logger.warning(
                "Context engine '%s' not found — falling back to built-in compressor",
                _engine_name,
            )
    # else: config says "compressor" — use built-in, don't auto-activate plugins

    if _selected_engine is not None:
        agent.context_compressor = _selected_engine
        # External engines own compaction policy: the host compression
        # threshold (including the Codex gpt-5.5 autoraise above) only
        # configures the built-in ContextCompressor and never reaches the
        # plugin, so the autoraise notice would announce a change that does
        # not apply. Drop it. (#44439)
        agent._compression_threshold_autoraised = None
        # Resolve context_length for plugin engines — mirrors switch_model() path
        from agent.model_metadata import get_model_context_length
        _plugin_ctx_len = get_model_context_length(
            agent.model,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            config_context_length=_effective_context_length,
            provider=agent.provider,
            custom_providers=_custom_providers,
        )
        # Per-model threshold overrides are part of the explicit
        # context-engine contract: assign them BEFORE the initial
        # update_model() call so the first resolution (which derives
        # threshold_percent/threshold_tokens for the initial model) already
        # sees the overrides. Assigning after update_model() left the initial
        # model on the engine's global threshold until the first /model
        # switch. Engines that override update_model() own their own policy
        # and may ignore the attribute.
        if compression_model_thresholds:
            agent.context_compressor.model_thresholds = compression_model_thresholds
        agent.context_compressor.update_model(
            model=agent.model,
            context_length=_plugin_ctx_len,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            provider=agent.provider,
            api_mode=agent.api_mode,
        )
        if not agent.quiet_mode:
            logger.info("Using context engine: %s", _selected_engine.name)
    else:
        agent.context_compressor = ContextCompressor(
            model=agent.model,
            threshold_percent=compression_threshold,
            protect_first_n=compression_protect_first,
            protect_last_n=compression_protect_last,
            summary_target_ratio=compression_target_ratio,
            summary_model_override=None,
            quiet_mode=agent.quiet_mode,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            config_context_length=_effective_context_length,
            provider=agent.provider,
            api_mode=agent.api_mode,
            abort_on_summary_failure=compression_abort_on_summary_failure,
            max_tokens=agent.max_tokens,
            model_thresholds=compression_model_thresholds,
            threshold_tokens_cap=compression_threshold_tokens,
            proactive_prune_tokens=compression_proactive_prune_tokens,
            proactive_prune_min_result_chars=compression_proactive_prune_min_chars,
            proactive_prune_min_reclaim_tokens=compression_proactive_prune_min_reclaim,
            min_tail_user_messages=compression_min_tail_users,
            tail_mode=compression_tail_mode,
        )
    _bind_session_state = getattr(agent.context_compressor, "bind_session_state", None)
    if callable(_bind_session_state):
        try:
            _bind_session_state(session_db=session_db, session_id=agent.session_id)
        except Exception:
            pass
    agent.compression_enabled = compression_enabled
    agent.compression_in_place = compression_in_place
    # Apply micro-compaction settings to the compressor (feature is opt-in)
    _cc = getattr(agent, "context_compressor", None)
    if _cc is not None and hasattr(_cc, "_micro_compact_enabled"):
        _cc._micro_compact_enabled = compression_micro_compact
    if _cc is not None and hasattr(_cc, "_micro_compact_every_n_turns"):
        _cc._micro_compact_every_n_turns = compression_micro_compact_every_n_turns
    if _cc is not None and hasattr(_cc, "_micro_compact_defrag_threshold_tokens"):
        _cc._micro_compact_defrag_threshold_tokens = (
            compression_micro_compact_defrag_tokens
        )
    agent.codex_app_server_auto_compaction = codex_app_server_auto_compaction
    agent.codex_responses_native_compaction = codex_responses_native_compaction
    agent.codex_responses_compact_threshold = codex_responses_compact_threshold
    agent.max_compression_attempts = compression_max_attempts
    agent.compression_idle_compact_after_seconds = (
        compression_idle_compact_after_seconds
    )

    # Reject models whose context window is below the minimum required
    # for reliable tool-calling workflows (64K tokens).
    _ctx = getattr(agent.context_compressor, "context_length", 0)
    _allow_lmstudio_explicit_below_floor = (
        str(getattr(agent, "provider", "") or "").strip().lower() == "lmstudio"
        and isinstance(agent._config_context_length, int)
        and not isinstance(agent._config_context_length, bool)
        and agent._config_context_length > 0
    )
    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:
        raise ValueError(
            f"Model {agent.model} has a context window of {_ctx:,} tokens, "
            f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
            f"by Hermes Agent.  Choose a model with at least "
            f"{MINIMUM_CONTEXT_LENGTH // 1000}K context.  If your server "
            f"reports a window smaller than the model's true window, set "
            f"model.context_length in config.yaml to the real value "
            f"(this must be at least {MINIMUM_CONTEXT_LENGTH // 1000}K)."
        )

    # Nous Hermes 3/4 are chat models, not tool-call-tuned. The interactive
    # CLI already warns via cli.py show_banner() (richer output + /model hint),
    # so skip platform=="cli" here to avoid emitting the warning twice per
    # startup. (Gateway/TUI/cron construct with quiet_mode=True and are already
    # gated off by the `not agent.quiet_mode` check above; this guard's active
    # job is the CLI dedup, and it leaves the door open for any non-quiet
    # non-CLI surface to still surface the warning.)
    if not agent.quiet_mode and (agent.platform or "cli") != "cli":
        try:
            from hermes_cli.model_switch import _check_hermes_model_warning

            _hermes_warn = _check_hermes_model_warning(agent.model or "")
            if _hermes_warn:
                _user_msg = (
                    "⚠ Nous Research Hermes 3 & 4 models are NOT agentic — they "
                    "lack reliable tool-calling for agent workflows (delegation, "
                    "cron, proactive tools). Consider an agentic model instead "
                    "(Claude, GPT, Gemini, Qwen-Coder, etc.)."
                )
                if hasattr(agent, "_emit_warning"):
                    status_output._emit_warning(agent, _user_msg)
                else:
                    print(f"\n{_user_msg}\n", file=sys.stderr)
                logger.warning(_hermes_warn)
        except Exception:
            pass

    # Inject context engine tool schemas (e.g. lcm_grep, lcm_describe, lcm_expand).
    # Skip names that are already present — the get_tool_definitions()
    # quiet_mode cache returned a shared list pre-#17335, so a stray
    # mutation here would poison subsequent agent inits in the same
    # Gateway process and trip provider-side 'duplicate tool name'
    # errors. Even with the cache fix, dedup is the right defense
    # against plugin paths that may register the same schemas via
    # ctx.register_tool(). Mirrors the memory tools dedup above.
    #
    # Respect the platform's enabled_toolsets configuration (#5544):
    # context engine tools follow the same gating pattern as memory
    # provider tools — without the gate, `platform_toolsets: telegram: []`
    # would still leak lcm_* tools into the tool surface and incur the
    # same local-model latency penalty.
    agent._context_engine_tool_names: set = set()
    if (
        hasattr(agent, "context_compressor")
        and agent.context_compressor
        and agent.tools is not None
        and (
            agent.enabled_toolsets is None
            or "context_engine" in agent.enabled_toolsets
        )
    ):
        _existing_tool_names = {
            t.get("function", {}).get("name")
            for t in agent.tools
            if isinstance(t, dict)
        }
        from agent.memory_manager import normalize_tool_schema as _normalize_tool_schema
        for _raw_schema in agent.context_compressor.get_tool_schemas():
            _schema = _normalize_tool_schema(_raw_schema)
            if _schema is None:
                # A schema with no resolvable name (e.g. an already-wrapped
                # entry) would append a nameless tool that strict providers
                # 400 on, disabling the whole toolset (#47707). Skip it.
                logger.warning(
                    "Context engine returned a tool schema with no resolvable "
                    "name; skipping to avoid poisoning the request (%r)",
                    _raw_schema,
                )
                continue
            _tname = _schema["name"]
            if _tname in _existing_tool_names:
                continue  # already registered via plugin/cache path
            _wrapped = {"type": "function", "function": _schema}
            agent.tools.append(_wrapped)
            agent.valid_tool_names.add(_tname)
            agent._context_engine_tool_names.add(_tname)
            _existing_tool_names.add(_tname)

    # Notify context engine of session start
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        try:
            agent.context_compressor.on_session_start(
                agent.session_id,
                hermes_home=str(get_hermes_home()),
                platform=agent.platform or "cli",
                model=agent.model,
                context_length=getattr(agent.context_compressor, "context_length", 0),
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as _ce_err:
            logger.debug("Context engine on_session_start: %s", _ce_err)

    agent._subdirectory_hints = SubdirectoryHintTracker(
        working_dir=os.getenv("TERMINAL_CWD") or None,
    )
    agent._user_turn_count = 0
    # Copilot x-initiator flag: first API call of a user turn sends "user" (#3040).
    agent._is_user_initiated_turn = False

    # Cumulative token usage for the session
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_api_calls = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"

    # ── Ollama num_ctx injection ──
    # Ollama defaults to 2048 context regardless of the model's capabilities.
    # When running against an Ollama server, detect the model's max context
    # and pass num_ctx on every chat request so the full window is used.
    # User override: set model.ollama_num_ctx in config.yaml to cap VRAM use.
    # If model.context_length is set, it caps num_ctx so the user's VRAM
    # budget is respected even when GGUF metadata advertises a larger window.
    agent._ollama_num_ctx: int | None = None
    _ollama_num_ctx_override = None
    if isinstance(_model_cfg, dict):
        _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
    if _ollama_num_ctx_override is not None:
        try:
            agent._ollama_num_ctx = int(_ollama_num_ctx_override)
        except (TypeError, ValueError):
            logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
    if agent._ollama_num_ctx is None and agent.base_url and is_local_endpoint(agent.base_url):
        try:
            # ``agent.api_key`` may be a callable (Entra token provider).
            # Ollama detection makes a manual HTTP request and expects a
            # string — Azure Foundry isn't a local endpoint so this branch
            # never fires for Entra, but guard defensively.
            _key_for_ollama = agent.api_key if isinstance(agent.api_key, str) else ""
            _detected = query_ollama_num_ctx(agent.model, agent.base_url, api_key=_key_for_ollama or "")
            if _detected and _detected > 0:
                agent._ollama_num_ctx = _detected
        except Exception as exc:
            logger.debug("Ollama num_ctx detection failed: %s", exc)
    # Cap auto-detected ollama_num_ctx to the user's explicit context_length.
    # Without this, GGUF metadata can advertise 256K+ which Ollama honours
    # by allocating that much VRAM — blowing up small GPUs even though the
    # user explicitly set a smaller context_length in config.yaml.
    if (
        agent._ollama_num_ctx
        and _config_context_length
        and _ollama_num_ctx_override is None  # don't override explicit ollama_num_ctx
        and agent._ollama_num_ctx > _config_context_length
    ):
        logger.info(
            "Ollama num_ctx capped: %d -> %d (model.context_length override)",
            agent._ollama_num_ctx, _config_context_length,
        )
        agent._ollama_num_ctx = _config_context_length
    if agent._ollama_num_ctx and not agent.quiet_mode:
        logger.info(
            "Ollama num_ctx: will request %d tokens (model max from /api/show)",
            agent._ollama_num_ctx,
        )

    # Codex gpt-5.x autoraise notice: show at most once per profile/config
    # state. Without the persisted marker the notice re-fires on every agent
    # init — and the gateway rebuilds the agent per inbound message, so Discord
    # etc. saw it repeatedly (#54432). A change in the raised threshold (or the
    # autoraised model) updates the marker state and re-notifies once. The
    # config display gate (compression.codex_gpt55_autoraise_notice) still
    # suppresses the banner entirely without disabling the threshold autoraise.
    _autoraise = getattr(agent, "_compression_threshold_autoraised", None) or {}
    _show_autoraise_notice = (
        bool(_autoraise)
        and compression_enabled
        and _codex_gpt55_autoraise_notice
        and not _codex_gpt55_autoraise_notice_seen(_autoraise)
    )

    if not agent.quiet_mode:
        if compression_enabled:
            # Report the active engine's own threshold — for a plugin engine
            # the host compression_threshold is not in effect, and mixing the
            # two printed a percent that contradicted the token count. (#44439)
            _active_threshold_pct = getattr(
                agent.context_compressor, "threshold_percent", compression_threshold
            )
            _cap_note = ""
            _cap = getattr(agent.context_compressor, "threshold_tokens_cap", None)
            if _cap and _cap > 0:
                _cap_note = f" (capped at {_cap:,} tokens)"
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (compress at {int(_active_threshold_pct*100)}% = {agent.context_compressor.threshold_tokens:,}{_cap_note})")
        else:
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (auto-compression disabled)")
        # Notice with the exact opt-back-out command. Printed inline at startup
        # for CLI users; gateway users get the same text replayed via
        # _compression_warning on turn 1 (set below).
        if _show_autoraise_notice:
            print(_build_codex_gpt5_autoraise_notice(
                _autoraise,
                context_length=getattr(agent.context_compressor, "context_length", None),
            ))

    # Check immediately so CLI users see the warning at startup.
    # Gateway status_callback is not yet wired, so any warning is stored
    # in _compression_warning and replayed in the first run_conversation().
    agent._compression_warning = None
    # Gateway parity for the Codex gpt-5.x autoraise notice: the startup print
    # above only reaches the CLI, so stash the same text here to be replayed
    # through status_callback on the first turn (Telegram/Discord/Slack/etc.).
    if _show_autoraise_notice:
        agent._compression_warning = _build_codex_gpt5_autoraise_notice(
            _autoraise,
            context_length=getattr(agent.context_compressor, "context_length", None),
        )

    # Mark shown so repeated inits in this profile (e.g. every gateway message)
    # stay silent. Recorded once, whether the notice went to the CLI print or
    # the gateway replay slot.
    if _show_autoraise_notice:
        _record_codex_gpt55_autoraise_notice(_autoraise)
    # Lazy feasibility check: deferred to the first turn that approaches the
    # compression threshold. Running it eagerly here costs ~400ms cold (network
    # probe of the auxiliary provider chain + /models lookup) on every agent
    # init, including short ``chat -q`` runs that never reach the threshold.
    # ``ensure_compression_feasibility_checked`` (called from
    # ``run_conversation``'s preflight) runs it at most once per agent.
    agent._compression_feasibility_checked = False

    # Snapshot primary runtime for per-turn restoration.  When fallback
    # activates during a turn, the next turn restores these values so the
    # preferred model gets a fresh attempt each time.  Uses a single dict
    # so new state fields are easy to add without N individual attributes.
    _cc = agent.context_compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        # Context engine state that _try_activate_fallback() overwrites.
        # Use getattr for model/base_url/api_key/provider since plugin
        # engines may not have these (they're ContextCompressor-specific).
        "compressor_model": getattr(_cc, "model", agent.model),
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(_cc, "api_key", ""),
        "compressor_provider": getattr(_cc, "provider", agent.provider),
        "compressor_context_length": _cc.context_length,
        "compressor_threshold_tokens": _cc.threshold_tokens,
    }
    if agent.api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })
