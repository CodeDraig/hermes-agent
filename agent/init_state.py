"""State-owned phases of create_agent initialization."""

from __future__ import annotations



import logging
import os
import threading
import time
from typing import Dict, List, Optional

from agent.iteration_budget import IterationBudget
from agent.memory_manager import StreamingContextScrubber
from agent.process_bootstrap import _install_safe_stdio
from agent.rate_limit_tracker import RateLimitState
from agent.session_activity import ActivityProvenance
from agent.think_scrubber import StreamingThinkScrubber
from agent.tool_guardrails import ToolCallGuardrailController, ToolGuardrailDecision
from hermes_constants import get_hermes_home

logger = logging.getLogger("run_agent")


def initialize_agent_identity(
    agent,
    *,
    base_url,
    provider,
    requested_provider,
    credential_pool,
    acp_command,
    acp_args,
    model,
    max_iterations,
    iteration_budget,
    save_trajectories,
    verbose_logging,
    quiet_mode,
    tool_progress_mode,
    ephemeral_system_prompt,
    platform,
    user_id,
    user_id_alt,
    user_name,
    chat_id,
    chat_name,
    chat_type,
    thread_id,
    gateway_session_key,
    skip_context_files,
    load_soul_identity,
    skip_background_review,
    pass_session_id,
    log_prefix_chars,
    log_prefix,
):
    import agent.provider_runtime as provider_runtime
    _install_safe_stdio()

    agent.model = model
    agent.max_iterations = max_iterations
    # Shared iteration budget — parent creates, children inherit.
    # Consumed by every LLM turn across parent + all subagents.
    agent.iteration_budget = iteration_budget or IterationBudget(max_iterations)
    agent.save_trajectories = save_trajectories
    agent.verbose_logging = verbose_logging
    agent.quiet_mode = quiet_mode
    agent.tool_progress_mode = tool_progress_mode
    agent.ephemeral_system_prompt = ephemeral_system_prompt
    agent.platform = platform
    agent._user_id = user_id  # Platform user identifier (gateway sessions)
    agent._user_id_alt = user_id_alt  # Optional stable alternate platform identifier
    agent._user_name = user_name
    agent._chat_id = chat_id
    agent._chat_name = chat_name
    agent._chat_type = chat_type
    agent._thread_id = thread_id
    agent._gateway_session_key = gateway_session_key  # Stable per-chat key (e.g. agent:main:telegram:dm:123)
    # Pluggable print function — CLI replaces this with _cprint so that
    # raw ANSI status lines are routed through prompt_toolkit's renderer
    # instead of going directly to stdout where patch_stdout's StdoutProxy
    # would mangle the escape sequences.  None = use builtins.print.
    agent._print_fn = None
    agent.background_review_callback = None  # Optional sync callback for gateway delivery
    agent.memory_notifications = "on"  # Memory update notifications: "off", "on", "verbose"
    agent.skip_context_files = skip_context_files
    agent.load_soul_identity = load_soul_identity
    # Background review (memory/skill) opt-out switch. When True, skips the
    # _spawn_background_review fork at end-of-turn -- avoids ~30K tokens /
    # event of extra LLM cost on cron-style sessions where review forks
    # provide no value (no human in the loop, no skill-creation pressure).
    # skip_memory=True already disables the memory-review trigger; this
    # flag is the explicit single-switch off for both review paths.
    agent.skip_background_review = bool(skip_background_review)
    agent.pass_session_id = pass_session_id
    agent.log_prefix_chars = log_prefix_chars
    agent.log_prefix = f"{log_prefix} " if log_prefix else ""
    # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
    provider_runtime.set_base_url(agent, base_url or "")
    provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
    agent.provider = provider_name or ""
    agent.requested_provider = (
        requested_provider.strip().lower()
        if isinstance(requested_provider, str) and requested_provider.strip()
        else agent.provider
    )
    agent._credential_pool = credential_pool
    agent.acp_command = acp_command
    agent.acp_args = list(acp_args or [])

    return provider_name


def initialize_execution_state(
    agent,
    *,
    tool_progress_callback,
    tool_start_callback,
    tool_complete_callback,
    thinking_callback,
    reasoning_callback,
    clarify_callback,
    step_callback,
    stream_delta_callback,
    interim_assistant_callback,
    status_callback,
    notice_callback,
    notice_clear_callback,
    event_callback,
    reaction_callback,
    tool_gen_callback,
    providers_allowed,
    providers_ignored,
    providers_order,
    provider_sort,
    provider_require_parameters,
    provider_data_collection,
    openrouter_min_coding_score,
    enabled_toolsets,
    disabled_toolsets,
    max_tokens,
    reasoning_config,
    service_tier,
    request_overrides,
    prefill_messages,
):
    import agent.provider_runtime as provider_runtime
    agent.tool_progress_callback = tool_progress_callback
    agent.tool_start_callback = tool_start_callback
    agent.tool_complete_callback = tool_complete_callback
    agent.suppress_status_output = False
    agent.thinking_callback = thinking_callback
    agent.reasoning_callback = reasoning_callback
    agent.clarify_callback = clarify_callback
    agent.step_callback = step_callback
    agent.stream_delta_callback = stream_delta_callback
    agent.interim_assistant_callback = interim_assistant_callback
    agent.status_callback = status_callback
    agent.notice_callback = notice_callback
    agent.notice_clear_callback = notice_clear_callback
    agent.event_callback = event_callback
    agent.reaction_callback = reaction_callback
    agent.tool_gen_callback = tool_gen_callback


    # Tool execution state — allows _vprint during tool execution
    # even when stream consumers are registered (no tokens streaming then)
    agent._executing_tools = False
    agent._tool_guardrails = ToolCallGuardrailController()
    agent._tool_guardrail_halt_decision: ToolGuardrailDecision | None = None

    # Interrupt mechanism for breaking out of tool loops
    agent._interrupt_requested = False
    agent._interrupt_message = None  # Optional message that triggered interrupt
    # Explicit hard cancellation is separate from redirect/message state. A
    # thread-safe Event makes the cause atomic for auxiliary stream pollers.
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id: int | None = None  # Set at run_conversation() start
    agent._interrupt_thread_signal_pending = False
    agent._client_lock = threading.RLock()
    agent._model_request_active = threading.Event()
    agent._supports_active_turn_redirect = True

    # /steer mechanism — inject a user note into the next tool result
    # without interrupting the agent. Unlike interrupt(), steer() does
    # NOT set _interrupt_requested; it waits for the current tool batch
    # to finish naturally, then the drain hook appends the text to the
    # last tool result's content so the model sees it on its next
    # iteration. Message-role alternation is preserved (we modify an
    # existing tool message rather than inserting a new user turn).
    agent._pending_steer: Optional[str] = None
    agent._pending_steer_lock = threading.Lock()

    # Active-turn redirect mechanism. A regular follow-up sent while the model
    # is generating is different from a hard /stop: preserve the valid turn
    # prefix, cancel only the in-flight model request, and rebuild its tail with
    # the correction. The loop drains this slot at a role-safe boundary.
    agent._pending_redirect: Optional[str] = None
    agent._pending_redirect_lock = threading.Lock()

    # Concurrent-tool worker thread tracking.  `_execute_tool_calls_concurrent`
    # runs each tool on its own ThreadPoolExecutor worker — those worker
    # threads have tids distinct from `_execution_thread_id`, so
    # `_set_interrupt(True, _execution_thread_id)` alone does NOT cause
    # `is_interrupted()` inside the worker to return True.  Track the
    # workers here so `interrupt()` / `clear_interrupt()` can fan out to
    # their tids explicitly.
    agent._tool_worker_threads: set[int] = set()
    agent._tool_worker_threads_lock = threading.Lock()

    # Subagent delegation state
    agent._delegate_depth = 0        # 0 = top-level agent, incremented for children
    agent._active_children = []      # Running child agents (for interrupt propagation)
    agent._active_children_lock = threading.Lock()

    # Background memory/skill review state (agent/background_review.py). Holds
    # the forked review create_agent while its run_conversation() is in flight, so
    # the NEXT live turn can proactively interrupt a still-running review
    # instead of letting the two race concurrently against the same
    # session_id/credentials (observed as doubled prompt-token counts and a
    # Ctrl+C-proof lockup when a live turn started before a review fired at
    # the end of the prior turn had finished).
    agent._background_review_agent = None
    agent._background_review_lock = threading.Lock()

    # Store OpenRouter provider preferences
    agent.providers_allowed = providers_allowed
    agent.providers_ignored = providers_ignored
    agent.providers_order = providers_order
    agent.provider_sort = provider_sort
    agent.provider_require_parameters = provider_require_parameters
    agent.provider_data_collection = provider_data_collection
    agent.openrouter_min_coding_score = openrouter_min_coding_score

    # Store toolset filtering options
    agent.enabled_toolsets = enabled_toolsets
    agent.disabled_toolsets = disabled_toolsets

    # Model response configuration
    agent.max_tokens = max_tokens  # None = use model default
    agent.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
    agent.service_tier = service_tier
    agent.request_overrides = dict(request_overrides or {})
    agent.prefill_messages = prefill_messages or []  # Prefilled conversation turns
    agent._force_ascii_payload = False

    # Anthropic prompt caching: auto-enabled for Claude models on native
    # Anthropic, OpenRouter, and third-party gateways that speak the
    # Anthropic protocol (``api_mode == 'anthropic_messages'``). Reduces
    # input costs by ~75% on multi-turn conversations. Uses four breakpoints:
    # the static system prefix, full system prompt, and last two messages
    # (falling back to system-and-3 when no static prefix is available). See
    # ``_anthropic_prompt_cache_policy`` for the layout-vs-transport decision.
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        provider_runtime.anthropic_prompt_cache_policy(agent)
    )
    agent._cache_disabled = False
    # Anthropic supports "5m" (default) and "1h" cache TTL tiers. Read from
    # config.yaml under prompt_caching.cache_ttl; unknown values keep "5m".
    # 1h tier costs 2x on write vs 1.25x for 5m, but amortizes across long
    # sessions with >5-minute pauses between turns (#14971).
    #
    # Setting cache_ttl to a falsy value (false / null / "off" / "disabled" /
    # "no" / "none") disables prompt caching entirely. This is useful for
    # OAuth subscription users where cache writes bill against "extra usage"
    # or for third-party proxies that inject their own cache_control markers
    # (#13477). The disable propagates through anthropic_prompt_cache_policy()
    # and restore_primary_runtime() so it survives /model switches and
    # fallback re-derivation (#33555).
    agent._cache_ttl = "5m"
    try:
        from hermes_cli.config import load_config_readonly as _load_pc_cfg

        from agent.provider_runtime import cache_ttl_means_disabled

        _pc_cfg = _load_pc_cfg().get("prompt_caching", {}) or {}
        _ttl = _pc_cfg.get("cache_ttl", "5m")
        if _ttl in {"5m", "1h"}:
            agent._cache_ttl = _ttl
        elif cache_ttl_means_disabled(_ttl):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False
            agent._cache_ttl = None
            agent._cache_disabled = True
    except Exception:
        pass

    # Iteration budget: the LLM is only notified when it actually exhausts
    # the iteration budget (api_call_count >= max_iterations).  At that
    # point we inject ONE message, allow one final API call, and if the
    # model doesn't produce a text response, force a user-message asking
    # it to summarise.  No intermediate pressure warnings — they caused
    # models to "give up" prematurely on complex tasks (#7915).
    agent._budget_exhausted_injected = False
    agent._budget_grace_call = False

    # Activity tracking — updated on each API call, tool execution, and
    # stream chunk.  Used by the gateway timeout handler to report what the
    # agent was doing when it was killed, and by the "still working"
    # notifications to show progress.
    agent._last_activity_ts: float = time.time()
    agent._last_activity_desc: str = "initializing"
    # Default / unmigrated paths and _touch_activity stamp unknown; named
    # provenances are stamped by compression writers (heartbeat / timeout / cooldown).
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    # Rate-limit durable SessionDB activity stamps from _touch_activity (#72016).
    agent._session_activity_last_persist_mono: float = 0.0
    agent._current_tool: str | None = None
    agent._api_call_count: int = 0
    # Opt-out flag for the between-turns MCP tool refresh (build_turn_context).
    # Set on internal forks (e.g. background_review) that must keep ``tools[]``
    # byte-identical to a parent for provider cache parity.
    agent._skip_mcp_refresh = False
    # Registry generation the current tool snapshot was derived from. Lets a
    # late/concurrent refresh reject a stale (older-generation) rebuild instead
    # of clobbering a newer one. Set adjacent to the tool snapshot below.
    agent._tool_snapshot_generation = 0
    # Rate limit tracking — updated from x-ratelimit-* response headers
    # after each API call.  Accessed by /usage slash command.
    agent._rate_limit_state: Optional["RateLimitState"] = None

    # Credits tracking (dev-only, L0 usage-aware-credits) — updated from
    # x-nous-credits-* response headers after each API call.  Session-start
    # remaining is latched the first time a header is ever seen so we can
    # report cumulative micros spent.  Surfaced behind HERMES_DEV_CREDITS.
    agent._credits_state = None
    agent._credits_session_start_micros = None
    # Threshold-notice latch (L4): active sticky-notice keys + the crossing gates.
    from agent.credits_tracker import new_credits_latch

    agent._credits_latch = new_credits_latch()

    # OpenRouter response cache hit counter — incremented when
    # X-OpenRouter-Cache-Status: HIT is seen in streaming response headers.
    agent._or_cache_hits: int = 0

    # Centralized logging — agent.log (INFO+) and errors.log (WARNING+)
    # both live under ~/.hermes/logs/.  Idempotent, so gateway mode
    # (which creates a new create_agent per message) won't duplicate handlers.
    from hermes_logging import setup_logging, setup_verbose_logging
    setup_logging(hermes_home=get_hermes_home())

    if agent.verbose_logging:
        setup_verbose_logging()
        logger.info("Verbose logging enabled (third-party library logs suppressed)")
    elif agent.quiet_mode:
        # In quiet mode (CLI default), keep console output clean —
        # but DO NOT raise per-logger levels. Doing so prevents the
        # root logger's file handlers (agent.log, errors.log) from
        # ever seeing the records, because Python checks
        # logger.isEnabledFor() before handler propagation. We rely
        # on the fact that hermes_logging.setup_logging() does not
        # install a console StreamHandler in quiet mode — so INFO
        # records flow to the file handlers but never reach a
        # console. Any future noise reduction belongs at the
        # handler level inside hermes_logging.py, not here.
        pass

    # Internal stream callback (set during streaming TTS).
    # Initialized here so _vprint can reference it before run_conversation.
    agent._stream_callback = None
    # Deferred paragraph break flag — set after tool iterations so a
    # single "\n\n" is prepended to the next real text delta.
    agent._stream_needs_break = False
    # Stateful scrubber for <memory-context> spans split across stream
    # deltas (#5719).  sanitize_context() alone can't survive chunk
    # boundaries because the block regex needs both tags in one string.
    agent._stream_context_scrubber = StreamingContextScrubber()
    # Stateful scrubber for reasoning/thinking tags in streamed deltas
    # (#17924).  Replaces the per-delta _strip_think_blocks regex that
    # destroyed downstream state (e.g. MiniMax-M2.7 streaming
    # '<think>' as delta1 and 'Let me check' as delta2 — the regex
    # erased delta1, so downstream state machines never learned a
    # block was open and leaked delta2 as content).
    agent._stream_think_scrubber = StreamingThinkScrubber()
    # Visible assistant text already delivered through live token callbacks
    # during the current model response. Used to avoid re-sending the same
    # commentary when the provider later returns it as a completed interim
    # assistant message.
    agent._current_streamed_assistant_text = ""
    # Completed interim messages delivered during the current user turn.
    # Unlike token-stream tracking, this spans Codex continuation/tool calls so
    # repeated commentary is not re-sent before normalization can deduplicate it.
    agent._delivered_interim_texts: set[str] = set()

    # Single-writer guard for the streaming delta sink (#65991). A stale/
    # superseded stream (e.g. one the stale-stream detector reconnected past,
    # whose socket abort raced and never actually stopped the old worker) must
    # NOT keep writing tokens into the turn alongside the retry's stream —
    # otherwise two coherent responses interleave token-by-token into one
    # transcript. Every streaming attempt claims a monotonic writer token; the
    # delta sink drops chunks whose calling thread holds a stale token. The
    # threading.local means threads that never claimed (non-streaming callers)
    # are never fenced, so the guard can only ever drop a superseded stream,
    # never the single legitimate writer.
    agent._stream_writer_lock = threading.Lock()
    agent._stream_writer_token = 0
    agent._stream_writer_tls = threading.local()
    agent._stream_writer_dropped = 0

    # Optional current-turn user-message override used when the API-facing
    # user message intentionally differs from the persisted transcript
    # (e.g. CLI voice mode adds a temporary prefix for the live call only).
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None

    # Cache anthropic image-to-text fallbacks per image payload/URL so a
    # single tool loop does not repeatedly re-run auxiliary vision on the
    # same image history.
    agent._anthropic_image_fallback_cache: Dict[str, str] = {}
