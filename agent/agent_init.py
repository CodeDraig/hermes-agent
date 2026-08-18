"""Ordered orchestration for :meth:`run_agent.AIAgent.__init__`.

The responsibility-owned phase modules perform the actual initialization;
this module keeps only the public argument contract and their dependency
order. Tests patch those owning modules directly rather than a façade here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def init_agent(
    agent,
    base_url: str = None,
    api_key: str = None,
    provider: str = None,
    api_mode: str = None,
    acp_command: str = None,
    acp_args: list[str] | None = None,
    model: str = "",
    max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    save_trajectories: bool = False,
    verbose_logging: bool = False,
    quiet_mode: bool = False,
    tool_progress_mode: str = "all",
    ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100,
    log_prefix: str = "",
    providers_allowed: List[str] = None,
    providers_ignored: List[str] = None,
    providers_order: List[str] = None,
    provider_sort: str = None,
    provider_require_parameters: bool = False,
    provider_data_collection: str = None,
    openrouter_min_coding_score: Optional[float] = None,
    session_id: str = None,
    tool_progress_callback: callable = None,
    tool_start_callback: callable = None,
    tool_complete_callback: callable = None,
    thinking_callback: callable = None,
    reasoning_callback: callable = None,
    clarify_callback: callable = None,
    step_callback: callable = None,
    stream_delta_callback: callable = None,
    interim_assistant_callback: callable = None,
    tool_gen_callback: callable = None,
    status_callback: callable = None,
    notice_callback: callable = None,
    notice_clear_callback: callable = None,
    event_callback: Optional[Callable[[str, dict], None]] = None,
    reaction_callback: Optional[Callable[[str], None]] = None,
    max_tokens: int = None,
    reasoning_config: Dict[str, Any] = None,
    service_tier: str = None,
    request_overrides: Dict[str, Any] = None,
    prefill_messages: List[Dict[str, Any]] = None,
    platform: str = None,
    user_id: str = None,
    user_id_alt: str = None,
    user_name: str = None,
    chat_id: str = None,
    chat_name: str = None,
    chat_type: str = None,
    thread_id: str = None,
    gateway_session_key: str = None,
    skip_context_files: bool = False,
    load_soul_identity: bool = False,
    skip_memory: bool = False,
    skip_background_review: bool = False,
    session_db=None,
    parent_session_id: str = None,
    iteration_budget: "IterationBudget" = None,
    fallback_providers: List[Dict[str, Any]] = None,
    credential_pool=None,
    checkpoints_enabled: bool = False,
    checkpoint_max_snapshots: int = 20,
    checkpoint_max_total_size_mb: int = 500,
    checkpoint_max_file_size_mb: int = 10,
    pass_session_id: bool = False,
    requested_provider: str = None,
):
    """
    Initialize the AI Agent.

    Args:
        base_url (str): Base URL for the model API (optional)
        api_key (str): API key for authentication (optional, uses env var if not provided)
        provider (str): Provider identifier (optional; used for telemetry/routing hints)
        requested_provider (str): Original provider identity before runtime canonicalization
        api_mode (str): API mode override: "chat_completions" or "codex_responses"
        model (str): Model name to use (default: "anthropic/claude-opus-4.6")
        max_iterations (int): Maximum number of tool calling iterations (default: 90)
        enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
        disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
        save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
        verbose_logging (bool): Enable verbose logging for debugging (default: False)
        quiet_mode (bool): Suppress progress output for clean CLI experience (default: False)
        ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 100)
        log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
        providers_allowed (List[str]): OpenRouter providers to allow (optional)
        providers_ignored (List[str]): OpenRouter providers to ignore (optional)
        providers_order (List[str]): OpenRouter providers to try in order (optional)
        provider_sort (str): Sort providers by price/throughput/latency (optional)
        openrouter_min_coding_score (float): Coding-score floor (0.0-1.0) for the
            openrouter/pareto-code router. Only applied when model == "openrouter/pareto-code".
            None or empty = let OpenRouter pick the strongest available coder.
        session_id (str): Pre-generated session ID for logging (optional, auto-generated if not provided)
        tool_progress_callback (callable): Callback function(tool_name, args_preview) for progress notifications
        clarify_callback (callable): Callback function(question, choices) -> str for interactive user questions.
            Provided by the platform layer (CLI or gateway). If None, the clarify tool returns an error.
        max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
        reasoning_config (Dict): OpenRouter reasoning configuration override (e.g. {"effort": "none"} to disable thinking).
            If None, defaults to {"enabled": True, "effort": "medium"} for OpenRouter. Set to disable/customize reasoning.
        prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
            Useful for injecting a few-shot example or priming the model's response style.
            Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            NOTE: Anthropic Sonnet 4.6+ and Opus 4.6+ reject a conversation that ends on an
            assistant-role message (400 error).  For those models use structured outputs or
            output_config.format instead of a trailing-assistant prefill.
        platform (str): The interface platform (for example "cli" or "telegram").
            Used to inject platform-specific formatting hints into the system prompt.
        skip_context_files (bool): If True, skip auto-injection of project context files
            (SOUL.md, .hermes.md, AGENTS.md, CLAUDE.md, .cursorrules) from the cwd / HERMES_HOME
            into the system prompt. Use this for batch processing and data generation to avoid
            polluting trajectories with user-specific persona or project instructions.
        load_soul_identity (bool): If True, still use ~/.hermes/SOUL.md as the primary
            identity even when skip_context_files=True. Project context files from the cwd
            remain skipped.
    """
    from agent.init_context import initialize_context
    from agent.init_runtime import initialize_provider_client, initialize_provider_route
    from agent.init_session import initialize_session
    from agent.init_state import initialize_agent_identity, initialize_execution_state
    from agent.init_tools import initialize_tools

    if fallback_providers is not None and not isinstance(fallback_providers, list):
        raise TypeError("fallback_providers must be a list of provider entries or None")

    provider_name = initialize_agent_identity(
        agent,
        base_url=base_url,
        provider=provider,
        requested_provider=requested_provider,
        credential_pool=credential_pool,
        acp_command=acp_command,
        acp_args=acp_args,
        model=model,
        max_iterations=max_iterations,
        iteration_budget=iteration_budget,
        save_trajectories=save_trajectories,
        verbose_logging=verbose_logging,
        quiet_mode=quiet_mode,
        tool_progress_mode=tool_progress_mode,
        ephemeral_system_prompt=ephemeral_system_prompt,
        platform=platform,
        user_id=user_id,
        user_id_alt=user_id_alt,
        user_name=user_name,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type=chat_type,
        thread_id=thread_id,
        gateway_session_key=gateway_session_key,
        skip_context_files=skip_context_files,
        load_soul_identity=load_soul_identity,
        skip_background_review=skip_background_review,
        pass_session_id=pass_session_id,
        log_prefix_chars=log_prefix_chars,
        log_prefix=log_prefix,
    )
    initialize_provider_route(
        agent,
        api_mode=api_mode,
        provider_name=provider_name,
        credential_pool=credential_pool,
    )

    initialize_execution_state(
        agent,
        tool_progress_callback=tool_progress_callback,
        tool_start_callback=tool_start_callback,
        tool_complete_callback=tool_complete_callback,
        thinking_callback=thinking_callback,
        reasoning_callback=reasoning_callback,
        clarify_callback=clarify_callback,
        step_callback=step_callback,
        stream_delta_callback=stream_delta_callback,
        interim_assistant_callback=interim_assistant_callback,
        status_callback=status_callback,
        notice_callback=notice_callback,
        notice_clear_callback=notice_clear_callback,
        event_callback=event_callback,
        reaction_callback=reaction_callback,
        tool_gen_callback=tool_gen_callback,
        providers_allowed=providers_allowed,
        providers_ignored=providers_ignored,
        providers_order=providers_order,
        provider_sort=provider_sort,
        provider_require_parameters=provider_require_parameters,
        provider_data_collection=provider_data_collection,
        openrouter_min_coding_score=openrouter_min_coding_score,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        max_tokens=max_tokens,
        reasoning_config=reasoning_config,
        service_tier=service_tier,
        request_overrides=request_overrides,
        prefill_messages=prefill_messages,
    )

    # Initialize LLM client via centralized provider router.
    # The router handles auth resolution, base URL, headers, and
    # Codex/Anthropic wrapping for all known providers.
    # raw_codex=True because the main agent needs direct responses.stream()
    # access for Codex Responses API streaming.
    initialize_provider_client(
        agent,
        api_key=api_key,
        base_url=base_url,
        fallback_providers=fallback_providers,
    )

    initialize_tools(
        agent,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )
    
    _agent_cfg = initialize_session(
        agent,
        session_id=session_id,
        checkpoints_enabled=checkpoints_enabled,
        checkpoint_max_snapshots=checkpoint_max_snapshots,
        checkpoint_max_total_size_mb=checkpoint_max_total_size_mb,
        checkpoint_max_file_size_mb=checkpoint_max_file_size_mb,
        session_db=session_db,
        parent_session_id=parent_session_id,
        reasoning_config=reasoning_config,
        max_tokens=max_tokens,
    )

    initialize_context(
        agent,
        config=_agent_cfg,
        base_url=base_url,
        platform=platform,
        skip_memory=skip_memory,
        session_db=session_db,
    )



__all__ = ["init_agent"]
