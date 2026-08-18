"""Tool-owned phase of create_agent initialization."""

from __future__ import annotations

import logging

from model_tools import check_toolset_requirements, get_tool_definitions

logger = logging.getLogger("run_agent")


def initialize_tools(agent, *, enabled_toolsets, disabled_toolsets):
    # A multiplexed gateway may enter a different HERMES_HOME after
    # ``model_tools`` was first imported. Ensure that profile's keyed plugin
    # manager has discovered its registrations before taking the tool snapshot.
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning("Plugin discovery failed during agent setup", exc_info=True)

    # Get available tools with filtering. Capture the registry generation this
    # snapshot is derived from FIRST, so a later concurrent refresh can tell
    # whether it holds a newer or staler view (see refresh_agent_mcp_tools).
    try:
        from tools.registry import registry as _snapshot_registry
        agent._tool_snapshot_generation = _snapshot_registry._generation
    except Exception:
        agent._tool_snapshot_generation = 0
    agent.tools = get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=agent.quiet_mode,
    )

    # Show tool configuration and store valid tool names for validation
    agent.valid_tool_names = set()
    if agent.tools:
        agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
        tool_names = sorted(agent.valid_tool_names)
        if not agent.quiet_mode:
            print(f"🛠️  Loaded {len(agent.tools)} tools: {', '.join(tool_names)}")
            # Show filtering info if applied
            if enabled_toolsets:
                print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
            if disabled_toolsets:
                print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
    elif not agent.quiet_mode:
        print("🛠️  No tools loaded (all tools filtered out or unavailable)")

    # Kanban worker/orchestrator lifecycle guidance is session-static:
    # the dispatcher decides at spawn time whether this process is a kanban
    # worker (kanban_show tool is present iff HERMES_KANBAN_TASK is set).
    # Resolving the ~835-token block once here avoids re-running the
    # membership test + reference on every system-prompt rebuild
    # (init + each context compression).
    from agent.prompt_builder import KANBAN_GUIDANCE
    agent._kanban_worker_guidance = (
        KANBAN_GUIDANCE if "kanban_show" in agent.valid_tool_names else ""
    )

    # Check tool requirements
    if agent.tools and not agent.quiet_mode:
        requirements = check_toolset_requirements()
        missing_reqs = [name for name, available in requirements.items() if not available]
        if missing_reqs:
            print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")

    # Show trajectory saving status
    if agent.save_trajectories and not agent.quiet_mode:
        print("📝 Trajectory saving enabled")

    # Show ephemeral system prompt status
    if agent.ephemeral_system_prompt and not agent.quiet_mode:
        prompt_preview = agent.ephemeral_system_prompt[:60] + "..." if len(agent.ephemeral_system_prompt) > 60 else agent.ephemeral_system_prompt
        print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")

    # Show prompt caching status
    if agent._use_prompt_caching and not agent.quiet_mode:
        if agent._use_native_cache_layout and agent.provider == "anthropic":
            source = "native Anthropic"
        elif agent._use_native_cache_layout:
            source = "Anthropic-compatible endpoint"
        else:
            source = "Claude via OpenRouter"
        print(f"💾 Prompt caching: ENABLED ({source}, {agent._cache_ttl} TTL)")
