"""Direct contract checks for the decomposed agent initialization phases."""

from types import SimpleNamespace

import agent.init_context as init_context
import agent.init_runtime as init_runtime
import agent.init_session as init_session
import agent.init_state as init_state
import agent.init_tools as init_tools
from agent.agent_init import init_agent


def test_init_agent_keeps_phase_order(monkeypatch):
    calls = []

    def identity(agent, **kwargs):
        calls.append("identity")
        return "openai"

    def route(agent, **kwargs):
        calls.append("route")

    def execution(agent, **kwargs):
        calls.append("execution")

    def client(agent, **kwargs):
        calls.append("client")

    def tools(agent, **kwargs):
        calls.append("tools")

    def session(agent, **kwargs):
        calls.append("session")
        return {}

    def context(agent, **kwargs):
        calls.append("context")

    monkeypatch.setattr(init_state, "initialize_agent_identity", identity)
    monkeypatch.setattr(init_runtime, "initialize_provider_route", route)
    monkeypatch.setattr(init_state, "initialize_execution_state", execution)
    monkeypatch.setattr(init_runtime, "initialize_provider_client", client)
    monkeypatch.setattr(init_tools, "initialize_tools", tools)
    monkeypatch.setattr(init_session, "initialize_session", session)
    monkeypatch.setattr(init_context, "initialize_context", context)

    init_agent(SimpleNamespace(), model="test/model", fallback_providers=[])

    assert calls == ["identity", "route", "execution", "client", "tools", "session", "context"]


def test_init_agent_rejects_non_list_fallback_providers():
    try:
        init_agent(SimpleNamespace(), fallback_providers={"provider": "openai", "model": "gpt-5"})
    except TypeError as exc:
        assert str(exc) == "fallback_providers must be a list of provider entries or None"
    else:
        raise AssertionError("non-list fallback_providers should be rejected")
