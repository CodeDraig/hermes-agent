"""Memory-provider access through the process startup plugin registry."""

from __future__ import annotations

from pathlib import Path


def discover_memory_providers() -> dict[str, dict]:
    """Return registered memory providers keyed by their native names."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()
    manager = get_plugin_manager()
    return {
        name: {
            "name": name,
            "provider": provider,
            "available": bool(provider.is_available()),
        }
        for name, provider in manager._memory_providers.items()
    }


def load_memory_provider(name: str):
    """Return the named startup-registered memory provider, if present."""
    normalized = str(name or "").strip().lower()
    if not normalized:
        return None
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()
    return get_plugin_manager()._memory_providers.get(normalized)


def find_provider_dir(name: str) -> Path | None:
    """Return the installed plugin directory that registered ``name``."""
    provider = load_memory_provider(name)
    if provider is None:
        return None
    module = __import__(provider.__class__.__module__, fromlist=["__name__"])
    module_file = getattr(module, "__file__", None)
    return Path(module_file).resolve().parent if module_file else None

