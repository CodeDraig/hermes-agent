"""Plugin ownership surface after platform-plugin excision."""

from hermes_cli.plugins import PluginContext, _VALID_PLUGIN_KINDS


def test_platform_registration_is_not_public():
    assert not hasattr(PluginContext, "register_platform")


def test_non_platform_plugin_kinds_remain():
    assert {"standalone", "backend", "exclusive", "model-provider"} <= _VALID_PLUGIN_KINDS
    assert "platform" not in _VALID_PLUGIN_KINDS


def test_non_platform_registration_methods_remain():
    assert hasattr(PluginContext, "register_tool")
    assert hasattr(PluginContext, "register_hook")
    assert hasattr(PluginContext, "register_context_engine")
    assert hasattr(PluginContext, "register_memory_provider")
