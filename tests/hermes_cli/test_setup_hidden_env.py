"""Setup surfaces ask only for what a platform can't start without.

The knobs in SETUP_HIDDEN_ENV_SUFFIXES are self-configuring (/sethome) or
already correct by default, so they don't belong on a setup form. They must
still be reachable everywhere else — this is a presentation change, not a
feature removal.
"""

import pytest

from hermes_cli.setup_hidden_env import is_setup_hidden_env


class TestIsSetupHiddenEnv:
    @pytest.mark.parametrize(
        "key",
        [
            "TELEGRAM_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL_NAME",
            "TELEGRAM_ALLOW_ALL_USERS",
            "TELEGRAM_REPLY_TO_MODE",
            "MATTERMOST_REPLY_MODE",
            "TELEGRAM_PROXY",
        ],
    )
    def test_self_configuring_knobs_are_hidden(self, key):
        assert is_setup_hidden_env(key)


class TestStillConfigurable:
    def test_gateway_still_honors_the_env_vars(self, tmp_path, monkeypatch):
        """Nothing was removed from the product — only from the setup form."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:t")
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "999")
        monkeypatch.setenv("TELEGRAM_REPLY_TO_MODE", "off")

        from gateway.config import Platform, load_gateway_config

        telegram = load_gateway_config().platforms[Platform.TELEGRAM]
        assert telegram.home_channel is not None
        assert telegram.home_channel.chat_id == "999"
        assert telegram.reply_to_mode == "off"
