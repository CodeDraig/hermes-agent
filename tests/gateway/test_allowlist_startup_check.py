"""Tests for the startup allowlist warning check in gateway/run.py."""

import os
from unittest.mock import patch


def _would_warn():
    """Replicate the startup allowlist warning logic. Returns True if warning fires."""
    _any_allowlist = any(
        os.getenv(v)
        for v in ("TELEGRAM_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                   "GATEWAY_ALLOWED_USERS")
    )
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
        os.getenv(v, "").lower() in {"true", "1", "yes"}
        for v in ("TELEGRAM_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS")
    )
    return not _any_allowlist and not _allow_all


class TestAllowlistStartupCheck:

    def test_no_config_emits_warning(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn() is True

