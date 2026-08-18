"""Pairing-store grants and Telegram allowlist mirroring."""

import os
from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


def _runner(*, paired: bool):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: paired)
    return runner


def _source(user_id: str):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id=user_id,
        user_name="Human",
        is_bot=False,
    )


def test_paired_user_is_authorized_outside_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner")
    assert _runner(paired=True)._is_user_authorized(_source("paired")) is True


def test_allowlisted_unpaired_user_is_authorized(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner")
    assert _runner(paired=False)._is_user_authorized(_source("owner")) is True


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import importlib
    import gateway.pairing as pairing

    importlib.reload(pairing)
    return pairing.PairingStore()


def test_approval_mirrors_into_configured_telegram_allowlist(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner")
    captured = {}
    import hermes_cli.config as config

    monkeypatch.setattr(
        config,
        "save_env_value",
        lambda key, value: (
            captured.__setitem__(key, value),
            os.environ.__setitem__(key, value),
        ),
    )
    code = store.generate_code("telegram", "new-user", "")

    store.approve_code("telegram", code)

    assert captured["TELEGRAM_ALLOWED_USERS"] == "owner,new-user"


def test_revoke_removes_telegram_allowlist_grant(store, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner,new-user")
    captured = {}
    import hermes_cli.config as config

    monkeypatch.setattr(
        config,
        "save_env_value",
        lambda key, value: captured.__setitem__(key, value),
    )
    store._approve_user("telegram", "new-user", "")

    assert store.revoke("telegram", "new-user") is True
    assert captured["TELEGRAM_ALLOWED_USERS"] == "owner"
