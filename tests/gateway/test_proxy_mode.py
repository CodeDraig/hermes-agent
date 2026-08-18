"""HTTP proxy resolution used by retained gateway transports."""

from gateway.platforms.base import resolve_proxy_url


def _clear_proxy_environment(monkeypatch):
    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


def test_no_proxy_bypasses_matching_host(monkeypatch):
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "api.telegram.org")

    assert resolve_proxy_url(target_hosts="api.telegram.org") is None


def test_no_proxy_bypasses_cidr_target(monkeypatch):
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "149.154.160.0/20")

    assert resolve_proxy_url(target_hosts=["149.154.167.220"]) is None
