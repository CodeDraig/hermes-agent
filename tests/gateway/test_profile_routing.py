"""Tests for gateway/profile_routing.py — profile-based routing."""

import pytest
from gateway.profile_routing import (
    ProfileRoute,
    parse_profile_routes,
    match_profile_route,
)


class TestProfileRoute:
    def test_specificity_thread(self):
        r = ProfileRoute(name="t", platform="mattermost", profile="p",
                         scope_id="g", chat_id="c", thread_id="t")
        assert r.specificity == 14  # 2 + 4 + 8


    def test_frozen(self):
        r = ProfileRoute(name="x", platform="mattermost", profile="p")
        with pytest.raises(AttributeError):
            r.name = "y"


class TestProfileRouteMatching:
    def test_exact_thread_match(self):
        r = ProfileRoute(name="t", platform="mattermost", profile="trader",
                         scope_id="111", chat_id="222", thread_id="333")
        assert r.matches("mattermost", scope_id="111", chat_id="222", thread_id="333")
        assert not r.matches("mattermost", scope_id="111", chat_id="222", thread_id="444")


    def test_scope_and_chat_are_conjunctive(self):
        # A route declaring BOTH scope_id and chat_id requires both to match.
        # Regression guard: previously chat_id was checked first and returned
        # True before scope_id was ever consulted.
        r = ProfileRoute(name="sc", platform="mattermost", profile="scoped",
                         scope_id="111", chat_id="222")
        # Both match (direct channel) -> match
        assert r.matches("mattermost", scope_id="111", chat_id="222")
        # Both match via parent (thread inside the channel) -> match
        assert r.matches("mattermost", scope_id="111", chat_id="333", parent_chat_id="222")
        # chat matches but guild differs -> NO match (the bug this guards)
        assert not r.matches("mattermost", scope_id="999", chat_id="222")
        # guild matches but chat differs -> NO match
        assert not r.matches("mattermost", scope_id="111", chat_id="333")


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []


class TestMatchProfileRoute:


    def test_no_match_returns_none(self):
        routes = [
            ProfileRoute(name="r", platform="telegram", profile="p"),
        ]
        assert match_profile_route(routes, "mattermost") is None


class TestSessionKeyIntegration:
    def test_default_profile_key(self):
        from gateway.session import build_session_key, SessionSource, Platform
        src = SessionSource(platform=Platform.TELEGRAM, chat_id="123",
                            chat_type="channel", user_id="456")
        key = build_session_key(src)
        assert key.startswith("agent:main:")


class TestParentChatIdMatching:
    """Thread messages carry thread_id as chat_id; parent_chat_id is the channel."""

    def test_channel_route_matches_via_parent_chat_id(self):
        r = ProfileRoute(name="ch", platform="mattermost", profile="trader",
                         chat_id="222")
        assert r.matches("mattermost", chat_id="333", parent_chat_id="222")


    def test_match_profile_route_with_parent_chat_id(self):
        routes = [
            ProfileRoute(name="ch", platform="mattermost", profile="trader",
                         chat_id="222"),
        ]
        m = match_profile_route(routes, "mattermost", chat_id="333", parent_chat_id="222")
        assert m is not None
        assert m.profile == "trader"


class TestParentThreadMatching:
    """Thread messages match a route through their direct parent."""


    def test_forum_post_comment_matches_channel_not_thread_id(self):
        """Verify that thread_id matching is distinct from parent_chat_id matching."""
        routes = [
            ProfileRoute(name="channel", platform="mattermost", profile="channel_profile",
                         chat_id="parent_channel_123"),
            ProfileRoute(name="post", platform="mattermost", profile="post_profile",
                         thread_id="post_thread_456"),
        ]
        m = match_profile_route(routes, "mattermost", chat_id="post_thread_456",
                                 parent_chat_id="parent_channel_123")
        assert m is not None
        assert m.profile == "channel_profile"
