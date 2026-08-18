"""Tests for the central command registry and autocomplete."""

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    COMMANDS,
    COMMANDS_BY_CATEGORY,
    CommandDef,
    GATEWAY_KNOWN_COMMANDS,
    SUBCOMMANDS,
    SlashCommandAutoSuggest,
    SlashCommandCompleter,
    _CMD_NAME_LIMIT,
    _clamp_command_names,
    _sanitize_telegram_name,
    gateway_help_lines,
    resolve_command,
    telegram_bot_commands,
    telegram_menu_commands,
    telegram_menu_max_commands,
)


def _completions(completer: SlashCommandCompleter, text: str):
    return list(
        completer.get_completions(
            Document(text=text),
            CompleteEvent(completion_requested=True),
        )
    )


# ---------------------------------------------------------------------------
# CommandDef registry tests
# ---------------------------------------------------------------------------

class TestCommandRegistry:


    def test_save_command_supports_formats(self):
        cmd = resolve_command("save")
        assert cmd is not None
        assert cmd.name == "save"
        # /save is a cross-platform session export: json (default), md, html
        assert not cmd.cli_only
        for token in ("json", "md", "html"):
            assert token in (cmd.args_hint or "")

    def test_no_duplicate_canonical_names(self):
        names = [cmd.name for cmd in COMMAND_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_no_alias_collides_with_canonical_name(self):
        """An alias must not shadow another command's canonical name."""
        canonical_names = {cmd.name for cmd in COMMAND_REGISTRY}
        for cmd in COMMAND_REGISTRY:
            for alias in cmd.aliases:
                if alias in canonical_names:
                    # reset -> new is intentional (reset IS an alias for new)
                    target = next(c for c in COMMAND_REGISTRY if c.name == alias)
                    # This should only happen if the alias points to the same entry
                    assert resolve_command(alias).name == cmd.name or alias == cmd.name, \
                        f"Alias '{alias}' of '{cmd.name}' shadows canonical '{target.name}'"





# ---------------------------------------------------------------------------
# resolve_command tests
# ---------------------------------------------------------------------------

class TestResolveCommand:


    def test_topic_is_gateway_command(self):
        topic = resolve_command("topic")
        assert topic is not None
        assert topic.name == "topic"
        assert "topic" in GATEWAY_KNOWN_COMMANDS

    def test_context_command_registered_with_ctx_alias(self):
        ctx = resolve_command("context")
        assert ctx is not None
        assert ctx.name == "context"
        assert resolve_command("ctx").name == "context"
        assert "all" in (ctx.subcommands or ())
        # Available on both CLI and gateway surfaces
        assert not ctx.cli_only and not ctx.gateway_only
        assert "context" in GATEWAY_KNOWN_COMMANDS




# ---------------------------------------------------------------------------
# Derived dicts (backwards compat)
# ---------------------------------------------------------------------------

class TestDerivedDicts:


    def test_commands_dict_includes_aliases(self):
        assert "/bg" in COMMANDS
        assert "/reset" in COMMANDS
        assert "/q" in COMMANDS
        assert "/exit" in COMMANDS
        assert "/reload_mcp" in COMMANDS
        assert "/gateway" in COMMANDS

    def test_commands_by_category_covers_all_categories(self):
        registry_categories = {cmd.category for cmd in COMMAND_REGISTRY if not cmd.gateway_only}
        assert set(COMMANDS_BY_CATEGORY.keys()) == registry_categories


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------

class TestGatewayKnownCommands:

    def test_includes_config_gated_cli_only(self):
        """Commands with gateway_config_gate are always in GATEWAY_KNOWN_COMMANDS."""
        for cmd in COMMAND_REGISTRY:
            if cmd.gateway_config_gate:
                assert cmd.name in GATEWAY_KNOWN_COMMANDS, \
                    f"config-gated command '{cmd.name}' should be in GATEWAY_KNOWN_COMMANDS"


    def test_is_frozenset(self):
        assert isinstance(GATEWAY_KNOWN_COMMANDS, frozenset)


class TestGatewayHelpLines:

    def test_excludes_cli_only_commands_without_config_gate(self):
        import re
        lines = gateway_help_lines()
        joined = "\n".join(lines)
        for cmd in COMMAND_REGISTRY:
            if cmd.cli_only and not cmd.gateway_config_gate:
                # Word-boundary match so `/reload` doesn't match `/reload-mcp`
                pattern = rf'`/{re.escape(cmd.name)}(?![-_\w])'
                assert not re.search(pattern, joined), \
                    f"cli_only command /{cmd.name} should not be in gateway help"

    def test_includes_alias_note_for_bg(self):
        lines = gateway_help_lines()
        bg_line = [l for l in lines if "/background" in l]
        assert len(bg_line) == 1
        assert "/bg" in bg_line[0]


class TestTelegramBotCommands:
    def test_returns_list_of_tuples(self):
        cmds = telegram_bot_commands()
        assert len(cmds) > 10
        for name, desc in cmds:
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_no_hyphens_in_command_names(self):
        """Telegram does not support hyphens in command names."""
        for name, _ in telegram_bot_commands():
            assert "-" not in name, f"Telegram command '{name}' contains a hyphen"


    def test_includes_builtin_commands_with_required_args(self):
        """Built-in arg-taking commands (e.g. /queue, /steer, /background)
        are now included because their handlers return usage text when
        invoked without arguments — issue #24312."""
        names = {name for name, _ in telegram_bot_commands()}
        assert "background" in names
        assert "queue" in names
        assert "steer" in names


# ---------------------------------------------------------------------------
# Config-gated gateway commands
# ---------------------------------------------------------------------------

class TestGatewayConfigGate:
    """Tests for the gateway_config_gate mechanism on CommandDef."""


    def test_verbose_in_gateway_known_commands(self):
        """Config-gated commands are always recognized by the gateway."""
        assert "verbose" in GATEWAY_KNOWN_COMMANDS

    def test_config_gate_excluded_from_help_when_off(self, tmp_path, monkeypatch):
        """When the config gate is falsy, the command should not appear in help."""
        # Write a config with the gate off (default)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("display:\n  tool_progress_command: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        lines = gateway_help_lines()
        joined = "\n".join(lines)
        assert "`/verbose" not in joined



# ---------------------------------------------------------------------------
# Autocomplete (SlashCommandCompleter)
# ---------------------------------------------------------------------------

class TestSlashCommandCompleter:
    # -- basic prefix completion -----------------------------------------



    # -- exact-match trailing space --------------------------------------


    # -- non-slash input returns nothing ---------------------------------



    # -- skill commands via provider ------------------------------------

    def test_skill_commands_are_completed_from_provider(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/gif-search": {"description": "Search for GIFs across providers"},
            }
        )

        completions = _completions(completer, "/gif")

        assert len(completions) == 1
        assert completions[0].text == "gif-search"
        assert completions[0].display_text == "/gif-search"
        assert completions[0].display_meta_text == "⚡ Search for GIFs across providers"



    def test_skill_provider_exception_is_swallowed(self):
        """A broken provider should not crash autocomplete."""
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should return builtin matches only, no crash
        completions = _completions(completer, "/he")
        texts = {item.text for item in completions}
        assert "help" in texts




# ── Stacked slash-skill completion ──────────────────────────────────────


def _stacked_completer(**extra_skills):
    skills = {
        "/skill-a": {"description": "Skill A"},
        "/skill-b": {"description": "Skill B"},
        "/skill-c": {"description": "Skill C"},
        **extra_skills,
    }
    return SlashCommandCompleter(skill_commands_provider=lambda: skills)


class TestStackedSkillCompletion:
    """Second+ leading skill tokens keep getting completions (stacked
    slash-skill invocations, Claude Code v2.1.199 port follow-up)."""


    def test_no_completions_for_instruction_text(self):
        assert _completions(_stacked_completer(), "/skill-a do the") == []
        assert _completions(_stacked_completer(), "/skill-a ") == []


    def test_cap_stops_completions(self):
        skills = {f"/stk-{i}": {"description": f"S{i}"} for i in range(8)}
        completer = SlashCommandCompleter(skill_commands_provider=lambda: skills)
        text = " ".join(f"/stk-{i}" for i in range(5)) + " /stk-"
        assert _completions(completer, text) == []


# ── SUBCOMMANDS extraction ──────────────────────────────────────────────


class TestSubcommands:
    def test_explicit_subcommands_extracted(self):
        """Commands with explicit subcommands on CommandDef are extracted."""
        assert "/skills" in SUBCOMMANDS
        assert "install" in SUBCOMMANDS["/skills"]


    def test_commands_without_subcommands_not_in_dict(self):
        """Plain commands should not appear in SUBCOMMANDS."""
        assert "/help" not in SUBCOMMANDS
        assert "/quit" not in SUBCOMMANDS
        assert "/clear" not in SUBCOMMANDS


# ── Subcommand tab completion ───────────────────────────────────────────


class TestSubcommandCompletion:






    def test_tools_enable_skips_already_listed(self, monkeypatch):
        """If the user already typed a name, don't suggest it again."""
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda *_a, **_k: set(),
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_plugin_toolset_keys",
            lambda: set(),
        )

        completions = _completions(SlashCommandCompleter(), "/tools enable spotify ")
        texts = {c.text for c in completions}
        assert "spotify" not in texts


    def _fake_gateway(self, monkeypatch, platforms):
        """Patch load_gateway_config with a fake whose connected platforms are
        the keys of `platforms` (name -> home as None or a (chat_id, name) tuple).
        """
        from types import SimpleNamespace

        enums = {name: SimpleNamespace(value=name) for name in platforms}
        homes = {
            name: (None if home is None else SimpleNamespace(chat_id=home[0], name=home[1]))
            for name, home in platforms.items()
        }
        fake = SimpleNamespace(
            get_connected_platforms=lambda: list(enums.values()),
            get_home_channel=lambda p: homes[p.value],
        )
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: fake)

    def test_handoff_completes_connected_platforms(self, monkeypatch):
        """`/handoff ` offers connected platforms, with or without a home channel."""
        self._fake_gateway(
            monkeypatch,
            {
                "telegram": ("123", "Me"),
                "mattermost": None,  # no home channel yet -> still listed
            },
        )

        texts = {c.text for c in _completions(SlashCommandCompleter(), "/handoff ")}
        assert texts == {"telegram", "mattermost"}





# ── Ghost text (SlashCommandAutoSuggest) ────────────────────────────────


def _suggestion(text: str, completer=None) -> str | None:
    """Get ghost text suggestion for given input."""
    suggest = SlashCommandAutoSuggest(completer=completer)
    doc = Document(text=text)

    class FakeBuffer:
        pass

    result = suggest.get_suggestion(FakeBuffer(), doc)
    return result.text if result else None


class TestGhostText:
    def test_command_name_suggestion(self):
        """/he → 'lp'"""
        assert _suggestion("/he") == "lp"


    # -- stacked slash-skill ghost text -----------------------------------


    def test_stacked_skill_ghost_text_skips_used(self):
        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: {
                "/alpha": {"description": "A"},
                "/beta": {"description": "B"},
            }
        )
        assert _suggestion("/alpha /a", completer=completer) is None
        assert _suggestion("/alpha /b", completer=completer) == "eta"


# ---------------------------------------------------------------------------
# Telegram command name sanitization
# ---------------------------------------------------------------------------


class TestSanitizeTelegramName:
    """Tests for _sanitize_telegram_name() — Telegram requires [a-z0-9_] only."""

    def test_hyphens_replaced_with_underscores(self):
        assert _sanitize_telegram_name("my-skill-name") == "my_skill_name"





    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_telegram_name("a---b") == "a_b"
        assert _sanitize_telegram_name("a-+-b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_telegram_name("-leading") == "leading"
        assert _sanitize_telegram_name("trailing-") == "trailing"
        assert _sanitize_telegram_name("-both-") == "both"






class TestClampCommandNamesTriples:
    """Tests for _clamp_command_names with 3-tuples (name, desc, cmd_key).

    Skill entries pass through _clamp_command_names as 3-tuples so the
    original cmd_key survives name truncation.  Before the fix in PR #18951,
    the code stripped cmd_key into a side-dict keyed by the *original*
    (name, desc) pair — after truncation the lookup key no longer matched,
    silently losing the cmd_key.
    """


    def test_long_name_preserves_cmd_key(self):
        long = "a" * 50
        cmd_key = f"/{long}"
        result = _clamp_command_names([(long, "desc", cmd_key)], set())
        assert len(result) == 1
        name, desc, key = result[0]
        assert len(name) == _CMD_NAME_LIMIT
        assert key == cmd_key, "cmd_key must survive name clamping"

    def test_collision_preserves_cmd_key(self):
        prefix = "x" * _CMD_NAME_LIMIT
        long = "x" * 50
        result = _clamp_command_names(
            [(long, "desc", "/long-skill")], reserved={prefix},
        )
        assert len(result) == 1
        name, _desc, key = result[0]
        assert name == "x" * (_CMD_NAME_LIMIT - 1) + "0"
        assert key == "/long-skill"




class TestTelegramMenuCommands:
    """Integration: telegram_menu_commands enforces the 32-char limit."""









    def test_external_dir_skills_included_in_telegram_menu(self, tmp_path, monkeypatch):
        """External skills (``skills.external_dirs``) must appear in the Telegram menu.

        Regression test for #8110 — external skills were visible to the
        agent and CLI but silently excluded from gateway slash menus
        because ``_collect_gateway_skill_entries`` only accepted skills
        whose path started with ``SKILLS_DIR``.

        Also verifies the trailing-slash boundary: a directory that
        simply shares a prefix with a configured ``external_dirs`` entry
        (``/tmp/my-skills-extra`` vs ``/tmp/my-skills``) must NOT be
        admitted.
        """
        from unittest.mock import patch

        local_dir = tmp_path / "skills"
        local_dir.mkdir()
        external_dir = tmp_path / "my-skills"
        external_dir.mkdir()
        lookalike_dir = tmp_path / "my-skills-extra"
        lookalike_dir.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_dir}\n"
        )

        fake_cmds = {
            "/local-one": {
                "name": "local-one",
                "description": "Local",
                "skill_md_path": f"{local_dir}/local-one/SKILL.md",
                "skill_dir": f"{local_dir}/local-one",
            },
            "/morning-briefing": {
                "name": "morning-briefing",
                "description": "External skill",
                "skill_md_path": f"{external_dir}/morning-briefing/SKILL.md",
                "skill_dir": f"{external_dir}/morning-briefing",
            },
            "/lookalike-skill": {
                "name": "lookalike-skill",
                "description": "Lives in a sibling dir that shares a prefix",
                "skill_md_path": f"{lookalike_dir}/lookalike-skill/SKILL.md",
                "skill_dir": f"{lookalike_dir}/lookalike-skill",
            },
        }

        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", local_dir),
            patch(
                "agent.skill_utils.get_external_skills_dirs",
                return_value=[external_dir],
            ),
        ):
            menu, _ = telegram_menu_commands(max_commands=100)

        menu_names = {n for n, _ in menu}
        assert "local_one" in menu_names, "local skill must appear"
        assert "morning_briefing" in menu_names, (
            "external skill from skills.external_dirs must appear (fixes #8110)"
        )
        assert "lookalike_skill" not in menu_names, (
            "prefix-match sibling directories must not be admitted"
        )

    def test_special_chars_in_skill_names_sanitized(self, tmp_path, monkeypatch):
        """Skills with +, /, or other special chars produce valid Telegram names."""
        from unittest.mock import patch
        import re

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/jellyfin-+-jellystat-24h-summary": {
                "name": "Jellyfin + Jellystat 24h Summary",
                "description": "Test",
                "skill_md_path": f"{fake_skills_dir}/jellyfin/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/jellyfin",
            },
            "/sonarr-v3/v4-api": {
                "name": "Sonarr v3/v4 API",
                "description": "Test",
                "skill_md_path": f"{fake_skills_dir}/sonarr/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/sonarr",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, _ = telegram_menu_commands(max_commands=100)

        # Every name must match Telegram's [a-z0-9_] requirement
        tg_valid = re.compile(r"^[a-z0-9_]+$")
        for name, _ in menu:
            assert tg_valid.match(name), f"Invalid Telegram command name: {name!r}"

    def test_empty_sanitized_names_excluded(self, tmp_path, monkeypatch):
        """Skills whose names sanitize to empty string are silently dropped."""
        from unittest.mock import patch

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/+++": {
                "name": "+++",
                "description": "All special chars",
                "skill_md_path": f"{fake_skills_dir}/bad/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/bad",
            },
            "/valid-skill": {
                "name": "valid-skill",
                "description": "Normal skill",
                "skill_md_path": f"{fake_skills_dir}/valid/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/valid",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, _ = telegram_menu_commands(max_commands=100)

        menu_names = {n for n, _ in menu}
        # The valid skill should be present, the empty one should not
        assert "valid_skill" in menu_names
        # No empty string in menu names
        assert "" not in menu_names


# ---------------------------------------------------------------------------
# Plugin slash command integration
# ---------------------------------------------------------------------------

class TestPluginCommandEnumeration:
    """Plugin commands registered via ctx.register_command() must be surfaced
    by the retained gateway command enumerators.
    """

    def _patch_plugin_commands(self, monkeypatch, commands):
        """Monkeypatch hermes_cli.plugins.get_plugin_commands() to a fixed dict."""
        from hermes_cli import plugins as _plugins_mod

        monkeypatch.setattr(
            _plugins_mod, "get_plugin_commands", lambda: dict(commands)
        )



    def test_plugin_command_with_hyphens_sanitized_for_telegram(self, monkeypatch):
        """Plugin names containing hyphens must be underscore-normalized for Telegram."""
        self._patch_plugin_commands(monkeypatch, {
            "my-plugin-cmd": {
                "handler": lambda _a: "ok",
                "description": "desc",
                "args_hint": "",
                "plugin": "p",
            }
        })
        names = {name for name, _desc in telegram_bot_commands()}
        assert "my_plugin_cmd" in names
        assert "my-plugin-cmd" not in names



    def test_plugin_enumerator_handles_missing_plugin_manager(self, monkeypatch):
        """Enumerators must never raise when plugin discovery raises."""
        from hermes_cli import plugins as _plugins_mod

        def _boom():
            raise RuntimeError("plugin system down")

        monkeypatch.setattr(_plugins_mod, "get_plugin_commands", _boom)

        tg_names = {name for name, _desc in telegram_bot_commands()}
        assert "status" in tg_names
