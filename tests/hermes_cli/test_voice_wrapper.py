"""Tests for prompt-toolkit voice shortcut configuration helpers."""

class TestNormalizeVoiceRecordKeyForPromptToolkit:
    """Voice shortcut spellings normalize to prompt_toolkit key names."""



    def test_non_string_falls_back_to_default(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit(None) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit(1) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit(True) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit({}) == "c-b"


    def test_super_win_fall_back_to_default_in_cli(self):
        """prompt_toolkit has no super modifier, so ``super+b`` / ``win+o``
        would crash the classic CLI at startup if passed through. Fall
        back to the documented default; the CLI binding site is
        expected to warn before falling back."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("super+b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("win+o") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("windows+o") == "c-b"

    # Round-10 Copilot review regressions on #19835.




    def test_pt_key_to_sequence(self):
        from hermes_cli.voice import pt_key_to_sequence

        assert pt_key_to_sequence("c-b") == ("c-b",)
        assert pt_key_to_sequence("a-v") == ("escape", "v")
        assert pt_key_to_sequence("a-space") == ("escape", "space")


class TestVoiceRecordKeyFromConfig:
    """Round-11 Copilot review regression on #19835.

    ``load_config()`` preserves YAML scalar overrides, so a hand-edited
    ``voice: true`` or ``voice: cmd+b`` made the naive
    ``cfg.get('voice', {}).get('record_key')`` chain raise
    AttributeError before voice could run. The shape-safe extractor
    returns None for every malformed shape so the call-site fallback
    (``normalize_…`` / ``format_…``) surfaces the documented default.
    """




    def test_missing_record_key_returns_none(self):
        from hermes_cli.voice import voice_record_key_from_config

        assert voice_record_key_from_config({"voice": {"beep_enabled": True}}) is None
        assert voice_record_key_from_config({}) is None

    def test_normalizer_accepts_extractor_output_directly(self):
        """voice_record_key_from_config + normalize_… must compose —
        None / non-string scalars all fall back to c-b."""
        from hermes_cli.voice import (
            normalize_voice_record_key_for_prompt_toolkit,
            voice_record_key_from_config,
        )

        for raw in (None, True, 1, "cmd+b", ["ctrl+b"]):
            extracted = voice_record_key_from_config({"voice": raw})
            assert normalize_voice_record_key_for_prompt_toolkit(extracted) == "c-b"


class TestFormatVoiceRecordKeyForStatus:
    """Round-10 Copilot review regression on #19835.

    ``/voice status`` used to print the raw scalar (``True`` / ``1``)
    for non-string configs even though the actual binding falls back
    to Ctrl+B. The formatter routes through the same normalizer so
    status always matches what the CLI actually binds.
    """

    def test_ctrl_and_alt_letter_keys_render_canonically(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("ctrl+b") == "Ctrl+B"
        assert format_voice_record_key_for_status("ctrl+o") == "Ctrl+O"
        assert format_voice_record_key_for_status("alt+r") == "Alt+R"



    def test_non_string_scalar_falls_back_to_ctrl_b_label(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        # Copilot round-10 regression: previously /voice status printed
        # the raw scalar ("True" / "1") even though the actual binding
        # fell back to Ctrl+B.
        assert format_voice_record_key_for_status(True) == "Ctrl+B"
        assert format_voice_record_key_for_status(1) == "Ctrl+B"
        assert format_voice_record_key_for_status(None) == "Ctrl+B"
        assert format_voice_record_key_for_status({}) == "Ctrl+B"

    def test_malformed_configs_fall_back_to_ctrl_b(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("ctrl+spcae") == "Ctrl+B"
        assert format_voice_record_key_for_status("ctrl+alt+r") == "Ctrl+B"
        assert format_voice_record_key_for_status("") == "Ctrl+B"
        assert format_voice_record_key_for_status("  ") == "Ctrl+B"

