from unittest.mock import patch

import pytest










def test_code_scoped_stamp_wins_over_home_stamp(tmp_path):
    """The stamp next to the running code is authoritative over $HERMES_HOME.

    Models a host git install whose $HERMES_HOME is shared with (and stamped
    'docker' by) a co-located container. The code-scoped stamp must win so the
    host install is correctly identified as 'git' and 'hermes update' works.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (code / ".install_method").write_text("git\n")
    (home / ".install_method").write_text("docker\n")  # container contamination
    with patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "git"






def test_stamp_install_method_writes_code_scoped(tmp_path):
    """stamp_install_method writes next to the code, not into $HERMES_HOME."""
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    with patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import stamp_install_method
        stamp_install_method("git", project_root=code)
    assert (code / ".install_method").read_text().strip() == "git"
    assert not (home / ".install_method").exists()


def test_container_git_checkout_is_detected_as_git(tmp_path):
    """Container detection does not override a real Git checkout."""
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "git"




