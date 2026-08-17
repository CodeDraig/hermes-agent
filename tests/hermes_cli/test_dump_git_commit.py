"""Tests for hermes_cli.dump._get_git_commit — git SHA resolution for ``hermes dump``.

``hermes dump`` prints the running commit so support bug reports identify the
exact version. Source installs resolve it live via ``git rev-parse``.
"""

from unittest.mock import MagicMock, patch


def test_get_git_commit_uses_live_git_when_available(tmp_path):
    """Source install: ``git rev-parse --short=8 HEAD`` wins; no fallback."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    git_result = MagicMock(returncode=0, stdout="deadbeef\n")
    with patch("hermes_cli.dump.subprocess.run", return_value=git_result) as mock_run:
        commit = dump._get_git_commit(repo_dir)

    assert commit == "deadbeef"
    mock_run.assert_called_once()


def test_get_git_commit_date_empty_when_git_fails(tmp_path):
    """No git checkout means the dump line drops the date."""
    from hermes_cli import dump

    repo_dir = tmp_path / "no-git-here"
    repo_dir.mkdir()

    failed = MagicMock(returncode=128, stdout="")
    with patch("hermes_cli.dump.subprocess.run", return_value=failed):
        date = dump._get_git_commit_date(repo_dir)

    assert date == ""

