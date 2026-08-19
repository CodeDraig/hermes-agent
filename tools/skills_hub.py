"""Private GitHub repository transport for reviewed Hermes skills.

There is deliberately no public catalog or search API here.  A configured
``skills.repository`` (``owner/repo``) is the only remote source and every
operation names one reviewed skill explicitly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from hermes_constants import get_hermes_home


SKILLS_DIR = get_hermes_home() / "skills"
HUB_DIR = SKILLS_DIR / ".repository"
LOCK_PATH = HUB_DIR / "lock.json"
AUDIT_PATH = HUB_DIR / "audit.jsonl"
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillRepositoryError(RuntimeError):
    """A private-repository operation could not complete safely."""


def configured_repository() -> str:
    """Return the required ``owner/repo`` repository setting."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        value = cfg_get(load_config_readonly(), "skills", "repository", default="")
    except Exception as exc:
        raise SkillRepositoryError(f"Could not load skills.repository: {exc}") from exc
    repo = str(value or "").strip().strip("/")
    if len(repo.split("/")) != 2 or any(not part for part in repo.split("/")):
        raise SkillRepositoryError(
            "skills.repository must be configured as a GitHub owner/repo"
        )
    return repo


def validate_skill_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not _SKILL_NAME_RE.fullmatch(candidate):
        raise SkillRepositoryError(
            "Skill name must use lowercase letters, digits, and single hyphens"
        )
    return candidate


class GitHubAuth:
    """Resolve GitHub authentication from Hermes secrets or ``gh``."""

    def __init__(self) -> None:
        self._token: str | None = None

    def token(self) -> str:
        if self._token is not None:
            return self._token
        try:
            from agent.secret_scope import get_secret

            token = get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN")
        except Exception:
            token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not token:
            try:
                completed = subprocess.run(
                    ["gh", "auth", "token"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key not in {"GITHUB_TOKEN", "GH_TOKEN"}
                    },
                )
                if completed.returncode == 0:
                    token = completed.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                token = ""
        if not token:
            raise SkillRepositoryError(
                "GitHub authentication required (GITHUB_TOKEN, GH_TOKEN, or gh auth login)"
            )
        self._token = token
        return token

    def get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def is_authenticated(self) -> bool:
        try:
            self.token()
        except SkillRepositoryError:
            return False
        return True


@dataclass(frozen=True)
class RemoteSkill:
    name: str
    files: dict[str, bytes]
    revision: str

    @property
    def content_hash(self) -> str:
        return bundle_content_hash(self.files)


def bundle_content_hash(files: dict[str, bytes | str] | Path) -> str:
    digest = hashlib.sha256()
    if isinstance(files, Path):
        entries: Iterable[tuple[str, bytes]] = (
            (path.relative_to(files).as_posix(), path.read_bytes())
            for path in sorted(files.rglob("*"))
            if path.is_file()
        )
    else:
        entries = (
            (name, value.encode("utf-8") if isinstance(value, str) else value)
            for name, value in sorted(files.items())
        )
    for name, content in entries:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


class HubLockFile:
    """Small provenance lock for skills installed from the private repository."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write(self, value: dict[str, dict[str, Any]]) -> None:
        from utils import atomic_json_write

        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.path, value, indent=2)

    def get_installed(self, name: str) -> dict[str, Any] | None:
        return self._read().get(name)

    def list_installed(self) -> list[dict[str, Any]]:
        return [dict(name=name, **entry) for name, entry in sorted(self._read().items())]

    def record(self, remote: RemoteSkill, install_path: Path) -> None:
        value = self._read()
        value[remote.name] = {
            "repository": configured_repository(),
            "revision": remote.revision,
            "content_hash": remote.content_hash,
            "install_path": install_path.relative_to(SKILLS_DIR).as_posix(),
        }
        self._write(value)

    def remove(self, name: str) -> None:
        value = self._read()
        if value.pop(name, None) is not None:
            self._write(value)


class PrivateSkillRepository:
    API_ROOT = "https://api.github.com"

    def __init__(
        self,
        repository: str | None = None,
        auth: GitHubAuth | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.repository = repository or configured_repository()
        self.auth = auth or GitHubAuth()
        self.client = client or httpx.Client(timeout=30, follow_redirects=True)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self.client.request(
            method,
            f"{self.API_ROOT}/repos/{self.repository}{path}",
            headers=self.auth.get_headers(),
            **kwargs,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise SkillRepositoryError(
                f"GitHub {method} {path} failed ({response.status_code}): {detail}"
            )
        return response

    def default_branch(self) -> str:
        value = self._request("GET", "").json().get("default_branch")
        if not isinstance(value, str) or not value:
            raise SkillRepositoryError("Configured repository has no default branch")
        return value

    def branch_head(self, branch: str) -> str:
        value = self._request("GET", f"/git/ref/heads/{branch}").json()
        try:
            return str(value["object"]["sha"])
        except (KeyError, TypeError) as exc:
            raise SkillRepositoryError("Could not resolve repository branch head") from exc

    def fetch(self, name: str) -> RemoteSkill:
        name = validate_skill_name(name)
        branch = self.default_branch()
        revision = self.branch_head(branch)
        tree = self._request("GET", f"/git/trees/{revision}?recursive=1").json()
        prefix = f"skills/{name}/"
        entries = [
            entry
            for entry in tree.get("tree", [])
            if isinstance(entry, dict)
            and entry.get("type") == "blob"
            and str(entry.get("path", "")).startswith(prefix)
        ]
        if not entries:
            raise SkillRepositoryError(f"Skill {name!r} was not found in {self.repository}")
        files: dict[str, bytes] = {}
        for entry in entries:
            path = str(entry["path"])
            relative = path[len(prefix) :]
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise SkillRepositoryError(f"Repository returned unsafe skill path: {path}")
            blob = self._request("GET", f"/git/blobs/{entry['sha']}").json()
            if blob.get("encoding") != "base64":
                raise SkillRepositoryError(f"Unsupported blob encoding for {path}")
            files[relative] = base64.b64decode(str(blob.get("content", "")))
        if "SKILL.md" not in files:
            raise SkillRepositoryError(f"Skill {name!r} has no SKILL.md")
        return RemoteSkill(name=name, files=files, revision=revision)

    def publish(self, skill_dir: Path) -> str:
        skill_dir = skill_dir.resolve()
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
            raise SkillRepositoryError(f"No SKILL.md found at {skill_dir}")
        name = validate_skill_name(skill_dir.name)
        branch = self.default_branch()
        base_sha = self.branch_head(branch)
        base_commit = self._request("GET", f"/git/commits/{base_sha}").json()
        base_tree = base_commit.get("tree", {}).get("sha")
        if not base_tree:
            raise SkillRepositoryError("Could not resolve base repository tree")

        tree_entries: list[dict[str, Any]] = []
        local_paths: set[str] = set()
        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(skill_dir).as_posix()
            repository_path = f"skills/{name}/{relative}"
            local_paths.add(repository_path)
            blob = self._request(
                "POST",
                "/git/blobs",
                json={
                    "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "encoding": "base64",
                },
            ).json()
            tree_entries.append(
                {
                    "path": repository_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": str(blob["sha"]),
                }
            )
        base_entries = self._request(
            "GET", f"/git/trees/{base_tree}?recursive=1"
        ).json().get("tree", [])
        prefix = f"skills/{name}/"
        for entry in base_entries:
            repository_path = str(entry.get("path", "")) if isinstance(entry, dict) else ""
            if (
                repository_path.startswith(prefix)
                and entry.get("type") == "blob"
                and repository_path not in local_paths
            ):
                tree_entries.append(
                    {
                        "path": repository_path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": None,
                    }
                )
        new_tree = self._request(
            "POST", "/git/trees", json={"base_tree": base_tree, "tree": tree_entries}
        ).json()["sha"]
        commit = self._request(
            "POST",
            "/git/commits",
            json={
                "message": f"Update skill: {name}",
                "tree": new_tree,
                "parents": [base_sha],
            },
        ).json()["sha"]

        # Re-read immediately before the compare-and-fast-forward update.  A
        # concurrent commit makes our new commit non-fast-forward, so GitHub's
        # ``force: false`` rejects it without changing the branch.
        if self.branch_head(branch) != base_sha:
            raise SkillRepositoryError(
                "Repository changed while publishing; no branch update was attempted"
            )
        self._request(
            "PATCH", f"/git/refs/heads/{branch}", json={"sha": commit, "force": False}
        )
        return str(commit)


def ensure_hub_dirs() -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    HUB_DIR.mkdir(parents=True, exist_ok=True)


def install_remote_skill(remote: RemoteSkill, *, force: bool = False) -> Path:
    ensure_hub_dirs()
    target = SKILLS_DIR / remote.name
    if target.exists() and not force:
        raise SkillRepositoryError(f"Skill {remote.name!r} is already installed")
    backup = None
    if target.exists():
        backup_root = HUB_DIR / "backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"{remote.name}-{remote.revision[:12]}"
        if backup.exists():
            raise SkillRepositoryError(f"Backup already exists: {backup}")
        target.rename(backup)
    staging = Path(tempfile.mkdtemp(prefix=f".{remote.name}-", dir=SKILLS_DIR))
    try:
        for relative, content in remote.files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        staging.rename(target)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and not target.exists():
            backup.rename(target)
        raise
    HubLockFile().record(remote, target)
    append_audit_log("INSTALL", remote.name, remote.revision)
    return target


def check_for_skill_updates(name: str) -> list[dict[str, Any]]:
    name = validate_skill_name(name)
    installed = HubLockFile().get_installed(name)
    if installed is None:
        raise SkillRepositoryError(f"Skill {name!r} is not repository-managed")
    remote = PrivateSkillRepository().fetch(name)
    status = "current" if remote.content_hash == installed.get("content_hash") else "update_available"
    return [{"name": name, "status": status, "revision": remote.revision, "bundle": remote}]


def uninstall_skill(name: str) -> tuple[bool, str]:
    name = validate_skill_name(name)
    installed = HubLockFile().get_installed(name)
    if installed is None:
        return False, f"Skill {name!r} is not repository-managed"
    path = (SKILLS_DIR / str(installed.get("install_path", name))).resolve()
    try:
        path.relative_to(SKILLS_DIR.resolve())
    except ValueError:
        return False, "Recorded install path escapes the skills directory"
    import shutil

    shutil.rmtree(path, ignore_errors=False)
    HubLockFile().remove(name)
    append_audit_log("UNINSTALL", name, "")
    return True, f"Uninstalled {name}"


def append_audit_log(action: str, name: str, detail: str) -> None:
    from datetime import datetime, timezone

    ensure_hub_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "name": name,
        "detail": detail,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
