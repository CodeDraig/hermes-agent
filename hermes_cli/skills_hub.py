"""CLI and slash-command surface for the private reviewed-skill repository."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console

from tools.skills_hub import (
    HubLockFile,
    PrivateSkillRepository,
    SKILLS_DIR,
    SkillRepositoryError,
    check_for_skill_updates,
    configured_repository,
    install_remote_skill,
    uninstall_skill,
    validate_skill_name,
)


_console = Console()


def _scan_remote(remote, *, force: bool = False) -> None:
    from tools.skills_guard import format_scan_report, scan_skill, should_allow_install

    with tempfile.TemporaryDirectory(prefix="hermes-skill-review-") as tmp:
        root = Path(tmp) / remote.name
        root.mkdir()
        for relative, content in remote.files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        result = scan_skill(root, source=f"private:{configured_repository()}")
        allowed, reason = should_allow_install(result, force=force)
        if not allowed:
            raise SkillRepositoryError(
                f"Skill scan blocked {remote.name}: {reason}\n{format_scan_report(result)}"
            )


def do_inspect(name: str, *, console: Console | None = None) -> dict[str, Any]:
    remote = PrivateSkillRepository().fetch(validate_skill_name(name))
    text = remote.files["SKILL.md"].decode("utf-8", errors="replace")
    (console or _console).print(text)
    return {"name": remote.name, "revision": remote.revision, "content": text}


def do_install(
    name: str,
    *,
    force: bool = False,
    yes: bool = False,
    console: Console | None = None,
) -> Path:
    del yes  # explicit CLI operation is the authorization boundary
    remote = PrivateSkillRepository().fetch(validate_skill_name(name))
    _scan_remote(remote, force=force)
    path = install_remote_skill(remote, force=force)
    (console or _console).print(f"[bold green]Installed:[/] {path}")
    return path


def do_check(name: str, *, console: Console | None = None) -> list[dict[str, Any]]:
    results = check_for_skill_updates(validate_skill_name(name))
    display = [{key: value for key, value in row.items() if key != "bundle"} for row in results]
    (console or _console).print_json(json.dumps(display))
    return results


def do_update(
    name: str,
    *,
    force: bool = False,
    console: Console | None = None,
) -> Path | None:
    result = check_for_skill_updates(validate_skill_name(name))[0]
    if result["status"] == "current":
        (console or _console).print(f"[dim]{name} is current.[/]")
        return None
    remote = result["bundle"]
    installed = HubLockFile().get_installed(name) or {}
    local = SKILLS_DIR / str(installed.get("install_path", name))
    recorded_hash = str(installed.get("content_hash", ""))
    from tools.skills_hub import bundle_content_hash

    if local.exists() and recorded_hash and bundle_content_hash(local) != recorded_hash and not force:
        raise SkillRepositoryError(
            f"{name} has local edits; pass --force to replace the reviewed copy"
        )
    _scan_remote(remote, force=force)
    path = install_remote_skill(remote, force=True)
    (console or _console).print(f"[bold green]Updated:[/] {path}")
    return path


def do_publish(skill_path: str, *, console: Console | None = None) -> str:
    path = Path(skill_path)
    if not path.is_absolute():
        path = SKILLS_DIR / path
    path = path.resolve()
    from tools.skills_guard import format_scan_report, scan_skill

    result = scan_skill(path, source="self")
    if result.verdict == "dangerous":
        raise SkillRepositoryError(
            f"Skill scan blocked publication\n{format_scan_report(result)}"
        )
    commit = PrivateSkillRepository().publish(path)
    (console or _console).print(
        f"[bold green]Published {path.name} to {configured_repository()} at {commit}[/]"
    )
    return commit


def do_list(*, console: Console | None = None) -> list[dict[str, Any]]:
    rows = HubLockFile().list_installed()
    (console or _console).print_json(json.dumps(rows))
    return rows


def do_audit(name: str, *, deep: bool = False, console: Console | None = None) -> Any:
    name = validate_skill_name(name)
    path = SKILLS_DIR / name
    from tools.skills_guard import format_scan_report, scan_skill

    del deep
    result = scan_skill(path, source="local")
    (console or _console).print(format_scan_report(result))
    return result


def skills_command(args) -> Any:
    action = getattr(args, "skills_action", None)
    try:
        if action == "inspect":
            return do_inspect(args.name)
        if action == "install":
            return do_install(args.name, force=args.force, yes=args.yes)
        if action == "check":
            return do_check(args.name)
        if action == "update":
            return do_update(args.name, force=args.force)
        if action == "publish":
            return do_publish(args.skill_path)
        if action == "list":
            return do_list()
        if action == "audit":
            return do_audit(args.name, deep=args.deep)
        if action == "uninstall":
            ok, message = uninstall_skill(args.name)
            _console.print(message)
            return ok
        if action == "config":
            from hermes_cli.skills_config import configure_skills

            return configure_skills()
        if action in {"reset", "modified", "diff", "opt-out", "opt-in"}:
            return _bundled_skill_command(action, args)
        raise SkillRepositoryError("Specify a skills action")
    except SkillRepositoryError as exc:
        _console.print(f"[bold red]Error:[/] {exc}")
        return False


def _bundled_skill_command(action: str, args) -> Any:
    from tools import skills_sync

    if action == "reset":
        return skills_sync.reset_bundled_skill(args.name, restore=args.restore)
    if action == "modified":
        return skills_sync.list_user_modified_bundled_skills()
    if action == "diff":
        return skills_sync.diff_bundled_skill(args.name)
    if action == "opt-out":
        result = skills_sync.set_bundled_skills_opt_out(True)
        if args.remove:
            skills_sync.remove_pristine_bundled_skills()
        return result
    if action == "opt-in":
        result = skills_sync.set_bundled_skills_opt_out(False)
        if args.sync:
            skills_sync.sync_skills()
        return result
    raise SkillRepositoryError(f"Unknown bundled skill action: {action}")


def handle_skills_slash(argument: str, **_: Any) -> str:
    """Handle explicit repository operations from ``/skills``."""
    parts = argument.strip().split()
    if not parts:
        return "Usage: /skills <inspect|install|check|update|list> [skill-name]"
    action = parts[0].lower()
    name = parts[1] if len(parts) > 1 else ""
    try:
        if action == "list":
            rows = HubLockFile().list_installed()
            return "\n".join(row["name"] for row in rows) or "No repository-managed skills."
        if not name:
            raise SkillRepositoryError(f"/skills {action} requires a skill name")
        if action == "inspect":
            return PrivateSkillRepository().fetch(name).files["SKILL.md"].decode(
                "utf-8", errors="replace"
            )
        if action == "install":
            remote = PrivateSkillRepository().fetch(name)
            _scan_remote(remote)
            return f"Installed {install_remote_skill(remote)}"
        if action == "check":
            return str(check_for_skill_updates(name)[0]["status"])
        if action == "update":
            path = do_update(name)
            return f"Updated {path}" if path else f"{name} is current"
        raise SkillRepositoryError(f"Unknown /skills action: {action}")
    except SkillRepositoryError as exc:
        return f"Error: {exc}"
