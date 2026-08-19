"""Explicit Git-backed plugin management for manifest-version-2 plugins."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_cli.config import load_config, save_config
from hermes_constants import get_hermes_home
from utils import atomic_json_write


class PluginOperationError(RuntimeError):
    pass


class PluginScanBlocked(PluginOperationError):
    pass


_OWNER_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _plugins_dir() -> Path:
    path = get_hermes_home() / "plugins"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path() -> Path:
    return _plugins_dir() / ".install-metadata.json"


def _read_metadata() -> dict[str, dict[str, str]]:
    try:
        data = json.loads(_metadata_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_metadata(data: dict[str, dict[str, str]]) -> None:
    atomic_json_write(_metadata_path(), data)


def _git_url(identifier: str) -> str:
    value = identifier.strip()
    if _OWNER_REPO.fullmatch(value):
        return f"https://github.com/{value}.git"
    if value.startswith(("https://", "ssh://", "git@")):
        return value
    raise PluginOperationError(
        "Plugin source must be a Git URL or owner/repo shorthand"
    )


def _git(args: list[str], *, cwd: Optional[Path] = None) -> str:
    git = shutil.which("git")
    if not git:
        raise PluginOperationError("git is required for plugin management")
    result = subprocess.run(
        [git, *args],
        cwd=str(cwd) if cwd else None,
        env=noninteractive_git_env(),
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.returncode:
        raise PluginOperationError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "plugin.yaml"
    if not path.is_file():
        raise PluginOperationError("plugin.yaml is required")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PluginOperationError(f"Invalid plugin.yaml: {exc}") from exc
    if not isinstance(data, dict):
        raise PluginOperationError("plugin.yaml must contain a mapping")
    if data.get("manifest_version") != 2:
        raise PluginOperationError("plugin.yaml manifest_version must be 2")
    name = data.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise PluginOperationError("plugin.yaml name is invalid")
    return data


def _scan(directory: Path, source: str, *, force: bool) -> None:
    from tools.plugin_guard import scan_plugin, should_allow_plugin_install

    result = scan_plugin(directory, source=source)
    allowed, reason = should_allow_plugin_install(result, force=force)
    if allowed is not True:
        raise PluginScanBlocked(f"Security scan blocked plugin install: {reason}")


def _target_for(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise PluginOperationError("Invalid plugin name")
    return _plugins_dir() / name


def cmd_install(
    identifier: str,
    *,
    force: bool = False,
    enable: Optional[bool] = None,
    ref: Optional[str] = None,
) -> None:
    source = _git_url(identifier)
    with tempfile.TemporaryDirectory(prefix="hermes-plugin-") as raw_tmp:
        checkout = Path(raw_tmp) / "checkout"
        _git(["clone", "--quiet", "--", source, str(checkout)])
        if ref:
            _git(["fetch", "--quiet", "origin", ref], cwd=checkout)
            _git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout)
        commit = _git(["rev-parse", "HEAD"], cwd=checkout)
        manifest = _read_manifest(checkout)
        _scan(checkout, source, force=force)
        name = manifest["name"]
        target = _target_for(name)
        if target.exists() and not force:
            raise PluginOperationError(f"Plugin {name!r} is already installed")
        backup = None
        if target.exists():
            backup_root = _plugins_dir() / ".backups"
            backup_root.mkdir(exist_ok=True)
            backup = backup_root / f"{name}-{commit[:12]}"
            if backup.exists():
                raise PluginOperationError(f"Backup target already exists: {backup}")
            target.rename(backup)
        try:
            shutil.copytree(checkout, target, ignore=shutil.ignore_patterns(".git"))
        except BaseException:
            if backup is not None and not target.exists():
                backup.rename(target)
            raise
        metadata = _read_metadata()
        metadata[name] = {"source": source, "commit": commit, "ref": ref or ""}
        _write_metadata(metadata)
        if enable is True:
            cmd_enable(name)
        elif enable is False:
            cmd_disable(name)
        print(f"Installed {name} at {commit[:12]}; activation changes apply next start.")


def cmd_update(name: str) -> None:
    target = _target_for(name)
    if not target.is_dir():
        raise PluginOperationError(f"Plugin {name!r} is not installed")
    meta = _read_metadata().get(name) or {}
    source = meta.get("source")
    if not source:
        raise PluginOperationError("Plugin has no recorded Git source")
    cmd_install(source, force=True, ref=meta.get("ref") or None)


def cmd_remove(name: str) -> None:
    target = _target_for(name)
    if not target.is_dir():
        raise PluginOperationError(f"Plugin {name!r} is not installed")
    backup_root = _plugins_dir() / ".backups"
    backup_root.mkdir(exist_ok=True)
    backup = backup_root / f"removed-{name}"
    if backup.exists():
        raise PluginOperationError(f"Removal backup already exists: {backup}")
    target.rename(backup)
    metadata = _read_metadata()
    metadata.pop(name, None)
    _write_metadata(metadata)
    print(f"Removed {name} to {backup}; change applies next start.")


def _get_enabled_set() -> set[str]:
    value = (load_config().get("plugins") or {}).get("enabled")
    return {str(item) for item in value} if isinstance(value, list) else set()


def _get_disabled_set() -> set[str]:
    value = (load_config().get("plugins") or {}).get("disabled")
    return {str(item) for item in value} if isinstance(value, list) else set()


def _save_enabled_set(enabled: set[str]) -> None:
    config = load_config()
    config.setdefault("plugins", {})["enabled"] = sorted(enabled)
    save_config(config)


def _save_disabled_set(disabled: set[str]) -> None:
    config = load_config()
    config.setdefault("plugins", {})["disabled"] = sorted(disabled)
    save_config(config)


def cmd_enable(name: str) -> None:
    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    enabled.add(name)
    disabled.discard(name)
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    print(f"Enabled {name}; change applies next start.")


def cmd_disable(name: str) -> None:
    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    enabled.discard(name)
    disabled.add(name)
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    print(f"Disabled {name}; change applies next start.")


def _read_manifest_info(directory: Path, prefix: str = ""):
    try:
        data = _read_manifest(directory)
    except PluginOperationError:
        return None
    key = f"{prefix}/{directory.name}" if prefix else directory.name
    return (data["name"], str(data.get("version") or ""), str(data.get("description") or ""), key)


def _discover_all_plugins() -> list[tuple[str, str, str, str, str, str]]:
    from hermes_cli.plugins import get_bundled_plugins_dir

    entries = []
    for root, source in ((get_bundled_plugins_dir(), "bundled"), (_plugins_dir(), "user")):
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.glob("**/plugin.yaml")):
            directory = manifest_path.parent
            info = _read_manifest_info(directory)
            if info:
                name, version, description, key = info
                entries.append((name, version, description, source, str(directory), key))
    return entries


def _plugin_status(name: str, enabled: set[str], disabled: set[str], key: str = "") -> str:
    identity = key or name
    if identity in disabled or name in disabled:
        return "disabled"
    if identity in enabled or name in enabled:
        return "enabled"
    return "available next start"


def cmd_list(args: Any = None) -> None:
    enabled, disabled = _get_enabled_set(), _get_disabled_set()
    for name, version, description, source, _path, key in _discover_all_plugins():
        print(f"{key}\t{version}\t{source}\t{_plugin_status(name, enabled, disabled, key)}\t{description}")


def cmd_show(name: str) -> None:
    for entry in _discover_all_plugins():
        if name in {entry[0], entry[5]}:
            print(json.dumps({"name": entry[0], "version": entry[1], "description": entry[2], "source": entry[3], "path": entry[4], "key": entry[5]}, indent=2))
            return
    raise PluginOperationError(f"Plugin {name!r} was not found")


def cmd_plugin_doctor(target: str = ".", *, ci: bool = False) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    result = doctor_plugin(Path(target))
    print(result)
    if ci and getattr(result, "errors", None):
        raise SystemExit(1)


def plugins_command(args) -> None:
    action = getattr(args, "plugins_action", None)
    if action == "install":
        enabled = True if args.enable else False if args.no_enable else None
        cmd_install(args.identifier, force=args.force, enable=enabled, ref=args.ref)
    elif action == "update":
        cmd_update(args.name)
    elif action in {"remove", "rm", "uninstall"}:
        cmd_remove(args.name)
    elif action in {"list", "ls"}:
        cmd_list(args)
    elif action == "enable":
        cmd_enable(args.name)
    elif action == "disable":
        cmd_disable(args.name)
    elif action in {"show", "info"}:
        cmd_show(args.name)
    elif action == "doctor":
        cmd_plugin_doctor(args.target, ci=args.ci)
    else:
        cmd_list(args)
