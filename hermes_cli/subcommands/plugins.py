"""Parser for explicit Git-backed plugin management."""

from __future__ import annotations

from typing import Callable


def build_plugins_parser(subparsers, *, cmd_plugins: Callable) -> None:
    parser = subparsers.add_parser(
        "plugins",
        help="Manage manifest-version-2 plugins from explicit Git sources",
    )
    actions = parser.add_subparsers(dest="plugins_action")

    install = actions.add_parser("install", help="Install from a Git URL or owner/repo")
    install.add_argument("identifier")
    install.add_argument("--force", "-f", action="store_true")
    install.add_argument("--ref", help="Fetch and install this Git ref as a detached commit")
    activation = install.add_mutually_exclusive_group()
    activation.add_argument("--enable", action="store_true")
    activation.add_argument("--no-enable", action="store_true")

    update = actions.add_parser("update", help="Update from the recorded Git source")
    update.add_argument("name")

    remove = actions.add_parser("remove", aliases=["rm", "uninstall"])
    remove.add_argument("name")

    actions.add_parser("list", aliases=["ls"])

    enable = actions.add_parser("enable", help="Enable on the next process start")
    enable.add_argument("name")
    disable = actions.add_parser("disable", help="Disable on the next process start")
    disable.add_argument("name")

    show = actions.add_parser("show", aliases=["info"])
    show.add_argument("name")

    doctor = actions.add_parser("doctor", help="Validate a manifest-version-2 plugin")
    doctor.add_argument("target", nargs="?", default=".")
    doctor.add_argument("--ci", action="store_true")

    parser.set_defaults(func=cmd_plugins)
