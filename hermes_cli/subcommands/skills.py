"""Parser for explicit private-repository and local skill operations."""

from __future__ import annotations

from typing import Callable


def build_skills_parser(subparsers, *, cmd_skills: Callable) -> None:
    parser = subparsers.add_parser(
        "skills",
        help="Manage reviewed skills",
        description=(
            "Inspect, install, update, and publish named skills through the "
            "private GitHub repository configured at skills.repository."
        ),
    )
    actions = parser.add_subparsers(dest="skills_action")

    inspect = actions.add_parser("inspect", help="Preview one reviewed skill")
    inspect.add_argument("name")

    install = actions.add_parser("install", help="Install one reviewed skill")
    install.add_argument("name")
    install.add_argument("--force", action="store_true")
    install.add_argument("--yes", "-y", action="store_true")

    check = actions.add_parser("check", help="Check one installed skill")
    check.add_argument("name")

    update = actions.add_parser("update", help="Update one installed skill")
    update.add_argument("name")
    update.add_argument("--force", action="store_true")

    publish = actions.add_parser("publish", help="Publish one local skill")
    publish.add_argument("skill_path")

    actions.add_parser("list", help="List repository-managed skills")
    audit = actions.add_parser("audit", help="Audit one installed skill")
    audit.add_argument("name")
    audit.add_argument("--deep", action="store_true")
    uninstall = actions.add_parser("uninstall", help="Uninstall one reviewed skill")
    uninstall.add_argument("name")

    reset = actions.add_parser("reset", help="Reset bundled-skill tracking")
    reset.add_argument("name")
    reset.add_argument("--restore", action="store_true")
    actions.add_parser("modified", help="List modified bundled skills")
    diff = actions.add_parser("diff", help="Diff one bundled skill")
    diff.add_argument("name")
    opt_out = actions.add_parser("opt-out", help="Disable bundled-skill seeding")
    opt_out.add_argument("--remove", action="store_true")
    opt_in = actions.add_parser("opt-in", help="Enable bundled-skill seeding")
    opt_in.add_argument("--sync", action="store_true")
    actions.add_parser("config", help="Configure local skill enablement")
    parser.set_defaults(func=cmd_skills)
