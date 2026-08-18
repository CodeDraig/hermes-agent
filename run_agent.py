#!/usr/bin/env python3
"""Executable CLI entry point. Agent runtime APIs live under ``agent``."""


def main(*args, **kwargs):
    from agent.agent_cli import main as run

    return run(*args, **kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
