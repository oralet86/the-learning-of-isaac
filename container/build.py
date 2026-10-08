#!/usr/bin/env python3
"""Build the game container image.

The image needs a few values that the Python code also relies on (where the
game and save live, which port the sidecar listens on). They are defined once
in Python and passed to the Containerfile as build arguments here, so the two
cannot drift apart.

Usage::

    python -m container.build
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from container.sidecar.config import DEFAULT_GAME_DIR, DEFAULT_SAVE_SOURCE_DIR
from container.sidecar.protocol import DEFAULT_PORT

DEFAULT_IMAGE = "localhost/tloi-isaac:dev"
DEFAULT_RUNTIME = "podman"

# Where the sidecar sources (and constants.py) live inside the image.
APP_DIR = Path("/opt/app")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTAINERFILE = Path("container", "Containerfile")


def build_command(image: str, runtime: str) -> list[str]:
    """Return the container CLI invocation that builds *image*.

    The build context is the repository root (the image needs ``constants.py``
    as well as ``container/``); ``.containerignore`` there keeps everything
    else out of it.

    Args:
        image: Tag for the built image.
        runtime: Container CLI to invoke.
    """
    build_args = {
        "APP_DIR": APP_DIR,
        "GAME_DIR": DEFAULT_GAME_DIR,
        "SAVE_SOURCE_DIR": DEFAULT_SAVE_SOURCE_DIR,
        "SIDECAR_PORT": DEFAULT_PORT,
    }
    command = [runtime, "build", "--tag", image, "--file", str(_CONTAINERFILE)]
    for name, value in build_args.items():
        command += ["--build-arg", f"{name}={value}"]
    return [*command, "."]


def main() -> None:
    """Build the image, failing loudly if the build fails."""
    parser = argparse.ArgumentParser(description="Build the game container image.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    subprocess.run(build_command(args.image, args.runtime), check=True, cwd=_REPO_ROOT)


if __name__ == "__main__":
    main()
