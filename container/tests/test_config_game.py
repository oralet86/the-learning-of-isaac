"""Tests for sidecar configuration and the game's options file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from container.sidecar.config import SidecarConfig
from container.sidecar.game import options_path, render_options, write_options

if TYPE_CHECKING:
    from pathlib import Path


def test_config_defaults_and_env_overrides() -> None:
    assert SidecarConfig.from_env({}) == SidecarConfig()

    config = SidecarConfig.from_env(
        {"GAME_WIDTH": "320", "GAME_HEIGHT": "180", "SIDECAR_PORT": "6000"}
    )

    assert (config.game_width, config.game_height, config.port) == (320, 180, 6000)


@pytest.mark.parametrize(
    "environ",
    [{"GAME_WIDTH": "0"}, {"GAME_HEIGHT": "-5"}, {"TICK_HZ": "0"}],
)
def test_config_rejects_out_of_range_values(environ: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="positive"):
        SidecarConfig.from_env(environ)


def test_rendered_options_request_the_window_size() -> None:
    lines = render_options(480, 270).splitlines()

    assert lines[0] == "[Options]"
    assert "WindowWidth=480" in lines
    assert "WindowHeight=270" in lines
    assert "Fullscreen=0" in lines
    assert "EnableMods=0" in lines


def test_write_options_creates_parent_directories(tmp_path: Path) -> None:
    path = options_path(tmp_path, "root")

    write_options(path, 320, 180)

    assert path.read_text() == render_options(320, 180)
    assert path.is_relative_to(tmp_path / "drive_c" / "users" / "root")
