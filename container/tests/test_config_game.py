"""Tests for sidecar configuration and the game's options file."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pytest

from constants import STEP_HZ
from container.build import APP_DIR, build_command
from container.sidecar.config import (
    DEFAULT_GAME_DIR,
    DEFAULT_SAVE_SOURCE_DIR,
    SidecarConfig,
)
from container.sidecar.display import screen_size
from container.sidecar.game import (
    clear_run_state,
    install_save,
    launch_spec,
    options_path,
    render_options,
    write_options,
)
from container.sidecar.protocol import DEFAULT_PORT

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
    [{"GAME_WIDTH": "0"}, {"GAME_HEIGHT": "-5"}],
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


def test_config_control_rate_is_the_shared_constant() -> None:
    assert SidecarConfig.from_env({"TICK_HZ": "60"}).tick_hz == STEP_HZ


def test_install_save_copies_the_single_save_into_slot_one(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "persistentgamedata_8e097dd2.dat").write_bytes(b"unlocked")
    target = tmp_path / "saves"
    target.mkdir()
    (target / "persistentgamedata1.dat").write_bytes(b"changed by earlier runs")

    install_save(source, target)

    assert (target / "persistentgamedata1.dat").read_bytes() == b"unlocked"


@pytest.mark.parametrize("names", [[], ["a.dat", "b.dat"]])
def test_install_save_requires_exactly_one_save(
    tmp_path: Path, names: list[str]
) -> None:
    for name in names:
        (tmp_path / name).write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="exactly one"):
        install_save(tmp_path, tmp_path / "saves")


def test_clear_run_state_removes_only_in_progress_runs(tmp_path: Path) -> None:
    for name in ("gamestate1.dat", "gamestate2.dat", "persistentgamedata1.dat"):
        (tmp_path / name).write_bytes(b"x")

    clear_run_state(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["persistentgamedata1.dat"]


def test_screen_is_larger_than_the_game_window_and_never_tiny() -> None:
    small = screen_size(240, 135)
    large = screen_size(1920, 1080)

    assert small == (1024, 640)
    assert large[0] > 1920
    assert large[1] > 1080


def test_build_command_passes_the_python_constants_to_the_image() -> None:
    command = build_command("example:tag", "podman")

    assert command[:4] == ["podman", "build", "--tag", "example:tag"]
    assert command[-1] == "."
    build_args = {
        arg.split("=", 1)[0]: arg.split("=", 1)[1]
        for flag, arg in pairwise(command)
        if flag == "--build-arg"
    }
    assert build_args == {
        "APP_DIR": str(APP_DIR),
        "GAME_DIR": str(DEFAULT_GAME_DIR),
        "SAVE_SOURCE_DIR": str(DEFAULT_SAVE_SOURCE_DIR),
        "SIDECAR_PORT": str(DEFAULT_PORT),
    }


def test_launch_spec_renders_in_software_without_a_gpu() -> None:
    command, env = launch_spec(None)

    assert command == ["wine", "isaac-ng.exe"]
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_launch_spec_uses_virtualgl_with_a_gpu(tmp_path: Path) -> None:
    command, env = launch_spec(tmp_path / "renderD129")

    assert command[:3] == ["vglrun", "-d", "egl"]
    assert command[-2:] == ["wine", "isaac-ng.exe"]
    assert "LIBGL_ALWAYS_SOFTWARE" not in env


def test_config_gpu_device_is_optional() -> None:
    assert SidecarConfig.from_env({}).gpu_device is None
    assert SidecarConfig.from_env({"GPU_DEVICE": ""}).gpu_device is None

    config = SidecarConfig.from_env({"GPU_DEVICE": "/dev/dri/renderD129"})

    assert str(config.gpu_device) == "/dev/dri/renderD129"


def test_config_rejects_idle_limit_shorter_than_a_resume() -> None:
    # Such a limit makes games stepped in lockstep pause each other forever.
    with pytest.raises(ValueError, match="IDLE_PAUSE_S"):
        SidecarConfig.from_env({"IDLE_PAUSE_S": "0.25"})

    assert SidecarConfig.from_env({"IDLE_PAUSE_S": "0"}).idle_pause_s == 0.0
