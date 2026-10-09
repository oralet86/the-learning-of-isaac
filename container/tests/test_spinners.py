"""Tests for `container.sidecar.spinners`.

The parsing and selection run against a fake ``/proc`` tree; only the two
syscalls that actually change scheduling are left to the real container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from container.sidecar.spinners import (
    ThreadSample,
    find_processes,
    sample_threads,
    spinning_threads,
)

if TYPE_CHECKING:
    from pathlib import Path


def _stat_line(tid: int, comm: str, utime: int, stime: int) -> str:
    """Build a /proc stat line with the CPU times in their real positions."""
    after_comm = ["S", "1", "1", "1", "0", "-1", "0", "0", "0", "0", "0"]
    return f"{tid} ({comm}) " + " ".join([*after_comm, str(utime), str(stime), "0"])


def _add_process(
    proc: Path, pid: int, comm: str, threads: dict[int, tuple[int, int, int]]
) -> None:
    """Create a fake process; each thread is ``(utime, stime, voluntary switches)``."""
    (proc / str(pid) / "task").mkdir(parents=True)
    (proc / str(pid) / "comm").write_text(comm + "\n")
    for tid, (utime, stime, voluntary) in threads.items():
        task = proc / str(pid) / "task" / str(tid)
        task.mkdir()
        (task / "stat").write_text(_stat_line(tid, comm, utime, stime))
        (task / "status").write_text(
            f"Name:\t{comm}\nvoluntary_ctxt_switches:\t{voluntary}\n"
            "nonvoluntary_ctxt_switches:\t7\n"
        )


def test_find_processes_matches_truncated_command_names(tmp_path: Path) -> None:
    _add_process(tmp_path, 26, "isaac-ng.exe", {26: (0, 0, 0)})
    _add_process(tmp_path, 40, "wineserver", {40: (0, 0, 0)})
    _add_process(tmp_path, 77, "a-very-long-pro", {77: (0, 0, 0)})
    (tmp_path / "meminfo").write_text("not a process")

    assert find_processes("isaac-ng.exe", tmp_path) == [26]
    assert find_processes("a-very-long-process-name", tmp_path) == [77]
    assert find_processes("missing", tmp_path) == []


def test_sample_threads_reads_cpu_time_and_voluntary_switches(tmp_path: Path) -> None:
    _add_process(
        tmp_path, 26, "isaac-ng.exe", {26: (100, 20, 900), 147: (7000, 500, 3)}
    )

    assert sample_threads(26, tmp_path) == {
        26: ThreadSample(120, 900),
        147: ThreadSample(7500, 3),
    }


def test_sample_threads_copes_with_parentheses_in_the_name(tmp_path: Path) -> None:
    _add_process(tmp_path, 9, "odd (name) x", {9: (5, 6, 1)})

    assert sample_threads(9, tmp_path) == {9: ThreadSample(11, 1)}


def test_spinner_is_a_thread_that_runs_without_ever_blocking() -> None:
    before = {
        147: ThreadSample(5000, 3),  # busy-wait: lots of CPU, never blocks
        150: ThreadSample(10, 40),  # asleep: no CPU at all
        151: ThreadSample(200, 100),  # real work: CPU, but blocks regularly
    }
    after = {
        147: ThreadSample(5099, 3),
        150: ThreadSample(10, 40),
        151: ThreadSample(290, 160),
        999: ThreadSample(500, 0),  # appeared mid-sample: no baseline
    }

    assert spinning_threads(before, after, 1.0, 100) == [147]


def test_spinner_is_still_found_when_it_only_gets_part_of_a_core() -> None:
    # On a saturated machine a busy-wait thread is preempted constantly and
    # may get a fraction of a core, but it still never blocks voluntarily.
    before = {147: ThreadSample(5000, 3)}
    after = {147: ThreadSample(5030, 3)}

    assert spinning_threads(before, after, 1.0, 100) == [147]


def test_spinning_threads_never_reports_excluded_threads() -> None:
    before = {26: ThreadSample(0, 0), 147: ThreadSample(0, 0)}
    after = {26: ThreadSample(100, 0), 147: ThreadSample(100, 0)}

    spinners = spinning_threads(before, after, 1.0, 100, exclude=frozenset({26}))

    assert spinners == [147]
