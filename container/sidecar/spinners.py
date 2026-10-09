"""Finding and demoting busy-wait threads inside the game process.

One of the game's own threads (its code is in ``isaac-ng.exe``, not in a
library) spins without ever sleeping under Wine and burns a full CPU core per
game without doing anything the agent needs. No game option stops it, with
or without an audio device, so the sidecar moves any such thread to the
kernel's idle scheduling class and pins it to one core shared by every
container: the threads then only use CPU time nobody else wants, and all
games together waste at most one core.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

_PROC = Path("/proc")
_SAMPLE_S = 1.0
# A spinner never blocks, so it makes (almost) no voluntary context switches,
# however little CPU it gets on a busy machine. Requiring some CPU use as well
# keeps threads that are simply asleep out of the result.
_MAX_VOLUNTARY_SWITCHES = 2
_MIN_CPU_SHARE = 0.1
# Fields of /proc/<pid>/task/<tid>/stat after the "(comm)" part, zero-based.
_UTIME_FIELD = 11
_STIME_FIELD = 12
_VOLUNTARY_SWITCHES_KEY = "voluntary_ctxt_switches:"


@dataclass(frozen=True)
class ThreadSample:
    """Scheduler counters of one thread at one moment.

    Attributes:
        cpu_ticks: CPU time used so far, in kernel clock ticks.
        voluntary_switches: Times the thread gave up the CPU by blocking.
    """

    cpu_ticks: int
    voluntary_switches: int


def find_processes(name: str, proc_root: Path = _PROC) -> list[int]:
    """Return the PIDs of processes whose command name is *name*.

    The kernel truncates command names to 15 characters, so *name* is
    truncated the same way before comparing.
    """
    wanted = name[:15]
    pids: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == wanted:
                pids.append(int(entry.name))
        except OSError:
            # The process exited while /proc was being scanned.
            continue
    return sorted(pids)


def sample_threads(pid: int, proc_root: Path = _PROC) -> dict[int, ThreadSample]:
    """Return the scheduler counters of every thread of *pid*."""
    samples: dict[int, ThreadSample] = {}
    for task in (proc_root / str(pid) / "task").iterdir():
        try:
            stat = (task / "stat").read_text()
            status = (task / "status").read_text()
        except OSError:
            # The thread exited while its directory was being read.
            continue
        # The command name may contain spaces and parentheses; everything
        # after the last ")" is whitespace-separated.
        fields = stat.rsplit(")", 1)[1].split()
        voluntary = next(
            int(line.split()[1])
            for line in status.splitlines()
            if line.startswith(_VOLUNTARY_SWITCHES_KEY)
        )
        samples[int(task.name)] = ThreadSample(
            int(fields[_UTIME_FIELD]) + int(fields[_STIME_FIELD]), voluntary
        )
    return samples


def spinning_threads(
    before: dict[int, ThreadSample],
    after: dict[int, ThreadSample],
    interval_s: float,
    ticks_per_second: int,
    exclude: frozenset[int] = frozenset(),
) -> list[int]:
    """Return threads that ran between two samples without ever blocking.

    A thread doing real work blocks now and then (on a lock, a timer, I/O);
    a busy-wait loop never does. That holds on a saturated machine too, where
    a spinner may get only a fraction of a core.

    Args:
        before: Per-thread counters at the start of the interval.
        after: Per-thread counters at the end of the interval.
        interval_s: Length of the interval in seconds.
        ticks_per_second: Kernel clock ticks per second.
        exclude: Thread IDs never reported (e.g. the game's main thread).
    """
    budget = interval_s * ticks_per_second
    spinners: list[int] = []
    for tid, now in after.items():
        if tid not in before or tid in exclude:
            continue
        earlier = before[tid]
        cpu_share = (now.cpu_ticks - earlier.cpu_ticks) / budget
        blocked = now.voluntary_switches - earlier.voluntary_switches
        if cpu_share >= _MIN_CPU_SHARE and blocked <= _MAX_VOLUNTARY_SWITCHES:
            spinners.append(tid)
    return sorted(spinners)


def demote_thread(tid: int, cpu: int) -> None:
    """Move *tid* to the idle scheduling class and pin it to *cpu*."""
    os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(tid, {cpu})


def demote_spinning_threads(process_name: str) -> list[int]:
    """Demote every busy-wait thread of the game except its main thread.

    Call this while the game is idling in a room: nothing legitimate needs a
    whole core there, so whatever does is a spinner. Samples for about a
    second.

    Args:
        process_name: Command name of the game process.

    Returns:
        The thread IDs that were demoted.
    """
    pids = find_processes(process_name)
    before = {pid: sample_threads(pid) for pid in pids}
    time.sleep(_SAMPLE_S)
    shared_cpu = max(os.sched_getaffinity(0))
    demoted: list[int] = []
    for pid in pids:
        try:
            after = sample_threads(pid)
        except OSError:
            continue
        # The main thread (tid == pid) runs the game loop and is left alone.
        for tid in spinning_threads(
            before[pid],
            after,
            _SAMPLE_S,
            os.sysconf("SC_CLK_TCK"),
            exclude=frozenset({pid}),
        ):
            try:
                if os.sched_getscheduler(tid) == os.SCHED_IDLE:
                    # Demoted by an earlier call; it still looks like a spinner.
                    continue
                demote_thread(tid, shared_cpu)
            except OSError:
                logger.exception("could not demote spinning thread %d", tid)
                continue
            demoted.append(tid)
    logger.info(
        "demoted %d busy-wait thread(s) to idle priority on CPU %d: %s",
        len(demoted),
        shared_cpu,
        demoted,
    )
    return demoted
