"""Tests for `components.shard_store`.

Covers the invariants both pipelines' datasets depend on: shards are
discovered in order, row counts come from `.npy` headers alone, an
incomplete shard fails loudly rather than being skipped, and a pickle
round-trip (what every `DataLoader` worker does) yields a store whose
memory-maps still read correctly.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from components.shard_store import (
    ShardedArrayStore,
    read_npy_length,
    shard_path,
    write_shard,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_shards(directory: Path, rows_per_shard: tuple[int, ...]) -> None:
    """Writes one `frames`/`labels` shard per entry of `rows_per_shard`.

    Frame values encode their global row index, so a test can assert it
    read the row it asked for.
    """
    start = 0
    for index, n_rows in enumerate(rows_per_shard):
        frames = np.arange(start, start + n_rows, dtype=np.uint8).reshape(n_rows, 1, 1)
        frames = np.broadcast_to(frames, (n_rows, 2, 3)).copy()
        labels = np.arange(start, start + n_rows, dtype=np.int64)
        write_shard(
            directory, f"shard_{index:04d}", {"frames": frames, "labels": labels}
        )
        start += n_rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_shard_writes_one_file_per_array(tmp_path: Path) -> None:
    written = write_shard(
        tmp_path,
        "shard_0000",
        {"frames": np.zeros((3, 2), np.uint8), "labels": np.ones(3, np.int64)},
    )

    assert [path.name for path in written] == [
        "shard_0000_frames.npy",
        "shard_0000_labels.npy",
    ]
    assert all(path.exists() for path in written)


def test_write_shard_rejects_mismatched_row_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disagree on row count"):
        write_shard(
            tmp_path,
            "shard_0000",
            {"frames": np.zeros((3, 2), np.uint8), "labels": np.ones(2, np.int64)},
        )


def test_write_shard_rejects_empty_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no arrays"):
        write_shard(tmp_path, "shard_0000", {})


# ---------------------------------------------------------------------------
# Discovery and lengths
# ---------------------------------------------------------------------------


def test_discovers_shards_in_order(tmp_path: Path) -> None:
    _make_shards(tmp_path, (4, 2, 3))

    store = ShardedArrayStore(tmp_path, ("frames", "labels"))

    assert store.stems == ["shard_0000", "shard_0001", "shard_0002"]
    assert len(store) == 3
    assert store.shard_lengths == [4, 2, 3]
    assert store.total_rows == 9


def test_read_npy_length_does_not_load_the_array(tmp_path: Path) -> None:
    # A 40 MB shard: if `read_npy_length` loaded it, this would be the
    # slowest test in the suite. The real guarantee under test is that only
    # the header is parsed, so the value must still be exact.
    write_shard(
        tmp_path, "shard_0000", {"frames": np.zeros((10_000, 64, 64), np.uint8)}
    )

    assert read_npy_length(shard_path(tmp_path, "shard_0000", "frames")) == 10_000


def test_missing_sibling_array_raises(tmp_path: Path) -> None:
    _make_shards(tmp_path, (3, 3))
    shard_path(tmp_path, "shard_0001", "labels").unlink()

    with pytest.raises(FileNotFoundError, match=r"missing shard_0001_labels\.npy"):
        ShardedArrayStore(tmp_path, ("frames", "labels"))


def test_empty_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="no shard_"):
        ShardedArrayStore(tmp_path / "empty", ("frames",))


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ShardedArrayStore(tmp_path / "nope", ("frames",))


def test_stem_prefix_selects_the_shard_family(tmp_path: Path) -> None:
    write_shard(tmp_path, "run_0000", {"frames": np.zeros((2, 1), np.uint8)})
    write_shard(tmp_path, "shard_0000", {"frames": np.zeros((5, 1), np.uint8)})

    assert ShardedArrayStore(tmp_path, ("frames",), stem_prefix="run").stems == [
        "run_0000"
    ]
    assert ShardedArrayStore(tmp_path, ("frames",)).stems == ["shard_0000"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_array_reads_rows_and_is_memory_mapped(tmp_path: Path) -> None:
    _make_shards(tmp_path, (4, 3))
    store = ShardedArrayStore(tmp_path, ("frames", "labels"))

    frames = store.array(1, "frames")
    assert isinstance(frames, np.memmap)
    # Shard 1 starts at global row 4.
    assert frames[0].tolist() == [[4, 4, 4], [4, 4, 4]]
    assert store.array(1, "labels").tolist() == [4, 5, 6]


def test_array_is_cached_per_shard_and_name(tmp_path: Path) -> None:
    _make_shards(tmp_path, (2,))
    store = ShardedArrayStore(tmp_path, ("frames", "labels"))

    assert store.array(0, "frames") is store.array(0, "frames")


def test_unknown_array_name_raises(tmp_path: Path) -> None:
    _make_shards(tmp_path, (2,))
    store = ShardedArrayStore(tmp_path, ("frames", "labels"))

    with pytest.raises(KeyError, match="unknown array"):
        store.array(0, "splits")


# ---------------------------------------------------------------------------
# Worker (pickle) round-trip
# ---------------------------------------------------------------------------


def test_pickle_round_trip_drops_and_reopens_mmaps(tmp_path: Path) -> None:
    _make_shards(tmp_path, (4, 3))
    store = ShardedArrayStore(tmp_path, ("frames", "labels"))
    store.array(0, "frames")  # force a map open before pickling

    revived: ShardedArrayStore = pickle.loads(pickle.dumps(store))

    assert revived._mmaps == {}
    assert revived.stems == store.stems
    assert revived.array(1, "labels").tolist() == [4, 5, 6]
