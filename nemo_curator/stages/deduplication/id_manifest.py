# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic, coordinator-free dedup ID assignment backed by a Lance dataset.

Replaces the Ray-actor ``IdGenerator`` (``id_generator.py``) for pipelines
that must be splittable across a Slurm array, where each array task
typically runs its own separate Ray cluster and can't share a single named
actor. Instead of assigning IDs via RPC to a live coordinator, every ID is
computed once, up front, from a file's position in a canonical manifest:
``id = file_idx << ROW_INDEX_BITS | row_idx``.

The manifest is a Lance dataset with one row per source file, written as
its own fragment (single-row append). Lance already derives
``fragment_id = row_address >> 32`` from its native stable row addressing
(see ``nemo_curator/utils/lance.py``), i.e. Lance's own row-addressing
convention already matches the bit layout used here. Reusing it directly
means ``file_idx`` is literally Lance's own ``fragment_id`` -- assignment,
ordering, and uniqueness on append come from Lance's commit protocol, not
code written here.

Note: a fragment's own ``count_rows()`` is always 1 in this dataset (one
manifest row per fragment) -- it does not describe the source file's row
count. The source file's actual row count is the ``row_count`` column,
used explicitly to compute the file's id range.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.utils.atomic_io import write_json_atomically, write_json_atomically_if_absent
from nemo_curator.utils.file_utils import get_fs
from nemo_curator.utils.lance import LANCE_FRAGID_COLUMN, add_lance_metadata_columns
from nemo_curator.utils.retry_manifest import CompletionManifest, read_completion_manifests

ROW_INDEX_BITS = 32
ROW_INDEX_MASK = (1 << ROW_INDEX_BITS) - 1

# Task metadata key stages use to pass the manifest location downstream.
ID_MANIFEST_DIR_METADATA_KEY = "id_manifest_dir"

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("row_count", pa.int64()),
        pa.field("file_size", pa.int64()),
        pa.field("content_hash", pa.string()),
    ]
)

_PARTS_DIRNAME = ".parts"
_PARTS_COMPLETION_NAMESPACE = "id_manifest_parts"
_LOCK_FILENAME = ".lock"

_ROW_COUNTERS = {
    ".parquet": lambda fs, path: pq.ParquetFile(fs.open(path, "rb")).metadata.num_rows,
    ".jsonl": lambda fs, path: sum(1 for _ in fs.open(path, "rb")),
    ".json": lambda fs, path: sum(1 for _ in fs.open(path, "rb")),
}


def encode_id_range(file_idx: int, row_count: int) -> tuple[int, int]:
    """Compute the inclusive ``[id_start, id_end]`` range for a file.

    ``id_end < id_start`` for a zero-row file (empty range), by design.
    """
    if file_idx < 0:
        msg = f"file_idx must be non-negative, got {file_idx}"
        raise ValueError(msg)
    if row_count < 0:
        msg = f"row_count must be non-negative, got {row_count}"
        raise ValueError(msg)
    id_start = file_idx << ROW_INDEX_BITS
    id_end = id_start + row_count - 1
    return id_start, id_end


def decode_id(id_value: int) -> tuple[int, int]:
    """Recover ``(file_idx, row_idx)`` from a previously assigned id."""
    if id_value < 0:
        msg = f"id_value must be non-negative, got {id_value}"
        raise ValueError(msg)
    return id_value >> ROW_INDEX_BITS, id_value & ROW_INDEX_MASK


@dataclass(frozen=True)
class ManifestFileEntry:
    """One source file's manifest entry, with its assigned id range."""

    path: str
    row_count: int
    file_size: int
    content_hash: str
    file_idx: int

    @property
    def id_start(self) -> int:
        return encode_id_range(self.file_idx, self.row_count)[0]

    @property
    def id_end(self) -> int:
        return encode_id_range(self.file_idx, self.row_count)[1]


def _content_fingerprint(fs: Any, path: str, file_size: int) -> str:  # noqa: ANN401
    """Cheap fingerprint (size + mtime), not a full-file hash.

    Detects most accidental changes (edits, truncation, regeneration)
    without paying the cost of hashing file contents at scale. Not a
    defense against adversarial tampering.
    """
    info = fs.info(path)
    mtime = info.get("mtime") or info.get("LastModified") or info.get("last_modified")
    digest_input = f"{path}:{file_size}:{mtime}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]


def scan_file_entry(path: str, storage_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Scan one source file for its manifest fields (no full read)."""
    fs = get_fs(path, storage_options)
    suffix = Path(path).suffix.lower()
    counter = _ROW_COUNTERS.get(suffix)
    if counter is None:
        msg = f"Cannot determine row count for file with extension {suffix!r}: {path}"
        raise ValueError(msg)
    file_size = fs.info(path)["size"]
    row_count = counter(fs, path)
    content_hash = _content_fingerprint(fs, path, file_size)
    return {"path": path, "row_count": row_count, "file_size": file_size, "content_hash": content_hash}


class IdManifest:
    """Deterministic, Lance-backed dedup-id manifest.

    See module docstring for the ``file_idx == fragment_id`` design and its
    concurrency constraints.
    """

    def __init__(self, manifest_dir: str, storage_options: dict[str, Any] | None = None) -> None:
        self.manifest_dir = str(manifest_dir)
        self.storage_options = storage_options or {}

    def _open(self) -> lance.LanceDataset:
        return lance.dataset(self.manifest_dir, storage_options=self.storage_options or None)

    def _dataset_exists(self) -> bool:
        try:
            self._open()
        except (ValueError, OSError, FileNotFoundError):
            return False
        return True

    @classmethod
    def from_disk(cls, manifest_dir: str, storage_options: dict[str, Any] | None = None) -> IdManifest:
        manifest = cls(manifest_dir, storage_options)
        manifest._open()  # noqa: SLF001 - raises if the dataset doesn't exist yet
        return manifest

    def _parts_dir(self) -> Path:
        return Path(self.manifest_dir) / _PARTS_DIRNAME

    def _lock_path(self) -> Path:
        return Path(self.manifest_dir) / _LOCK_FILENAME

    class _WriterLock:
        """Guards against two concurrent finalize()/extend() calls racing.

        This is not the completeness signal for sharded generation (see
        ``finalize``'s ``total_shards`` check) -- it only prevents two
        invocations of this process from interleaving fragment appends.
        """

        def __init__(self, manifest: IdManifest) -> None:
            self._manifest = manifest

        def __enter__(self) -> IdManifest._WriterLock:
            lock_path = self._manifest._lock_path()  # noqa: SLF001
            if not write_json_atomically_if_absent(lock_path, {"status": "locked"}):
                msg = (
                    f"Another finalize()/extend() is already in progress for manifest "
                    f"{self._manifest.manifest_dir} (lock file {lock_path} exists)"
                )
                raise RuntimeError(msg)
            return self

        def __exit__(self, *_exc_info: object) -> bool:
            self._manifest._lock_path().unlink(missing_ok=True)  # noqa: SLF001
            return False

    @classmethod
    def build(
        cls,
        file_paths: list[str],
        manifest_dir: str,
        storage_options: dict[str, Any] | None = None,
    ) -> IdManifest:
        """Convenience wrapper: scan + finalize in one call, single process."""
        manifest = cls(manifest_dir, storage_options)
        manifest.build_part(file_paths, part_id="0", total_shards=1, shard_index=0)
        return manifest.finalize(total_shards=1)

    def build_part(
        self,
        file_paths: list[str],
        part_id: str,
        total_shards: int | None = None,
        shard_index: int | None = None,
    ) -> Path:
        """Scan ``file_paths`` and record an unordered partial contribution.

        Safe for many concurrent callers (e.g. one per Slurm array task):
        each writes its own uniquely-named part file, no shared state. Does
        not assign fragment ids -- that only happens in ``finalize``.
        """
        entries = [scan_file_entry(path, self.storage_options) for path in file_paths]
        part_path = self._parts_dir() / f"part_{part_id}.json"
        write_json_atomically(part_path, {"entries": entries}, sort_keys=True)

        identity: dict[str, object] = {"part_id": part_id}
        if total_shards is not None:
            identity["total_shards"] = total_shards
            identity["shard_index"] = shard_index
        CompletionManifest(
            checkpoint_path=self.manifest_dir,
            namespace=_PARTS_COMPLETION_NAMESPACE,
            identity=identity,
        ).mark_completed({"part_path": str(part_path)})
        return part_path

    def finalize(self, total_shards: int | None = None) -> IdManifest:
        """Commit all pending ``build_part`` contributions to the manifest.

        Single-writer only (see ``_WriterLock``). If ``total_shards`` is
        given, refuses to run unless a completion record exists for every
        expected ``shard_index`` in ``range(total_shards)`` -- this makes
        completeness verifiable rather than assumed (no array task can
        know on its own whether sibling tasks have finished).
        """
        with self._WriterLock(self):
            if total_shards is not None:
                records = read_completion_manifests(self.manifest_dir, namespace=_PARTS_COMPLETION_NAMESPACE)
                completed_shard_indices = {
                    record.payload["shard_index"] for record in records if "shard_index" in record.payload
                }
                missing = sorted(set(range(total_shards)) - completed_shard_indices)
                if missing:
                    msg = (
                        f"Cannot finalize manifest {self.manifest_dir}: missing build_part "
                        f"contributions for shard indices {missing} (of {total_shards} expected)"
                    )
                    raise ValueError(msg)

            all_entries = self._read_all_parts()
            existing = self._existing_entries()
            new_entries = self._resolve_new_entries(all_entries, existing)
            self._append_entries(new_entries)
        return self

    def extend(self, new_file_paths: list[str]) -> IdManifest:
        """Add new files to an existing manifest without disturbing prior ranges."""
        part_id = f"extend-{uuid.uuid4().hex}"
        self.build_part(new_file_paths, part_id=part_id)
        return self.finalize()

    def _read_all_parts(self) -> dict[str, dict[str, Any]]:
        all_entries: dict[str, dict[str, Any]] = {}
        for part_file in sorted(self._parts_dir().glob("part_*.json")):
            payload = json.loads(part_file.read_text())
            for entry in payload["entries"]:
                prior = all_entries.get(entry["path"])
                if prior is not None and (prior["row_count"], prior["content_hash"]) != (
                    entry["row_count"],
                    entry["content_hash"],
                ):
                    msg = f"Conflicting scan results for {entry['path']!r} across build_part contributions"
                    raise ValueError(msg)
                all_entries[entry["path"]] = entry
        return all_entries

    def _existing_entries(self) -> dict[str, dict[str, Any]]:
        if not self._dataset_exists():
            return {}
        table = self._open().to_table(columns=["path", "row_count", "content_hash"])
        return {row["path"]: row for row in table.to_pylist()}

    def _resolve_new_entries(
        self,
        all_entries: dict[str, dict[str, Any]],
        existing: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        new_entries = []
        for path in sorted(all_entries):
            entry = all_entries[path]
            prior = existing.get(path)
            if prior is None:
                new_entries.append(entry)
                continue
            if (prior["row_count"], prior["content_hash"]) != (entry["row_count"], entry["content_hash"]):
                msg = (
                    f"Path {path!r} is already registered in manifest {self.manifest_dir} "
                    "with different row_count/content_hash -- reusing a logical identity with "
                    "incompatible content is not allowed"
                )
                raise ValueError(msg)
            # Already present with matching content: nothing to append.
        return new_entries

    def _append_entries(self, entries: list[dict[str, Any]]) -> None:
        # Appended sequentially, one fragment per file, so fragment_id
        # order matches the canonical (sorted-by-path) order callers see.
        for entry in entries:
            table = pa.Table.from_pylist([entry], schema=MANIFEST_SCHEMA)
            mode = "append" if self._dataset_exists() else "create"
            lance.write_dataset(
                table,
                self.manifest_dir,
                mode=mode,
                max_rows_per_file=1,
                storage_options=self.storage_options or None,
            )

    def validate(self, deep: bool = False) -> list[str]:
        """Return a list of validation errors (empty means valid)."""
        errors: list[str] = []
        dataset = self._open()
        table = dataset.to_table(columns=["path", "row_count", "file_size", "content_hash"])
        paths = table["path"].to_pylist()
        row_counts = table["row_count"].to_pylist()
        file_sizes = table["file_size"].to_pylist()

        seen: dict[str, int] = {}
        for i, path in enumerate(paths):
            if path in seen:
                errors.append(f"Duplicate path in manifest: {path!r} (rows {seen[path]} and {i})")
            else:
                seen[path] = i

        for fragment in dataset.get_fragments():
            if fragment.count_rows() != 1:
                errors.append(
                    f"Fragment {fragment.fragment_id} has {fragment.count_rows()} rows, expected "
                    "exactly 1 (one manifest entry per fragment)"
                )

        for path, row_count, file_size in zip(paths, row_counts, file_sizes, strict=True):
            if row_count is None or row_count < 0:
                errors.append(f"Invalid row_count for {path!r}: {row_count!r}")
            if file_size is None or file_size < 0:
                errors.append(f"Invalid file_size for {path!r}: {file_size!r}")

        if deep:
            for path, row_count in zip(paths, row_counts, strict=True):
                try:
                    actual = scan_file_entry(path, self.storage_options)
                except (OSError, FileNotFoundError) as e:
                    errors.append(f"Could not re-scan {path!r}: {e}")
                    continue
                if actual["row_count"] != row_count:
                    errors.append(
                        f"Row count changed for {path!r}: manifest has {row_count}, "
                        f"actual is {actual['row_count']}"
                    )
        return errors

    def list_entries(self) -> list[ManifestFileEntry]:
        """All entries, ordered by ``file_idx`` (the manifest's canonical order)."""
        dataset = self._open()
        table = dataset.scanner(
            columns=["path", "row_count", "file_size", "content_hash"],
            with_row_address=True,
            with_row_id=True,
        ).to_table()
        table = add_lance_metadata_columns(table)
        entries = [
            ManifestFileEntry(
                path=row["path"],
                row_count=row["row_count"],
                file_size=row["file_size"],
                content_hash=row["content_hash"],
                file_idx=row[LANCE_FRAGID_COLUMN],
            )
            for row in table.to_pylist()
        ]
        return sorted(entries, key=lambda entry: entry.file_idx)

    def get_id_range(self, path: str) -> tuple[int, int]:
        """The ``(id_start, id_end)`` range reserved for ``path``."""
        escaped = path.replace("'", "''")
        dataset = self._open()
        table = dataset.scanner(
            filter=f"path = '{escaped}'",
            with_row_address=True,
            with_row_id=True,
        ).to_table()
        if table.num_rows == 0:
            msg = f"Path {path!r} is not present in manifest {self.manifest_dir}"
            raise KeyError(msg)
        if table.num_rows > 1:
            msg = f"Path {path!r} has {table.num_rows} entries in manifest {self.manifest_dir}, expected 1"
            raise ValueError(msg)
        table = add_lance_metadata_columns(table)
        file_idx = table[LANCE_FRAGID_COLUMN][0].as_py()
        row_count = table["row_count"][0].as_py()
        return encode_id_range(file_idx, row_count)

    def file_for_id(self, id_value: int) -> str:
        """The durable "id -> original source file" lookback."""
        file_idx, _ = decode_id(id_value)
        dataset = self._open()
        fragment = dataset.get_fragment(file_idx)
        if fragment is None:
            msg = f"No manifest fragment {file_idx} (decoded from id {id_value}) in {self.manifest_dir}"
            raise KeyError(msg)
        table = fragment.to_table(columns=["path"])
        return table["path"][0].as_py()


def assign_ids_for_batch(manifest: IdManifest, filepath: str | list[str], num_rows: int) -> np.ndarray:
    """Compute the id array for a read batch spanning one or more files.

    Shared by both the CPU (``BaseReader``, pandas) and GPU
    (``DeduplicationIO``, cuDF) read paths: a batch read from multiple files
    concatenates each file's rows in order (no interleaving), so per-file id
    ranges from the manifest can be assigned positionally, without needing a
    per-row "source file" column. Raises if the manifest's row counts for
    ``filepath`` don't sum to ``num_rows`` -- a sign the source files changed
    since the manifest was built.
    """
    paths = filepath if isinstance(filepath, list) else [filepath]
    id_arrays = []
    offset = 0
    for path in paths:
        min_id, max_id = manifest.get_id_range(path)
        row_count = max_id - min_id + 1
        if offset + row_count > num_rows:
            msg = (
                f"Manifest row_count for {path!r} ({row_count}) exceeds the rows remaining in the "
                f"read batch ({num_rows - offset}); {path!r} may have changed since the manifest was built"
            )
            raise ValueError(msg)
        id_arrays.append(np.arange(min_id, max_id + 1))
        offset += row_count

    if offset != num_rows:
        msg = (
            f"Manifest row counts for {paths} sum to {offset}, but the read batch has {num_rows} rows; "
            "the source files may have changed since the manifest was built"
        )
        raise ValueError(msg)

    return np.concatenate(id_arrays) if id_arrays else np.array([], dtype=np.int64)
