# modality: text

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

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.deduplication.id_manifest import (  # noqa: E402
    ROW_INDEX_BITS,
    IdManifest,
    assign_ids_for_batch,
    decode_id,
    encode_id_range,
    scan_file_entry,
)


def _write_jsonl(path: Path, num_rows: int) -> None:
    with path.open("w") as f:
        for i in range(num_rows):
            f.write(json.dumps({"text": f"row-{i}"}) + "\n")


def _write_parquet(path: Path, num_rows: int) -> None:
    table = pa.table({"text": [f"row-{i}" for i in range(num_rows)]})
    pq.write_table(table, path)


class TestEncodeDecode:
    def test_round_trip(self):
        for file_idx, row_idx in [(0, 0), (0, 5), (3, 0), (3, 999), (1_000_000, 42)]:
            id_start, _ = encode_id_range(file_idx, row_idx + 1)
            assert decode_id(id_start + row_idx) == (file_idx, row_idx)

    def test_encode_range_boundaries(self):
        id_start, id_end = encode_id_range(0, 10)
        assert id_start == 0
        assert id_end == 9

        id_start, id_end = encode_id_range(1, 10)
        assert id_start == 1 << ROW_INDEX_BITS
        assert id_end == id_start + 9

    def test_encode_empty_file_range_is_before_start(self):
        id_start, id_end = encode_id_range(5, 0)
        assert id_end == id_start - 1

    def test_encode_rejects_negative_inputs(self):
        with pytest.raises(ValueError, match="file_idx"):
            encode_id_range(-1, 10)
        with pytest.raises(ValueError, match="row_count"):
            encode_id_range(0, -1)

    def test_decode_rejects_negative_id(self):
        with pytest.raises(ValueError, match="id_value"):
            decode_id(-1)


class TestScanFileEntry:
    def test_jsonl(self, tmp_path: Path):
        path = tmp_path / "a.jsonl"
        _write_jsonl(path, 7)
        entry = scan_file_entry(str(path))
        assert entry["path"] == str(path)
        assert entry["row_count"] == 7
        assert entry["file_size"] > 0
        assert entry["content_hash"]

    def test_parquet(self, tmp_path: Path):
        path = tmp_path / "a.parquet"
        _write_parquet(path, 11)
        entry = scan_file_entry(str(path))
        assert entry["row_count"] == 11

    def test_unsupported_extension(self, tmp_path: Path):
        path = tmp_path / "a.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="row count"):
            scan_file_entry(str(path))


class TestIdManifestBuild:
    def test_build_assigns_sequential_file_idx_in_path_order(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "b.jsonl", 3)
        _write_jsonl(files_dir / "a.jsonl", 5)

        manifest = IdManifest.build(
            [str(files_dir / "b.jsonl"), str(files_dir / "a.jsonl")],
            str(tmp_path / "manifest"),
        )
        entries = manifest.list_entries()
        assert [Path(e.path).name for e in entries] == ["a.jsonl", "b.jsonl"]
        assert entries[0].file_idx == 0
        assert entries[1].file_idx == 1
        assert entries[0].row_count == 5
        assert entries[1].row_count == 3

    def test_build_is_reproducible(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 4)
        _write_jsonl(files_dir / "b.jsonl", 6)
        paths = [str(files_dir / "a.jsonl"), str(files_dir / "b.jsonl")]

        manifest1 = IdManifest.build(paths, str(tmp_path / "manifest1"))
        manifest2 = IdManifest.build(paths, str(tmp_path / "manifest2"))

        entries1 = [(e.path, e.file_idx, e.id_start, e.id_end) for e in manifest1.list_entries()]
        entries2 = [(e.path, e.file_idx, e.id_start, e.id_end) for e in manifest2.list_entries()]
        # Paths differ only by manifest dir, not by source file path, so compare
        # file_idx/id_start/id_end assignment, not the manifest dir itself.
        assert [(f, i, s, e) for _, i, s, e in entries1] == [(f, i, s, e) for _, i, s, e in entries2]

    def test_get_id_range_and_file_for_id_round_trip(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 4)
        path = str(files_dir / "a.jsonl")

        manifest = IdManifest.build([path], str(tmp_path / "manifest"))
        id_start, id_end = manifest.get_id_range(path)
        assert id_end - id_start + 1 == 4
        assert manifest.file_for_id(id_start) == path
        assert manifest.file_for_id(id_end) == path

    def test_get_id_range_missing_path_raises(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 4)
        manifest = IdManifest.build([str(files_dir / "a.jsonl")], str(tmp_path / "manifest"))
        with pytest.raises(KeyError):
            manifest.get_id_range(str(files_dir / "missing.jsonl"))


class TestIdManifestExtend:
    def test_extend_preserves_prior_ranges(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 3)
        _write_jsonl(files_dir / "b.jsonl", 5)
        manifest_dir = str(tmp_path / "manifest")

        manifest = IdManifest.build([str(files_dir / "a.jsonl")], manifest_dir)
        original_range = manifest.get_id_range(str(files_dir / "a.jsonl"))

        manifest.extend([str(files_dir / "b.jsonl")])
        assert manifest.get_id_range(str(files_dir / "a.jsonl")) == original_range

        new_range = manifest.get_id_range(str(files_dir / "b.jsonl"))
        assert new_range[0] > original_range[1]

    def test_extend_rejects_incompatible_reuse(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        path = files_dir / "a.jsonl"
        _write_jsonl(path, 3)
        manifest_dir = str(tmp_path / "manifest")

        manifest = IdManifest.build([str(path)], manifest_dir)
        _write_jsonl(path, 4)  # same path, different content now
        with pytest.raises(ValueError, match="incompatible content"):
            manifest.extend([str(path)])


class TestIdManifestValidate:
    def test_validate_clean_manifest(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 3)
        manifest = IdManifest.build([str(files_dir / "a.jsonl")], str(tmp_path / "manifest"))
        assert manifest.validate() == []

    def test_validate_deep_detects_changed_row_count(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        path = files_dir / "a.jsonl"
        _write_jsonl(path, 3)
        manifest = IdManifest.build([str(path)], str(tmp_path / "manifest"))

        _write_jsonl(path, 9)
        errors = manifest.validate(deep=True)
        assert any("Row count changed" in e for e in errors)


class TestIdManifestShardedGeneration:
    def test_build_part_and_finalize_matches_single_build(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 3)
        _write_jsonl(files_dir / "b.jsonl", 5)
        paths = [str(files_dir / "a.jsonl"), str(files_dir / "b.jsonl")]

        direct = IdManifest.build(paths, str(tmp_path / "direct"))

        sharded = IdManifest(str(tmp_path / "sharded"))
        sharded.build_part([paths[0]], part_id="0", total_shards=2, shard_index=0)
        sharded.build_part([paths[1]], part_id="1", total_shards=2, shard_index=1)
        sharded.finalize(total_shards=2)

        direct_entries = [(Path(e.path).name, e.file_idx, e.row_count) for e in direct.list_entries()]
        sharded_entries = [(Path(e.path).name, e.file_idx, e.row_count) for e in sharded.list_entries()]
        assert direct_entries == sharded_entries

    def test_finalize_raises_when_shard_missing(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 3)

        manifest = IdManifest(str(tmp_path / "manifest"))
        manifest.build_part([str(files_dir / "a.jsonl")], part_id="0", total_shards=2, shard_index=0)
        with pytest.raises(ValueError, match="missing build_part"):
            manifest.finalize(total_shards=2)

    def test_finalize_succeeds_once_all_shards_present(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 3)
        _write_jsonl(files_dir / "b.jsonl", 3)

        manifest = IdManifest(str(tmp_path / "manifest"))
        manifest.build_part([str(files_dir / "a.jsonl")], part_id="0", total_shards=2, shard_index=0)
        manifest.build_part([str(files_dir / "b.jsonl")], part_id="1", total_shards=2, shard_index=1)
        manifest.finalize(total_shards=2)
        assert len(manifest.list_entries()) == 2


class TestIdManifestConcurrency:
    def test_finalize_rejects_concurrent_lock_holder(self, tmp_path: Path):
        manifest = IdManifest(str(tmp_path / "manifest"))
        lock_path = manifest._lock_path()  # noqa: SLF001
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("{}")
        try:
            with pytest.raises(RuntimeError, match="already in progress"):
                manifest.finalize()
        finally:
            lock_path.unlink(missing_ok=True)


class TestAssignIdsForBatch:
    def test_single_file_batch(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        path = str(files_dir / "a.jsonl")
        _write_jsonl(files_dir / "a.jsonl", 5)
        manifest = IdManifest.build([path], str(tmp_path / "manifest"))

        ids = assign_ids_for_batch(manifest, path, 5)
        id_start, id_end = manifest.get_id_range(path)
        assert list(ids) == list(range(id_start, id_end + 1))

    def test_multi_file_batch_positional_assignment(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        _write_jsonl(files_dir / "a.jsonl", 2)
        _write_jsonl(files_dir / "b.jsonl", 3)
        paths = [str(files_dir / "a.jsonl"), str(files_dir / "b.jsonl")]
        manifest = IdManifest.build(paths, str(tmp_path / "manifest"))

        ids = assign_ids_for_batch(manifest, paths, 5)
        a_start, a_end = manifest.get_id_range(paths[0])
        b_start, b_end = manifest.get_id_range(paths[1])
        assert list(ids) == list(range(a_start, a_end + 1)) + list(range(b_start, b_end + 1))

    def test_mismatched_row_count_raises(self, tmp_path: Path):
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        path = str(files_dir / "a.jsonl")
        _write_jsonl(files_dir / "a.jsonl", 5)
        manifest = IdManifest.build([path], str(tmp_path / "manifest"))

        with pytest.raises(ValueError, match="Manifest row counts"):
            assign_ids_for_batch(manifest, path, 4)
