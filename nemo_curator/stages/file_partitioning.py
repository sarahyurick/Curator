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

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import (
    _split_files_as_per_blocksize,
    get_all_file_paths_and_size_under,
    infer_dataset_name_from_path,
    parse_bytes_string_to_int,
)

if TYPE_CHECKING:
    from nemo_curator.stages.deduplication.id_manifest import IdManifest, ManifestFileEntry

# Mirrors nemo_curator.stages.deduplication.id_manifest.ID_MANIFEST_DIR_METADATA_KEY.
# Kept as a plain constant (rather than importing that module, which pulls in the
# optional `lance` dependency) so this widely-used stage stays importable without
# the `[lance]` extra when build_id_manifest is never used.
_ID_MANIFEST_DIR_METADATA_KEY = "id_manifest_dir"


@dataclass
class FilePartitioningStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Stage that partitions input file paths into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.

    Parameters
    ----------
    file_paths: str | list[str]
        Path to the input files.
    files_per_partition: int | None = None
        Number of files per partition.
        If both files_per_partition and blocksize are not provided,
        then default to files_per_partition = 1 and enforce a blocksize <= 512 MB per partition safeguard.
        Errors if both files_per_partition and blocksize are provided.
    blocksize: int | str | None = None
        Target size of the partitions. A blocksize of 512 MB or less is recommended.
        Errors if both files_per_partition and blocksize are provided.
        Note: For compressed files, the compressed size is used for blocksize estimation.
    file_extensions: list[str] | None = None
        File extensions to filter.
    storage_options: dict[str, Any] | None = None
        Storage options to pass to the file system.
    limit: int | None = None
        Maximum number of partitions to create.
    build_id_manifest: bool = False
        Whether this stage should build an ``IdManifest`` over the discovered
        files and use it to drive dedup-id assignment for downstream readers
        with ``_generate_ids``/``_assign_ids``. Off by default: most
        ``FilePartitioningStage`` users (video/audio readers, intermediate
        cache listings, etc.) don't assign dedup ids and shouldn't pay
        manifest-build cost or be constrained to jsonl/parquet row counting.
        Readers that do need dedup ids (``JsonlReader``/``ParquetReader``
        with ``_generate_ids`` or ``_assign_ids``) set this automatically.
        Slurm-array sharding of the emitted ``FileGroupTask``s (if a Slurm
        array is active) is unaffected either way -- it's handled generically
        by the executor, hashing each task's deterministic ``task_id``, the
        same as for every other source stage; this stage has no Slurm-specific
        logic of its own.
    id_manifest_dir: str | None = None
        Directory for the ``IdManifest``, when ``build_id_manifest=True``.
        Required when ``file_paths`` is a list (no single directory to derive
        a default from); defaults to
        ``<file_paths>/.nemo_curator_id_manifest`` when ``file_paths`` is a
        single directory path.
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    limit: int | None = None
    build_id_manifest: bool = False
    id_manifest_dir: str | None = None
    name: str = "file_partitioning"

    def __post_init__(self):
        """Initialize default values."""
        if self.files_per_partition is not None and self.blocksize is not None:
            msg = "Both 'files_per_partition' and 'blocksize' were specified, but only one is allowed"
            raise ValueError(msg)
        if self.file_extensions is None:
            self.file_extensions = [".jsonl", ".json", ".parquet"]
        if self.storage_options is None:
            self.storage_options = {}
        if self.build_id_manifest and self.id_manifest_dir is None:
            if not isinstance(self.file_paths, str):
                msg = (
                    "id_manifest_dir must be provided explicitly when file_paths is a list "
                    "(there is no single directory to derive a default location from)"
                )
                raise ValueError(msg)
            self.id_manifest_dir = self.file_paths.rstrip("/") + "/.nemo_curator_id_manifest"

        # self.blocksize is the value set by the user
        # self._blocksize is the value used internally
        if self.blocksize is not None:
            self._blocksize = parse_bytes_string_to_int(self.blocksize)
        else:
            self._blocksize = parse_bytes_string_to_int("512MB")

        if self._blocksize > parse_bytes_string_to_int("512MB"):
            msg = (
                f"Blocksize is greater than 512 MB, which is not recommended: {self.blocksize} "
                "Consider using a smaller blocksize to avoid potential memory issues."
            )
            logger.warning(msg)

        self.resources = Resources(cpus=0.5)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def process(self, _: EmptyTask) -> list[FileGroupTask]:
        """Process the initial task to create file group tasks.

        This stage expects a simple Task with file paths information
        and outputs multiple FileGroupTasks for parallel processing.
        """
        if self.build_id_manifest:
            files, files_with_sizes = self._list_files_via_manifest()
        else:
            sort_by_size = self.blocksize is not None
            files_with_sizes = self._get_file_list_with_sizes(sort_by_size)
            files = [file[0] for file in files_with_sizes]
            logger.info(f"Found {len(files)} files")

        if len(files) == 0:
            logger.warning(f"No files found under {self.file_paths}")
            return []

        # Partition files
        if self.files_per_partition:
            partitions = self._partition_by_count(files, self.files_per_partition)
        elif self.blocksize:
            partitions = self._partition_by_size(files_with_sizes, self._blocksize)
        else:
            # Default to one file per partition
            logger.info("No partitions specified, defaulting to one file per partition")
            partitions = self._partition_by_count(files, 1)

        # Build a dictionary of path: size of all files
        path_to_size: dict[str, int] = dict(files_with_sizes)

        # Check that no files have size less than 0 (since -1 is used to indicate unknown size)
        if any(size < 0 for size in path_to_size.values()):
            msg = "Skipping storage limit check because some files have unknown size"
            logger.warning(msg)
        else:
            # Verify storage size of input files is not greater than self._blocksize (512 MB by default)
            # This should be a very quick check per file, so we do it first before reading the data
            for partition in partitions:
                total_storage_size = sum(path_to_size[path] for path in partition)
                # Scenario 1: The user specified blocksize and the partition created is too large
                # This means at least one file is larger than the blocksize
                if self.blocksize is not None and total_storage_size > self._blocksize:
                    msg = (
                        f"File group task has exceeded the storage limit per partition: {partition}. "
                        f"Total storage size is {total_storage_size} bytes (limit {self._blocksize} bytes). "
                        "Please increase blocksize if possible (the maximum recommended blocksize is 512 MB). "
                        "Any individual file(s) larger than the storage limit should be split into smaller chunks using nemo_curator.utils.split_large_files."
                    )
                    logger.warning(msg)
                # Scenario 2: The user did not specify blocksize and the partition created is too large
                elif total_storage_size > self._blocksize:
                    msg = (
                        f"File group task has exceeded the storage limit per partition: {partition}. "
                        f"Total storage size is {total_storage_size} bytes (limit {self._blocksize} bytes). "
                        "Please reduce files_per_partition if possible, or set blocksize instead (the maximum recommended blocksize is 512 MB). "
                        "Any individual file(s) larger than the storage limit should be split into smaller chunks using nemo_curator.utils.split_large_files."
                    )
                    logger.warning(msg)

        # Create FileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)
        metadata_extra = {_ID_MANIFEST_DIR_METADATA_KEY: self.id_manifest_dir} if self.build_id_manifest else {}

        for i, file_group in enumerate(partitions):
            if self.limit is not None and len(tasks) >= self.limit:
                # We should revisit this behavior.
                # https://github.com/NVIDIA-NeMo/Curator/issues/948
                logger.info(f"Reached limit of {self.limit} file groups")
                break
            file_task = FileGroupTask(
                dataset_name=dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "source_files": file_group,  # Add source files for deterministic naming during write stage
                    **metadata_extra,
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)

        logger.info(f"Created {len(tasks)} file groups from {len(files)} files")
        return tasks

    def _list_files_via_manifest(self) -> tuple[list[str], list[tuple[str, int]]]:
        """Discover files and build/load the IdManifest over all of them.

        Listing order doesn't affect partition quality (``_partition_by_size``
        re-sorts by size internally); files are ordered by the manifest's
        canonical (path-sorted) ``file_idx`` order instead. Slurm-array
        sharding of the resulting FileGroupTasks (if any) happens generically
        in the executor, same as for every other source stage -- this stage
        has no Slurm-specific logic of its own.
        """
        discovered_files = [path for path, _ in self._get_file_list_with_sizes(sort_by_size=False)]
        logger.info(f"Found {len(discovered_files)} files")
        if len(discovered_files) == 0:
            return [], []

        manifest = self._get_or_build_manifest(discovered_files)
        entries: list[ManifestFileEntry] = manifest.list_entries()

        files = [entry.path for entry in entries]
        files_with_sizes = [(entry.path, entry.file_size) for entry in entries]
        return files, files_with_sizes

    def _get_or_build_manifest(self, discovered_files: list[str]) -> "IdManifest":
        """Load the manifest if it already covers every discovered file, else build it.

        Concurrent callers pointed at the same ``id_manifest_dir`` (e.g. many
        Slurm array tasks launched around the same time, before any manifest
        exists yet) must not all try to build it at once: ``IdManifest.build()``
        fails fast with ``RuntimeError`` on writer-lock contention rather than
        blocking, so a naive "just call build()" here would crash every task
        except whichever one happens to win the lock. Instead: try a read-only
        load first (safe under unlimited concurrency); if that comes up empty
        or incomplete, try to build; if another process is already building it
        (lock contention), wait for that build to finish and load its result
        instead of failing this task.
        """
        # Imported lazily: id_manifest.py pulls in the optional `lance` dependency,
        # and most FilePartitioningStage users never set build_id_manifest=True.
        from nemo_curator.stages.deduplication.id_manifest import IdManifest

        manifest = self._try_load_complete_manifest(discovered_files)
        if manifest is not None:
            return manifest

        try:
            return IdManifest.build(discovered_files, self.id_manifest_dir, storage_options=self.storage_options)
        except RuntimeError:
            logger.info(
                f"Another process is already building the IdManifest at {self.id_manifest_dir}; "
                "waiting for it to finish instead of racing it."
            )
            return self._wait_for_manifest(discovered_files)

    def _try_load_complete_manifest(self, discovered_files: list[str]) -> "IdManifest | None":
        from nemo_curator.stages.deduplication.id_manifest import IdManifest

        try:
            manifest = IdManifest.from_disk(self.id_manifest_dir, storage_options=self.storage_options)
        except (ValueError, OSError, FileNotFoundError):
            return None
        existing_paths = {entry.path for entry in manifest.list_entries()}
        if set(discovered_files) <= existing_paths:
            return manifest
        return None

    def _wait_for_manifest(
        self,
        discovered_files: list[str],
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 5.0,
    ) -> "IdManifest":
        # If the process building it fails outright (not just contention), waiters
        # here time out rather than surfacing that failure directly -- the build
        # error itself will be visible in that process's own logs/exit status.
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            manifest = self._try_load_complete_manifest(discovered_files)
            if manifest is not None:
                return manifest
            time.sleep(poll_interval_seconds)
        msg = (
            f"Timed out after {timeout_seconds}s waiting for a concurrently-building IdManifest "
            f"at {self.id_manifest_dir} to finish"
        )
        raise TimeoutError(msg)

    def _get_file_list_with_sizes(self, sort_by_size: bool = True) -> list[tuple[str, int]]:
        """
        Get the list of files to process.
        """
        logger.debug(f"Getting file list with sizes for {self.file_paths}")
        if isinstance(self.file_paths, str):
            # Directory: list contents (recursively) and filter extensions
            output_ls = get_all_file_paths_and_size_under(
                self.file_paths,
                recurse_subdirectories=True,
                keep_extensions=self.file_extensions,
                storage_options=self.storage_options,
                sort_by_size=sort_by_size,
            )
        elif isinstance(self.file_paths, list):
            output_ls = []
            for path in self.file_paths:
                output_ls.extend(
                    get_all_file_paths_and_size_under(
                        path,
                        recurse_subdirectories=False,
                        keep_extensions=self.file_extensions,
                        storage_options=self.storage_options,
                        sort_by_size=sort_by_size,
                    )
                )
        else:
            msg = f"Invalid file paths: {self.file_paths}, must be a string or list of strings"
            raise TypeError(msg)
        return sorted(output_ls, key=lambda x: x[1] if sort_by_size else x[0])

    def _get_dataset_name(self, files: list[str]) -> str:
        """Extract dataset name from file paths (fsspec-compatible)."""
        if not files:
            return "dataset"

        return infer_dataset_name_from_path(files[0])

    def _partition_by_count(self, files: list[str], count: int) -> list[list[str]]:
        """Partition files by count."""
        partitions = []
        for i in range(0, len(files), count):
            partitions.append(files[i : i + count])
        return partitions

    def _partition_by_size(self, files: list[tuple[str, int]], blocksize: int | str) -> list[list[str]]:
        """Partition files by target size.
        Args:
            files: A list of tuples (file_path, file_size)
            blocksize: The target size of the partitions
        Returns:
            A list of lists, where each inner list contains the file paths of the files in the partitionN
        """
        sorted_files = sorted(files, key=lambda x: x[1])
        return _split_files_as_per_blocksize(sorted_files, blocksize)
