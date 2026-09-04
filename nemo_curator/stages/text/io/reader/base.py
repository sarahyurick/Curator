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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
import pyarrow as pa
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask, LanceReadTask

ReaderTask: TypeAlias = FileGroupTask | LanceReadTask
ReaderData: TypeAlias = pd.DataFrame | pa.Table


@dataclass(frozen=True)
class ReaderOutput:
    data: ReaderData
    metadata: dict[str, Any] | None = None


@dataclass
class BaseReader(ProcessingStage[ReaderTask, DocumentBatch]):
    """Common base for tabular readers.

    Subclasses must implement read_task for their input task type.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    _generate_ids: bool = False
    _assign_ids: bool = False
    # Permit valid zero-row results.
    allow_empty: bool = False
    # Lazily loaded IdManifest cache; the manifest location comes from task
    # metadata (stamped by FilePartitioningStage), not fixed stage config, so
    # it can't be loaded in setup().
    _manifest: Any = field(default=None, init=False, repr=False, compare=False)
    _manifest_dir_loaded: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or [])
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

            output_fields.append(CURATOR_DEDUP_ID_STR)
        return ["data"], output_fields

    def setup(self, _: WorkerMetadata | None = None) -> None:
        # No actor to start: ID ranges are precomputed in an IdManifest and
        # loaded lazily per-task (see _get_manifest), since the manifest
        # location comes from task metadata, not fixed stage config.
        pass

    def process(self, task: ReaderTask) -> DocumentBatch:
        output = self.read_task(task, dict(self.read_kwargs or {}), self.fields)
        self._validate_result(task, output.data)
        return self._document_batch(task, output)

    def _document_batch(self, task: ReaderTask, output: ReaderOutput) -> DocumentBatch:
        batch = DocumentBatch(
            dataset_name=task.dataset_name,
            data=output.data,
            _metadata=output.metadata if output.metadata is not None else task._metadata,
        )
        if self._generate_ids or self._assign_ids:
            batch_key = self._id_generator_key(task)
            if self._generate_ids:
                self._generate_ids_func(batch_key, batch, task)
            else:
                self._assign_ids_func(batch_key, batch, task)

        return batch

    def _get_manifest(self, task: ReaderTask) -> Any:  # noqa: ANN401
        from nemo_curator.stages.deduplication.id_manifest import ID_MANIFEST_DIR_METADATA_KEY, IdManifest

        manifest_dir = (task._metadata or {}).get(ID_MANIFEST_DIR_METADATA_KEY)
        if manifest_dir is None:
            msg = (
                f"Task metadata is missing {ID_MANIFEST_DIR_METADATA_KEY!r}, required when "
                "_generate_ids or _assign_ids is True. It is normally stamped by FilePartitioningStage."
            )
            raise RuntimeError(msg)
        if self._manifest is None or self._manifest_dir_loaded != manifest_dir:
            storage_options = (self.read_kwargs or {}).get("storage_options")
            self._manifest = IdManifest.from_disk(manifest_dir, storage_options=storage_options)
            self._manifest_dir_loaded = manifest_dir
        return self._manifest

    def _validate_result(self, task: ReaderTask, result: ReaderData) -> None:
        if self.allow_empty:
            return
        if (
            (result is None)
            or (isinstance(result, pd.DataFrame) and result.empty)
            or (isinstance(result, pa.Table) and result.num_rows == 0)
        ):
            msg = f"No data read from files in task {task.task_id}"
            raise ValueError(msg)

    # Subclass responsibilities -------------------------------------------------
    def read_task(
        self,
        task: ReaderTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # ID helpers ----------------------------------------------------------------
    #
    # Manifest ranges are precomputed (assigned once, up front, when the
    # manifest was built) rather than allocated on first use, so "assign"
    # and "generate" are now the same lookup -- there's no longer a
    # meaningful distinction between "already registered elsewhere" and
    # "register now". Both flags are kept for config-surface compatibility.
    @staticmethod
    def _id_generator_key(task: ReaderTask) -> str | list[str]:
        # TODO(NMCUR-315): Use the deterministic task ID for FileGroupTask as well.
        # Keep returning file paths for backward compatibility -- IdManifest is keyed
        # by file path, so only FileGroupTask (real paths) is supported for now.
        if isinstance(task, FileGroupTask):
            return task.data
        return task.get_deterministic_id()

    @staticmethod
    def _append_ids(batch: DocumentBatch, ids: Any) -> None:  # noqa: ANN401
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if isinstance(batch.data, pd.DataFrame):
            batch.data[CURATOR_DEDUP_ID_STR] = ids
        else:
            batch.data = batch.data.append_column(CURATOR_DEDUP_ID_STR, pa.array(ids, type=pa.int64()))

    def _assign_ids_func(self, batch_key: str | list[str], batch: DocumentBatch, task: ReaderTask) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
        from nemo_curator.stages.deduplication.id_manifest import assign_ids_for_batch

        if CURATOR_DEDUP_ID_STR in batch.get_columns():
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {batch_key}, not re-assigning IDs")
            return

        # Only loaded once we know a manifest is actually needed -- a batch that
        # already carries CURATOR_DEDUP_ID_STR (handled above) shouldn't require
        # one at all.
        manifest = self._get_manifest(task)
        ids = assign_ids_for_batch(manifest, batch_key, batch.num_items)
        self._append_ids(batch, ids)

    def _generate_ids_func(self, batch_key: str | list[str], batch: DocumentBatch, task: ReaderTask) -> None:
        self._assign_ids_func(batch_key, batch, task)

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: self._generate_ids or self._assign_ids}


@dataclass
class BaseFileReader(BaseReader):
    """Base reader for file-group readers that consume lists of paths."""

    def read_task(
        self,
        task: FileGroupTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        return ReaderOutput(self.read_data(task.data, read_kwargs, fields))

    def read_data(
        self,
        file_paths: list[str],
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderData:  # pragma: no cover - abstract
        raise NotImplementedError
