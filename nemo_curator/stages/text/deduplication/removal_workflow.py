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
from typing import TYPE_CHECKING, Any, Literal, Optional

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_default_file_extensions

from .removal import TextDuplicatesRemovalStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor


@dataclass
class TextDuplicatesRemovalWorkflow(WorkflowBase):
    # required args
    input_path: str | None
    ids_to_remove_path: str
    output_path: str

    # input args
    input_filetype: Literal["parquet", "jsonl"] = "parquet"
    input_fields: list[str] | None = None
    id_field: str | None = CURATOR_DEDUP_ID_STR
    input_files_per_partition: int | None = None
    input_blocksize: str | None = None
    input_file_extensions: list[str] | None = None
    input_task_limit: int | None = None
    input_kwargs: dict[str, Any] | None = None

    # ids_to_remove args
    duplicate_id_field: str = "id"
    duplicate_id_read_kwargs: dict[str, Any] | None = None

    # id manifest args -- must point at the same manifest directory the
    # upstream workflow (fuzzy/semantic/exact) used to assign
    # CURATOR_DEDUP_ID_STR to input_path's files, so re-reading here resolves
    # to the exact same ids (see IdManifest / id_manifest.py). Storage options
    # for the manifest come from input_kwargs["storage_options"], same as the
    # input files themselves.
    id_manifest_dir: str | None = None

    # output args
    output_file_extension: str | None = None
    output_filetype: Literal["parquet", "jsonl"] = "parquet"
    output_kwargs: dict[str, Any] | None = None
    output_fields: list[str] | None = None
    output_mode: Literal["ignore", "overwrite", "append", "error"] | None = None
    drop_id_field: bool = False

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        if self.id_manifest_dir is None and self.id_field == CURATOR_DEDUP_ID_STR:
            logger.warning(
                f"Using {CURATOR_DEDUP_ID_STR} as id_field for removal stage, even though id_manifest_dir is not set."
            )
        if self.drop_id_field and self.output_fields and self.id_field in self.output_fields:
            msg = f"Cannot drop id_field {self.id_field!r} when it is included in output_fields."
            raise ValueError(msg)

    def _generate_stages(self, initial_tasks: list[FileGroupTask] | None = None) -> list[ProcessingStage]:
        stages = []

        if initial_tasks is None:
            if self.input_path is None:
                msg = "input_path is required when initial_tasks is None"
                raise ValueError(msg)

            if self.input_filetype not in ("parquet", "jsonl"):
                msg = f"Invalid input filetype: {self.input_filetype}"
                raise ValueError(msg)

            from nemo_curator.stages.file_partitioning import FilePartitioningStage

            stages.append(
                FilePartitioningStage(
                    file_paths=self.input_path,
                    files_per_partition=self.input_files_per_partition,
                    blocksize=self.input_blocksize,
                    file_extensions=(self.input_file_extensions or get_default_file_extensions(self.input_filetype)),
                    storage_options=(self.input_kwargs or {}).get("storage_options"),
                    limit=self.input_task_limit,
                    build_id_manifest=self.id_manifest_dir is not None,
                    id_manifest_dir=self.id_manifest_dir,
                )
            )
        else:
            fields_to_ignore = ["input_path", "input_files_per_partition", "input_blocksize", "input_file_extensions"]
            logger.warning(f"Initial tasks provided, ignoring {fields_to_ignore}")

        if self.input_filetype == "parquet":
            from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage

            read_stage = ParquetReaderStage
        elif self.input_filetype == "jsonl":
            from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage

            read_stage = JsonlReaderStage
        else:
            msg = f"Invalid input filetype: {self.input_filetype}"
            raise ValueError(msg)

        stages.append(
            read_stage(
                fields=self.input_fields,
                read_kwargs=self.input_kwargs,
                _generate_ids=False,
                _assign_ids=self.id_manifest_dir is not None,
            )
        )

        stages.append(
            TextDuplicatesRemovalStage(
                ids_to_remove_path=self.ids_to_remove_path,
                id_field=self.id_field,
                duplicate_id_field=self.duplicate_id_field,
                read_kwargs=self.duplicate_id_read_kwargs,
                drop_id_field=self.drop_id_field,
            )
        )

        if self.output_filetype == "parquet":
            from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

            write_stage = ParquetWriter
        elif self.output_filetype == "jsonl":
            from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

            write_stage = JsonlWriter
        else:
            msg = f"Invalid output filetype: {self.output_filetype}"
            raise ValueError(msg)

        stages.append(
            write_stage(
                path=self.output_path,
                **({"file_extension": self.output_file_extension} if self.output_file_extension else {}),
                write_kwargs=self.output_kwargs or {},
                fields=self.output_fields,
                **({"mode": self.output_mode} if self.output_mode else {}),
            )
        )

        return stages

    @staticmethod
    def _count_removed_duplicates(tasks: list[FileGroupTask] | None) -> int:
        """Sum num_removed metadata reported by downstream stages."""
        total_removed = 0
        for task in tasks or []:
            metadata = getattr(task, "_metadata", {}) or {}
            total_removed += metadata.get("num_removed", 0)
        return total_removed

    def run(
        self, executor: Optional["BaseExecutor"] = None, initial_tasks: list[FileGroupTask] | None = None
    ) -> WorkflowRunResult:
        pipeline = Pipeline(
            name="text_duplicates_removal_workflow",
            description="Text duplicates removal workflow",
            stages=self._generate_stages(initial_tasks),
        )
        workflow_result = WorkflowRunResult(workflow_name="text_duplicates_removal")
        if (
            self.input_task_limit is not None
            and initial_tasks is not None
            and len(initial_tasks) > self.input_task_limit
        ):
            logger.warning(
                f"Initial tasks provided ({len(initial_tasks)}) is greater than input_task_limit ({self.input_task_limit}), truncating to {self.input_task_limit}"
            )
            initial_tasks = initial_tasks[: self.input_task_limit]

        if executor is None:
            from nemo_curator.backends.xenna import XennaExecutor

            executor = XennaExecutor()

        if initial_tasks is not None and self.id_manifest_dir is not None:
            # FilePartitioningStage (which normally stamps this) is skipped
            # when initial_tasks is provided directly, so stamp it here.
            from nemo_curator.stages.deduplication.id_manifest import ID_MANIFEST_DIR_METADATA_KEY

            for task in initial_tasks:
                task._metadata = {**(task._metadata or {}), ID_MANIFEST_DIR_METADATA_KEY: self.id_manifest_dir}

        start_time = time.time()
        output_tasks = pipeline.run(executor, initial_tasks=initial_tasks)
        execution_time = time.time() - start_time
        num_duplicates_removed = self._count_removed_duplicates(output_tasks)

        workflow_result.add_pipeline_tasks("removal", output_tasks)
        workflow_result.add_metadata("total_time", execution_time)
        workflow_result.add_metadata("num_duplicates_removed", num_duplicates_removed)
        return workflow_result
