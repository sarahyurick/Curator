# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from critics.coverage import CoverageApplyStage, CoveragePrepareStage

from nemo_curator.tasks import DocumentBatch


def test_coverage_snapshot_and_main_survive_final_decision_changes(c0_record: dict) -> None:
    batch = CoveragePrepareStage("pair_semantic_judgment").process(
        DocumentBatch(dataset_name="coverage", data=pa.Table.from_pylist([c0_record]))
    )
    row = batch.to_pyarrow().to_pylist()[0]
    row["coverage_review"] = {
        "a_loss_span_id": "",
        "b_loss_span_id": "",
        "a_context_span_id": "S001",
        "b_context_span_id": "S001",
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "shared_anchor_ids": ["S001"],
        "explanation": "Same service.",
    }
    output = (
        CoverageApplyStage("pair_semantic_judgment")
        .process(DocumentBatch(dataset_name="coverage", data=pa.Table.from_pylist([row])))
        .to_pyarrow()
        .to_pylist()[0]
    )
    assert output["coverage_action"] == "KEEP_MAIN"
    assert output["pair_semantic_judgment"] == c0_record["pair_semantic_judgment"]
    output["final_decision"]["a_can_replace_b"] = "no"
    assert output["coverage_decision"]["a_can_replace_b"] == "yes"
