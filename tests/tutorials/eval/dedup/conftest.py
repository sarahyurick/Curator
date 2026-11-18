# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Imports and span-aligned records for the standalone tutorial modules."""

import importlib.util
import sys
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).parents[4] / "tutorials/eval/dedup"
sys.path.insert(0, str(TUTORIAL))


@pytest.fixture
def critic_module():
    name = "dedup_run_critics"
    spec = importlib.util.spec_from_file_location(name, TUTORIAL / "6_run_critics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c0_record():
    shared = "Service hours and prices are unchanged. "
    texts = {"A": shared + "City Alpha", "B": shared + "City Beta"}
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_text": shared,
            "b_text": shared,
            "a_start_char": 0,
            "a_end_char": len(shared),
            "b_start_char": 0,
            "b_end_char": len(shared),
        }
    ]
    for side in ("A", "B"):
        spans.append(
            {
                "span_id": side + "001",
                "kind": side + "_ONLY",
                "text": texts[side][len(shared) :],
                "start_char": len(shared),
                "end_char": len(texts[side]),
                "delta_start_char": len(shared),
                "delta_end_char": len(texts[side]),
            }
        )
    values = {
        "span_content_profile_a": "substantive_main",
        "span_content_profile_b": "substantive_main",
        "span_translation_status": "not_translation",
        "span_shared_basis": "verified_substantive_record",
        "span_a_delta": "universal_ui_or_repetition",
        "span_b_delta": "semantically_covered",
        "span_hard_conflict": "none",
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    return {
        "pair_id": "service",
        "text_a": texts["A"],
        "text_b": texts["B"],
        "truncated": False,
        "semantic_diff": {
            "status": "COMPLETE",
            "truncated": False,
            "truncated_a": False,
            "truncated_b": False,
            "span_counts": {"SHARED": 1, "A_ONLY": 1, "B_ONLY": 1},
            "spans": spans,
        },
        "pair_semantic_judgment": {
            key: {"score": value, "reasoning": "S001: same service and conditions."} for key, value in values.items()
        },
    }
