# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pyarrow as pa
import pytest
from critics.c0 import C0AuditStage, main_errors
from critics.models import Decision

from nemo_curator.tasks import DocumentBatch


def test_covered_service_slots_remain_valid_without_rewriting(c0_record: dict) -> None:
    before = copy.deepcopy(c0_record)
    assert main_errors(c0_record, "pair_semantic_judgment", 240) == []
    assert c0_record == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("span_b_delta", "same_record_content_extension"),
        ("span_a_delta", "material_non_main_message_change"),
        ("span_hard_conflict", "legal_context"),
        ("span_shared_basis", "none"),
    ],
)
def test_literal_ledger_contradictions_are_invalid(c0_record: dict, field: str, value: str) -> None:
    c0_record["pair_semantic_judgment"][field]["score"] = value
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


def _negative(record: dict) -> None:
    raw = record["pair_semantic_judgment"]
    updates = {
        "a_can_replace_b": "no",
        "b_can_replace_a": "no",
        "relation_type": "related_non_duplicate",
        "material_difference": "major",
        "primary_material_difference": "other_material",
        "span_translation_status": "complete_faithful",
    }
    for key, value in updates.items():
        raw[key]["score"] = value
    raw["a_can_replace_b"]["reasoning"] = 'B001: "City Beta" is the removed-side phrase.'
    raw["b_can_replace_a"]["reasoning"] = 'A001: "City Alpha" is the removed-side phrase.'


def test_translation_negatives_are_not_inferred_harmless_from_coverage(c0_record: dict) -> None:
    _negative(c0_record)
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)
    c0_record["pair_semantic_judgment"]["span_translation_status"]["score"] = "not_translation"
    assert any("covered removed-side" in issue for issue in main_errors(c0_record, "pair_semantic_judgment", 240))


@pytest.mark.parametrize(
    "reasoning",
    [
        'A001: "City Alpha" is missing.',
        'B999: "City Beta" is missing.',
        'B001: "a fabricated obligation" is missing.',
        'B001: "Service hours" is missing.',
        'B001: "   " is missing.',
        'S001: "Service hours" is missing.',
    ],
)
def test_no_requires_existing_removed_side_citation_and_literal_quote(c0_record: dict, reasoning: str) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = reasoning
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


@pytest.mark.parametrize("delimiters", [('"', '"'), ("'", "'"), ("“", "”"), ("\u2018", "\u2019"), ("`", "`")])
def test_literal_loss_quotes_accept_typography_without_rewriting(c0_record: dict, delimiters: tuple[str, str]) -> None:
    _negative(c0_record)
    opening, closing = delimiters
    raw = c0_record["pair_semantic_judgment"]
    raw["a_can_replace_b"]["reasoning"] = f"B001: B's lost phrase is {opening}City Beta{closing}."
    raw["b_can_replace_a"]["reasoning"] = f"A001: A's lost phrase is {opening}City Alpha{closing}."
    decision = Decision.from_judge(raw).model_dump()
    c0_record.update(coverage_decision=decision, final_decision=decision, coverage_action="KEEP_MAIN")
    table = pa.Table.from_pylist([c0_record])
    output = C0AuditStage().process(DocumentBatch(dataset_name="audit", data=table)).to_pyarrow()
    assert output.select(table.column_names).to_pylist() == table.to_pylist()
    assert all(output.column(f"c0_{layer}_status").to_pylist() == ["valid"] for layer in ("main", "coverage", "final"))


@pytest.mark.parametrize(
    ("opening", "closing", "phrase"),
    [("'", "'", "City Beta's office"), ("\u2018", "\u2019", "City Beta\u2019s office")],
)
def test_single_quoted_loss_can_contain_an_apostrophe(
    c0_record: dict, opening: str, closing: str, phrase: str
) -> None:
    _negative(c0_record)
    c0_record["text_b"] = c0_record["text_b"].replace("City Beta", phrase)
    span = next(span for span in c0_record["semantic_diff"]["spans"] if span["span_id"] == "B001")
    span.update(text=phrase, end_char=len(c0_record["text_b"]), delta_end_char=len(c0_record["text_b"]))
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = (
        f"B001: B's lost phrase is {opening}{phrase}{closing}."
    )
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def test_nested_existing_quote_remains_valid(c0_record: dict) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = (
        "B001: 'The literal \"City Beta\" is absent from A.'"
    )
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def _display_with_context(record: dict) -> None:
    for span in record["semantic_diff"]["spans"]:
        if span["kind"] != "SHARED":
            text = record["text_" + span["kind"][0].lower()]
            span.update(start_char=0, end_char=len(text), text=text)


@pytest.mark.parametrize(
    "phrase", ["unchanged. City Beta", "unchanged. City", "Service hours and prices are unchanged. City Beta"]
)
def test_literal_quote_can_include_displayed_context_and_the_lost_delta(c0_record: dict, phrase: str) -> None:
    _negative(c0_record)
    _display_with_context(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = f"B001: '{phrase}' is missing."
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


@pytest.mark.parametrize(
    ("core", "phrase"), [("Straße", "unchanged. STRASSE"), ("City\nBeta", "unchanged. City Beta")]
)
def test_displayed_quote_matching_preserves_unicode_and_whitespace_normalization(
    c0_record: dict, core: str, phrase: str
) -> None:
    _negative(c0_record)
    c0_record["text_b"] = c0_record["text_b"].replace("City Beta", core)
    span = next(span for span in c0_record["semantic_diff"]["spans"] if span["span_id"] == "B001")
    span.update(delta_end_char=len(c0_record["text_b"]))
    _display_with_context(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = f"B001: '{phrase}' is missing."
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


@pytest.mark.parametrize(
    "phrase",
    ["Service hours and prices are unchanged.", "City Alpha", "unchanged. City Alpha", "a fabricated obligation"],
)
def test_displayed_padding_alone_and_other_side_quotes_remain_invalid(c0_record: dict, phrase: str) -> None:
    _negative(c0_record)
    _display_with_context(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = f"B001: '{phrase}' is missing."
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


def test_shared_context_with_the_same_words_as_the_delta_does_not_prove_a_loss(c0_record: dict) -> None:
    _negative(c0_record)
    shared = "City Beta is a shared example. "
    for side in ("A", "B"):
        core = "City Alpha" if side == "A" else "City Beta"
        c0_record["text_" + side.lower()] = shared + core
        span = next(span for span in c0_record["semantic_diff"]["spans"] if span["span_id"] == side + "001")
        span.update(
            start_char=len(shared),
            end_char=len(shared + core),
            text=core,
            delta_start_char=len(shared),
            delta_end_char=len(shared + core),
        )
    c0_record["semantic_diff"]["spans"][0].update(
        a_text=shared, b_text=shared, a_end_char=len(shared), b_end_char=len(shared)
    )
    _display_with_context(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = (
        "B001: 'City Beta is a shared example.' is missing."
    )
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


@pytest.mark.parametrize(
    "reasoning",
    [
        "B001: Beta's role isn't covered.",
        "B001: Beta\u2019s role isn\u2019t covered.",
        "B001: 'a fabricated obligation' is missing.",
        "B001: \u2018a fabricated obligation\u2019 is missing.",
        "B001: 'City Alpha' is missing.",
        "B001: \u2018City Alpha\u2019 is missing.",
        "B001: 'City Beta is missing.",
        "B001: \u2018City Beta is missing.",
        "B999: 'City Beta' is missing.",
        "A001: 'City Alpha' is missing.",
    ],
)
def test_single_quotes_still_require_a_literal_in_the_cited_removed_span(c0_record: dict, reasoning: str) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = reasoning
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


def test_complete_shared_quote_must_identify_removed_side(c0_record: dict) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = 'S001 (B): "Service hours" is lost.'
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def test_incomplete_packet_requires_fully_unresolved_summary(c0_record: dict) -> None:
    c0_record["truncated"] = True
    c0_record["semantic_diff"].update(truncated=True, truncated_a=True, status="INCOMPLETE_LIMIT")
    assert main_errors(c0_record, "pair_semantic_judgment", 240)
    raw = c0_record["pair_semantic_judgment"]
    for field in Decision.model_fields:
        raw[field]["score"] = "unresolved"
    raw["confidence_tier"]["score"] = "low"
    raw["primary_risk_factor"]["score"] = "extraction_or_payload_limit"
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def test_missing_ledger_and_cross_field_enums_are_invalid(c0_record: dict) -> None:
    raw = c0_record["pair_semantic_judgment"]
    del raw["span_shared_basis"]
    raw["primary_material_difference"]["score"] = "state_or_version"
    errors = main_errors(c0_record, "pair_semantic_judgment", 240)
    assert any("span_shared_basis" in error for error in errors)
    assert any("primary_material_difference" in error for error in errors)


def test_audit_retains_rows_raw_answers_and_all_layer_validity(c0_record: dict) -> None:
    decision = Decision.from_judge(c0_record["pair_semantic_judgment"]).model_dump()
    c0_record.update(coverage_decision=decision, final_decision=decision, coverage_action="KEEP_MAIN")
    bad = copy.deepcopy(c0_record)
    bad["pair_id"] = "invalid"
    bad["pair_semantic_judgment"]["span_b_delta"]["score"] = "same_record_content_extension"
    missing = copy.deepcopy(c0_record)
    missing.update(pair_id="missing", pair_semantic_judgment=None, coverage_decision=None, final_decision=None)
    table = pa.Table.from_pylist([c0_record, bad, missing])
    output = C0AuditStage().process(DocumentBatch(dataset_name="audit", data=table)).to_pyarrow()
    assert output.select(table.column_names).to_pylist() == table.to_pylist()
    assert [row["c0_final_status"] for row in output.to_pylist()] == ["valid", "invalid", "invalid"]
    assert all(
        output.column(f"c0_{layer}_status").to_pylist() == ["valid", "invalid", "invalid"]
        for layer in ("main", "coverage", "final")
    )


def test_unanswered_critic_is_invalid_without_erasing_main(c0_record: dict) -> None:
    decision = Decision.from_judge(c0_record["pair_semantic_judgment"]).model_dump()
    c0_record.update(coverage_decision=decision, final_decision=decision, coverage_action="UNVALIDATED_KEEP_MAIN")
    result = C0AuditStage()._audit(c0_record)
    assert result["c0_main_status"] == "valid"
    assert result["c0_coverage_status"] == result["c0_final_status"] == "invalid"
