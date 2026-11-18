# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only validation for the optional C0 span-retention policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import yaml

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from .models import Decision
from .spans import DEFAULT_MAX_EVIDENCE_CHARS, SpanPacket
from .stage_utils import replace_columns

C0_ROOT = Path(__file__).resolve().parent / "policies" / "c0"
C0_CONFIG = C0_ROOT / "judge_config" / "fuzzy_pair_judge.yaml"
C0_PROMPTS = C0_ROOT / "critics"
_MATERIAL_DELTAS = {
    "same_record_content_extension",
    "record_identity_role_or_state_change",
    "material_non_main_message_change",
    "other_substantive_content",
}
_COVERED_DELTAS = {"none", "semantically_covered", "universal_ui_or_repetition"}
_VERIFIED_BASES = {"verified_substantive_record", "verified_equivalent_non_main_message"}
_DIRECTIONS = (("a_can_replace_b", "B"), ("b_can_replace_a", "A"))


@lru_cache(maxsize=1)
def _score_options() -> dict[str, dict]:
    config = yaml.safe_load(C0_CONFIG.read_text(encoding="utf-8"))
    scores = config["execution"]["stages"][0]["judges"][0]["scores"]
    return {score["name"]: score["options"] for score in scores}


def _ledger_issues(values: dict) -> list[str]:
    """Apply the literal coherence constraints used by the frozen C0 delivery audit."""
    errors = []
    for field, side in _DIRECTIONS:
        delta = "span_" + side.lower() + "_delta"
        if values[field] == "yes" and values[delta] in _MATERIAL_DELTAS:
            errors.append(f"{field}: uncovered removed-side contribution")
    if any(values[field] == "yes" for field, _ in _DIRECTIONS):
        if values["span_hard_conflict"] not in {"none", "unresolved"}:
            errors.append("yes with material hard conflict")
        if values["span_shared_basis"] not in _VERIFIED_BASES:
            errors.append("yes without verified basis")
    if (
        values["span_hard_conflict"] == "none"
        and values["span_shared_basis"] in _VERIFIED_BASES
        and values["span_translation_status"] == "not_translation"
    ):
        for field, side in _DIRECTIONS:
            if values[field] == "no" and values["span_" + side.lower() + "_delta"] in _COVERED_DELTAS:
                errors.append(f"{field}: covered removed-side contribution")
    return errors


def _context_quote_covers_delta(phrase: str, displayed: dict) -> bool:
    """A displayed context quote is evidence only if it overlaps the cited core difference."""
    start = int(displayed["delta_start_char"]) - int(displayed["start_char"])
    end = int(displayed["delta_end_char"]) - int(displayed["start_char"])
    normalized, positions = [], []
    for token in re.finditer(r"\S+", displayed["text"]):
        if normalized:
            normalized.append(" ")
            positions.append(token.start())
        for offset, char in enumerate(token.group()):
            folded = char.casefold()
            normalized.append(folded)
            positions.extend([token.start() + offset] * len(folded))
    source = "".join(normalized)
    cursor = source.find(phrase)
    while cursor >= 0:
        if positions[cursor] < end and positions[cursor + len(phrase) - 1] >= start:
            return True
        cursor = source.find(phrase, cursor + 1)
    return False


def _loss_evidence_errors(raw: dict, packet: SpanPacket, displayed_spans: dict[str, dict]) -> list[str]:
    errors = []
    for field, side in _DIRECTIONS:
        if raw[field]["score"] != "no":
            continue
        reasoning = raw[field].get("reasoning") or ""
        match = re.match(r"\s*\[?([ABS]\d{3})\b", reasoning)
        span = packet.spans.get(match[1]) if match else None
        excerpt = span.excerpt(side) if span else None
        if excerpt is None or (span.kind == "SHARED" and not re.search(rf"\b{side}\b", reasoning)):
            errors.append(f"{field}: missing or wrong-side loss citation")
            continue
        quotes = re.findall(r'"([^"\n]+)"|“([^”\n]+)”|`([^`\n]+)`', reasoning)
        # Keep nested double quotes visible and avoid treating word apostrophes as delimiters.
        quotes.extend(re.findall(r"(?<!\w)'([^\n]+?)'(?!\w)|\u2018([^\n]+?)\u2019(?!\w)", reasoning))
        phrases = [" ".join(next(part for part in quote if part).split()).casefold() for quote in quotes]
        source = " ".join(excerpt.text.split()).casefold()
        if not any(
            phrase
            and (
                phrase in source
                or (span.kind != "SHARED" and _context_quote_covers_delta(phrase, displayed_spans[span.span_id]))
            )
            for phrase in phrases
        ):
            errors.append(f"{field}: missing literal loss quote in cited span")
    return errors


def main_errors(record: dict, source_judge: str, max_evidence_chars: int) -> list[str]:
    """Check the native fifteen-score response without rewriting any answers."""
    raw = record.get(source_judge)
    if not isinstance(raw, dict):
        return ["missing main judgment"]
    values, errors = {}, []
    for name, options in _score_options().items():
        value = raw.get(name)
        score = value.get("score") if isinstance(value, dict) else None
        if not isinstance(score, str) or score not in options:
            errors.append(f"{name}: missing or invalid score")
        else:
            values[name] = score
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("reasoning"), str)
            or not value["reasoning"].strip()
        ):
            errors.append(f"{name}: missing reasoning")
    if errors:
        return errors
    decision = Decision.from_judge(raw)
    errors.extend(decision.inconsistencies())
    errors.extend(_ledger_issues(values))
    try:
        packet = SpanPacket.from_record(record, max_evidence_chars=max_evidence_chars)
    except ValueError as error:
        return [*errors, f"invalid source packet: {error}"]
    unresolved = decision.a_can_replace_b == decision.b_can_replace_a == "unresolved"
    missing = any(not record[key].strip() for key in ("text_a", "text_b"))
    if (not packet.complete or missing) and not unresolved:
        errors.append("incomplete or empty source requires fully unresolved summary")
    errors.extend(
        _loss_evidence_errors(raw, packet, {span["span_id"]: span for span in record["semantic_diff"]["spans"]})
    )
    return errors


@dataclass
class C0AuditStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Annotate main, coverage and final validity while retaining raw rows and decisions."""

    source_judge: str = "pair_semantic_judgment"
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    name: str = "c0_audit"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "text_a",
            "text_b",
            "semantic_diff",
            "truncated",
            self.source_judge,
            "coverage_decision",
            "final_decision",
            "coverage_action",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            f"c0_{layer}_{suffix}" for layer in ("main", "coverage", "final") for suffix in ("status", "errors")
        ]

    def _audit(self, record: dict) -> dict:
        errors = main_errors(record, self.source_judge, self.max_evidence_chars)
        result = {}
        for layer, column in (
            ("main", self.source_judge),
            ("coverage", "coverage_decision"),
            ("final", "final_decision"),
        ):
            found = list(errors)
            try:
                value = (
                    Decision.from_judge(record.get(column))
                    if layer == "main"
                    else Decision.model_validate(record.get(column))
                )
                found.extend(value.inconsistencies())
            except ValueError:
                value = None
                found.append(f"missing or malformed {layer} decision")
            if layer != "main" and str(record.get("coverage_action", "")).startswith("UNVALIDATED"):
                found.append("unvalidated coverage review")
            if layer == "final" and str(record.get("subject_action", "")).startswith("UNVALIDATED"):
                found.append("unvalidated subject review")
            result[f"c0_{layer}_errors"] = sorted(set(found))
            result[f"c0_{layer}_status"] = (
                "invalid"
                if found
                else "unresolved"
                if "unresolved" in (value.a_can_replace_b, value.b_can_replace_a)
                else "valid"
            )
        return result

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        clashes = set(self.outputs()[1]) & set(table.column_names)
        if clashes:
            msg = f"{self.name} would overwrite input columns: {sorted(clashes)}"
            raise ValueError(msg)
        audits = [self._audit(record) for record in table.to_pylist()]
        columns = {
            name: pa.array(
                [audit[name] for audit in audits],
                type=pa.list_(pa.string()) if name.endswith("errors") else pa.string(),
            )
            for name in self.outputs()[1]
        }
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=replace_columns(table, columns),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
