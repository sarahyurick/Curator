# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from critics.c0 import C0_CONFIG, C0_PROMPTS, C0_ROOT, C0AuditStage
from critics.coverage import coverage_column
from critics.spans import SpanPacket
from critics.subject import subject_column, verifier_column
from data_designer.engine.processing.ginja.environment import UserTemplateSandboxEnvironment
from data_designer.engine.validation import ViolationLevel, validate_data_designer_config
from jinja2 import StrictUndefined

from nemo_curator.eval.llm_judge.workflow import build_config_builder


def test_c0_builds_with_native_data_designer_and_renders_main(c0_record: dict) -> None:
    raw = yaml.safe_load(C0_CONFIG.read_text())
    builder, _ = build_config_builder(
        C0_CONFIG,
        endpoint="http://127.0.0.1:9999/v1",
        models=raw["models"],
        judges=raw["execution"]["stages"][0]["judges"],
    )
    config = builder.build()
    assert config
    violations = validate_data_designer_config(
        columns=config.columns,
        processor_configs=config.processors or [],
        allowed_references=[column.name for column in config.columns] + list(c0_record),
    )
    assert not [violation for violation in violations if violation.level == ViolationLevel.ERROR]
    environment = UserTemplateSandboxEnvironment(allowed_references=list(c0_record), undefined=StrictUndefined)
    for name in ("system.jinja", "pair.jinja"):
        rendered = environment.render_template((C0_CONFIG.parent / name).read_text(), c0_record)
        assert rendered
    manifest = json.loads((C0_ROOT / "provenance.json").read_text())
    for path, checksum in manifest["sha256"].items():
        assert hashlib.sha256((C0_ROOT / path).read_bytes()).hexdigest() == checksum


def test_native_critic_columns_load_and_render_the_same_c0_policy(c0_record: dict) -> None:
    packet = SpanPacket.from_record(c0_record).for_prompt()
    context = {**c0_record, "coverage_packet": packet, "subject_packet": packet, "subject_candidate": {}}
    environment = UserTemplateSandboxEnvironment(allowed_references=list(context), undefined=StrictUndefined)
    for factory, name in (
        (coverage_column, "coverage"),
        (subject_column, "subject"),
        (verifier_column, "subject_verifier"),
    ):
        column = factory("judge", C0_PROMPTS)
        assert column.system_prompt == (C0_PROMPTS / f"{name}_system.jinja").read_text()
        assert column.prompt == (C0_PROMPTS / f"{name}_pair.jinja").read_text()
        assert column.skip is not None
        assert column.output_format
        assert environment.render_template(column.prompt, context)
        assert environment.render_template(column.system_prompt, context)


def test_main_prompt_marks_core_inside_padded_float_offset_spans(c0_record: dict) -> None:
    for span in c0_record["semantic_diff"]["spans"]:
        if span["kind"] == "SHARED":
            continue
        text = c0_record["text_" + span["kind"][0].lower()]
        span.update(start_char=0.0, end_char=float(len(text)), text=text)
        span["delta_start_char"] = float(span["delta_start_char"])
        span["delta_end_char"] = float(span["delta_end_char"])
    environment = UserTemplateSandboxEnvironment(allowed_references=list(c0_record), undefined=StrictUndefined)
    rendered = environment.render_template((C0_CONFIG.parent / "pair.jinja").read_text(), c0_record)
    assert '<own_side_core span_id="A001">City Alpha</own_side_core>' in rendered
    assert '<own_side_core span_id="B001">City Beta</own_side_core>' in rendered
    assert "Literal citation opener: A001: `Service hours and prices are unchanged. City Alpha`" in rendered
    assert "Literal citation opener: B001: `Service hours and prices are unchanged. City Beta`" in rendered


def test_c0_workflow_adds_audit_after_critics_without_changing_legacy(
    critic_module: ModuleType, tmp_path: Path
) -> None:
    options = {"judge_config": C0_CONFIG, "input_path": str(tmp_path / "in"), "output_path": str(tmp_path / "out")}
    legacy = critic_module.CriticWorkflow(**options)
    current = critic_module.CriticWorkflow(**options, c0=True, subject=True)
    assert legacy.critic_prompt_dir is None
    assert not any(isinstance(stage, C0AuditStage) for stage in legacy.postprocessing_stages)
    assert current.critic_prompt_dir == C0_PROMPTS
    assert isinstance(current.postprocessing_stages[-1], C0AuditStage)


def test_subject_pipeline_keeps_audit_last_and_uses_c0(
    critic_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(critic_module.ray, "init", lambda **_kwargs: None)
    workflow = critic_module.CriticWorkflow(
        judge_config=C0_CONFIG,
        input_path=str(tmp_path / "in"),
        output_path=str(tmp_path / "out"),
        c0=True,
        subject=True,
    )
    stages = workflow._build_judge_stages(endpoint="http://127.0.0.1:9999/v1")
    assert stages[0][0] == "coverage"
    assert isinstance(workflow.postprocessing_stages[-1], C0AuditStage)
    assert [stage.name for stage in workflow.postprocessing_stages][1:-1] == [
        "subject_prepare",
        "ndd_subject_review",
        "subject_apply",
        "ndd_subject_verifier_review",
        "subject_verifier_apply",
    ]
