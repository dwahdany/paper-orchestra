from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from paper_orchestra.agents.outline import (
    OUTLINE_PAGE,
    Outline,
    OutlineAgent,
    normalize_aspect_ratio,
    normalize_data_source,
    normalize_outline,
    normalize_plot_type,
)
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import Capabilities, LLMParseError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT
from paper_orchestra.prompts.outline import ASPECT_RATIOS, OUTLINE_SYSTEM_PROMPT

TEMPLATE = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\section{Introduction}\nIntroduction here.\n\\end{document}\n"
)

OUTLINE_KEYS = {"plotting_plan", "intro_related_work_plan", "section_plan"}


def model_outline() -> dict:
    """What a model might return: right shape, sloppy values, a few stray keys."""
    return {
        "plotting_plan": [
            {
                "figure_id": "Fig Main Results!",
                "title": "Main results",
                "plot_type": "Bar Chart",
                "data_source": "experimental_log.md",
                "objective": "Compare accuracy across baselines.",
                "aspect_ratio": "16 : 9",
                "caption": "a stray key",
            },
            {
                "figure_id": "fig_main_results",
                "title": "Overview",
                "plot_type": "architecture diagram",
                "data_source": "idea.md and experimental_log.md",
                "objective": "Show the architecture diagram of the method.",
                "aspect_ratio": "2:1",
            },
            {
                "figure_id": "",
                "title": "Ablation Study",
                "plot_type": "plot",
                "objective": "Line plot of the ablation.",
            },
        ],
        "intro_related_work_plan": {
            "introduction_strategy": {
                "hook_hypothesis": "Hook.",
                "problem_gap_hypothesis": "Gap.",
                "search_directions": ["surveys on widgets"],
            },
            "related_work_strategy": {
                "overview": "Two clusters.",
                "subsections": [
                    {
                        "subsection_title": "2.1 Widget Models",
                        "methodology_cluster": "Widgets",
                        "sota_investigation_mission": "Find widget models.",
                        "limitation_hypothesis": "They drift.",
                        "limitation_search_queries": ["widget drift"],
                        "bridge_to_our_method": "We do not drift.",
                    }
                ],
            },
        },
        "section_plan": [
            {
                "section_title": "Abstract",
                "content_bullets": ["State the problem."],
                "citation_hints": ["research paper introducing 'WidgetNet'"],
            },
            {
                "section_title": "3. Method",
                "subsections": [
                    {
                        "subsection_title": "3.1 Core",
                        "content_bullets": ["Define the loss."],
                        "citation_hints": [
                            "Kingma et al. (Adam: A Method for Stochastic Optimization)"
                        ],
                        "estimated_words": 400,
                    },
                    {"subsection_title": "3.2 Details"},
                ],
            },
        ],
        "notes": "a stray top-level key",
    }


def make_ctx(tmp_path: Path, client: FakeLLM, *, memory: bool = True, **cfg) -> RunContext:
    base = tmp_path / "run"
    paths = RunPaths(base_dir=base)
    for path, text in (
        (paths.idea, "We propose WidgetNet."),
        (paths.experimental_log, "WidgetNet reaches 91.2% on WidgetBench."),
        (paths.template_tex, TEMPLATE),
        (paths.guidelines, "Eight pages. Double blind."),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = PipelineConfig(
        base_dir=base, template_dir=tmp_path / "template", research_cutoff="2024-10", **cfg
    )
    if memory:
        harness = ContextHarness(
            project=Wiki(base / "memory"),
            transcripts=TranscriptLog(base / "memory" / "transcripts"),
            client_for=lambda model: client,
        )
    else:
        harness = ContextHarness(client_for=lambda model: client)
    return RunContext(cfg=config, paths=paths, harness=harness, compute=LocalBackend())


# -- normalization -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plot", "plot"),
        (" Diagram ", "diagram"),
        ("Bar Chart", "plot"),
        ("line graph", "plot"),
        ("heatmap", "plot"),
        ("architecture diagram", "diagram"),
        ("computational graph diagram", "diagram"),
        ("schematic", "diagram"),
        ("", "diagram"),
    ],
)
def test_normalize_plot_type(value, expected):
    assert normalize_plot_type(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("16:9", "16:9"),
        (" 16 : 9 ", "16:9"),
        ("16x9", "16:9"),
        ("32/18", "16:9"),
        ("1:1", "1:1"),
        ("2:1", "16:9"),
        ("3:1", "21:9"),
        ("1:3", "1:4"),
        ("wide", "16:9"),
        ("", "16:9"),
        ("0:9", "16:9"),
    ],
)
def test_normalize_aspect_ratio(value, expected):
    assert normalize_aspect_ratio(value) == expected
    assert normalize_aspect_ratio(value) in ASPECT_RATIOS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("idea.md", "idea.md"),
        ("Idea", "idea.md"),
        ("experimental_log.md", "experimental_log.md"),
        ("the experiment log", "experimental_log.md"),
        ("both", "both"),
        ("idea.md and experimental_log.md", "both"),
        ("", "both"),
        ("somewhere else", "both"),
    ],
)
def test_normalize_data_source(value, expected):
    assert normalize_data_source(value) == expected


def test_normalize_outline_makes_figure_ids_safe_and_unique():
    outline = normalize_outline(
        {
            "plotting_plan": [
                {"figure_id": "Fig: A/B", "plot_type": "plot", "objective": "x"},
                {"figure_id": "fig_a_b", "plot_type": "plot", "objective": "x"},
                {"figure_id": "FIG A B", "plot_type": "plot", "objective": "x"},
                {"figure_id": "../../etc", "plot_type": "plot", "objective": "x"},
                {"figure_id": "  ", "title": "", "plot_type": "plot", "objective": "x"},
            ]
        }
    )
    ids = [figure["figure_id"] for figure in outline["plotting_plan"]]
    assert ids == ["fig_a_b", "fig_a_b_2", "fig_a_b_3", "etc", "fig_5"]


# -- prompt ------------------------------------------------------------------


def test_the_prompt_example_matches_the_schema_and_has_single_braces():
    start = OUTLINE_SYSTEM_PROMPT.index("```json\n") + len("```json\n")
    example = json.loads(OUTLINE_SYSTEM_PROMPT[start : OUTLINE_SYSTEM_PROMPT.index("```", start)])
    validated = Outline.model_validate(example).model_dump()
    assert validated == example  # the schema loses nothing of the documented shape
    assert "{{" not in OUTLINE_SYSTEM_PROMPT and "}}" not in OUTLINE_SYSTEM_PROMPT
    assert "\\section{...}" in OUTLINE_SYSTEM_PROMPT
    assert "{cutoff_date}" not in OUTLINE_SYSTEM_PROMPT
    for ratio in ASPECT_RATIOS:
        assert f'"{ratio}"' in OUTLINE_SYSTEM_PROMPT
    assert OUTLINE_SYSTEM_PROMPT.endswith(UNIVERSAL_NO_LEAKAGE_PROMPT)


def test_the_schema_stays_within_the_structured_output_limits():
    schema = Outline.model_json_schema()
    nodes = [schema, *schema["$defs"].values()]
    optional = sum(len(set(node["properties"]) - set(node.get("required", []))) for node in nodes)
    assert optional <= 24  # more optional fields than this and the API rejects the schema
    assert "additionalProperties" not in json.dumps(schema)  # no dict[str, X] fields


# -- agent -------------------------------------------------------------------


def test_run_writes_a_validated_normalized_outline(tmp_path):
    requests: list[FakeRequest] = []

    def responder(request: FakeRequest) -> str:
        requests.append(request)
        return json.dumps(model_outline())

    ctx = make_ctx(tmp_path, FakeLLM(responder))
    outline = OutlineAgent(ctx).run()

    assert isinstance(outline, dict) and set(outline) == OUTLINE_KEYS
    assert json.loads(ctx.paths.outline.read_text(encoding="utf-8")) == outline

    figures = outline["plotting_plan"]
    assert [f["figure_id"] for f in figures] == [
        "fig_main_results",
        "fig_main_results_2",
        "ablation_study",
    ]
    assert [f["plot_type"] for f in figures] == ["plot", "diagram", "plot"]
    assert [f["aspect_ratio"] for f in figures] == ["16:9", "16:9", "16:9"]
    assert [f["data_source"] for f in figures] == ["experimental_log.md", "both", "both"]
    # A chart type given as plot_type is kept where the plotting agent looks for it.
    assert figures[0]["objective"] == (
        "Compare accuracy across baselines. (requested as: Bar Chart)"
    )
    assert figures[1]["objective"] == "Show the architecture diagram of the method."
    assert set(figures[0]) == {
        "figure_id",
        "title",
        "plot_type",
        "data_source",
        "objective",
        "aspect_ratio",
    }

    plan = outline["intro_related_work_plan"]
    assert plan["introduction_strategy"]["search_directions"] == ["surveys on widgets"]
    assert plan["related_work_strategy"]["subsections"][0]["limitation_search_queries"] == [
        "widget drift"
    ]

    abstract, method = outline["section_plan"]
    # Content given at section level is kept as a subsection instead of being dropped.
    assert abstract == {
        "section_title": "Abstract",
        "subsections": [
            {
                "subsection_title": "Abstract",
                "content_bullets": ["State the problem."],
                "citation_hints": ["research paper introducing 'WidgetNet'"],
            }
        ],
    }
    assert method["subsections"][1] == {
        "subsection_title": "3.2 Details",
        "content_bullets": [],
        "citation_hints": [],
    }
    assert "estimated_words" not in method["subsections"][0]

    (request,) = requests
    assert request.model == ctx.cfg.models.writer
    assert request.label == "outline.OutlineAgent"
    assert request.schema is Outline
    assert request.system.startswith("You are a senior AI researcher")
    assert "Strict Knowledge Isolation" in request.system
    assert "# Memory" in request.system  # the main authoring call reads memory
    for tag, content in (
        ("idea", "We propose WidgetNet."),
        ("experimental_log", "WidgetNet reaches 91.2% on WidgetBench."),
        ("template", TEMPLATE.strip()),
        ("guidelines", "Eight pages. Double blind."),
    ):
        assert f"<{tag}>\n{content}\n</{tag}>" in request.text
    assert "<research_cutoff>2024-10</research_cutoff>" in request.text
    assert "senior AI researcher" not in request.text


def test_run_records_the_outline_in_memory(tmp_path):
    ctx = make_ctx(tmp_path, FakeLLM(lambda request: json.dumps(model_outline())))
    outline = OutlineAgent(ctx).run()

    page = ctx.harness.project.read(OUTLINE_PAGE)
    assert page.kind == "outline"
    assert "3 planned figure(s), 2 section(s), 2 citation hint(s)" in page.summary
    fenced = page.body.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(fenced) == outline
    assert "[[paper/outline]]" in ctx.harness.project.rebuild_index()

    events = [e for e in ctx.harness.transcripts.events() if e["type"] == "outline"]
    assert len(events) == 1
    assert (events[0]["stage"], events[0]["figures"], events[0]["sections"]) == ("outline", 3, 2)


def test_run_works_with_memory_disabled(tmp_path):
    client = FakeLLM(lambda request: json.dumps(model_outline()))
    ctx = make_ctx(tmp_path, client, memory=False)
    outline = OutlineAgent(ctx).run()
    assert ctx.paths.outline.is_file() and len(outline["section_plan"]) == 2
    assert "# Memory" not in client.requests[0].system
    assert not (ctx.paths.base_dir / "memory").exists()


def test_run_falls_back_to_text_json_without_native_structured_output(tmp_path):
    def responder(request: FakeRequest) -> str:
        assert request.schema is None
        assert "JSON Schema" in request.text and '"plotting_plan"' in request.text
        return "Here is the outline.\n```json\n" + json.dumps(model_outline()) + "\n```"

    client = FakeLLM(
        responder,
        capabilities=Capabilities(
            tools=False, memory_tool=False, web_search=False, structured_output=False
        ),
    )
    outline = OutlineAgent(make_ctx(tmp_path, client)).run()
    assert len(outline["plotting_plan"]) == 3


def test_an_unparseable_answer_is_asked_for_again(tmp_path):
    client = FakeLLM(["not json at all", '{"plotting_plan": []}', json.dumps(model_outline())])
    outline = OutlineAgent(make_ctx(tmp_path, client)).run()
    assert len(client.requests) == 3
    assert outline["section_plan"][1]["section_title"] == "3. Method"


def test_run_fails_when_the_model_never_returns_an_outline(tmp_path):
    client = FakeLLM(["nope"] * 3)
    ctx = make_ctx(tmp_path, client)
    with pytest.raises(LLMParseError):
        OutlineAgent(ctx).run()
    assert not ctx.paths.outline.exists()


def test_an_outline_without_sections_is_rejected(tmp_path):
    empty = model_outline() | {"section_plan": []}
    ctx = make_ctx(tmp_path, FakeLLM(lambda request: json.dumps(empty)))
    with pytest.raises(RuntimeError, match="empty section_plan"):
        OutlineAgent(ctx).run()
    assert not ctx.paths.outline.exists()


@pytest.mark.parametrize("missing", ["idea", "experimental_log", "template_tex", "guidelines"])
def test_a_missing_input_fails_before_any_model_call(tmp_path, missing):
    client = FakeLLM(lambda request: json.dumps(model_outline()))
    ctx = make_ctx(tmp_path, client)
    path = getattr(ctx.paths, missing)
    path.unlink()
    with pytest.raises(FileNotFoundError, match=path.name):
        OutlineAgent(ctx).run()
    assert client.requests == []


def test_a_malformed_cutoff_fails_before_any_model_call(tmp_path):
    client = FakeLLM(lambda request: json.dumps(model_outline()))
    ctx = make_ctx(tmp_path, client)
    ctx.cfg.research_cutoff = "2024/10"
    with pytest.raises(ValueError, match="YYYY-MM"):
        OutlineAgent(ctx).run()
    assert client.requests == []


def test_run_logs_without_printing(tmp_path, caplog, capsys):
    caplog.set_level(logging.INFO, logger="paper_orchestra")
    ctx = make_ctx(tmp_path, FakeLLM(lambda request: json.dumps(model_outline())))
    OutlineAgent(ctx).run()
    assert "Outline written to" in caplog.text
    assert capsys.readouterr().out == ""
