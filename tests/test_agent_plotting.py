from __future__ import annotations

import ast
import base64
import inspect
import io
import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image as PILImage

from paper_orchestra.agents import plotting
from paper_orchestra.agents.plotting import (
    FIGURE_FILE,
    SAVE_FIGURE_EPILOGUE,
    FewShotSelection,
    FigureCritique,
    PlottingAgent,
    figure_size,
    normalize_aspect_ratio,
    plan_figures,
    run_figure_code,
    task_name,
    to_png,
)
from paper_orchestra.compute.base import CompileResult, ComputeBackend, ExecResult
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import PAPER_BANANA_ENV, ModelConfig, PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import Capabilities, LLMConfigError, ToolError
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.prompts import plotting as prompts

LOGGER = "paper_orchestra.agents.plotting"

CODE = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])"
ACCEPT = json.dumps(
    {"needs_revision": False, "critic_suggestions": "No changes needed.", "revised_description": ""}
)


def image_bytes(color: str = "white", fmt: str = "PNG", size: tuple[int, int] = (64, 36)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


PNG = image_bytes()


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


def revise(suggestions: str, description: str) -> str:
    return json.dumps(
        {
            "needs_revision": True,
            "critic_suggestions": suggestions,
            "revised_description": description,
        }
    )


def rendered(png: bytes = PNG) -> ExecResult:
    return ExecResult(returncode=0, files={FIGURE_FILE: png})


def crashed(stderr: str = "Traceback\nNameError: name 'sns' is not defined") -> ExecResult:
    return ExecResult(returncode=1, stderr=stderr)


class FakeCompute(ComputeBackend):
    """Records what it is asked to run and answers from a script; runs nothing."""

    name = "fake"

    def __init__(self, behaviour: Callable[[str, int], ExecResult] | None = None) -> None:
        self._behaviour = behaviour or (lambda code, call: rendered())
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0) -> ExecResult:
        with self._lock:
            self.calls.append(
                {"code": code, "inputs": inputs, "outputs": list(outputs), "timeout": timeout}
            )
            call = len(self.calls)
        return self._behaviour(code, call)

    def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0) -> CompileResult:
        raise AssertionError("the plotting agent does not compile LaTeX")


def title_of(request: FakeRequest) -> str:
    return re.search(r"Title: (.+)", request.text).group(1)


def first_turn_text(request: FakeRequest) -> str:
    """The opening user message of the conversation ``request`` belongs to."""
    return request.history[0][0] if request.history else request.text


def scripted_llm(**overrides) -> FakeLLM:
    """A model that plans, styles, codes, accepts and captions; override any step by name."""
    script = {
        "plotting.plan": lambda r: f"**Detailed Description:** plan for {title_of(r)}",
        "plotting.style": lambda r: f"Detailed Description: styled plan for {title_of(r)}",
        "plotting.code": fenced(CODE),
        "plotting.critic": ACCEPT,
        "plotting.caption": lambda r: f"**Caption:** Caption of {title_of(r)}.",
    }
    script.update({f"plotting.{step}": reply for step, reply in overrides.items()})
    return FakeLLM(script)


def make_ctx(
    tmp_path: Path,
    llm: FakeLLM,
    compute: ComputeBackend | None = None,
    *,
    image_llm: FakeLLM | Exception | None = None,
    memory: bool = True,
    **cfg_options,
) -> RunContext:
    base = tmp_path / "run"
    paths = RunPaths(base_dir=base)
    paths.materials_dir.mkdir(parents=True)
    paths.idea.write_text("IDEA: we propose FooNet.\n", encoding="utf-8")
    paths.experimental_log.write_text("LOG: FooNet 91.2, baseline 85.0.\n", encoding="utf-8")
    models = ModelConfig(
        writer="fake",
        reflection="fake",
        plotting="fake",
        fast="fake",
        image="fake-image" if image_llm is not None else None,
    )
    cfg = PipelineConfig(
        base_dir=base, template_dir=tmp_path / "template", models=models, **cfg_options
    )

    def client_for(model: str):
        if model != "fake-image":
            return llm
        if isinstance(image_llm, Exception):
            raise image_llm
        return image_llm

    if memory:
        harness = ContextHarness(
            project=Wiki(base / "memory"),
            transcripts=TranscriptLog(base / "memory" / "transcripts"),
            client_for=client_for,
        )
    else:
        harness = ContextHarness(client_for=client_for)
    return RunContext(cfg=cfg, paths=paths, harness=harness, compute=compute or FakeCompute())


def figure(figure_id: str = "fig_accuracy", plot_type: str = "plot", **extra) -> dict:
    return {
        "figure_id": figure_id,
        "title": f"Title {figure_id}",
        "plot_type": plot_type,
        "data_source": "experimental_log.md",
        "objective": "Compare the accuracy of all methods",
        "aspect_ratio": "4:3",
        **extra,
    }


def requests_for(llm: FakeLLM, label: str) -> list[FakeRequest]:
    return [request for request in llm.requests if request.label == label]


def notes(ctx: RunContext) -> list:
    wiki = ctx.harness.project
    return [wiki.read(info.path) for info in wiki.pending_notes()]


# -- the plot path and its outputs -------------------------------------------------


def test_plot_is_planned_styled_rendered_reviewed_and_captioned(tmp_path):
    llm = scripted_llm()
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, llm, compute)

    results = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert llm.labels() == [
        "plotting.plan",
        "plotting.style",
        "plotting.code",
        "plotting.critic",
        "plotting.caption",
    ]
    assert results == [
        {
            "figure_id": "fig_accuracy",
            "title": "Title fig_accuracy",
            "task_name": "plot",
            "description": "styled plan for Title fig_accuracy",
            "caption": "Caption of Title fig_accuracy.",
            "aspect_ratio": "4:3",
            "critic_history": [
                {
                    "round": 0,
                    "suggestions": "No changes needed.",
                    "revised_description": "styled plan for Title fig_accuracy",
                    "outcome": "accepted",
                }
            ],
            "image_path": "figures/fig_accuracy.png",
        }
    ]

    paths = ctx.paths
    assert json.loads(paths.plotting_results.read_text(encoding="utf-8")) == results
    assert "base64" not in paths.plotting_results.read_text(encoding="utf-8")
    generated = paths.base_dir / results[0]["image_path"]
    assert generated == paths.generated_figures_dir / "fig_accuracy.png"
    assert generated.read_bytes() == PNG
    assert (paths.figures_dir / "fig_accuracy.png").read_bytes() == PNG
    assert json.loads(paths.figures_info.read_text(encoding="utf-8")) == [
        {"name": "fig_accuracy.png", "caption": "Caption of Title fig_accuracy."}
    ]

    # The model's script reaches the compute backend with the save epilogue appended.
    (call,) = compute.calls
    assert call["code"] == CODE + "\n" + SAVE_FIGURE_EPILOGUE
    assert call["outputs"] == [FIGURE_FILE]
    assert call["inputs"] is None
    assert call["timeout"] == plotting.CODE_TIMEOUT_SECONDS

    plan, style, code, critic, caption = llm.requests
    assert "<raw_data>\n<experimental_log>\nLOG: FooNet 91.2" in plan.text
    assert "<idea>" not in plan.text
    assert "<visual_intent>\nTitle: Title fig_accuracy\nObjective: Compare" in plan.text
    assert plan.system == prompts.PLOT_PLANNER_SYSTEM and not plan.images
    assert "<detailed_description>\nplan for Title fig_accuracy\n</detailed_description>" in (
        style.text
    )
    assert prompts.DEFAULT_PLOT_STYLE_GUIDE.strip() in style.text
    assert "<figure_description>\nstyled plan for Title fig_accuracy\n" in code.text
    assert "aspect ratio of 4:3" in code.text and "figsize=(8.5, 6.4)" in code.text
    assert "statistical plot illustrator" in code.system
    assert "it is stopped after 120 seconds" in code.system
    assert [image.data for image in critic.images] == [PNG]
    assert critic.schema is FigureCritique and critic.system == prompts.PLOT_CRITIC_SYSTEM
    assert "Target plot for critique:" in critic.text
    assert [image.data for image in caption.images] == [PNG]
    assert "<task_type>\nplot\n</task_type>" in caption.text

    page = ctx.harness.project.read("wiki/figures/fig_accuracy.md")
    assert page.meta["kind"] == "figure" and page.meta["file"] == "figures/fig_accuracy.png"
    assert "Caption of Title fig_accuracy." in page.meta["summary"]
    assert "## Caption\n\nCaption of Title fig_accuracy." in page.body
    assert "## Description\n\nstyled plan for Title fig_accuracy" in page.body
    assert "matplotlib code written by fake and run by the 'fake' compute backend" in page.body
    assert "planned zero-shot" in page.body and "round 1: accepted" in page.body


def test_only_the_code_call_reads_memory(tmp_path):
    seen: dict[str, str] = {}

    def code(request: FakeRequest) -> str:
        with pytest.raises(ToolError):
            request.memory.create("/memories/notes/x.md", "the code call may not write")
        seen["lesson"] = request.memory.view("/memories/wiki/lessons/figure-code.md")
        return fenced(CODE)

    llm = scripted_llm(code=code)
    ctx = make_ctx(tmp_path, llm)
    ctx.harness.write_page(
        "wiki/lessons/figure-code.md",
        "Importing seaborn fails in the figure sandbox; use matplotlib only.",
        {"summary": "seaborn is not installed in the figure sandbox", "kind": "lesson"},
    )

    assert len(PlottingAgent(ctx).run({"plotting_plan": [figure()]})) == 1

    (code_request,) = requests_for(llm, "plotting.code")
    assert "# Memory" in code_request.system
    assert "seaborn is not installed in the figure sandbox" in code_request.system
    assert "Memory is read-only in this session." in code_request.system
    assert "Importing seaborn fails" in seen["lesson"]
    others = [request for request in llm.requests if request.label != "plotting.code"]
    assert len(others) == 4
    for request in others:
        assert request.memory is None and not request.tools
        assert "# Memory" not in request.system


def test_runs_without_memory(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm(), memory=False)
    results = PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    assert [result["figure_id"] for result in results] == ["fig_accuracy"]
    assert (ctx.paths.figures_dir / "fig_accuracy.png").is_file()
    assert not ctx.paths.memory_dir.exists()


def test_data_source_selects_the_material(tmp_path):
    llm = scripted_llm()
    ctx = make_ctx(tmp_path, llm, figure_workers=1)
    plan = [
        figure("fig_idea", data_source="idea.md"),
        figure("fig_both", data_source="both"),
        figure("fig_unknown", data_source="appendix"),
    ]
    assert len(PlottingAgent(ctx).run({"plotting_plan": plan})) == 3
    idea_only, both, unknown = requests_for(llm, "plotting.plan")
    assert "<idea>\nIDEA: we propose FooNet.\n</idea>" in idea_only.text
    assert "<experimental_log>" not in idea_only.text
    for request in (both, unknown):
        assert "<idea>" in request.text and "<experimental_log>" in request.text


def test_empty_plan_writes_empty_results_and_keeps_provided_figures(tmp_path):
    llm = scripted_llm()
    ctx = make_ctx(tmp_path, llm)
    provided = [{"name": "teaser.png", "caption": "A provided teaser."}]
    ctx.paths.figures_dir.mkdir(parents=True)
    ctx.paths.figures_info.write_text(json.dumps(provided), encoding="utf-8")

    assert PlottingAgent(ctx).run({}) == []
    assert PlottingAgent(ctx).run({"plotting_plan": []}) == []
    assert json.loads(ctx.paths.plotting_results.read_text(encoding="utf-8")) == []
    assert json.loads(ctx.paths.figures_info.read_text(encoding="utf-8")) == provided
    assert llm.requests == []


def test_malformed_plan(tmp_path, caplog):
    ctx = make_ctx(tmp_path, scripted_llm())
    with pytest.raises(ValueError, match="must be a list"):
        PlottingAgent(ctx).run({"plotting_plan": {"figure_id": "x"}})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = PlottingAgent(ctx).run({"plotting_plan": ["not an object", figure()]})
    assert [result["figure_id"] for result in results] == ["fig_accuracy"]
    assert "Entry 1 of the plotting plan is not an object" in caplog.text


def test_unusable_plotting_model_fails_the_stage(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm())

    def no_client(model: str):
        raise LLMConfigError("no credentials for the plotting model")

    ctx.harness._client_for = no_client
    with pytest.raises(LLMConfigError, match="no credentials"):
        PlottingAgent(ctx).run({"plotting_plan": [figure()]})


# -- diagrams ----------------------------------------------------------------------


def test_diagram_is_drawn_with_code_without_an_image_model(tmp_path):
    llm = scripted_llm()
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, llm, compute)

    (result,) = PlottingAgent(ctx).run(
        {"plotting_plan": [figure("fig_overview", "diagram", data_source="idea.md")]}
    )

    assert result["task_name"] == "diagram" and result["image_path"] == "figures/fig_overview.png"
    assert len(compute.calls) == 1 and llm.image_prompts == []
    plan, style, code, critic, _ = llm.requests
    assert plan.system == prompts.DIAGRAM_PLANNER_SYSTEM
    assert "<methodology_section>\n<idea>\nIDEA: we propose FooNet." in plan.text
    assert "<diagram_caption>\nTitle: Title fig_overview" in plan.text
    for request in (plan, style, critic):
        assert prompts.DIAGRAM_AS_CODE_NOTE in request.text
    assert prompts.DEFAULT_DIAGRAM_STYLE_GUIDE.strip() in style.text
    # The dedicated diagram prompt: patches, arrows and labels, no external assets.
    assert "FancyBboxPatch" in code.system and "FancyArrowPatch" in code.system
    assert "No external images" in code.system
    assert "statistical plot illustrator" not in code.system
    assert "Draw the diagram described above" in code.text
    assert critic.system == prompts.DIAGRAM_CRITIC_SYSTEM
    assert "Target diagram for critique:" in critic.text


def test_diagram_uses_the_image_model_when_it_can_generate_images(tmp_path):
    jpeg = image_bytes("navy", "JPEG")
    image_llm = FakeLLM({}, image=jpeg)
    diagram_replies = iter([revise("Add the decoder.", "plan with a decoder"), ACCEPT])
    llm = scripted_llm(
        critic=lambda r: next(diagram_replies) if "Target diagram" in r.text else ACCEPT
    )
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, llm, compute, image_llm=image_llm)

    results = PlottingAgent(ctx).run(
        {"plotting_plan": [figure("fig_overview", "diagram"), figure("fig_scores", "plot")]}
    )

    by_id = {result["figure_id"]: result for result in results}
    diagram = by_id["fig_overview"]
    assert [entry["outcome"] for entry in diagram["critic_history"]] == ["revised", "accepted"]
    assert diagram["description"] == "plan with a decoder"
    # The diagram never touched the compute backend; the plot is always code.
    assert len(compute.calls) == 1
    assert len(requests_for(llm, "plotting.code")) == 1
    first, second = image_llm.image_prompts
    assert "<figure_description>\nstyled plan for Title fig_overview\n" in first
    assert "<figure_description>\nplan with a decoder\n" in second
    assert "Do not include a figure title" in first

    # The JPEG the image model returned is stored as a real PNG under the .png name.
    stored = (ctx.paths.generated_figures_dir / "fig_overview.png").read_bytes()
    assert stored.startswith(b"\x89PNG") and PILImage.open(io.BytesIO(stored)).size == (64, 36)
    diagram_requests = [r for r in llm.requests if "Title fig_overview" in r.text]
    for request in diagram_requests:
        assert prompts.DIAGRAM_AS_CODE_NOTE not in request.text
    critic = next(r for r in diagram_requests if r.label == "plotting.critic")
    assert critic.images[0].media_type == "image/png"
    page = ctx.harness.project.read("wiki/figures/fig_overview.md")
    assert "Produced by: image model fake-image" in page.body


@pytest.mark.parametrize(
    "image_llm",
    [
        FakeLLM({}, image=None),
        FakeLLM({}, image=b"<html>quota exceeded</html>"),
        FakeLLM({}, image=PNG[: len(PNG) // 2]),
        FakeLLM({}, image=PNG, capabilities=Capabilities(image_generation=False)),
        LLMConfigError("the image provider SDK is not installed"),
    ],
    ids=["no-image", "not-an-image", "truncated", "cannot-generate", "client-unavailable"],
)
def test_diagram_falls_back_to_code_when_the_image_model_fails(tmp_path, caplog, image_llm):
    llm = scripted_llm()
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, llm, compute, image_llm=image_llm)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure("fig_overview", "diagram")]})

    assert result["image_path"] == "figures/fig_overview.png"
    assert len(compute.calls) == 1
    assert (ctx.paths.figures_dir / "fig_overview.png").read_bytes() == PNG
    assert "matplotlib code instead" in caplog.text
    (code,) = requests_for(llm, "plotting.code")
    assert "FancyBboxPatch" in code.system


# -- the critic loop ---------------------------------------------------------------


def test_critic_revises_then_accepts(tmp_path):
    first, second = image_bytes("white"), image_bytes("blue")
    critic_replies = iter(
        [revise("Make the bars blue.", "Detailed Description: blue bars"), ACCEPT]
    )
    llm = scripted_llm(
        critic=lambda r: next(critic_replies),
        code=lambda r: fenced(f"{CODE}\n# version {r.turn}"),
    )
    compute = FakeCompute(lambda code, call: rendered(first if call == 1 else second))
    ctx = make_ctx(tmp_path, llm, compute)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert result["description"] == "blue bars"
    assert result["critic_history"] == [
        {
            "round": 0,
            "suggestions": "Make the bars blue.",
            "revised_description": "blue bars",
            "outcome": "revised",
        },
        {
            "round": 1,
            "suggestions": "No changes needed.",
            "revised_description": "blue bars",
            "outcome": "accepted",
        },
    ]
    # Each critic round looks at the render it is judging; the caption sees the final one.
    round_one, round_two = requests_for(llm, "plotting.critic")
    assert round_one.images[0].data == first and round_two.images[0].data == second
    assert "<detailed_description>\nblue bars\n" in round_two.text
    (caption,) = requests_for(llm, "plotting.caption")
    assert caption.images[0].data == second and "blue bars" in caption.text
    assert (ctx.paths.generated_figures_dir / "fig_accuracy.png").read_bytes() == second

    # The revision is a second turn of the same code conversation.
    original, revision = requests_for(llm, "plotting.code")
    assert (original.turn, revision.turn) == (0, 1)
    assert "<reviewer_suggestions>\nMake the bars blue.\n" in revision.text
    assert "<figure_description>\nblue bars\n" in revision.text
    assert revision.history[0][1] == fenced(f"{CODE}\n# version 0")
    assert [call["code"].split("\n# version ")[1][0] for call in compute.calls] == ["0", "1"]
    assert "round 1: revised; round 2: accepted" in (
        ctx.harness.project.read("wiki/figures/fig_accuracy.md").body
    )


@pytest.mark.parametrize(("rounds", "renders"), [(0, 1), (2, 3)])
def test_critic_rounds_are_bounded(tmp_path, rounds, renders):
    counter = iter(range(100))
    llm = scripted_llm(critic=lambda r: revise("More contrast.", f"revision {next(counter)}"))
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, llm, compute, plotting_max_critic_rounds=rounds)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert len(requests_for(llm, "plotting.critic")) == rounds
    assert len(compute.calls) == renders
    assert [entry["outcome"] for entry in result["critic_history"]] == ["revised"] * rounds
    if rounds:
        assert result["description"] == f"revision {rounds - 1}"


@pytest.mark.parametrize(
    "reply",
    [
        revise("Looks right.", "No changes needed."),
        revise("Looks right.", "  "),
        json.dumps({"needs_revision": False, "revised_description": "ignored rewrite"}),
    ],
    ids=["sentinel", "blank", "not-needed"],
)
def test_critic_without_a_usable_revision_accepts_the_figure(tmp_path, reply):
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, scripted_llm(critic=reply), compute)
    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    (entry,) = result["critic_history"]
    assert entry["outcome"] == "accepted"
    assert (
        entry["revised_description"]
        == result["description"]
        == "styled plan for Title fig_accuracy"
    )
    assert len(compute.calls) == 1


def test_unparseable_critic_keeps_the_render(tmp_path, caplog):
    llm = scripted_llm(critic="I cannot judge this figure.")
    ctx = make_ctx(tmp_path, llm)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    assert result["critic_history"] == []
    assert result["caption"] == "Caption of Title fig_accuracy."
    assert "critic round 1 failed" in caplog.text
    assert (ctx.paths.figures_dir / "fig_accuracy.png").is_file()


def test_failed_revision_render_keeps_the_previous_figure(tmp_path, caplog):
    llm = scripted_llm(critic=revise("Use a log axis.", "log-axis plot"))
    compute = FakeCompute(
        lambda code, call: rendered() if call == 1 else crashed("ValueError: log")
    )
    ctx = make_ctx(tmp_path, llm, compute, max_repair_rounds=1)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    # One good render, then the revision and its single repair both fail; the loop stops.
    assert len(compute.calls) == 3
    assert len(requests_for(llm, "plotting.critic")) == 1
    assert result["description"] == "styled plan for Title fig_accuracy"
    assert result["critic_history"] == [
        {
            "round": 0,
            "suggestions": "Use a log axis.",
            "revised_description": "log-axis plot",
            "outcome": "render_failed",
        }
    ]
    assert (ctx.paths.generated_figures_dir / "fig_accuracy.png").read_bytes() == PNG
    assert "keeping the previous render" in caplog.text
    (note,) = notes(ctx)
    assert note.kind == "lesson" and "ValueError: log" in note.body


def test_caption_failure_falls_back_to_the_title(tmp_path, caplog):
    ctx = make_ctx(tmp_path, scripted_llm(caption="  "))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    assert result["caption"] == "Title fig_accuracy"
    assert "no caption could be generated" in caplog.text


def test_model_without_vision_skips_the_critic(tmp_path, caplog):
    blind = Capabilities(tools=True, memory_tool=True, structured_output=True, image_input=False)
    llm = scripted_llm()
    llm.capabilities = blind
    ctx = make_ctx(tmp_path, llm)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    assert result["critic_history"] == []
    assert requests_for(llm, "plotting.critic") == []
    assert all(not request.images for request in llm.requests)
    assert "cannot read images" in caplog.text


# -- figure code failures ----------------------------------------------------------


def test_failing_code_is_repaired_with_the_error_output(tmp_path):
    stderr = "x" * 5000 + "\nNameError: name 'sns' is not defined"
    llm = scripted_llm(code=lambda r: fenced(f"{CODE}\n# attempt {r.turn}"))
    compute = FakeCompute(lambda code, call: crashed(stderr) if call == 1 else rendered())
    ctx = make_ctx(tmp_path, llm, compute)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert result["image_path"] == "figures/fig_accuracy.png"
    assert len(compute.calls) == 2
    first, repair = requests_for(llm, "plotting.code")
    assert (first.turn, repair.turn) == (0, 1)
    assert "It exited with status 1." in repair.text
    assert "NameError: name 'sns' is not defined" in repair.text
    # Only the tail of a long error output is fed back.
    assert len(repair.text) < 2600
    assert "# attempt 1" in compute.calls[1]["code"]

    (note,) = notes(ctx)
    assert note.kind == "lesson" and note.title == "Figure code repaired: fig_accuracy"
    assert "NameError: name 'sns' is not defined" in note.body
    events = [e for e in ctx.harness.transcripts.events() if e["type"] == "figure_code"]
    assert [(e["attempt"], e["ok"]) for e in events] == [(1, False), (2, True)]


@pytest.mark.parametrize(
    ("failure", "problem"),
    [
        (ExecResult(returncode=-9, timed_out=True), "stopped after the time limit of 120 seconds"),
        (ExecResult(returncode=0, stdout="done"), "finished without producing figure.png"),
        (rendered(b"<html>not an image</html>"), "is not a valid image"),
        (rendered(PNG[: len(PNG) // 2]), "is not a valid image"),
    ],
    ids=["timeout", "missing-output", "not-an-image", "truncated-image"],
)
def test_other_render_failures_are_fed_back_too(tmp_path, failure, problem):
    llm = scripted_llm()
    compute = FakeCompute(lambda code, call: failure if call == 1 else rendered())
    ctx = make_ctx(tmp_path, llm, compute)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert result["image_path"] == "figures/fig_accuracy.png"
    _, repair = requests_for(llm, "plotting.code")
    assert problem in repair.text
    assert (ctx.paths.generated_figures_dir / "fig_accuracy.png").read_bytes() == PNG


def test_permanent_code_failure_skips_only_that_figure(tmp_path, caplog):
    def code(request: FakeRequest) -> str:
        return fenced("BROKEN = 1" if "fig_broken" in first_turn_text(request) else CODE)

    llm = scripted_llm(code=code)
    compute = FakeCompute(
        lambda code, call: (
            crashed("ZeroDivisionError: division by zero") if "BROKEN" in code else rendered()
        )
    )
    ctx = make_ctx(tmp_path, llm, compute, max_repair_rounds=2, figure_workers=3)
    plan = [figure("fig_first"), figure("fig_broken"), figure("fig_last")]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = PlottingAgent(ctx).run({"plotting_plan": plan})

    assert [result["figure_id"] for result in results] == ["fig_first", "fig_last"]
    # One attempt plus max_repair_rounds repairs, then the figure is given up.
    assert sum("BROKEN" in call["code"] for call in compute.calls) == 3
    broken_turns = [
        request.turn
        for request in requests_for(llm, "plotting.code")
        if "fig_broken" in first_turn_text(request)
    ]
    assert sorted(broken_turns) == [0, 1, 2]

    paths = ctx.paths
    assert sorted(p.name for p in paths.generated_figures_dir.iterdir()) == [
        "fig_first.png",
        "fig_last.png",
    ]
    assert [entry["name"] for entry in json.loads(paths.figures_info.read_text("utf-8"))] == [
        "fig_first.png",
        "fig_last.png",
    ]
    assert json.loads(paths.plotting_results.read_text("utf-8")) == results
    wiki = ctx.harness.project
    assert wiki.read("wiki/figures/fig_first.md").meta["status"] == "produced"
    broken_page = wiki.read("wiki/figures/fig_broken.md")
    assert broken_page.meta["status"] == "failed" and "file" not in broken_page.meta
    assert "no file to include" in broken_page.meta["summary"]
    assert "still failed after 3 attempt(s)" in broken_page.body

    assert "Figure 'fig_broken' failed and is skipped" in caplog.text
    assert "Plotting produced 2 of 3 figure(s); skipped: fig_broken." in caplog.text
    (note,) = notes(ctx)
    assert note.kind == "lesson" and note.title == "Figure skipped: fig_broken"
    assert "still failed after 3 attempt(s)" in note.body
    assert "ZeroDivisionError: division by zero" in note.body
    statuses = {
        e["figure_id"]: e["status"]
        for e in ctx.harness.transcripts.events()
        if e["type"] == "figure"
    }
    assert statuses == {"fig_first": "ok", "fig_broken": "failed", "fig_last": "ok"}


def test_every_figure_failing_returns_an_empty_list(tmp_path, caplog):
    def plan(request: FakeRequest) -> str:
        raise RuntimeError("planner exploded")

    ctx = make_ctx(tmp_path, scripted_llm(plan=plan))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = PlottingAgent(ctx).run({"plotting_plan": [figure("a"), figure("b")]})
    assert results == []
    assert json.loads(ctx.paths.plotting_results.read_text("utf-8")) == []
    assert json.loads(ctx.paths.figures_info.read_text("utf-8")) == []
    assert "Plotting produced 0 of 2 figure(s); skipped: a, b." in caplog.text
    assert {note.kind for note in notes(ctx)} == {"observation"}


def test_missing_material_skips_the_figure(tmp_path, caplog):
    ctx = make_ctx(tmp_path, scripted_llm())
    ctx.paths.experimental_log.unlink()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = PlottingAgent(ctx).run(
            {"plotting_plan": [figure("fig_log"), figure("fig_idea", data_source="idea.md")]}
        )
    assert [result["figure_id"] for result in results] == ["fig_idea"]
    assert "required file not found" in caplog.text


# -- figure ids and file names -----------------------------------------------------


def test_unsafe_and_duplicate_figure_ids_become_safe_unique_file_names(tmp_path):
    raw_ids = ["../../etc/passwd", "Fig 1: Overview!", "fig_dup", "fig_dup", "", None, "FIG_DUP"]
    plan = [{**figure(f"n{index}"), "figure_id": raw} for index, raw in enumerate(raw_ids)]
    ctx = make_ctx(tmp_path, scripted_llm(), figure_workers=4)

    results = PlottingAgent(ctx).run({"plotting_plan": plan})

    ids = [result["figure_id"] for result in results]
    assert ids == [
        "etc_passwd",
        "Fig_1_Overview",
        "fig_dup",
        "fig_dup_2",
        "figure_5",
        "figure_6",
        "FIG_DUP_3",
    ]
    assert len({figure_id.lower() for figure_id in ids}) == len(ids)
    paths = ctx.paths
    expected = sorted(f"{figure_id}.png" for figure_id in ids)
    assert sorted(p.name for p in paths.generated_figures_dir.iterdir()) == expected
    assert sorted(p.name for p in paths.figures_dir.iterdir()) == sorted([*expected, "info.json"])
    for result in results:
        target = (paths.base_dir / result["image_path"]).resolve()
        assert target.parent == paths.generated_figures_dir.resolve() and target.is_file()
        assert ctx.harness.project.exists(f"wiki/figures/{result['figure_id']}.md")
    # Nothing escaped the run directory.
    assert not (tmp_path / "etc").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"run"}


def test_generated_figures_do_not_overwrite_provided_ones(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm())
    paths = ctx.paths
    provided_png = image_bytes("green")
    paths.provided_figures_dir.mkdir()
    (paths.provided_figures_dir / "fig_accuracy.png").write_bytes(provided_png)
    paths.figures_dir.mkdir(parents=True)
    (paths.figures_dir / "fig_accuracy.png").write_bytes(provided_png)
    provided_entry = {"name": "fig_accuracy.png", "caption": "Provided by the authors."}
    paths.figures_info.write_text(json.dumps([provided_entry]), encoding="utf-8")

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure("fig_accuracy")]})

    assert result["figure_id"] == "fig_accuracy_2"
    assert (paths.figures_dir / "fig_accuracy.png").read_bytes() == provided_png
    assert (paths.figures_dir / "fig_accuracy_2.png").read_bytes() == PNG
    assert json.loads(paths.figures_info.read_text("utf-8")) == [
        provided_entry,
        {"name": "fig_accuracy_2.png", "caption": "Caption of Title fig_accuracy."},
    ]


def test_figures_info_drops_entries_of_an_earlier_plotting_run(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm())
    paths = ctx.paths
    for directory in (paths.generated_figures_dir, paths.figures_dir):
        directory.mkdir(parents=True)
        (directory / "fig_old.png").write_bytes(PNG)
    paths.figures_info.write_text(
        json.dumps(
            [
                {"name": "teaser.pdf", "caption": "Provided."},
                {"name": "fig_old.png", "caption": "From the earlier run."},
                {"name": "fig_accuracy.png", "caption": "Stale caption."},
            ]
        ),
        encoding="utf-8",
    )
    PlottingAgent(ctx).run({"plotting_plan": [figure()]})
    assert json.loads(paths.figures_info.read_text("utf-8")) == [
        {"name": "teaser.pdf", "caption": "Provided."},
        {"name": "fig_accuracy.png", "caption": "Caption of Title fig_accuracy."},
    ]


# -- PaperBanana few-shot data -----------------------------------------------------


@pytest.mark.parametrize("configured", [False, True], ids=["unset", "missing-directory"])
def test_without_paper_banana_figures_are_zero_shot_and_it_is_logged_once(
    tmp_path, monkeypatch, caplog, configured
):
    if configured:
        monkeypatch.setenv(PAPER_BANANA_ENV, str(tmp_path / "no-such-checkout"))
    llm = scripted_llm()
    ctx = make_ctx(tmp_path, llm, figure_workers=3)
    plan = [figure("fig_a"), figure("fig_b", "diagram"), figure("fig_c")]

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert len(PlottingAgent(ctx).run({"plotting_plan": plan})) == 3

    mentions = [record for record in caplog.records if PAPER_BANANA_ENV in record.getMessage()]
    assert len(mentions) == 1 and "zero-shot" in mentions[0].getMessage()
    assert mentions[0].levelno == (logging.WARNING if configured else logging.INFO)
    assert requests_for(llm, "plotting.retrieve") == []
    assert all(not request.images for request in requests_for(llm, "plotting.plan"))
    assert all("Example 1:" not in request.text for request in requests_for(llm, "plotting.plan"))


def make_paper_banana(root: Path) -> None:
    task_dir = root / "data" / "PaperBananaBench" / "plot"
    (task_dir / "images").mkdir(parents=True)
    (task_dir / "images" / "ref_0.png").write_bytes(image_bytes("red"))
    (task_dir / "images" / "ref_1.jpg").write_bytes(image_bytes("green", "JPEG"))
    (root / "secret.png").write_bytes(image_bytes("black"))
    pool = [
        {
            "id": "ref_0",
            "visual_intent": "Bar chart of BLEU",
            "content": {"BLEU": [1, 2]},
            "path_to_gt_image": "images/ref_0.png",
        },
        {
            "id": "ref_1",
            "visual_intent": "Line chart of loss",
            "content": "loss: " + "0.5 " * 2000,
            "path_to_gt_image": "images/ref_1.jpg",
        },
        {"id": "ref_2", "visual_intent": "Lost image", "path_to_gt_image": "images/gone.png"},
        {"id": "ref_evil", "visual_intent": "Escapes", "path_to_gt_image": "../../../secret.png"},
        {"visual_intent": "entry without an id"},
    ]
    (task_dir / "ref.json").write_text(json.dumps(pool), encoding="utf-8")
    guides = root / "style_guides"
    guides.mkdir()
    (guides / "neurips2025_plot_style_guide.md").write_text("PB PLOT GUIDE: use teal.", "utf-8")


def test_paper_banana_examples_and_style_guide_are_used(tmp_path, monkeypatch):
    root = tmp_path / "PaperBanana"
    make_paper_banana(root)
    monkeypatch.setenv(PAPER_BANANA_ENV, str(root))
    selection = {"selected_ids": ["ref_1", "ref_9", "ref_2", "ref_evil", "ref_0", "ref_1"]}
    llm = scripted_llm(retrieve=json.dumps(selection))
    ctx = make_ctx(tmp_path, llm)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert llm.labels()[:2] == ["plotting.retrieve", "plotting.plan"]
    retrieve, plan, style = llm.requests[:3]
    assert retrieve.schema is FewShotSelection and retrieve.effort == "low"
    assert retrieve.system == prompts.PLOT_RETRIEVER_SYSTEM and retrieve.memory is None
    for ref_id in ("ref_0", "ref_1", "ref_2", "ref_evil"):
        assert f'<candidate id="{ref_id}">' in retrieve.text
    assert retrieve.text.count("<candidate id=") == 4
    assert "[truncated]" in retrieve.text and len(retrieve.text) < 4000
    assert "<target>\n<visual_intent>\nTitle: Title fig_accuracy" in retrieve.text
    assert retrieve.text.index("<candidate_pool>") < retrieve.text.index("<target>")

    # Unknown ids, a lost image and a path outside the checkout are dropped; order is kept.
    assert [image.data for image in plan.images] == [
        image_bytes("green", "JPEG"),
        image_bytes("red"),
    ]
    assert [image.media_type for image in plan.images] == ["image/jpeg", "image/png"]
    assert "Example 1:\n<raw_data>\nloss: 0.5" in plan.text
    assert 'Example 2:\n<raw_data>\n{"BLEU": [1, 2]}\n</raw_data>' in plan.text
    assert "<visual_intent>\nBar chart of BLEU\n</visual_intent>\nReference plot of example 2:" in (
        plan.text
    )
    assert "Example 3:" not in plan.text
    assert "PB PLOT GUIDE: use teal." in style.text
    assert prompts.DEFAULT_PLOT_STYLE_GUIDE.strip() not in style.text
    assert "planned with 2 PaperBanana few-shot example(s)" in (
        ctx.harness.project.read(f"wiki/figures/{result['figure_id']}.md").body
    )


def test_failed_retrieval_plans_zero_shot(tmp_path, monkeypatch, caplog):
    root = tmp_path / "PaperBanana"
    make_paper_banana(root)
    monkeypatch.setenv(PAPER_BANANA_ENV, str(root))
    llm = scripted_llm(retrieve="I do not know which ones to pick.")
    ctx = make_ctx(tmp_path, llm)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = [figure(), figure("fig_overview", "diagram")]
        assert len(PlottingAgent(ctx).run({"plotting_plan": plan})) == 2
    assert "few-shot retrieval failed" in caplog.text
    assert all(not request.images for request in requests_for(llm, "plotting.plan"))
    # The checkout has no diagram data: that task is zero-shot with the built-in guide.
    assert "diagrams are planned zero-shot" in caplog.text
    assert "No usable diagram style guide" in caplog.text


# -- no in-process execution -------------------------------------------------------


def test_figure_code_is_never_executed_in_this_process(tmp_path):
    sentinel = tmp_path / "run" / "executed-in-process"
    hostile = (
        f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('executed')\n"
        "raise SystemExit('model code ran in the orchestrator process')"
    )
    compute = FakeCompute()
    ctx = make_ctx(tmp_path, scripted_llm(code=fenced(hostile)), compute)

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    assert result["image_path"] == "figures/fig_accuracy.png"
    assert not sentinel.exists()
    assert compute.calls[0]["code"].startswith(hostile)


def test_agent_module_has_no_way_to_run_code_itself():
    tree = ast.parse(inspect.getsource(plotting))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"exec", "eval", "compile", "__import__"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"matplotlib", "subprocess", "runpy", "importlib", "multiprocessing"}


# -- the save epilogue, in a real subprocess -----------------------------------------


def test_real_backend_renders_the_figure_the_script_left_open(tmp_path):
    script = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(4, 3))\n"
        "ax.bar(['FooNet', 'baseline'], [91.2, 85.0])\n"
        "plt.show()"
    )
    ctx = make_ctx(tmp_path, scripted_llm(code=fenced(script)), LocalBackend())

    (result,) = PlottingAgent(ctx).run({"plotting_plan": [figure()]})

    with PILImage.open(ctx.paths.base_dir / result["image_path"]) as picture:
        assert picture.format == "PNG"
        # 4 x 3 inches at 200 dpi, trimmed or padded a little by the tight bounding box.
        assert 600 <= picture.width <= 900 and 450 <= picture.height <= 700


def test_epilogue_reports_a_script_that_drew_nothing(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm(), LocalBackend())
    with pytest.raises(plotting._CodeFailure) as caught:
        run_figure_code(ctx, "print('nothing drawn')")
    assert caught.value.problem == "It exited with status 3."
    assert "No matplotlib figure was open" in caught.value.output

    with pytest.raises(plotting._CodeFailure) as caught:
        run_figure_code(ctx, "raise ValueError('boom')")
    assert "ValueError: boom" in caught.value.output


def test_epilogue_keeps_a_figure_the_script_saved_itself(tmp_path):
    ctx = make_ctx(tmp_path, scripted_llm(), LocalBackend())
    own = image_bytes("orange")
    script = (
        "import base64\n"
        f"open('figure.png', 'wb').write(base64.b64decode({base64.b64encode(own).decode()!r}))"
    )
    assert run_figure_code(ctx, script) == own


# -- helpers and prompts -----------------------------------------------------------


def test_plan_normalisation_helpers():
    assert [
        task_name(kind) for kind in ("plot", "Diagram", "Bar Chart", "scatter plot", "", "x")
    ] == [
        "plot",
        "diagram",
        "plot",
        "plot",
        "diagram",
        "diagram",
    ]
    assert normalize_aspect_ratio(" 21:9 ") == "21:9"
    assert [normalize_aspect_ratio(value) for value in ("wide", None, "0:3", "4/3")] == ["16:9"] * 4
    assert figure_size("16:9") == (9.8, 5.5)
    assert figure_size("1:1") == (7.3, 7.3)
    assert figure_size("1:4") == (3.7, 14.7)
    (spec,) = plan_figures([{"figure_id": "a b", "plot_type": "Radar Chart", "aspect_ratio": 3}])
    assert (spec.figure_id, spec.task, spec.aspect_ratio, spec.title) == ("a_b", "plot", "16:9", "")
    assert spec.intent == "Title: \nObjective: "


def test_to_png_validates_and_converts():
    assert to_png(PNG) == PNG
    converted = to_png(image_bytes("red", "JPEG"))
    with PILImage.open(io.BytesIO(converted)) as picture:
        assert picture.format == "PNG" and picture.mode == "RGB"
    transparent = io.BytesIO()
    PILImage.new("RGBA", (8, 8), (255, 0, 0, 0)).save(transparent, format="WEBP", lossless=True)
    with PILImage.open(io.BytesIO(to_png(transparent.getvalue()))) as picture:
        assert picture.format == "PNG" and picture.mode == "RGBA"
    for bad in (b"", b"plain text", PNG[:20], PNG[: len(PNG) // 2]):
        with pytest.raises(ValueError):
            to_png(bad)


def test_label_stripping():
    strip = plotting._strip_description_label
    assert strip("**Detailed Description:** a chart") == "a chart"
    assert strip("Detailed Description: a chart") == "a chart"
    assert strip("## Detailed Description\n\na chart") == "a chart"
    assert strip("Detailed description of the pipeline: boxes") == (
        "Detailed description of the pipeline: boxes"
    )
    caption = plotting._clean_caption
    assert caption("Caption: Accuracy.\n") == "Accuracy."
    assert caption("**Caption:** Accuracy.") == "Accuracy."
    assert caption("Figure 3: Accuracy\nper method.") == "Accuracy per method."
    assert caption("Fig. 2. Accuracy.") == "Accuracy."
    assert caption("Figure-ground segmentation results.") == "Figure-ground segmentation results."
    assert caption("  \n") is None


def test_prompts_render_without_stray_braces():
    material = "<idea>\nidea\n</idea>"
    rendered_prompts = []
    for task in ("plot", "diagram"):
        rendered_prompts += [
            prompts.retriever_system(task),
            prompts.planner_system(task),
            prompts.stylist_system(task),
            prompts.critic_system(task),
            prompts.code_system(task, timeout_seconds=120),
            prompts.default_style_guide(task),
            prompts.planner_request(task, material=material, intent="i", drawn_with_code=True),
            prompts.stylist_request(
                task,
                description="d",
                style_guide="g",
                material=material,
                intent="i",
                drawn_with_code=True,
            ),
            prompts.critic_request(
                task, description="d", material=material, intent="i", drawn_with_code=True
            ),
            prompts.code_request(task, description="d", aspect_ratio="16:9", figsize=(9.8, 5.5)),
            prompts.caption_request(task, material=material, intent="i", description="d"),
        ]
    rendered_prompts += [
        prompts.CAPTION_SYSTEM,
        prompts.IMAGE_MODEL_SYSTEM,
        prompts.image_model_request("d"),
        prompts.code_revision_request(description="d", suggestions="s"),
        prompts.code_repair_request(problem="It exited with status 1.", output=""),
    ]
    for text in rendered_prompts:
        assert "{" not in text and "}" not in text, text[:80]
        assert text.strip()
    assert "(the script printed nothing)" in rendered_prompts[-1]
    # A description with braces (JSON data, LaTeX) passes through the builders unchanged.
    braces = 'data = {"a": 1} and $x_{i}$'
    assert braces in prompts.code_request(
        "plot", description=braces, aspect_ratio="1:1", figsize=(7.3, 7.3)
    )
    assert braces in prompts.code_repair_request(problem="p", output=braces)
