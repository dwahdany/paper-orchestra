"""The whole pipeline, offline, with the real agents.

A scripted model stands in for the provider and a tiny in-memory Semantic
Scholar for the network, but everything else is real: the stage runner, every
agent, the memory harness (with the model using the memory tool), figure code
executed by the local compute backend and the paper compiled by pdflatex.
"""

from __future__ import annotations

import base64
import json
import re
import signal
import threading

import pytest
from rapidfuzz import fuzz

from paper_orchestra import pipeline
from paper_orchestra.agents.literature import LiteratureAgent
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import MemoryConfig, ModelConfig, PipelineConfig
from paper_orchestra.context import RunPaths
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness
from paper_orchestra.utils.pdf import load_paper, page_count
from paper_orchestra.utils.scholar import SemanticScholar

pytestmark = pytest.mark.latex

REVISION_MARKER = "We state the scope of each claim explicitly."
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

LIBRARY = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "date": "2017-06-12",
        "abstract": "We propose the Transformer, based solely on attention mechanisms.",
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang"],
        "year": 2016,
        "date": "2015-12-10",
        "abstract": "We present a residual learning framework to ease the training of networks.",
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": ["Diederik P. Kingma", "Jimmy Ba"],
        "year": 2015,
        "date": "2014-12-22",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization.",
    },
]

OUTLINE = {
    "plotting_plan": [
        {
            "figure_id": "fig_main_results",
            "title": "Main results",
            "plot_type": "plot",
            "data_source": "experimental_log.md",
            "objective": "Bar chart comparing our method with the baselines.",
            "aspect_ratio": "4:3",
        }
    ],
    "intro_related_work_plan": {
        "introduction_strategy": {
            "hook_hypothesis": "Attention-based models dominate.",
            "problem_gap_hypothesis": "They are costly to train.",
            "search_directions": ["foundational attention models"],
        },
        "related_work_strategy": {
            "overview": "One cluster of prior architectures.",
            "subsections": [
                {
                    "subsection_title": "2.1 Deep Architectures",
                    "methodology_cluster": "Residual networks",
                    "sota_investigation_mission": "Find the residual learning line of work.",
                    "limitation_hypothesis": "Depth alone does not give efficiency.",
                    "limitation_search_queries": ["residual learning image recognition"],
                    "bridge_to_our_method": "We reuse residual connections more cheaply.",
                }
            ],
        },
    },
    "section_plan": [
        {
            "section_title": "3. Methodology",
            "subsections": [
                {
                    "subsection_title": "3.1 Training",
                    "content_bullets": ["State the optimizer."],
                    "citation_hints": [
                        "Kingma and Ba (Adam: A Method for Stochastic Optimization)"
                    ],
                },
                {"subsection_title": "3.2 Details", "content_bullets": [], "citation_hints": []},
            ],
        }
    ],
}

FIGURE_CODE = """```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["Baseline", "Ours"], [1.0, 1.2], color=["#888888", "#1f77b4"])
ax.set_ylabel("Relative score")
fig.tight_layout()
```"""


class FakeS2Response:
    def __init__(self, data: list[dict]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._data = data

    def json(self) -> dict:
        return {"total": len(self._data), "data": self._data}


class FakeS2Session:
    """Returns the library entries whose titles are closest to the query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def get(self, url, *, headers, params, timeout):
        query = params["query"]
        self.queries.append(query)
        ranked = sorted(LIBRARY, key=lambda p: -fuzz.ratio(query.lower(), p["title"].lower()))
        records = [
            {
                "paperId": f"{index:040d}",
                "title": paper["title"],
                "authors": [{"authorId": "1", "name": name} for name in paper["authors"]],
                "venue": "NeurIPS",
                "year": paper["year"],
                "abstract": paper["abstract"],
                "citationCount": 1000,
                "journal": None,
                "publicationDate": paper["date"],
            }
            for index, paper in enumerate(ranked[: params["limit"]])
        ]
        return FakeS2Response(records)


def review(overall: int) -> str:
    return json.dumps(
        {
            "Summary": "A clear systems paper.",
            "Strengths": ["Simple method."],
            "Weaknesses": ["Claims could be scoped more carefully."],
            "Originality": 2,
            "Quality": 3,
            "Clarity": 3,
            "Significance": 2,
            "Questions": ["What is the scope of the main claim?"],
            "Limitations": ["Evaluated on one benchmark."],
            "EthicalConcerns": False,
            "Soundness": 3,
            "Presentation": 3,
            "Contribution": 2,
            "Overall": overall,
            "Confidence": 4,
            "Decision": "Accept" if overall >= 6 else "Reject",
        }
    )


class Model:
    """The scripted model.  Each branch answers the way a capable model would."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.memory_views: list[str] = []
        #: Label of a model call during which the user presses Ctrl-C (once).
        self.interrupt_during: str | None = None

    def __call__(self, request: FakeRequest) -> str:
        label = request.label
        if label == self.interrupt_during:
            self.interrupt_during = None
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        if label == "outline.OutlineAgent":
            return json.dumps(OUTLINE)
        if label == "literature.discovery":
            return self._discover(request)
        if label == "literature.LiteratureAgent":
            return self._intro_and_related_work(request)
        if label == "plotting.plan":
            return "A bar chart with two bars: the baseline and our method."
        if label == "plotting.style":
            return "A clean two-bar chart, grey baseline and blue for ours, labelled axes."
        if label == "plotting.code":
            return FIGURE_CODE
        if label == "plotting.critic":
            return json.dumps({"needs_revision": False})
        if label == "plotting.caption":
            return "Our method compared with the baseline."
        if label == "section_writing.SectionWritingAgent":
            return self._full_paper(request)
        if label in ("refinement.review", "refinement.review.meta"):
            return review(6 if REVISION_MARKER in request.text else 5)
        if label == "refinement.revise":
            return self._revise()
        if label == "refinement.format_review":
            return json.dumps({"figure_and_tables": [], "other_issues": []})
        if label == "dream.project":
            return self._dream_project(request)
        if label == "dream.lab":
            return self._dream_lab(request)
        raise AssertionError(f"unexpected model call: {label}")

    def _discover(self, request: FakeRequest) -> str:
        text = request.text
        if "<hint>" in text:
            wanted = [LIBRARY[2]]
        elif "residual" in text.lower():
            wanted = [LIBRARY[1]]
        else:
            wanted = [LIBRARY[0]]
        return json.dumps(
            {
                "section_name": "any",
                "candidates": [
                    {"title": p["title"], "year": p["year"], "reason": "Foundational work."}
                    for p in wanted
                ],
            }
        )

    def _intro_and_related_work(self, request: FakeRequest) -> str:
        block = re.search(r"<citation_checklist>\n(.*?)\n</citation_checklist>", request.text, re.S)
        keys = block.group(1).split()
        assert len(keys) == 3, keys
        cites = ", ".join(keys)
        template = self.paths.template_tex.read_text(encoding="utf-8")
        template = template.replace(
            "Introduction here.",
            f"Attention-based models dominate modern learning~\\cite{{{cites}}}.",
        ).replace(
            "Related work here.",
            f"Residual architectures~\\cite{{{keys[0]}}} precede our approach.",
        )
        return f"```latex\n{template}\n```"

    def _full_paper(self, request: FakeRequest) -> str:
        # Use memory the way a model would: look around, then leave one lesson.
        assert request.memory is not None
        self.memory_views.append(request.memory.view("/memories"))
        request.memory.create(
            "/memories/notes/table-style.md",
            "Lesson: tables in this template need the booktabs package for rules.",
        )
        figure = json.loads(self.paths.figures_info.read_text(encoding="utf-8"))[0]
        tex = self.paths.template_tex.read_text(encoding="utf-8")
        author_block = re.search(r"\\author\{.*?\n\}", tex, re.S).group(0)
        replacements = {
            "My Title": "Cheaper Attention with Residual Shortcuts",
            author_block: "\\author{Anonymous Authors}",
            "Abstract here.": "We study a cheaper attention block and report consistent gains.",
            "Methodology here.": "Our block adds a shortcut around attention and is trained "
            "with a standard adaptive optimizer.",
            "Experimental setup, results, and analysis here.": (
                "Figure~\\ref{fig:main} summarises the comparison.\n"
                "\\begin{figure}[t]\n\\centering\n"
                f"\\includegraphics[width=0.9\\linewidth]{{figures/{figure['name']}}}\n"
                f"\\caption{{{figure['caption']}}}\n\\label{{fig:main}}\n\\end{{figure}}\n"
                "\\begin{table}[t]\n\\centering\n\\begin{tabular}{lc}\n\\toprule\n"
                "Method & Relative score \\\\\n\\midrule\nBaseline & lower \\\\\n"
                "Ours & higher \\\\\n\\bottomrule\n\\end{tabular}\n"
                "\\caption{Qualitative summary.}\n\\label{tab:main}\n\\end{table}"
            ),
            "Conclusion here.": "A shortcut around attention is a cheap and effective change.",
        }
        for old, new in replacements.items():
            assert old in tex, old
            tex = tex.replace(old, new)
        return f"```latex\n{tex}\n```"

    def _revise(self) -> str:
        current = (self.paths.refinement_dir / "initial_draft.tex").read_text(encoding="utf-8")
        revised = current.replace(
            "A shortcut around attention is a cheap and effective change.",
            f"A shortcut around attention is a cheap and effective change. {REVISION_MARKER}",
        )
        assert revised != current
        plan = {
            "addressed_weaknesses": ["Scoped the main claim."],
            "integrated_answers": ["Stated the scope in the conclusion."],
            "actions_taken": ["Edited the conclusion."],
        }
        return f"```json\n{json.dumps(plan)}\n```\n```latex\n{revised}\n```"

    def _dream_project(self, request: FakeRequest) -> str:
        if "synthesis" in request.text.lower() and not request.history:
            request.memory.create(
                "/memories/wiki/synthesis/architectures.md",
                "---\nsummary: How the collected architecture papers relate to our block\n"
                "kind: synthesis\n---\nResidual connections and attention are the two lines "
                "of work our block combines; see the pages under literature/.",
            )
            return "Wrote one synthesis page."
        request.memory.create(
            "/memories/wiki/lessons/latex-tables.md",
            "---\nsummary: Table rules need a package to be loaded\nkind: lesson\n"
            "portable: true\n---\n"
            "RULE: load the package that provides table rules before using them.\n"
            "WHY: the build fails with an undefined control sequence otherwise.\n"
            "EVIDENCE: noted by the section writer during drafting.",
        )
        return "Folded the notes into one lesson."

    def _dream_lab(self, request: FakeRequest) -> str:
        assert "sources" not in request.memory.view("/memories")
        request.memory.create(
            "/memories/wiki/lessons/latex.md",
            "---\nsummary: LaTeX pitfalls seen across papers\nkind: lesson\n---\n"
            "RULE: load the package that provides table rules before using them.",
        )
        return "Folded one lesson into the lab memory."


@pytest.fixture
def run(tmp_path, repo_root, monkeypatch):
    example = json.loads((repo_root / "examples/cvpr_example.json").read_text())
    raw = tmp_path / "raw_materials"
    raw.mkdir()
    (raw / "idea_sparse.md").write_text(example["idea"], encoding="utf-8")
    (raw / "experimental_log.md").write_text(example["log"], encoding="utf-8")

    cfg = PipelineConfig(
        base_dir=tmp_path / "run",
        template_dir=repo_root / "src" / "paper_orchestra" / "templates" / "cvpr2025",
        use_plotting=True,
        research_cutoff="2025-01",
        max_reflections=1,
        review_ensemble=2,
        plotting_max_critic_rounds=1,
        models=ModelConfig(writer="fake", reflection="fake", plotting="fake", fast="fake"),
        memory=MemoryConfig(lab_dir=tmp_path / "lab", dream="stages"),
    )
    paths = pipeline.prepare_run_dir(cfg, raw)
    model = Model(paths)
    client = FakeLLM(model)
    session = FakeS2Session()
    monkeypatch.setattr(
        pipeline,
        "_literature_agent",
        lambda ctx: LiteratureAgent(
            ctx, scholar=SemanticScholar(session, sleep=lambda _: None, max_retries=1)
        ),
    )

    def execute():
        harness = ContextHarness.create(cfg, client_for=lambda _model: client)
        return pipeline.run_paper(cfg, harness=harness, compute=LocalBackend()), harness

    return cfg, paths, client, model, session, execute


def test_pipeline_writes_a_paper_and_remembers_what_it_learned(run):
    _cfg, paths, client, model, session, execute = run
    result, harness = execute()

    assert result.success, result.error
    state = json.loads(paths.state.read_text())
    assert list(state["stages"]) == [
        "ingest",
        "outline",
        "literature",
        "plotting",
        "section_writing",
        "refinement",
        "finalize",
    ]
    assert {stage["status"] for stage in state["stages"].values()} == {"completed"}

    # -- the paper -----------------------------------------------------------
    assert paths.final_pdf.read_bytes().startswith(b"%PDF") and page_count(paths.final_pdf) >= 1
    final_tex = paths.final_tex.read_text()
    assert REVISION_MARKER in final_tex  # the revision raised the score and was kept
    assert "Anonymous Authors" in final_tex and "My Title" not in final_tex
    text = load_paper(paths.final_pdf)
    assert "Cheaper Attention with Residual Shortcuts" in text and "References" in text
    for author in ("Vaswani", "Kingma"):
        assert author in text  # the bibliography was resolved by bibtex

    # -- the stage outputs keep their public names ---------------------------
    outline = json.loads(paths.outline.read_text())
    intro_plan = outline["intro_related_work_plan"]["introduction_strategy"]
    assert len(intro_plan["citation_candidates"]) == 1
    citation_map = json.loads(paths.citation_map.read_text())
    assert len(citation_map) == 3
    bib = paths.references_bib.read_text()
    assert all(f"{{{key}," in bib for key in citation_map)
    figures = json.loads(paths.plotting_results.read_text())
    assert [f["figure_id"] for f in figures] == ["fig_main_results"]
    assert (paths.base_dir / figures[0]["image_path"]).read_bytes().startswith(b"\x89PNG")
    worklog = json.loads((paths.refinement_dir / "content_refinement_worklog.json").read_text())
    assert worklog["v1"]["outcome"] == "ACCEPTED_SCORE_INCREASE"
    for name in ("initial_draft.pdf", "refined_paper_v1.pdf", "final_refined_paper.pdf"):
        assert (paths.refinement_dir / name).is_file()
    assert (paths.refinement_dir / "peer_reviews" / "review_v0.json").is_file()
    usage = json.loads(paths.usage.read_text())
    assert usage["total"]["requests"] == len(client.requests)
    assert {
        "OutlineAgent",
        "SectionWritingAgent",
        "ContentRefinementAgent",
        "Total_Time_Span",
    } <= set(json.loads(paths.time_breakdown.read_text()))

    # -- every agent went through the harness --------------------------------
    labels = set(client.labels())
    assert labels >= {
        "outline.OutlineAgent",
        "literature.discovery",
        "literature.LiteratureAgent",
        "plotting.plan",
        "plotting.code",
        "plotting.critic",
        "plotting.caption",
        "section_writing.SectionWritingAgent",
        "refinement.review",
        "refinement.review.meta",
        "refinement.revise",
        "refinement.format_review",
        "dream.project",
        "dream.lab",
    }
    by_label = {request.label: request for request in client.requests}
    # Reviewers and bulk calls are plain model calls; authors get the memory.
    for plain in ("refinement.review", "literature.discovery", "plotting.plan"):
        assert by_label[plain].memory is None and "# Memory" not in (by_label[plain].system or "")
    writer = by_label["section_writing.SectionWritingAgent"]
    assert "# Memory" in writer.system and "[[sources/experimental_log]]" in writer.system
    assert len(session.queries) == 3

    # -- memory: records, notes consolidated, index, transcripts -------------
    wiki = harness.project
    pages = {info.path for info in wiki.list()}
    assert {
        "wiki/sources/idea.md",
        "wiki/sources/experimental_log.md",
        "wiki/paper/outline.md",
    } <= pages
    assert len([p for p in pages if p.startswith("wiki/literature/")]) == 3
    assert "wiki/figures/fig_main_results.md" in pages
    assert {"wiki/reviews/v0.md", "wiki/reviews/v1.md"} <= pages
    assert {"wiki/synthesis/architectures.md", "wiki/lessons/latex-tables.md"} <= pages
    assert wiki.read("wiki/sources/idea.md").body.strip() == (
        paths.idea.read_text(encoding="utf-8").strip()
    )
    assert wiki.pending_notes() == []  # the end-of-run dream consolidated every note
    assert any("table-style" in info.path for info in wiki.list("notes/consolidated"))
    index = wiki.injected_context()
    assert "[[lessons/latex-tables]]" in index and "[[synthesis/architectures]]" in index
    report = wiki.lint()
    assert report.broken_links == [] and not report.index_over_budget
    assert "/memories/wiki" in model.memory_views[0]
    events = list(harness.transcripts.events())
    assert {"stage_start", "stage_end", "compile", "review", "dream", "note"} <= {
        event["type"] for event in events
    }

    # -- lab memory: the process lesson, and nothing about this paper --------
    lab = harness.lab
    assert lab.exists("wiki/lessons/latex.md") and lab.pending_notes() == []
    lab_text = "\n".join(lab.read_text(info.path) for info in lab.list())
    assert "table rules" in lab_text
    assert "Cheaper Attention" not in lab_text and "shortcut" not in lab_text.lower()
    memory_report = json.loads((paths.base_dir / pipeline.MEMORY_REPORT_NAME).read_text())
    assert [entry["scope"] for entry in memory_report["reports"]] == ["project", "lab"]
    assert all(entry["llm_used"] and not entry["error"] for entry in memory_report["reports"])


def test_rerunning_a_finished_run_reuses_every_stage(run):
    _cfg, paths, client, _model, _session, execute = run
    first, _ = execute()
    assert first.success, first.error
    calls = len(client.requests)
    pdf = paths.final_pdf.read_bytes()

    second, _ = execute()

    assert second.success, second.error
    authoring = [r.label for r in client.requests[calls:] if not r.label.startswith("dream.")]
    assert authoring == []  # no stage was run again
    assert paths.final_pdf.read_bytes() == pdf


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs POSIX signals")
def test_ctrl_c_during_plotting_then_the_same_command_finishes_the_paper(run):
    _cfg, paths, client, model, _session, execute = run
    # The folder ships the original paper's figure, as a benchmark folder does.  With plotting
    # on, every figure is generated and nobody is shown this one or its caption.
    caption = "ORIGINAL-CAPTION our method reaches the best score."
    paths.provided_figures_dir.mkdir()
    (paths.provided_figures_dir / "teaser.png").write_bytes(PNG)
    (paths.provided_figures_dir / "info.json").write_text(
        json.dumps([{"name": "teaser.png", "caption": caption}])
    )

    model.interrupt_during = "plotting.code"
    # A process started in the background ignores the signal unless it asks for it.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            execute()
    finally:
        signal.signal(signal.SIGINT, previous or signal.SIG_DFL)

    # Literature and plotting ran on their own threads; both stopped before the interrupt
    # was passed on, and nothing went on to write the paper.
    assert [t.name for t in threading.enumerate() if t.name.startswith("stage-")] == []
    state = json.loads(paths.state.read_text())
    assert state["status"] == "interrupted"
    assert state["stages"]["plotting"]["status"] == "interrupted"
    assert state["stages"]["literature"]["status"] in ("completed", "interrupted")
    assert state["stages"]["section_writing"]["status"] == "pending"
    assert "section_writing.SectionWritingAgent" not in client.labels()
    spent = len(client.requests)
    assert json.loads(paths.usage.read_text())["total"]["requests"] == spent

    result, harness = execute()

    assert result.success, result.error
    state = json.loads(paths.state.read_text())
    assert {stage["status"] for stage in state["stages"].values()} == {"completed"}
    assert state["stages"]["outline"]["reused"] and not state["stages"]["plotting"]["reused"]
    assert paths.final_pdf.read_bytes().startswith(b"%PDF")
    # The literature synthesis exists whether or not the interrupt came before it.
    assert state["stages"]["literature"]["after_done"] is True
    assert harness.project.exists("wiki/synthesis/architectures.md")
    # Both invocations are on the books.
    assert json.loads(paths.usage.read_text())["total"]["requests"] == len(client.requests)

    # The paper has the generated figure only.
    figures = json.loads(paths.figures_info.read_text())
    assert [figure["name"] for figure in figures] == ["fig_main_results.png"]
    assert not (paths.figures_dir / "teaser.png").exists()
    assert "teaser" not in paths.final_tex.read_text()
    assert not harness.project.exists("wiki/figures/teaser.md")
    for request in client.requests:
        assert "ORIGINAL-CAPTION" not in f"{request.system}\n{request.text}", request.label
