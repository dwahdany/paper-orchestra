from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from paper_orchestra.agents.literature import (
    ABSTRACT_LIMIT,
    CACHE_DIR,
    CandidatePaper,
    DiscoveryResult,
    LiteratureAgent,
    LiteratureResult,
    PaperData,
    _ScholarCache,
    bibtex_entry,
    check_document,
    collect_search_tasks,
    escape_bibtex,
    generate_bibtex,
    inject_citations,
    make_citation_key,
    unique_citation_key,
)
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.config import ModelConfig, PipelineConfig
from paper_orchestra.context import RunContext, RunPaths
from paper_orchestra.llm.base import Capabilities, LLMError, WebSearch
from paper_orchestra.llm.fake import FakeLLM, FakeRequest
from paper_orchestra.memory import ContextHarness, TranscriptLog, Wiki
from paper_orchestra.prompts.literature import (
    DISCOVERY_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    min_cited,
    writer_user_prompt,
)
from paper_orchestra.utils.latex import bib_keys, cite_keys
from paper_orchestra.utils.scholar import (
    SemanticScholar,
    SemanticScholarError,
    normalize_title,
    title_similarity,
)

TEMPLATE = r"""\documentclass{article}
\usepackage{cvpr}  % venue style
\usepackage[capitalize]{cleveref}
\title{My Title}
\begin{document}
\maketitle
\begin{abstract}
Abstract here.
\end{abstract}
\section{Introduction}
Introduction here.
\section{Related Work}
Related work here.
\section{Methodology}
Methodology here.
\bibliography{references}
\end{document}
"""

ADAM_HINT = "Kingma et al. (Adam: A Method for Stochastic Optimization)"

SURVEY = "Lovelace2021SurveyWidgets"
FOUNDATIONS = "Gauss2019WidgetFoundations"
DRIFT = "Scully2022WidgetDrift"
ADAM = "Kingma2014AdamMethod"


def make_outline() -> dict:
    return {
        "plotting_plan": [],
        "intro_related_work_plan": {
            "introduction_strategy": {
                "hook_hypothesis": "Widgets matter.",
                "problem_gap_hypothesis": "Widgets drift.",
                "search_directions": ["widget surveys"],
            },
            "related_work_strategy": {
                "overview": "One cluster.",
                "subsections": [
                    {
                        "subsection_title": "2.1 Widget Models",
                        "methodology_cluster": "Widgets",
                        "sota_investigation_mission": "Find widget models.",
                        "limitation_hypothesis": "They drift.",
                        "limitation_search_queries": ["widget drift failure"],
                        "bridge_to_our_method": "We do not drift.",
                    }
                ],
            },
        },
        "section_plan": [
            {
                "section_title": "3. Method",
                "subsections": [
                    {
                        "subsection_title": "3.1 Core",
                        "content_bullets": ["Define the loss.", "State the optimizer."],
                        "citation_hints": [ADAM_HINT],
                    },
                    {
                        "subsection_title": "3.2 Details",
                        "content_bullets": [],
                        "citation_hints": [],
                    },
                ],
            }
        ],
    }


def s2_paper(
    title: str,
    authors: list[str],
    year: int | None,
    date: str | None,
    *,
    abstract: str | None = "An abstract.",
    venue: str = "NeurIPS",
    journal: dict | None = None,
    citations: int = 10,
) -> dict:
    return {
        "paperId": "0" * 40,
        "title": title,
        "authors": [{"authorId": "1", "name": name} for name in authors],
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": citations,
        "journal": journal,
        "publicationDate": date,
    }


LIBRARY = [
    s2_paper(
        "A Survey of Widgets",
        ["Ada Lovelace", "Bob Stone"],
        2021,
        "2021-03-02",
        abstract="We survey widgets.\nSecond line with 100% & more.",
        citations=321,
    ),
    s2_paper("Widget Foundations", ["Carl Gauss"], 2019, "2019-05-01", abstract="F" * 2000),
    s2_paper("Widget Drift in Practice", ["Dana Scully"], 2022, "2022-07-09"),
    s2_paper(
        "Adam: A Method for Stochastic Optimization",
        ["Diederik P. Kingma", "Jimmy Ba"],
        2014,
        "2014-12-22",
        venue="International Conference on Learning Representations",
        journal={"name": "CoRR", "volume": "abs/1412.6980"},
    ),
    s2_paper("Future Widgets", ["Eve Future"], 2025, "2025-01-10"),
    s2_paper("Abstractless Widgets", ["Nora Null"], 2020, "2020-02-02", abstract=None),
]

DISCOVERY = {
    "widget surveys": [
        {"title": "A Survey of Widgets", "year": 2021, "reason": "The standard survey."},
        {"title": "Widget Foundations", "year": 2019, "reason": "Founded the field."},
        {"title": "Totally Invented Paper Title About Nothing", "year": 2022, "reason": "?"},
        {"title": "Future Widgets", "year": 2025, "reason": "Recent."},
        {"title": "Abstractless Widgets", "year": 2020, "reason": "No abstract."},
    ],
    "widget drift failure": [
        {"title": "Widget Drift in Practice", "year": 2022, "reason": "Documents drift."},
        {"title": "a survey of  widgets", "year": 2021, "reason": "Also covers drift."},
    ],
    ADAM_HINT: [
        {"title": "Adam: A Method for Stochastic Optimization", "year": 2015, "reason": "Adam."}
    ],
}


class FakeResponse:
    def __init__(self, status: int, data: list[dict] | None = None) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._data = data or []

    def json(self) -> dict:
        return {"total": len(self._data), "data": self._data}


class FakeS2Session:
    """A tiny Semantic Scholar: returns the ``limit`` library titles closest to the query."""

    def __init__(self, library: list[dict] = LIBRARY, *, failing: tuple[str, ...] = ()) -> None:
        self.library = library
        self.failing = failing
        self.queries: list[str] = []

    def get(self, url, *, headers, params, timeout):
        query = params["query"]
        self.queries.append(query)
        if any(needle in query for needle in self.failing):
            return FakeResponse(500)
        ranked = sorted(self.library, key=lambda p: -title_similarity(query, p["title"]))
        return FakeResponse(200, ranked[: params["limit"]])


def make_scholar(session: FakeS2Session | None = None) -> SemanticScholar:
    return SemanticScholar(session or FakeS2Session(), sleep=lambda _: None, max_retries=1)


def checklist(request: FakeRequest) -> list[str]:
    block = re.search(r"<citation_checklist>\n(.*?)\n</citation_checklist>", request.text, re.S)
    keys = block.group(1).split() if block else []
    return [] if keys == ["(empty)"] else keys


def filled(keys: list[str], *, template: str = TEMPLATE) -> str:
    intro = f"Widgets matter \\cite{{{', '.join(keys)}}}." if keys else "Widgets matter."
    return template.replace("Introduction here.", intro).replace(
        "Related work here.", "Prior work drifts."
    )


def fenced(latex: str) -> str:
    return f"Here is the file.\n```latex\n{latex}```\n"


def good_writer(request: FakeRequest) -> str:
    return fenced(filled(checklist(request)))


class Script:
    """Model behaviour for a test: discovery answers by task, plus a writer."""

    def __init__(
        self,
        discovery: dict | None = None,
        writer: Callable[[FakeRequest], str] = good_writer,
    ) -> None:
        self.discovery = DISCOVERY if discovery is None else discovery
        self.writer = writer

    def __call__(self, request: FakeRequest) -> str:
        if request.label != "literature.discovery":
            return self.writer(request)
        for needle, answer in self.discovery.items():
            if f">{needle}</" in request.text:
                if isinstance(answer, Exception):
                    raise answer
                if isinstance(answer, str):
                    return answer
                return json.dumps({"candidates": answer})
        raise AssertionError(f"no scripted discovery answer for: {request.text[:200]}")


def make_ctx(
    tmp_path: Path,
    client: FakeLLM,
    *,
    name: str = "run",
    lab: bool = True,
    cutoff: str = "2024-10",
    **cfg,
) -> RunContext:
    base = tmp_path / name
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
        base_dir=base,
        template_dir=tmp_path / "template",
        research_cutoff=cutoff,
        models=ModelConfig(writer="fake-writer", fast="fake-fast"),
        **cfg,
    )
    harness = ContextHarness(
        project=Wiki(base / "memory"),
        lab=Wiki(tmp_path / "lab", name="lab", read_only=()) if lab else None,
        transcripts=TranscriptLog(base / "memory" / "transcripts"),
        client_for=lambda model: client,
        default_effort="high",
    )
    return RunContext(cfg=config, paths=paths, harness=harness, compute=LocalBackend())


def run_agent(
    tmp_path: Path,
    script: Script | None = None,
    *,
    session: FakeS2Session | None = None,
    outline: dict | None = None,
    **ctx_kwargs,
) -> tuple[LiteratureResult, RunContext, FakeLLM, FakeS2Session]:
    client = FakeLLM(script or Script())
    session = session or FakeS2Session()
    ctx = make_ctx(tmp_path, client, **ctx_kwargs)
    result = LiteratureAgent(ctx, scholar=make_scholar(session)).run(outline or make_outline())
    return result, ctx, client, session


def discovery_requests(client: FakeLLM) -> list[FakeRequest]:
    return [r for r in client.requests if r.label == "literature.discovery"]


def writer_requests(client: FakeLLM) -> list[FakeRequest]:
    return [r for r in client.requests if r.label == "literature.LiteratureAgent"]


def notes(ctx: RunContext) -> list[tuple[str, str]]:
    wiki = ctx.harness.project
    return [(info.kind, wiki.read(info.path).body) for info in wiki.pending_notes()]


def all_candidate_keys(outline: dict) -> set[str]:
    plan = outline["intro_related_work_plan"]
    keys = set(plan["introduction_strategy"]["citation_candidates"])
    for cluster in plan["related_work_strategy"]["subsections"]:
        keys |= set(cluster["citation_candidates"])
    for section in outline["section_plan"]:
        for subsection in section["subsections"]:
            keys |= set(subsection["citation_candidates"])
    return keys


def paper_data(key: str = "Key2020Some", **overrides) -> PaperData:
    fields = {
        "citation_key": key,
        "title": "Some Title",
        "authors": ["Ann Author"],
        "venue": "NeurIPS",
        "year": 2020,
        "abstract": "Abstract.",
        "citation_count": 1,
        "found_in_section": "Introduction",
        "sections": ["Introduction"],
        "reason": "",
    }
    return PaperData(**(fields | overrides))


# -- search tasks ------------------------------------------------------------


def test_collect_search_tasks_covers_the_three_families_in_outline_order():
    outline = make_outline()
    clusters = outline["intro_related_work_plan"]["related_work_strategy"]["subsections"]
    clusters.append(
        {"subsection_title": "2.2 Gadgets", "methodology_cluster": "Gadget tuning"}  # no queries
    )
    outline["section_plan"][0]["subsections"][1]["citation_hints"] = [ADAM_HINT, "  ", ADAM_HINT]
    outline["section_plan"].append(
        {"section_title": "5. Conclusion", "citation_hints": ["paper introducing 'WidgetBench'"]}
    )
    tasks = collect_search_tasks(outline)

    assert [(t.section, t.focus, t.search_type) for t in tasks] == [
        ("Introduction", "widget surveys", "exploration"),
        ("Related Work: 2.1 Widget Models", "widget drift failure", "exploration"),
        ("Related Work: 2.2 Gadgets", "2.2 Gadgets Gadget tuning", "exploration"),
        ("3. Method - 3.1 Core", ADAM_HINT, "targeted"),
        ("3. Method - 3.2 Details", ADAM_HINT, "targeted"),
        ("5. Conclusion", "paper introducing 'WidgetBench'", "targeted"),
    ]
    assert tasks[0].context == "Hook: Widgets matter.. Gap: Widgets drift."
    assert tasks[1].context == "Mission: Find widget models.. Hypothesis: They drift."
    assert tasks[3].context == (
        "Must-have citation for section covering: Define the loss. State the optimizer."
    )
    assert collect_search_tasks({}) == []


def test_inject_citations_lists_each_paper_where_it_was_found_and_drops_hints():
    outline = make_outline()
    outline["section_plan"].append({"section_title": "5. Conclusion", "citation_hints": ["x"]})
    papers = [
        paper_data("B", sections=["Introduction", "3. Method - 3.1 Core"]),
        paper_data("A", sections=["Related Work: 2.1 Widget Models", "Introduction"]),
        paper_data("C", sections=["5. Conclusion"]),
    ]
    updated = inject_citations(outline, papers)
    plan = updated["intro_related_work_plan"]
    assert plan["introduction_strategy"]["citation_candidates"] == ["B", "A"]  # discovery order
    assert plan["related_work_strategy"]["subsections"][0]["citation_candidates"] == ["A"]
    core, details = updated["section_plan"][0]["subsections"]
    assert core["citation_candidates"] == ["B"] and details["citation_candidates"] == []
    assert "citation_hints" not in core and "citation_hints" not in details
    assert updated["section_plan"][1] == {
        "section_title": "5. Conclusion",
        "citation_candidates": ["C"],
    }

    bare = inject_citations({"section_plan": []}, papers)
    assert bare["intro_related_work_plan"]["introduction_strategy"]["citation_candidates"] == [
        "B",
        "A",
    ]


# -- discovery schema --------------------------------------------------------


def test_discovery_answers_tolerate_sloppy_candidates(caplog):
    result = DiscoveryResult.model_validate(
        {
            "section_name": "Introduction",
            "candidates": [
                {"title": "Good Paper", "year": "2021", "reason": "r", "url": "http://x"},
                {"title": "No Year Given"},
                {"title": "Odd Year", "year": "circa 2019?", "reason": None},
                {"title": "Not A Number", "year": float("nan"), "reason": 7},
                {"title": None, "year": 2020},
                {"year": 2020, "reason": "no title"},
                "A Bare Title",
                ["a", "list"],
                {"title": "   "},
            ],
        }
    )
    assert [(c.title, c.year, c.reason) for c in result.candidates] == [
        ("Good Paper", 2021, "r"),
        ("No Year Given", None, ""),
        ("Odd Year", 2019, ""),
        ("Not A Number", None, "7"),
        ("A Bare Title", None, ""),
    ]
    assert "Ignored 4 candidate(s) without a title" in caplog.text
    schema = DiscoveryResult.model_json_schema()
    assert schema["$defs"]["CandidatePaper"]["required"] == ["title", "year", "reason"]
    assert CandidatePaper(title="t", year=2020.0, reason="").year == 2020


def test_discovery_answers_tolerate_a_null_list_and_titles_that_are_not_text(caplog):
    # "No candidate" may come back as null, and one unusable title must not cost the others.
    assert DiscoveryResult.model_validate({"candidates": None}).candidates == []
    result = DiscoveryResult.model_validate(
        {
            "candidates": [
                {"title": 1984, "year": 1949, "reason": "a number"},
                {"title": ["A", "List"], "year": 2020},
                {"title": {"text": "An Object"}},
                {"title": "Kept", "year": 2020, "reason": "r"},
            ]
        }
    )
    assert [(c.title, c.year) for c in result.candidates] == [("Kept", 2020)]
    assert "Ignored 3 candidate(s) without a title" in caplog.text


# -- citation keys and BibTeX ------------------------------------------------


def test_citation_keys_are_ascii_and_never_invent_a_year():
    assert make_citation_key(["Kaiming He"], 2016, "Deep Residual Learning for Image") == (
        "He2016DeepResidual"
    )
    assert make_citation_key(["Jürgen Müller"], 2020, "The β-VAE: a study") == "Muller2020Vae"
    assert make_citation_key(["李 飞飞"], 2009, "ImageNet") == "Unknown2009Imagenet"
    assert make_citation_key([], None, "Of the and") == "UnknownndPaper"
    assert make_citation_key(["Cher"], 2001, "On-line Learning") == "Cher2001OnlineLearning"
    # Letters that have no base-plus-accent form are transliterated, not dropped.
    assert make_citation_key(["Anders Søgaard"], 2019, "Æsthetics of Łódź") == (
        "Sogaard2019AestheticsLodz"
    )
    assert make_citation_key(["Đorđe Đurić"], 2019, "Straße") == "Duric2019Strasse"


def test_unique_citation_key_ignores_case_and_never_runs_out_of_suffixes():
    taken: set[str] = set()
    assert unique_citation_key("He2016Deep", taken) == "He2016Deep"
    assert unique_citation_key("He2016Deep", taken) == "He2016Deepa"
    assert unique_citation_key("HE2016DEEP", taken) == "HE2016DEEPb"
    assert unique_citation_key("He2016Deepa", taken) == "He2016Deepaa"
    many = [unique_citation_key("X", taken) for _ in range(60)]
    assert len({key.casefold() for key in many}) == 60
    assert many[27] == "Xaa" and all(re.fullmatch(r"X[a-z]*", key) for key in many)


def test_escape_bibtex_handles_specials_accents_and_unknown_characters():
    assert escape_bibtex("Q&A with 100% accuracy on A_B #1 {unbalanced") == (
        r"Q\&A with 100\% accuracy on A\_B \#1 \textbraceleft{}unbalanced"
    )
    assert escape_bibtex("a\\b ~ x^2 $y$ }") == (
        r"a\textbackslash{}b \textasciitilde{} x\textasciicircum{}2 \$y\$ \textbraceright{}"
    )
    assert escape_bibtex("Müller, Łukasz, Søren, naïve, Čech, façade") == (
        r"M{\"{u}}ller, {\L}ukasz, S{\o}ren, na{\"{\i}}ve, {\v{C}}ech, fa{\c{c}}ade"
    )
    assert escape_bibtex("β-VAE and Γ – “quoted” … 深度") == (
        r"{$\beta$}-VAE and {$\Gamma$} -- ``quoted'' \ldots{}"
    )
    assert escape_bibtex("  multi\nline\ttitle ") == "multi line title"
    assert escape_bibtex("李 飞飞") == ""
    # BibTeX counts braces whether or not they are escaped, so they must pair up as written.
    for text in ("Q&A {x", "}{", "a\\b", "{{{", "x} y {z", "Müller}"):
        escaped = escape_bibtex(text)
        assert escaped.count("{") == escaped.count("}")
        depth = 0
        for char in escaped:
            depth += (char == "{") - (char == "}")
            assert depth >= 0


def test_escape_bibtex_only_uses_what_the_default_font_encoding_has():
    # The ogonek (\k), d-with-stroke, eth and thorn exist only in T1, which the venue
    # templates do not load: the letters keep their base form instead of raising an error.
    assert escape_bibtex("Stanisław Jastrzębski, Kazlauskų, Įvairūs") == (
        r"Stanis{\l}aw Jastrzebski, Kazlausku, Ivair{\={u}}s"
    )
    assert escape_bibtex("Đorđe Þór Guðmundsson") == r"Dorde Th{\'{o}}r Gudmundsson"
    assert "\\k" not in escape_bibtex("ąęįųĄĘĮŲ")
    # Hyphens, minus and multiplication signs of publisher metadata are kept, not deleted.
    publisher_title = "Parameter\u2010Efficient non\u2011local 3×3 \u2212 5 µm O\u02bcNeil"
    assert escape_bibtex(publisher_title) == (
        r"Parameter-Efficient non-local 3{$\times$}3 - 5 {$\mu$}m O'Neil"
    )
    # In OT1 text a bare < > | prints as an inverted exclamation mark, question mark and dash.
    assert escape_bibtex("a < b > c | d") == r"a \textless{} b \textgreater{} c \textbar{} d"


def test_escape_bibtex_only_emits_greek_macros_that_latex_defines():
    # Unicode calls the letter "lamda"; \lamda would be an undefined control sequence.
    assert escape_bibtex("TD(λ) with Λ, ο and ϝ") == r"TD({$\lambda$}) with {$\Lambda$}, o and d"
    defined = {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
        "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi",
        "Psi", "Omega",
    }  # fmt: skip
    greek_and_coptic = "".join(chr(code) for code in range(0x370, 0x400))
    assert set(re.findall(r"\\([A-Za-z]+)", escape_bibtex(greek_and_coptic))) == defined


def test_escape_bibtex_escapes_specials_hidden_in_compatibility_characters():
    # A full-width ampersand reduces to "&", which must not reach the .bbl unescaped.
    assert escape_bibtex("Ａ＆Ｂ ５０％ ｛ｘ｝ ＿ ＃") == (
        r"A\&B 50\% \textbraceleft{}x\textbraceright{} \_ \#"
    )
    # TeX stops at a control character ("Text line contains an invalid character").
    assert escape_bibtex("a\x00b\x01 c\x7f d\x1fe") == "ab c d e"
    blocks = (range(0x00, 0x3100), range(0xFB00, 0xFB50), range(0xFE50, 0xFF70))
    for code in (code for block in blocks for code in block):
        escaped = escape_bibtex(chr(code))
        # What is left once the macros are taken out has to be inert text (a bare ~ is a
        # tie, which is what a no-break space becomes).
        plain = re.sub(r"\\[A-Za-z]+|\\[^A-Za-z]", "", escaped)
        assert not set(plain) & set("\\&%#_^<>|"), (hex(code), escaped)
        assert plain.count("$") % 2 == 0, (hex(code), escaped)
        assert escaped.count("{") == escaped.count("}"), (hex(code), escaped)


def test_bibtex_entry_types_follow_the_source():
    preprint = paper_data(
        "A", venue="arXiv.org", journal="ArXiv", volume="abs/2106.09685", pages="1-12"
    )
    assert bibtex_entry(preprint) == (
        "@article{A,\n  title={{Some Title}},\n  author={Ann Author},\n"
        "  journal={arXiv preprint arXiv:2106.09685},\n  year={2020}\n}"
    )
    article = paper_data(
        "B", authors=["Ann Author", "Bo Li"], journal="Nature", volume="521", pages="436 - 444"
    )
    assert bibtex_entry(article) == (
        "@article{B,\n  title={{Some Title}},\n  author={Ann Author and Bo Li},\n"
        "  journal={Nature},\n  year={2020},\n  volume={521},\n  pages={436--444}\n}"
    )
    proceedings = paper_data(
        "C",
        journal="2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        volume="1",
        pages="770-778",
    )
    assert bibtex_entry(proceedings) == (
        "@inproceedings{C,\n  title={{Some Title}},\n  author={Ann Author},\n"
        "  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},\n"
        "  year={2020},\n  pages={770--778}\n}"
    )
    venue_only = paper_data("D", authors=[], year=None)
    assert bibtex_entry(venue_only) == (
        "@inproceedings{D,\n  title={{Some Title}},\n  author={Unknown},\n  booktitle={NeurIPS}\n}"
    )
    unpublished = paper_data("E", venue="arXiv")
    assert "journal={arXiv preprint}" in bibtex_entry(unpublished)

    bib = generate_bibtex([preprint, article, proceedings, venue_only, unpublished])
    assert bib_keys(bib) == {"A", "B", "C", "D", "E"} and bib.endswith("}\n")
    assert generate_bibtex([]) == ""


@pytest.mark.latex
def test_generated_bibtex_compiles_with_hostile_metadata(tmp_path):
    papers = [
        paper_data(
            "Hostile2020",
            title="Q&A with 100% accuracy on A_B #1 {unbalanced: β-VAE ~ x^2 $cheap$ \\o/ naïve",
            authors=["Jürgen Müller", "Łukasz Kaiser", "Søren Ørsted", "李 飞飞"],
            journal="Journal of Machine Learning & Research",
            volume="12_3",
            pages="1-2",
        ),
        paper_data("Plain2019", title="Plain", venue="Proceedings of Things #5", year=2019),
        paper_data("Preprint2021", venue="arXiv.org", journal="ArXiv", volume="abs/2101.00001"),
    ]
    (tmp_path / "references.bib").write_text(generate_bibtex(papers), encoding="utf-8")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "Text \\cite{Hostile2020,Plain2019,Preprint2021}.\n"
        "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    result = LocalBackend().compile_latex(tmp_path, "paper")
    assert result.ok and result.errors == [], result.feedback()
    # The log of the last pass: every citation resolved, and BibTeX had nothing to say.
    assert "undefined" not in (tmp_path / "paper.log").read_text(errors="replace")
    bibtex_log = (tmp_path / "paper.blg").read_text(errors="replace")
    assert "Warning--" not in bibtex_log and "error message" not in bibtex_log
    bbl = (tmp_path / "paper.bbl").read_text()
    assert bbl.count("\\bibitem") == 3 and "{\\L}ukasz" in bbl


@pytest.mark.latex
def test_generated_bibtex_needs_no_font_encoding_beyond_the_default(tmp_path):
    papers = [
        paper_data(
            "Houlsby2019",
            title="Parameter\u2010Efficient TD(λ): a < b | c > d, 3×3 \u2212 5 µm",
            authors=[
                "Stanisław Jastrzębski",
                "Đorđe Miladinović",
                "Þór Guðmundsson",
                "Jonas Kazlauskų",
            ],
        )
    ]
    (tmp_path / "references.bib").write_text(generate_bibtex(papers), encoding="utf-8")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nText \\cite{Houlsby2019}.\n"
        "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    result = LocalBackend().compile_latex(tmp_path, "paper")
    assert result.ok and result.errors == [], result.feedback()
    # The log of the last pass, read directly: nothing went wrong inside the .bbl either.
    log_text = (tmp_path / "paper.log").read_text(errors="replace")
    assert "LaTeX Error" not in log_text and "Undefined control sequence" not in log_text
    bbl = (tmp_path / "paper.bbl").read_text()
    assert "Jastrzebski" in bbl and "Dorde" in bbl and "Parameter-Efficient" in bbl


def test_author_names_bibtex_cannot_parse_are_taken_literally():
    odd = paper_data(
        "Odd2020",
        authors=[
            "Dept. of Computer Science, Univ. of X, City, Country",  # an affiliation
            "Ed ] Itor",
            ",",
            "Smith, Jr., John",
            "Ann Author,",
        ],
    )
    assert (
        "  author={{Dept. of Computer Science, Univ. of X, City, Country} and {Ed ] Itor} "
        "and Smith, Jr., John and Ann Author},\n"
    ) in bibtex_entry(odd)


@pytest.mark.latex
def test_generated_bibtex_compiles_with_author_year_citations_and_odd_names(tmp_path):
    papers = [
        paper_data(
            "Odd2020",
            authors=["Itor ]", "Dept. of Computer Science, Univ. of X, City, Country", ","],
        ),
        paper_data("Plain2021", authors=["Stanisław Jastrzębski", "Ann Author"], year=2021),
    ]
    (tmp_path / "references.bib").write_text(generate_bibtex(papers), encoding="utf-8")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\usepackage{natbib}\n\\begin{document}\n"
        "\\citet{Odd2020} and \\citep{Plain2021}.\n"
        "\\bibliographystyle{plainnat}\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    result = LocalBackend().compile_latex(tmp_path, "paper")
    assert result.ok and result.errors == [], result.feedback()
    # The log of the last pass, read directly: the .aux and the .bbl were read cleanly.
    log_text = (tmp_path / "paper.log").read_text(errors="replace")
    for symptom in ("LaTeX Error", "natbib Error", "Missing \\endcsname"):
        assert symptom not in log_text
    assert "error message" not in (tmp_path / "paper.blg").read_text(errors="replace")


# -- the whole stage ---------------------------------------------------------


def test_run_end_to_end(tmp_path):
    original = make_outline()
    result, ctx, _, session = run_agent(tmp_path, outline=original)
    paths = ctx.paths

    # Four of the seven distinct candidates are real, early enough and have an abstract.
    assert result.papers == 4
    keys = [SURVEY, FOUNDATIONS, DRIFT, ADAM]
    assert list(result.citation_map) == keys

    # Every artefact uses the same citation keys.
    assert bib_keys(result.bibtex) == set(keys)
    assert all_candidate_keys(result.outline) == set(keys)
    wiki = ctx.harness.project
    assert {Path(info.path).stem for info in wiki.list("wiki/literature")} == set(keys)
    assert cite_keys(result.latex) == set(keys)

    # Each part of the outline lists what its own searches found; hints are gone.
    plan = result.outline["intro_related_work_plan"]
    assert plan["introduction_strategy"]["citation_candidates"] == [SURVEY, FOUNDATIONS]
    assert plan["related_work_strategy"]["subsections"][0]["citation_candidates"] == [
        SURVEY,
        DRIFT,
    ]
    core, details = result.outline["section_plan"][0]["subsections"]
    assert core["citation_candidates"] == [ADAM] and details["citation_candidates"] == []
    assert "citation_hints" not in core and "citation_hints" not in details
    assert original == make_outline()  # the caller's outline is left alone

    # Files: the old names in the stage directory, and copies where later stages read them.
    out = paths.literature_dir
    assert sorted(p.name for p in out.iterdir()) == [
        "citation_map.json",
        "outline_v1.json",
        "references.bib",
        "updated_template.tex",
    ]
    assert json.loads((out / "outline_v1.json").read_text()) == result.outline
    assert json.loads(paths.outline.read_text()) == result.outline
    assert (out / "updated_template.tex").read_text() == result.latex
    assert paths.template_tex.read_text() == result.latex
    assert (out / "references.bib").read_text() == result.bibtex
    assert paths.references_bib.read_text() == result.bibtex
    assert json.loads(paths.citation_map.read_text()) == result.citation_map

    survey = result.citation_map[SURVEY]
    assert survey == {
        "citation_key": SURVEY,
        "title": "A Survey of Widgets",
        "authors": ["Ada Lovelace", "Bob Stone"],
        "venue": "NeurIPS",
        "year": 2021,
        "abstract": "We survey widgets.\nSecond line with 100% & more.",
    }
    clipped = result.citation_map[FOUNDATIONS]["abstract"]
    assert clipped == "F" * ABSTRACT_LIMIT + "... [Truncated]"
    assert "@article{Kingma2014AdamMethod," in result.bibtex
    assert "journal={CoRR}" in result.bibtex and "volume={abs/1412.6980}" in result.bibtex

    # One lookup per distinct title; the repeated candidate costs nothing.
    assert sorted(session.queries) == sorted(
        [
            "A Survey of Widgets",
            "Widget Foundations",
            "Totally Invented Paper Title About Nothing",
            "Future Widgets",
            "Abstractless Widgets",
            "Widget Drift in Practice",
            "Adam: A Method for Stochastic Optimization",
        ]
    )

    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert event["papers"] == 4 and event["cited"] == 4
    assert (event["tasks"], event["failed_tasks"], event["candidates"]) == (3, 0, 8)
    assert (event["verified"], event["duplicates"], event["lookups"]) == (4, 1, 7)
    assert (event["unmatched"], event["no_abstract"], event["after_cutoff"]) == (2, 1, 0)
    assert notes(ctx) == []


def test_discovery_calls_are_cheap_memory_free_web_searches(tmp_path):
    _, _, client, _ = run_agent(tmp_path)
    requests = discovery_requests(client)
    assert len(requests) == 3
    for request in requests:
        assert request.model == "fake-fast"
        assert request.effort == "low"
        assert request.system == DISCOVERY_SYSTEM_PROMPT  # no memory block
        assert request.memory is None and request.tools == {}
        # Search results and a response format do not combine: JSON is asked for in text.
        assert request.schema is None and "JSON Schema" in request.text
        assert "<research_cutoff>2024-10</research_cutoff>" in request.text
        assert "Use web search" in request.text

    by_task = {("<hint>" in r.text): r for r in requests}
    targeted, exploration = by_task[True], by_task[False]
    assert targeted.web_search == WebSearch(max_uses=3)
    assert f"<hint>{ADAM_HINT}</hint>" in targeted.text
    assert "Define the loss. State the optimizer." in targeted.text
    assert "exactly one candidate" in targeted.text
    assert exploration.web_search == WebSearch(max_uses=6)
    assert "<project_problem>Widgets drift.</project_problem>" in exploration.text
    assert "Find 10-15 highly relevant" in exploration.text


def test_the_writer_gets_everything_in_tags_and_reads_memory(tmp_path):
    _, _, client, _ = run_agent(tmp_path)
    (request,) = writer_requests(client)
    assert request.model == "fake-writer" and request.effort == "high"
    assert request.system.startswith(WRITER_SYSTEM_PROMPT)
    assert "# Memory" in request.system and request.memory is not None
    assert "Strict Knowledge Isolation" in request.system
    assert "{{" not in WRITER_SYSTEM_PROMPT and "\\cite{key}" in WRITER_SYSTEM_PROMPT
    assert "\\usepackage[capitalize]{cleveref}" in WRITER_SYSTEM_PROMPT

    text = request.text
    assert f"<template>\n{TEMPLATE.strip()}\n</template>" in text
    assert "<idea>\nWe propose WidgetNet.\n</idea>" in text
    assert (
        "<experimental_log>\nWidgetNet reaches 91.2% on WidgetBench.\n</experimental_log>" in text
    )
    assert "<guidelines>\nEight pages. Double blind.\n</guidelines>" in text
    assert "<research_cutoff>2024-10</research_cutoff>" in text
    assert checklist(request) == [SURVEY, FOUNDATIONS, DRIFT, ADAM]
    assert "You have the abstracts of 4 collected papers. Cite at least 3 of them" in text

    plan = json.loads(text.split("<intro_related_work_plan>\n")[1].split("\n</intro_")[0])
    assert plan["introduction_strategy"]["citation_candidates"] == [SURVEY, FOUNDATIONS]
    papers = json.loads(text.split("<collected_papers>\n")[1].split("\n</collected_papers>")[0])
    assert [p["citation_key"] for p in papers] == [SURVEY, FOUNDATIONS, DRIFT, ADAM]
    assert papers[0]["found_in_section"] == "Introduction"
    assert papers[0]["sections"] == ["Introduction", "Related Work: 2.1 Widget Models"]
    assert papers[0]["reason"] == "The standard survey."
    assert papers[3]["publication_date"] == "2014-12-22" and papers[3]["year"] == 2014


def test_wiki_pages_quote_the_abstract_and_link_to_the_outline(tmp_path):
    result, ctx, _, _ = run_agent(tmp_path)
    wiki = ctx.harness.project

    page = wiki.read(f"wiki/literature/{SURVEY}.md")
    assert page.meta["kind"] == "literature"
    assert page.meta["citation_key"] == SURVEY
    assert page.meta["authors"] == ["Ada Lovelace", "Bob Stone"]
    assert (page.meta["year"], page.meta["venue"], page.meta["citation_count"]) == (
        2021,
        "NeurIPS",
        321,
    )
    assert page.summary == "Ada Lovelace, Bob Stone (2021), NeurIPS: A Survey of Widgets"
    assert "> We survey widgets.\n> Second line with 100% & more.\n" in page.body
    assert "\\cite{Lovelace2021SurveyWidgets}" in page.body
    assert "The standard survey." in page.body
    assert "- Introduction\n- Related Work: 2.1 Widget Models\n" in page.body
    assert "[[paper/outline]]" in page.body

    outline_page = wiki.read("wiki/paper/outline.md")
    assert "4 verified paper(s)" in outline_page.summary
    for key in result.citation_map:
        assert f"[[literature/{key}]]" in outline_page.body
    stored = json.loads(outline_page.body.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert stored == result.outline
    assert wiki.lint().broken_links == []


def test_stale_literature_pages_are_removed_but_other_pages_are_kept(tmp_path):
    client = FakeLLM(Script())
    ctx = make_ctx(tmp_path, client)
    wiki = ctx.harness.project
    wiki.write("wiki/literature/Old2019Stale.md", "gone", {"kind": "literature", "summary": "s"})
    wiki.write("wiki/literature/themes.md", "kept", {"kind": "synthesis", "summary": "s"})
    LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    stems = {Path(info.path).stem for info in wiki.list("wiki/literature")}
    assert stems == {SURVEY, FOUNDATIONS, DRIFT, ADAM, "themes"}


@pytest.mark.parametrize("workers", [1, 8])
def test_results_do_not_depend_on_the_number_of_workers(tmp_path, workers):
    result, _, _, _ = run_agent(tmp_path, literature_workers=workers)
    assert list(result.citation_map) == [SURVEY, FOUNDATIONS, DRIFT, ADAM]


def test_without_web_search_discovery_uses_native_structured_output(tmp_path, caplog):
    client = FakeLLM(
        Script(),
        capabilities=Capabilities(
            tools=True, memory_tool=True, web_search=False, structured_output=True
        ),
    )
    ctx = make_ctx(tmp_path, client)
    result = LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    assert result.papers == 4
    for request in discovery_requests(client):
        assert request.web_search is None
        assert request.schema is DiscoveryResult
        assert "Web search is not available" in request.text
        assert "Use web search" not in request.text
    assert "has no web search" in caplog.text


# -- cutoff ------------------------------------------------------------------


def test_cutoff_edges(tmp_path):
    library = [
        s2_paper("Edge Last Day Before", ["A One"], 2024, "2024-09-30"),
        s2_paper("Edge First Day Of Cutoff Month", ["B Two"], 2024, "2024-10-01"),
        s2_paper("Edge Year Only In Cutoff Year", ["C Three"], 2024, None),
        s2_paper("Edge Year Only Earlier Year", ["D Four"], 2023, None),
        s2_paper("Edge Without Any Date", ["E Five"], None, None),
        s2_paper("Edge Date Only", ["F Six"], None, "2022-02-02"),
    ]
    script = Script(
        {
            "widget surveys": [{"title": p["title"], "year": 2024, "reason": ""} for p in library],
            "widget drift failure": [],
            ADAM_HINT: [],
        }
    )
    result, _, _, _ = run_agent(tmp_path, script, session=FakeS2Session(library))
    titles = {entry["title"]: entry["year"] for entry in result.citation_map.values()}
    assert titles == {
        "Edge Last Day Before": 2024,
        "Edge Year Only Earlier Year": 2023,
        "Edge Date Only": 2022,  # the year comes from the date, never from the model's guess
    }
    assert "Six2022EdgeDate" in result.citation_map


def test_the_registry_applies_the_cutoff_even_if_the_lookup_did_not(tmp_path):
    class LeakyScholar:
        def find_paper(self, title, *, year_hint=None, cutoff=None):
            return next((p for p in LIBRARY if p["title"] == title), None)

    client = FakeLLM(Script())
    ctx = make_ctx(tmp_path, client)
    result = LiteratureAgent(ctx, scholar=LeakyScholar()).run(make_outline())
    assert "Future Widgets" not in {entry["title"] for entry in result.citation_map.values()}
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert event["after_cutoff"] == 1 and event["verified"] == 4
    lab = ctx.harness.lab
    cached = "\n".join(lab.read(info.path).body for info in lab.list(CACHE_DIR))
    assert "A Survey of Widgets" in cached and "Future Widgets" not in cached


@pytest.mark.parametrize("cutoff", ["2024", "2024/10", "October 2024"])
def test_a_malformed_cutoff_fails_before_any_call(tmp_path, cutoff):
    client = FakeLLM(Script())
    session = FakeS2Session()
    ctx = make_ctx(tmp_path, client, cutoff=cutoff)
    with pytest.raises(ValueError, match="YYYY-MM"):
        LiteratureAgent(ctx, scholar=make_scholar(session)).run(make_outline())
    assert client.requests == [] and session.queries == []


@pytest.mark.parametrize("missing", ["idea", "experimental_log", "template_tex", "guidelines"])
def test_a_missing_input_fails_before_any_call(tmp_path, missing):
    client = FakeLLM(Script())
    ctx = make_ctx(tmp_path, client)
    getattr(ctx.paths, missing).unlink()
    with pytest.raises(FileNotFoundError, match=getattr(ctx.paths, missing).name):
        LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    assert client.requests == []


def test_control_characters_and_broken_unicode_in_metadata_are_dropped(tmp_path):
    library = [
        s2_paper(
            "Trans\u00adformers \x00Are\x01 Great\ud83d",  # soft hyphen, NUL, half an emoji
            ["Ann\u200b Author\x07"],
            2020,
            "2020-01-01",
            abstract="Line\x00 one\ud83d.\nLine two.",
        )
    ]
    script = Script(
        {
            "widget surveys": [{"title": "Transformers Are Great", "year": 2020, "reason": ""}],
            "widget drift failure": [],
            ADAM_HINT: [],
        }
    )
    result, ctx, _, _ = run_agent(tmp_path, script, session=FakeS2Session(library))

    ((key, entry),) = result.citation_map.items()
    assert key == "Author2020TransformersGreat"
    assert entry["title"] == "Transformers Are Great"
    assert entry["authors"] == ["Ann Author"]
    assert entry["abstract"] == "Line one.\nLine two."
    assert "title={{Transformers Are Great}}" in result.bibtex
    assert "author={Ann Author}" in result.bibtex
    # An unpaired surrogate cannot be encoded: every output file and page was still written.
    assert json.loads(ctx.paths.citation_map.read_text(encoding="utf-8")) == result.citation_map
    page = ctx.harness.project.read(f"wiki/literature/{key}.md")
    assert "> Line one.\n> Line two.\n" in page.body


def test_semantic_scholar_fields_of_an_unexpected_type_count_as_unknown(tmp_path):
    odd = s2_paper("Odd Shape Paper", ["A B"], 2020, "2020-01-01") | {
        "authors": 7,
        "citationCount": "many",
        "venue": None,
        "journal": "J",
    }
    textless = s2_paper("Second Odd Paper", ["A B"], 2020, "2020-01-01") | {"abstract": ["x"]}
    script = Script(
        {
            "widget surveys": [
                {"title": "Odd Shape Paper", "year": 2020, "reason": ""},
                {"title": "Second Odd Paper", "year": 2020, "reason": ""},
            ],
            "widget drift failure": [],
            ADAM_HINT: [],
        }
    )
    result, ctx, _, _ = run_agent(tmp_path, script, session=FakeS2Session([odd, textless]))

    ((key, entry),) = result.citation_map.items()
    assert (entry["title"], entry["authors"], entry["venue"]) == ("Odd Shape Paper", [], "arXiv")
    assert "author={Unknown}" in result.bibtex
    assert ctx.harness.project.read(f"wiki/literature/{key}.md").meta["citation_count"] == 0
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert (event["verified"], event["no_abstract"]) == (1, 1)


# -- citation key collisions -------------------------------------------------


def test_colliding_keys_are_made_unique_before_anything_uses_them(tmp_path):
    library = [
        s2_paper("Deep Learning for Widgets", ["Jane Smith"], 2020, "2020-01-01"),
        s2_paper("Deep Learning of Gadgets", ["John SMITH"], 2020, "2020-02-01"),
        s2_paper("The Deep Learning Primer", ["Ann Smith"], 2020, "2020-03-01"),
    ]
    script = Script(
        {
            "widget surveys": [{"title": library[0]["title"], "year": 2020, "reason": ""}],
            "widget drift failure": [
                {"title": library[1]["title"], "year": 2020, "reason": ""},
                {"title": library[0]["title"], "year": 2020, "reason": ""},
            ],
            ADAM_HINT: [{"title": library[2]["title"], "year": 2020, "reason": ""}],
        }
    )
    result, ctx, client, _ = run_agent(tmp_path, script, session=FakeS2Session(library))

    keys = ["Smith2020DeepLearning", "Smith2020DeepLearninga", "Smith2020DeepLearningb"]
    assert list(result.citation_map) == keys
    assert [entry["title"] for entry in result.citation_map.values()] == [
        "Deep Learning for Widgets",
        "Deep Learning of Gadgets",
        "The Deep Learning Primer",
    ]
    assert [m.group(1) for m in re.finditer(r"@\w+\{(\w+),", result.bibtex)] == keys
    plan = result.outline["intro_related_work_plan"]
    assert plan["introduction_strategy"]["citation_candidates"] == [keys[0]]
    assert plan["related_work_strategy"]["subsections"][0]["citation_candidates"] == [
        keys[0],
        keys[1],
    ]
    assert result.outline["section_plan"][0]["subsections"][0]["citation_candidates"] == [keys[2]]
    assert checklist(writer_requests(client)[0]) == keys
    wiki = ctx.harness.project
    assert {Path(info.path).stem for info in wiki.list("wiki/literature")} == set(keys)
    for key, entry in result.citation_map.items():
        assert wiki.read(f"wiki/literature/{key}.md").meta["title"] == entry["title"]
    assert cite_keys(result.latex) == set(keys)


# -- lab cache ---------------------------------------------------------------


def test_the_lab_cache_saves_lookups_across_runs(tmp_path):
    first, ctx, _, session = run_agent(tmp_path, name="first")
    assert len(session.queries) == 7

    lab = ctx.harness.lab
    pages = lab.list(CACHE_DIR)
    assert pages and all(info.kind == "cache" for info in pages)
    stored = "\n".join(lab.read(info.path).body for info in pages)
    assert "A Survey of Widgets" in stored and "Ada Lovelace" in stored
    assert "paperId" not in stored and "authorId" not in stored  # bibliographic fields only
    assert "Future Widgets" not in stored  # nothing past the cutoff is kept
    assert "WidgetNet" not in stored and "Widgets drift." not in stored  # nothing of our paper
    assert lab.lint().oversize == []

    second, _, _, again = run_agent(tmp_path, name="second")
    # Only the titles that have no match before the cutoff are asked for again.
    assert sorted(again.queries) == ["Future Widgets", "Totally Invented Paper Title About Nothing"]
    assert second.citation_map == first.citation_map
    assert second.bibtex == first.bibtex


def test_cache_hits_are_checked_against_the_current_cutoff(tmp_path):
    run_agent(tmp_path, name="first")  # fills the cache under cutoff 2024-10

    # An earlier cutoff: the cached 2022 paper must not come back, the 2021 survey may.
    down = FakeS2Session(failing=("Widget", "Adam", "Totally"))
    earlier, ctx, _, _ = run_agent(tmp_path, name="earlier", session=down, cutoff="2021-06")
    assert list(earlier.citation_map) == [SURVEY, FOUNDATIONS, ADAM]
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert event["cache_hits"] == 4  # survey, foundations, abstractless, adam
    assert "Widget Drift in Practice" in " ".join(down.queries)  # it had to be asked for again

    # A later cutoff may be owed newer papers, so cached answers are not trusted for it.
    later, _, _, session = run_agent(tmp_path, name="later", cutoff="2025-06")
    assert len(session.queries) == 7
    assert "Future2025FutureWidgets" in later.citation_map


def test_without_a_lab_wiki_every_run_asks_semantic_scholar(tmp_path):
    _, ctx, _, first = run_agent(tmp_path, name="first", lab=False)
    _, _, _, second = run_agent(tmp_path, name="second", lab=False)
    assert len(first.queries) == len(second.queries) == 7
    assert ctx.harness.lab is None and not (tmp_path / "lab").exists()


def test_the_cache_stays_bounded_and_survives_a_damaged_page(tmp_path, caplog):
    client = FakeLLM(Script())
    harness = make_ctx(tmp_path, client).harness
    cache = _ScholarCache(harness, "2024-10")

    # Enough titles that land on one shard page to overflow it several times.
    page = _ScholarCache._page(normalize_title("Paper 0"))
    titles = [
        f"Paper {i}"
        for i in range(4000)
        if _ScholarCache._page(normalize_title(f"Paper {i}")) == page
    ][:40]
    assert len(titles) == 40
    for title in titles:
        cache.put(title, {"title": title, "abstract": "```" + "a" * 1500, "year": 2020})

    body = harness.read_page(page, scope="lab")
    assert len(body) < 40_000  # under the wiki's soft page size, so the linter stays quiet
    assert cache.get(titles[0]) is None  # the oldest entries were evicted
    newest = cache.get(titles[-1])
    assert newest["title"] == titles[-1] and newest["abstract"].startswith("```aaa")
    assert len(harness.lab.list(CACHE_DIR)) == 1

    cache.put(titles[-1], {"title": "replaced"})
    assert cache.get(titles[-1]) == {"title": "replaced"}

    harness.write_page(page, "someone tidied this page up", {"summary": "s"}, scope="lab")
    assert cache.get(titles[-1]) is None
    assert "unreadable literature cache page" in caplog.text
    cache.put(titles[-1], {"title": "again"})
    assert cache.get(titles[-1]) == {"title": "again"}


# -- repair turns ------------------------------------------------------------


def scripted_writer(*replies: Callable[[list[str]], str]) -> Callable[[FakeRequest], str]:
    """A writer that gives the n-th reply on the n-th turn of the conversation."""

    first_turn: dict[str, list[str]] = {}

    def writer(request: FakeRequest) -> str:
        if request.turn == 0:
            first_turn["keys"] = checklist(request)
        return replies[min(request.turn, len(replies) - 1)](first_turn["keys"])

    return writer


def test_unknown_keys_and_truncation_are_repaired_in_the_same_conversation(tmp_path, caplog):
    def broken(keys: list[str]) -> str:
        document = filled([*keys, "Hallucinated2020Key"])
        return fenced(document.replace("\\end{document}\n", ""))

    writer = scripted_writer(broken, lambda keys: fenced(filled(keys)))
    result, ctx, client, _ = run_agent(tmp_path, Script(writer=writer))

    first, second = writer_requests(client)
    assert (first.turn, second.turn) == (0, 1)
    assert second.history[0][0] == first.text  # same conversation
    assert "Hallucinated2020Key" in second.text
    assert "\\end{document}" in second.text and "truncated" in second.text
    assert "complete corrected template.tex" in second.text

    assert result.latex == filled([SURVEY, FOUNDATIONS, DRIFT, ADAM])
    assert ctx.paths.template_tex.read_text() == result.latex
    assert "needs a repair turn" in caplog.text
    ((kind, note),) = notes(ctx)
    assert kind == "lesson"
    assert "Hallucinated2020Key" in note and "Resolved after 1 repair turn(s)" in note


def test_a_changed_preamble_is_repaired(tmp_path):
    def cleverref(keys: list[str]) -> str:
        return fenced(filled(keys).replace("{cleveref}", "{cleverref}"))

    writer = scripted_writer(cleverref, lambda keys: fenced(filled(keys)))
    result, _, client, _ = run_agent(tmp_path, Script(writer=writer))
    assert len(writer_requests(client)) == 2
    assert "preamble" in writer_requests(client)[1].text
    assert "\\usepackage[capitalize]{cleveref}" in result.latex


def test_a_lost_preamble_or_an_unwritten_section_is_repaired(tmp_path):
    def body_only(keys: list[str]) -> str:
        return fenced("\\section{Introduction}\nWidgets \\cite{" + keys[0] + "}.\n")

    def half_written(keys: list[str]) -> str:
        return fenced(filled(keys).replace("Prior work drifts.", "Related work here."))

    writer = scripted_writer(body_only, half_written, lambda keys: fenced(filled(keys)))
    result, _, client, _ = run_agent(tmp_path, Script(writer=writer))
    _, second, third = writer_requests(client)
    assert "missing \\documentclass" in second.text
    assert "Related Work section still contains the template placeholder" in third.text
    assert result.latex == filled([SURVEY, FOUNDATIONS, DRIFT, ADAM])


def test_a_reply_without_latex_is_asked_for_again(tmp_path):
    writer = scripted_writer(lambda keys: "I would rather not.", lambda keys: fenced(filled(keys)))
    result, _, client, _ = run_agent(tmp_path, Script(writer=writer))
    assert len(writer_requests(client)) == 2
    assert "did not contain the LaTeX document" in writer_requests(client)[1].text
    assert result.papers == 4 and "\\cite{" in result.latex


def test_the_best_version_is_kept_when_repairs_run_out(tmp_path, caplog):
    def two_problems(keys: list[str]) -> str:
        document = filled([*keys, "Ghost2020"])
        return fenced(document.replace("Prior work drifts.", "Related work here."))

    def one_problem(keys: list[str]) -> str:
        return fenced(filled([*keys, "Ghost2020"]))

    writer = scripted_writer(two_problems, one_problem, two_problems)
    result, ctx, client, _ = run_agent(tmp_path, Script(writer=writer), max_repair_rounds=2)

    assert len(writer_requests(client)) == 3  # the first answer and two repair turns
    assert result.latex == filled([SURVEY, FOUNDATIONS, DRIFT, ADAM, "Ghost2020"])
    assert "Keeping the best Introduction/Related Work draft" in caplog.text
    ((kind, note),) = notes(ctx)
    assert kind == "lesson" and "Still unresolved after 2 repair turn(s)" in note
    assert "Ghost2020" in note


def test_a_complete_draft_is_kept_over_a_truncated_one_with_as_many_problems(tmp_path, caplog):
    def complete_with_two_problems(keys: list[str]) -> str:
        document = filled([*keys, "Ghost2020"])
        return fenced(document.replace("Prior work drifts.", "Related work here."))

    def truncated(keys: list[str]) -> str:  # also two problems: no \end, 'document' unbalanced
        return fenced(filled(keys).replace("\\end{document}\n", ""))

    writer = scripted_writer(complete_with_two_problems, truncated)
    result, _, client, _ = run_agent(tmp_path, Script(writer=writer), max_repair_rounds=1)

    assert len(writer_requests(client)) == 2
    # The later draft is cut off and cannot be used; the earlier one can, so the stage goes on.
    assert "\\end{document}" in result.latex and "Ghost2020" in result.latex
    assert "Keeping the best Introduction/Related Work draft" in caplog.text


def test_a_document_that_stays_incomplete_fails_the_stage(tmp_path):
    def truncated(keys: list[str]) -> str:
        return fenced(filled(keys).replace("\\end{document}\n", ""))

    client = FakeLLM(Script(writer=scripted_writer(truncated)))
    ctx = make_ctx(tmp_path, client, max_repair_rounds=1)
    with pytest.raises(RuntimeError, match=r"still incomplete after 1 repair turn"):
        LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    assert len(writer_requests(client)) == 2
    # Nothing was put in place for later stages.
    assert not ctx.paths.literature_dir.exists()
    assert ctx.paths.template_tex.read_text() == TEMPLATE
    assert not ctx.paths.references_bib.exists() and not ctx.paths.outline.exists()
    assert ctx.harness.project.list("wiki/literature") == []


def test_no_usable_reply_at_all_fails_the_stage(tmp_path):
    client = FakeLLM(Script(writer=lambda request: "No."))
    ctx = make_ctx(tmp_path, client, max_repair_rounds=0)
    with pytest.raises(RuntimeError, match="returned no LaTeX document"):
        LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    assert len(writer_requests(client)) == 1


def test_too_few_citations_earn_exactly_one_request_for_more(tmp_path):
    writer = scripted_writer(lambda keys: fenced(filled(keys[:1])))
    result, ctx, client, _ = run_agent(tmp_path, Script(writer=writer))
    _, second = writer_requests(client)
    assert "only 1 of the 4 collected papers are cited" in second.text
    assert "at least 3" in second.text
    assert f"Not cited yet: {FOUNDATIONS}, {DRIFT}, {ADAM}" in second.text
    assert cite_keys(result.latex) == {SURVEY}  # accepted: usable, just thinly cited
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert event["cited"] == 1


def test_check_document_separates_blocking_from_advisory_problems():
    keys = ["A", "B", "C"]
    blocking, advisory = check_document(filled(keys), template=TEMPLATE, citation_keys=keys)
    assert (blocking, advisory) == ([], [])

    commented = filled(keys).replace("\\maketitle", "\\maketitle % \\cite{NotReal}")
    assert check_document(commented, template=TEMPLATE, citation_keys=keys) == ([], [])

    untouched_preamble = filled(keys).replace("  % venue style", "")
    assert check_document(untouched_preamble, template=TEMPLATE, citation_keys=keys) == ([], [])

    blocking, advisory = check_document(filled(["A", "Z"]), template=TEMPLATE, citation_keys=keys)
    assert len(blocking) == 1 and "Z" in blocking[0]
    assert len(advisory) == 1 and "Not cited yet: B, C" in advisory[0]

    blocking, _ = check_document(TEMPLATE, template=TEMPLATE, citation_keys=[])
    assert len(blocking) == 2 and all("placeholder" in problem for problem in blocking)


def test_a_single_collected_paper_is_asked_for_and_checked_as_one_not_zero():
    assert [min_cited(count) for count in (0, 1, 2, 4, 10, 80)] == [0, 1, 1, 3, 9, 72]
    prompt = writer_user_prompt(
        template=TEMPLATE,
        plan={},
        idea="",
        experimental_log="",
        guidelines="",
        papers=[{"citation_key": "A"}],
        citation_keys=["A"],
        cutoff="2024-10",
    )
    assert "abstracts of 1 collected paper. Cite at least 1 of them" in prompt
    _, advisory = check_document(filled([]), template=TEMPLATE, citation_keys=["A"])
    assert len(advisory) == 1 and "should cite at least 1. Not cited yet: A" in advisory[0]
    assert check_document(filled(["A"]), template=TEMPLATE, citation_keys=["A"]) == ([], [])


# -- nothing verified --------------------------------------------------------


def test_no_verified_papers_is_a_warning_not_a_crash(tmp_path, caplog):
    script = Script(
        {
            "widget surveys": [{"title": "Totally Invented Paper Title", "year": 2020}],
            "widget drift failure": [],
            ADAM_HINT: [{"title": "Future Widgets", "year": 2025, "reason": "too new"}],
        }
    )
    result, ctx, client, _ = run_agent(tmp_path, script)

    assert result.papers == 0 and result.citation_map == {} and result.bibtex == ""
    assert ctx.paths.references_bib.read_text() == ""
    assert json.loads(ctx.paths.citation_map.read_text()) == {}
    assert all_candidate_keys(result.outline) == set()
    assert "citation_hints" not in result.outline["section_plan"][0]["subsections"][0]
    assert "\\cite" not in result.latex and "Widgets matter." in result.latex

    (request,) = writer_requests(client)
    assert "No verified papers could be collected" in request.text
    assert "<citation_checklist>\n(empty)\n</citation_checklist>" in request.text
    assert "Cite at least" not in request.text

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("No verified papers were collected" in message for message in warnings)
    ((kind, note),) = notes(ctx)
    assert kind == "observation" and "references.bib is empty" in note
    assert "3 search task(s) (0 failed, 1 without candidates), 2 candidate(s)" in note
    assert ctx.harness.project.list("wiki/literature") == []


def test_citations_are_rejected_when_the_bibliography_is_empty(tmp_path):
    empty = {task: [] for task in DISCOVERY}
    writer = scripted_writer(
        lambda keys: fenced(filled(["Vaswani2017Attention"])), lambda keys: fenced(filled([]))
    )
    result, _, client, _ = run_agent(tmp_path, Script(empty, writer))
    assert len(writer_requests(client)) == 2
    assert "Vaswani2017Attention" in writer_requests(client)[1].text
    assert "\\cite" not in result.latex


def test_an_outline_without_search_tasks_still_gets_its_sections_written(tmp_path):
    outline = make_outline()
    outline["intro_related_work_plan"]["introduction_strategy"]["search_directions"] = []
    outline["intro_related_work_plan"]["related_work_strategy"]["subsections"] = []
    outline["section_plan"][0]["subsections"][0]["citation_hints"] = []
    result, _, client, session = run_agent(tmp_path, outline=outline)
    assert result.papers == 0 and session.queries == []
    assert discovery_requests(client) == [] and len(writer_requests(client)) == 1


# -- failures ----------------------------------------------------------------


def test_a_failed_search_task_is_logged_counted_and_survived(tmp_path, caplog):
    script = Script(DISCOVERY | {"widget drift failure": LLMError("overloaded (req_123)")})
    result, ctx, _, _ = run_agent(tmp_path, script)

    assert list(result.citation_map) == [SURVEY, FOUNDATIONS, ADAM]
    assert "Search task 2/3 failed [Related Work: 2.1 Widget Models]" in caplog.text
    assert "overloaded (req_123)" in caplog.text
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert (event["tasks"], event["failed_tasks"]) == (3, 1)
    ((kind, note),) = notes(ctx)
    assert kind == "observation" and "1 of 3 search task(s) failed" in note


def test_an_answer_that_never_parses_fails_only_its_task(tmp_path):
    script = Script(DISCOVERY | {"widget surveys": "I found many papers but will not list them."})
    result, ctx, client, _ = run_agent(tmp_path, script)
    assert list(result.citation_map) == [DRIFT, SURVEY, ADAM]
    # Asked three times (the harness re-asks unparseable answers) before the task is given up.
    assert sum("<search_task>widget surveys<" in r.text for r in discovery_requests(client)) == 3
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert event["failed_tasks"] == 1


def test_the_stage_fails_when_every_search_task_fails(tmp_path):
    script = Script(dict.fromkeys(DISCOVERY, LLMError("no credentials")))
    client = FakeLLM(script)
    ctx = make_ctx(tmp_path, client)
    with pytest.raises(RuntimeError, match="all 3 literature search tasks failed") as failure:
        LiteratureAgent(ctx, scholar=make_scholar()).run(make_outline())
    assert isinstance(failure.value.__cause__, LLMError)
    assert writer_requests(client) == [] and not ctx.paths.literature_dir.exists()


def test_a_failed_lookup_is_logged_counted_and_survived(tmp_path, caplog):
    session = FakeS2Session(failing=("Foundations",))
    result, ctx, _, _ = run_agent(tmp_path, session=session)

    assert list(result.citation_map) == [SURVEY, DRIFT, ADAM]
    assert session.queries.count("Widget Foundations") == 2  # retried once, as configured
    assert "Semantic Scholar lookup failed for 'Widget Foundations'" in caplog.text
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert (event["lookup_errors"], event["skipped_lookups"]) == (1, 0)
    ((kind, note),) = notes(ctx)
    assert kind == "observation" and "1 Semantic Scholar lookup(s) failed" in note


def test_semantic_scholar_being_down_stops_the_lookups_but_not_the_stage(tmp_path, caplog):
    many = [{"title": f"Candidate Number {i}", "year": 2020, "reason": ""} for i in range(12)]
    script = Script({"widget surveys": many, "widget drift failure": [], ADAM_HINT: many[:1]})
    session = FakeS2Session(failing=("Candidate",))
    result, ctx, _, _ = run_agent(tmp_path, script, session=session)

    assert result.papers == 0 and result.bibtex == ""
    assert len(session.queries) == 5 * 2  # five titles, each tried twice, then no more
    assert "Semantic Scholar failed 5 times in a row" in caplog.text
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert (event["lookup_errors"], event["skipped_lookups"]) == (5, 8)
    kinds = sorted(kind for kind, _ in notes(ctx))
    assert kinds == ["observation", "observation"]  # incomplete search + empty bibliography


def test_the_cache_still_answers_while_semantic_scholar_is_down(tmp_path):
    run_agent(tmp_path, name="first")  # fills the cache

    unknown = [{"title": f"Candidate Number {i}", "year": 2020, "reason": ""} for i in range(6)]
    script = Script(
        {
            "widget surveys": [*unknown, DISCOVERY["widget surveys"][0]],
            "widget drift failure": DISCOVERY["widget drift failure"],
            ADAM_HINT: DISCOVERY[ADAM_HINT],
        }
    )
    down = FakeS2Session(failing=("Candidate", "Widget", "Adam"))
    result, ctx, _, _ = run_agent(tmp_path, script, name="second", session=down)

    assert list(result.citation_map) == [SURVEY, DRIFT, ADAM]  # all three from the cache
    assert len(down.queries) == 5 * 2  # the outage was noticed after five titles
    (event,) = [e for e in ctx.harness.transcripts.events() if e["type"] == "literature"]
    assert (event["lookup_errors"], event["skipped_lookups"], event["cache_hits"]) == (5, 1, 3)


def test_scholar_errors_are_the_only_lookup_failures_that_are_tolerated(tmp_path):
    class BrokenScholar:
        def find_paper(self, title, *, year_hint=None, cutoff=None):
            raise KeyError("a bug, not an outage")

    ctx = make_ctx(tmp_path, FakeLLM(Script()))
    with pytest.raises(KeyError):
        LiteratureAgent(ctx, scholar=BrokenScholar()).run(make_outline())

    class DownScholar:
        def find_paper(self, title, *, year_hint=None, cutoff=None):
            raise SemanticScholarError("down")

    ctx = make_ctx(tmp_path, FakeLLM(Script()), name="down")
    assert LiteratureAgent(ctx, scholar=DownScholar()).run(make_outline()).papers == 0


def test_the_input_outline_may_be_reused_after_a_failure(tmp_path):
    outline = make_outline()
    before = copy.deepcopy(outline)
    ctx = make_ctx(tmp_path, FakeLLM(Script(writer=lambda request: "No.")), max_repair_rounds=0)
    with pytest.raises(RuntimeError):
        LiteratureAgent(ctx, scholar=make_scholar()).run(outline)
    assert outline == before
