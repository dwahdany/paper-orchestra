# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompts of the literature agent: candidate discovery and Introduction / Related Work writing.

Nothing here is rendered with ``str.format``.  System prompts are constants and
user messages are assembled by the builder functions, so LaTeX braces appear
exactly as the model should read them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT

# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

DISCOVERY_SYSTEM_PROMPT = """You help a research team build the bibliography of a paper they are writing by finding real, published academic papers.

Every candidate you return is looked up in a bibliographic database by its title afterwards, and a candidate that cannot be matched there is discarded. The exact official title and the correct publication year are therefore worth more than a long list: leave out any paper you are not sure exists under that title.

For each candidate give the exact title, the year of publication, and one sentence on what the paper contributes and why it fits the task."""

_SEARCH_FIRST = (
    "Use web search to find the papers. The exact titles and years are the thing to verify, "
    "so search before answering instead of relying on memory."
)
_NO_SEARCH = (
    "Web search is not available in this session, so include only papers whose exact title "
    "and year you are certain of."
)


def _cutoff_line(cutoff: str) -> str:
    if not cutoff.strip():
        return ""
    return (
        f"<research_cutoff>{cutoff.strip()}</research_cutoff>\n"
        "Only papers published before the research cutoff month are in scope.\n\n"
    )


def targeted_discovery_prompt(*, hint: str, context: str, cutoff: str, web_search: bool) -> str:
    """Resolve one citation hint from the section plan to the paper it names."""
    return (
        "Identify the specific academic paper described by this hint.\n\n"
        f"<hint>{hint}</hint>\n"
        f"<context>{context}</context>\n"
        f"{_cutoff_line(cutoff)}"
        f"- {_SEARCH_FIRST if web_search else _NO_SEARCH}\n"
        "- If the hint refers to a dataset, model architecture or baseline (for example "
        '"research paper introducing ..."), find the primary, original paper that introduced it.\n'
        "- Return exactly one candidate, the paper that matches best. If you cannot identify "
        "it with confidence, return no candidate rather than a guess."
    )


def exploration_discovery_prompt(
    *, focus: str, context: str, core_problem: str, cutoff: str, web_search: bool
) -> str:
    """Find the literature for one search direction of the outline."""
    return (
        "Find 10-15 highly relevant, influential academic papers matching the search task. "
        "Do not force results: return only papers that are strictly relevant, even if that "
        "is fewer.\n\n"
        f"<search_task>{focus}</search_task>\n"
        f"<task_context>{context}</task_context>\n"
        f"<project_problem>{core_problem}</project_problem>\n"
        f"{_cutoff_line(cutoff)}"
        f"- {_SEARCH_FIRST if web_search else _NO_SEARCH} Look for real papers on arXiv and "
        "at venues such as CVPR, NeurIPS, ICML and ICLR.\n"
        "- Return only papers published at top-tier conferences or journals, highly cited "
        "works, or the exact state of the art relevant to the task."
    )


# ---------------------------------------------------------------------------
# Introduction and Related Work
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = (
    r"""You are a senior AI researcher writing the Introduction and Related Work sections of a paper.

The user message contains template.tex, the skeleton of the paper. Fill in exactly two sections, Introduction and Related Work, and leave everything else untouched: later stages write the other sections, and the venue style depends on the preamble. The preamble (packages, styles, macros), the title block and every other section therefore stay identical to the original. For example, `\usepackage[capitalize]{cleveref}` is correct as it stands; there is no `cleverref` package.

Inputs in the user message:
* `<template>`: template.tex.
* `<intro_related_work_plan>`: your primary guide for structure and arguments.
* `<idea>` and `<experimental_log>`: use them so that the Introduction frames the technical contribution and the results accurately.
* `<guidelines>`: the venue's writing guidelines. Follow them where they apply to these two sections (for example anonymity and how to refer to prior work).
* `<collected_papers>`: the papers collected for citation, each with its citation key and abstract.
* `<citation_checklist>`: the citation keys you may use.
* `<research_cutoff>`: the month at which the literature search stopped.

Citation rules:
- Cite only the papers in `<collected_papers>`. They were verified against a bibliographic database and are the whole bibliography, so a citation to anything else cannot resolve.
- Every `\cite{key}` uses a key exactly as it is spelled in `<citation_checklist>`.
- Introduction: cite key statistics, foundational models (CLIP and the like) and broad problem statements.
- Related Work: make deep comparative citations and group distinct works (for example "Several methods [A, B, C] ...").
- Timeline rule: do not treat papers published after the research cutoff as prior baselines to beat. Treat them strictly as concurrent work.
- Evaluation rule: do not claim that our method beats, or achieves state of the art over, a specific cited paper unless that paper is explicitly evaluated against in the experimental log. Frame other recent papers strictly as concurrent, orthogonal or conceptual work.

Output: return the complete updated template.tex in a single ```latex fenced block, with the two sections filled in and everything else identical to the original.
"""
    + "\n"
    + UNIVERSAL_NO_LEAKAGE_PROMPT
)

#: Share of the collected papers the two sections are asked to cite.
MIN_CITED_FRACTION = 0.9


def writer_user_prompt(
    *,
    template: str,
    plan: Mapping[str, Any],
    idea: str,
    experimental_log: str,
    guidelines: str,
    papers: Sequence[Mapping[str, Any]],
    citation_keys: Sequence[str],
    cutoff: str,
) -> str:
    """The first user turn of the writing conversation: the inputs, then the request."""
    if citation_keys:
        count = len(citation_keys)
        requirement = (
            f"You have the abstracts of {count} collected papers. Cite at least "
            f"{int(count * MIN_CITED_FRACTION)} of them across the two sections."
        )
        checklist = "\n".join(citation_keys)
    else:
        requirement = (
            "No verified papers could be collected for this paper, so the bibliography is "
            r"empty. Write both sections without any \cite command and do not refer to "
            "specific prior publications."
        )
        checklist = "(empty)"
    return (
        f"<template>\n{template.strip()}\n</template>\n\n"
        "<intro_related_work_plan>\n"
        f"{json.dumps(plan, indent=1, ensure_ascii=False)}\n"
        "</intro_related_work_plan>\n\n"
        f"<idea>\n{idea.strip()}\n</idea>\n\n"
        f"<experimental_log>\n{experimental_log.strip()}\n</experimental_log>\n\n"
        f"<guidelines>\n{guidelines.strip()}\n</guidelines>\n\n"
        "<collected_papers>\n"
        f"{json.dumps(list(papers), indent=1, ensure_ascii=False)}\n"
        "</collected_papers>\n\n"
        f"<citation_checklist>\n{checklist}\n</citation_checklist>\n\n"
        f"<research_cutoff>{cutoff.strip() or 'none'}</research_cutoff>\n\n"
        "Write the Introduction and Related Work sections and return the complete updated "
        f"template.tex. {requirement}"
    )


def writer_repair_prompt(problems: Sequence[str]) -> str:
    """A follow-up turn listing what an automatic check found wrong with the last document."""
    listed = "\n".join(f"- {problem}" for problem in problems)
    return (
        "An automatic check of the document you returned found problems that have to be "
        "fixed before it can be used:\n"
        f"{listed}\n\n"
        "Return the complete corrected template.tex in a single ```latex fenced block. Keep "
        "everything that was not flagged as it is, and cite only keys from "
        "<citation_checklist>."
    )
