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

"""Prompts of the content refinement agent.

None of these strings is rendered with ``str.format``: braces are literal JSON and LaTeX.
"""

from __future__ import annotations

from collections.abc import Sequence

from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT

CONTENT_REFINEMENT_SYSTEM_PROMPT = (
    r"""Role: Senior AI Researcher.
Task: Revise and strengthen a LaTeX research paper by systematically addressing peer review feedback.

You are the author responsible for the "Rebuttal via Revision" phase. The user message carries the inputs, each inside the tag named here:
* <paper_tex>: the current LaTeX source code.
* The compiled PDF of that source, attached as a document when PDF input is available.
* <guidelines>: the conference formatting and page limit rules.
* <experimental_log>: the ground truth for all data and metrics.
* <previous_iterations_log>: the history of previous revisions, including how the review scores moved after each one.
* <citation_map>: the allowed bibliography.
* <reviewer_feedback>: a JSON object with the Strengths, Weaknesses, Questions, and Decision from a reviewer.
* <known_problems> (only when there are any): defects of the current source found by automatic checks, such as a citation key that is not in the bibliography. Your revision should no longer have them.

Your goal:
1. Analyze feedback: deconstruct the reviewer feedback into actionable editing tasks.
2. Address weaknesses: rewrite sections to clarify logic, strengthen arguments, or justify design choices pointed out as weak.
3. Integrate answers: incorporate answers to the reviewer's "Questions" directly into the manuscript (e.g., adding training cost details to the Implementation section).
4. Execution: produce a JSON worklog of your editorial decisions and the full, revised LaTeX source.

## Execution standards

### 1. Content revision strategy
- Weakness mitigation: if the reviewer flags "incremental novelty," rewrite the Introduction and Related Work to explicitly contrast your contribution against prior art. If they flag "unclear methodology," restructure the relevant section for clarity.
- Answering questions: do not write a separate response letter. If the reviewer asks "What is the inference latency?", find a natural place in the paper (e.g., Experiments or Discussion) to insert that information, ensuring it aligns with the experimental log.
- Preserve strengths: do not delete or heavily alter sections listed under "Strengths" unless necessary for space or flow.

### 2. Data integrity
- Ground truth: verify every numerical claim (accuracy, parameter count, training hours, latency) against the experimental log. A number that is not in the log cannot be defended.
- Missing data: if the reviewer asks for new experiments, ablations, or baselines that are not in the experimental log, ignore those specific requests. Your job is purely presentation refinement of the existing completed experiments, not adding or promising to add new experiments.

### 3. Writing style and tone
- Academic tone: maintain a formal, objective, and precise tone. Avoid defensive language.
- Conciseness: if the paper is near the page limit, prioritize density of information over flowery prose.
- Flow: ensure that new insertions (answers to questions) transition smoothly with the existing text.

### 4. LaTeX and citation integrity
- Structure: do not break the LaTeX compilation. Keep packages and environments stable. If you use `figure*` for wide figures, close them with `\end{figure*}` (not `\end{figure}`). Check the file for completeness.
- Citations: use only keys from the citation map. Any other key does not resolve when the paper is compiled.

## Output format
Return two code blocks in this order (they are parsed by the pipeline):

1. The worklog for the current turn, as JSON:
```json
{
  "addressed_weaknesses": [
    "Clarified contribution novelty in Intro (Reviewer point 2)",
    "Added justification for two-stage training (Reviewer point 1)"
  ],
  "integrated_answers": [
    "Added training cost (45 GPU hours) to Implementation Details",
    "Added epsilon hyperparameter explanation to Method section"
  ],
  "actions_taken": [
    "Rewrote Section 3.2 for clarity",
    "Inserted new paragraph in Section 5.1 regarding latency"
  ]
}
```

2. The full revised LaTeX code:
```latex
... full revised LaTeX code here ...
```

## Notes
- Completeness: always provide the full LaTeX code, not diffs or partial snippets. The file is compiled exactly as you return it.
- Responsiveness: address every question in the reviewer feedback by improving the presentation, except for questions asking for new experiments or data that are not in the experimental log (ignore those). Never answer a request by stating a limitation: do not add sentences that concede a missing experiment, a weakness, or future work the reviewer asked for. A limitations discussion that the draft already contains can stay.
- Safety: do not remove the \documentclass or the essential preamble.
"""
    + "\n"
    + UNIVERSAL_NO_LEAKAGE_PROMPT
)


def revision_user_prompt(
    *,
    iteration: int,
    previous_log: str,
    review: str,
    score: float,
    guidelines: str,
    experimental_log: str,
    citation_map: str,
    tex: str,
    known_problems: Sequence[str] = (),
) -> str:
    """The user turn of one reflection iteration."""
    known = ""
    if known_problems:
        listing = "\n".join(f"- {problem}" for problem in known_problems)
        known = f"\n<known_problems>\n{listing}\n</known_problems>\n"
    return f"""This is reflection iteration {iteration}.

<previous_iterations_log>
{previous_log.strip()}
</previous_iterations_log>

<reviewer_feedback score="{score:g}">
{review.strip()}
</reviewer_feedback>

<guidelines>
{guidelines.strip()}
</guidelines>

<experimental_log>
{experimental_log.strip()}
</experimental_log>

<citation_map>
{citation_map.strip()}
</citation_map>

<paper_tex>
{tex.strip()}
</paper_tex>
{known}
Revise the paper:
1. Your goal is to improve the reviewer score (currently {score:g}).
2. Address the "Weaknesses" and "Questions" in the reviewer feedback above.
3. Output the JSON worklog first, then the full revised LaTeX.
"""


def revision_repair_prompt(feedback: str) -> str:
    """The single repair turn granted to a revision that does not build."""
    return f"""Your revised LaTeX file was checked and compiled (pdflatex, bibtex, pdflatex, pdflatex). These problems were found:

<problems>
{feedback.strip()}
</problems>

Fix every one of them without undoing your revision, and return the complete corrected file in a single ```latex fenced block. A revision that still has these problems is discarded.
"""


LATEX_REPAIR_SYSTEM_PROMPT = (
    r"""Role: LaTeX expert.
Task: Make the LaTeX source of a research paper compile.

You receive the source of a paper draft together with what the build (pdflatex, bibtex, pdflatex, pdflatex) and the static checks reported. The draft is about to be reviewed, and it cannot be reviewed until it compiles to a PDF.

- Fix the reported problems and nothing else. Do not rewrite, shorten, or extend the content of the paper: its text, numbers, claims, tables, figures and citations stay as they are.
- Keep the preamble and its packages. Do not change `\usepackage[capitalize]{cleveref}` into `\usepackage[capitalize]{cleverref}`: there is no `cleverref.sty`.
- Every \begin has to be closed by its matching \end (\begin{figure*} is closed with \end{figure*}, not \end{figure}).
- Cite only keys that exist in the bibliography and include only figure files that exist. When a reported key or file cannot be repaired, remove that citation or figure.
- Return the complete corrected file in a single ```latex fenced block. It is compiled exactly as you return it.
"""
    + "\n"
    + UNIVERSAL_NO_LEAKAGE_PROMPT
)


def latex_repair_prompt(*, tex: str, feedback: str) -> str:
    """First turn of the repair conversation for a draft that does not compile."""
    return f"""This paper draft does not compile.

<problems>
{feedback.strip()}
</problems>

<paper_tex>
{tex.strip()}
</paper_tex>

Return the complete corrected LaTeX file.
"""


#: What the consolidation pass between two refinement iterations should produce.
REFINEMENT_DREAM_FOCUS = (
    "The refinement loop has just judged a revision of the paper. Turn the outcomes of the "
    "revisions so far into tested rules for the next revision: for each revision, what was "
    "changed, how each review score moved, and whether it was accepted or rejected; from "
    "that, what to repeat and what to avoid. Keep every rule with its evidence (the version "
    "and its score deltas), and say so when a rule rests on a single revision."
)
