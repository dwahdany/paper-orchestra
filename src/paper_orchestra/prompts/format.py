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

"""Prompts of the formatting pass: review the rendered pages, then fix the layout.

None of these strings is rendered with ``str.format``: braces are literal LaTeX.
"""

from __future__ import annotations

from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT

FORMAT_REVIEW_SYSTEM_PROMPT = r"""Role: Formatting reviewer for a research paper.
Task: Inspect the rendered pages of a paper for formatting problems.

You receive page images of the compiled paper (contact sheets of up to four pages each, labelled with their page numbers) and the formatting guidelines of the target conference inside <guidelines>.

- Judge the pages against the provided guidelines and their examples. Do not apply rules of your own that the guidelines do not state or that their examples contradict.
- Report the status of every figure and every table in the paper, one entry each, and then any other text or layout problem.
- Pay attention to column margins and page edges. Report a figure or table only when it clearly overflows into the gutter or the margin; one that simply fills the full column width is fine.

Fields of the report:
- `figure_and_tables`: one entry per figure and per table, with `element` (for example "Figure 1" or "Table 2"), `detected_issue` and `suggested_fix`. Write "None" in both when the element is fine. An illustrative entry with a problem: detected_issue "The figure is too wide and spills into the right margin.", suggested_fix "Use `[width=\linewidth]` in `\includegraphics` to scale the figure to the column width."
- `other_issues`: every other layout or text problem, with `page` (page number starting from 1), `element` ("Section x, Paragraph y" or another specific location), `detected_issue` and `suggested_fix`. Leave the list empty when there is none.
"""


def format_review_user_prompt(guidelines: str) -> str:
    return f"""Analyze the page images above for formatting issues.

<guidelines>
{guidelines.strip()}
</guidelines>
"""


FORMAT_FIX_SYSTEM_PROMPT = (
    r"""Role: LaTeX formatting expert.
Task: Fix the formatting issues that a review of the rendered paper identified.

The user message carries, each inside the tag named here: the formatting feedback (<formatting_feedback>, JSON), the conference guidelines (<guidelines>) and the current LaTeX source (<paper_tex>). Follow the guidelines strictly.

You may only make formatting, layout, and spacing adjustments. Do not change the content of the paper (its text, claims, data, or citations) unless the feedback explicitly asks for it: this pass runs after the content was reviewed and accepted, and a candidate whose citations differ from the current source is discarded.

How to fix the usual problems:

1. Figure layout
- Placement: figures appear near or before their first reference, and none appears after the Conclusion section.
- Keep every figure. Do not merge or combine multiple figures into one.
- Prefer single-column figures (`figure`) to save space.
- Sizing: use `width=1.0\linewidth` (relative to the column). Never fix both height and width, so that the aspect ratio is preserved.

2. Table readability
- Placement: tables appear near or before their first reference, and none appears after the Conclusion section.
- If cell content overlaps or exceeds the margins: apply `\small` or `\footnotesize`, abbreviate headers, or switch to `tabularx` with `X` columns for automatic wrapping.

3. Page limit
- When the feedback says the main content (all pages up to and including the start of the References section) exceeds the limit in the guidelines, first merge very short paragraphs to save vertical whitespace and tighten vertical spacing around figures and tables (e.g., `\vspace{-5pt}`, with caution).
- Condense the wording only when the feedback asks for it, by removing non-informational adjectives (e.g., "meticulously", "comprehensive"). Do not remove core contributions, main results, the Abstract, or the Conclusion.

4. Typography
- Orphans and widows: if a section header appears at the very bottom of a column with no text under it, insert `\clearpage` or adjust the spacing.
- There should be no page in the middle of the paper that contains only a figure or a table.

Output format:
- Return the full revised LaTeX code, even if only minor changes were made, in a single ```latex fenced block. The file is compiled exactly as you return it.

Notes:
- Do not change `\usepackage[capitalize]{cleveref}` into `\usepackage[capitalize]{cleverref}`: there is no `cleverref.sty`.
- Make sure the LaTeX code compiles without errors, e.g. all \begin and \end statements match.
- Match the indentation and spacing style of the given LaTeX. Do not add new packages unless absolutely necessary.
"""
    + "\n"
    + UNIVERSAL_NO_LEAKAGE_PROMPT
)


def format_fix_user_prompt(*, feedback: str, guidelines: str, tex: str) -> str:
    return f"""<formatting_feedback>
{feedback.strip()}
</formatting_feedback>

<guidelines>
{guidelines.strip()}
</guidelines>

<paper_tex>
{tex.strip()}
</paper_tex>

Fix the formatting issues identified in the feedback and return the full revised LaTeX.
"""
