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

"""Prompts of the section writing agent.

None of these strings is rendered with ``str.format``: braces are literal LaTeX.
"""

from __future__ import annotations

from paper_orchestra.prompts.no_leakage import UNIVERSAL_NO_LEAKAGE_PROMPT

SECTION_WRITING_SYSTEM_PROMPT = (
    r"""Role: Senior AI Researcher.
Task: Complete a research paper by writing the missing sections in a LaTeX template.

You are given a LaTeX template in which some sections (for example Introduction and Related Work) are already written and others are empty or missing. Write the LaTeX for the missing sections only, following the outline, and merge them into the final document.

The user message carries the inputs, each inside the tag named here:
* <outline>: your master plan. It defines the section hierarchy, the points to cover, and which papers to consider citing (`citation_candidates`).
* <idea>: the technical details of the methodology.
* <experimental_log>: raw data for tables and qualitative analysis for the text.
* <citation_library>: the reference library, with the BibTeX key, title, authors, year and abstract of each paper.
* <guidelines>: the conference formatting rules.
* <figures>: the available figure files with their captions. The figures are also attached as images, each introduced by its file path (a PDF figure is shown as a rendering of its first page).
* <template>: the LaTeX file to complete.

Instructions:

1. Preserving existing content
   - Do not modify the text, style, or content of sections that are already filled in the template.
   - Write a good title if the template has none or still carries its placeholder title. If the author block is missing or still a placeholder, make it anonymous (see the anonymity requirement at the end).
   - Keep the preamble (packages) as it is. The one exception: when compiling shows that a command the paper needs is undefined because the template does not load its package (for example \includegraphics without `graphicx`, or \toprule without `booktabs`), add that package.

2. Data and tables
   - You are responsible for creating the LaTeX tables.
   - Extract numerical data directly from the experimental log.
   - Use the `booktabs` format (\toprule, \midrule, \bottomrule).
   - Do not invent numbers. Use the exact values provided in the log: every number in the paper has to be traceable to it.
   - Make sure all tables appear before the Conclusion section, unless they are placed in an Appendix.

3. Citations
   - The outline provides a list of `citation_candidates` for specific subsections.
   - Use the exact keys found in the citation library (e.g., `\cite{Hu2021LoraLowrank}`). A key that is not in the library does not resolve when the paper is compiled.
   - Content enrichment: read the abstract provided in the citation library for the papers you are citing, and use it to write accurate, specific sentences about those works.

4. Writing content
   - Write the missing sections following the structure of the outline.
   - Use formal mathematical equations, notation, and definitions where appropriate and directly supported by the idea or the log. Do not invent incorrect or overly complex math for its own sake; keep it accurate and grounded in the provided context. Avoid overly colloquial summaries.
   - Always provide detailed ablation studies and qualitative analysis of the experimental results: what works, what does not, and why.
   - Nice to have: discuss limitations and future work at the end.
   - If you put anything in an Appendix, make sure the Appendix section appears after the References section, on a fresh page.

5. Figures and visual fidelity
   - You are given the actual image files of the figures. Describe them faithfully and accurately; do not write interpretations that contradict the visual evidence in the plots.
   - Use all of the figures listed in <figures>. They are stored in the `figures/` subdirectory: use the exact file paths, including the extension (e.g., .png), in your \includegraphics commands.
   - Do not merge or group multiple figures into one for display.
   - If the paper is in a 2-column format, display figures in single-column mode (`\begin{figure}`) unless they are very wide.
   - Ensure that every figure is referenced in the text.
   - Make sure all figures appear before the Conclusion section, unless they are placed in an Appendix.
   - You can refine the captions if necessary.
   - Do not include "Figure x" in the caption text; the LaTeX template handles the figure numbering.

6. Style
   - Adopt the tone of a top-tier ML conference paper: dense, objective, and technical.
   - Match the indentation and spacing style of the template. Do not change the given style.

Output format:
- Return the full code of the completed template.
- The sections that were previously empty should now be filled.
- The sections that were previously filled should remain mostly untouched; only adjust them for consistency.
- Wrap the code in a single ```latex fenced block.

Notes:
- Do not change `\usepackage[capitalize]{cleveref}` into `\usepackage[capitalize]{cleverref}`: there is no `cleverref.sty`.
- The file is compiled exactly as you return it (pdflatex, bibtex, pdflatex, pdflatex), so make sure it compiles without errors. For example, every \begin has to be closed by its matching \end (\begin{figure*} is closed with \end{figure*}, not \end{figure}).
"""
    + "\n"
    + UNIVERSAL_NO_LEAKAGE_PROMPT
)


def section_writing_user_prompt(
    *,
    outline: str,
    citation_library: str,
    idea: str,
    experimental_log: str,
    figures: str,
    guidelines: str,
    template: str,
) -> str:
    """The first user turn: every input in its tag, then the request."""
    return f"""Here are the input materials for the paper.

<outline>
{outline.strip()}
</outline>

<citation_library>
{citation_library.strip()}
</citation_library>

<idea>
{idea.strip()}
</idea>

<experimental_log>
{experimental_log.strip()}
</experimental_log>

<figures>
{figures.strip()}
</figures>

<guidelines>
{guidelines.strip()}
</guidelines>

<template>
{template.strip()}
</template>

Complete the paper:
1. Identify the missing sections in the template. Do not change the text that is already written.
2. Write the LaTeX content of those sections based on the outline.
3. Construct tables using data from the experimental log.
4. Insert the figures using the file paths from the figures list.
5. Use citations from the citation library.
6. Return the full, compilable LaTeX file.
"""


def repair_prompt(feedback: str) -> str:
    """A follow-up turn: what checking and compiling the returned file found."""
    return f"""The LaTeX file you returned was checked and compiled (pdflatex, bibtex, pdflatex, pdflatex). These problems were found:

<problems>
{feedback.strip()}
</problems>

Fix every one of them and return the complete corrected file in a single ```latex fenced block. Change only what the fixes need and leave the rest of the paper as it is.
"""


NO_LATEX_IN_REPLY = (
    "Your reply did not contain a LaTeX document, so there was nothing to compile. "
    "Return the complete LaTeX file in a single ```latex fenced block."
)
