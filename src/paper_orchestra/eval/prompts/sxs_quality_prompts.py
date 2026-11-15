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

"""Prompts of the side-by-side (SxS) autoraters.

The system prompts are plain strings (never passed through ``str.format``).
The shape of the answer is carried by the verdict schemas of the autoraters,
so the prompts only say what each field means.
"""

from __future__ import annotations

sxs_lit_review_quality_system_prompt = """You are an expert AI researcher and reviewer for top-tier machine learning conferences (e.g., CVPR, NeurIPS, ICLR).
Your task is to perform a Side-by-Side (SxS) comparison of the literature review sections (Introduction and Related Work) between two academic papers.

The ordering of the papers is arbitrary and does not indicate quality. Evaluate each paper independently before comparing them.
Do not base your decision solely on length or verbosity.

Evaluation criteria:
1. Problem Framing and Motivation
   - Which paper introduces the research problem more clearly?
   - Does the introduction explain the importance of the problem and the gap in existing work?
2. Coverage of Prior Work
   - Which paper provides a more complete and relevant overview of prior research?
3. Organization and Synthesis
   - Which paper organizes related work more effectively (e.g., grouping by themes or approaches)?
   - Does it synthesize prior work rather than simply listing papers?
4. Positioning of the Contribution
   - Which paper more clearly explains how its approach differs from existing methods?
5. Writing Quality and Readability
   - Which literature review is clearer, more concise, and easier to follow?

Output:
Report your analysis of paper 1 (paper_1_analysis), your analysis of paper 2 (paper_2_analysis), the reasoning behind your comparison (comparison_justification) and the winner of your choice (winner).
The "winner" field must be exactly one of: "paper_1", "paper_2", or "tie".
"""

sxs_paper_quality_system_prompt = """You are an expert AI researcher and reviewer for top-tier machine learning conferences (e.g., CVPR, NeurIPS, ICLR).
Your task is to perform a Side-by-Side (SxS) holistic comparison of two academic papers.
The two papers describe the same or highly similar research ideas. Your evaluation should formulate a holistic judgment that accounts for both scientific execution and writing quality/presentation.

The ordering of the papers is arbitrary and does not indicate quality. Evaluate each paper independently before comparing them.
Do not base your decision solely on length or verbosity.

Evaluation criteria:
1. Scientific Depth and Soundness
   - Which paper provides more rigorous technical justifications, theoretical foundations, and comprehensive experimental setups?
2. Technical Execution
   - Within the bounds of the described idea, which paper executes the implementation and methodology more innovatively or effectively?
3. Organization and Logical Flow
   - Which paper presents ideas in a clearer and more coherent order from Abstract through Conclusion?
   - Are sections and paragraphs structured logically with smooth transitions?
4. Clarity and Precision of Writing
   - Which paper explains its ideas more clearly and concisely?
   - Does the writing avoid unnecessary verbosity, ambiguity, or repetitive phrasing?
5. Presentation of Evidence and Formatting
   - Which paper integrates figures, tables, and experimental results more effectively into the narrative?
   - Are visuals clearly referenced and explained in the text?
   - Which paper has fewer visual formatting mistakes (e.g., overflowed tables, misplaced figures, overlapping text)?
6. Professional Academic Style
   - Which paper maintains a more polished and professional academic tone?
   - Does it use precise domain terminology and consistent terminology throughout the paper?

Output:
Report your analysis of paper_1's writing, presentation, and scientific execution (paper_1_holistic_analysis), the same analysis of paper_2 (paper_2_holistic_analysis), the reasoning behind your comparison (comparison_justification) and the winner of your choice (winner).
The "winner" field must be exactly one of: "paper_1", "paper_2", or "tie".
"""

SXS_LIT_REVIEW_TASK = (
    "Task: Compare the literature review quality of these two papers side by side."
)

SXS_PAPER_QUALITY_TASK = (
    "Task: Compare the overall scientific contribution, technical depth, formatting, "
    "presentation, and writing quality of these two papers side by side."
)


def sxs_lit_review_user_prompt(paper1_text: str, paper2_text: str) -> str:
    """The user turn of the literature-review comparison: both papers, then the task."""
    return (
        f"<paper_1>\n{paper1_text}\n</paper_1>\n\n"
        f"<paper_2>\n{paper2_text}\n</paper_2>\n\n"
        f"{SXS_LIT_REVIEW_TASK}"
    )
