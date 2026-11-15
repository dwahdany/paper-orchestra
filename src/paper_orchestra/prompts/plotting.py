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

"""Prompts of the figure pipeline: retriever, planner, stylist, visualizer, critic, caption.

Ported from the PaperBanana prompts.  Role instructions are system prompts; the
inputs of a call are rendered into the user message inside XML-style tags by
the ``*_request`` builders.  The source material of the paper keeps its
``<idea>`` / ``<experimental_log>`` tags and is wrapped in the tag that names
its role for the task: ``<methodology_section>`` for diagrams, ``<raw_data>``
for plots.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

TaskName = Literal["plot", "diagram"]

_MATERIAL_TAG: dict[str, str] = {"plot": "raw_data", "diagram": "methodology_section"}
_INTENT_TAG: dict[str, str] = {"plot": "visual_intent", "diagram": "diagram_caption"}


def _tag(name: str, body: str) -> str:
    return f"<{name}>\n{body.strip()}\n</{name}>"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " [truncated]"


def _example_fields(item: Mapping[str, Any]) -> tuple[str, str]:
    """``(content, intent)`` of a PaperBanana reference item, as text."""
    content = item.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    intent = item.get("visual_intent", item.get("caption", ""))
    return content, str(intent)


# -- retriever -------------------------------------------------------------------

DIAGRAM_RETRIEVER_SYSTEM = """\
# Background and goal
We are building an AI system that automatically generates method diagrams for academic papers. Given a paper's methodology section and a figure caption, the system creates a high-quality illustrative diagram that visualizes the described method.

To help the system learn what kind of diagram to create, we use few-shot learning: it is shown reference examples of similar diagrams and learns from them what to draw for the target.

# Your task
You are the retrieval agent. Select the most relevant reference diagrams from a candidate pool; they will serve as few-shot examples for the diagram generation model.

The user message contains:
- <candidate_pool>: existing diagrams, each with an id, a caption and a methodology section (long sections are truncated);
- <target>: the methodology section and the caption of the diagram we need to generate.

Select the 10 candidates that would be most helpful as examples for teaching the system how to draw the target diagram.

# Selection logic (topic + intent)
Find examples that match the target in both domain and diagram type.

1. Match the research topic (use methodology and caption).
   - What is the domain? (for example Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications)
   - Select candidates that belong to the same research domain, because similar domains share similar terminology (for example "Actor-Critic" in RL).

2. Match the visual intent (use caption and keywords).
   - What type of diagram is implied? (for example "Framework", "Pipeline", "Detailed Module", "Performance Chart")
   - Select candidates with similar visual structures, because a "Framework" diagram is useless as an example for drawing a "Performance Bar Chart", even in the same domain.

Ranking priority:
1. Best match: same topic and same visual intent (target "Agent Framework" -> candidate "Agent Framework"; target "Dataset Construction Pipeline" -> candidate "Dataset Construction Pipeline").
2. Second best: same visual intent (target "Agent Framework" -> candidate "Vision Framework"). Structure is more important than topic for drawing.
3. Avoid: different visual intent (target "Pipeline" -> candidate "Bar Chart").

# Output
Return the ids of the selected diagrams, best match first, copied exactly from the candidate pool (such as "ref_1").
"""

PLOT_RETRIEVER_SYSTEM = """\
# Background and goal
We are building an AI system that automatically generates statistical plots. Given a plot's raw data and the visual intent, the system creates a high-quality visualization that effectively presents the data.

To help the system learn what kind of plot to create, we use few-shot learning: it is shown reference examples of similar plots and learns from them what to create for the target data.

# Your task
You are the retrieval agent. Select the most relevant reference plots from a candidate pool; they will serve as few-shot examples for the plot generation model.

The user message contains:
- <candidate_pool>: reference plots, each with an id, a visual intent and raw data (long data is truncated);
- <target>: the raw data and the visual intent of the plot we need to generate.

Select the 10 candidates that would be most helpful as examples for teaching the system how to create the target plot.

# Selection logic (data type + visual intent)
Find examples that match the target in both data characteristics and plot type.

1. Match the data characteristics (use raw data and visual intent).
   - What type of data is it? (for example categorical vs numerical, single series vs multi-series, temporal vs comparative)
   - What are the data dimensions? (for example 1D, 2D, 3D)
   - Select candidates with similar data structures and characteristics, because different data types require different visualization approaches.

2. Match the visual intent (use the visual intent).
   - What type of plot is implied? (for example "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart")
   - Select candidates with similar plot types, because a "bar chart" example is more useful for generating another bar chart than a "scatter plot" example.

Ranking priority:
1. Best match: same data type and same plot type (target "multi-series line chart" -> candidate "multi-series line chart").
2. Second best: same plot type with compatible data.
3. Avoid: different plot type.

# Output
Return the ids of the selected plots, best match first, copied exactly from the candidate pool (such as "ref_0").
"""


def retriever_system(task: TaskName) -> str:
    return PLOT_RETRIEVER_SYSTEM if task == "plot" else DIAGRAM_RETRIEVER_SYSTEM


def retrieval_request(
    task: TaskName,
    *,
    material: str,
    intent: str,
    candidates: Sequence[Mapping[str, Any]],
    max_content_chars: int,
) -> str:
    """The candidate pool first (it is the same for every figure of a task), then the target."""
    material_tag, intent_tag = _MATERIAL_TAG[task], _INTENT_TAG[task]
    pool: list[str] = []
    for item in candidates:
        content, candidate_intent = _example_fields(item)
        pool.append(
            f'<candidate id="{item["id"]}">\n'
            f"{_tag(intent_tag, candidate_intent)}\n"
            f"{_tag(material_tag, _clip(content, max_content_chars))}\n"
            "</candidate>"
        )
    pool_text = "\n".join(pool)
    target = f"{_tag(intent_tag, intent)}\n{_tag(material_tag, material)}"
    return (
        f"{_tag('candidate_pool', pool_text)}\n\n"
        f"{_tag('target', target)}\n\n"
        "Based on the target and the candidate pool, select the 10 most relevant examples."
    )


# -- planner ---------------------------------------------------------------------

DIAGRAM_PLANNER_SYSTEM = """\
I am working on a task: given the methodology section of a paper and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will give you the text of the methodology section (in <methodology_section>; it is the source material of the paper) and the figure caption (in <diagram_caption>). Your output is a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task and grasp the principles for generating such figures, I may also provide several examples, each with its reference diagram. Learn from these examples when you write your figure description.

Make your description as detailed as possible, because vague or unclear specifications make the generated figure worse, not better. Semantically, clearly describe each element and their connections. Formally, include details such as the background style (typically pure white or very light pastel), colors, line thickness and icon styles.

Reply with the description only; it is passed on to the next stage as it is.
"""

PLOT_PLANNER_SYSTEM = """\
I am working on a task: given raw data (typically in tabular or JSON form) and the visual intent of the desired plot, automatically generate a corresponding statistical plot that is both accurate and aesthetically pleasing. I will give you the raw data (in <raw_data>; it is the source material of the paper, so use the parts the visual intent refers to) and the visual intent (in <visual_intent>). Your output is a detailed description of a plot that effectively represents the data. The description has to include all the raw data points to be plotted.

To help you understand the task and grasp the principles for generating such plots, I may also provide several examples, each with its reference plot. Learn from the content presentation and the aesthetic design (for example the color schemes) of these examples when you write your plot description.

Make your description as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate the coordinates of every raw data point to be drawn, so that the plot is accurate. For presentation, specify the exact aesthetic parameters: specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement and grid styles.

Reply with the description only; it is passed on to the next stage as it is.
"""

#: Added to planner, stylist and critic requests when a diagram will be drawn by plotting
#: code instead of an image model, so the description stays within what code can draw.
DIAGRAM_AS_CODE_NOTE = (
    "This diagram is drawn by a matplotlib script from boxes, simple geometric shapes, arrows "
    "and text labels. Describe only what can be drawn that way: express icons as simple "
    "labelled shapes, and leave out photographic images, 3D renderings and textures."
)


def planner_system(task: TaskName) -> str:
    return PLOT_PLANNER_SYSTEM if task == "plot" else DIAGRAM_PLANNER_SYSTEM


def planner_example(task: TaskName, index: int, item: Mapping[str, Any]) -> str:
    """Text of one few-shot example; its reference image follows as the next part."""
    content, intent = _example_fields(item)
    return (
        f"Example {index}:\n"
        f"{_tag(_MATERIAL_TAG[task], content)}\n"
        f"{_tag(_INTENT_TAG[task], intent)}\n"
        f"Reference {task} of example {index}:"
    )


def planner_request(task: TaskName, *, material: str, intent: str, drawn_with_code: bool) -> str:
    if task == "plot":
        ask = (
            "Based on the raw data and the visual intent above, provide a detailed description "
            "of the plot to be generated."
        )
    else:
        ask = (
            "Based on the methodology section and the diagram caption above, provide a detailed "
            "description of the figure to be generated. Do not include a figure title."
        )
        if drawn_with_code:
            ask += f"\n\n{DIAGRAM_AS_CODE_NOTE}"
    return f"{_tag(_MATERIAL_TAG[task], material)}\n\n{_tag(_INTENT_TAG[task], intent)}\n\n{ask}"


# -- stylist ---------------------------------------------------------------------

DIAGRAM_STYLIST_SYSTEM = """\
## Role
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## Task
Our goal is to generate high-quality, publication-ready diagrams, given the methodology section and the caption of the desired diagram. The diagram should illustrate the logic of the methodology section while adhering to the scope defined by the caption. Before you, a planner agent has already generated a preliminary description of the target diagram. However, this description may lack specific aesthetic details, such as element shapes, color palettes and background styling. Your task is to refine and enrich this description based on the provided style guidelines, so that the final generated image is a high-quality, publication-ready diagram that adheres to the NeurIPS 2025 aesthetic standards where appropriate.

## Input
- <detailed_description>: the preliminary description of the figure.
- <style_guidelines>: the NeurIPS 2025 style guidelines.
- <methodology_section>: contextual content from the methodology section.
- <diagram_caption>: the target diagram caption.

Focus primarily on the detailed description and the style guidelines. The methodology section and the diagram caption are provided for context only: do not regenerate a description from scratch based on them while ignoring the detailed description we already have.

## Instructions
1. Preserve semantic content. Do not alter the semantic content, logic or structure of the diagram: your job is aesthetic refinement, not content editing. If you find some phrases or descriptions too verbose, you may simplify them, referring to the methodology section to keep them semantically accurate.
2. Preserve high-quality aesthetics and intervene only when necessary. First evaluate the aesthetic quality implied by the input description. If it already describes a high-quality, professional and visually appealing diagram (for example nice 3D icons, rich textures, good color harmony), preserve it. Apply strict style-guide adjustments only if the current description lacks detail, looks outdated or is visually cluttered. The goal is specific refinement, not blind standardization.
3. Respect diversity. Different domains have different styles. If the input describes a specific style that works well (for example an illustrative style for agents), keep it.
4. Enrich details. If the input is plain, enrich it with the specific visual attributes (colors, fonts, line styles, layout adjustments) defined in the guidelines.
5. Handle icons with care. Icons may carry specific semantic meanings. Some have conventional technical meanings (for example snowflake = frozen/non-trainable, flame = trainable): when you encounter such icons, check the methodology section for their intent before changing them. Purely decorative or symbolic icons can be freely enhanced and beautified; agent papers, for example, often use cute 2D robot avatars to represent agents.

## Output
Reply with the final polished detailed description only, without conversational text or explanations: your reply is passed verbatim to the stage that draws the figure.
"""

PLOT_STYLIST_SYSTEM = """\
## Role
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## Task
You are given a preliminary description of a statistical plot to be generated. This description may lack specific aesthetic details, such as color palettes, background styling and font choices.

Your task is to refine and enrich this description based on the provided style guidelines, so that the final generated image is a high-quality, publication-ready plot that adheres to the NeurIPS 2025 aesthetic standards.

## Instructions
1. Enrich details. Focus on specifying the visual attributes (colors, fonts, line styles, layout adjustments) defined in the guidelines.
2. Preserve content. Do not alter the semantic content, logic or quantitative results of the plot: your job is aesthetic refinement, not content editing.
3. Context awareness. Use the raw data and the visual intent to understand the emphasis of the plot, so that the style supports the content effectively.

## Input
- <detailed_description>: the preliminary description of the plot.
- <style_guidelines>: the NeurIPS 2025 style guidelines.
- <raw_data>: the raw data to be visualized.
- <visual_intent>: the visual intent of the desired plot.

## Output
Reply with the final polished detailed description only, without conversational text or explanations: your reply is passed verbatim to the stage that draws the figure.
"""

#: Used when no PaperBanana checkout provides the full NeurIPS 2025 style guides.
DEFAULT_DIAGRAM_STYLE_GUIDE = """\
General conventions of method diagrams at top-tier machine learning venues:
- Background: pure white or a very light pastel tint. Related modules are grouped in softly tinted, rounded containers.
- Color: a restrained palette of soft pastels (blues, greens, oranges, purples) with darker outlines of the same hue and one accent color for the key contribution. The same concept has the same color everywhere.
- Shapes: rounded rectangles for modules and processes, cylinders or stacked sheets for data, small circles for operations. Outlines are thin (about 1 to 1.5 pt).
- Arrows: solid dark-gray arrows for the main flow, dashed arrows for auxiliary flows (feedback, gradients, optional paths). Arrows are orthogonal or gently curved and never cross labels.
- Typography: a clean sans-serif font, short labels, bold for module names and regular weight for details, italic math symbols, sizes that stay legible at column width.
- Layout: one clear reading direction (left to right or top to bottom), aligned elements, generous whitespace, no figure title or caption text inside the image.
"""

DEFAULT_PLOT_STYLE_GUIDE = """\
General conventions of statistical plots at top-tier machine learning venues:
- Background and frame: white background, top and right spines removed or a light full frame, thin light-gray dashed grid lines behind the data (horizontal grid only for bar charts).
- Color: a muted, colorblind-friendly categorical palette. The proposed method has the most saturated color, baselines use softer tones or grays; heatmaps use a perceptually uniform sequential colormap.
- Marks: lines about 1.5 to 2 pt wide with a distinct marker per series, bars with thin dark edges and value labels where they help, error bars with small caps.
- Typography: a sans-serif font, axis labels 10 to 12 pt, tick labels and legend 9 to 10 pt, axes labelled with quantity and unit, no title inside the plot because the caption carries it.
- Legend: frameless or lightly framed, placed where it does not cover data.
- Layout: tight layout without clipped labels and with consistent number formatting on the ticks.
"""


def stylist_system(task: TaskName) -> str:
    return PLOT_STYLIST_SYSTEM if task == "plot" else DIAGRAM_STYLIST_SYSTEM


def default_style_guide(task: TaskName) -> str:
    return DEFAULT_PLOT_STYLE_GUIDE if task == "plot" else DEFAULT_DIAGRAM_STYLE_GUIDE


def stylist_request(
    task: TaskName,
    *,
    description: str,
    style_guide: str,
    material: str,
    intent: str,
    drawn_with_code: bool,
) -> str:
    request = (
        f"{_tag('detailed_description', description)}\n\n"
        f"{_tag('style_guidelines', style_guide)}\n\n"
        f"{_tag(_MATERIAL_TAG[task], material)}\n\n"
        f"{_tag(_INTENT_TAG[task], intent)}"
    )
    if task == "diagram" and drawn_with_code:
        request += f"\n\n{DIAGRAM_AS_CODE_NOTE}"
    return request


# -- visualizer ------------------------------------------------------------------

_PLOT_CODE_ROLE = """\
You are an expert statistical plot illustrator. Write code that generates high-quality statistical plots based on user requests.

Follow the description exactly: plot every data point it lists with the values it gives, and use its colors, fonts, sizes and layout. Do not put a figure title or the caption text inside the image unless the description asks for it; the paper adds the caption.
"""

_DIAGRAM_CODE_ROLE = """\
You are an expert scientific diagram illustrator who draws method diagrams as code. Write a matplotlib script that renders the diagram the user describes as a clean, vector-like figure.

How to draw it:
- Build the diagram from matplotlib primitives only: FancyBboxPatch, Rectangle, Circle, Ellipse and Polygon patches for boxes and shapes, FancyArrowPatch or ax.annotate with arrowprops for arrows and connectors, and ax.text for labels. No external images, icon fonts or other assets exist; compose simple icons from these primitives, and where the description asks for something that cannot be drawn this way (a photographic icon, a 3D rendering, a texture), substitute a simple labelled shape that keeps its meaning.
- Use one axes that fills the figure, with a fixed coordinate system whose ranges match the aspect ratio (for example x from 0 to 160 and y from 0 to 90 for 16:9), an equal aspect and the axis switched off.
- Compute positions and sizes from a few layout variables (columns, rows, box width, gaps) instead of scattering literal coordinates, so that elements are aligned and evenly spaced.
- Connect boxes edge to edge: arrows start and end at box borders and never run through a box or a label.
- Make every label fit inside its shape: size boxes from their text, break long labels into several lines, and keep font sizes readable when the figure is printed at column width.
- Use the colors, line widths and styles of the description. Where it is silent, use a white background, soft pastel fills with darker outlines of the same hue, rounded boxes and dark-gray arrows.
- Do not draw a figure title or the caption text inside the image; the paper adds the caption.
"""

_CODE_CONTRACT = """\
How the script is run:
- It runs unattended in a fresh, empty working directory without a display. It must not use the network, and it is stopped after {timeout_seconds} seconds.
- Only the Python standard library, numpy and matplotlib are guaranteed to be installed, so import nothing else (no seaborn, pandas or scipy). No input files exist: embed every value the figure needs in the script.
- Create one figure with matplotlib.pyplot and leave it open when the script ends. It is then saved for you as figure.png with a tight bounding box, so do not call plt.show() or plt.close().
- Use the fonts that ship with matplotlib (DejaVu Sans). LaTeX text rendering (text.usetex) is not available; mathtext ($...$) is.

Reply with the complete Python script in one ```python fenced block and nothing else.
"""


def code_system(task: TaskName, *, timeout_seconds: int) -> str:
    """System prompt of the call that writes the matplotlib script for a figure."""
    role = _PLOT_CODE_ROLE if task == "plot" else _DIAGRAM_CODE_ROLE
    return f"{role}\n{_CODE_CONTRACT.format(timeout_seconds=timeout_seconds)}"


def code_request(
    task: TaskName, *, description: str, aspect_ratio: str, figsize: tuple[float, float]
) -> str:
    width, height = figsize
    if task == "plot":
        ask = (
            "Use Python and matplotlib to generate a statistical plot based on the detailed "
            "description above."
        )
    else:
        ask = "Draw the diagram described above with Python and matplotlib."
    return (
        f"{_tag('figure_description', description)}\n\n"
        f"{ask} The figure is planned with an aspect ratio of {aspect_ratio}: unless the "
        f"description fixes another size, create it with figsize=({width}, {height})."
    )


def code_revision_request(*, description: str, suggestions: str) -> str:
    """Follow-up turn after a critic asked for changes to the rendered figure.

    ``suggestions`` may be empty: a critic can revise the description without commenting.
    """
    blocks = [
        "A reviewer looked at the figure your script rendered and asked for changes; the "
        "description was revised accordingly."
    ]
    if suggestions.strip():
        blocks.append(_tag("reviewer_suggestions", suggestions))
    blocks += [
        _tag("figure_description", description),
        "Update the script so that the figure matches the revised description, and reply with "
        "the complete script.",
    ]
    return "\n\n".join(blocks)


def code_repair_request(*, problem: str, output: str) -> str:
    """Follow-up turn after the script failed to produce a figure."""
    shown = output.strip() or "(the script printed nothing)"
    return (
        f"The script did not produce the figure. {problem}\n\n"
        f"{_tag('error_output', shown)}\n\n"
        "Find the cause, fix it, and reply with the complete corrected script."
    )


IMAGE_MODEL_SYSTEM = (
    "You are an expert scientific diagram illustrator. Generate high-quality scientific "
    "diagrams based on user requests."
)


def image_model_request(description: str) -> str:
    """Prompt for a diagram rendered directly by an image-generation model."""
    return (
        "Render an image based on the following detailed description. Do not include a figure "
        f"title in the image.\n\n{_tag('figure_description', description)}"
    )


# -- critic ----------------------------------------------------------------------

DIAGRAM_CRITIC_SYSTEM = """\
## Role
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## Task
Conduct a sanity check and provide a critique of the target diagram based on its content and presentation. Make sure it is aligned with the provided methodology section and diagram caption.

You are also given the detailed description that the current diagram was generated from. If you identify areas for improvement in the diagram, list your specific critique and provide a revised version of the detailed description that incorporates these corrections.

## Critique and revision rules

1. Content
   - Fidelity and alignment: the diagram accurately reflects the method described in the methodology section and aligns with the diagram caption. Reasonable simplifications are allowed, but no critical component may be omitted or misrepresented, and the diagram must not contain hallucinated content. Consistency with the provided methodology section and diagram caption is the most important thing.
   - Text QA: check for typographical errors, nonsensical text or unclear labels within the diagram, and suggest specific corrections.
   - Validation of examples: verify the accuracy of illustrative examples. If the diagram includes specific examples to aid understanding (for example molecular formulas, attention maps, mathematical expressions), make sure they are factually correct and logically consistent. If an example is incorrect, provide the correct version.
   - Caption exclusion: the figure caption text (for example "Figure 1: Overview...") is not part of the image itself. The caption stays separate.

2. Presentation
   - Clarity and readability: evaluate the overall visual clarity. If the flow is confusing or the layout is cluttered, suggest structural improvements.
   - Legend management: the description and the diagram may include a text-based legend explaining the color coding. This is typically redundant, so remove such descriptions if you find them.

Your revised description should primarily modify the original description rather than rewrite it from scratch. Where a part of the original description has obvious problems and needs to be described again, make it as detailed as possible: semantically, clearly describe each element and their connections; formally, include details such as background, colors, line thickness and icon styles. Vague or unclear specifications make the generated figure worse, not better.

## Input
- The target diagram: the generated figure, as an image.
- <detailed_description>: the detailed description of the figure.
- <methodology_section>: contextual content from the methodology section.
- <diagram_caption>: the target diagram caption.

## Output
- needs_revision: true when the diagram should be generated again from a revised description, false when it is good as it is.
- critic_suggestions: your detailed critique and specific suggestions for improvement. If the diagram is good as it is, write "No changes needed."
- revised_description: the fully revised detailed description, incorporating all your suggestions. Leave it empty when no changes are needed.
"""

PLOT_CRITIC_SYSTEM = """\
## Role
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## Task
Conduct a sanity check and provide a critique of the target plot based on its content and presentation. Make sure it is aligned with the provided raw data and visual intent.

You are also given the detailed description that the current plot was generated from. If you identify areas for improvement in the plot, list your specific critique and provide a revised version of the detailed description that incorporates these corrections.

## Critique and revision rules

1. Content
   - Data fidelity and alignment: the plot accurately represents all data points from the raw data and aligns with the visual intent. All quantitative values are correct, and no data is hallucinated, omitted or misrepresented.
   - Text QA: check for typographical errors, nonsensical text or unclear labels within the plot (axis labels, legend entries, annotations), and suggest specific corrections.
   - Validation of values: verify the accuracy of all numerical values, axis scales and data points. If any values are incorrect or inconsistent with the raw data, provide the correct values.
   - Caption exclusion: the figure caption text (for example "Figure 1: Performance comparison...") is not part of the image itself. The caption stays separate.

2. Presentation
   - Clarity and readability: evaluate the overall visual clarity. If the plot is confusing, cluttered or hard to interpret, suggest structural improvements (for example better axis labeling, a clearer legend, a more appropriate plot type).
   - Overlap and layout: check for overlapping elements that reduce readability, such as text labels obscured by heavy hatching, grid lines or other chart elements (for example pie chart labels inside dark slices). If overlaps exist, suggest adjusting element positions (for example moving labels outside the chart, using leader lines, or adjusting transparency).
   - Legend management: the description and the plot may include a text-based legend explaining symbols or colors. This is typically redundant in a well-designed plot, so remove such descriptions if you find them.

Your revised description replaces the current one and the plot is drawn again from it, so it has to stay complete: modify the original description rather than rewrite it from scratch, and keep every data point the plot needs.

## Input
- The target plot: the generated plot, as an image.
- <detailed_description>: the detailed description of the plot.
- <raw_data>: the raw data to be visualized.
- <visual_intent>: the visual intent of the desired plot.

## Output
- needs_revision: true when the plot should be generated again from a revised description, false when it is good as it is.
- critic_suggestions: your detailed critique and specific suggestions for improvement. If the plot is good as it is, write "No changes needed."
- revised_description: the fully revised detailed description, incorporating all your suggestions. Leave it empty when no changes are needed.
"""


def critic_system(task: TaskName) -> str:
    return PLOT_CRITIC_SYSTEM if task == "plot" else DIAGRAM_CRITIC_SYSTEM


def critic_intro(task: TaskName) -> str:
    """Text that precedes the rendered image in the critic's user message."""
    return f"Target {task} for critique:"


def critic_request(
    task: TaskName, *, description: str, material: str, intent: str, drawn_with_code: bool
) -> str:
    """Text that follows the rendered image in the critic's user message."""
    request = (
        f"{_tag('detailed_description', description)}\n\n"
        f"{_tag(_MATERIAL_TAG[task], material)}\n\n"
        f"{_tag(_INTENT_TAG[task], intent)}"
    )
    if task == "diagram" and drawn_with_code:
        request += f"\n\n{DIAGRAM_AS_CODE_NOTE}"
    return request


# -- caption ---------------------------------------------------------------------

CAPTION_SYSTEM = """\
You write figure captions for academic papers. You are given the kind of figure, the source material it was made from, the overall intent of the figure, its detailed description and, when available, the rendered figure. Write the final caption for this figure.

Requirements:
- The caption is concise and informative and can be used directly as a caption in an academic paper.
- It does not start with a "Figure X:" or "Caption X:" prefix, because the LaTeX template adds the numbering automatically.
- It is plain text without markdown formatting (no bold, italics or similar).

Reply with the caption text only.
"""


def caption_request(task: TaskName, *, material: str, intent: str, description: str) -> str:
    return (
        f"{_tag('task_type', task)}\n\n"
        f"{_tag('context', material)}\n\n"
        f"{_tag('figure_intent', intent)}\n\n"
        f"{_tag('figure_description', description)}\n\n"
        "Provide the final caption for this figure."
    )
