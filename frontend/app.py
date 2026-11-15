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

"""Streamlit demo of PaperOrchestra.

Collects an idea, an experimental log and optional figures, runs
``paper-orchestra write`` as a child process and shows what the run produces:
stage progress, the outline, the citation map, figures, refinement rounds and
the wiki memory.  Launch it with ``uv run streamlit run frontend/app.py``.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import frontend_utils as fu
import streamlit as st

from paper_orchestra.config import load_env
from paper_orchestra.context import RunPaths

st.set_page_config(
    page_title="Paper Orchestra - AI Paper Writer",
    page_icon="🎻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Credentials and SMTP settings may live in the repository's .env; the pipeline process inherits them.
load_env(fu.REPO_ROOT / ".env")

NO_IMAGE_MODEL = "None (draw diagrams with code)"
DREAM_LABELS = {
    "stages": "After each stage",
    "end": "At the end of the run",
    "off": "Off",
}
#: Timeline CSS class and icon per stage status.
STAGE_STYLE = {
    "completed": ("finished", "✓"),
    "running": ("active", "⏳"),
    "failed": ("failed", "✗"),
    "interrupted": ("failed", "■"),
    "skipped": ("", "–"),
    "pending": ("", "○"),
}


def apply_custom_css() -> None:
    st.markdown(
        """
    <style>
        .stApp { background: #f8fafc; color: #0f172a; }
        [data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #e2e8f0; }
        .card { background: white; border-radius: 12px; padding: 2rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); margin-bottom: 1.5rem; }
        .timeline-container { padding: 1rem; }
        .timeline-item { border-left: 2px solid #e2e8f0; padding-left: 1.5rem; position: relative; margin-bottom: 1.5rem; }
        .timeline-item.active { border-left-color: #2563eb; }
        .timeline-item.finished { border-left-color: #10b981; }
        .timeline-item.failed { border-left-color: #dc2626; }
        .timeline-icon { position: absolute; left: -11px; top: 0; width: 20px; height: 20px; border-radius: 50%; background: white; border: 2px solid #cbd5e1; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #64748b; }
        .timeline-item.active .timeline-icon { border-color: #b45309; background: #fffbeb; color: #b45309; }
        .timeline-item.finished .timeline-icon { border-color: #10b981; background: #10b981; color: white; }
        .timeline-item.failed .timeline-icon { border-color: #dc2626; background: #dc2626; color: white; }
        .timeline-detail { color: #64748b; font-size: 0.85rem; margin: 0; }
        .gallery-item { background: white; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
        .stButton > button { background-color: #ecfdf5 !important; color: #065f46 !important; border: 1px solid #a7f3d0 !important; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: 600; width: 100%; transition: all 0.2s; }
        .stButton > button:hover { transform: translateY(-1px); background-color: #d1fae5 !important; box-shadow: 0 4px 6px -1px rgba(5, 150, 105, 0.2); }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: white !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "locked": False,
        "hide_inputs": False,
        "figures": [],
        "figure_counter": 0,
        "run": None,
        "run_options": None,
        "run_directory": None,
        "wiki": None,
        "email_confirmed": False,
        "error_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_running() -> bool:
    run = st.session_state.run
    return run is not None and run.running


def run_paths() -> RunPaths | None:
    directory = st.session_state.run_directory
    return RunPaths(Path(directory)) if directory else None


# ------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------


def render_sidebar() -> dict[str, Any]:
    locked = st.session_state.locked
    st.sidebar.title("🎻 Configuration")

    if st.sidebar.button("Load CVPR Example", disabled=locked):
        try:
            example = fu.load_example()
        except (OSError, ValueError) as error:
            st.sidebar.error(f"Error loading example: {error}")
        else:
            st.session_state.idea_text_main = example["idea"]
            st.session_state.log_text_main = example["log"]
            if example["figures"]:
                st.session_state.figures = example["figures"]
            st.sidebar.success("🎉 Example loaded successfully!")
            time.sleep(1)
            st.rerun()

    model = st.sidebar.selectbox("Select Model", fu.MODEL_CHOICES, disabled=locked)
    template = st.sidebar.selectbox("Select Template", fu.TEMPLATE_CHOICES, disabled=locked)
    plotting = st.sidebar.selectbox(
        "Enable Figure Generation", ["Yes", "No"], index=0, disabled=locked
    )
    image_model = st.sidebar.selectbox(
        "Diagram Image Model",
        [NO_IMAGE_MODEL, *fu.IMAGE_MODEL_CHOICES],
        disabled=locked or plotting == "No",
        help="Diagrams are drawn by model-written plotting code unless an image model is chosen. "
        "A Gemini image model needs Gemini credentials.",
    )
    compute = st.sidebar.selectbox(
        "Compute Backend",
        fu.COMPUTE_CHOICES,
        disabled=locked,
        help="Where model-written figure code runs and LaTeX compiles: on this machine, "
        "or in network-isolated Modal sandboxes.",
    )
    memory = st.sidebar.checkbox(
        "Wiki Memory",
        value=True,
        disabled=locked,
        help="Keep a wiki of what the run learns (sources, literature, reviews, lessons) "
        "that every agent can consult.",
    )
    dream = st.sidebar.selectbox(
        "Memory Consolidation (Dreaming)",
        fu.DREAM_CHOICES,
        format_func=DREAM_LABELS.get,
        disabled=locked or not memory,
        help="When a separate session folds notes into wiki pages and tested rules.",
    )

    st.sidebar.markdown("Your Email (for the completion notification, optional)")
    email = st.sidebar.text_input(
        "Email", key="email_input", label_visibility="collapsed", disabled=locked
    )
    if st.sidebar.button("Confirm Email", disabled=locked):
        st.session_state.email_confirmed = fu.valid_email_address(email.strip())
        if not st.session_state.email_confirmed:
            st.sidebar.warning("⚠️ Please enter one valid email address first.")
    if st.session_state.get("email_confirmed"):
        st.sidebar.success("📧 Email confirmed successfully!")

    return {
        "model": model,
        "template": template,
        "use_plotting": plotting == "Yes",
        "image_model": None if image_model == NO_IMAGE_MODEL else image_model,
        "compute": compute,
        "memory": memory,
        "dream": dream,
        "email": email.strip() or None,
    }


# ------------------------------------------------------------------------
# RUN CONTROL
# ------------------------------------------------------------------------


def start_run(choices: dict[str, Any]) -> None:
    """Validate the inputs, prepare the run directory and launch the pipeline."""
    idea = (st.session_state.get("idea_text_main") or "").strip()
    experimental_log = (st.session_state.get("log_text_main") or "").strip()
    if not idea or not experimental_log:
        st.session_state.error_message = (
            "❌ Please fill in all required fields (Idea and Experimental Log)."
        )
        st.rerun()
    if choices["email"] and not fu.valid_email_address(choices["email"]):
        # Better now than after 40 minutes, when the notification would be refused.
        st.session_state.error_message = (
            "❌ The email address is not valid. Correct it, or clear the field to run "
            "without a notification."
        )
        st.rerun()

    options = fu.RunOptions(
        run_dir=fu.new_run_directory(),
        template=choices["template"],
        model=choices["model"],
        use_plotting=choices["use_plotting"],
        image_model=choices["image_model"],
        compute=choices["compute"],
        memory=choices["memory"],
        dream=choices["dream"],
    )
    try:
        fu.write_raw_materials(options.run_dir, idea, experimental_log, st.session_state.figures)
    except (OSError, ValueError) as error:
        st.session_state.error_message = f"❌ Could not prepare the run directory: {error}"
        st.rerun()

    recipient = choices["email"]
    run = fu.PipelineRun(
        fu.build_write_command(options),
        cwd=fu.REPO_ROOT,
        log_path=options.run_dir / fu.PIPELINE_LOG_NAME,
        on_finish=lambda finished: fu.notify_completion(finished, options.run_dir, recipient),
    )
    run.start()

    st.session_state.run = run
    st.session_state.run_options = options
    st.session_state.run_directory = str(options.run_dir)
    st.session_state.wiki = None
    st.session_state.saved_idea = idea
    st.session_state.saved_log = experimental_log
    st.session_state.locked = True
    st.session_state.hide_inputs = True
    st.rerun()


def reset_for_new_paper() -> None:
    st.session_state.run = None
    st.session_state.run_options = None
    st.session_state.run_directory = None
    st.session_state.wiki = None
    st.session_state.locked = False
    st.session_state.hide_inputs = False
    st.rerun()


# ------------------------------------------------------------------------
# UI COMPONENT FUNCTIONS
# ------------------------------------------------------------------------


def render_inputs(choices: dict[str, Any]) -> None:
    locked = st.session_state.locked
    if st.session_state.get("error_message"):
        st.error(st.session_state.error_message)
        st.session_state.error_message = ""

    st.header("📝 Input Materials")
    column_1, column_2 = st.columns(2)

    with column_1:
        st.subheader("💡 Research Idea")
        idea_input_type = st.radio(
            "Idea Input Type", ["Text", "Upload"], disabled=locked, key="idea_input_type_main"
        )
        if idea_input_type == "Text":
            st.markdown(
                "Enter your idea here <span style='color:red'>*</span>", unsafe_allow_html=True
            )
            st.text_area(
                "Idea",
                key="idea_text_main",
                label_visibility="collapsed",
                disabled=locked,
                height=300,
            )
        else:
            st.markdown(
                "Upload Idea file (.md/.txt) <span style='color:red'>*</span>",
                unsafe_allow_html=True,
            )
            uploaded_idea = st.file_uploader(
                "Upload Idea",
                type=["md", "txt"],
                label_visibility="collapsed",
                disabled=locked,
                key="idea_uploader_main",
            )
            if uploaded_idea:
                st.session_state.idea_text_main = uploaded_idea.getvalue().decode(
                    "utf-8", errors="replace"
                )

    with column_2:
        st.subheader("📊 Experimental Log")
        log_input_type = st.radio(
            "Exp Log Input Type", ["Text", "Upload"], disabled=locked, key="log_input_type_main"
        )
        if log_input_type == "Text":
            st.markdown(
                "Enter experimental log here <span style='color:red'>*</span>",
                unsafe_allow_html=True,
            )
            st.text_area(
                "Exp Log",
                key="log_text_main",
                label_visibility="collapsed",
                disabled=locked,
                height=300,
            )
        else:
            st.markdown(
                "Upload Exp Log file (.md/.txt) <span style='color:red'>*</span>",
                unsafe_allow_html=True,
            )
            uploaded_log = st.file_uploader(
                "Upload Exp Log",
                type=["md", "txt"],
                label_visibility="collapsed",
                disabled=locked,
                key="log_uploader_main",
            )
            if uploaded_log:
                st.session_state.log_text_main = uploaded_log.getvalue().decode(
                    "utf-8", errors="replace"
                )

    st.subheader("🖼️ Upload Figures")
    figure_column_1, figure_column_2 = st.columns([1, 2])
    with figure_column_1:
        figure_file = st.file_uploader(
            "Upload figure",
            type=["png", "jpg", "jpeg", "pdf"],
            disabled=locked,
            key="figure_uploader_main",
        )
    with figure_column_2:
        figure_caption = st.text_input("Figure Caption", disabled=locked, key="figure_caption_main")
        button_column, _ = st.columns([1, 2])
        with button_column:
            st.markdown(
                """
            <style>
                .stButton > button { background-color: #e0e7ff !important; color: #3730a3 !important; border: 1px solid #c7d2fe !important; }
            </style>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Add Figure", disabled=locked or not figure_file or not figure_caption):
                st.session_state.figure_counter += 1
                st.session_state.figures.append(
                    {
                        "id": st.session_state.figure_counter,
                        "file": figure_file,
                        "name": figure_file.name,
                        "caption": figure_caption,
                    }
                )
                st.success("🎉 Figure added successfully!")
                time.sleep(1)
                st.rerun()

    render_uploaded_figures()

    st.markdown(
        """
    <style>
        .stButton > button[kind="primary"] { background-color: #2563eb !important; color: white !important; font-size: 1.25rem !important; padding: 1rem 2rem !important; border: none !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important; }
        .stButton > button[kind="primary"]:hover { background-color: #1d4ed8 !important; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    if st.button("🚀 Begin Writing", type="primary", disabled=locked):
        start_run(choices)


def render_uploaded_figures() -> None:
    """The figures added so far, with editable captions until the run starts."""
    locked = st.session_state.locked
    if not st.session_state.figures:
        if not locked:
            st.markdown(
                "<p style='color: gray; font-style: italic;'>No figures uploaded yet.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("ℹ️ No user figures provided for this run.")
        return
    for index, figure in enumerate(st.session_state.figures):
        # Keyed by the figure, not its position, so deleting one leaves the others' widgets alone.
        widget_id = figure.get("id", "example")
        st.markdown(
            f'<div class="gallery-item" style="margin-bottom: 1rem;"><p style="font-weight: bold;">{html.escape(figure["name"])}</p></div>',
            unsafe_allow_html=True,
        )
        if figure["name"].lower().endswith(".pdf"):
            st.caption("PDF figure (no preview).")
        else:
            try:
                st.image(figure["file"], width="stretch")
            except Exception as error:
                # A file that is not really an image must stay on the page with its delete
                # button: an exception here would end the script above the rest of the form.
                st.warning(f"This file cannot be shown as an image ({type(error).__name__}).")
        new_caption = st.text_input(
            "Edit Caption",
            value=figure["caption"],
            key=f"edit_caption_{widget_id}",
            disabled=locked,
        )
        if new_caption != figure["caption"]:
            st.session_state.figures[index]["caption"] = new_caption
        if st.button("🗑️ Delete Image", key=f"delete_figure_{widget_id}", disabled=locked):
            st.session_state.figures.pop(index)
            st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)


def read_json_file(path: Path) -> Any:
    """Parsed JSON, or ``None`` when the file is absent or still being written."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Error loading {path.name}: {error}")
        return None


def render_gallery_tab() -> None:
    st.subheader("Uploaded Figures")
    render_uploaded_figures()

    st.subheader("Generated Figures")
    paths = run_paths()
    generated = read_json_file(paths.plotting_results) if paths else None
    if not isinstance(generated, list) or not generated:
        if not is_running():
            st.info("No generated figures.")
        return
    for figure in generated:
        if not isinstance(figure, dict):
            continue
        image_path = paths.base_dir / str(figure.get("image_path", ""))
        if image_path.is_file():
            st.markdown(f"**{figure.get('title', 'Figure')}**")
            st.image(str(image_path), caption=figure.get("caption", ""))
            st.markdown("<hr>", unsafe_allow_html=True)


def render_json_tab(path: Path | None, title: str, missing_message: str) -> None:
    st.subheader(title)
    content = read_json_file(path) if path else None
    if content is not None:
        st.json(content)
    elif not is_running():
        st.info(missing_message)


def page_order(path: Path) -> list[int | str]:
    """Sort key that puts ``grid_2.png`` before ``grid_10.png``."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def render_pdf_and_snapshots(pdf_path: Path, snapshots_directory: Path, download_key: str) -> None:
    if not pdf_path.is_file():
        return
    st.download_button(
        label="📥 Download PDF",
        data=pdf_path.read_bytes(),
        file_name=pdf_path.name,
        mime="application/pdf",
        key=download_key,
    )
    if snapshots_directory.is_dir():
        image_files = sorted(
            (path for path in snapshots_directory.iterdir() if path.suffix in (".jpg", ".png")),
            key=page_order,
        )
        if image_files:
            st.markdown("**Snapshots:**")
            for column, image_file in zip(st.columns(3), image_files[:3], strict=False):
                with column:
                    st.image(str(image_file), width="stretch")


def stage_detail(stage: fu.StageProgress) -> str:
    """One line under a timeline entry: how the stage is doing."""
    minutes, seconds = divmod(int(stage.seconds), 60)
    took = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
    if stage.status == "completed":
        return "Reused from the earlier run." if stage.reused else f"Completed in {took}."
    if stage.status == "running":
        return f"Running (attempt {stage.attempts})." if stage.attempts > 1 else "Running."
    if stage.status == "failed":
        outcome = " The run continued without it." if stage.tolerated else ""
        return (
            f"Failed after {stage.attempts} attempt(s): {stage.error or 'unknown error'}.{outcome}"
        )
    if stage.status == "interrupted":
        return "Interrupted."
    if stage.status == "skipped":
        return "Skipped."
    return ""


def render_timeline_tab(run: fu.PipelineRun, progress: fu.RunProgress) -> None:
    st.subheader("Writing Progress")
    caption = f"{progress.done} of {len(progress.stages)} stages completed"
    if progress.active:
        caption += f" · now: {', '.join(progress.active)}"
    st.progress(progress.fraction, text=caption)

    for stage in progress.stages:
        state_class, icon = STAGE_STYLE.get(stage.status, STAGE_STYLE["pending"])
        st.markdown(
            f'<div class="timeline-item {state_class}"><div class="timeline-icon">{icon}</div>'
            f'<p style="font-weight: bold; margin-bottom: 0.25rem;">{html.escape(stage.label)}</p>'
            f'<p class="timeline-detail">{html.escape(stage_detail(stage))}</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("View Pipeline Log", expanded=run.running):
        log_tail = run.log_tail()
        if log_tail:
            st.code(log_tail, language=None)
        else:
            st.text("No logs yet.")

    paths = run_paths()
    if run.running or paths is None:
        return
    archive = fu.latex_source_zip(paths.base_dir)
    if not paths.final_pdf.is_file() and archive is None:
        return
    st.markdown("### 📥 Download Results")
    column_download_1, column_download_2 = st.columns(2)
    if paths.final_pdf.is_file():
        with column_download_1:
            st.download_button(
                label="📄 Download Final PDF",
                data=paths.final_pdf.read_bytes(),
                file_name="final_paper.pdf",
                mime="application/pdf",
                key="download_final_pdf_tab",
            )
    if archive is not None:
        with column_download_2:
            st.download_button(
                label="📦 Download LaTeX Source",
                data=archive,
                file_name="latex_source.zip",
                mime="application/zip",
                key="download_final_latex_tab",
            )


def render_inputs_tab() -> None:
    st.header("📥 User Inputs")
    st.markdown(
        "<h3 style='background-color: #ffe6e6; padding: 5px; border-radius: 3px;'>Idea</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.get("saved_idea", "*(No input provided or process not started)*"))
    st.markdown(
        "<h3 style='background-color: #ffffcc; padding: 5px; border-radius: 3px;'>Experimental Log</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.get("saved_log", "*(No input provided or process not started)*"))


def render_refinement_tab() -> None:
    st.header("Content & Format Refinement Progress")
    paths = run_paths()
    if paths is None:
        st.info("No refinement progress available yet.")
        return
    work_dir = paths.refinement_dir
    screenshots = work_dir / "pdf_screenshots"
    initial_pdf_path = work_dir / "initial_draft.pdf"
    worklog = read_json_file(work_dir / "content_refinement_worklog.json")

    st.markdown("### Content Refinement")
    columns = st.columns(2)
    if initial_pdf_path.is_file():
        with columns[0], st.expander("Content Refinement V0 (Initial)"):
            st.write("Initial draft compiled.")
            render_pdf_and_snapshots(
                initial_pdf_path, screenshots / "content_refinement" / "v0", "download_content_0"
            )
    if not isinstance(worklog, dict):
        if not is_running() and not initial_pdf_path.is_file():
            st.info("No refinement progress available yet.")
        return

    rounds = {key: entry for key, entry in worklog.items() if isinstance(entry, dict)}
    for index, (version_key, entry) in enumerate(rounds.items()):
        with columns[(index + 1) % 2], st.expander(f"Content Refinement {version_key.upper()}"):
            st.write(f"**Outcome:** {entry.get('outcome')}")
            st.write(f"**Total Gain:** {entry.get('total_gain')}")
            st.write(f"**Total Drop:** {entry.get('total_drop')}")
            if "scores_before" in entry and "scores_after" in entry:
                st.markdown("**Scores:**")
                for axis in ["Overall", "Originality", "Quality", "Clarity"]:
                    st.write(
                        f"- {axis}: {entry['scores_before'].get(axis, 0)} -> {entry['scores_after'].get(axis, 0)}"
                    )
            render_pdf_and_snapshots(
                work_dir / f"refined_paper_{version_key}.pdf",
                screenshots / "content_refinement" / version_key,
                f"download_content_{version_key}",
            )

    format_worklog = read_json_file(work_dir / "format_refinement_worklog.json")
    if not isinstance(format_worklog, dict):
        return
    selected_content = fu.last_accepted_version(worklog)
    selected_label = selected_content.upper() if selected_content else "V0 (Initial)"
    st.info(
        f"💡 `Content Refinement {selected_label}` is selected at the output of the content "
        "refinement phase, now move on to formatting refinement..."
    )

    st.markdown("### Format Refinement")
    format_columns = st.columns(2)
    with format_columns[0], st.expander("Format Refinement V0 (Initial)"):
        st.write("Input from content refinement phase.")
        source_pdf = work_dir / (
            f"refined_paper_{selected_content}.pdf" if selected_content else "initial_draft.pdf"
        )
        if source_pdf.is_file():
            st.download_button(
                label="📥 Download PDF",
                data=source_pdf.read_bytes(),
                file_name=source_pdf.name,
                mime="application/pdf",
                key="download_format_0",
            )
    format_rounds = {k: entry for k, entry in format_worklog.items() if isinstance(entry, dict)}
    for index, (version_key, entry) in enumerate(format_rounds.items()):
        iteration = fu.version_number(version_key)
        with format_columns[(index + 1) % 2], st.expander(f"Format Refinement V{iteration}"):
            if entry.get("outcome"):
                st.write(f"**Outcome:** {entry['outcome']}")
            feedback = entry.get("formatting_feedback") or {}
            if "figure_and_tables" in feedback:
                st.markdown("**Figure & Table Issues:**")
                st.json(feedback["figure_and_tables"])
            render_pdf_and_snapshots(
                work_dir / f"formatted_candidate_{version_key}.pdf",
                screenshots / "format_refinement" / version_key,
                f"download_format_{iteration}",
            )

    if (work_dir / "final_refined_paper.pdf").is_file():
        selected_format = fu.last_accepted_version(format_worklog)
        if selected_format:
            st.info(
                f"💡 `Format Refinement V{fu.version_number(selected_format)}` is selected as "
                "the final output of this pipeline."
            )
        else:
            st.info(
                f"💡 No formatting change was accepted: `Content Refinement {selected_label}` "
                "is the final output of this pipeline."
            )
    elif not is_running():
        st.info("💡 Refinement pipeline ended without producing a final separated PDF.")


def render_memory_tab() -> None:
    st.header("Wiki Memory")
    paths = run_paths()
    if paths is None:
        st.info("No run yet.")
        return
    if st.session_state.wiki is None:
        st.session_state.wiki = fu.open_run_wiki(paths.base_dir)
    wiki = st.session_state.wiki
    if wiki is None:
        options = st.session_state.run_options
        if options is not None and not options.memory:
            st.info("Memory is switched off for this run.")
        else:
            st.info("The run has not written any memory yet.")
        return

    st.caption(f"Memory directory: `{paths.memory_dir}`")
    index_column, page_column = st.columns(2)
    with index_column:
        st.markdown("### Injected index (`MEMORY.md`)")
        st.caption("Loaded into every agent session: standing notes and the page catalog.")
        st.markdown(fu.memory_index_markdown(wiki) or "*The index is being rewritten.*")
    with page_column:
        st.markdown("### Wiki pages")
        entries = fu.list_wiki_pages(wiki)
        if not entries:
            st.info("No wiki pages yet.")
            return
        by_path = {entry.path: entry for entry in entries}
        # A dream may have moved or removed the page that was open.
        if st.session_state.get("memory_page") not in by_path:
            st.session_state.pop("memory_page", None)
        selected = st.selectbox(
            "Open a page",
            list(by_path),
            format_func=lambda path: by_path[path].label,
            key="memory_page",
        )
        entry = by_path[selected]
        page = fu.read_wiki_page(wiki, selected)
        if page is None:
            st.warning("This page no longer exists.")
            return
        meta, body = page
        st.caption(f"`{entry.path}`" + (f" · {entry.summary}" if entry.summary else ""))
        if meta:
            with st.expander("Front matter"):
                st.json(meta)
        st.markdown(body)


def render_run_status(run: fu.PipelineRun, progress: fu.RunProgress) -> None:
    paths = run_paths()
    if run.running:
        st.info(
            "🚀 Paper generation in progress! This may take 35-45 minutes. "
            "Please do not close this tab."
        )
        if st.button("⏹ Stop Run", key="stop_run"):
            with st.spinner("Stopping the pipeline..."):
                run.stop()
            st.rerun()
        return
    if run.launch_error:
        st.error(f"❌ {run.launch_error}")
    elif paths is not None and fu.run_succeeded(run, paths.base_dir):
        st.success("🎉 Paper writing complete!")
    elif run.stopped:
        st.warning("Paper writing was stopped.")
    else:
        reason = f" {progress.error}" if progress.error else ""
        st.error(f"❌ Paper writing failed or stopped unexpectedly. Check logs.{reason}")
    if run.notification:
        st.caption(run.notification)
    if st.button("Start a New Paper", key="new_paper"):
        reset_for_new_paper()


def render_tab(render: Callable[..., None], *arguments: Any) -> None:
    """Draw one tab and keep a failure inside it.

    The page refreshes itself at the end of ``render_run``.  An exception
    escaping from a tab (a half-written output file, a page the run just
    moved) would end the script before that and leave a running pipeline with
    a frozen page.  Streamlit's own rerun and stop signals are not
    ``Exception`` subclasses, so they still pass through.
    """
    try:
        render(*arguments)
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
        # Logged once per problem, not on every refresh of the page.
        if st.session_state.get(f"tab_problem_{render.__name__}") != problem:
            st.session_state[f"tab_problem_{render.__name__}"] = problem
            fu.log.exception("The %s part of the demo page could not be drawn.", render.__name__)
        st.error(f"This tab could not be drawn: {problem}")


def render_run() -> None:
    run: fu.PipelineRun = st.session_state.run
    options: fu.RunOptions = st.session_state.run_options
    # Sampled once: if the run ends while this page is being drawn, the refresh at the
    # bottom still happens and the next pass shows the final state.
    was_running = run.running
    progress = fu.read_progress(
        options.run_dir, use_plotting=options.use_plotting, process_running=was_running
    )
    render_run_status(run, progress)

    paths = options.paths
    (
        tab_timeline,
        tab_inputs,
        tab_gallery,
        tab_outline,
        tab_citation,
        tab_refinement,
        tab_memory,
    ) = st.tabs(
        [
            "📊 Progress Timeline",
            "📥 Inputs",
            "🖼️ Figures Gallery",
            "📝 Outline",
            "🔗 Citation Map",
            "📈 Refinement Progress",
            "🧠 Memory",
        ]
    )
    with tab_timeline:
        render_tab(render_timeline_tab, run, progress)
    with tab_inputs:
        render_tab(render_inputs_tab)
    with tab_gallery:
        render_tab(render_gallery_tab)
    with tab_outline:
        render_tab(render_json_tab, paths.outline, "Paper Outline", "No outline generated yet.")
    with tab_citation:
        render_tab(
            render_json_tab, paths.citation_map, "Citation Map", "No citation map generated yet."
        )
    with tab_refinement:
        render_tab(render_refinement_tab)
    with tab_memory:
        render_tab(render_memory_tab)

    if was_running:
        time.sleep(fu.POLL_SECONDS)
        st.rerun()


# ------------------------------------------------------------------------
# MAIN PAGE ROUTING
# ------------------------------------------------------------------------

apply_custom_css()
initialize_session_state()
sidebar_choices = render_sidebar()

st.title("🎻 Paper Orchestra Demo")
st.markdown(
    "Demonstrating the automated paper writing process as detailed in "
    "[PaperOrchestra](https://arxiv.org/abs/2604.05018)."
)

if st.session_state.hide_inputs and st.session_state.run is not None:
    render_run()
else:
    render_inputs(sidebar_choices)
