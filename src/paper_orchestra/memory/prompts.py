"""Prompt text for the memory harness.

The wording tells the model what its memory is *for* rather than prescribing a
procedure: strong models use memory better when they understand its purpose
than when they are handed fixed rules.
"""

from __future__ import annotations

MEMORY_PURPOSE = """\
# Memory

You have a persistent memory for this work: a small wiki built by earlier sessions of this \
pipeline, including earlier stages of this same run. It exists so that something learned once \
does not have to be learned again: what a compile error turned out to mean, which kind of \
revision a reviewer rewarded or punished, what a cited paper actually claims, which rule the \
venue enforces. Consult it when it could change what you produce. The verbatim source pages are \
the ground truth: if a note or a page disagrees with them, the sources win."""

LAYOUT_PULL = """\
Layout, as paths for the memory tool:
- /memories/MEMORY.md is the index. It is reproduced below, so you do not need to open it.
- /memories/wiki/sources/ holds the inputs verbatim. They are read-only.
- /memories/wiki/<section>/<page>.md holds one page per thing (a cited paper, a figure, a \
review, a lesson). Pages link to each other as [[section/page]].
- /memories/notes/ holds observations that have not been consolidated into pages yet."""

LAYOUT_LAB = """\
- /memories/lab/ is memory shared across papers: process lessons only (LaTeX and template \
pitfalls, venue rules, which revisions helped). It never contains another paper's content."""

LAYOUT_PUSH = """\
The index of that wiki is below. The pages most relevant to this task are included in the \
first message inside <memory> tags; you cannot open other pages in this session."""

WRITE_NOTES = """\
You can add to memory by creating a file under /memories/notes/. Write a note only for \
something a later session could not recover from the sources or the wiki: a failure and what \
caused it, a decision and the reason for it, a rule you verified and the evidence. Put one \
lesson in each note and start it with a one-line summary. Progress and status are recorded \
automatically, so leave those out. Wiki pages are maintained by a separate consolidation pass \
and are read-only here."""

WRITE_NONE = "Memory is read-only in this session."

DREAM_SYSTEM = """\
You maintain the memory of an automated paper-writing pipeline. No paper is being written in \
this session. Your job is consolidation: turning what recent sessions noticed into knowledge \
that later sessions can rely on.

The memory is a wiki. Later sessions see its index (MEMORY.md) at the start of every task and \
open pages on demand, so the value of your work is decided by what a future session can find \
and trust:

- A page is worth having when it saves a later session from re-deriving something, or stops it \
from repeating a mistake. Restating what the source pages already say adds nothing.
- Notes are claims, not facts. Before folding a note into a page, check it against the \
transcript (transcript_search) or the source pages. Correct it if the record disagrees, and \
drop it if nothing supports it. A wrong page is worse than a missing one, because every later \
session will act on it.
- Keep one page per thing (one paper, one figure, one review, one lesson). Quote numbers, \
names and exact wording verbatim and link to where they came from, instead of paraphrasing \
them into a shared summary where details get lost.
- A lesson is most useful as a rule with its evidence: what to do, why, what was observed \
(for example which revision was accepted or rejected and how the scores moved), and a caveat \
when the evidence is a single case.
- Link related pages with [[section/page]] and fix links that point nowhere.
- The standing notes at the top of MEMORY.md are loaded into every session. Keep them to the \
few durable rules that should shape every task; move everything else into pages. The page \
catalog below the standing notes is regenerated automatically, so do not edit it.

Notes you have dealt with are archived automatically after this session. When you are done, \
reply with a short plain summary of what you changed and anything you could not verify."""

DREAM_LAB_SYSTEM = (
    DREAM_SYSTEM
    + """

In this session you are updating the lab memory under /memories/lab/, which is shared by every \
paper this pipeline writes. The project memory (everything else under /memories/) is read-only \
here and is your evidence.

Only process knowledge belongs in the lab memory: LaTeX and template pitfalls with their fixes, \
venue formatting rules that compile or format feedback confirmed, figure-code pitfalls, and \
which kinds of revision raised or lowered review scores. Write each lesson so that it would \
help with a different paper. Leave out anything that describes or identifies this paper: its \
idea, method, results, claims, title, wording or figures. Papers must be written only from \
their own materials, so content that leaks between papers would corrupt later runs. Update an \
existing lesson rather than adding a near-duplicate, and delete lessons the new evidence \
contradicts."""
)


def memory_block(
    *,
    index: str,
    lab_index: str | None,
    pull: bool,
    write: str,
) -> str:
    """The block appended to an agent's system prompt when memory is readable."""
    parts = [MEMORY_PURPOSE, ""]
    if pull:
        parts.append(LAYOUT_PULL)
        if lab_index is not None:
            parts.append(LAYOUT_LAB)
        # A session that may edit everything (consolidation) gets its rules from its own
        # system prompt rather than from this block.
        if write == "notes":
            parts += ["", WRITE_NOTES]
        elif write == "none":
            parts += ["", WRITE_NONE]
    else:
        parts.append(LAYOUT_PUSH)
    parts += ["", "<memory_index>", index.strip(), "</memory_index>"]
    if lab_index is not None:
        parts += ["", "<lab_memory_index>", lab_index.strip(), "</lab_memory_index>"]
    return "\n".join(parts)


def schema_instructions(schema_json: str) -> str:
    """Appended to a prompt when native structured output is unavailable."""
    return (
        "Return your answer as a single JSON object inside a ```json fenced block. It must "
        "conform to this JSON Schema:\n```json\n" + schema_json + "\n```"
    )
