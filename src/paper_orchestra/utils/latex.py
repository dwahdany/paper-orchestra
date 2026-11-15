"""LaTeX helpers shared by the compute backends and the writing agents.

* the command sequence used to build a paper,
* a log parser that turns pdflatex/bibtex output into short, actionable
  error and warning lists, and
* static checks that catch the usual model mistakes (unknown citation keys,
  missing figure files, unbalanced environments, leftover template
  placeholders) before and after compiling.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

GRAPHICS_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")

_COMMENT = re.compile(r"(?<!\\)%.*")
_CITE = re.compile(r"\\[cC]ite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
_BIB_ENTRY = re.compile(r"@\s*([a-zA-Z]+)\s*\{\s*([^,\s]+)\s*,")
_INCLUDEGRAPHICS = re.compile(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]*)\}")
_BEGIN = re.compile(r"\\begin\{([^}]+)\}")
_END = re.compile(r"\\end\{([^}]+)\}")
_VERBATIM = re.compile(r"\\begin\{(verbatim\*?|lstlisting|minted)\}.*?\\end\{\1\}", re.DOTALL)

_UNDEFINED_CITE = re.compile(r"Citation [`']([^']+)' (?:on page \d+ )?undefined")
_UNDEFINED_REF = re.compile(r"Reference [`']([^']+)' on page \d+ undefined")
_MISSING_FILE = re.compile(r"File [`']([^']+)' not found")
_OVERFULL = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\)")
_FILE_LINE_ERROR = re.compile(r"^(?:\./)?[^\s:]+\.(?:tex|sty|cls|bib|bst|bbl):\d+: ")
#: pdflatex hard-wraps its log at this many characters.
_LOG_WRAP = 79

#: TeX primitives and macros that read or write files, run code, or build control
#: sequence names at run time (which would let a document hide any of the others).
#: ``openin_any``/``openout_any`` do not cover pdfTeX's ``\pdfobj ... file``, which
#: embeds an arbitrary file in the PDF, so documents are also screened statically.
_FORBIDDEN_COMMANDS = (
    "pdfobj",
    "pdffiledump",
    "pdfmdfivesum",
    "pdffilesize",
    "pdffilemoddate",
    "pdfximage",
    "openin",
    "openout",
    "read",
    "readline",
    "write",
    "immediate",
    "directlua",
    "latelua",
    "luaexec",
    "special",
    "csname",
    "begincsname",
    "catcode",
    "scantokens",
    "@nameuse",
    "@namedef",
    "@input",
    "@@input",
    "UseName",
    "NameUse",
    "ExplSyntaxOn",
    "ShellEscape",
    "CatchFileDef",
    "CatchFileEdef",
    "embedfile",
    "attachfile",
    "textattachfile",
    "includepdf",
    "verbatiminput",
    "lstinputlisting",
    "inputminted",
)
_FORBIDDEN_COMMAND = re.compile(
    r"\\(" + "|".join(re.escape(name) for name in _FORBIDDEN_COMMANDS) + r")(?![A-Za-z@])"
)
#: Packages whose whole purpose is embedding or reading external files.
_FORBIDDEN_PACKAGES = frozenset(
    {"embedfile", "attachfile", "attachfile2", "navigator", "catchfile", "shellesc",
     "pdfpages", "bashful", "pythontex", "minted", "write18", "filecontents"}
)  # fmt: skip
_USEPACKAGE = re.compile(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
#: Commands whose argument names a file the document pulls in.
_FILE_ARGUMENT = re.compile(
    r"\\(input|include|InputIfFileExists|includegraphics|bibliography|addbibresource|"
    r"subfile|subimport|import)\*?\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}"
)
_BARE_INPUT = re.compile(r"\\input\s+([^\s{\\]+)")

#: Placeholder bodies shipped in the bundled templates.  If one survives into a
#: draft, the corresponding section was never written.
TEMPLATE_PLACEHOLDERS = (
    "Abstract here.",
    "Introduction here.",
    "Related work here.",
    "Methodology here.",
    "Experimental setup, results, and analysis here.",
    "Conclusion here.",
    "My Title",
)


def latex_commands(tex_name: str) -> list[list[str]]:
    """pdflatex, bibtex, pdflatex, pdflatex.  Shell escape is always off."""
    pdflatex = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-no-shell-escape",
        "-file-line-error",
        f"{tex_name}.tex",
    ]
    return [pdflatex, ["bibtex", tex_name], pdflatex, pdflatex]


def strip_comments(tex: str) -> str:
    return _COMMENT.sub("", tex)


def cite_keys(tex: str) -> set[str]:
    """Every key used in a ``\\cite``-family command (comments ignored)."""
    keys: set[str] = set()
    for match in _CITE.finditer(strip_comments(tex)):
        keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    keys.discard("*")
    return keys


def bib_keys(bib_text: str) -> set[str]:
    return {
        key
        for entry_type, key in _BIB_ENTRY.findall(bib_text)
        if entry_type.lower() not in ("comment", "string", "preamble")
    }


def graphics_paths(tex: str) -> list[str]:
    return [m.group(1).strip() for m in _INCLUDEGRAPHICS.finditer(strip_comments(tex))]


def resolve_graphic(workdir: Path, name: str) -> Path | None:
    """Find the file an ``\\includegraphics`` argument refers to, if it exists."""
    candidate = workdir / name
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        for ext in GRAPHICS_EXTENSIONS:
            with_ext = candidate.with_name(candidate.name + ext)
            if with_ext.is_file():
                return with_ext
    return None


def _outside_workspace(name: str) -> bool:
    name = name.strip()
    return (
        name.startswith(("/", "~", "|", "\\"))
        or ".." in name.replace("\\", "/").split("/")
        or re.match(r"^[A-Za-z]:[\\/]", name) is not None
    )


def unsafe_constructs(tex: str) -> list[str]:
    """Reasons a model-written document must not be compiled as it is (empty = fine).

    A paper needs none of these: commands that read or write files other than its
    own figures, bibliography and inputs inside the workspace, commands that run
    code, and commands that assemble control sequence names at run time.  The
    screen is deliberately conservative; it is a second line of defence next to
    running TeX without credentials, not a sandbox.
    """
    body = strip_comments(tex)
    problems: list[str] = []
    for name in dict.fromkeys(_FORBIDDEN_COMMAND.findall(body)):
        problems.append(
            f"\\{name} is not allowed in the paper source (it can read or write files, "
            "run code or hide other commands); remove it"
        )
    for packages in _USEPACKAGE.findall(body):
        for package in (p.strip() for p in packages.split(",")):
            if package in _FORBIDDEN_PACKAGES:
                problems.append(f"the package '{package}' is not allowed; remove it")
    arguments = [m.group(2) for m in _FILE_ARGUMENT.finditer(body)]
    arguments += _BARE_INPUT.findall(body)
    for argument in dict.fromkeys(arguments):
        for name in argument.split(","):
            if _outside_workspace(name):
                problems.append(
                    f"'{name.strip()}' refers to a file outside the paper's own directory; "
                    "only files inside it may be used"
                )
    return problems


def unbalanced_environments(tex: str) -> list[str]:
    """Environments whose ``\\begin`` and ``\\end`` counts differ."""
    body = _VERBATIM.sub("", strip_comments(tex))
    begins = Counter(_BEGIN.findall(body))
    ends = Counter(_END.findall(body))
    problems = []
    for env in sorted(set(begins) | set(ends)):
        if begins[env] != ends[env]:
            problems.append(f"environment '{env}': {begins[env]} \\begin vs {ends[env]} \\end")
    return problems


def static_check(
    tex: str,
    *,
    workdir: str | Path | None = None,
    allowed_citations: set[str] | None = None,
    check_placeholders: bool = True,
) -> list[str]:
    """Cheap structural checks.  Returns human-readable problems (empty = clean).

    ``allowed_citations`` defaults to the keys of ``<workdir>/references.bib``
    when that file exists.
    """
    problems: list[str] = unsafe_constructs(tex)
    stripped = strip_comments(tex)
    if "\\documentclass" not in stripped:
        problems.append("missing \\documentclass")
    if "\\begin{document}" not in stripped:
        problems.append("missing \\begin{document}")
    if "\\end{document}" not in stripped:
        problems.append("missing \\end{document} (the document looks truncated)")
    problems.extend(unbalanced_environments(tex))

    work = Path(workdir) if workdir is not None else None
    if allowed_citations is None and work is not None:
        bib = work / "references.bib"
        if bib.is_file():
            allowed_citations = bib_keys(bib.read_text(encoding="utf-8", errors="replace"))
    if allowed_citations is not None:
        unknown = sorted(cite_keys(tex) - allowed_citations)
        if unknown:
            shown = ", ".join(unknown[:25]) + (" ..." if len(unknown) > 25 else "")
            problems.append(f"{len(unknown)} citation key(s) are not in the bibliography: {shown}")

    if work is not None:
        missing = sorted({g for g in graphics_paths(tex) if resolve_graphic(work, g) is None})
        if missing:
            problems.append("figure file(s) not found: " + ", ".join(missing[:25]))

    if check_placeholders:
        for placeholder in TEMPLATE_PLACEHOLDERS:
            if placeholder in stripped:
                problems.append(f"template placeholder still present: '{placeholder}'")
    return problems


def _unwrap(log_text: str) -> list[str]:
    """Split a pdflatex log into logical lines, undoing its hard wrap at 79 columns."""
    lines: list[str] = []
    carry = ""
    for raw in log_text.splitlines():
        line = carry + raw
        # A line of exactly the wrap width continues on the next one.
        if len(raw) == _LOG_WRAP and not raw.startswith("$ "):
            carry = line
            continue
        carry = ""
        lines.append(line)
    if carry:
        lines.append(carry)
    return lines


def summarize_build(outputs: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` for a whole build, given ``(program, output)`` per step.

    Errors are collected from every step.  Warnings come from the last pdflatex
    pass only: citations and references are always "undefined" on the first
    pass, and only what is still unresolved at the end is a real problem.
    """
    errors: list[str] = []
    for _, text in outputs:
        for error in parse_latex_log(text)[0]:
            if error not in errors:
                errors.append(error)
    final = next((text for program, text in reversed(outputs) if program == "pdflatex"), "")
    return errors, parse_latex_log(final)[1]


def parse_latex_log(
    log_text: str, *, overfull_threshold_pt: float = 25.0
) -> tuple[list[str], list[str]]:
    """Extract ``(errors, warnings)`` from pdflatex / bibtex output.

    Errors are ``! ...`` blocks and ``file:line: message`` lines, each with the
    offending source line when LaTeX printed one.  Warnings cover undefined
    citations and references, missing files and badly overfull boxes; they are
    de-duplicated.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_errors: set[str] = set()
    seen_warnings: set[str] = set()

    def add_error(message: str) -> None:
        message = " ".join(message.split())
        if message and message not in seen_errors:
            seen_errors.add(message)
            errors.append(message)

    def add_warning(message: str) -> None:
        if message not in seen_warnings:
            seen_warnings.add(message)
            warnings.append(message)

    lines = _unwrap(log_text)
    joined = "\n".join(lines)
    for index, line in enumerate(lines):
        is_bang = line.startswith("! ")
        if is_bang or _FILE_LINE_ERROR.match(line):
            message = line[2:] if is_bang else line
            context = ""
            for follow in lines[index + 1 : index + 8]:
                if follow.startswith("l.") and follow[2:3].isdigit():
                    context = follow
                    break
            add_error(f"{message} [{context}]" if context else message)
        elif line.startswith("I found no "):
            # BibTeX saying the document has no citations, bibliography or style.  A paper
            # that cites nothing is legitimate, and when citations were expected the
            # undefined-citation warnings of the final pass report it.
            continue
        elif line.startswith("I couldn't open") or ("---line " in line and " of file " in line):
            add_error(f"bibtex: {line}")

    for key in _UNDEFINED_CITE.findall(joined):
        add_warning(f"undefined citation '{key}'")
    for key in _UNDEFINED_REF.findall(joined):
        add_warning(f"undefined reference '{key}'")
    for name in _MISSING_FILE.findall(joined):
        add_warning(f"file not found: {name}")
    worst = max((float(v) for v in _OVERFULL.findall(joined)), default=0.0)
    if worst >= overfull_threshold_pt:
        add_warning(f"overfull hbox up to {worst:.0f}pt (content spills into the margin)")
    return errors, warnings
