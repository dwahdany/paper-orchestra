"""Keeping one paper's content out of the memory shared between papers.

The lab memory holds lessons about how to do the work.  Before a lesson may
enter it, :class:`LeakScanner` checks mechanically that it carries nothing of
the paper it was learned on: no result-like number, no distinctive name and no
run of wording from that paper's materials.

Two decisions shape the scan.

* **What is the paper, and what is the venue.**  The paper is everything the
  run was given and everything it wrote: inputs, outline, captions, reviews,
  and every file in its workspace.  The venue is what the template directory
  ships, recognised by content, never by file name: once a stage has written
  into the workspace's ``template.tex``, that file is the paper.  Lessons may
  quote venue material freely.
* **What is content, and what is form.**  LaTeX command names and the
  pipeline's own record formats (JSON keys, the outcome line of a review
  record, compile diagnostics) are not the paper's words, although they appear
  in its files.  Lessons quote them all the time.

The scan is a filter on form, not on meaning.  It stops the copies that happen
in practice, a lesson that carries its evidence along; it cannot stop a
paraphrase that shares no name, number or phrase with the paper.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from paper_orchestra.memory.wiki import Wiki, tokenize

#: Sections of the project wiki that hold the paper's own content.  (The literature
#: section is public bibliographic data; draft pages only point at files, which are
#: scanned from disk.)
CONTENT_SECTIONS = ("wiki/sources/", "wiki/paper/", "wiki/figures/", "wiki/reviews/")
#: Source pages that hold the venue's template and guidelines as shipped.
VENUE_SOURCE_PREFIXES = ("wiki/sources/guidelines", "wiki/sources/template")
_TEXT_SUFFIXES = frozenset({".md", ".txt", ".tex", ".json", ".csv", ".tsv", ".yaml", ".yml"})
_VENUE_SUFFIXES = frozenset({".tex", ".md", ".txt", ".sty", ".cls", ".bst"})
MAX_PROTECTED_FILE_BYTES = 2_000_000
SHINGLE_TOKENS = 5

_TEX_UNIT = r"(?:pt|em|ex|cm|mm|in|bp|pc|sp|px|dpi)\b"
#: A number in any of the spellings results are written in: with thousands separators,
#: without a leading zero, followed by a unit or a subscript (``2.3x``, ``110M``,
#: ``87.3_{\pm 0.2}``).  Layout is not a result: a TeX length (``12.3pt``) and the factor
#: of a length command (``0.48\textwidth``) are left out.
_NUMBER = re.compile(
    r"(?<![\w.])(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?!\d|\.\d)"
    rf"(?!{_TEX_UNIT})"
    r"(?!\s*\\(?:textwidth|linewidth|columnwidth|textheight|paperwidth|paperheight|hsize|vsize)\b)"
)
#: A decimal comma (``43,43``), which the pattern above reads as two list items.
_DECIMAL_COMMA = re.compile(r"(?<![\w.,])(\d+),(\d{1,2})(?![\w,]|\.\d)")
_YEAR = re.compile(r"(?:19|20)\d\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HOME_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)[\w.-]+")
_CREDENTIAL = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}|(?i:\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*\S{6,})"
)
_CONTROL = re.compile(r"\\[A-Za-z@]+\*?")
#: Words with a capital after the first letter and at least one lower-case letter (FooNet).
_CAMEL = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*[a-z])[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?![A-Za-z0-9])"
)
#: Upper-case names of three or more characters, with optional suffixes (CIFAR-10).
_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*(?![A-Za-z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_LOWER_WORD = re.compile(r"(?<![A-Za-z0-9])[a-z]{2,}(?![A-Za-z0-9])")
#: Identifiers, paths and slugs: tokens joined by ``_ : / . -`` (``fig:tsam_overview``).
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_:/.-][A-Za-z0-9]+)+")
_IDENTIFIER_PART = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_TITLE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{([^{}]*)")
#: "We propose Falcon", "we call our method Falcon": where a paper names what it built.
_COINED = re.compile(
    r"\b(?:propose|introduce|present|call|name|term|dub)s?\s+"
    r"(?:(?:it|our|the|this|a|an|new|novel|method|model|approach|framework|system)\s+){0,4}"
    r"([A-Z][A-Za-z0-9]{2,})"
)
_JSON_KEY = re.compile(r'"[^"\n]{1,60}"\s*:')
#: Fields of the pipeline's records that hold diagnostics and outcomes, not the paper's prose.
_RECORD_FIELDS = frozenset({"problems", "outcome", "error", "errors", "warnings", "status"})

#: Names that say nothing about a particular paper: tools, formats, venues and the common
#: vocabulary of the field.  Any other distinctive name found in a paper's materials is
#: treated as identifying that paper.
COMMON_TERMS = frozenset(
    """
    TeX LaTeX BibTeX BibLaTeX pdfTeX pdfLaTeX XeTeX XeLaTeX LuaTeX LuaLaTeX TikZ PGFPlots PGF
    CTAN OT1 UTF PDF DVI EPS PNG JPG JPEG SVG GIF DPI RGB CMYK URL DOI ISBN ISSN HTML XML JSON
    YAML CSV TSV API CLI CPU GPU RAM OOM ASCII arXiv OpenReview GitHub ORCID
    IEEE ACM CVPR ICCV ECCV WACV ICLR ICML NeurIPS NIPS AAAI IJCAI ACL EMNLP NAACL COLING
    KDD SIGIR SIGGRAPH TMLR JMLR PMLR TPAMI IJCV
    NumPy SciPy PyTorch TensorFlow JAX CUDA OpenCV
    SOTA MLP CNN RNN LSTM GRU GAN VAE SGD NLP LLM AUC ROC MSE MAE RMSE BLEU ROUGE
    TODO FIXME NOTE WARNING ERROR EVIDENCE CONFLICT RULE WHY
    """.split()  # noqa: SIM905
)
_COMMON_LOWER = frozenset(term.lower() for term in COMMON_TERMS)
#: Capitalised words that follow "we present ..." without being a name.
_NOT_COINED = frozenset(
    [
        "Table",
        "Tables",
        "Figure",
        "Figures",
        "Fig",
        "Section",
        "Sections",
        "Appendix",
        "Algorithm",
        "Equation",
        "Eq",
        "Theorem",
        "Lemma",
        "Proposition",
        "Corollary",
        "Definition",
    ]
)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _canonical(raw: str) -> str:
    """``2,908`` -> ``2908``, ``87.30`` -> ``87.3``, ``.568`` -> ``0.568``, ``007`` -> ``7``."""
    whole, _, fraction = raw.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _readings(raw: str) -> list[str]:
    """Canonical forms a matched number can stand for.

    ``12,345`` is either one number with a thousands separator or two list
    items, so both readings are returned, the joined one first.
    """
    readings = [_canonical(raw)]
    if "," in raw:
        whole, _, fraction = raw.partition(".")
        parts = whole.split(",")
        if fraction:
            parts[-1] = f"{parts[-1]}.{fraction}"
        readings.extend(_canonical(part) for part in parts)
    return readings


def numbers_in(text: str) -> set[str]:
    """Every number in ``text`` in canonical form, under each possible reading."""
    found: set[str] = set()
    for raw in _NUMBER.findall(text):
        found.update(_readings(raw))
    return found


def _digits(number: str) -> int:
    """How many significant digits a canonical number has."""
    whole, _, fraction = number.partition(".")
    return len((whole + fraction).lstrip("0"))


def _variants(number: str) -> set[str]:
    """Other ways the same result gets written: as a percentage or as a fraction, and
    rounded to fewer decimals.  Only for numbers that stay specific (three or more
    significant digits), so that coincidences remain rare."""
    found: set[str] = set()
    try:
        value = Decimal(number)
    except InvalidOperation:
        return found
    for scaled in (value * 100, value / 100):
        text = _canonical(format(scaled, "f"))
        if _digits(text) >= 3 and _digits(number) >= 3:
            found.add(text)
    _, _, fraction = number.partition(".")
    for places in range(len(fraction)):
        rounded = _canonical(format(value.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP), "f"))
        if _digits(rounded) >= 3:
            found.add(rounded)
    found.discard(number)
    return found


def _significant(number: str) -> bool:
    """Numbers specific enough that a wrong one matters: decimals with three or more
    digits, or integers of four or more digits that are not calendar years."""
    whole, _, fraction = number.partition(".")
    if fraction:
        return len(whole + fraction) >= 3
    return len(whole) >= 4 and not _YEAR.fullmatch(whole)


def _identifying(number: str) -> bool:
    """Numbers that could identify a result: at least two significant digits, and for
    integers at least three digits that are not a round figure or a calendar year."""
    whole, _, fraction = number.partition(".")
    if fraction:
        return len((whole + fraction).lstrip("0")) >= 2
    return len(whole) >= 3 and len(whole.rstrip("0")) >= 2 and not _YEAR.fullmatch(whole)


# ---------------------------------------------------------------------------
# Names and wording
# ---------------------------------------------------------------------------


def _shingles(text: str) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    return {tuple(tokens[i : i + SHINGLE_TOKENS]) for i in range(len(tokens) - SHINGLE_TOKENS + 1)}


def _is_common(name: str) -> bool:
    """Whether ``name`` is general vocabulary, also in the plural (``GPUs``, ``PDFs``)."""
    lowered = name.lower()
    return lowered in _COMMON_LOWER or (lowered.endswith("s") and lowered[:-1] in _COMMON_LOWER)


def _ordinary_words(text: str) -> set[str]:
    """Words the text writes in lower case: evidence that they are words, not names."""
    return set(_LOWER_WORD.findall(_CONTROL.sub(" ", text)))


@dataclass
class _Names:
    #: Long mixed-case names, lower-cased: matched whatever the case (FooNet, foonet).
    folded: set[str] = field(default_factory=set)
    #: Names matched only in their own spelling: short or word-like mixed-case names (mAP,
    #: which must not make "map" a name) and plain capitalised names a paper coins (Falcon).
    exact: set[str] = field(default_factory=set)
    #: Upper-case names (TSAM), and the part before a suffix (CIFAR for CIFAR-10).
    upper: set[str] = field(default_factory=set)

    def minus(self, other: _Names) -> _Names:
        return _Names(
            self.folded - other.folded, self.exact - other.exact, self.upper - other.upper
        )


def _names(text: str, ordinary: set[str]) -> _Names:
    """Distinctive names in ``text``.  LaTeX command names are not names.

    ``ordinary`` holds the words written in lower case somewhere in the
    materials.  An upper-case word that is also used in lower case is an
    ordinary word written loudly (a heading, ``NOT``), not a name.
    """
    plain = _CONTROL.sub(" ", text)
    names = _Names()
    for match in _CAMEL.findall(plain):
        if _is_common(match):
            continue
        if sum(char.isalpha() for char in match) <= 4 or match.lower() in ordinary:
            names.exact.add(match)
        else:
            names.folded.add(match.lower())
    for match in _ACRONYM.findall(plain):
        for name in {match, match.split("-")[0]}:
            if len(name) >= 3 and not _is_common(name) and name.lower() not in ordinary:
                names.upper.add(name)
    coined = [match.group(1) for match in _COINED.finditer(plain)]
    for title in _TITLE.findall(text):
        head, colon, _ = title.partition(":")
        if colon and len(head.split()) <= 3:
            coined.extend(_WORD.findall(_CONTROL.sub(" ", head)))
    for name in coined:
        if (
            name[0].isupper()
            and name not in _NOT_COINED
            and not _is_common(name)
            and name.lower() not in ordinary
        ):
            names.exact.add(name)
    return names


def _wording(label: str, text: str) -> str:
    """The part of a protected text that is the paper's wording.

    The pipeline's record formats are left out: a lesson that quotes "Overall
    5 -> 6; sub-axis gain 1, drop 0" or a compile error repeats the pipeline,
    not the paper.
    """
    if label.startswith("wiki/reviews/"):
        return _JSON_KEY.sub(" ", text.split("\n## Outcome", 1)[0])
    if label.endswith(".json"):
        try:
            data = json.loads(text)
        except ValueError:
            return _JSON_KEY.sub(" ", text)
        strings: list[str] = []

        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for name, item in value.items():
                    walk(item, str(name))
            elif isinstance(value, list):
                for item in value:
                    walk(item, key)
            elif isinstance(value, str) and key not in _RECORD_FIELDS:
                strings.append(value)

        walk(data)
        return "\n".join(strings)
    return text


class LeakScanner:
    """Decides mechanically whether a text may enter the memory shared between papers.

    ``sources`` are the paper's own materials and ``allowed`` the venue's (the
    template and guidelines, which lessons legitimately quote).  ``wording``
    is the text to compare wording against when it differs from ``sources``
    (see :func:`paper_materials`).  A text is turned down when it contains an
    e-mail address, a home-directory path or a credential, or when it repeats
    something from the paper's materials: a result-like number, a distinctive
    name (a method, dataset or system), or a run of wording.
    """

    def __init__(
        self,
        sources: Sequence[str],
        allowed: Sequence[str] = (),
        *,
        wording: Sequence[str] | None = None,
    ) -> None:
        corpus = "\n".join(sources)
        permitted = "\n".join(allowed)
        self.has_sources = bool(corpus.strip())
        venue_numbers = numbers_in(permitted)
        self._numbers = {n for n in numbers_in(corpus) - venue_numbers if _identifying(n)}
        #: The same results as percentages or fractions, and rounded: not in the materials
        #: in that spelling, but the paper's all the same.
        self._number_variants = (
            {variant for number in self._numbers for variant in _variants(number)}
            - self._numbers
            - venue_numbers
        )
        ordinary = _ordinary_words(corpus) | _ordinary_words(permitted)
        self._names = _names(corpus, ordinary).minus(_names(permitted, ordinary))
        #: Upper-case names of four letters or more, lower-cased.  Such a name is matched in
        #: any case and inside identifiers (``tsam``, ``Tsam``, ``fig:tsam_overview``): its
        #: lower-case form is known not to be a word the materials use.
        self._upper_folded = {name.lower() for name in self._names.upper if len(name) >= 4}
        #: Long mixed-case names, to be found when written apart (``Prompt Weaver``,
        #: ``prompt_weaver``).
        self._long_folded = sorted(name for name in self._names.folded if len(name) >= 8)
        worded = corpus if wording is None else "\n".join(wording)
        self._shingles = _shingles(worded) - _shingles(permitted)

    @classmethod
    def from_materials(cls, materials: Materials) -> LeakScanner:
        return cls(
            [text for _, text in materials.content],
            materials.venue,
            wording=[_wording(label, text) for label, text in materials.content],
        )

    def _named(self, text: str) -> str | None:
        """A name from the paper's materials that ``text`` uses, in whatever spelling."""
        names = self._names
        scan = _CONTROL.sub(" ", text)
        for name in dict.fromkeys(_ACRONYM.findall(scan)):
            base = name.split("-")[0]
            if (
                name in names.upper
                or base in names.upper
                or base.rstrip("0123456789") in names.upper
            ):
                return name
        for word in dict.fromkeys(_WORD.findall(scan)):
            singular = word[:-1] if word.endswith("s") else word
            if {word.lower(), singular.lower()} & (names.folded | self._upper_folded):
                return word
            if word in names.exact or singular in names.exact or singular in names.upper:
                return word
        if self._long_folded:
            # Written apart, with an underscore or across a LaTeX command.
            squashed = re.sub(r"[^a-z0-9]+", "", scan.lower())
            for name in self._long_folded:
                if name in squashed:
                    return name
        return None

    def reason(self, text: str) -> str | None:
        """Why ``text`` must not enter the shared memory, or ``None`` if it may."""
        match = _EMAIL.search(text)
        if match:
            return f"it contains an e-mail address ({match.group(0)})"
        match = _HOME_PATH.search(text)
        if match:
            return f"it contains a personal file-system path ({match.group(0)})"
        if _CREDENTIAL.search(text):
            return "it contains something that looks like a credential"
        if not self.has_sources:
            return None
        spelled = [(raw, _readings(raw)) for raw in dict.fromkeys(_NUMBER.findall(text))]
        spelled += [
            (match.group(0), [_canonical(f"{match.group(1)}.{match.group(2)}")])
            for match in _DECIMAL_COMMA.finditer(text)
        ]
        for raw, readings in spelled:
            for number in readings:
                if number in self._numbers or number in self._number_variants:
                    return f"it repeats a number from this paper's materials ({raw})"
        name = self._named(text)
        if name:
            return f"it names something from this paper's materials ({name})"
        shared = _shingles(text) & self._shingles
        if shared:
            return (
                "it repeats wording from this paper's materials (\"" + " ".join(min(shared)) + '")'
            )
        return None


def leak_reason(text: str, sources: Sequence[str], allowed: Sequence[str] = ()) -> str | None:
    """One-off form of :class:`LeakScanner`: why ``text`` may not be shared, or ``None``."""
    return LeakScanner(sources, allowed).reason(text)


# ---------------------------------------------------------------------------
# What is the paper, and what is the venue
# ---------------------------------------------------------------------------


@dataclass
class Materials:
    #: ``(label, text)`` of everything that holds the paper.
    content: list[tuple[str, str]] = field(default_factory=list)
    #: Text the venue ships with its template; lessons may quote it.
    venue: list[str] = field(default_factory=list)


def protected_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file() and not file.is_symlink():
                    yield file


def _read_protected(file: Path) -> str | None:
    try:
        if file.stat().st_size > MAX_PROTECTED_FILE_BYTES:
            return None
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paper_materials(
    project: Wiki, paths: Sequence[Path] = (), venue_dirs: Sequence[Path] = ()
) -> Materials:
    """Sort a run's texts into the paper's and the venue's.

    The paper: the content pages of the project wiki (inputs, outline, figure
    captions, reviews) and every text file under ``paths``, the run's files
    and directories that the harness was told to protect.  The venue: the
    template and guidelines pages that ingest wrote from the template as
    shipped, and every text file in ``venue_dirs``.  A file under ``paths``
    whose text is identical to a shipped file is the venue's copy; any other,
    whatever its name, is the paper's.
    """
    materials = Materials()
    for info in project.list("wiki/sources"):
        if info.path.startswith(VENUE_SOURCE_PREFIXES):
            materials.venue.append(project.read_text(info.path))
    shipped: set[str] = set()
    # The paper's files win: a template directory that happens to contain the run (a
    # template given as ".", say) must not turn the paper into venue material.
    own = [path.resolve() for path in (*paths, project.root)]
    for file in protected_files(venue_dirs):
        resolved = file.resolve()
        if any(root == resolved or root in resolved.parents for root in own):
            continue
        if file.suffix.lower() in _VENUE_SUFFIXES:
            text = _read_protected(file)
            if text:
                materials.venue.append(text)
                shipped.add(_digest(text))
    for section in CONTENT_SECTIONS:
        for info in project.list(section.rstrip("/")):
            if not info.path.startswith(VENUE_SOURCE_PREFIXES):
                materials.content.append((info.path, project.read_text(info.path)))
    for file in protected_files(paths):
        if file.suffix.lower() in _TEXT_SUFFIXES:
            text = _read_protected(file)
            if text and _digest(text) not in shipped:
                materials.content.append((file.name, text))
    return materials


def content_sources(
    project: Wiki, paths: Sequence[Path] = (), venue_dirs: Sequence[Path] = ()
) -> list[str]:
    """Every text that holds the paper's own content (see :func:`paper_materials`)."""
    return [text for _, text in paper_materials(project, paths, venue_dirs).content]


def venue_materials(
    project: Wiki, paths: Sequence[Path] = (), venue_dirs: Sequence[Path] = ()
) -> list[str]:
    """The venue's template and guidelines as shipped (see :func:`paper_materials`)."""
    return paper_materials(project, paths, venue_dirs).venue
