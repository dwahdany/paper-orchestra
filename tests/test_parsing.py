from __future__ import annotations

import pytest

from paper_orchestra.llm.parsing import (
    parse_json,
    parse_json_and_latex,
    parse_json_object,
    parse_latex,
    parse_python,
)

DOC = "\\documentclass{article}\n\\begin{document}\nHi\n\\end{document}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Thought...\n```JSON\n{"a": [1, 2,],}\n```\ntrailing', {"a": [1, 2]}),
        ('prefix {"path": "C:\\dir", "b": 2} suffix', {"path": "C:\\dir", "b": 2}),
        ("Here: [1, 2, 3] done", [1, 2, 3]),
        ('{"outer": {"inner": [1, {"x": "}"}]}}', {"outer": {"inner": [1, {"x": "}"}]}}),
        ('```json\nnot json\n```\n```json\n{"ok": true}\n```', {"ok": True}),
    ],
)
def test_parse_json_variants(text, expected):
    assert parse_json(text) == expected


@pytest.mark.parametrize("text", [None, "", "no json here", 42])
def test_parse_json_returns_none_when_hopeless(text):
    assert parse_json(text) is None


def test_parse_json_object_rejects_arrays():
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_latex_prefers_longest_fenced_block():
    text = f"A snippet:\n```latex\n\\cite{{x}}\n```\nFull file:\n```latex\n{DOC}\n```\n"
    assert parse_latex(text) == DOC + "\n"


def test_parse_latex_accepts_unfenced_document_and_rejects_prose():
    assert parse_latex(DOC) == DOC + "\n"
    assert parse_latex(f"```\n{DOC}\n```") == DOC + "\n"
    assert parse_latex("I could not write the paper.") is None
    assert parse_latex("") is None


def test_parse_json_and_latex():
    text = f'```json\n{{"actions_taken": ["x"]}}\n```\n```latex\n{DOC}\n```'
    assert parse_json_and_latex(text) == {"json": {"actions_taken": ["x"]}, "latex": DOC + "\n"}
    assert parse_json_and_latex(f"```latex\n{DOC}\n```")["json"] is None
    assert parse_json_and_latex('```json\n{"a": 1}\n```') is None


def test_parse_python():
    assert parse_python("```python\nimport x\n```") == "import x\n"
    assert parse_python("```\nimport y\n```") == "import y\n"
    assert parse_python("import z") == "import z\n"
    assert parse_python("  ") is None
