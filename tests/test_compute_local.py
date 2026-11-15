from __future__ import annotations

import shutil

import pytest

from paper_orchestra.compute.base import CompileResult, ComputeError
from paper_orchestra.compute.local import LocalBackend, safe_relative, scrubbed_env
from paper_orchestra.utils.latex import (
    bib_keys,
    cite_keys,
    latex_commands,
    parse_latex_log,
    static_check,
)


@pytest.fixture
def backend() -> LocalBackend:
    return LocalBackend()


def test_run_python_isolated_environment_and_io(backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("PYTHONSTARTUP", "/nonexistent")
    result = backend.run_python(
        "import os\n"
        "print(os.environ.get('ANTHROPIC_API_KEY'))\n"
        "print(open('data/in.txt').read())\n"
        "open('out.txt', 'w').write('done')\n",
        inputs={"data/in.txt": b"payload"},
        outputs=["out.txt", "missing.txt"],
    )
    assert result.ok
    assert result.stdout.split() == ["None", "payload"]
    assert result.files == {"out.txt": b"done"}


def test_run_python_reports_failure_and_timeout(backend):
    failed = backend.run_python("raise ValueError('boom')")
    assert not failed.ok and failed.returncode != 0 and "boom" in failed.stderr
    slow = backend.run_python("import time\ntime.sleep(30)", timeout=1)
    assert slow.timed_out and not slow.ok


def test_run_python_renders_a_matplotlib_figure(backend):
    result = backend.run_python(
        "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.savefig('figure.png')\n",
        outputs=["figure.png"],
        timeout=120,
    )
    assert result.ok, result.stderr
    assert result.files["figure.png"].startswith(b"\x89PNG")


@pytest.mark.parametrize("name", ["/abs/path", "../up.txt", "a/../../b", ""])
def test_unsafe_file_names_are_rejected(backend, name):
    with pytest.raises(ComputeError):
        safe_relative(name)
    with pytest.raises(ComputeError):
        backend.run_python("pass", inputs={name: b"x"})


def test_scrubbed_env_has_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    env = scrubbed_env(tmp_path)
    assert "OPENAI_API_KEY" not in env and env["HOME"] == str(tmp_path) and "PATH" in env


def test_latex_commands_disable_shell_escape():
    commands = latex_commands("paper")
    assert [c[0] for c in commands] == ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    assert all("-no-shell-escape" in c for c in commands if c[0] == "pdflatex")


def test_compile_missing_tex_raises(backend, tmp_path):
    with pytest.raises(ComputeError):
        backend.compile_latex(tmp_path, "nope")


@pytest.mark.latex
@pytest.mark.parametrize("template", ["cvpr2025", "iclr2025"])
def test_bundled_templates_compile(backend, tmp_path, repo_root, template):
    shutil.copytree(repo_root / "templates" / template, tmp_path, dirs_exist_ok=True)
    result = backend.compile_latex(tmp_path, "template", output_pdf=tmp_path / "out" / "paper.pdf")
    assert result.ok and result.pdf_path == tmp_path / "out" / "paper.pdf"
    assert result.pdf_path.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "template.pdf").exists()


@pytest.mark.latex
def test_compile_reports_errors_with_line_context(backend, tmp_path):
    (tmp_path / "bad.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\undefinedmacro text \\cite{Missing2024} \\ref{sec:none}\n\\end{document}\n"
    )
    result = backend.compile_latex(tmp_path, "bad")
    assert result.ok  # nonstopmode still yields a PDF
    assert any("Undefined control sequence" in e and "l.3" in e for e in result.errors)
    assert "undefined citation 'Missing2024'" in result.warnings
    assert "undefined reference 'sec:none'" in result.warnings
    assert "Undefined control sequence" in result.feedback()


@pytest.mark.latex
def test_shell_escape_is_blocked_and_fatal_errors_fail(backend, tmp_path):
    (tmp_path / "x.tex").write_text(
        "\\documentclass{article}\\begin{document}"
        "\\immediate\\write18{touch pwned}hi\\end{document}"
    )
    backend.compile_latex(tmp_path, "x")
    assert not (tmp_path / "pwned").exists()
    (tmp_path / "y.tex").write_text("\\documentclass{article}\\begin{document}\\input{nofile}")
    result = backend.compile_latex(tmp_path, "y")
    assert not result.ok and result.pdf_path is None and result.errors


def test_static_check_catches_common_model_mistakes(tmp_path):
    (tmp_path / "references.bib").write_text(
        "@article{Known2020,\n title={T}\n}\n@string{x = {y}}\n"
    )
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "ok.png").write_bytes(b"\x89PNG")
    tex = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\begin{figure*}\\includegraphics[width=1\\linewidth]{figures/ok}\\end{figure}\n"
        "\\includegraphics{figures/missing.png}\n"
        "\\citep{Known2020, Unknown2021} % \\cite{Commented2022}\n"
        "Introduction here.\n"
        "\\begin{verbatim}\\begin{figure}\\end{verbatim}\n"
    )
    problems = static_check(tex, workdir=tmp_path)
    text = "\n".join(problems)
    assert "missing \\end{document}" in text
    assert "environment 'figure*': 1 \\begin vs 0 \\end" in text
    assert "environment 'figure': 0 \\begin vs 1 \\end" in text
    assert "Unknown2021" in text and "Known2020" not in text and "Commented2022" not in text
    assert "figures/missing.png" in text and "figures/ok" not in text.replace("figures/ok.png", "")
    assert "template placeholder still present: 'Introduction here.'" in text
    assert cite_keys(tex) == {"Known2020", "Unknown2021"}
    assert bib_keys((tmp_path / "references.bib").read_text()) == {"Known2020"}


def test_static_check_clean_document():
    tex = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n"
    assert static_check(tex) == []


def test_parse_latex_log_dedupes_and_flags_overfull():
    log = (
        "! Undefined control sequence.\nl.12 \\foo\n"
        "! Undefined control sequence.\nl.12 \\foo\n"
        "LaTeX Warning: Citation `A' on page 1 undefined on input line 3.\n"
        "LaTeX Warning: Citation `A' on page 2 undefined on input line 9.\n"
        "Overfull \\hbox (80.5pt too wide) in paragraph at lines 5--6\n"
        "Overfull \\hbox (2.0pt too wide) in paragraph at lines 8--9\n"
    )
    errors, warnings = parse_latex_log(log)
    assert errors == ["Undefined control sequence. [l.12 \\foo]"]
    assert warnings == [
        "undefined citation 'A'",
        "overfull hbox up to 80pt (content spills into the margin)",
    ]


def test_compile_result_feedback_falls_back_to_log_tail():
    result = CompileResult(ok=False, log_tail="fatal: emergency stop")
    assert "emergency stop" in result.feedback()
    assert "timed out" in CompileResult(ok=False, timed_out=True).feedback()
