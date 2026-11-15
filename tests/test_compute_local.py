from __future__ import annotations

import os
import shutil
import time

import pytest

from paper_orchestra.compute.base import CompileResult, ComputeError
from paper_orchestra.compute.local import LocalBackend, safe_relative, scrubbed_env
from paper_orchestra.utils.latex import (
    bib_keys,
    cite_keys,
    latex_commands,
    parse_latex_log,
    static_check,
    summarize_build,
    unsafe_constructs,
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
    shutil.copytree(
        repo_root / "src" / "paper_orchestra" / "templates" / template, tmp_path, dirs_exist_ok=True
    )
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
    assert len(result.errors) == 1  # the same error from three passes is reported once
    assert "undefined citation 'Missing2024'" in result.warnings
    assert "undefined reference 'sec:none'" in result.warnings
    assert "Undefined control sequence" in result.feedback()


@pytest.mark.latex
def test_resolved_citations_and_references_are_not_reported(backend, tmp_path):
    """First-pass 'undefined' warnings disappear once bibtex and the later passes resolve them."""
    (tmp_path / "references.bib").write_text(
        "@article{Known2020,\n title={A title},\n author={Doe, Jane},\n journal={J},\n year={2020}\n}\n"
    )
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\section{Intro}\\label{sec:intro}\n"
        "See \\cite{Known2020} and Section~\\ref{sec:intro}.\n"
        "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n"
    )
    result = backend.compile_latex(tmp_path, "paper")
    assert result.ok and result.errors == [] and result.warnings == []


@pytest.mark.latex
def test_paper_without_citations_builds_clean(backend, tmp_path):
    (tmp_path / "references.bib").write_text("")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nNo citations here.\n"
        "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n"
    )
    result = backend.compile_latex(tmp_path, "paper")
    assert result.ok and result.errors == []


UNSAFE_DOCUMENTS = {
    "pdfobj": "\\immediate\\pdfobj stream file {SECRET}",
    "filedump": "\\edef\\x{\\pdffiledump length 9 {SECRET}}\\x",
    "hidden name": "\\csname pdfobj\\endcsname stream file {SECRET}",
    "nameuse": "\\makeatletter\\@nameuse{pdfobj} stream file {SECRET}",
    "shell": "\\write18{cat SECRET}",
    "input absolute": "\\input{SECRET}",
    "input parent": "\\input{../secret.txt}",
    "bare input": "\\input ../secret.txt ",
    "home": "\\includegraphics{~/private.png}",
    "embedfile": "\\usepackage{embedfile}",
    "catcode": "\\catcode`\\!=0 !pdfobj stream file {SECRET}",
}


@pytest.mark.parametrize("name", sorted(UNSAFE_DOCUMENTS))
def test_documents_that_could_read_or_run_things_are_never_compiled(backend, tmp_path, name):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    work = tmp_path / "work"
    work.mkdir()
    body = UNSAFE_DOCUMENTS[name].replace("SECRET", str(secret))
    (work / "paper.tex").write_text(
        f"\\documentclass{{article}}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"
    )
    result = backend.compile_latex(work, "paper")
    assert not result.ok and result.pdf_path is None and result.errors
    assert not (work / "paper.pdf").exists() and not (work / "paper.log").exists()
    assert "not allowed" in result.feedback() or "outside the paper" in result.feedback()


def test_unsafe_constructs_does_not_flag_ordinary_papers(repo_root):
    for template in ("cvpr2025", "iclr2025"):
        tex = (
            repo_root / "src" / "paper_orchestra" / "templates" / template / "template.tex"
        ).read_text()
        assert unsafe_constructs(tex) == []
    ordinary = (
        "\\documentclass{article}\\usepackage{graphicx,booktabs,amsmath}\n"
        "\\newcommand{\\readout}{r}\\newcommand{\\writer}{w} % \\pdfobj in a comment\n"
        "\\begin{document}\\input{sections/intro}\\includegraphics[width=1in]{figures/a.png}\n"
        "\\readout \\writer \\bibliography{references}\\end{document}\n"
    )
    assert unsafe_constructs(ordinary) == []
    assert static_check("\\documentclass{a}\\begin{document}\\write18{ls}\\end{document}")


@pytest.mark.latex
def test_tex_home_is_private_and_removed_on_close(backend, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "private.tex").write_text("HOMESECRET")
    monkeypatch.setenv("HOME", str(home))
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}Hi\\end{document}\n"
    )
    assert backend.compile_latex(tmp_path, "paper").ok
    tex_home = backend._private_tex_home()
    assert tex_home != home and tex_home.is_dir()
    backend.close()
    backend.close()
    assert not tex_home.exists()


@pytest.mark.latex
def test_tex_environment_blocks_reads_outside_its_directory(tmp_path, monkeypatch):
    """The second line of defence, exercised directly: even a document that got past the
    static screen cannot \\input files outside its directory or see credentials."""
    from paper_orchestra.compute import local

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    work = tmp_path / "work"
    work.mkdir()
    home = tmp_path / "texhome"
    home.mkdir()
    (work / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        f"\\input{{{secret}}}\n\\input{{../secret.txt}}\n\\input{{/proc/self/environ}}\n"
        "\\end{document}\n"
    )
    env = local.latex_env(home)
    assert "ANTHROPIC_API_KEY" not in env and env["HOME"] == str(home)
    local._run(
        ["pdflatex", "-interaction=nonstopmode", "-no-shell-escape", "paper.tex"],
        cwd=work,
        env=env,
        timeout=120,
    )
    log = (work / "paper.log").read_text(errors="replace")
    assert "TOPSECRET" not in log and "sk-secret-value" not in log
    pdf = work / "paper.pdf"
    if pdf.exists():
        assert b"TOPSECRET" not in pdf.read_bytes()


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


def test_run_python_keeps_only_the_tail_of_huge_output(backend):
    result = backend.run_python("import sys\nfor i in range(400000):\n    print('line', i)\n")
    assert result.ok and len(result.stdout) <= 200_000
    assert result.stdout.rstrip().endswith("line 399999")


def test_interrupt_kills_the_child_process_group(tmp_path, monkeypatch):
    import subprocess

    from paper_orchestra.compute import local

    killed: list[int] = []
    real_wait = subprocess.Popen.wait
    state = {"first": True}

    def interrupted_wait(self, timeout=None):
        if state["first"]:
            state["first"] = False
            raise KeyboardInterrupt
        return real_wait(self, timeout)

    real_kill = local._kill_group
    monkeypatch.setattr(subprocess.Popen, "wait", interrupted_wait)
    monkeypatch.setattr(local, "_kill_group", lambda p: (killed.append(p.pid), real_kill(p)))
    with pytest.raises(KeyboardInterrupt):
        local._run(
            [local.sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=local.scrubbed_env(tmp_path),
            timeout=120,
        )
    assert len(killed) == 1


def test_bibtex_notice_and_bbl_errors():
    errors, _ = parse_latex_log("I found no \\citation commands---while reading file paper.aux\n")
    assert errors == []
    errors, _ = parse_latex_log(
        "./paper.bbl:12: Misplaced alignment tab character &.\nl.12 Smith & Jones\n"
    )
    assert errors == ["./paper.bbl:12: Misplaced alignment tab character &. [l.12 Smith & Jones]"]


def test_summarize_build_takes_warnings_from_the_last_pass_only():
    first = "LaTeX Warning: Citation `A' on page 1 undefined on input line 3.\n! Undefined control sequence.\nl.4 \\foo\n"
    last = "LaTeX Warning: Reference `sec:x' on page 1 undefined on input line 9.\n! Undefined control sequence.\nl.4 \\foo\n"
    errors, warnings = summarize_build(
        [("pdflatex", first), ("bibtex", "I couldn't open file name `x.aux'\n"), ("pdflatex", last)]
    )
    assert errors == [
        "Undefined control sequence. [l.4 \\foo]",
        "bibtex: I couldn't open file name `x.aux'",
    ]
    assert warnings == ["undefined reference 'sec:x'"]


def test_long_command_header_is_not_glued_to_the_log():
    header = "$ pdflatex -interaction=nonstopmode -no-shell-escape -file-line-error paper.tex"
    assert len(header) == 79
    errors, _ = parse_latex_log(header + "\n! Undefined control sequence.\nl.1 \\x\n")
    assert errors == ["Undefined control sequence. [l.1 \\x]"]


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


def test_check_reports_a_missing_tex_install(monkeypatch):
    backend = LocalBackend()
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    (problem,) = backend.check()
    assert "pdflatex/bibtex were not found" in problem and "--compute modal" in problem
    assert backend.check(latex=False) == []
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: True))
    assert backend.check() == []


def test_run_python_enforces_the_memory_limit_and_stops_at_the_timeout():
    hungry = "x = bytearray(1024 ** 3)\nprint('allocated')\n"
    limited = LocalBackend(memory_limit_mb=256).run_python(hungry)
    assert not limited.ok and "MemoryError" in limited.stderr and "allocated" not in limited.stdout

    spawning = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    slow = LocalBackend().run_python(spawning, timeout=1)
    assert slow.timed_out and time.monotonic() - started < 10
    grandchild = int(slow.stdout.split()[0])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the script's own child survived the timeout")


@pytest.mark.latex
def test_compile_timeout_is_reported_and_leaves_no_pdf(tmp_path):
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}\\loop\\iftrue\\repeat\\end{document}"
    )
    started = time.monotonic()
    result = LocalBackend().compile_latex(tmp_path, "paper", timeout=1.5)
    assert result.timed_out and not result.ok and result.pdf_path is None
    assert time.monotonic() - started < 15
    assert "timed out" in result.feedback()
