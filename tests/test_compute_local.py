from __future__ import annotations

import contextlib
import errno
import inspect
import itertools
import json
import mmap
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import paper_orchestra.compute.base as base
import paper_orchestra.compute.local as local
from paper_orchestra.compute import JobResult, JobSpec
from paper_orchestra.compute.base import BackendUnusable, CompileResult, ComputeError
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


def test_a_missing_tex_install_is_reported_as_an_unusable_backend(tmp_path, monkeypatch):
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}", encoding="utf-8"
    )
    monkeypatch.setattr(LocalBackend, "latex_available", staticmethod(lambda: False))
    with pytest.raises(BackendUnusable, match="pdflatex/bibtex not found") as caught:
        LocalBackend().compile_latex(tmp_path, "paper")
    # Callers stop instead of retrying: the next attempt would fail the same way.
    assert caught.value.unusable and isinstance(caught.value, ComputeError)
    assert ComputeError("the sandbox was preempted").unusable is False


# -- experiment jobs ---------------------------------------------------------------------


class JobDirs:
    """A source directory, a run directory and the place working directories are made in."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "runs" / "1"
        self.scratch = root / "scratch"
        self.scratch.mkdir()

    def source(self, main: str = "", **files: str) -> Path:
        source = self.root / "code"
        source.mkdir(exist_ok=True)
        (source / "main.py").write_text(textwrap.dedent(main), encoding="utf-8")
        for name, text in files.items():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_text(text, encoding="utf-8")
        return source

    def log(self) -> str:
        return (self.run / "log.txt").read_bytes().decode("utf-8")


@pytest.fixture
def job(tmp_path, monkeypatch) -> Iterator[JobDirs]:
    """Directories for one job.  Afterwards no working directory may be left behind."""
    dirs = JobDirs(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(dirs.scratch))
    yield dirs
    assert list(dirs.scratch.iterdir()) == [], "a working directory was left behind"


def run_main(
    backend: LocalBackend, job: JobDirs, source: Path, **options
) -> tuple[JobResult, list]:
    spec = JobSpec(command=["python", "main.py"], **options.pop("spec", {}))
    seen: list[str] = []
    options.setdefault("on_log", seen.append)
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, **options)
    return result, seen


def assert_gone(pid: int, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} is still running")


def test_run_job_runs_the_command_in_a_copy_and_returns_its_outputs(backend, job, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("PYTHONSTARTUP", "/nonexistent")
    source = job.source(
        """
        import json, os, sys
        from pathlib import Path

        print("in the source directory:", Path.cwd().samefile(sys.argv[1]))
        print("key:", os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("PYTHONSTARTUP"))
        print("seed:", os.environ["SEED"], "unbuffered:", os.environ["PYTHONUNBUFFERED"])
        print("home is private:", not Path.home().samefile(sys.argv[2]))
        print("stdin:", repr(sys.stdin.read()))
        print(Path("data/in.txt").read_text())
        Path("data/in.txt").write_text("changed by the job")
        Path("scratch.txt").write_text("not asked for")
        Path("artifacts/deep").mkdir(parents=True)
        Path("artifacts/curve.png").write_bytes(b"\\x89PNG")
        Path("artifacts/deep/table.csv").write_text("a,b\\n1,2\\n")
        Path("metrics.json").write_text(json.dumps({"accuracy": 0.91, "note": "ok"}))
        print("to stderr", file=sys.stderr)
        """,
        **{"data/in.txt": "payload"},
    )
    spec = JobSpec(
        command=["python", "main.py", str(source), os.environ["HOME"]],
        env={"SEED": "7"},
        timeout=60,
    )
    seen: list[str] = []
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)

    assert result.ok and result.returncode == 0, job.log()
    assert (result.timed_out, result.cancelled, result.error) == (False, False, None)
    assert (result.backend, result.handle, result.cost_usd) == ("local", None, None)
    assert result.log_path == job.run / "log.txt" and result.outputs_dir == job.run
    assert 0 < result.seconds < 30
    assert seen == [
        "in the source directory: False",
        "key: None None",
        "seed: 7 unbuffered: 1",
        "home is private: True",
        "stdin: ''",
        "payload",
        "to stderr",
    ]
    assert job.log().splitlines() == seen
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.91, "note": "ok"}
    assert (job.run / "artifacts" / "curve.png").read_bytes() == b"\x89PNG"
    assert (job.run / "artifacts" / "deep" / "table.csv").read_text() == "a,b\n1,2\n"
    assert sorted(p.name for p in job.run.iterdir()) == [
        "artifacts",
        "job.json",  # what the job started, for whoever has to clean up after a dead worker
        "log.txt",
        "metrics.json",
    ]
    # The person's files are as they were: the job worked on a copy.
    assert (source / "data" / "in.txt").read_text() == "payload"
    assert sorted(p.name for p in source.iterdir()) == ["data", "main.py"]
    # The local backend is not a sandbox, and says so instead of refusing.
    assert any("cannot block the network" in note for note in result.notes)
    assert "cannot block" not in job.log()


def test_python_in_a_command_is_the_interpreter_of_the_backend(job, monkeypatch, tmp_path):
    """Not whatever ``python`` the PATH finds: packages were checked for this one."""
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    source = job.source("import sys\nprint(sys.executable)\n")
    for name in ("python", "python3"):
        spec = JobSpec(command=[name, "main.py"])
        seen: list[str] = []
        result = LocalBackend().run_job(
            spec, source_dir=source, run_dir=job.run, on_log=seen.append
        )
        assert result.ok and seen == [sys.executable]


def test_a_failing_job_is_a_result_not_an_error_of_the_backend(backend, job):
    source = job.source("import sys\nprint('partial')\nsys.exit(3)\n")
    job.run.mkdir(parents=True)
    (job.run / "metrics.json").write_text('{"accuracy": 0.99}')  # of an earlier attempt
    (job.run / "handle.txt").write_text("sb-old")
    (job.run / "cost.json").write_text('{"cost_usd": 3.5, "seconds": 9.0, "final": true}')
    result, seen = run_main(backend, job, source)
    assert not result.ok and result.returncode == 3 and result.error is None
    assert seen == ["partial"]
    # Nothing an earlier run left behind passes for the output, or the cost, of this one.
    assert sorted(p.name for p in job.run.iterdir()) == ["job.json", "log.txt"]

    crashed, _ = run_main(backend, job, job.source("raise ValueError('boom')\n"))
    assert crashed.returncode == 1 and "ValueError: boom" in job.log()


def test_a_command_that_cannot_be_started_is_the_jobs_failure(backend, job):
    source = job.source()
    spec = JobSpec(command=["definitely-not-a-program", "--flag"])
    result = backend.run_job(spec, source_dir=source, run_dir=job.run)
    assert result.returncode == 127 and result.error is None and not result.ok
    assert "could not be started" in job.log() and "definitely-not-a-program" in job.log()


def test_output_is_streamed_while_the_job_runs(backend, job):
    source = job.source(
        """
        import time
        print("first")
        time.sleep(1.5)
        print("second")
        """
    )
    seen: list[tuple[str, float, str]] = []

    def on_log(line: str) -> None:
        seen.append((line, time.monotonic(), job.log()))

    result, _ = run_main(backend, job, source, on_log=on_log)
    assert result.ok
    (first, first_at, log_then), (second, second_at, _) = seen
    assert (first, second) == ("first", "second")
    assert second_at - first_at > 1.0  # not delivered together when the job ended
    assert log_then == "first\n"  # the file had it by then, too


def test_output_without_a_final_newline_and_binary_output(backend, job):
    result, seen = run_main(
        backend, job, job.source("import sys\nsys.stdout.write('no newline at the end')\n")
    )
    assert result.ok and seen == ["no newline at the end"]
    assert job.log() == "no newline at the end"

    binary = "import sys\nsys.stdout.buffer.write(bytes(range(256)) * 300 + b'\\nend')\n"
    result, seen = run_main(backend, job, job.source(binary))
    assert result.ok and seen[-1] == "end"
    assert "\ufffd" in job.log() and job.log().endswith("\nend")


def test_what_a_finished_command_left_running_is_killed(backend, job):
    """The command forks and exits; its child holds the output open and would run on."""
    source = job.source(
        """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        """
    )
    started = time.monotonic()
    result, seen = run_main(backend, job, source, spec={"timeout": 30})
    assert result.ok and not result.timed_out
    assert time.monotonic() - started < 8
    assert_gone(int(seen[0]))


def test_what_the_job_started_is_killed_even_when_nothing_gives_it_away(backend, job):
    """The children write nowhere, so the output ends with the command and no open pipe
    points to them; and when the caller is interrupted, nobody asks whether the command
    has left anything behind.  Either way the whole group is ended."""
    source = job.source(
        """
        import subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(child.pid, flush=True)
        if "--stay" in sys.argv:
            time.sleep(60)
        """
    )
    result, seen = run_main(backend, job, source, spec={"timeout": 30})
    assert result.ok
    assert_gone(int(seen[0]))

    pids: list[int] = []

    def on_log(line: str) -> None:
        pids.append(int(line))
        raise KeyboardInterrupt

    spec = JobSpec(command=["python", "main.py", "--stay"], timeout=30)
    with pytest.raises(KeyboardInterrupt):
        backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=on_log)
    assert_gone(pids[0])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_what_left_the_process_group_of_the_job_is_killed_too(backend, job):
    """coreutils timeout and shells with job control give what they start its own group."""
    source = job.source(
        """
        import subprocess, sys, time
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        child = subprocess.Popen(sleeper, process_group=0)
        print(child.pid, flush=True)
        if "--stay" in sys.argv:
            time.sleep(60)
        """
    )
    started = time.monotonic()
    result, seen = run_main(backend, job, source, spec={"timeout": 30})
    assert result.ok and time.monotonic() - started < 8  # not held up by the open pipe
    assert_gone(int(seen[0]))

    spec = JobSpec(command=["python", "main.py", "--stay"], timeout=1)
    seen = []
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)
    assert result.timed_out
    assert_gone(int(seen[0]))


def test_a_stop_request_is_heard_while_the_source_is_copied(backend, job):
    source = job.source("open('ran.txt', 'w').write('x')\n", **{"a/b/c/d.txt": "deep"})
    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return True

    result, seen = run_main(
        backend, job, source, should_stop=should_stop, spec={"outputs": ("ran.txt",)}
    )
    assert result.cancelled and result.returncode is None and result.error is None
    assert len(asked) == 1 and not (job.run / "ran.txt").exists()
    assert seen == ["[paper-orchestra] The job was stopped on request, before it had started."]


def test_a_stop_request_is_heard_between_the_files_of_one_directory(backend, job, monkeypatch):
    """A flat directory of data files can take minutes to copy."""
    monkeypatch.setattr(base, "CHECK_SECONDS", 0.0)
    files = {f"data_{index:03d}.bin": "x" for index in range(200)}
    source = job.source("open('ran.txt', 'w').write('x')\n", **files)
    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) > 20

    result, _ = run_main(
        backend, job, source, should_stop=should_stop, spec={"outputs": ("ran.txt",)}
    )
    assert result.cancelled and result.returncode is None and len(asked) == 21
    assert not (job.run / "ran.txt").exists()


def test_a_job_that_ignores_the_request_to_stop_is_killed(backend, job, monkeypatch):
    monkeypatch.setattr(base, "KILL_GRACE", 0.5)
    source = job.source(
        """
        import os, signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(os.getpid(), child.pid, flush=True)
        while True:
            time.sleep(0.1)
        """
    )
    started = time.monotonic()
    result, seen = run_main(backend, job, source, spec={"timeout": 1})
    assert result.timed_out and not result.cancelled and not result.ok
    assert result.returncode == 128 + signal.SIGKILL and result.error is None
    assert 1.0 <= result.seconds < 8 and time.monotonic() - started < 10
    for pid in map(int, seen[0].split()):
        assert_gone(pid)
    assert (
        seen[-1]
        == "[paper-orchestra] The job was stopped because it ran longer than its time limit of 1 s."
    )
    assert job.log().endswith(seen[-1] + "\n")


def test_a_job_that_is_asked_to_stop_may_finish_what_it_was_writing(backend, job):
    source = job.source(
        """
        import json, signal, sys, time

        def stop(*_):
            json.dump({"steps": 3, "complete": "no"}, open("metrics.json", "w"))
            print("stopping cleanly", flush=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, stop)
        print("training", flush=True)
        time.sleep(60)
        """
    )
    result, seen = run_main(backend, job, source, spec={"timeout": 1})
    assert result.timed_out and result.returncode == 0 and not result.ok
    assert seen[:2] == ["training", "stopping cleanly"]
    assert json.loads((job.run / "metrics.json").read_text()) == {"steps": 3, "complete": "no"}


def test_a_timeout_while_output_pours_in(backend, job, monkeypatch):
    """A job printing in a loop is stopped on time, and fills neither disk nor memory."""
    monkeypatch.setattr(base, "MAX_LOG_BYTES", 200_000)
    monkeypatch.setattr(base, "LOG_TAIL_BYTES", 10_000)
    source = job.source("import sys\nwhile True:\n    sys.stdout.write('spam ' * 20 + '\\n')\n")
    started = time.monotonic()
    result, seen = run_main(backend, job, source, spec={"timeout": 1})
    assert result.timed_out and time.monotonic() - started < 15
    assert (job.run / "log.txt").stat().st_size < 200_000 + 10_000 + 1_000
    lines = job.log().splitlines()
    spam = "spam " * 20
    [marker] = [index for index, line in enumerate(lines) if "left out here" in line]
    assert 1000 < marker <= 200_000 // 101 + 1
    assert set(lines[: marker - 1]) == {spam}  # the start of the output...
    assert spam.startswith(lines[marker - 1])  # ...up to where it was cut...
    assert set(lines[marker + 2 : -1]) == {spam}  # ...and its end
    assert lines[-1].startswith("[paper-orchestra] The job was stopped because it ran longer")
    assert len(seen) < marker + 10 and sum("no longer shown live" in line for line in seen) == 1
    assert any("printed more than 195 KB" in note for note in result.notes)


def test_a_stop_request_reaches_a_silent_job(backend, job):
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(60)\n")
    seen: list[str] = []
    asked: list[float] = []

    def should_stop() -> bool:
        asked.append(time.monotonic())
        return bool(seen) and len(asked) > 3

    started = time.monotonic()
    result, _ = run_main(backend, job, source, on_log=seen.append, should_stop=should_stop)
    assert result.cancelled and not result.timed_out and not result.ok and result.error is None
    assert result.returncode == 128 + signal.SIGTERM
    assert time.monotonic() - started < 8
    assert max(b - a for a, b in itertools.pairwise(asked)) < 1.0  # asked while blocked
    assert_gone(int(seen[0]))
    assert seen[-1] == "[paper-orchestra] The job was stopped on request."


def test_the_log_is_complete_when_the_job_is_killed(backend, job, monkeypatch):
    monkeypatch.setattr(base, "KILL_GRACE", 0.2)
    source = job.source(
        """
        import signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for index in range(300):
            print("step", index)
        sys.stdout.write("half a li")
        sys.stdout.flush()
        time.sleep(60)
        """
    )
    result, seen = run_main(backend, job, source, spec={"timeout": 1.5})
    assert result.timed_out and result.returncode == 137
    lines = job.log().splitlines()
    assert lines[:300] == [f"step {index}" for index in range(300)]
    assert lines[300] == "half a li" and lines[301].startswith("[paper-orchestra] ")
    assert seen == lines


def test_outputs_that_are_links_are_not_followed(backend, job, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    source = job.source(
        f"""
        import os
        os.makedirs("artifacts")
        os.symlink({str(secret)!r}, "artifacts/leak.txt")
        os.symlink({str(tmp_path)!r}, "artifacts/leak_dir")
        os.symlink({str(secret)!r}, "metrics.json")
        os.symlink("real.txt", "artifacts/inside.txt")
        os.mkfifo("artifacts/pipe")
        open("artifacts/real.txt", "w").write("real")
        """
    )
    result, _ = run_main(backend, job, source)
    assert result.ok
    copied = sorted(p.relative_to(job.run).as_posix() for p in job.run.rglob("*"))
    assert copied == ["artifacts", "artifacts/real.txt", "job.json", "log.txt"]
    assert not any(p.is_symlink() for p in job.run.rglob("*"))
    notes = "\n".join(result.notes)
    for name in (
        "artifacts/leak.txt",
        "artifacts/leak_dir",
        "metrics.json",
        "artifacts/inside.txt",
    ):
        assert name in notes
    assert "symbolic links are never followed" in notes and "artifacts/pipe" in notes
    assert "symbolic links are never followed" in job.log()  # the person sees it in the log too
    assert "TOPSECRET" not in job.log()


def test_an_output_behind_a_linked_directory_is_not_followed(backend, job, tmp_path):
    """The output is named with a directory, and the job made that directory a link."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "metrics.json").write_text("TOPSECRET")
    source = job.source(f"import os\nos.symlink({str(outside)!r}, 'results')\n")
    result, _ = run_main(backend, job, source, spec={"outputs": ("results/metrics.json",)})
    assert result.ok
    assert sorted(p.name for p in job.run.iterdir()) == ["job.json", "log.txt"]
    [note] = [note for note in result.notes if "symbolic links are never followed" in note]
    assert "results" in note and note in job.log()


def test_the_end_of_the_output_is_not_lost_to_a_slow_listener(backend, job, monkeypatch):
    """The job has exited before its output was read, and whoever listens takes longer
    over it than the backend waits for a pipe that something else holds open."""
    monkeypatch.setattr(base, "DRAIN_SECONDS", 0.2)
    source = job.source(
        """
        import sys
        for index in range(4000):
            sys.stdout.write(f"line {index:05d} " + "x" * 100 + "\\n")
        print("the last line")
        """
    )
    seen: list[str] = []

    def on_log(line: str) -> None:
        seen.append(line)
        time.sleep(0.0005)

    result, _ = run_main(backend, job, source, on_log=on_log)
    assert result.ok
    assert len(seen) == 4001 and seen[-1] == "the last line"
    assert job.log().endswith("the last line\n")
    assert not any("kept its output open" in note for note in result.notes)


DAEMON = """
import os, sys, time
ready, tell = os.pipe()
if os.fork() == 0:
    os.setsid()
    print(os.getpid(), flush=True)
    os.write(tell, b"x")
    os.close(tell)
    if "--drop-environment" in sys.argv:
        # Nothing of the job is left on it: not its group, its session or its mark.
        os.execve(sys.executable, [sys.executable, "-c", "import time; time.sleep(60)"], {})
    time.sleep(60)
    os._exit(0)
os.close(tell)
os.read(ready, 1)
print("the command is done", flush=True)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_a_daemon_in_a_session_of_its_own_is_found_by_the_jobs_mark_and_stopped(backend, job):
    """It used to be out of reach, and ran on with the job's output held open."""
    started = time.monotonic()
    result, seen = run_main(backend, job, job.source(DAEMON))
    daemon = int(seen[0])
    try:
        assert result.ok and time.monotonic() - started < 10
        assert "the command is done" in seen
        assert_gone(daemon)
        [note] = [note for note in result.notes if "outside its own session" in note]
        assert note.startswith("1 process that the job had started") and "was stopped" in note
        assert note in job.log()
        assert not any("kept its output open" in note for note in result.notes)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(daemon, signal.SIGKILL)


DETACHED = """
import json, os, subprocess, sys, time
# A helper in a session of its own that holds none of the job's pipes: nothing but the
# mark in its environment says whose it is.
helper = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(helper.pid, flush=True)
json.dump({"a": 1}, open("metrics.json", "w"))
if "--stay" in sys.argv:
    time.sleep(120)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_a_detached_helper_does_not_outlive_the_job_however_the_job_ends(backend, job):
    source = job.source(DETACHED)
    result, seen = run_main(backend, job, source)
    helper = int(seen[0])
    try:
        assert result.ok
        assert_gone(helper)
        assert any("outside its own session" in note for note in result.notes)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(helper, signal.SIGKILL)

    spec = JobSpec(command=["python", "main.py", "--stay"], timeout=1)
    seen = []
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)
    helper = int(seen[0])
    try:
        assert result.timed_out
        assert_gone(helper)
        assert any("outside its own session" in note for note in result.notes)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(helper, signal.SIGKILL)

    # Nor an orchestrator that is interrupted while the job runs.
    pids: list[int] = []

    def on_log(line: str) -> None:
        pids.append(int(line))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=on_log)
    assert_gone(pids[0])


@pytest.mark.skipif(os.name != "posix", reason="starts a session of its own")
def test_output_that_an_escaped_process_holds_open_is_given_up_and_reported(
    backend, job, monkeypatch
):
    """A daemon that left the session and dropped its environment is out of reach: the
    job still ends, and the person is told that something was left behind."""
    monkeypatch.setattr(base, "DRAIN_SECONDS", 0.5)
    source = job.source(DAEMON)
    spec = JobSpec(command=["python", "main.py", "--drop-environment"])
    seen: list[str] = []
    started = time.monotonic()
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)
    daemon = int(seen[0])
    try:
        assert result.ok and time.monotonic() - started < 10
        assert "the command is done" in seen
        [note] = [note for note in result.notes if "kept its output open" in note]
        assert "no longer carries the job's mark" in note and note in job.log()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(daemon, signal.SIGKILL)


def test_a_working_directory_inside_the_source_is_refused(job):
    """Copying a directory into itself would never end."""
    source = job.source("print('never runs')\n")
    backend = LocalBackend(scratch_dir=source / "scratch")
    result, seen = run_main(backend, job, source)
    assert result.returncode is None and not result.ok
    assert "inside the source directory" in result.error and str(source) in result.error
    assert len(result.error) < 600 and seen == [f"[paper-orchestra] {result.error}"]
    assert not (source / "scratch").exists()


def test_a_huge_artifact_is_left_out_and_reported(backend, job, monkeypatch):
    monkeypatch.setattr(base, "MAX_OUTPUT_BYTES", 20_000)
    source = job.source(
        """
        import os
        os.makedirs("artifacts")
        open("metrics.json", "w").write('{"loss": 0.5}')
        open("artifacts/a_huge.bin", "wb").write(b"x" * 500_000)
        open("artifacts/b_small.txt", "w").write("small")
        """
    )
    result, _ = run_main(backend, job, source)
    assert result.ok  # the job did its work; what was not copied is a note
    assert (job.run / "metrics.json").exists() and (job.run / "artifacts" / "b_small.txt").exists()
    assert not (job.run / "artifacts" / "a_huge.bin").exists()
    [note] = [note for note in result.notes if "a_huge.bin" in note]
    assert "488 KB" in note and "limited to 20 KB in total" in note
    assert note in job.log()


def test_too_many_output_files_are_capped(backend, job, monkeypatch):
    monkeypatch.setattr(base, "MAX_OUTPUT_FILES", 5)
    source = job.source(
        """
        import os
        os.makedirs("artifacts")
        for index in range(40):
            open(f"artifacts/{index:02d}.txt", "w").write("x")
        """
    )
    result, _ = run_main(backend, job, source)
    assert len(list((job.run / "artifacts").iterdir())) == 5
    assert any("Only the first 5 files" in note and "35 more" in note for note in result.notes)


def test_the_source_is_copied_without_repository_data_or_credentials(backend, job, tmp_path):
    source = job.source(
        """
        import json, os
        names = sorted(
            os.path.join(root, name)[2:] for root, dirs, files in os.walk(".") for name in dirs + files
        )
        print(json.dumps(names))
        print(os.path.getsize("data/big.bin"), os.readlink("sub/link"), os.listdir("sub/link"))
        print(os.access("tool.sh", os.X_OK))
        """,
        **{
            ".git/objects/pack/pack-1.pack": "x" * 100_000,
            ".git/config": "[core]",
            ".env": "API_KEY=sk-secret",
            ".env.local": "API_KEY=sk-secret",
            "pkg/__pycache__/mod.cpython-312.pyc": "bytecode",
            "pkg/mod.py": "VALUE = 1",
            "node_modules/left-pad/index.js": "x",
            ".gitignore": "*.pyc",
            "tool.sh": "#!/bin/sh\n",
            "sub/keep.txt": "keep",
        },
    )
    (source / "tool.sh").chmod(0o755)
    (source / "data").mkdir()
    with open(source / "data" / "big.bin", "wb") as big:
        big.truncate(64 * 1024 * 1024)
    (source / "sub" / "link").symlink_to("../data")
    (source / "abs_link").symlink_to(tmp_path)
    (source / "up_link").symlink_to("../../outside")
    os.mkfifo(source / "pipe")

    started = time.monotonic()
    result, seen = run_main(backend, job, source)
    assert result.ok, job.log()
    assert time.monotonic() - started < 30
    lines = [line for line in seen if not line.startswith("[paper-orchestra]")]
    assert json.loads(lines[0]) == [
        ".gitignore",
        "data",
        "data/big.bin",
        "main.py",
        "pkg",
        "pkg/mod.py",
        "sub",
        "sub/keep.txt",
        "sub/link",
        "tool.sh",
    ]
    assert lines[1] == f"{64 * 1024 * 1024} ../data ['big.bin']"
    assert lines[2] == "True"
    notes = "\n".join(result.notes)
    for name in (".git", ".env", ".env.local", "pkg/__pycache__", "node_modules"):
        assert name in notes
    for name in ("abs_link", "up_link", "pipe"):
        assert name in notes
    assert ".git/config" not in notes  # the directory is named once, not every file in it
    assert seen[0].startswith("[paper-orchestra] Left out of the working directory")
    assert (source / ".git" / "config").exists() and (source / ".env").exists()


def test_an_interrupt_leaves_no_process_and_no_directory(backend, job):
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(60)\n")
    pids: list[int] = []

    def on_log(line: str) -> None:
        pids.append(int(line))
        raise KeyboardInterrupt  # Ctrl-C arrives while the job runs

    with pytest.raises(KeyboardInterrupt):
        run_main(backend, job, source, on_log=on_log)
    assert_gone(pids[0])
    assert list(job.scratch.iterdir()) == []
    assert job.log() == f"{pids[0]}\n"

    def should_stop() -> bool:
        if len(pids) > 1:  # once the job is running
            raise RuntimeError("the state file is unreadable")
        return False

    with pytest.raises(RuntimeError, match="state file"):
        run_main(backend, job, source, on_log=pids.append, should_stop=should_stop)
    assert_gone(int(pids[1]))
    assert list(job.scratch.iterdir()) == []


def test_a_working_directory_the_job_locked_is_still_removed(backend, job):
    source = job.source(
        """
        import os
        os.makedirs("locked/inner")
        open("locked/inner/file.txt", "w").write("x")
        os.chmod("locked/inner", 0)
        os.chmod("locked", 0)
        """
    )
    result, _ = run_main(backend, job, source)
    assert result.ok


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses PR_SET_PDEATHSIG")
def test_a_job_does_not_outlive_an_orchestrator_that_is_killed(job, tmp_path):
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(120)\n")
    driver = (
        "import sys\n"
        "from pathlib import Path\n"
        "from paper_orchestra.compute import JobSpec, LocalBackend\n"
        # Told where to work: what is patched in this process does not reach that one,
        # and what it leaves behind must not pile up in the system's temporary directory.
        "LocalBackend(scratch_dir=sys.argv[3]).run_job(JobSpec(command=['python', 'main.py']),\n"
        "    source_dir=Path(sys.argv[1]), run_dir=Path(sys.argv[2]))\n"
    )
    orchestrator = subprocess.Popen(
        [sys.executable, "-c", driver, str(source), str(job.run), str(job.scratch)]
    )
    try:
        deadline = time.monotonic() + 20
        log = job.run / "log.txt"
        while not (log.exists() and log.read_text().strip()):
            assert time.monotonic() < deadline and orchestrator.poll() is None
            time.sleep(0.05)
        pid = int(log.read_text())
    finally:
        orchestrator.kill()
        orchestrator.wait()
    assert_gone(pid)
    # Nobody was left to remove the working directory; the job's record says where it is.
    [leftover] = job.scratch.iterdir()
    assert leftover.name.startswith("paper-orchestra-job-")
    record = json.loads((job.run / "job.json").read_text())
    assert record["scratch"] == str(leftover) and record["session"] == pid
    notes = LocalBackend.end_job(record)
    assert not leftover.exists() and any(str(leftover) in note for note in notes)


KILLED_WORKER = (
    "import sys\n"
    "from pathlib import Path\n"
    "from paper_orchestra.compute import JobSpec, LocalBackend\n"
    "LocalBackend(scratch_dir=sys.argv[3]).run_job(\n"
    "    JobSpec(command=['python', 'main.py'], timeout=300),\n"
    "    source_dir=Path(sys.argv[1]), run_dir=Path(sys.argv[2]))\n"
)


def killed_worker(job: JobDirs, source: Path, lines: int) -> list[str]:
    """Run ``source`` under a worker of its own, kill that worker with SIGKILL once the
    job has printed ``lines`` lines, and return those lines."""
    worker = subprocess.Popen(
        [sys.executable, "-c", KILLED_WORKER, str(source), str(job.run), str(job.scratch)]
    )
    try:
        deadline = time.monotonic() + 20
        log = job.run / "log.txt"
        # Whole lines: a job's output is not buffered, and one print arrives in pieces.
        while not (log.exists() and log.read_text().count("\n") >= lines):
            assert time.monotonic() < deadline and worker.poll() is None
            time.sleep(0.05)
        return log.read_text().splitlines()
    finally:
        worker.kill()
        worker.wait()


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat_file:
            return stat_file.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_what_a_killed_worker_left_behind_is_ended_and_removed_from_the_jobs_record(job):
    """kill -9, an OOM kill or a crash: the job's main process dies with the worker, what
    it had started runs on, and its working directory stays."""
    source = job.source(
        """
        import os, subprocess, sys, time
        sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
        child = subprocess.Popen(sleeper)                       # same group, same session
        grouped = subprocess.Popen(sleeper, process_group=0)    # a group of its own
        detached = subprocess.Popen(sleeper, start_new_session=True)
        os.makedirs("locked/inner")
        os.chmod("locked", 0)
        print(os.getpid(), child.pid, grouped.pid, detached.pid, flush=True)
        time.sleep(300)
        """
    )
    main, *children = (int(pid) for pid in killed_worker(job, source, 1)[0].split())
    try:
        assert_gone(main)  # the kernel ends it with the worker
        assert all(alive(pid) for pid in children), "the scenario needs survivors"
        [scratch] = job.scratch.iterdir()

        record = json.loads((job.run / "job.json").read_text())
        assert set(record) == {"session", "session_started", "scratch"}
        assert record["session"] == main and record["scratch"] == str(scratch)
        assert isinstance(record["session_started"], str) and record["session_started"]

        # What would be ended can be asked first, by whoever wants to know (the web app).
        assert LocalBackend.job_processes(record) == set(children)
        assert all(alive(pid) for pid in children) and scratch.is_dir()
        # The engine keeps the record inside the run's own; either shape is understood.
        notes = LocalBackend.end_job({"status": "running", "job": record})
        for pid in children:
            assert_gone(pid)
        assert not scratch.exists() and LocalBackend.job_processes(record) == set()
        assert len(notes) == 2
        assert notes[0].startswith("Stopped 3 processes that the run had left running")
        assert notes[1] == f"Removed the working directory {scratch} that the run had left behind."
        # Settling it again finds nothing left to do.
        assert LocalBackend.end_job(record) == []
    finally:
        for pid in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        for leftover in job.scratch.iterdir():
            subprocess.run(["chmod", "-R", "u+rwx", str(leftover)], check=False)
            shutil.rmtree(leftover, ignore_errors=True)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_cleaning_up_never_touches_what_is_not_the_jobs(job, tmp_path):
    """A record is read long after it was written: the number of its session may belong
    to somebody else's processes by then, and its path is only a string in a file."""
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.tex").write_text("years of work")
    lookalike = tmp_path / "paper-orchestra-job-link"
    lookalike.symlink_to(precious, target_is_directory=True)
    try:
        # The same session number, but not the session that was started back then.
        stale = {"session": bystander.pid, "session_started": "12345@another-boot"}
        assert LocalBackend.end_job({**stale, "scratch": str(precious)}) == []
        assert LocalBackend.end_job({**stale, "scratch": str(lookalike)}) == []
        assert LocalBackend.end_job({**stale, "scratch": "/"}) == []
        assert bystander.poll() is None and (precious / "thesis.tex").exists()

        # Records that are not job records at all: of a Modal run, empty, or damaged.
        damaged = (
            {},
            {"job": None},
            {"session": "7", "scratch": 3},
            {"session": True},
            None,
            ["scratch"],
            {"scratch": f"{job.scratch}/paper-orchestra-job-\ud800"},  # no name of a file
            {"scratch": f"{job.scratch}/paper-orchestra-job-a\0b"},
            {"job": {"job": {"scratch": str(precious)}}},
        )
        for record in damaged:
            assert LocalBackend.end_job(record) == []
            assert LocalBackend.job_processes(record) == set()
        assert LocalBackend.end_job({"session": 0, "session_started": "", "scratch": ""}) == []
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()

    # A working directory that this backend made is removed, also without any process.
    scratch = job.scratch / "paper-orchestra-job-abc123"
    (scratch / "work").mkdir(parents=True)
    notes = LocalBackend.end_job({"session": 0, "session_started": "", "scratch": str(scratch)})
    assert not scratch.exists() and len(notes) == 1 and str(scratch) in notes[0]


on_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="looks processes up in /proc"
)

#: A process of the person's that has nothing to do with any job: it leads a session of
#: its own and has a child in it, as a shell with a program running in it does.
BYSTANDER = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(300)\n"
)


def job_record(scratch: Path | str, session: int = 0, started: str = "") -> dict:
    return {"session": session, "session_started": started, "scratch": str(scratch)}


@on_linux
def test_a_record_that_names_a_session_of_the_persons_ends_nothing(job):
    """run.json and job.json are files of a project directory, and a project can come
    from somebody else.  When a process was started can be read by anybody in /proc:
    a record that named a session of the person's with its start used to get the whole
    session killed as soon as the run was settled."""
    bystander = subprocess.Popen(
        [sys.executable, "-c", BYSTANDER], start_new_session=True, stdout=subprocess.PIPE
    )
    child = int(bystander.stdout.readline())
    leftover = job.scratch / "paper-orchestra-job-left"
    (leftover / "work").mkdir(parents=True)
    try:
        started = local._session_started(bystander.pid)
        assert started, "the start of the session could not be read"
        # By itself, and next to a working directory that really was left behind.
        for scratch in ("", leftover):
            record = job_record(scratch, bystander.pid, started)
            notes = LocalBackend.end_job({"status": "running", "job": record})
            assert bystander.poll() is None and alive(child), "a session of the person's ended"
            assert LocalBackend.job_processes(record) == set()
        assert not leftover.exists() and len(notes) == 1 and "Removed" in notes[0]
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(bystander.pid, signal.SIGKILL)
        bystander.wait()


def started_job(job: JobDirs, source: Path) -> tuple[subprocess.Popen, int, Path]:
    """A job under a worker of its own that goes on running: the worker, the process of
    the job's command, and the job's working directory."""
    worker = subprocess.Popen(
        [sys.executable, "-c", KILLED_WORKER, str(source), str(job.run), str(job.scratch)]
    )
    log = job.run / "log.txt"
    deadline = time.monotonic() + 30
    try:
        while not (log.exists() and "\n" in log.read_text()):
            assert time.monotonic() < deadline and worker.poll() is None
            time.sleep(0.05)
    except BaseException:
        worker.kill()
        worker.wait()
        raise
    [scratch] = job.scratch.iterdir()
    return worker, int(log.read_text().split()[0]), scratch


@on_linux
def test_a_record_that_names_the_working_directory_of_a_running_job_ends_nothing(job):
    """Working directories are made in the temporary directory, where anybody can list
    them.  A record that names the one of a job that is running (a planted record, or
    the record in a copy of a project that was made while its job ran) used to get that
    job killed and its directory removed."""
    source = job.source("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(300)\n")
    worker, command, scratch = started_job(job, source)
    try:
        own = json.loads((job.run / "job.json").read_text())
        assert own["scratch"] == str(scratch) and own["session"] == command
        for record in (job_record(scratch), own, {"job": own}):
            notes = LocalBackend.end_job(record)
            assert worker.poll() is None and alive(command), "a running job was ended"
            assert (scratch / "work" / "main.py").is_file(), "its working directory was removed"
            assert len(notes) == 1 and str(scratch) in notes[0]
            assert "is running" in notes[0] and "Nothing was stopped or removed" in notes[0]
            assert LocalBackend.job_processes(record) == set()
    finally:
        worker.kill()
        worker.wait()
    # Once its worker is gone, the same record is what the leftovers are cleaned up by.
    assert_gone(command)
    assert LocalBackend.end_job(own) == [
        f"Removed the working directory {scratch} that the run had left behind."
    ]


#: A job with a child, and a helper that detached from all of them: its parent ended,
#: so it is nobody's descendant any more.  Prints the two process ids.
DETACHING_JOB = """
import subprocess, sys, time
sleep = [sys.executable, "-c", "import time; time.sleep(300)"]
child = subprocess.Popen(sleep)
start = "import subprocess, sys; "
start += "print(subprocess.Popen(sys.argv[1:], start_new_session=True).pid, flush=True)"
middle = subprocess.Popen([sys.executable, "-c", start, *sleep], stdout=subprocess.PIPE)
detached = int(middle.stdout.readline())
middle.wait()
print(child.pid, detached, flush=True)
time.sleep(300)
"""

#: Stands for a worker of another project: it starts the job given as its last argument
#: but one, with the mark of the working directory given as its last, and holds no lock.
OTHER_WORKER = """
import os, subprocess, sys, time
job, scratch = sys.argv[-2:]
command = subprocess.Popen(
    [sys.executable, "-c", job],
    env={**os.environ, "PAPER_ORCHESTRA_JOB": scratch},
    start_new_session=True,
    stdout=subprocess.PIPE,
)
print(command.pid, command.stdout.readline().decode().strip(), flush=True)
time.sleep(300)
"""


@on_linux
@pytest.mark.parametrize("spelling", ["module", "script"])
def test_what_runs_under_a_living_worker_is_that_workers_job_and_is_left_alone(
    job, tmp_path, spelling
):
    """Not every working directory is held by a lock (a file system without locks, a
    job of an older version).  A job whose processes descend from a living
    ``paper-orchestra research ...`` process is that worker's, whatever a record says:
    nothing of it is touched, not even what has detached from the worker."""
    scratch = job.scratch / "paper-orchestra-job-of-another-project"
    (scratch / "work").mkdir(parents=True)
    (scratch / "work" / "checkpoint.bin").write_text("hours of work")
    if spelling == "module":
        argv = [sys.executable, "-c", OTHER_WORKER, "-m", "paper_orchestra", "research", "run"]
    else:
        script = tmp_path / "bin" / "paper-orchestra"
        script.parent.mkdir()
        script.write_text(OTHER_WORKER)
        argv = [sys.executable, str(script), "research", "advance"]
    argv += ["another-project", DETACHING_JOB, str(scratch)]
    worker = subprocess.Popen(argv, stdout=subprocess.PIPE)
    # The job's command, its child and the helper that detached.
    pids = [int(pid) for pid in worker.stdout.readline().split()]
    record = job_record(scratch)
    try:
        assert all(alive(pid) for pid in pids), "the scenario needs a running job"
        notes = LocalBackend.end_job(record)
        assert all(alive(pid) for pid in pids), "a job of a living worker was ended"
        assert (scratch / "work" / "checkpoint.bin").read_text() == "hours of work"
        assert len(notes) == 1 and str(scratch) in notes[0]
        assert f"process {worker.pid}" in notes[0] and "Nothing was stopped or removed" in notes[0]
        assert LocalBackend.job_processes(record) == set()

        # The worker dies (kill -9); its job runs on.  Now it is a leftover.
        worker.kill()
        worker.wait()
        assert all(alive(pid) for pid in pids)
        assert LocalBackend.job_processes(record) == set(pids)
        notes = LocalBackend.end_job(record)
        for pid in pids:
            assert_gone(pid)
        assert notes[0].startswith("Stopped 3 processes") and not scratch.exists()
    finally:
        worker.kill()
        worker.wait()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(scratch, ignore_errors=True)


@on_linux
@pytest.mark.parametrize("spelled_otherwise", ["the record", "the mark"])
def test_a_job_is_the_same_job_however_its_working_directory_is_spelled(
    job, tmp_path, spelled_otherwise
):
    """A mark names a directory, and so does a record.  A record that led to a working
    directory by another spelling than the one in the job's mark (through a link that
    came with the project, or by the other name of a temporary directory that is
    reached through a link) found none of the job's processes.  So a job under a living
    worker was not seen to be running, and what a dead worker had left was removed from
    under its processes, which nothing found again."""
    scratch = job.scratch / "paper-orchestra-job-spelled"
    (scratch / "work").mkdir(parents=True)
    (scratch / "work" / "checkpoint.bin").write_text("hours of work")
    link = tmp_path / "project" / "tmp"
    link.parent.mkdir()
    link.symlink_to(job.scratch, target_is_directory=True)
    by_link = link / scratch.name
    named, marked = (by_link, scratch) if spelled_otherwise == "the record" else (scratch, by_link)
    argv = [sys.executable, "-c", OTHER_WORKER, "-m", "paper_orchestra", "research", "run"]
    worker = subprocess.Popen(
        [*argv, "another-project", DETACHING_JOB, str(marked)], stdout=subprocess.PIPE
    )
    pids = [int(pid) for pid in worker.stdout.readline().split()]
    record = job_record(named)
    try:
        assert all(alive(pid) for pid in pids), "the scenario needs a running job"
        notes = LocalBackend.end_job(record)
        assert all(alive(pid) for pid in pids), "a job of a living worker was ended"
        assert (scratch / "work" / "checkpoint.bin").read_text() == "hours of work"
        assert len(notes) == 1 and f"process {worker.pid}" in notes[0]
        assert LocalBackend.job_processes(record) == set()

        worker.kill()
        worker.wait()
        assert LocalBackend.job_processes(record) == set(pids)
        notes = LocalBackend.end_job(record)
        for pid in pids:
            assert_gone(pid)
        assert notes[0].startswith("Stopped 3 processes") and not scratch.exists()
    finally:
        worker.kill()
        worker.wait()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_mark_names_a_directory_only_by_a_path_that_leads_to_it(tmp_path, monkeypatch):
    scratch = tmp_path / "paper-orchestra-job-a"
    scratch.mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / scratch.name).mkdir()
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "paper-orchestra-job-b").symlink_to(scratch, target_is_directory=True)

    def names(mark: str) -> bool:
        return local._names(os.fsencode(mark), str(scratch))

    assert names(str(scratch)) and names(str(tmp_path / "link" / scratch.name))
    # Another directory of the same name, a path that leads nowhere, and no path at all.
    assert not names(str(tmp_path / "other" / scratch.name))
    assert not names(str(tmp_path / "gone" / scratch.name))
    assert not names("") and not names(f"{scratch}\0") and not names(f"{scratch}/")
    # A relative one names something else for everybody who reads it: also for one who
    # happens to stand where it leads to the directory.
    monkeypatch.chdir(tmp_path)
    assert os.path.samefile(scratch.name, scratch) and not names(scratch.name)
    # Nor a name of its own that leads there: a backend's mark ends in the name it made.
    assert not names(str(tmp_path / "paper-orchestra-job-b"))


#: A process of a job that asks, from a child of its own, what would be ended for the
#: job's record; it has one more child, which only waits.
ASKS_ABOUT_ITSELF = """
import json, subprocess, sys
waits = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
look = "import json, sys; from paper_orchestra.compute.local import LocalBackend; "
look += "print(json.dumps(sorted(LocalBackend.job_processes({'scratch': sys.argv[1]}))))"
asked = subprocess.run([sys.executable, "-c", look, sys.argv[1]], capture_output=True, text=True)
print(json.dumps({"waits": waits.pid, "found": asked.stdout, "error": asked.stderr}), flush=True)
waits.kill()
"""


@on_linux
def test_whoever_asks_is_never_among_the_processes_to_end_nor_what_it_descends_from(job):
    scratch = job.scratch / "paper-orchestra-job-nested"
    scratch.mkdir()
    env = {**os.environ, local.JOB_ENV: str(scratch), "TMPDIR": str(job.scratch)}
    try:
        asking = subprocess.run(
            [sys.executable, "-c", ASKS_ABOUT_ITSELF, str(scratch)],
            env=env,
            start_new_session=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        said = json.loads(asking.stdout)
        assert json.loads(said["found"]) == [said["waits"]], said["error"]
    finally:
        scratch.rmdir()


@pytest.mark.parametrize(
    ("command", "is_one"),
    [
        (["python", "-P", "-m", "paper_orchestra", "research", "run", "/projects/p"], True),
        (["/venv/bin/python3", "/venv/bin/paper-orchestra", "research", "advance", "p"], True),
        (["paper-orchestra", "research", "demo", "p"], True),
        (["C:\\venv\\Scripts\\paper-orchestra.exe", "research", "cancel", "p", "E1"], True),
        (["python", "/src/paper_orchestra/__main__.py", "research", "write", "p"], True),
        (["python", "-m", "paper_orchestra", "ui", "--workspace", "w"], False),
        (["python", "-m", "paper_orchestra", "run", "--idea", "idea.md"], False),
        (["python", "-m", "pytest", "tests/test_research_engine.py"], False),
        (["vim", "src/paper_orchestra/research/experiments.py"], False),
        (["python", "-c", "import time; time.sleep(300)", "research"], False),
        (["research"], False),
        ([], False),
    ],
)
def test_which_command_lines_read_as_a_worker_of_this_package(command, is_one):
    assert local._reads_as_worker(command) is is_one


@on_linux
def test_only_a_working_directory_that_this_backend_makes_is_cleaned_up(job, tmp_path, monkeypatch):
    """The path in a record is a string in a file.  What is ended and removed by it is a
    directory named as this backend names them, directly in the place where it makes
    them, a real directory, and the person's own."""
    elsewhere = tmp_path / "elsewhere" / "paper-orchestra-job-abc"
    nested = job.scratch / "deeper" / "paper-orchestra-job-abc"
    foreign = job.scratch / "paper-orchestra-job-foreign"
    for directory in (elsewhere, nested, foreign):
        (directory / "work").mkdir(parents=True)
        (directory / "work" / "notes.txt").write_text("kept")
    # A process that carries the mark of such a directory is not a job's either.
    marked = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, local.JOB_ENV: str(elsewhere)},
        start_new_session=True,
    )
    try:
        for directory in (elsewhere, nested):
            [note] = LocalBackend.end_job(job_record(directory))
            assert (directory / "work" / "notes.txt").read_text() == "kept"
            assert str(directory) in note and "Nothing was stopped or removed" in note
            assert str(job.scratch) in note and "remove it yourself" in note
            assert LocalBackend.job_processes(job_record(directory)) == set()
        assert marked.poll() is None

        with monkeypatch.context() as patch:
            patch.setattr(os, "geteuid", lambda: os.getuid() + 1)  # as if somebody else's
            assert LocalBackend.end_job(job_record(foreign)) == []
        assert (foreign / "work" / "notes.txt").read_text() == "kept"
        # A spelling that leads to it by another way is not the name it was made under.
        assert LocalBackend.end_job(job_record(f"{foreign}/")) == []
        assert LocalBackend.end_job(job_record(f"{foreign}/../{foreign.name}")) == []
        assert LocalBackend.end_job(job_record(os.path.relpath(foreign))) == []
        assert (foreign / "work" / "notes.txt").read_text() == "kept"

        # A backend that was told to make its working directories elsewhere knows them.
        there = LocalBackend(scratch_dir=elsewhere.parent)
        assert there.job_processes(job_record(elsewhere)) == {marked.pid}
        notes = there.end_job(job_record(elsewhere))
        assert marked.wait(timeout=10) == -signal.SIGKILL
        assert not elsewhere.exists() and notes[0].startswith("Stopped 1 process ")
    finally:
        marked.kill()
        marked.wait()
        shutil.rmtree(job.scratch / "deeper", ignore_errors=True)
        shutil.rmtree(foreign, ignore_errors=True)


@on_linux
def test_what_a_leftover_starts_while_the_others_are_ended_is_ended_too(job, monkeypatch):
    """One look is not enough: what a process started just before it was killed is found
    by the next look, as it is when a job that runs is ended."""
    scratch = job.scratch / "paper-orchestra-job-forking"
    scratch.mkdir()
    env = {**os.environ, local.JOB_ENV: str(scratch)}
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    first = subprocess.Popen(sleeper, env=env, start_new_session=True)
    late: list[subprocess.Popen] = []
    really_kill = local._kill

    def kill(pid: int) -> bool:
        if not late:  # as if the process had started this one the moment before it died
            late.append(subprocess.Popen(sleeper, env=env, start_new_session=True))
        return really_kill(pid)

    monkeypatch.setattr(local, "_kill", kill)
    try:
        notes = LocalBackend.end_job(job_record(scratch))
        assert first.wait(timeout=10) == -signal.SIGKILL
        assert late[0].wait(timeout=10) == -signal.SIGKILL, "what it started was left running"
        assert notes[0].startswith("Stopped 2 processes") and not scratch.exists()
    finally:
        for process in (first, *late):
            process.kill()
            process.wait()
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.mark.skipif(os.name != "posix", reason="a working directory is held by a kernel lock")
def test_a_directory_that_cannot_be_looked_into_is_not_taken_for_one_nobody_uses(job, monkeypatch):
    """Whether a job is running in a working directory is told by the lock on it.  When
    the directory could not even be opened to try that lock (this process had no file
    descriptor left), nothing was known; that was taken for "nobody is there", and the
    directory was removed."""
    scratch = job.scratch / "paper-orchestra-job-unknown"
    (scratch / "work").mkdir(parents=True)
    record = job_record(scratch)
    really_open = os.open
    refused: list[int] = []

    def opening(failure: int, times: int):
        def open_(path, flags, *args, **kwargs):
            if os.fspath(path) == str(scratch) and len(refused) < times:
                refused.append(failure)
                raise OSError(failure, os.strerror(failure), os.fspath(path))
            return really_open(path, flags, *args, **kwargs)

        return open_

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", opening(errno.EMFILE, times=100))
        [note] = LocalBackend.end_job(record)
        assert LocalBackend.job_processes(record) == set()
    assert (scratch / "work").is_dir(), "a directory nothing was known about was removed"
    assert str(scratch) in note and os.strerror(errno.EMFILE) in note
    assert "Nothing was stopped or removed" in note and "remove it yourself" in note

    # A job that took the access to its own directory away has still left it behind: it
    # is the person's own directory, and is opened for them again.
    refused.clear()
    scratch.chmod(0o000)
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", opening(errno.EACCES, times=1))
        assert LocalBackend.end_job(record) == [
            f"Removed the working directory {scratch} that the run had left behind."
        ]
    assert refused == [errno.EACCES] and not scratch.exists()

    # One that is gone meanwhile is nothing to hold, and nothing to say anything about.
    assert local.lock_directory(str(scratch)) is None
    (job.scratch / "paper-orchestra-job-link").symlink_to(job.root, target_is_directory=True)
    assert local.lock_directory(str(job.scratch / "paper-orchestra-job-link")) is None
    (job.scratch / "paper-orchestra-job-link").unlink()


@pytest.mark.skipif(os.name != "posix", reason="a working directory is held by a kernel lock")
def test_a_job_runs_also_where_its_working_directory_cannot_be_held(backend, job, monkeypatch):
    """The lock is there for whoever cleans up later.  A job does not fail for want of it."""
    really_open = os.open

    def open_(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY and os.path.basename(os.fspath(path)).startswith(
            local.SCRATCH_PREFIX
        ):
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE), os.fspath(path))
        return really_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_)
    result, seen = run_main(backend, job, job.source("print('ran')\n"))
    assert result.ok and seen == ["ran"], result.error


@on_linux
def test_a_process_that_is_already_being_killed_is_not_one_that_was_stopped(job, monkeypatch):
    """The command of a job dies with its worker, a moment after it.  Whether it had
    finished dying when the run was settled used to decide how many processes the run
    was said to have left running."""
    # A process on which the kill signal is pending reads like this one, which was
    # killed and not collected yet; one that runs does not.
    killed = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        killed.kill()
        os.waitid(os.P_PID, killed.pid, os.WEXITED | os.WNOWAIT)
        assert local._being_killed(killed.pid) and not local._being_killed(running.pid)
        assert not local._being_killed(os.getpid())
    finally:
        for process in (killed, running):
            process.kill()
            process.wait()
    assert not local._being_killed(killed.pid)  # nor one that is gone

    scratch = job.scratch / "paper-orchestra-job-dying"
    scratch.mkdir()
    env = {**os.environ, local.JOB_ENV: str(scratch)}
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    dying = subprocess.Popen(sleeper, env=env, start_new_session=True)
    left = subprocess.Popen(sleeper, env=env, start_new_session=True)
    signalled: list[int] = []

    def kill(pid: int) -> bool:
        signalled.append(pid)
        os.kill(pid, signal.SIGKILL)
        return True

    really = local._being_killed
    monkeypatch.setattr(local, "_being_killed", lambda pid: pid == dying.pid or really(pid))
    monkeypatch.setattr(local, "_kill", kill)
    monkeypatch.setattr(local, "_REAP_SECONDS", 0.5)
    try:
        assert LocalBackend.job_processes(job_record(scratch)) == {left.pid}
        notes = LocalBackend.end_job(job_record(scratch))
        assert signalled == [left.pid] and dying.poll() is None
        assert notes[0].startswith("Stopped 1 process that the run had left running")
    finally:
        for process in (dying, left):
            process.kill()
            process.wait()
        shutil.rmtree(scratch, ignore_errors=True)


COPYING_WORKER = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from paper_orchestra.compute import JobSpec, LocalBackend\n"
    "def should_stop():\n"
    "    # Asked while the source is copied: the copy takes its time.\n"
    "    Path(sys.argv[4]).write_text('copying')\n"
    "    time.sleep(120)\n"
    "LocalBackend(scratch_dir=sys.argv[3]).run_job(\n"
    "    JobSpec(command=['python', 'main.py'], timeout=300),\n"
    "    source_dir=Path(sys.argv[1]), run_dir=Path(sys.argv[2]), should_stop=should_stop)\n"
)


@pytest.mark.skipif(os.name != "posix", reason="a working directory is held by a kernel lock")
def test_a_worker_killed_while_the_source_is_copied_leaves_a_record_to_clean_up_by(job):
    """A repository of gigabytes takes a while to copy.  The copy that a worker leaves
    half-made when it is killed is as large as the one of a job that had started, and
    the record next to the run names it from the moment it exists."""
    source = job.source("print('never reached')\n", **{"data/table.csv": "1,2,3\n"})
    copying = job.root / "copying"
    command = [sys.executable, "-c", COPYING_WORKER]
    worker = subprocess.Popen([*command, str(source), str(job.run), str(job.scratch), str(copying)])
    try:
        deadline = time.monotonic() + 30
        while not copying.exists():
            assert time.monotonic() < deadline and worker.poll() is None
            time.sleep(0.05)
        [scratch] = job.scratch.iterdir()
        record = json.loads((job.run / "job.json").read_text())
        # No process of the job exists yet: the record names no session, and says where
        # the directory is.
        assert record == {"session": 0, "session_started": "", "scratch": str(scratch)}
        # While the worker lives, the directory is the one it is copying into, whoever
        # holds a record that names it: no process says so yet, the backend's lock does.
        [note] = LocalBackend.end_job(record)
        assert "A job is running in it now" in note and (scratch / "work").is_dir()
    finally:
        worker.kill()
        worker.wait()

    notes = LocalBackend.end_job(record)
    assert not scratch.exists()
    assert notes == [f"Removed the working directory {scratch} that the run had left behind."]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="looks processes up in /proc")
def test_what_a_detached_helper_started_is_ended_with_it(backend, job):
    """A helper that left the job's session is found by the job's mark.  What it started
    in its own session without the mark (``env -i``) is the job's just the same."""
    source = job.source(
        """
        import subprocess, sys, time
        inner = "import subprocess, sys, time\\n"
        inner += "child = subprocess.Popen(['env', '-i', sys.executable, '-c', "
        inner += "'import time; time.sleep(300)'])\\n"
        inner += "print(child.pid, flush=True)\\ntime.sleep(300)\\n"
        helper = subprocess.Popen(
            [sys.executable, "-c", inner],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print(helper.pid, int(helper.stdout.readline()), flush=True)
        helper.stdout.close()
        if "--stay" in sys.argv:
            time.sleep(300)
        """
    )
    result, seen = run_main(backend, job, source)
    helper, unmarked = (int(pid) for pid in seen[0].split())
    try:
        assert result.ok
        assert_gone(helper)
        assert_gone(unmarked)
        [note] = [note for note in result.notes if "outside its own session" in note]
        assert note.startswith("2 processes")
    finally:
        for pid in (helper, unmarked):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)

    # The same when the worker was killed and the run is cleaned up from its record.
    (job.root / "code" / "main.py").write_text(
        (job.root / "code" / "main.py").read_text().replace('"--stay" in sys.argv', "True")
    )
    (job.run / "log.txt").unlink()  # the first run's: the next worker starts a new one
    main_line = killed_worker(job, job.root / "code", 1)[0]
    helper, unmarked = (int(pid) for pid in main_line.split())
    try:
        assert alive(helper) and alive(unmarked), "the scenario needs survivors"
        notes = LocalBackend.end_job(json.loads((job.run / "job.json").read_text()))
        assert_gone(helper)
        assert_gone(unmarked)
        assert notes[0].startswith("Stopped 2 processes")
    finally:
        for pid in (helper, unmarked):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        for leftover in job.scratch.iterdir():
            shutil.rmtree(leftover, ignore_errors=True)


def test_outputs_that_came_with_the_source_are_not_results_of_the_run(backend, job):
    """A repository often holds a metrics.json and figures of earlier work."""
    source = job.source(
        """
        import json, os
        print("stale files in the working directory:", sorted(
            name for name in ("metrics.json", "artifacts") if os.path.lexists(name)))
        os.makedirs("out")
        json.dump({"accuracy": 0.5}, open("out/metrics.json", "w"))
        """,
        **{
            "metrics.json": '{"accuracy": 0.97, "note": "last month, other model"}',
            "artifacts/old_curve.png": "an old figure",
        },
    )
    result, seen = run_main(backend, job, source)

    assert result.ok  # the command did end well; whether it measured anything is for the caller
    assert "stale files in the working directory: []" in seen
    assert not (job.run / "metrics.json").exists() and not (job.run / "artifacts").exists()
    [note] = [note for note in result.notes if "that was there before the run" in note]
    assert "removed artifacts, metrics.json" in note and f"[paper-orchestra] {note}" in seen
    # The person's files are where they were.
    assert (source / "metrics.json").exists() and (source / "artifacts" / "old_curve.png").exists()

    # What the command writes under those names is its own, and is collected as before.
    source = job.source(
        """
        import json, os
        os.makedirs("artifacts")
        open("artifacts/new_curve.png", "w").write("new")
        json.dump({"accuracy": 0.5}, open("metrics.json", "w"))
        """
    )
    result, _ = run_main(backend, job, source)
    assert json.loads((job.run / "metrics.json").read_text()) == {"accuracy": 0.5}
    assert [p.name for p in (job.run / "artifacts").iterdir()] == ["new_curve.png"]
    assert any("that was there before the run" in note for note in result.notes)


MASKING_WORKER = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "from paper_orchestra.compute import JobSpec, LocalBackend\n"
    "result = LocalBackend(scratch_dir=sys.argv[3]).run_job(\n"
    "    JobSpec(command=['python', 'main.py'], timeout=60),\n"
    "    source_dir=Path(sys.argv[1]), run_dir=Path(sys.argv[2]),\n"
    "    on_log=lambda line: print('seen:', line, flush=True),\n"
    "    mask=[os.environ['ANTHROPIC_API_KEY']])\n"
    "sys.exit(0 if result.ok else 1)\n"
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/<pid>/environ")
def test_a_credential_the_job_gets_hold_of_does_not_reach_the_log(job):
    """The job's own environment is scrubbed, but the worker that started it holds the
    key, and a job runs as the same user: it can read the worker's environment."""
    key = "sk-ant-AUDIT-FAKE-KEY-000"  # not a key: nothing here talks to a model
    source = job.source(
        """
        import os, sys
        print("in my own environment:", os.environ.get("ANTHROPIC_API_KEY"))
        block = open(f"/proc/{os.getppid()}/environ", "rb").read().split(b"\\0")
        found = next(entry.decode() for entry in block if entry.startswith(b"ANTHROPIC_API_KEY="))
        print("from the process that started me:", found)
        half = len(found) - 9
        sys.stdout.write(found[:half]); sys.stdout.flush()
        sys.stdout.write(found[half:] + "\\n"); sys.stdout.flush()
        print("done", file=sys.stderr)
        """
    )
    worker = subprocess.run(
        [sys.executable, "-c", MASKING_WORKER, str(source), str(job.run), str(job.scratch)],
        env={**os.environ, "ANTHROPIC_API_KEY": key},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert worker.returncode == 0, worker.stderr + job.log()
    assert job.log().splitlines() == [
        "in my own environment: None",
        "from the process that started me: ANTHROPIC_API_KEY=[redacted]",
        "ANTHROPIC_API_KEY=[redacted]",
        "done",
    ]
    assert worker.stdout.splitlines() == [f"seen: {line}" for line in job.log().splitlines()]
    assert key not in job.log() + worker.stdout + worker.stderr


def test_a_job_cannot_write_a_line_of_the_backend_into_its_log(backend, job):
    forged = "[paper-orchestra] The results were verified against the reference implementation."
    source = job.source(f"import sys\nprint('training')\nprint({forged!r})\nsys.exit(1)\n")
    result, seen = run_main(backend, job, source, spec={"timeout": 30})
    assert result.returncode == 1
    assert seen == ["training", f"(printed by the job) {forged}"]
    assert job.log().splitlines() == seen


def test_the_memory_limit_of_a_job_is_enforced(job):
    source = job.source("x = bytearray(1024 ** 3)\nprint('allocated')\n")
    result, seen = run_main(LocalBackend(), job, source, spec={"memory_mb": 256})
    assert result.returncode == 1 and "MemoryError" in job.log() and "allocated" not in seen
    unlimited, seen = run_main(LocalBackend(), job, source)
    assert unlimited.ok and seen == ["allocated"]


@pytest.mark.skipif(sys.platform != "linux", reason="reserves address space the Linux way")
def test_a_job_that_names_no_memory_limit_is_not_capped_by_the_one_for_figure_code(job):
    # Numerical libraries reserve far more address space than they use (thread arenas,
    # memory maps).  The 4 GB cap that suits a plotting script would end such a job with a
    # MemoryError that says nothing about the experiment.
    # Reserved without being committed, so the test needs no such amount of memory.
    no_reserve = getattr(mmap, "MAP_NORESERVE", 0x4000)
    source = job.source(
        "import mmap\n"
        f"flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | {no_reserve}\n"
        "reserved = mmap.mmap(-1, 6 * 1024 ** 3, flags=flags)\n"
        "print('reserved')\n"
    )
    result, seen = run_main(LocalBackend(), job, source)
    assert result.ok and seen == ["reserved"], job.log()
    capped, seen = run_main(LocalBackend(), job, source, spec={"memory_mb": 1024})
    assert capped.returncode == 1 and "reserved" not in seen


def test_supports_says_why_a_job_cannot_run_here(backend):
    assert backend.supports(JobSpec(command=["python", "main.py"])) == []
    assert backend.supports(JobSpec(command=["python", "main.py"], network=True, cpu=2)) == []

    [gpu] = backend.supports(JobSpec(command=["python", "main.py"], gpu="T4"))
    assert "GPU (T4)" in gpu and "Modal backend" in gpu

    installed = ("numpy", "pillow>=10", "PyYAML", "matplotlib[all]==3.*")
    assert backend.supports(JobSpec(command=["x"], packages=installed)) == []
    wanted = (*installed, "definitely-not-installed>=2", "scikit-not-here")
    [missing] = backend.supports(JobSpec(command=["x"], packages=wanted))
    assert "definitely-not-installed>=2, scikit-not-here" in missing
    assert "numpy" not in missing and "pillow" not in missing
    assert "not installed" in missing and "Modal backend" in missing and "installs" in missing

    both = backend.supports(JobSpec(command=[], gpu="T4", packages=("nope-nope",)))
    assert len(both) == 3  # the malformed command, the GPU and the package


def test_import_names_come_from_the_installed_distribution_or_a_guess():
    from paper_orchestra.compute.local import missing_packages

    assert missing_packages(["Pillow", "pyyaml", "numpy>=1"]) == []
    assert missing_packages(["scikit-nothing", "not_there==1.0", "-r nope.txt"]) == [
        "scikit-nothing",
        "not_there==1.0",
    ]


def test_a_job_the_backend_cannot_run_is_refused_with_the_reasons(backend, job):
    source = job.source("open('ran.txt', 'w').write('x')\n")
    seen: list[str] = []
    spec = JobSpec(command=["python", "main.py"], gpu="T4", outputs=("ran.txt",))
    result = backend.run_job(spec, source_dir=source, run_dir=job.run, on_log=seen.append)
    assert result.returncode is None and not result.ok and result.seconds == 0
    assert result.error.startswith("The local backend cannot run this job.")
    assert "GPU (T4)" in result.error
    assert seen == [f"[paper-orchestra] {result.error}"] and result.error in job.log()
    assert not (job.run / "ran.txt").exists()

    for outputs in (("../escape",), ("log.txt",), ("/etc/hostname",)):
        spec = JobSpec(command=["python", "main.py"], outputs=outputs)
        refused = backend.run_job(spec, source_dir=source, run_dir=job.run)
        assert refused.returncode is None and repr(outputs[0]) in refused.error

    missing = backend.run_job(
        JobSpec(command=["python", "main.py"]), source_dir=job.root / "nowhere", run_dir=job.run
    )
    assert "does not exist" in missing.error and str(job.root / "nowhere") in missing.error


def test_a_run_directory_that_cannot_be_written_raises(backend, job):
    blocker = job.root / "file"
    blocker.write_text("x")
    with pytest.raises(ComputeError, match="run directory"):
        backend.run_job(JobSpec(command=["true"]), source_dir=job.source(), run_dir=blocker / "run")


def test_working_directories_are_made_where_the_backend_is_told_to(job, tmp_path):
    scratch = tmp_path / "big-disk" / "scratch"
    backend = LocalBackend(scratch_dir=scratch)
    source = job.source("import os\nprint(os.getcwd())\n")
    result, seen = run_main(backend, job, source)
    assert result.ok and Path(seen[0]).parent.parent == scratch
    assert list(scratch.iterdir()) == []

    blocker = tmp_path / "file"
    blocker.write_text("x")
    refused, _ = run_main(LocalBackend(scratch_dir=blocker / "scratch"), job, source)
    assert refused.returncode is None
    assert "could not make a working directory" in refused.error


def test_local_describe_says_what_jobs_can_ask_for(backend):
    jobs = backend.describe()["jobs"]
    assert {key: jobs[key] for key in ("supported", "gpu", "network_isolation", "packages")} == {
        "supported": True,
        "gpu": False,
        "network_isolation": False,
        "packages": False,
    }
    assert jobs["reattach"] is False and "not a sandbox" in jobs["summary"]
    with pytest.raises(ComputeError, match="cannot be picked up again"):
        backend.attach("anything", run_dir=Path("unused"))


def test_run_python_and_compile_latex_signatures_are_unchanged():
    assert str(inspect.signature(LocalBackend.run_python)) == (
        "(self, code: 'str', *, inputs: 'Mapping[str, bytes] | None' = None, "
        "outputs: 'Sequence[str]' = (), timeout: 'float' = 120.0) -> 'ExecResult'"
    )
    assert str(inspect.signature(LocalBackend.compile_latex)) == (
        "(self, workdir: 'str | Path', tex_name: 'str', *, "
        "output_pdf: 'str | Path | None' = None, timeout: 'float' = 240.0) -> 'CompileResult'"
    )
