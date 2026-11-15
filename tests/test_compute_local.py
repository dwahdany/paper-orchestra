from __future__ import annotations

import contextlib
import inspect
import itertools
import json
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
    assert sorted(p.name for p in job.run.iterdir()) == ["artifacts", "log.txt", "metrics.json"]
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
    result, seen = run_main(backend, job, source)
    assert not result.ok and result.returncode == 3 and result.error is None
    assert seen == ["partial"]
    # Nothing an earlier run left behind passes for the output of this one.
    assert sorted(p.name for p in job.run.iterdir()) == ["log.txt"]

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
    assert copied == ["artifacts", "artifacts/real.txt", "log.txt"]
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
    assert sorted(p.name for p in job.run.iterdir()) == ["log.txt"]
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


@pytest.mark.skipif(os.name != "posix", reason="starts a session of its own")
def test_output_that_an_escaped_process_holds_open_is_given_up_and_reported(
    backend, job, monkeypatch
):
    """A daemon in a session of its own is out of reach: the job still ends, and the
    person is told that something was left behind."""
    monkeypatch.setattr(base, "DRAIN_SECONDS", 0.5)
    source = job.source(
        """
        import os, time
        ready, tell = os.pipe()
        if os.fork() == 0:
            os.setsid()
            print(os.getpid(), flush=True)
            os.write(tell, b"x")
            os.close(tell)
            time.sleep(60)
            os._exit(0)
        os.close(tell)
        os.read(ready, 1)
        print("the command is done", flush=True)
        """
    )
    started = time.monotonic()
    result, seen = run_main(backend, job, source)
    daemon = int(seen[0])
    try:
        assert result.ok and time.monotonic() - started < 10
        assert "the command is done" in seen
        [note] = [note for note in result.notes if "kept its output open" in note]
        assert note in job.log()
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
    # Nobody was left to remove the working directory; that is the one thing a kill costs.
    [leftover] = job.scratch.iterdir()
    assert leftover.name.startswith("paper-orchestra-job-")
    shutil.rmtree(leftover)


def test_the_memory_limit_of_a_job_is_enforced(job):
    source = job.source("x = bytearray(1024 ** 3)\nprint('allocated')\n")
    result, seen = run_main(LocalBackend(), job, source, spec={"memory_mb": 256})
    assert result.returncode == 1 and "MemoryError" in job.log() and "allocated" not in seen
    unlimited, seen = run_main(LocalBackend(memory_limit_mb=None), job, source)
    assert unlimited.ok and seen == ["allocated"]


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
