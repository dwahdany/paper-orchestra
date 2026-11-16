"""The job contract and the plumbing both backends share (``compute/base.py``)."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path

import pytest

import paper_orchestra.compute.base as base
from paper_orchestra.compute import (
    ComputeBackend,
    ComputeError,
    JobResult,
    JobSpec,
)
from paper_orchestra.compute.base import (
    NOTE_PREFIX,
    JobLog,
    OutputEntry,
    choose_outputs,
    delivered_lines,
    open_run,
    requirement_name,
    scan_outputs,
    spec_problems,
    supervise,
)

#: What the log puts before a line of the job that begins like a line of the backend.
JOB_MARK = "(printed by the job) "

# -- the contract ------------------------------------------------------------------------


def test_job_spec_and_result_have_the_fields_and_defaults_of_the_contract():
    spec = JobSpec(command=["python", "main.py"])
    assert dataclasses.asdict(spec) == {
        "command": ["python", "main.py"],
        "timeout": 900.0,
        "gpu": None,
        "network": False,
        "packages": (),
        "env": {},
        "cpu": None,
        "memory_mb": None,
        "outputs": ("metrics.json", "artifacts"),
    }
    assert JobSpec(command=["a"]).env is not JobSpec(command=["b"]).env

    result = JobResult(returncode=0)
    contract = {
        "returncode": 0,
        "timed_out": False,
        "cancelled": False,
        "seconds": 0.0,
        "log_path": None,
        "outputs_dir": None,
        "backend": "",
        "handle": None,
        "cost_usd": None,
        "error": None,
    }
    described = dataclasses.asdict(result)
    assert {key: described[key] for key in contract} == contract
    # The one addition: what the person should know that is neither output nor a failure.
    assert set(described) - set(contract) == {"notes"} and result.notes == []


@pytest.mark.parametrize(
    ("result", "ok"),
    [
        (JobResult(returncode=0), True),
        (JobResult(returncode=3), False),
        (JobResult(returncode=None), False),
        (JobResult(returncode=0, timed_out=True), False),
        (JobResult(returncode=0, cancelled=True), False),
        (JobResult(returncode=0, error="the sandbox was lost"), False),
    ],
)
def test_a_result_is_ok_only_when_the_job_itself_ended_well(result, ok):
    assert result.ok is ok


def test_the_signatures_are_the_ones_of_the_specification():
    run_job = inspect.signature(ComputeBackend.run_job)
    assert list(run_job.parameters) == [
        "self",
        "spec",
        "source_dir",
        "run_dir",
        "on_log",
        "should_stop",
        "mask",
    ]
    keyword_only = [p.name for p in run_job.parameters.values() if p.kind is p.KEYWORD_ONLY]
    assert keyword_only == ["source_dir", "run_dir", "on_log", "should_stop", "mask"]
    assert run_job.parameters["on_log"].default is None
    assert run_job.parameters["should_stop"].default is None
    assert run_job.parameters["mask"].default == ()
    assert list(inspect.signature(ComputeBackend.supports).parameters) == ["self", "spec"]


def test_every_backend_takes_a_job_and_picks_one_up_with_the_same_arguments():
    """The seam with the engine: what it passes is fixed, whichever backend it talks to."""
    from paper_orchestra.compute.local import LocalBackend
    from paper_orchestra.compute.modal_backend import ModalBackend

    for backend in (ComputeBackend, LocalBackend, ModalBackend):
        run_job = inspect.signature(backend.run_job)
        assert list(run_job.parameters)[1:] == [
            "spec",
            "source_dir",
            "run_dir",
            "on_log",
            "should_stop",
            "mask",
        ]
        attach = inspect.signature(backend.attach)
        assert list(attach.parameters) == [
            "self",
            "handle",
            "run_dir",
            "spec",
            "started",
            "on_log",
            "should_stop",
            "mask",
        ]
        assert [p.name for p in attach.parameters.values() if p.kind is p.KEYWORD_ONLY] == [
            "run_dir",
            "spec",
            "started",
            "on_log",
            "should_stop",
            "mask",
        ]
        defaults = {name: attach.parameters[name].default for name in ("spec", "started", "mask")}
        assert defaults == {"spec": None, "started": None, "mask": ()}


def test_a_backend_without_job_support_says_so(tmp_path):
    class FiguresOnly(ComputeBackend):
        name = "figures"

        def run_python(self, code, *, inputs=None, outputs=(), timeout=120.0):
            raise NotImplementedError

        def compile_latex(self, workdir, tex_name, *, output_pdf=None, timeout=240.0):
            raise NotImplementedError

    backend = FiguresOnly()
    spec = JobSpec(command=["python", "main.py"])
    [reason] = backend.supports(spec)
    assert "figures" in reason and "does not run experiment jobs" in reason
    with pytest.raises(ComputeError, match="does not run experiment jobs"):
        backend.run_job(spec, source_dir=tmp_path, run_dir=tmp_path / "run")
    with pytest.raises(ComputeError, match="cannot be picked up again"):
        backend.attach("sb-1", run_dir=tmp_path / "run")
    assert backend.describe()["jobs"] == {
        "supported": False,
        "gpu": False,
        "network_isolation": False,
        "packages": False,
        "reattach": False,
    }


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"command": []}, "no command"),
        ({"command": ["python", 3]}, "command"),
        ({"command": [""]}, "command"),
        ({"timeout": 0}, "time limit"),
        ({"timeout": float("inf")}, "time limit"),
        ({"timeout": float("nan")}, "time limit"),
        ({"outputs": ("../escape",)}, "'../escape'"),
        ({"outputs": ("/etc/passwd",)}, "'/etc/passwd'"),
        ({"outputs": ("log.txt",)}, "'log.txt'"),
        ({"outputs": ("run.json/inner",)}, "'run.json/inner'"),
        ({"outputs": ("handle.txt",)}, "'handle.txt'"),
        # The engine's snapshot of the code that ran, its copy of the repository and the
        # backend's own records: a job that could name them as outputs could rewrite them.
        ({"outputs": ("code",)}, "'code'"),
        ({"outputs": ("code/main.py",)}, "'code/main.py'"),
        ({"outputs": ("repo",)}, "'repo'"),
        ({"outputs": ("cost.json",)}, "'cost.json'"),
        ({"outputs": ("job.json",)}, "'job.json'"),
        ({"outputs": ("bad\0name",)}, "'bad\\x00name'"),
        ({"outputs": "metrics.json"}, "outputs of the job must be a list"),
        ({"packages": "numpy"}, "packages of the job must be a list"),
        ({"env": {"BAD-NAME": "1"}}, "'BAD-NAME'"),
        ({"env": {"1X": "1"}}, "'1X'"),
        ({"env": {"OK": 5}}, "'OK'"),
        ({"cpu": 0}, "cpu"),
        ({"memory_mb": 8}, "memory"),
        ({"gpu": "T4; rm -rf"}, "GPU"),
        ({"gpu": "T4:0"}, "GPU"),
        ({"packages": ("--index-url=https://evil.example",)}, "'--index-url=https://evil.example'"),
        ({"packages": ("numpy; import os",)}, "'numpy; import os'"),
        ({"packages": ("pkg @ https://example.org/x.whl",)}, "'pkg @ https://example.org/x.whl'"),
        ({"packages": ("-r requirements.txt",)}, "'-r requirements.txt'"),
    ],
)
def test_malformed_specs_are_described_for_a_person(changes, expected):
    spec = dataclasses.replace(JobSpec(command=["python", "main.py"]), **changes)
    problems = spec_problems(spec)
    assert len(problems) == 1 and expected in problems[0], problems
    assert problems[0][0].isupper() and problems[0].endswith(".")


def test_well_formed_specs_have_no_problems():
    spec = JobSpec(
        command=["python", "main.py", "--seed", "1"],
        timeout=3600,
        gpu="A100-80GB:2",
        network=True,
        packages=("numpy", "torch>=2.4,<3", "scikit-learn==1.5.*", "pandas[excel] ~= 2.2"),
        env={"SEED": "1", "_private": ""},
        cpu=0.5,
        memory_mb=512,
        outputs=("metrics.json", "artifacts", "nested/dir/table.csv"),
    )
    assert spec_problems(spec) == []
    assert spec_problems(JobSpec(command=["x"], gpu="h100!")) == []


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("numpy", "numpy"),
        ("  scikit-learn >= 1.3 ", "scikit-learn"),
        ("pandas[excel]~=2.2", "pandas"),
        ("Pillow==10.*", "Pillow"),
        ("ruamel.yaml<1", "ruamel.yaml"),
        ("-e .", None),
        ("git+https://example.org/x.git", None),
        ("", None),
        ("numpy scipy", None),
    ],
)
def test_requirement_name(requirement, name):
    assert requirement_name(requirement) == name


# -- the run directory -------------------------------------------------------------------


def test_open_run_creates_the_directory_and_removes_outputs_of_an_earlier_run(tmp_path):
    run = tmp_path / "runs" / "1"
    assert open_run(run, ("metrics.json", "artifacts")) == run / "log.txt"
    assert run.is_dir()

    # What an earlier attempt left must not pass for what this run produced.
    (run / "metrics.json").write_text('{"accuracy": 0.99}')
    (run / "artifacts" / "deep").mkdir(parents=True)
    (run / "artifacts" / "deep" / "old.png").write_bytes(b"old")
    (run / "run.json").write_text("{}")
    (run / "log.txt").write_text("old log")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (run / "linked").symlink_to(outside, target_is_directory=True)

    # Nor may the backend's records of an earlier attempt pass for this one's.
    for name in ("handle.txt", "cost.json", "job.json"):
        (run / name).write_text("of the earlier attempt")
    (run / "code").mkdir()

    open_run(run, ("metrics.json", "artifacts", "linked", "never-there"))
    assert sorted(p.name for p in run.iterdir()) == ["code", "log.txt", "run.json"]
    (run / "code").rmdir()
    assert (run / "log.txt").read_bytes() == b""
    assert (outside / "keep.txt").read_text() == "keep"  # the link went, not what it led to

    # Names that a job may not use as outputs are not removed either.
    (run / "m").write_text("kept")
    open_run(run, "m")  # a bare string is not a list of one-letter outputs
    open_run(run, ("run.json", "../outside", "log.txt"))
    assert sorted(p.name for p in run.iterdir()) == ["log.txt", "m", "run.json"]
    assert (outside / "keep.txt").exists()


def test_open_run_reports_a_directory_it_cannot_use(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ComputeError, match="run directory") as raised:
        open_run(blocker / "run", ())
    assert str(blocker) in str(raised.value)


# -- the log -----------------------------------------------------------------------------


def lines_of(chunks: list[bytes], **options) -> tuple[list[str], str]:
    """What ``on_log`` receives and what the log holds after feeding ``chunks``."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.txt"
        seen: list[str] = []
        with JobLog(path, seen.append, **options) as log:
            for chunk in chunks:
                log.feed(chunk)
        return seen, path.read_bytes().decode("utf-8")  # no newline translation


def test_output_reaches_the_file_and_the_callback_as_it_is_produced(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    with JobLog(path, seen.append) as log:
        log.feed(b"epoch 1\nepo")
        # A person tailing the file sees everything so far, including the unfinished line.
        assert path.read_bytes() == b"epoch 1\nepo"
        assert seen == ["epoch 1"]
        log.feed(b"ch 2\n")
        assert path.read_bytes() == b"epoch 1\nepoch 2\n"
        assert seen == ["epoch 1", "epoch 2"]
        assert log.raw_bytes == 16
    assert seen == ["epoch 1", "epoch 2"]


def test_output_that_never_ends_with_a_newline_is_still_delivered():
    seen, text = lines_of([b"working", b"... done"])
    assert seen == ["working... done"] and text == "working... done"
    assert lines_of([]) == ([], "")


def test_binary_output_becomes_valid_text():
    blob = bytes(range(256)) * 4
    seen, text = lines_of([blob[:300], blob[300:]])
    assert "\ufffd" in text and "ABC" in text
    assert all(isinstance(line, str) for line in seen)
    # A multi-byte character cut by a chunk boundary is not damaged.
    snowman = "snow \u2603 man\n".encode()
    assert lines_of([snowman[:6], snowman[6:]]) == (["snow \u2603 man"], "snow \u2603 man\n")
    # ...and one cut by the end of the output is replaced, not dropped silently.
    assert lines_of([snowman[:6]])[1] == "snow \ufffd"


def test_progress_bars_and_endless_lines_reach_the_callback_in_pieces(monkeypatch):
    seen, text = lines_of([b"10%\r20%\r30%\r\ndone\n\n\nend\n"])
    assert seen == ["10%", "20%", "30%", "done", "end"]  # blank lines are not events
    assert text == "10%\r20%\r30%\r\ndone\n\n\nend\n"  # the file keeps the output as it was

    monkeypatch.setattr(base, "MAX_LINE_CHARS", 10)
    seen, text = lines_of([b"x" * 25, b"y" * 7 + b"\nshort\n"])
    assert seen == ["x" * 10, "x" * 10, "xxxxxyyyyy", "yy", "short"]
    assert text == "x" * 25 + "y" * 7 + "\nshort\n"


def test_the_lines_do_not_depend_on_how_the_output_was_cut_into_chunks(monkeypatch):
    monkeypatch.setattr(base, "MAX_LINE_CHARS", 40)
    rng = random.Random(7)
    pieces = ["line\n", "x" * 130, "\r", "\r\n", "caf\u00e9 \u2603\n", "tail", "\n\n", "\U0001f600"]
    data = "".join(rng.choice(pieces) for _ in range(400)).encode("utf-8")
    whole = lines_of([data])
    for _ in range(25):
        cuts = sorted(rng.sample(range(len(data)), 60))
        chunks = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)], strict=True)]
        assert lines_of(chunks) == whole


def test_a_runaway_job_cannot_fill_the_disk_or_the_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "MAX_LOG_BYTES", 5_000)
    monkeypatch.setattr(base, "LOG_TAIL_BYTES", 1_000)
    path = tmp_path / "log.txt"
    seen: list[str] = []
    line = b"spam spam spam spam spam spam spam spam\n"  # 40 bytes
    tracemalloc.start()
    try:
        with JobLog(path, seen.append) as log:
            for _ in range(50_000):  # 2 MB
                log.feed(line)
            log.feed(b"the last line\n")
            log.note("The job was stopped.")
            assert path.stat().st_size <= 5_000
            _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    text = path.read_text(encoding="utf-8")
    assert len(text) < 5_000 + 1_000 + 600
    assert text.startswith("spam spam") and text.endswith(f"{NOTE_PREFIX}The job was stopped.\n")
    assert "the last line\n" in text
    [marker] = [row for row in text.splitlines() if "left out" in row]
    assert marker.startswith(NOTE_PREFIX) and "1.9 MB" in marker
    # The callback is told once and is then left alone: it feeds an activity stream.
    assert len(seen) < 200
    assert sum("no longer shown live" in row for row in seen) == 1
    assert seen[-1] == f"{NOTE_PREFIX}The job was stopped."
    assert log.truncated


def test_notes_of_the_backend_stand_on_their_own_lines(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    with JobLog(path, seen.append) as log:
        log.note("Starting.")
        log.feed(b"half a line")
        log.note("The job was stopped because it ran longer than its time limit of 5 s.")
        log.feed(b" and the rest\n")
    assert path.read_text() == (
        f"{NOTE_PREFIX}Starting.\n"
        "half a line\n"
        f"{NOTE_PREFIX}The job was stopped because it ran longer than its time limit of 5 s.\n"
        " and the rest\n"
    )
    assert seen == [
        f"{NOTE_PREFIX}Starting.",
        "half a line",
        f"{NOTE_PREFIX}The job was stopped because it ran longer than its time limit of 5 s.",
        " and the rest",
    ]


def test_a_log_works_without_a_callback_and_closes_twice(tmp_path):
    log = JobLog(tmp_path / "log.txt")
    log.feed(b"a\nb")
    log.note("n")
    log.close()
    log.close()
    assert (tmp_path / "log.txt").read_text() == f"a\nb\n{NOTE_PREFIX}n\n"


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs a device that is always full")
def test_a_log_that_cannot_be_written_is_an_error_of_the_run_directory():
    """A full disk is one of the few things ``run_job`` raises for, and it says which."""
    log = JobLog(Path("/dev/full"))
    with pytest.raises(ComputeError, match="/dev/full") as raised:
        log.feed(b"epoch 1\n")
    assert "cannot be written" in str(raised.value) and "No space left" in str(raised.value)
    with pytest.raises(ComputeError, match="cannot be written"):
        log.note("The job was stopped.")
    log.close()  # still closes, and still twice
    log.close()


def test_appending_keeps_what_an_earlier_process_wrote(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("epoch 1\nepo")
    with JobLog(path, append=True) as log:
        log.note("The sandbox is gone.")
    assert path.read_text() == f"epoch 1\nepo\n{NOTE_PREFIX}The sandbox is gone.\n"


def test_replaying_a_log_skips_the_lines_that_were_already_delivered(tmp_path):
    """Re-attaching rebuilds the log from the start but reports each line only once."""
    path = tmp_path / "log.txt"
    first: list[str] = []
    log = JobLog(path, first.append)
    log.feed(b"one\ntwo\n\nthr")
    log.note("a note of the backend")
    # The engine dies here: nothing is closed.
    log._file.flush()
    assert first == ["one", "two", "thr", f"{NOTE_PREFIX}a note of the backend"]
    # "thr" was pushed out by the note; without the note it would not have been delivered.
    assert delivered_lines(path) == 3
    assert delivered_lines(tmp_path / "missing.txt") == 0

    second: list[str] = []
    with JobLog(path, second.append, skip_lines=3) as replay:
        replay.feed(b"one\ntwo\n\nthr")
        replay.feed(b"ee\nfour\n")
    # "three" counts as the third line the person has seen (they saw its beginning).
    assert second == ["four"]
    assert path.read_text() == "one\ntwo\n\nthree\nfour\n"


# -- what a job prints is never a credential and never the backend speaking ---------------

SECRET = "sk-ant-api03-THIS-IS-A-SECRET"


def test_a_masked_string_never_reaches_the_file_or_the_callback(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    other = "ghp_tokenOfTheRepository"
    with JobLog(path, seen.append, mask=[SECRET, other, ""]) as log:
        log.feed(f"key: {SECRET}\n".encode())
        log.feed(f"twice {SECRET}{SECRET} and {other}\nclean\n".encode())
        log.note(f"The command could not be started: no such file {SECRET}.")
    text = path.read_text()
    assert SECRET not in text and other not in text
    assert text == (
        "key: ***\n"
        "twice ****** and ***\n"
        "clean\n"
        f"{NOTE_PREFIX}The command could not be started: no such file ***.\n"
    )
    assert seen == text.splitlines()


def test_a_masked_string_that_arrives_in_two_pieces_is_still_masked(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    data = f"before {SECRET} after\n".encode()
    with JobLog(path, seen.append, mask=[SECRET]) as log:
        for index in range(len(data)):  # one byte at a time: every possible cut
            log.feed(data[index : index + 1])
            assert SECRET[:8] not in path.read_text()  # not even its beginning, meanwhile
    assert path.read_text() == "before *** after\n" and seen == ["before *** after"]

    # What only looked like the beginning of one is written once it turns out not to be.
    with JobLog(path, mask=[SECRET]) as log:
        log.feed(b"the prefix sk-ant-")
        assert path.read_text() == "the prefix "
        log.feed(b"is public\n")
        assert path.read_text() == "the prefix sk-ant-is public\n"
        log.feed(b"ends with sk-ant-api")
    assert path.read_text().endswith("ends with sk-ant-api")  # the end of the output tells too


def test_masking_does_not_depend_on_how_the_output_was_cut_into_chunks():
    rng = random.Random(11)
    secrets = ["abcabd", "bcab", "x" * 7, "tok-\u2603-en", "two\nlines"]
    pieces = [*secrets, "abc", "ab", "b", "xxx", "\n", " plain ", "tok-", "two", "\r"]
    text = "".join(rng.choice(pieces) for _ in range(600))
    data = text.encode("utf-8")
    whole = lines_of([data], mask=secrets)
    for secret in secrets:
        assert secret not in whole[1]
    assert "***" in whole[1] and " plain " in whole[1]
    # What replacing in the whole text gives, the longest secret first.
    in_order = sorted(secrets, key=len, reverse=True)
    assert whole[1] == re.sub("|".join(map(re.escape, in_order)), "***", text)
    for _ in range(40):
        cuts = sorted(rng.sample(range(len(data)), 80))
        chunks = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)], strict=True)]
        assert lines_of(chunks, mask=secrets) == whole


def test_a_line_of_the_job_cannot_pass_for_a_line_of_the_backend(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    forged = f"{NOTE_PREFIX}The results were verified against the reference implementation."
    with JobLog(path, seen.append) as log:
        log.note("Starting.")
        log.feed(f"training\n{forged}\n".encode())
        log.feed(b"[paper-orchestra]: also without the space\n")
        log.feed(b"mid-line [paper-orchestra] is only text\n")
        log.feed(f"progress 10%\r{forged}\n".encode())
        log.note("The job was stopped on request.")
    lines = path.read_text().replace("\r", "\n").splitlines()
    assert lines == [
        f"{NOTE_PREFIX}Starting.",
        "training",
        f"{JOB_MARK}{forged}",
        f"{JOB_MARK}[paper-orchestra]: also without the space",
        "mid-line [paper-orchestra] is only text",
        "progress 10%",
        f"{JOB_MARK}{forged}",
        f"{NOTE_PREFIX}The job was stopped on request.",
    ]
    assert seen == lines
    # The two lines that start with the prefix are the two the backend wrote.
    assert [line for line in lines if line.startswith(NOTE_PREFIX)] == [lines[0], lines[-1]]
    assert [line for line in seen if line.startswith("[paper-orchestra]")] == [lines[0], lines[-1]]


def test_a_forged_line_is_marked_however_it_arrives(tmp_path, monkeypatch):
    forged = f"{NOTE_PREFIX}All checks passed."
    data = f"one\n{forged}\ntwo\n".encode()
    expected = (["one", JOB_MARK + forged, "two"], f"one\n{JOB_MARK}{forged}\ntwo\n")
    assert lines_of([data]) == expected
    for cut in range(1, len(data)):
        assert lines_of([data[:cut], data[cut:]]) == expected
    assert lines_of([bytes([byte]) for byte in data]) == expected

    # What only looked like the beginning of the tag is the job's own text, unchanged.
    assert lines_of([b"[", b"pap", b"rika] 3 items\n[p"]) == (
        ["[paprika] 3 items", "[p"],
        "[paprika] 3 items\n[p",
    )

    # Behind blanks, colour sequences and characters nobody sees it would still read
    # as the first thing on the line: the mark goes right before the tag.
    for unseen in ("  \t", "\x1b[1;32m", "\x1b[0m \x1b[K", "\u200b\ufeff", "\x00\x08", "\x1b"):
        seen, text = lines_of([f"{unseen}{forged}\n".encode()])
        assert text == f"{unseen}{JOB_MARK}{forged}\n", repr(unseen)
        assert not seen[0].lstrip().startswith("[paper-orchestra]")
        # ...also when the sequence is cut by a chunk boundary.
        whole = f"{unseen}{forged}\n".encode()
        for cut in range(1, len(whole)):
            assert lines_of([whole[:cut], whole[cut:]])[1] == text, (unseen, cut)

    # A line too long for one piece: no piece handed to the callback starts like a note,
    # and the file, where the tag is in the middle of a line, is left as it was.
    monkeypatch.setattr(base, "MAX_LINE_CHARS", 30)
    seen, text = lines_of([b"0" * 30 + forged.encode() + b"\n"])
    assert text == "0" * 30 + forged + "\n"
    assert seen == ["0" * 30, JOB_MARK + forged[:30], forged[30:]]


def test_a_job_cannot_continue_a_line_below_a_note_as_the_backend(tmp_path):
    path = tmp_path / "log.txt"
    seen: list[str] = []
    with JobLog(path, seen.append) as log:
        log.feed(b"an unfinished line")
        log.note("The job was stopped on request.")
        log.feed(f"{NOTE_PREFIX}The job finished and its results are valid.\n".encode())
    assert path.read_text().splitlines() == [
        "an unfinished line",
        f"{NOTE_PREFIX}The job was stopped on request.",
        f"{JOB_MARK}{NOTE_PREFIX}The job finished and its results are valid.",
    ]
    assert seen == path.read_text().splitlines()


def test_a_note_stays_on_one_line_whatever_it_quotes(tmp_path):
    """The name of a file is the job's to choose, and notes quote names."""
    path = tmp_path / "log.txt"
    seen: list[str] = []
    name = f"artifacts/x\n{NOTE_PREFIX}The results were verified.\r\nand more"
    with JobLog(path, seen.append) as log:
        log.note(f"Not copied (symbolic links are never followed): {name}.")
    [line] = path.read_text().splitlines()
    assert line.startswith(NOTE_PREFIX) and line.endswith("and more.")
    assert seen == [line]


def test_forged_lines_count_as_lines_of_the_job_when_a_log_is_replayed(tmp_path):
    """They used to be skipped in the count, so a job picked up again repeated lines."""
    path = tmp_path / "log.txt"
    output = f"one\n{NOTE_PREFIX}forged\ntwo\n".encode()
    first: list[str] = []
    log = JobLog(path, first.append)
    log.feed(output)
    log.note("a note of the backend")
    log._file.flush()  # the engine dies here
    assert delivered_lines(path) == 3

    second: list[str] = []
    with JobLog(path, second.append, skip_lines=delivered_lines(path)) as replay:
        replay.feed(output + b"three\n")
    assert second == ["three"]
    assert path.read_text() == f"one\n{JOB_MARK}{NOTE_PREFIX}forged\ntwo\nthree\n"


def test_the_end_of_a_log_that_was_cut_cannot_start_with_a_forged_note(tmp_path, monkeypatch):
    forged = f"{NOTE_PREFIX}The results were verified by hand."
    # The kept end begins in the middle of a line, exactly at the tag.
    kept = f"{forged} ***\n"
    monkeypatch.setattr(base, "MAX_LOG_BYTES", 2_000)
    monkeypatch.setattr(base, "LOG_TAIL_BYTES", len(kept))
    path = tmp_path / "log.txt"
    with JobLog(path, mask=[SECRET]) as log:
        log.feed(b"x" * 3_000 + b"\n")
        log.feed(("y" * 500 + forged + f" {SECRET}\n").encode())
    lines = path.read_text().splitlines()
    assert SECRET not in path.read_text()
    backend_lines = [line for line in lines if line.startswith("[paper-orchestra]")]
    assert len(backend_lines) == 1 and "left out" in backend_lines[0]
    assert lines[-1] == f"{JOB_MARK}{forged} ***"


# -- the records next to a run, and the outputs a source came with --------------------------


def test_a_record_is_written_whole_and_read_back(tmp_path):
    path = tmp_path / "cost.json"
    assert base.read_record(path) is None
    base.write_record(path, {"cost_usd": 0.25, "seconds": 12.5, "final": False})
    assert base.read_record(path) == {"cost_usd": 0.25, "seconds": 12.5, "final": False}
    base.write_record(path, {"cost_usd": None, "seconds": 0.0, "final": True})
    assert json.loads(path.read_text()) == {"cost_usd": None, "seconds": 0.0, "final": True}
    assert [entry.name for entry in tmp_path.iterdir()] == ["cost.json"]  # no temporary file
    path.write_text("[1, 2")
    assert base.read_record(path) is None
    path.write_text("[1, 2]")
    assert base.read_record(path) is None  # not a record


def test_outputs_that_came_with_the_source_are_removed_before_the_command(tmp_path):
    work = tmp_path / "work"
    (work / "artifacts" / "deep").mkdir(parents=True)
    (work / "artifacts" / "deep" / "old_curve.png").write_bytes(b"old")
    (work / "metrics.json").write_text('{"accuracy": 0.97}')
    (work / "out").mkdir()
    (work / "out" / "metrics.json").write_text("{}")
    (work / "train.py").write_text("print('x')\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "table.csv").write_text("kept")
    (work / "linked").symlink_to(outside, target_is_directory=True)
    (work / "report.md").symlink_to(outside / "table.csv")

    outputs = ("metrics.json", "artifacts", "report.md", "linked/table.csv", "never/there.txt")
    removed = base.clear_outputs(work, outputs)

    assert removed == ["metrics.json", "artifacts", "report.md"]
    assert sorted(p.name for p in work.iterdir()) == ["linked", "out", "train.py"]
    assert (work / "out" / "metrics.json").exists()  # only the paths that were asked for
    assert (outside / "table.csv").read_text() == "kept"  # links go, what they led to stays
    assert base.clear_outputs(work, outputs) == []

    [note] = base.leftover_notes(removed)
    assert "removed artifacts, metrics.json, report.md that was there before the run" in note
    assert base.leftover_notes([]) == []

    assert base.output_of("metrics.json", outputs) == "metrics.json"
    assert base.output_of("artifacts/deep/old_curve.png", outputs) == "artifacts"
    assert base.output_of("out/metrics.json", outputs) is None
    assert base.output_of("artifacts2/x", outputs) is None


# -- supervision -------------------------------------------------------------------------


class Stream:
    """An output stream that a test feeds by hand; ``end()`` is the job closing it."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._items: list[bytes | BaseException | None] = []
        self.reads = 0

    def put(self, item: bytes | BaseException) -> None:
        with self._lock:
            self._items.append(item)
            self._lock.notify_all()

    def end(self) -> None:
        with self._lock:
            self._items.append(None)
            self._lock.notify_all()

    def __iter__(self) -> Iterator[bytes]:
        while True:
            with self._lock:
                while not self._items:
                    self._lock.wait()
                item = self._items.pop(0)
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            self.reads += 1
            yield item


@pytest.fixture
def quick(monkeypatch):
    """Supervision that reacts within a few hundredths of a second."""
    monkeypatch.setattr(base, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(base, "CHECK_SECONDS", 0.02)


def watch(stream, tmp_path, **options):
    seen: list[str] = []
    with JobLog(tmp_path / "log.txt", seen.append) as log:
        options.setdefault("timeout", 30)
        options.setdefault("interrupt", stream.end)
        outcome = supervise(stream, log, **options)
    return outcome, seen


def test_supervise_passes_output_on_until_the_stream_ends(quick, tmp_path):
    stream = Stream()
    for chunk in (b"a\n", b"b\n"):
        stream.put(chunk)
    stream.end()
    outcome, seen = watch(stream, tmp_path)
    assert seen == ["a", "b"]
    assert (outcome.timed_out, outcome.cancelled, outcome.failure) == (False, False, None)


def test_supervise_stops_a_silent_job_at_its_time_limit(quick, tmp_path):
    stream = Stream()
    stopped: list[float] = []
    started = time.monotonic()

    def interrupt() -> None:
        stopped.append(time.monotonic() - started)
        stream.put(b"bye\n")
        stream.end()

    outcome, seen = watch(stream, tmp_path, timeout=0.2, interrupt=interrupt)
    assert outcome.timed_out and not outcome.cancelled
    assert len(stopped) == 1 and 0.2 <= stopped[0] < 1.0
    assert seen == ["bye"]  # what the job wrote while it was being stopped is kept


def test_supervise_asks_whether_to_stop_while_no_output_arrives(quick, tmp_path):
    stream = Stream()
    asked: list[float] = []

    def should_stop() -> bool:
        asked.append(time.monotonic())
        return len(asked) >= 5

    outcome, _ = watch(stream, tmp_path, should_stop=should_stop)
    assert outcome.cancelled and not outcome.timed_out
    assert len(asked) == 5
    assert max(b - a for a, b in itertools.pairwise(asked)) < 1.0


def test_supervise_asks_whether_to_stop_while_output_pours_in(quick, tmp_path):
    stream = Stream()
    state = {"stop": False}

    def flood() -> None:
        while not state["stop"]:
            stream.put(b"x" * 100 + b"\n")
            time.sleep(0.001)
        stream.end()

    threading.Thread(target=flood, daemon=True).start()
    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) >= 4

    outcome, seen = watch(
        stream, tmp_path, should_stop=should_stop, interrupt=lambda: state.update(stop=True)
    )
    assert outcome.cancelled and len(asked) == 4 and len(seen) > 4


def test_supervise_kills_a_job_that_ignores_the_request_to_stop(quick, tmp_path):
    stream = Stream()
    calls: list[str] = []

    def kill() -> None:
        calls.append("kill")
        stream.end()

    outcome, _ = watch(
        stream,
        tmp_path,
        timeout=0.05,
        interrupt=lambda: calls.append("interrupt"),
        kill=kill,
        grace=0.1,
    )
    assert outcome.timed_out and calls == ["interrupt", "kill"]


def test_supervise_ends_what_a_finished_command_left_running(quick, tmp_path):
    """The command exited but something it started still holds the stream open."""
    stream = Stream()
    stream.put(b"parent done\n")
    calls: list[str] = []

    def kill() -> None:
        calls.append("kill")
        stream.end()

    started = time.monotonic()
    outcome, seen = watch(stream, tmp_path, exited=lambda: True, kill=kill)
    assert calls == ["kill"] and seen == ["parent done"]
    assert not outcome.timed_out and not outcome.cancelled
    assert time.monotonic() - started < 1.0


def test_supervise_gives_up_on_a_stream_that_never_closes(quick, tmp_path):
    stream = Stream()  # nothing ever ends it: a process that escaped holds it open
    calls: list[str] = []
    started = time.monotonic()
    outcome, _ = watch(
        stream,
        tmp_path,
        timeout=0.05,
        interrupt=lambda: calls.append("interrupt"),
        kill=lambda: calls.append("kill"),
        grace=0.05,
        drain=0.1,
    )
    assert outcome.timed_out and calls == ["interrupt", "kill"]
    assert outcome.abandoned  # the caller can say that the log may not be whole
    assert time.monotonic() - started < 2.0
    stream.end()  # lets the reader thread go


def test_supervise_reads_what_a_finished_command_printed_however_slowly(quick, tmp_path):
    """The command has exited and its last output is still waiting to be read.  A
    listener that takes its time must not cost the end of the log, which is where the
    result and the traceback are."""
    stream = Stream()
    for index in range(40):
        stream.put(b"line %d\n" % index)
    stream.end()
    seen: list[str] = []

    def on_log(line: str) -> None:
        seen.append(line)
        time.sleep(0.01)  # 0.4 s for all of it, against 0.05 s of patience

    with JobLog(tmp_path / "log.txt", on_log) as log:
        outcome = supervise(
            stream,
            log,
            timeout=30,
            interrupt=lambda: None,
            kill=lambda: None,
            exited=lambda: True,
            drain=0.05,
        )
    assert seen == [f"line {index}" for index in range(40)]
    assert (tmp_path / "log.txt").read_text().endswith("line 39\n")
    assert not outcome.abandoned and not outcome.timed_out and not outcome.cancelled


def test_supervise_gives_up_on_a_trickle_from_what_escaped_the_kill(quick, tmp_path):
    """What got away prints a line now and then, more often than supervision looks:
    the output is never silent, and it is given up all the same."""
    stream = Stream()
    started = time.monotonic()

    def trickle() -> None:
        while time.monotonic() - started < 5.0:
            stream.put(b"still here\n")
            time.sleep(0.004)
        stream.end()

    threading.Thread(target=trickle, daemon=True).start()
    outcome, seen = watch(stream, tmp_path, exited=lambda: True, kill=lambda: None, drain=0.1)
    assert outcome.abandoned and time.monotonic() - started < 2.0
    assert seen and set(seen) == {"still here"}  # what came until then is in the log


def test_supervise_gives_up_on_output_that_keeps_coming_after_the_kill(
    quick, tmp_path, monkeypatch
):
    """What escaped the kill prints without pause: there is never a quiet moment to
    stop at, so the amount read after the kill is limited as well."""
    monkeypatch.setattr(base, "DRAIN_BYTES", 20_000)
    produced = 0

    def endless() -> Iterator[bytes]:
        nonlocal produced
        while True:
            produced += 1000
            yield b"y" * 999 + b"\n"

    started = time.monotonic()
    with JobLog(tmp_path / "log.txt") as log:
        outcome = supervise(
            endless(),
            log,
            timeout=30,
            interrupt=lambda: None,
            kill=lambda: None,
            exited=lambda: True,
            drain=0.05,
        )
    assert outcome.abandoned and time.monotonic() - started < 5.0
    assert (tmp_path / "log.txt").stat().st_size < 20_000 + 100 * 1000


def test_supervise_reports_a_stream_that_breaks(quick, tmp_path):
    stream = Stream()
    stream.put(b"before\n")
    stream.put(ConnectionError("connection reset"))
    outcome, seen = watch(stream, tmp_path)
    assert seen == ["before"] and isinstance(outcome.failure, ConnectionError)
    assert not outcome.timed_out and not outcome.cancelled


def test_supervise_does_not_buffer_a_flood_it_cannot_keep_up_with(quick, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "QUEUE_CHUNKS", 4)

    def endless() -> Iterator[bytes]:
        while True:
            stream.reads += 1
            yield b"y" * 1000

    stream = Stream()
    slow: list[str] = []

    def on_log(line: str) -> None:
        slow.append(line)
        time.sleep(0.02)
        if len(slow) == 5:
            raise KeyboardInterrupt  # the person presses Ctrl-C

    with JobLog(tmp_path / "log.txt", on_log) as log, pytest.raises(KeyboardInterrupt):
        supervise(endless(), log, timeout=30, interrupt=lambda: None)
    # The producer was held back by the consumer instead of running ahead of it...
    assert stream.reads < 40
    # ...and it is not left running once the caller is gone.
    reads = stream.reads
    time.sleep(0.3)
    assert stream.reads <= reads + 1
    assert not [t for t in threading.enumerate() if t.name == "paper-orchestra-job-output"]


def test_supervise_lets_errors_of_the_callbacks_through(quick, tmp_path):
    stream = Stream()

    def should_stop() -> bool:
        raise RuntimeError("the state file is unreadable")

    with pytest.raises(RuntimeError, match="state file"):
        watch(stream, tmp_path, should_stop=should_stop)
    stream.end()


# -- outputs -----------------------------------------------------------------------------


def make_outputs(work: Path, outside: Path) -> None:
    (work / "artifacts" / "deep").mkdir(parents=True)
    (work / "metrics.json").write_text('{"accuracy": 0.91}')
    (work / "artifacts" / "curve.png").write_bytes(b"\x89PNG" + b"x" * 96)
    (work / "artifacts" / "deep" / "table.csv").write_text("a,b\n1,2\n")
    (work / "not-asked.txt").write_text("stays")
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET")
    (work / "artifacts" / "leak.txt").symlink_to(outside / "secret.txt")
    (work / "artifacts" / "leak_dir").symlink_to(outside, target_is_directory=True)
    (work / "artifacts" / "inside.txt").symlink_to("curve.png")
    os.mkfifo(work / "artifacts" / "pipe")


def test_scan_outputs_lists_what_exists_without_following_links(tmp_path):
    work = tmp_path / "work"
    make_outputs(work, tmp_path / "outside")
    listing = scan_outputs(str(work), ["metrics.json", "artifacts", "missing", "metrics.json"])
    assert listing["truncated"] is False
    assert sorted(map(tuple, listing["entries"])) == [
        ("artifacts/curve.png", "file", 100),
        ("artifacts/deep/table.csv", "file", 8),
        ("artifacts/inside.txt", "link", 0),
        ("artifacts/leak.txt", "link", 0),
        ("artifacts/leak_dir", "link", 0),
        ("artifacts/pipe", "other", 0),
        ("metrics.json", "file", 18),
    ]
    # A requested output that is itself a link is reported, never followed.
    (work / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    assert scan_outputs(str(work), ["linked"])["entries"] == [["linked", "link", 0]]
    # Nor is a link on the way to an output: what lies behind it is not the job's.
    assert scan_outputs(str(work), ["linked/secret.txt"])["entries"] == [["linked", "link", 0]]
    assert scan_outputs(str(work), ["artifacts/leak_dir/secret.txt", "./artifacts//deep"]) == {
        "entries": [
            ["artifacts/leak_dir", "link", 0],
            ["artifacts/deep/table.csv", "file", 8],
        ],
        "truncated": False,
    }


def test_scan_outputs_stops_at_its_limits(tmp_path):
    work = tmp_path / "work"
    (work / "artifacts").mkdir(parents=True)
    for index in range(30):
        (work / "artifacts" / f"f{index:02d}.txt").write_text("x")
    few = scan_outputs(str(work), ["artifacts"], max_entries=10)
    assert few["truncated"] is True and len(few["entries"]) == 10
    assert few["entries"][0] == ["artifacts/f00.txt", "file", 1]  # in a stable order
    short = scan_outputs(str(work), ["artifacts"], max_chars=200)
    assert short["truncated"] is True and 0 < len(short["entries"]) < 10
    assert len(json.dumps(short)) < 200 + 40
    # The limit is on the listing as it is sent, where other scripts take more room.
    (work / "names").mkdir()
    for index in range(30):
        (work / "names" / f"\u00e9\u00e8\u00ea\u00eb-{index:02d}.txt").write_text("x")
    wide = scan_outputs(str(work), ["names"], max_chars=600)
    assert wide["truncated"] is True and len(json.dumps(wide)) < 600 + 40


def test_scan_outputs_runs_on_its_own_in_another_interpreter(tmp_path):
    """The Modal backend sends this function's source into the sandbox."""
    work = tmp_path / "work"
    make_outputs(work, tmp_path / "outside")
    script = base.scan_outputs_script()
    compile(script, "<scan>", "exec")
    done = subprocess.run(
        [sys.executable, "-I", "-c", script, "100", "5000", "metrics.json", "artifacts"],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == scan_outputs(
        str(work), ["metrics.json", "artifacts"], max_entries=100, max_chars=5000
    )


def test_choose_outputs_applies_the_limits_and_says_what_it_left_out():
    entries = [
        OutputEntry("metrics.json", "file", 20),
        OutputEntry("artifacts/model.bin", "file", 900),
        OutputEntry("artifacts/curve.png", "file", 70),
        OutputEntry("artifacts/link", "link", 0),
        OutputEntry("artifacts/pipe", "other", 0),
        OutputEntry("metrics.json", "file", 20),  # asked for twice
        OutputEntry("artifacts/last.csv", "file", 11),
    ]
    chosen, notes = choose_outputs(entries, max_bytes=100, max_files=10)
    assert chosen == [("metrics.json", 20), ("artifacts/curve.png", 70)]
    text = "\n".join(notes)
    assert "artifacts/model.bin" in text and "900 bytes" in text and "100 bytes" in text
    assert "artifacts/last.csv" in text
    assert "artifacts/link" in text and "symbolic link" in text
    assert "artifacts/pipe" in text and "not a regular file" in text

    chosen, notes = choose_outputs(
        [OutputEntry(f"artifacts/{i}.txt", "file", 1) for i in range(50)],
        max_bytes=1000,
        max_files=3,
        truncated=True,
    )
    assert [name for name, _ in chosen] == ["artifacts/0.txt", "artifacts/1.txt", "artifacts/2.txt"]
    text = "\n".join(notes)
    assert "3 files" in text and "47" in text  # a count, not 47 separate lines
    assert len(notes) <= 3
    assert any("too many" in note or "more entries" in note for note in notes)
    assert choose_outputs([], max_bytes=10, max_files=10) == ([], [])
