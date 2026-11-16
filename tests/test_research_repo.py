"""Repository mode: a copy of the person's repository for one run (``research/repo.py``),
and experiments that run inside it."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from paper_orchestra.compute import JobResult, JobSpec
from paper_orchestra.compute.base import JobStopped
from paper_orchestra.compute.local import LocalBackend
from paper_orchestra.research import experiments, repo
from paper_orchestra.research.model import ProjectError
from paper_orchestra.research.project import Project

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

TRAIN = """\
import json

from lib.util import scale

LEARNING_RATE = 0.1


def main():
    with open("metrics.json", "w") as handle:
        json.dump({"score": scale(LEARNING_RATE)}, handle)
    print("trained with", LEARNING_RATE)


if __name__ == "__main__":
    main()
"""

UTIL = """\
def scale(value):
    return value * 10
"""

PATCH = """\
--- a/train.py
+++ b/train.py
@@ -2,7 +2,7 @@

 from lib.util import scale

-LEARNING_RATE = 0.1
+LEARNING_RATE = 0.25


 def main():
"""


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=Tester", "-c", "user.email=tester@example.com", *args],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


def plain_tree(root: Path) -> Path:
    """The files of a small repository, without any version control."""
    root.mkdir(parents=True)
    (root / "train.py").write_text(TRAIN)
    (root / "lib").mkdir()
    (root / "lib" / "util.py").write_text(UTIL)
    (root / "README.md").write_text("# A small repository\n")
    return root


@pytest.fixture
def tree(tmp_path) -> Path:
    return plain_tree(tmp_path / "tree")


@pytest.fixture
def repository(tmp_path) -> Path:
    """A git repository with ignored, secret, large and environment files in its tree."""
    root = plain_tree(tmp_path / "repository")
    (root / ".gitignore").write_text("data/\n*.log\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    (root / "data").mkdir()
    (root / "data" / "ignored.bin").write_bytes(b"\0" * 64)
    (root / "debug.log").write_text("ignored\n")
    (root / ".env").write_text("TOKEN=secret\n")
    (root / "venv").mkdir()
    (root / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (root / "venv" / "site.py").write_text("x = 1\n")
    (root / "notes.txt").write_text("not committed, not ignored\n")
    (root / "weights.bin").write_bytes(b"w" * 5000)
    return root


def snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` with a digest of its content, ``.git`` included."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def copied(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


# -- copying a local repository ----------------------------------------------------------


@needs_git
def test_a_git_repository_is_copied_as_git_sees_it(repository, tmp_path):
    status = git(repository, "status", "--porcelain")
    before = snapshot(repository)
    dest = tmp_path / "copy"

    made = repo.copy_local(repository, dest, max_file_bytes=1000)

    assert copied(dest) == {"train.py", "lib/util.py", "README.md", ".gitignore", "notes.txt"}
    assert made.method == "git" and made.files == 5
    assert made.commit == git(repository, "rev-parse", "HEAD").strip()
    assert made.dirty is True  # notes.txt and the others are not committed
    reasons = {entry["path"]: entry["reason"] for entry in made.skipped}
    assert "virtual environment" in reasons["venv/pyvenv.cfg"]
    assert "larger than" in reasons["weights.bin"]
    assert ".env" in reasons and "data/ignored.bin" not in reasons  # ignored: not even listed
    assert not (dest / ".git").exists()
    # The person's working tree, index and history are exactly as they were.
    assert snapshot(repository) == before
    assert git(repository, "status", "--porcelain") == status


def test_without_git_the_directory_is_walked(repository, tmp_path):
    dest = tmp_path / "copy"
    made = repo.copy_local(repository, dest, use_git=False, max_file_bytes=1000)
    names = copied(dest)
    # .gitignore means nothing without git, so what it ignores is copied.
    assert {"train.py", "lib/util.py", "notes.txt", "debug.log", "data/ignored.bin"} <= names
    assert not any(name.startswith((".git/", "venv/")) for name in names)
    assert ".env" not in names and "weights.bin" not in names
    assert made.method == "walk" and made.commit is None and made.dirty is None
    listed = {entry["path"] for entry in made.skipped}
    assert {".git", ".env", "venv", "weights.bin"} <= listed


@needs_git
def test_a_copy_at_a_ref_is_the_committed_tree(repository, tmp_path):
    first = git(repository, "rev-parse", "HEAD").strip()
    (repository / "train.py").write_text("print('changed, not committed')\n")
    made = repo.copy_local(repository, tmp_path / "copy", ref="main")
    assert (tmp_path / "copy" / "train.py").read_text() == TRAIN
    assert copied(tmp_path / "copy") == {"train.py", "lib/util.py", "README.md", ".gitignore"}
    assert made.commit == first and made.ref == "main" and made.method == "archive"
    assert (repository / "train.py").read_text() == "print('changed, not committed')\n"

    with pytest.raises(repo.RepoError, match="no-such-branch"):
        repo.copy_local(repository, tmp_path / "other", ref="no-such-branch")
    with pytest.raises(repo.RepoError, match="git"):
        repo.copy_local(repository, tmp_path / "third", ref="main", use_git=False)


def test_a_copy_that_would_be_too_large_is_refused_with_the_numbers(tree, tmp_path):
    (tree / "big.csv").write_text("x" * 4000)
    with pytest.raises(repo.RepoError, match="larger than") as refused:
        repo.copy_local(tree, tmp_path / "copy", use_git=False, max_total_bytes=2000)
    assert ".gitignore" in str(refused.value)


def test_links_that_lead_out_of_the_repository_are_left_out(tree, tmp_path):
    (tmp_path / "secret.txt").write_text("outside\n")
    (tree / "leak").symlink_to(tmp_path / "secret.txt")
    (tree / "inside").symlink_to("README.md")
    made = repo.copy_local(tree, tmp_path / "copy", use_git=False)
    assert not (tmp_path / "copy" / "leak").exists()
    assert (tmp_path / "copy" / "inside").is_symlink()
    assert "leak" in {entry["path"] for entry in made.skipped}


@pytest.mark.parametrize("use_git", [pytest.param(True, marks=needs_git), False])
def test_a_project_inside_the_repository_is_not_copied_into_its_own_runs(tree, use_git):
    if use_git:
        git(tree, "init", "-q", "-b", "main")
    project = tree / "research" / "warmup"
    (project / "wiki").mkdir(parents=True)
    (project / "wiki" / "idea.md").write_text("The idea.\n")
    dest = project / "experiments" / "E1" / "runs" / "1" / "repo"

    made = repo.copy_local(tree, dest, use_git=use_git, exclude=[project])

    assert copied(dest) == {"train.py", "lib/util.py", "README.md"}
    reasons = {entry["path"]: entry["reason"] for entry in made.skipped}
    assert "project" in reasons["research/warmup"]
    # A copy made anywhere else inside the repository does not copy itself either.
    repo.copy_local(tree, tree / "out" / "copy", use_git=use_git)
    assert not (tree / "out" / "copy" / "out").exists()
    with pytest.raises(repo.RepoError, match="own directory"):
        repo.copy_local(tree, tree / "third", use_git=use_git, exclude=[tree])


def test_copying_stops_when_asked_to(tree, tmp_path):
    with pytest.raises(JobStopped):
        repo.copy_local(tree, tmp_path / "copy", use_git=False, should_stop=lambda: True)


def test_a_missing_repository_is_an_error_a_person_can_act_on(tmp_path):
    with pytest.raises(repo.RepoError, match="does not exist"):
        repo.copy_local(tmp_path / "nowhere", tmp_path / "copy")


def outcome_of(action, seconds: float = 30.0):
    """What ``action`` returned or raised, or the word ``hung`` when it did not end in time."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = action()
        except BaseException as exc:
            box["value"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    return "hung" if worker.is_alive() else box["value"]


@needs_git
def test_a_copy_at_a_ref_that_is_refused_or_stopped_ends_at_once(repository, tmp_path, monkeypatch):
    # More than a pipe holds: git is still writing when the copy gives up.
    for number in range(12):
        (repository / f"table{number}.csv").write_bytes(os.urandom(100_000))
    git(repository, "add", ".")
    git(repository, "commit", "-q", "-m", "tables")

    refused = outcome_of(
        lambda: repo.copy_local(repository, tmp_path / "a", ref="main", max_total_bytes=250_000)
    )
    assert isinstance(refused, repo.RepoError) and "larger than" in str(refused)

    monkeypatch.setattr(repo, "CHECK_SECONDS", 0.0)
    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) > 3

    stopped = outcome_of(
        lambda: repo.copy_local(repository, tmp_path / "b", ref="main", should_stop=should_stop)
    )
    assert isinstance(stopped, JobStopped)


@pytest.mark.parametrize("use_git", [pytest.param(True, marks=needs_git), False])
def test_a_file_that_cannot_be_copied_is_an_error_a_person_can_act_on(
    tree, tmp_path, monkeypatch, use_git
):
    if use_git:
        git(tree, "init", "-q", "-b", "main")
    place = repo._place

    def unreadable(source: Path, target: Path) -> None:
        if source.name == "util.py":
            raise PermissionError(13, "Permission denied", str(source))
        place(source, target)

    monkeypatch.setattr(repo, "_place", unreadable)
    with pytest.raises(repo.RepoError, match="Permission denied") as refused:
        repo.copy_local(tree, tmp_path / "copy", use_git=use_git)
    assert "util.py" in str(refused.value) and ".gitignore" in str(refused.value)


# -- patches -----------------------------------------------------------------------------

appliers = pytest.mark.parametrize("use_git", [pytest.param(True, marks=needs_git), False])

ADD_AND_DELETE = """\
diff --git a/lib/extra.py b/lib/extra.py
new file mode 100644
--- /dev/null
+++ b/lib/extra.py
@@ -0,0 +1,2 @@
+def extra():
+    return 1
diff --git a/README.md b/README.md
deleted file mode 100644
--- a/README.md
+++ /dev/null
@@ -1 +0,0 @@
-# A small repository
"""


@appliers
def test_a_patch_changes_adds_and_deletes_files(tree, use_git):
    touched = repo.apply_patch(tree, PATCH + ADD_AND_DELETE, use_git=use_git)
    assert "LEARNING_RATE = 0.25" in (tree / "train.py").read_text()
    assert (tree / "lib" / "extra.py").read_text() == "def extra():\n    return 1\n"
    assert not (tree / "README.md").exists()
    assert set(touched) == {"train.py", "lib/extra.py", "README.md"}


@appliers
def test_a_patch_that_does_not_apply_is_rejected_with_its_hunk_and_changes_nothing(tree, use_git):
    wrong = PATCH.replace("-LEARNING_RATE = 0.1", "-LEARNING_RATE = 0.3")
    before = snapshot(tree)
    with pytest.raises(repo.PatchError) as rejected:
        repo.apply_patch(tree, ADD_AND_DELETE + wrong, use_git=use_git)
    assert "train.py" in str(rejected.value)
    assert "-LEARNING_RATE = 0.3" in rejected.value.hunk
    assert "@@ -2,7 +2,7 @@" in str(rejected.value)
    assert snapshot(tree) == before  # not even the hunks that would have applied


def test_the_python_applier_finds_a_hunk_that_moved_and_forgives_a_wrong_count(tree):
    moved = PATCH.replace("@@ -2,7 +2,7 @@", "@@ -9,3 +9,3 @@")
    repo.apply_patch(tree, moved, use_git=False)
    assert "LEARNING_RATE = 0.25" in (tree / "train.py").read_text()


@needs_git
def test_a_patch_git_refuses_for_its_form_still_applies(tree):
    # Models miscount hunk headers; the text of the hunk says what is meant.
    miscounted = PATCH.replace("@@ -2,7 +2,7 @@", "@@ -2,9 +2,9 @@")
    repo.apply_patch(tree, miscounted, use_git=True)
    assert "LEARNING_RATE = 0.25" in (tree / "train.py").read_text()


@appliers
@pytest.mark.parametrize(
    ("hunk", "expected"),
    [
        # Lines added after line 2, with nothing around them to find the place by.
        ("@@ -2,0 +3 @@\n+inserted\n", "a\nb\ninserted\nc\nb\n"),
        # A changed line whose old text also ends the file.
        ("@@ -2 +2 @@\n-b\n+B\n", "a\nB\nc\nb\n"),
    ],
)
def test_a_hunk_without_context_changes_the_lines_it_names(tree, use_git, hunk, expected):
    (tree / "letters.txt").write_text("a\nb\nc\nb\n")
    repo.apply_patch(tree, "--- a/letters.txt\n+++ b/letters.txt\n" + hunk, use_git=use_git)
    assert (tree / "letters.txt").read_text() == expected


@appliers
def test_a_last_line_without_a_newline_is_not_taken_for_the_same_line_above_it(tree, use_git):
    (tree / "twice.txt").write_bytes(b"a\nx = 1\nx = 1")
    patch = (
        "--- a/twice.txt\n+++ b/twice.txt\n@@ -3 +2,0 @@\n-x = 1\n\\ No newline at end of file\n"
    )
    repo.apply_patch(tree, patch, use_git=use_git)
    assert (tree / "twice.txt").read_bytes() == b"a\nx = 1\n"


def test_a_patch_without_prefixes_is_applied_to_the_file_it_names(tree):
    (tree / "util.py").write_text("def scale(value):\n    return value\n")
    bare = (
        "--- lib/util.py\n+++ lib/util.py\n@@ -1,2 +1,2 @@\n def scale(value):\n"
        "-    return value * 10\n+    return value * 100\n"
    )
    for use_git in [True, False] if shutil.which("git") else [False]:
        work = tree.parent / f"work-{use_git}"
        shutil.copytree(tree, work)
        repo.apply_patch(work, bare, use_git=use_git)
        assert "value * 100" in (work / "lib" / "util.py").read_text()
        assert (work / "util.py").read_text() == "def scale(value):\n    return value\n"


@pytest.mark.parametrize(
    "path", ["../outside.py", "/etc/passwd", ".git/config", "lib/../../outside.py"]
)
def test_a_patch_may_not_reach_outside_the_copy(tree, tmp_path, path):
    patch = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x = 1\n"
    if path.startswith("/"):
        patch = f"--- /dev/null\n+++ {path}\n@@ -0,0 +1 @@\n+x = 1\n"
    for use_git in [True, False] if shutil.which("git") else [False]:
        with pytest.raises(repo.PatchError, match=r"outside|not allowed"):
            repo.apply_patch(tree, patch, use_git=use_git)
    assert not (tmp_path / "outside.py").exists()
    assert not (tree / ".git").exists()


def test_a_patch_does_not_write_through_a_link(tree, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree / "linked").symlink_to(outside)
    patch = "--- /dev/null\n+++ b/linked/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    with pytest.raises(repo.PatchError):
        repo.apply_patch(tree, patch, use_git=False)
    assert not (outside / "new.py").exists()


def test_text_that_is_not_a_patch_is_refused(tree):
    with pytest.raises(repo.PatchError, match="no changes"):
        repo.apply_patch(tree, "please change the learning rate to 0.25\n")
    with pytest.raises(repo.PatchError, match="rename"):
        repo.parse_patch(
            "diff --git a/a.py b/b.py\nsimilarity index 100%\nrename from a.py\nrename to b.py\n"
        )


def test_a_patch_keeps_the_line_endings_and_final_newline_of_the_file(tree):
    (tree / "dos.txt").write_bytes(b"one\r\ntwo\r\nthree")
    patch = (
        "--- a/dos.txt\n+++ b/dos.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+2\n three\n"
        "\\ No newline at end of file\n"
    )
    repo.apply_patch(tree, patch, use_git=False)
    assert (tree / "dos.txt").read_bytes() == b"one\r\n2\r\nthree"


# -- files of the experiment added to the copy -------------------------------------------


def test_files_are_added_but_never_replace_a_file_of_the_repository(tree):
    repo.add_files(tree, {"run_e1.py": b"print('hi')\n", "tools/helper.py": b"x = 1\n"})
    assert (tree / "tools" / "helper.py").read_bytes() == b"x = 1\n"
    with pytest.raises(repo.RepoError, match="already exists") as refused:
        repo.add_files(tree, {"train.py": b"print('replaced')\n"})
    assert "patch.diff" in str(refused.value)
    assert (tree / "train.py").read_text() == TRAIN
    with pytest.raises(repo.RepoError):
        repo.add_files(tree, {"../escape.py": b""})


# -- URLs --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("location", "url"),
    [
        ("https://github.com/owner/project.git", True),
        ("git@github.com:owner/project.git", True),
        ("ssh://git@example.org/owner/project", True),
        ("file:///home/someone/project", True),
        ("/home/someone/project", False),
        ("../project", False),
        ("C:\\Users\\someone\\project", False),
    ],
)
def test_a_location_is_a_url_or_a_path(location, url):
    assert repo.is_url(location) is url


@pytest.mark.parametrize(
    "url",
    [
        "file:///home/someone/project",
        "ext::sh -c 'touch /tmp/owned'",
        "https://-oProxyCommand=evil/x",
        "ssh://-oProxyCommand=evil/x",
        "https://example.org/a b",
        "ftp://example.org/project",
    ],
)
def test_urls_git_would_run_or_read_locally_are_refused(url):
    assert repo.url_problem(url)


def test_ordinary_urls_and_refs_are_accepted():
    assert repo.url_problem("https://github.com/owner/project.git") is None
    assert repo.url_problem("git@github.com:owner/project.git") is None
    assert repo.ref_problem("main") is None
    assert repo.ref_problem("v1.2.0") is None
    assert repo.ref_problem("3f2a9c1") is None
    assert repo.ref_problem("--upload-pack=evil")
    assert repo.ref_problem("a b")


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://github.com/owner/project.git", "https://github.com/owner/project.git"),
        ("https://ghp_s3cret@github.com/o/p.git", "https://***@github.com/o/p.git"),
        ("https://bot:ghp_s3cret@github.com/o/p.git", "https://bot:***@github.com/o/p.git"),
        ("ssh://git@example.org/owner/project", "ssh://git@example.org/owner/project"),
        ("ssh://me:s3cret@example.org/o/p", "ssh://me:***@example.org/o/p"),
        ("git@github.com:owner/project.git", "git@github.com:owner/project.git"),
        ("/home/someone/project", "/home/someone/project"),
    ],
)
def test_a_url_is_shown_without_its_password_or_token(url, shown):
    assert repo.display_url(url) == shown


# -- cloning inside the job --------------------------------------------------------------


@needs_git
def test_the_clone_happens_inside_the_job_and_the_command_runs_in_it(repository, tmp_path):
    git(repository, "add", "notes.txt")
    git(repository, "commit", "-q", "-m", "second")
    head = git(repository, "rev-parse", "HEAD").strip()
    seed = tmp_path / "seed"
    command = repo.clone_source(
        seed,
        url=str(repository),  # a path stands in for the URL: no test may use the network
        ref=None,
        patch=PATCH,
        files={"check.py": b"import os\nprint('notes', os.path.exists('notes.txt'))\n"},
        command=["python", "train.py"],
    )
    assert command[0] == "python" and not (seed / "train.py").exists()

    backend = LocalBackend(scratch_dir=tmp_path / "scratch")
    result = backend.run_job(
        JobSpec(command=command, timeout=60, network=True),
        source_dir=seed,
        run_dir=tmp_path / "run",
    )
    log = (tmp_path / "run" / "log.txt").read_text()
    assert result.ok, log
    assert json.loads((tmp_path / "run" / "metrics.json").read_text()) == {"score": 2.5}
    assert f"cloned {repository} at {head}" in log
    assert repo.clone_failure(result.returncode) is None


@needs_git
def test_a_clone_that_fails_says_which_step_failed(repository, tmp_path):
    backend = LocalBackend(scratch_dir=tmp_path / "scratch")

    def attempt(name: str, **changes) -> tuple[JobResult, str]:
        options = {"url": str(repository), "ref": None, "patch": None, "files": {}}
        options.update(changes)
        command = repo.clone_source(tmp_path / name, command=["python", "train.py"], **options)
        result = backend.run_job(
            JobSpec(command=command, timeout=60, network=True),
            source_dir=tmp_path / name,
            run_dir=tmp_path / f"{name}-run",
        )
        return result, (tmp_path / f"{name}-run" / "log.txt").read_text()

    result, log = attempt("gone", url=str(tmp_path / "no-such-repository"))
    kind, message = repo.clone_failure(result.returncode)
    assert kind == "setup" and "cloned" in message and "could not be cloned" in log

    wrong = PATCH.replace("-LEARNING_RATE = 0.1", "-LEARNING_RATE = 0.3")
    result, log = attempt("patch", patch=wrong)
    kind, message = repo.clone_failure(result.returncode)
    assert kind == "patch" and "patch.diff" in message and "patch.diff does not apply" in log

    result, log = attempt("exists", files={"train.py": b"print('replaced')\n"})
    kind, _ = repo.clone_failure(result.returncode)
    assert kind == "setup" and "train.py already exists" in log
    assert repo.clone_failure(0) is None and repo.clone_failure(1) is None


@needs_git
def test_the_clone_job_applies_a_patch_the_way_it_is_applied_here(repository, tmp_path):
    # A file of the same name one level up: a strip level guessed wrong would change it.
    (repository / "util.py").write_text(UTIL)
    (repository / "letters.txt").write_text("a\nb\nc\nb\n")
    git(repository, "add", "util.py", "letters.txt")
    git(repository, "commit", "-q", "-m", "second")
    bare = (
        "--- lib/util.py\n+++ lib/util.py\n@@ -1,2 +1,2 @@\n def scale(value):\n"
        "-    return value * 10\n+    return value * 100\n"
        "--- letters.txt\n+++ letters.txt\n@@ -2 +2 @@\n-b\n+B\n"
    )
    check = (
        "import json\n"
        "from lib.util import scale\n"
        "import util\n"
        "found = {'lib': scale(1), 'root': util.scale(1), 'letters': open('letters.txt').read()}\n"
        "json.dump(found, open('metrics.json', 'w'))\n"
    )
    command = repo.clone_source(
        tmp_path / "seed",
        url=str(repository),
        patch=bare,
        files={"check.py": check.encode()},
        command=["python", "check.py"],
    )
    result = LocalBackend(scratch_dir=tmp_path / "scratch").run_job(
        JobSpec(command=command, timeout=60, network=True),
        source_dir=tmp_path / "seed",
        run_dir=tmp_path / "run",
    )
    assert result.ok, (tmp_path / "run" / "log.txt").read_text()
    found = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert found == {"lib": 100, "root": 10, "letters": "a\nB\nc\nb\n"}

    mixed = PATCH + "--- letters.txt\n+++ letters.txt\n@@ -2 +2 @@\n-b\n+B\n"
    with pytest.raises(repo.PatchError, match="a/ and b/"):
        repo.clone_source(tmp_path / "other", url=str(repository), patch=mixed, command=["true"])


#: Stands in for git in a clone job: no test may reach a network.  It fails the fetch
#: when asked to, quoting the URL the way git does.
FAKE_GIT = """\
#!/bin/sh
case "$1" in
  remote) echo "$4" > .fake-git-url ;;
  fetch) if [ -n "$FAKE_GIT_FAILS" ]; then
           echo "fatal: unable to access '$(cat .fake-git-url)/': refused"; exit 128
         fi ;;
  rev-parse) echo 0123abc ;;
esac
exit 0
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
def test_the_clone_job_never_prints_the_password_of_its_url(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "git").write_text(FAKE_GIT)
    (tmp_path / "bin" / "git").chmod(0o755)
    url = "https://bot:s3cret-token@example.org/owner/project.git"
    seed = tmp_path / "seed"
    said = [sys.executable, "-c", "print('the command ran')"]
    command = repo.clone_source(seed, url=url, command=said)

    def job(**env: str) -> subprocess.CompletedProcess:
        path = f"{tmp_path / 'bin'}:/usr/bin:/bin"
        return subprocess.run(
            [sys.executable, command[1]],
            cwd=seed,
            env={"PATH": path, **env},
            capture_output=True,
            text=True,
        )

    failed = job(FAKE_GIT_FAILS="1")
    assert repo.clone_failure(failed.returncode) == repo.clone_failure(repo.EXIT_CLONE_FAILED)
    assert "could not be cloned" in failed.stdout and "https://bot:***@" in failed.stdout
    done = job()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "cloned https://bot:***@example.org/owner/project.git at 0123abc" in done.stdout
    assert "the command ran" in done.stdout
    assert "s3cret-token" not in failed.stdout + failed.stderr + done.stdout + done.stderr


# -- experiments in repo mode ------------------------------------------------------------

BODY = "## Objective\n\nSee what the learning rate does.\n"

RUN_E1 = """\
import json
import subprocess
import sys

subprocess.run([sys.executable, "train.py"], check=True)
with open("metrics.json") as handle:
    metrics = json.load(handle)
metrics["wrapped"] = True
with open("metrics.json", "w") as handle:
    json.dump(metrics, handle)
"""


def project_with(tmp_path, repository: Path | None, **settings) -> Project:
    if repository is not None:
        settings["repo"] = {"path": str(repository), "ref": None}
    return Project.create(tmp_path / "project", title="Learning rate", settings=settings)


def repo_experiment(project: Project, files: dict[str, str], **fields):
    """An accepted and approved experiment in repo mode, written by the designer."""
    fields = {"mode": "repo", **fields}
    page = project.create_page(
        "experiment", "Raise the learning rate", BODY, author="agent:designer", fields=fields
    )
    if files:
        experiments.write_code(project, page.id, files, author="agent:designer")
    project.review(page.path, "accept")
    return page


def local(tmp_path) -> LocalBackend:
    return LocalBackend(scratch_dir=tmp_path / "scratch")


@needs_git
def test_an_experiment_runs_in_a_patched_copy_and_the_repository_stays_as_it_was(
    repository, tmp_path
):
    project = project_with(tmp_path, repository)
    page = repo_experiment(
        project, {"patch.diff": PATCH, "run_e1.py": RUN_E1}, command="python run_e1.py"
    )
    status = git(repository, "status", "--porcelain")
    before = snapshot(repository)

    plan = experiments.load_plan(project, page.id)
    assert plan.mode == "repo" and plan.repo == str(repository)
    assert plan.command == ["python", "run_e1.py"]
    assert experiments.check_code(project, page.id) == []

    backend = local(tmp_path)
    experiments.approve(project, page.id, backends={"local": backend})
    record = experiments.run(project, page.id, compute=backend)

    assert record["status"] == "succeeded", record["error"]
    assert record["metrics"] == {"score": 2.5, "wrapped": True}  # 0.25 * 10: the patch was applied
    assert record["repo"]["commit"] == git(repository, "rev-parse", "HEAD").strip()
    assert record["repo"]["method"] == "git" and record["repo"]["dirty"] is True
    assert {entry["path"] for entry in record["skipped"]} >= {".env", "venv/pyvenv.cfg"}
    # The repository itself was never run in, patched or written to.
    assert snapshot(repository) == before
    assert git(repository, "status", "--porcelain") == status
    assert "LEARNING_RATE = 0.1" in (repository / "train.py").read_text()
    # The prepared copy is not kept: the patch and the commit say what ran.
    run_dir = project.experiment_dir(page.id) / "runs" / "1"
    assert not (run_dir / "repo").exists()
    assert (run_dir / "code" / "patch.diff").read_text() == PATCH
    result = project.page("E1-run1")
    assert "patch.diff" in result.body and record["repo"]["commit"][:12] in result.body


@needs_git
def test_a_patch_that_does_not_apply_fails_the_run_with_the_rejected_hunk(repository, tmp_path):
    project = project_with(tmp_path, repository)
    wrong = PATCH.replace("-LEARNING_RATE = 0.1", "-LEARNING_RATE = 0.3")
    page = repo_experiment(project, {"patch.diff": wrong}, command="python train.py")
    backend = local(tmp_path)
    experiments.approve(project, page.id, backends={"local": backend})
    seen = []

    def repair(plan, record, log_tail):
        seen.append(record["error"])
        return {"patch.diff": PATCH}

    record = experiments.run(project, page.id, compute=backend, repair=repair)

    first = experiments.runs(project, page.id)[0]
    assert first["status"] == "failed" and first["error_kind"] == "patch"
    assert "-LEARNING_RATE = 0.3" in first["error"] and "train.py" in first["error"]
    assert seen == [first["error"]]
    # The corrected patch is the runner's repair, and the next run uses it.
    assert record["number"] == 2 and record["status"] == "succeeded"
    assert record["metrics"] == {"score": 2.5}
    assert "LEARNING_RATE = 0.1" in (repository / "train.py").read_text()


@needs_git
def test_an_experiment_runs_at_the_ref_the_project_names(repository, tmp_path):
    project = project_with(tmp_path, repository)
    project.settings.update({"repo": {"path": str(repository), "ref": "main"}})
    project.save_settings()
    committed = git(repository, "rev-parse", "HEAD").strip()
    (repository / "train.py").write_text(TRAIN.replace("0.1", "0.7"))  # not committed
    page = repo_experiment(project, {}, command="python train.py")
    backend = local(tmp_path)
    experiments.approve(project, page.id, backends={"local": backend})

    record = experiments.run(project, page.id, compute=backend)

    assert record["status"] == "succeeded", record["error"]
    assert record["metrics"] == {"score": 1.0}  # the committed 0.1, not the 0.7 of the tree
    assert record["repo"]["method"] == "archive" and record["repo"]["commit"] == committed
    assert experiments.load_plan(project, page.id).repo_ref == "main"


def test_repo_mode_needs_a_repository(tmp_path):
    project = project_with(tmp_path, None)
    page = repo_experiment(project, {}, command="python train.py")
    with pytest.raises(ProjectError, match="repository") as refused:
        experiments.load_plan(project, page.id)
    assert "settings" in str(refused.value)


def test_an_agent_cannot_point_an_experiment_at_another_directory(repository, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    project = project_with(tmp_path, repository)
    page = repo_experiment(project, {}, command="python train.py", repo=str(private))
    with pytest.raises(ProjectError, match="not the repository this project was given"):
        experiments.load_plan(project, page.id)
    # A directory inside the project's repository is fine, and so is the person's own page.
    inside = repo_experiment(project, {}, command="python util.py", repo=str(repository / "lib"))
    assert experiments.load_plan(project, inside.id).repo == str(repository / "lib")
    own = project.create_page(
        "experiment",
        "Mine",
        BODY,
        author="human",
        fields={"mode": "repo", "repo": str(private), "command": "python x.py"},
    )
    assert experiments.load_plan(project, own.id).repo == str(private)


def test_a_repository_behind_a_url_needs_network_access_to_be_approved(tmp_path):
    project = project_with(tmp_path, None)
    project.settings.update({"repo": {"path": "https://example.org/owner/project.git"}})
    project.save_settings()
    page = repo_experiment(project, {"patch.diff": PATCH}, command="python train.py")
    backend = local(tmp_path)
    with pytest.raises(ProjectError, match="network") as refused:
        experiments.approve(project, page.id, backends={"local": backend})
    assert "https://example.org/owner/project.git" in str(refused.value)
    assert project.page(page.id).state == "planned"


class Recording:
    """A backend that records the job it is given and reports success."""

    name = "local"

    def __init__(self) -> None:
        self.jobs: list[tuple[JobSpec, list[str]]] = []

    def supports(self, spec):
        return []

    def check(self, **kwargs):
        return []

    def describe(self):
        return {"backend": self.name, "jobs": {"supported": True, "reattach": False}}

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None):
        names = sorted(p.relative_to(source_dir).as_posix() for p in Path(source_dir).rglob("*"))
        self.jobs.append((spec, names))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "log.txt").write_text("")
        (Path(run_dir) / "metrics.json").write_text('{"score": 1}')
        return JobResult(returncode=0, backend=self.name, seconds=0.1, outputs_dir=Path(run_dir))


def test_a_repository_behind_a_url_is_cloned_by_the_job_once_network_is_approved(tmp_path):
    project = project_with(tmp_path, None)
    project.settings.update(
        {"repo": {"path": "https://example.org/owner/project.git", "ref": "v1"}}
    )
    project.save_settings()
    page = repo_experiment(
        project, {"patch.diff": PATCH, "run_e1.py": RUN_E1}, command="python run_e1.py"
    )
    backend = Recording()
    experiments.approve(project, page.id, network=True, backends={"local": backend})
    record = experiments.run(project, page.id, compute=backend)

    assert record["status"] == "succeeded"
    (spec, names) = backend.jobs[0]
    assert spec.network is True and spec.command[0] == "python"
    # Nothing was cloned on this machine: the job gets the instructions and the experiment's files.
    assert any(name.endswith("bootstrap.py") for name in names)
    assert not any(name.endswith("train.py") for name in names)
    assert record["repo"] == {
        "source": "https://example.org/owner/project.git",
        "ref": "v1",
        "method": "clone",
    }


def test_the_password_of_a_repository_url_stays_out_of_records_and_pages(tmp_path):
    project = project_with(tmp_path, None)
    url = "https://bot:s3cret-token@example.org/owner/project.git"
    project.settings.update({"repo": {"path": url}})
    project.save_settings()
    page = repo_experiment(project, {}, command="python train.py")
    backend = Recording()
    with pytest.raises(ProjectError, match="network") as refused:
        experiments.approve(project, page.id, backends={"local": backend})
    assert "s3cret-token" not in str(refused.value)

    experiments.approve(project, page.id, network=True, backends={"local": backend})
    record = experiments.run(project, page.id, compute=backend)

    assert record["status"] == "succeeded"
    assert record["repo"]["source"] == "https://bot:***@example.org/owner/project.git"
    run_dir = project.experiment_dir(page.id) / "runs" / "1"
    kept = "".join(path.read_text() for path in run_dir.rglob("*") if path.is_file())
    assert "s3cret-token" not in kept + project.page("E1-run1").body
    assert "s3cret-token" not in json.dumps(project.events())
    # The job itself is told where to clone from.
    assert experiments.load_plan(project, page.id).repo == url


def test_a_repository_that_cannot_be_copied_fails_the_run_and_nothing_else(
    tree, tmp_path, monkeypatch
):
    project = project_with(tmp_path, tree)
    page = repo_experiment(project, {}, command="python train.py")
    backend = local(tmp_path)
    experiments.approve(project, page.id, backends={"local": backend})

    def unreadable(source: Path, target: Path) -> None:
        raise PermissionError(13, "Permission denied", str(source))

    monkeypatch.setattr(repo, "_place", unreadable)
    record = experiments.run(project, page.id, compute=backend)  # a record, not an exception
    assert (record["status"], record["error_kind"]) == ("failed", "setup")
    assert "Permission denied" in record["error"]
    assert project.page(page.id).state == "failed"
    assert not (project.experiment_dir(page.id) / "runs" / "1" / "repo").exists()
