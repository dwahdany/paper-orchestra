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
import time
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


def stored(directory: Path, data: str, *args: str) -> str:
    """The id git gives ``data`` when it is fed to ``git <args>`` on standard input."""
    done = subprocess.run(
        ["git", *args], cwd=directory, input=data, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the tree holds symbolic links")
def test_a_commit_that_names_a_link_and_a_file_alike_writes_nothing_outside_the_copy(
    repository, tmp_path
):
    """A commit is data like any other, and one that came with the directory can say what
    no working tree can hold: a link, and under the same name a directory or a file.
    Written out in that order, the file would go where the link leads; and a link that
    stays inside the copy as it is spelled (``d1/up/..``) can lead out of it."""
    blob = ["hash-object", "-w", "--stdin"]
    up = stored(repository, "..", *blob)
    out = stored(repository, "d1/up/..", *blob)
    onto = stored(repository, "d1/up/../victim.txt", *blob)
    evil = stored(repository, "written by the repository\n", *blob)
    train = git(repository, "rev-parse", "HEAD:train.py").strip()
    d1 = stored(repository, f"120000 blob {up}\tup\n", "mktree")
    inner = stored(repository, f"100644 blob {evil}\tevil.txt\n", "mktree")
    listing = (
        f"040000 tree {d1}\td1\n"
        f"120000 blob {out}\tl2\n040000 tree {inner}\tl2\n"
        f"120000 blob {onto}\tx\n100644 blob {evil}\tx\n"
        f"100644 blob {train}\ttrain.py\n"
    )
    crafted = stored(repository, listing, "mktree")
    commit = git(
        repository, "-c", "user.name=T", "-c", "user.email=t@example.com", "commit-tree", "-m",
        "crafted", crafted,
    ).strip()  # fmt: skip
    git(repository, "update-ref", "refs/heads/crafted", commit)
    run = tmp_path / "runs" / "1"
    run.mkdir(parents=True)
    (run / "victim.txt").write_text("the run's own record\n")

    made = repo.copy_local(repository, run / "repo", ref="crafted")

    assert (run / "repo" / "train.py").read_text() == TRAIN
    assert (run / "victim.txt").read_text() == "the run's own record\n"
    assert sorted(path.name for path in run.iterdir()) == ["repo", "victim.txt"]
    assert not os.path.exists(run / "evil.txt")
    left_out = {entry["path"] for entry in made.skipped}
    assert {"l2/evil.txt", "x"} <= left_out


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


@needs_git
def test_files_git_lists_behind_a_link_are_not_read_through_it(repository, tmp_path):
    """A directory git tracks became a link to somewhere else (the person moved their
    data, or the repository came from someone who meant it): git still lists the files
    it tracked there, and reading them would bring in what lies outside the repository,
    the private files of this machine among them."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "util.py").write_text("PRIVATE = 'not part of the repository'\n")
    shutil.rmtree(repository / "lib")
    (repository / "lib").symlink_to(private)
    assert "lib/util.py" in git(repository, "ls-files")

    made = repo.copy_local(repository, tmp_path / "copy")

    assert made.method == "git"
    assert not os.path.lexists(tmp_path / "copy" / "lib")
    assert (tmp_path / "copy" / "train.py").read_text() == TRAIN
    reasons = {entry["path"]: entry["reason"] for entry in made.skipped}
    assert "leads out of the repository" in reasons["lib/util.py"]
    assert "leads out of the repository" in reasons["lib"]


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
def test_a_listing_that_leads_out_of_the_repository_is_not_followed(tree, tmp_path, monkeypatch):
    """What git lists comes from the repository's index, a file like any other: one that
    came with the directory can name anything.  Nothing outside the repository is read
    for the copy, and nothing is written outside the copy."""
    git(tree, "init", "-q", "-b", "main")
    (tmp_path / "outside.txt").write_text("outside the repository\n")
    (tmp_path / "bin").mkdir()
    listing = tmp_path / "listing"
    names = [b"train.py", b"../outside.txt", b"lib/../../outside.txt"]
    listing.write_bytes(b"\0".join([*names, os.fsencode(tmp_path / "outside.txt"), b""]))
    forged = tmp_path / "bin" / "git"
    forged.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        f'  if [ "$arg" = ls-files ]; then exec cat {listing}; fi\n'
        "done\n"
        f'exec {shutil.which("git")} "$@"\n'
    )
    forged.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    dest = tmp_path / "runs" / "1" / "repo"

    made = repo.copy_local(tree, dest)

    assert made.method == "git" and copied(dest) == {"train.py"}
    assert copied(tmp_path / "runs") == {"1/repo/train.py"}
    assert (tmp_path / "outside.txt").read_text() == "outside the repository\n"
    left_out = {entry["path"] for entry in made.skipped}
    assert {"../outside.txt", "lib/../../outside.txt", str(tmp_path / "outside.txt")} <= left_out


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


def hostile(repository: Path, pwned: Path, **more: str) -> None:
    """Give ``repository`` a git configuration that starts a program wherever git lets a
    configuration start one, each of which leaves its environment in ``pwned``."""
    pwned.mkdir()

    def program(name: str) -> str:
        return f"sh -c 'env > {pwned}/{name}; cat'"

    (repository / ".gitattributes").write_text("* filter=evil diff=evil\nlib/* filter=other\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "attributes")
    settings = {
        "core.fsmonitor": f"sh -c 'env > {pwned}/fsmonitor; echo; false'",
        "filter.evil.clean": program("filter-clean"),
        "filter.evil.smudge": program("filter-smudge"),
        "filter.evil.required": "true",
        "filter.other.process": program("filter-process"),
        "diff.evil.textconv": program("textconv"),
        "diff.external": program("diff-external"),
        "core.pager": program("pager"),
        "core.sshCommand": program("ssh"),
        **more,
    }
    for key, value in settings.items():
        git(repository, "config", key, value)
    hooks = repository / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    for name in ("post-checkout", "post-index-change", "reference-transaction", "pre-commit"):
        (hooks / name).write_text(f"#!/bin/sh\nenv > '{pwned}/hook-{name}'\n")
        (hooks / name).chmod(0o755)
    # The time stamp differs from the index: to tell whether the file changed, git has
    # to look at its content, which is what the filters are for.
    os.utime(repository / "train.py", (1_600_000_000, 1_600_000_000))


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the programs are shell commands")
def test_reading_a_repository_starts_no_program_its_configuration_names(
    repository, tmp_path, monkeypatch
):
    """The directory may have come from anywhere, and the worker holds the model's key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-NOT-A-KEY-000")
    pwned = tmp_path / "pwned"
    hostile(repository, pwned)
    committed = git(repository, "rev-parse", "HEAD").strip()
    before = snapshot(repository)

    made = repo.copy_local(repository, tmp_path / "as-git-sees-it")
    assert sorted(os.listdir(pwned)) == []
    # Touched, not changed: git compared the content as it is, without the filter.
    assert made.method == "git" and made.commit == committed and made.dirty is False
    assert (tmp_path / "as-git-sees-it" / "train.py").read_text() == TRAIN

    at_ref = repo.copy_local(repository, tmp_path / "at-the-ref", ref="main")
    assert sorted(os.listdir(pwned)) == []
    assert at_ref.method == "archive" and at_ref.commit == committed
    assert (tmp_path / "at-the-ref" / "train.py").read_text() == TRAIN
    assert (tmp_path / "at-the-ref" / "lib" / "util.py").read_text() == UTIL

    repo.apply_patch(tmp_path / "at-the-ref", PATCH, use_git=True)
    assert sorted(os.listdir(pwned)) == []
    assert snapshot(repository) == before


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
def test_git_is_given_no_credential_and_nothing_that_points_it_elsewhere(
    repository, tmp_path, monkeypatch
):
    """git starts programs of its own and of the person's configuration.  What the worker
    holds (the model's key, a token) is not for them, and what would point git at another
    repository or another way of reaching one is not passed on."""
    (tmp_path / "bin").mkdir()
    seen = tmp_path / "environments"
    spy = tmp_path / "bin" / "git"
    spy.write_text(f'#!/bin/sh\nenv >> {seen}\nexec {shutil.which("git")} "$@"\n')
    spy.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-NOT-A-KEY-000")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_NOT-A-TOKEN-000")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/someone/.ssh/id_ed25519")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.pager")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "less")

    made = repo.copy_local(repository, tmp_path / "copy")
    at_ref = repo.copy_local(repository, tmp_path / "at-the-ref", ref="main")
    repo.apply_patch(tmp_path / "at-the-ref", PATCH, use_git=True)

    assert made.method == "git" and at_ref.method == "archive"
    given = seen.read_text()
    assert given.count("GIT_TERMINAL_PROMPT=0") >= 6  # every call went through the stand-in
    for name in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GIT_SSH_COMMAND", "GIT_CONFIG_COUNT"):
        assert f"{name}=" not in given, name
    assert "NOT-A-KEY" not in given and "NOT-A-TOKEN" not in given


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the programs are shell commands")
def test_whether_the_tree_is_dirty_is_told_without_running_filters(tree, tmp_path):
    git(tree, "init", "-q", "-b", "main")
    git(tree, "add", ".")
    git(tree, "commit", "-q", "-m", "first")
    pwned = tmp_path / "pwned"
    hostile(tree, pwned)

    # Touched, not changed: the content is compared as it is, and it is the same.
    assert repo.copy_local(tree, tmp_path / "clean").dirty is False
    (tree / "train.py").write_text(TRAIN + "# changed\n")
    assert repo.copy_local(tree, tmp_path / "changed").dirty is True
    assert os.listdir(pwned) == []
    git(tree, "commit", "-q", "-a", "-m", "second", "--no-verify")
    for name in os.listdir(pwned):  # what the commit itself started, here in the test
        (pwned / name).unlink()
    (tree / "notes.txt").write_text("not committed\n")
    assert repo.copy_local(tree, tmp_path / "untracked").dirty is True
    assert os.listdir(pwned) == []

    # A filter that cannot be switched off is not run either: the flag stays unknown, and
    # a copy at a ref, which git would write through the filter, is refused.
    (tree / ".gitattributes").write_text("* filter=a=b\n")
    git(tree, "config", "filter.a=b.clean", f"sh -c 'env > {pwned}/odd-clean; cat'")
    git(tree, "config", "filter.a=b.smudge", f"sh -c 'env > {pwned}/odd-smudge; cat'")
    os.utime(tree / "train.py", (1_500_000_000, 1_500_000_000))
    made = repo.copy_local(tree, tmp_path / "odd")
    assert made.dirty is None and made.method == "git"
    assert (tmp_path / "odd" / "train.py").exists()
    with pytest.raises(repo.RepoError, match="cannot be switched off") as refused:
        repo.copy_local(tree, tmp_path / "odd-ref", ref="main")
    assert "Remove the ref" in str(refused.value)
    assert os.listdir(pwned) == []


def partial_clone(tmp_path: Path, repository: Path) -> Path:
    """A clone of ``repository`` that holds the files of its newest commit only and names
    the remote git may fetch the others from when it needs them (a "blobless" clone)."""
    git(repository, "branch", "before")
    (repository / "train.py").write_text(TRAIN + "# second version\n")
    git(repository, "commit", "-q", "-a", "-m", "second")
    git(repository, "config", "uploadpack.allowFilter", "true")
    clone = tmp_path / "partial"
    git(tmp_path, "clone", "-q", "--filter=blob:none", f"file://{repository}", str(clone))
    missing = git(clone, "rev-list", "--objects", "--missing=print", "origin/before")
    assert "?" in missing, "the clone holds every file, so nothing would be fetched"
    return clone


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the programs are shell commands")
@pytest.mark.parametrize("transport", ["uploadpack", "ext", "ssh"])
@pytest.mark.parametrize("git_knows_the_switch", [True, False])
def test_a_copy_never_fetches_what_a_partial_clone_lacks(
    repository, tmp_path, monkeypatch, transport, git_knows_the_switch
):
    """Where a repository says git may fetch missing files from, git does so by itself
    when a command needs one, with the program its configuration names for the way
    there.  A copy runs none of them and contacts nobody: it takes what is on the disk."""
    clone = partial_clone(tmp_path, repository)
    pwned = tmp_path / "pwned"
    pwned.mkdir()
    if transport == "uploadpack":
        git(clone, "config", "remote.origin.uploadpack", f"touch {pwned}/ran; git-upload-pack")
    elif transport == "ext":
        git(clone, "config", "remote.origin.url", f"ext::sh -c touch% {pwned}/ran")
        git(clone, "config", "protocol.ext.allow", "always")
    else:
        git(clone, "config", "remote.origin.url", "ssh://git.example.invalid/x.git")
        git(clone, "config", "core.sshCommand", f"touch {pwned}/ran; false")
    if not git_knows_the_switch:
        # A git older than 2.45 ignores GIT_NO_LAZY_FETCH and tries to fetch: it must
        # then find that no way to another repository is open to it.
        environment = repo._git_env

        def older(ceiling: Path | None = None) -> dict[str, str]:
            env = environment(ceiling)
            env.pop("GIT_NO_LAZY_FETCH", None)
            return env

        monkeypatch.setattr(repo, "_git_env", older)
    before = snapshot(clone)

    # The newest commit is all there: it is copied, at the ref and as the tree stands.
    made = repo.copy_local(clone, tmp_path / "newest", ref="main")
    assert made.method == "archive"
    assert (tmp_path / "newest" / "train.py").read_text() == TRAIN + "# second version\n"
    assert repo.copy_local(clone, tmp_path / "as-it-stands").dirty is False

    # The older one is not, and is not fetched: the copy is refused with the way out.
    with pytest.raises(repo.RepoError) as refused:
        repo.copy_local(clone, tmp_path / "older", ref="origin/before")
    assert "partial clone" in str(refused.value) and "nothing is fetched" in str(refused.value)
    assert os.listdir(pwned) == []
    assert snapshot(clone) == before


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
def test_a_git_that_hangs_while_exporting_a_ref_is_stopped(repository, tmp_path, monkeypatch):
    """Nothing is read from it, so nothing came by that asked whether to go on."""
    (tmp_path / "bin").mkdir()
    slow = tmp_path / "bin" / "git"
    slow.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  if [ "$arg" = archive ]; then echo $$ > "$SLOW_GIT_PID"; exec sleep 20; fi\n'
        "done\n"
        f'exec {shutil.which("git")} "$@"\n'
    )
    slow.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SLOW_GIT_PID", str(tmp_path / "pid"))
    monkeypatch.setattr(repo, "_GIT_ENV", (*repo._GIT_ENV, "SLOW_GIT_PID"), raising=False)
    monkeypatch.setattr(repo, "CHECK_SECONDS", 0.0)

    def gone() -> bool:
        pid = int((tmp_path / "pid").read_text())
        (tmp_path / "pid").unlink()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) > 5

    started = time.monotonic()
    stopped = outcome_of(
        lambda: repo.copy_local(repository, tmp_path / "a", ref="main", should_stop=should_stop),
        seconds=8,
    )
    assert isinstance(stopped, JobStopped), stopped
    assert time.monotonic() - started < 6 and gone()

    monkeypatch.setattr(repo, "ARCHIVE_SECONDS", 0.5, raising=False)
    late = outcome_of(lambda: repo.copy_local(repository, tmp_path / "b", ref="main"), seconds=8)
    assert isinstance(late, repo.RepoError) and "took longer than 0.5 s" in str(late)
    assert gone()


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


@pytest.mark.parametrize(
    "path", ["metrics.json", "./metrics.json", "artifacts/table.csv", "artifacts/deep/fig.png"]
)
def test_a_patch_may_not_put_results_where_the_run_writes_them(tree, path):
    """A number in a file the designer wrote is not a number a run measured."""
    create = f'--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{{"accuracy": 0.9731}}\n'
    for use_git in [True, False] if shutil.which("git") else [False]:
        with pytest.raises(repo.PatchError, match="results come from running") as refused:
            repo.apply_patch(tree, PATCH + create, use_git=use_git)
        assert path in str(refused.value) and "metrics.json" in str(refused.value)
    # All or nothing: the change to train.py in the same patch was not applied either.
    assert (tree / "train.py").read_text() == TRAIN
    assert not (tree / "metrics.json").exists() and not (tree / "artifacts").exists()
    with pytest.raises(repo.PatchError, match="results come from running"):
        repo.clone_source(tree.parent / "seed", url="/nowhere", patch=create, command=["true"])


def test_a_patch_may_not_touch_results_the_repository_holds(tree):
    (tree / "metrics.json").write_text('{"accuracy": 0.5}\n')
    change = (
        "--- a/metrics.json\n+++ b/metrics.json\n@@ -1 +1 @@\n"
        '-{"accuracy": 0.5}\n+{"accuracy": 0.999}\n'
    )
    delete = '--- a/metrics.json\n+++ /dev/null\n@@ -1 +0,0 @@\n-{"accuracy": 0.5}\n'
    for patch in (change, delete):
        with pytest.raises(repo.PatchError, match="results come from running"):
            repo.parse_patch(patch)
        with pytest.raises(repo.PatchError, match="removed before it starts"):
            repo.apply_patch(tree, patch, use_git=False)
    assert json.loads((tree / "metrics.json").read_text()) == {"accuracy": 0.5}

    # A file of that name further down is the repository's own business.
    inner = "--- /dev/null\n+++ b/out/metrics.json\n@@ -0,0 +1 @@\n+{}\n"
    assert repo.apply_patch(tree, inner, use_git=False) == ["out/metrics.json"]


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
        # A password written with an @ in it: git takes the last one for the host's.
        ("https://bot:p@ss@github.com/o/p.git", "https://bot:***@github.com/o/p.git"),
        ("https://to@ken@github.com/o/p.git", "https://***@github.com/o/p.git"),
        ("https://github.com/owner/p@v2.git", "https://github.com/owner/p@v2.git"),
    ],
)
def test_a_url_is_shown_without_its_password_or_token(url, shown):
    assert repo.display_url(url) == shown


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        ("https://github.com/owner/project.git", None),
        ("https://ghp_s3cret@github.com/o/p.git", "ghp_s3cret"),
        ("https://bot:ghp_s3cret@github.com/o/p.git", "ghp_s3cret"),
        ("https://bot:p%40ss@github.com/o/p.git", "p%40ss"),
        ("https://bot:@github.com/o/p.git", None),
        ("ssh://git@example.org/owner/project", None),
        ("ssh://me:s3cret@example.org/o/p", "s3cret"),
        ("git@github.com:owner/project.git", None),
        ("/home/someone/project", None),
        ("", None),
        ("https://bot:p@ss@github.com/o/p.git", "p@ss"),
        ("https://to@ken@github.com/o/p.git", "to@ken"),
        ("https://github.com/owner/p@v2.git", None),
        ("https://@github.com/o/p.git", None),
    ],
)
def test_the_secret_of_a_url_is_named_for_masking(url, secret):
    assert repo.userinfo(url) == secret
    if secret:
        assert secret not in repo.display_url(url)
        # Nor is any part of it in what a tool is told the address is.
        assert not set(secret.split("@")) & set(repo._without_secret(url).split("@"))


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
def test_results_that_came_with_a_cloned_repository_are_not_results_of_the_run(
    repository, tmp_path
):
    (repository / "metrics.json").write_text('{"accuracy": 0.9912, "note": "of last year"}\n')
    (repository / "artifacts").mkdir()
    (repository / "artifacts" / "old_curve.png").write_bytes(b"old")
    git(repository, "add", "metrics.json", "artifacts")
    git(repository, "commit", "-q", "-m", "results of earlier work")
    look = (
        "import os, subprocess\n"
        "print('stale:', sorted(n for n in ('metrics.json', 'artifacts') if os.path.lexists(n)))\n"
        "print('fetch head kept:', os.path.exists('.git/FETCH_HEAD'))\n"
        "print('origin:', subprocess.run(['git', 'remote', 'get-url', 'origin'],\n"
        "      capture_output=True, text=True).stdout.strip())\n"
        "print('training... (this script writes no metrics)')\n"
    )
    seed = tmp_path / "seed"
    command = repo.clone_source(
        seed, url=str(repository), files={"look.py": look.encode()}, command=["python", "look.py"]
    )
    result = LocalBackend(scratch_dir=tmp_path / "scratch").run_job(
        JobSpec(command=command, timeout=60, network=True),
        source_dir=seed,
        run_dir=tmp_path / "run",
    )
    log = (tmp_path / "run" / "log.txt").read_text()
    assert result.ok, log
    assert "stale: []" in log and "fetch head kept: False" in log
    assert f"origin: {repository}" in log
    assert not (tmp_path / "run" / "metrics.json").exists()
    assert not (tmp_path / "run" / "artifacts").exists()
    [said] = [line for line in repo.clone_lines(log.splitlines()) if line.startswith("Removed")]
    assert "Removed metrics.json, artifacts that was there before the run" in said


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the repository holds a symbolic link")
def test_the_clone_job_removes_no_result_through_a_link_of_the_repository(repository, tmp_path):
    """A repository can hold a link where the run writes its results to.  What lies
    behind it is not the working directory's, and is never collected either."""
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "metrics.json").write_text('{"accuracy": 0.9}\n')
    (repository / "results").symlink_to(kept, target_is_directory=True)
    git(repository, "add", "results")
    git(repository, "commit", "-q", "-m", "results live elsewhere")
    seed = tmp_path / "seed"
    command = repo.clone_source(
        seed,
        url=str(repository),
        command=["python", "train.py"],
        outputs=("results/metrics.json", "metrics.json"),
    )
    result = LocalBackend(scratch_dir=tmp_path / "scratch").run_job(
        JobSpec(command=command, timeout=60, network=True, outputs=("metrics.json",)),
        source_dir=seed,
        run_dir=tmp_path / "run",
    )
    assert result.ok, (tmp_path / "run" / "log.txt").read_text()
    assert (kept / "metrics.json").read_text() == '{"accuracy": 0.9}\n'


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


#: Stands in for git in a clone job: no test may reach a network.  It writes down how it
#: was called (outside the working directory) and leaves the address where a git might:
#: in FETCH_HEAD, in a log of its own and, for a remote, in the configuration.  It fails
#: the fetch when asked to, quoting the URL the way git does.
FAKE_GIT = """\
#!/bin/sh
while [ "$1" = "-c" ]; do shift 2; done
echo "$@" >> "$FAKE_GIT_CALLS"
case "$1" in
  init) mkdir -p .git/logs ;;
  fetch) if [ -n "$FAKE_GIT_FAILS" ]; then
           echo "fatal: unable to access '$5/': refused"
           echo "fatal: Authentication failed for $FAKE_GIT_FAILS"
           exit 128
         fi
         echo "0123abc branch 'main' of $5" > .git/FETCH_HEAD
         echo "fetched from $5" > .git/logs/fetches ;;
  remote) echo "url = $4" > .git/config ;;
  rev-parse) echo 0123abc ;;
esac
exit 0
"""

#: What an experiment's command can find out about where its code came from.
SPY = """\
import json, os
print("the command ran")
print("job:", json.dumps(json.load(open(".paper-orchestra/job.json")), sort_keys=True))
print("config:", open(".git/config").read().strip())
print("environment:", json.dumps(dict(os.environ), sort_keys=True))
for folder, _, names in os.walk("."):
    for name in names:
        path = os.path.join(folder, name)
        print("file:", path, repr(open(path, errors="replace").read()))
"""


def clone_job(tmp_path: Path, url: str, **env: str) -> subprocess.CompletedProcess:
    """Run the clone job for ``url`` with the stand-in for git, and the spy as its command."""
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "git").write_text(FAKE_GIT)
    (tmp_path / "bin" / "git").chmod(0o755)
    seed = tmp_path / f"seed-{len(list(tmp_path.glob('seed-*')))}"
    command = repo.clone_source(seed, url=url, command=[sys.executable, "-c", SPY])
    calls = tmp_path / "git-calls"
    calls.unlink(missing_ok=True)
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "FAKE_GIT_CALLS": str(calls), **env}
    done = subprocess.run(
        [sys.executable, command[1]], cwd=seed, env=env, capture_output=True, text=True
    )
    done.calls = calls.read_text().splitlines()  # type: ignore[attr-defined]
    done.seed = seed  # type: ignore[attr-defined]
    return done


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
def test_the_clone_job_never_prints_the_password_of_its_url(tmp_path):
    url = "https://bot:s3cret-token@example.org/owner/project.git"
    failed = clone_job(tmp_path, url, FAKE_GIT_FAILS="bot:s3cret-token")
    assert repo.clone_failure(failed.returncode) == repo.clone_failure(repo.EXIT_CLONE_FAILED)
    assert "could not be cloned" in failed.stdout and "https://bot:***@" in failed.stdout
    # A secret taken out of what git said is marked as everywhere else in a log; the
    # address keeps the form it is shown in (display_url).
    assert "Authentication failed for bot:[redacted]" in failed.stdout
    done = clone_job(tmp_path, url)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "cloned https://bot:***@example.org/owner/project.git at 0123abc" in done.stdout
    assert "the command ran" in done.stdout
    assert "s3cret-token" not in failed.stdout + failed.stderr + done.stdout + done.stderr

    # A password is written into a URL in one form and quoted by git in another.
    encoded = clone_job(
        tmp_path, "https://bot:p%40ss-w0rd@example.org/o/p.git", FAKE_GIT_FAILS="p@ss-w0rd"
    )
    assert "Authentication failed for [redacted]" in encoded.stdout
    assert "p@ss-w0rd" not in encoded.stdout and "p%40ss-w0rd" not in encoded.stdout

    # ...or written with the @ as it is.  git then names the address without what stands
    # before the first @, which is not all of the password.
    as_git_names_it = "https://ss-w0rd@example.org/o/p.git"
    plain = clone_job(
        tmp_path, "https://bot:p@ss-w0rd@example.org/o/p.git", FAKE_GIT_FAILS=as_git_names_it
    )
    assert "Authentication failed for https://bot:***@example.org/o/p.git" in plain.stdout
    assert "ss-w0rd" not in plain.stdout + plain.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in for git is a shell script")
@pytest.mark.parametrize(
    ("url", "secret", "remote"),
    [
        (
            "https://ghp_SECRETTOKEN123@example.org/org/private.git",
            "ghp_SECRETTOKEN123",
            "https://example.org/org/private.git",
        ),
        (
            "https://bot:s3cret-token@example.org/org/private.git",
            "s3cret-token",
            "https://example.org/org/private.git",
        ),
        ("ssh://me:s3cret-token@example.org/o/p", "s3cret-token", "ssh://me@example.org/o/p"),
        (
            "https://bot:s3cret@token@example.org/org/private.git",
            "s3cret@token",
            "https://example.org/org/private.git",
        ),
    ],
)
def test_the_token_of_a_repository_url_is_out_of_the_commands_reach(tmp_path, url, secret, remote):
    """The job has to be told the token to clone at all.  The experiment's command runs
    next, in the same directory, and it is code a model wrote."""
    done = clone_job(tmp_path, url)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "the command ran" in done.stdout
    # The spy printed every file under its working directory, the job's description, the
    # remote and its environment: the token is in none of them.
    assert secret not in done.stdout + done.stderr
    assert f"config: url = {remote}" in done.stdout
    assert '"shown": "' + repo.display_url(url) + '"' in done.stdout
    assert f'"url": "{remote}"' in done.stdout  # what is left of the address in job.json
    # git was given the address for the fetch, and for nothing else.
    assert f"fetch -q --depth 1 {url} HEAD" in done.calls
    assert [call for call in done.calls if secret in call] == [f"fetch -q --depth 1 {url} HEAD"]
    assert f"remote add origin {remote}" in done.calls
    # What the stand-in wrote down of the fetch is gone or without the token.
    left = "".join(
        path.read_text(errors="replace") for path in done.seed.rglob("*") if path.is_file()
    )
    assert secret not in left and not (done.seed / ".git" / "FETCH_HEAD").exists()

    # Also when the clone fails: the job's own description no longer names it.
    failed = clone_job(tmp_path, url, FAKE_GIT_FAILS="1")
    assert failed.returncode == repo.EXIT_CLONE_FAILED
    assert secret not in (failed.seed / ".paper-orchestra" / "job.json").read_text()


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
def test_results_that_lie_in_the_repository_are_not_recorded_as_the_runs(repository, tmp_path):
    """The repository holds a metrics.json and figures of earlier work, and the command
    of the experiment writes its own somewhere else, or nowhere."""
    (repository / "metrics.json").write_text('{"accuracy": 0.97, "note": "last month"}\n')
    (repository / "artifacts").mkdir()
    (repository / "artifacts" / "old_curve.png").write_bytes(b"old")
    git(repository, "add", "metrics.json", "artifacts")
    git(repository, "commit", "-q", "-m", "results of earlier work")
    project = project_with(tmp_path, repository)
    quiet = "import os\nos.makedirs('out')\nopen('out/metrics.json', 'w').write('{}')\n"
    page = repo_experiment(project, {"quiet.py": quiet}, command="python quiet.py")
    backend = local(tmp_path)
    experiments.approve(project, page.id, backends={"local": backend})

    record = experiments.run(project, page.id, compute=backend)

    assert record["status"] == "failed", record
    assert "metrics.json" in record["error"] and record["metrics"] == {}
    assert record["artifacts"] == []
    # ...and the record says what became of them.
    assert any("before the run" in note and "metrics.json" in note for note in record["notes"])
    run_dir = project.experiment_dir(page.id) / "runs" / "1"
    assert not (run_dir / "metrics.json").exists() and not (run_dir / "artifacts").exists()
    assert (repository / "metrics.json").exists()  # the person's file is where it was

    # A patch that would write the result in the command's place is refused when the
    # code is checked, before anybody is asked to approve it.
    forged = '--- /dev/null\n+++ b/metrics.json\n@@ -0,0 +1 @@\n+{"accuracy": 0.999}\n'
    other = repo_experiment(project, {"patch.diff": forged}, command="python train.py")
    problems = experiments.check_code(project, other.id)
    assert any("results come from running" in str(problem) for problem in problems), problems


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

    def run_job(self, spec, *, source_dir, run_dir, on_log=None, should_stop=None, mask=()):
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
