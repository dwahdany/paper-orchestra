"""Worker processes of the web server: start, stop, death, adoption, shutdown.

Every worker here is a small Python script that behaves like the engine in one
respect (it stops on an interrupt, ignores it, dies, starts a job of its own).  It
is called ``paper-orchestra`` and started as ``paper-orchestra research <kind>
<project>``, as the real command is: a server only stops a process it did not start
when that process is, by its command line, a worker of the project.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_orchestra.research import Project
from paper_orchestra.research.model import format_stamp
from paper_orchestra.web import workers as workers_module
from paper_orchestra.web.workers import (
    RECORD_FILE,
    WORKER_LOG,
    WorkerBusy,
    WorkerError,
    Workers,
    default_command,
    mark_interrupted,
)

HOST = socket.gethostname()
LOCK_FILE = ".worker.lock"

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and signals")

# What the stand-in worker does, by its last argument.  Its command line is that of a
# worker: ``<this script> research <kind> <project> <mode>``.
WORKER_SCRIPT = r"""
import fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

# The project is the argument after the kind, as for the real command; a worker that was
# given its project in another way (by its id) finds the directory in the environment.
root = Path(os.environ.get("STAND_IN_ROOT") or sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "sleep"
state_file = root / "state.json"
if os.environ.get("STAND_IN_LOCK"):
    # Like the engine: the project's worker lock is held for as long as the process lives.
    lock = os.open(root / ".worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def write_state(status, phase_status):
    state = {"schema": 1, "status": status, "round": 1, "phase": "research",
             "phases": [{"round": 1, "phase": "research", "status": phase_status,
                         "started": "2026-01-01T00:00:00Z", "finished": None, "error": None}],
             "worker": {"pid": os.getpid()}, "error": None}
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    os.replace(temporary, state_file)


def ready():
    print("worker ready", mode, flush=True)
    (root / "ready").write_text(str(os.getpid()))


if mode == "quick":
    write_state("checkpoint", "completed")
    print("done at a checkpoint", flush=True)
    sys.exit(0)
if mode == "die":
    write_state("running", "running")
    ready()
    time.sleep(0.2)
    os._exit(3)
if mode == "ignore":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    write_state("running", "running")
    ready()
    time.sleep(120)
if mode == "term-only":
    # A worker of a server that was started in the background: the interrupt is ignored.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def on_term(signum, frame):
        write_state("stopped", "interrupted")
        (root / "terminated").write_text("yes")
        sys.exit(130)

    signal.signal(signal.SIGTERM, on_term)
    write_state("running", "running")
    ready()
    time.sleep(120)
if mode == "job":
    # Like the engine: a job in a session of its own, which the interrupt does not reach.
    job = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)"],
        start_new_session=True,
    )
    (root / "job.pid").write_text(str(job.pid))
if mode == "helper":
    # A child that stays in the worker's process group and ignores the interrupt.
    helper = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)"],
    )
    (root / "job.pid").write_text(str(helper.pid))
if mode == "late-job":
    # Ignores the interrupt, and only then starts a job in a session of its own.
    def on_interrupt(signum, frame):
        job = subprocess.Popen(
            [sys.executable, "-c",
             "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(120)"],
            start_new_session=True,
        )
        (root / "job.pid").write_text(str(job.pid))

    signal.signal(signal.SIGINT, on_interrupt)
    write_state("running", "running")
    ready()
    while True:
        time.sleep(0.05)
write_state("running", "running")
ready()
try:
    time.sleep(120)
except KeyboardInterrupt:
    write_state("stopped", "interrupted")
    (root / "interrupted").write_text("yes")
    sys.exit(130)
"""


def wait_for(condition, timeout: float = 60.0, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        time.sleep(0.02)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "paper-orchestra"
    path.parent.mkdir()
    path.write_text(WORKER_SCRIPT, encoding="utf-8")
    return path


def worker_command(script: Path, kind: str, root: Path | str, mode: str = "sleep") -> list[str]:
    """The command line of a stand-in worker of the project in ``root``."""
    return [sys.executable, str(script), "research", kind, str(root), mode]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "ws" / "proj", title="Warmup", brief="Does warmup matter?")


@pytest.fixture
def make_workers(script):
    """Build supervisors whose workers are the stand-in script; all are shut down after.
    ``make.settled`` lists the projects whose runs a supervisor was asked to settle."""
    made: list[Workers] = []
    settled: list[Path] = []

    def make(mode: str = "sleep", grace: float = 10.0, **options) -> Workers:
        def launcher(kind, root, options):
            return worker_command(script, kind, root, options.get("mode", mode))

        options.setdefault("settle_runs", settled.append)
        workers = Workers(launcher=launcher, grace=grace, **options)
        made.append(workers)
        return workers

    make.settled = settled
    yield make
    for workers in made:
        workers.shutdown(grace=1.0)


def started(workers: Workers, project: Project, **options) -> dict:
    status = workers.start(project.root, "advance", **options)
    wait_for(lambda: (project.root / "ready").exists(), message="the worker to be ready")
    return status


def state_of(project: Project) -> dict:
    return json.loads((project.root / "state.json").read_text())


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def test_start_runs_the_worker_in_its_own_session_and_logs_to_a_file(project, make_workers):
    workers = make_workers()
    status = started(workers, project, phase="research", instruction="Look at Adam first.")

    assert status["state"] == "running" and status["kind"] == "advance"
    assert status["phase"] == "research" and status["adopted"] is False
    pid = status["pid"]
    assert process_exists(pid)
    assert os.getsid(pid) == pid != os.getsid(0)  # leads a session of its own
    record = json.loads((project.root / RECORD_FILE).read_text())
    assert record["pid"] == pid and record["kind"] == "advance"
    assert record["options"] == {"phase": "research", "instruction": "Look at Adam first."}
    log = (project.root / WORKER_LOG).read_text()
    assert "[paper-orchestra]" in log and "starting:" in log
    wait_for(lambda: "worker ready sleep" in (project.root / WORKER_LOG).read_text())
    assert workers.status(project.root)["state"] == "running"
    assert workers.glance(project.root) == "running" and workers.known(project.root)
    assert workers.live() == 1


def test_a_second_start_is_refused_with_the_running_phase(project, make_workers):
    workers = make_workers()
    first = started(workers, project)
    with pytest.raises(WorkerBusy) as refused:
        workers.start(project.root, "advance", phase="design")
    assert refused.value.phase == "research"  # what state.json says is running
    assert refused.value.status["pid"] == first["pid"]
    assert "already has a worker" in str(refused.value)
    # ... also for another kind of work on the same project.
    with pytest.raises(WorkerBusy):
        workers.start(project.root, "run", experiments=["E1"])


def test_a_worker_that_finishes_leaves_the_project_idle(project, make_workers):
    workers = make_workers("quick")
    workers.start(project.root, "advance")
    assert workers.wait(project.root, timeout=60)
    status = workers.status(project.root)
    assert status["state"] == "idle"
    assert status["last_exit"]["code"] == 0 and status["last_exit"]["stopped"] is False
    assert status["last_exit"]["interrupted_phase"] is None
    assert state_of(project)["status"] == "checkpoint"  # the worker's own account stands
    assert not (project.root / RECORD_FILE).exists()
    assert "done at a checkpoint" in (project.root / WORKER_LOG).read_text()
    # The next worker appends to the same log.
    workers.start(project.root, "advance")
    assert workers.wait(project.root, timeout=60)
    assert (project.root / WORKER_LOG).read_text().count("starting:") == 2


def test_a_command_that_cannot_be_started_is_reported_and_nothing_runs(project):
    workers = Workers(launcher=lambda kind, root, options: ["/nonexistent/interpreter", "x"])
    try:
        with pytest.raises(WorkerError, match="could not be started"):
            workers.start(project.root, "advance")
        assert workers.status(project.root)["state"] == "idle"
        with pytest.raises(WorkerError, match="not a kind of worker"):
            workers.start(project.root, "dance")
    finally:
        workers.shutdown(grace=1.0)


def test_the_worker_gets_the_servers_environment_unbuffered(tmp_path, project):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\n"
        "print('ENV', os.environ.get('PYTHONUNBUFFERED'), os.environ.get('PO_TEST_MARK'),"
        " os.environ.get('COLUMNS'), sys.stdin.read() == '')\n"
    )
    workers = Workers(
        launcher=lambda kind, root, options: [sys.executable, str(probe)],
        environ={**os.environ, "PO_TEST_MARK": "from-the-server"},
    )
    try:
        workers.start(project.root, "advance")
        assert workers.wait(project.root, timeout=60)
        assert "ENV 1 from-the-server 200 True" in (project.root / WORKER_LOG).read_text()
    finally:
        workers.shutdown(grace=1.0)


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_stop_returns_at_once_and_the_worker_records_its_own_interruption(project, make_workers):
    workers = make_workers()
    started(workers, project)
    before = time.monotonic()
    status = workers.stop(project.root)
    assert time.monotonic() - before < 5.0  # the grace period alone is ten seconds
    assert status["state"] == "stopping"
    assert workers.stop(project.root)["state"] in ("stopping", "idle")  # asking twice is fine
    assert workers.wait(project.root, timeout=60)

    assert (project.root / "interrupted").exists()  # it saw the interrupt, not a kill
    state = state_of(project)
    assert state["status"] == "stopped" and state["phases"][0]["status"] == "interrupted"
    final = workers.status(project.root)
    assert final["state"] == "idle"
    assert final["last_exit"]["stopped"] is True and final["last_exit"]["code"] == 130
    assert final["last_exit"]["interrupted_phase"] is None  # nothing left for the server
    assert workers.stop(project.root)["state"] == "idle"  # nothing to stop is not an error


def test_a_worker_that_ignores_the_interrupt_is_killed_after_the_grace_period(
    project, make_workers
):
    workers = make_workers("ignore", grace=0.5)
    pid = started(workers, project)["pid"]
    before = time.monotonic()
    assert workers.stop(project.root)["state"] == "stopping"
    assert workers.wait(project.root, timeout=60)
    assert 0.4 <= time.monotonic() - before < 60
    assert not process_exists(pid)

    final = workers.status(project.root)
    assert final["last_exit"]["code"] == -signal.SIGKILL
    assert final["last_exit"]["interrupted_phase"] == "research"
    # It could not record anything itself, so the server did.
    state = state_of(project)
    assert state["status"] == "stopped" and state["worker"] is None and state["error"] is None
    assert state["phases"][0]["status"] == "interrupted" and state["phases"][0]["finished"]
    # The person asked for it: that is not an error in the activity stream.
    assert [event["type"] for event in project.events()] == ["project.created"]


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs /proc")
def test_a_worker_that_inherited_ignore_interrupt_is_asked_with_the_termination_signal(
    project, make_workers
):
    workers = make_workers("term-only", grace=120.0)
    started(workers, project)
    workers.stop(project.root)
    assert workers.wait(project.root, timeout=60)  # well inside the grace period
    assert (project.root / "terminated").exists()
    assert state_of(project)["status"] == "stopped"


def test_stop_also_ends_what_the_worker_started_in_other_sessions(project, make_workers):
    workers = make_workers("job", grace=0.5)
    started(workers, project)
    job = int((project.root / "job.pid").read_text())
    assert process_exists(job) and os.getsid(job) == job
    workers.stop(project.root)
    assert workers.wait(project.root, timeout=60)
    # The worker itself went at the interrupt; its job ignored it and was killed.
    assert (project.root / "interrupted").exists()
    wait_for(lambda: not process_exists(job), message="the job to be killed")


def test_what_the_worker_started_after_the_interrupt_is_killed_with_it(project, make_workers):
    workers = make_workers("late-job", grace=1.0)
    pid = started(workers, project)["pid"]
    workers.stop(project.root)
    wait_for(lambda: (project.root / "job.pid").exists(), message="the late job")
    job = int((project.root / "job.pid").read_text())
    assert workers.wait(project.root, timeout=60)
    assert not process_exists(pid)
    # It was not there when the stop began; it is looked for again before the kill.
    wait_for(lambda: not process_exists(job), message="the late job to be killed")


# ---------------------------------------------------------------------------
# A worker that dies
# ---------------------------------------------------------------------------


def test_a_worker_that_dies_is_recorded_as_interrupted_and_reported(project, make_workers):
    workers = make_workers("die")
    workers.start(project.root, "advance")
    assert workers.wait(project.root, timeout=60)

    status = workers.status(project.root)
    assert status["state"] == "idle"
    assert status["last_exit"] == {
        "kind": "advance",
        "code": 3,
        "at": status["last_exit"]["at"],
        "stopped": False,
        "interrupted_phase": "research",
    }
    state = state_of(project)
    assert state["status"] == "stopped" and state["worker"] is None
    assert state["phases"][0]["status"] == "interrupted"
    assert "exit code 3" in state["error"] and "worker.log" in state["error"]
    assert state["phases"][0]["error"] == state["error"]
    error = project.events()[-1]
    assert error["type"] == "error" and error["phase"] == "research"
    assert "ended with exit code 3" in error["summary"]
    assert "recorded as interrupted" in error["summary"]
    # Nothing stands in the way of resuming.
    workers.start(project.root, "advance", mode="quick")
    assert workers.wait(project.root, timeout=60)
    assert state_of(project)["status"] == "checkpoint"


def test_an_engine_that_fails_before_it_records_anything_is_reported(tmp_path, project):
    failing = tmp_path / "failing.py"
    failing.write_text("import sys\nprint('No credentials are configured.')\nsys.exit(1)\n")
    earlier = {"status": "checkpoint", "round": 1, "phase": "research", "phases": []}
    (project.root / "state.json").write_text(json.dumps(earlier))
    os.utime(project.root / "state.json", (time.time() - 60, time.time() - 60))
    workers = Workers(launcher=lambda kind, root, options: [sys.executable, str(failing)])
    try:
        workers.start(project.root, "advance")
        assert workers.wait(project.root, timeout=60)
        assert workers.status(project.root)["last_exit"]["code"] == 1
        assert state_of(project) == earlier  # nothing to settle: it never got that far
        error = project.events()[-1]
        assert error["type"] == "error" and "exit code 1" in error["summary"]
        assert "worker.log" in error["summary"]
        assert "No credentials are configured." in (project.root / WORKER_LOG).read_text()

        # A run or a write-up that ends the same way (the command refused, the project is
        # locked by another worker) left no account either.
        workers.start(project.root, "run", experiments=["E1"])
        assert workers.wait(project.root, timeout=60)
        error = project.events()[-1]
        assert (
            error["type"] == "error" and "The run worker ended with exit code 1" in error["summary"]
        )
    finally:
        workers.shutdown(grace=1.0)


def test_a_run_that_reported_its_own_failure_is_not_reported_twice(tmp_path, project):
    failing = tmp_path / "failing.py"
    failing.write_text(
        "import sys\n"
        "from paper_orchestra.research import Project\n"
        "Project.open(sys.argv[1]).emit('run.finished', 'E1 run 1 failed: exit code 2.')\n"
        "sys.exit(1)\n"
    )
    workers = Workers(
        launcher=lambda kind, root, options: [sys.executable, str(failing), str(root)]
    )
    try:
        for kind in ("run", "write"):
            workers.start(project.root, kind)
            assert workers.wait(project.root, timeout=60)
            assert workers.status(project.root)["last_exit"]["code"] == 1
        # The failed run is on record in its own event; the server adds nothing.
        assert [event["type"] for event in project.events()] == [
            "project.created",
            "run.finished",
            "run.finished",
        ]
    finally:
        workers.shutdown(grace=1.0)


def test_an_engine_that_recorded_its_own_failure_is_not_reported_twice(tmp_path, project):
    failing = tmp_path / "failing.py"
    failing.write_text(
        "import json, sys\n"
        "state = {'status': 'failed', 'error': 'The research phase failed twice.'}\n"
        "open(sys.argv[1] + '/state.json', 'w').write(json.dumps(state))\n"
        "sys.exit(1)\n"
    )
    workers = Workers(
        launcher=lambda kind, root, options: [sys.executable, str(failing), str(root)]
    )
    try:
        workers.start(project.root, "advance")
        assert workers.wait(project.root, timeout=60)
        assert [event["type"] for event in project.events()] == ["project.created"]
        assert state_of(project)["status"] == "failed"
    finally:
        workers.shutdown(grace=1.0)


def test_a_run_that_was_killed_does_not_leave_its_name_on_the_project(tmp_path, project):
    # `research run` names itself in state.json while the status stays what it was.  When
    # it is killed, the name stays behind with a heartbeat of a moment ago, and would read
    # as a worker that is alive but cannot be seen.
    killed = tmp_path / "killed.py"
    killed.write_text(
        "import json, os, socket, sys, time\n"
        "now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())\n"
        "worker = {'pid': os.getpid(), 'host': socket.gethostname(), 'started': now,\n"
        "          'heartbeat': now}\n"
        "state = {'status': 'checkpoint', 'round': 1, 'phase': 'design', 'phases': [],\n"
        "         'worker': worker}\n"
        "open(sys.argv[1] + '/state.json', 'w').write(json.dumps(state))\n"
        "os.kill(os.getpid(), 9)\n"
    )
    workers = Workers(launcher=lambda kind, root, options: [sys.executable, str(killed), str(root)])
    try:
        workers.start(project.root, "run", experiments=["E1"])
        assert workers.wait(project.root, timeout=60)
        status = workers.status(project.root)
        assert status["state"] == "idle" and status["last_exit"]["code"] == -signal.SIGKILL
        state = state_of(project)
        assert state["worker"] is None and state["status"] == "checkpoint"
        assert project.events()[-1]["type"] == "error"
        workers.start(project.root, "run", experiments=["E1"])  # nothing stands in the way
        assert workers.wait(project.root, timeout=60)
    finally:
        workers.shutdown(grace=1.0)


def test_the_name_of_a_worker_that_is_long_gone_is_taken_off_the_project(project, make_workers):
    stale = format_stamp(datetime.now(UTC) - timedelta(seconds=300))
    state = {
        "status": "checkpoint",
        "round": 1,
        "phase": "design",
        "phases": [],
        "worker": {"pid": dead_pid(), "host": socket.gethostname(), "heartbeat": stale},
    }
    (project.root / "state.json").write_text(json.dumps(state))
    workers = make_workers()
    assert workers.status(project.root)["state"] == "idle"
    after = state_of(project)
    assert after["worker"] is None and after["status"] == "checkpoint"
    assert workers.status(project.root)["last_exit"] is None  # no phase was interrupted


def test_mark_interrupted_only_touches_a_state_that_says_running(tmp_path):
    root = tmp_path
    assert mark_interrupted(root, "gone", by_person=False) is None  # no state.json
    state = {
        "status": "checkpoint",
        "phase": "research",
        "phases": [{"phase": "research", "status": "completed"}],
    }
    (root / "state.json").write_text(json.dumps(state))
    assert mark_interrupted(root, "gone", by_person=False) is None
    assert json.loads((root / "state.json").read_text()) == state

    state.update(status="running", worker={"pid": 1})
    state["phases"].append({"phase": "hypothesize", "status": "running", "finished": None})
    state["phase"] = "hypothesize"
    (root / "state.json").write_text(json.dumps(state))
    assert mark_interrupted(root, "The worker is gone.", by_person=False) == "hypothesize"
    after = json.loads((root / "state.json").read_text())
    assert after["status"] == "stopped" and after["worker"] is None
    assert after["error"] == "The worker is gone."
    assert [phase["status"] for phase in after["phases"]] == ["completed", "interrupted"]
    assert after["phases"][1]["error"] == "The worker is gone." and after["phases"][1]["finished"]


# ---------------------------------------------------------------------------
# A server that goes away, and the next one
# ---------------------------------------------------------------------------


def test_shutdown_stops_the_workers_and_refuses_new_ones(project, make_workers, tmp_path):
    other = Project.create(tmp_path / "ws" / "other", title="Other", brief="Another idea.")
    workers = make_workers()
    first = started(workers, project)["pid"]
    workers.start(other.root, "advance")
    wait_for(lambda: (other.root / "ready").exists())
    second = workers.status(other.root)["pid"]

    workers.shutdown(grace=5.0)
    assert not process_exists(first) and not process_exists(second)
    assert (project.root / "interrupted").exists() and (other.root / "interrupted").exists()
    assert workers.status(project.root)["state"] == "idle"
    with pytest.raises(WorkerError, match="shutting down"):
        workers.start(project.root, "advance")
    workers.reopen()
    workers.start(project.root, "advance", mode="quick")
    assert workers.wait(project.root, timeout=60)


def test_a_server_that_exits_does_not_wait_out_a_slow_stop(project, make_workers):
    workers = make_workers("ignore", grace=60.0)
    pid = started(workers, project)["pid"]
    assert workers.stop(project.root)["state"] == "stopping"  # would take a minute
    before = time.monotonic()
    workers.shutdown(grace=0.3)
    assert time.monotonic() - before < 45  # not the minute the first stop allowed
    assert not process_exists(pid)
    assert state_of(project)["status"] == "stopped"


def test_the_exit_hook_stops_workers_of_a_server_that_was_asked_to_and_did_not_shut_down(
    project, make_workers
):
    workers = make_workers(stop_on_exit=True)
    pid = started(workers, project)["pid"]
    workers_module._stop_all_at_exit()
    assert not process_exists(pid)
    assert (project.root / "interrupted").exists()


def test_the_end_of_a_server_leaves_its_workers_running(project, make_workers, tmp_path):
    # However the server ends, a worker belongs to its project: the next server finds it.
    other = Project.create(tmp_path / "ws" / "other", title="Other", brief="Another idea.")
    workers = make_workers()
    pid = started(workers, project, phase="research")["pid"]
    try:
        left = workers.close()
        assert [(worker["dir"], worker["pid"], worker["kind"]) for worker in left] == [
            (str(project.root), pid, "advance")
        ]
        assert left[0]["phase"] == "research"
        workers_module._stop_all_at_exit()  # the interpreter going down changes nothing
        with pytest.raises(WorkerError, match="shutting down"):
            workers.start(other.root, "advance")
        assert process_exists(pid) and not (project.root / "interrupted").exists()
        assert state_of(project)["status"] == "running"
        assert json.loads((project.root / RECORD_FILE).read_text())["pid"] == pid

        next_server = make_workers(grace=5.0)
        found = next_server.status(project.root)
        assert found["state"] == "running" and found["pid"] == pid and found["verified"] is True
        next_server.stop(project.root)
        assert next_server.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
    finally:
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)


def test_a_new_server_finds_the_worker_of_a_killed_one_and_can_stop_it(project, make_workers):
    old_server = make_workers()
    pid = started(old_server, project, phase="research")["pid"]

    new_server = make_workers(grace=5.0)  # knows nothing but the project's files
    found = new_server.status(project.root)
    assert found["state"] == "running" and found["adopted"] is True
    assert found["pid"] == pid and found["kind"] == "advance" and found["phase"] == "research"
    with pytest.raises(WorkerBusy):
        new_server.start(project.root, "advance")

    assert new_server.stop(project.root)["state"] == "stopping"
    assert new_server.wait(project.root, timeout=60)
    assert not process_exists(pid)
    assert (project.root / "interrupted").exists()
    assert new_server.status(project.root)["last_exit"]["stopped"] is True
    assert not (project.root / RECORD_FILE).exists()


def test_a_new_server_notices_that_the_worker_of_a_killed_one_died_too(project, make_workers):
    old_server = make_workers("ignore", grace=0.2)
    pid = started(old_server, project)["pid"]
    state = state_of(project)
    state["worker"] = {
        "pid": pid,
        "host": socket.gethostname(),
        "heartbeat": format_stamp(datetime.now(UTC)),  # it was alive a moment ago
    }
    (project.root / "state.json").write_text(json.dumps(state))
    record = (project.root / RECORD_FILE).read_text()
    old_server.stop(project.root)
    assert old_server.wait(project.root, timeout=60)
    # As the killed server left things: its record, and a state that says running.
    (project.root / RECORD_FILE).write_text(record)
    (project.root / "state.json").write_text(json.dumps(state))

    new_server = make_workers()
    status = new_server.status(project.root)
    assert status["state"] == "idle"  # known dead, however recent the heartbeat
    assert status["last_exit"]["interrupted_phase"] == "research"
    assert state_of(project)["status"] == "stopped"
    assert not (project.root / RECORD_FILE).exists()


def engine_from_a_terminal(
    script: Path,
    project: Project,
    *,
    named: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    heartbeat: datetime | None = None,
) -> subprocess.Popen:
    """An engine nobody's server started, as ``paper-orchestra research advance`` in a
    terminal would be: known only through ``state.json["worker"]``.  ``named`` is how
    the project was given on its command line (default: its directory)."""
    process = subprocess.Popen(
        worker_command(script, "advance", named or project.root),
        start_new_session=True,
        cwd=cwd,
        env={**os.environ, **(env or {})},
    )
    threading.Thread(target=process.wait, daemon=True).start()
    wait_for(lambda: (project.root / "ready").exists())
    state = state_of(project)
    state["worker"] = {
        "pid": process.pid,
        "host": socket.gethostname(),
        "started": format_stamp(datetime.now(UTC)),
        "heartbeat": format_stamp(heartbeat or datetime.now(UTC)),
    }
    (project.root / "state.json").write_text(json.dumps(state))
    return process


def test_an_engine_started_from_a_terminal_counts_as_the_projects_worker(
    script, project, make_workers
):
    process = engine_from_a_terminal(script, project)
    workers = make_workers(grace=5.0)
    status = workers.status(project.root)
    assert status["state"] == "running" and status["adopted"] is True
    assert status["pid"] == process.pid and status["phase"] == "research"
    with pytest.raises(WorkerBusy):
        workers.start(project.root, "advance")
    workers.stop(project.root)
    assert workers.wait(project.root, timeout=60)
    assert (project.root / "interrupted").exists()


def test_a_server_that_shuts_down_leaves_an_engine_started_from_a_terminal_running(
    script, project, make_workers
):
    # The person runs `research advance` in one terminal and the web app in another, to
    # watch it.  Closing the app must not end the work of the other terminal.
    process = engine_from_a_terminal(script, project)
    try:
        workers = make_workers(grace=1.0)
        assert workers.status(project.root)["state"] == "running"
        assert workers.known(project.root) and workers.live() == 0  # shown, not this server's
        workers.shutdown()
        workers_module._stop_all_at_exit()
        time.sleep(0.3)
        assert process.poll() is None and not (project.root / "interrupted").exists()
        assert state_of(project)["status"] == "running"
    finally:
        os.killpg(process.pid, signal.SIGKILL)


def test_a_worker_of_a_server_that_still_runs_is_not_stopped_by_another_servers_shutdown(
    project, make_workers
):
    # Two servers on one workspace (a second port): each stops what is its own.
    first = make_workers()
    pid = started(first, project)["pid"]
    # Both supervisors of this test live in one process; in the record, the server that
    # started the worker is another process that is alive.
    other_server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = json.loads((project.root / RECORD_FILE).read_text())
        assert record["server_pid"] == os.getpid()
        record.update(server_pid=other_server.pid, server_token=None)
        (project.root / RECORD_FILE).write_text(json.dumps(record))

        second = make_workers(grace=1.0)
        assert second.status(project.root)["adopted"] is True
        second.shutdown()
        time.sleep(0.3)
        assert process_exists(pid) and first.status(project.root)["state"] == "running"
        second.reopen()
        assert second.stop(project.root)["state"] == "stopping"  # the person may still stop it
        assert first.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
    finally:
        other_server.kill()
        other_server.wait()


PARENT_SCRIPT = r"""
import os, subprocess, sys
from pathlib import Path

script, root, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
# Like a shell script or a Makefile that runs the engine: same process group as the child.
child = subprocess.Popen([sys.executable, script, "research", "advance", str(root), mode])
(root / "child.pid").write_text(str(child.pid))
code = child.wait()
(root / "parent-went-on").write_text(str(code))
"""


@pytest.mark.parametrize("mode", ["sleep", "job", "helper"])
def test_stopping_an_engine_that_shares_its_process_group_spares_the_others_in_it(
    tmp_path, script, project, make_workers, mode
):
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(PARENT_SCRIPT, encoding="utf-8")
    parent = subprocess.Popen(
        [sys.executable, str(parent_script), str(script), str(project.root), mode],
        start_new_session=True,
    )
    try:
        wait_for(lambda: (project.root / "ready").exists())
        child = int((project.root / "child.pid").read_text())
        assert os.getpgid(child) == parent.pid != child  # the engine does not lead its group
        state = state_of(project)
        now = format_stamp(datetime.now(UTC))
        state["worker"] = {
            "pid": child,
            "host": socket.gethostname(),
            "started": now,
            "heartbeat": now,
        }
        (project.root / "state.json").write_text(json.dumps(state))

        workers = make_workers(grace=0.5)
        assert workers.status(project.root)["pid"] == child
        workers.stop(project.root)
        assert workers.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
        # The script that ran the engine saw it end and went on: it was neither
        # interrupted nor killed along with it.
        wait_for(lambda: (project.root / "parent-went-on").exists(), message="the parent")
        assert (project.root / "parent-went-on").read_text() == "130"
        assert parent.wait(timeout=10) == 0
        if mode != "sleep":
            # What the engine started is still ended: a job in a session of its own, and
            # a helper that stayed in the group the engine shares with the script.
            job = int((project.root / "job.pid").read_text())
            wait_for(lambda: not process_exists(job), message="the job to be killed")
    finally:
        with contextlib.suppress(OSError):
            os.killpg(parent.pid, signal.SIGKILL)
        parent.wait()


def test_an_experiment_run_from_a_terminal_counts_as_the_projects_worker(
    script, project, make_workers
):
    # `research run` holds the project as a worker without a phase of the engine running:
    # state.json names it under "worker" while the status stays what it was.
    process = engine_from_a_terminal(script, project)
    try:
        state = state_of(project)
        state.update(status="checkpoint")
        state["phases"][0]["status"] = "completed"
        (project.root / "state.json").write_text(json.dumps(state))

        workers = make_workers()
        status = workers.status(project.root)
        assert status["state"] == "running" and status["kind"] == "run"
        assert status["pid"] == process.pid and status["adopted"] is True
        with pytest.raises(WorkerBusy):
            workers.start(project.root, "advance")

        # A write-up started from a terminal says what it is, and is shown as that.
        state["worker"]["kind"] = "write"
        (project.root / "state.json").write_text(json.dumps(state))
        assert make_workers().status(project.root)["kind"] == "write"
        state["worker"]["kind"] = "something else"  # not a kind of work: judged as before
        (project.root / "state.json").write_text(json.dumps(state))
        assert make_workers().status(project.root)["kind"] == "run"

        # It finished and took its name out of state.json: the project is free at once,
        # even while that process is still around.
        state["worker"] = None
        (project.root / "state.json").write_text(json.dumps(state))
        final = workers.status(project.root)
        assert final["state"] == "idle" and final["last_exit"]["interrupted_phase"] is None
        assert state_of(project)["status"] == "checkpoint"  # nothing to settle
        assert [event["type"] for event in project.events()] == ["project.created"]
    finally:
        os.killpg(process.pid, signal.SIGKILL)


def test_a_worker_that_ends_leaves_the_state_of_another_that_is_alive_alone(
    tmp_path, script, project
):
    # A write-up runs (it does not hold the project against the engine), the person starts
    # `research advance` in a terminal, and then the write-up ends badly.  The phase that
    # reads as running is the engine's, and the engine is alive.
    writing = tmp_path / "writing.py"
    writing.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "while not (Path(sys.argv[1]) / 'go').exists():\n"
        "    time.sleep(0.02)\n"
        "sys.exit(1)\n"
    )
    workers = Workers(
        launcher=lambda kind, root, options: [sys.executable, str(writing), str(root)], grace=1.0
    )
    engine = None
    try:
        workers.start(project.root, "write", template="cvpr2025")
        engine = engine_from_a_terminal(script, project)
        before = state_of(project)
        (project.root / "go").write_text("")
        wait_for(lambda: workers.status(project.root)["kind"] != "write", message="its end")

        assert state_of(project) == before  # still running, still the engine's
        status = workers.status(project.root)
        assert status["state"] == "running" and status["pid"] == engine.pid
        assert status["last_exit"]["kind"] == "write" and status["last_exit"]["code"] == 1
        assert status["last_exit"]["interrupted_phase"] is None
        error = project.events()[-1]  # the write-up's own end is still reported
        assert error["type"] == "error" and "The write worker ended" in error["summary"]
        assert "interrupted" not in error["summary"]
    finally:
        workers.shutdown(grace=1.0)
        if engine is not None:
            os.killpg(engine.pid, signal.SIGKILL)


def test_the_end_of_a_worker_is_settled_once_however_many_notice_it(
    script, project, make_workers, monkeypatch
):
    process = engine_from_a_terminal(script, project)
    workers = make_workers()
    assert workers.status(project.root)["state"] == "running"
    entry = workers._entries[workers._key(project.root)]
    os.killpg(process.pid, signal.SIGKILL)
    wait_for(lambda: not process_exists(process.pid))

    settled = []
    settle = workers._settle

    def slow_settle(*args):
        settled.append(args)
        time.sleep(0.3)  # the second to notice arrives while the first is still at it
        return settle(*args)

    monkeypatch.setattr(workers, "_settle", slow_settle)
    noticing = [threading.Thread(target=workers._finished, args=(entry, None)) for _ in range(2)]
    for thread in noticing:
        thread.start()
        time.sleep(0.1)
    for thread in noticing:
        thread.join()
    assert len(settled) == 1
    assert workers.status(project.root)["last_exit"]["interrupted_phase"] == "research"
    assert [event["type"] for event in project.events()] == ["project.created", "error"]


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def write_running_state(project: Project, worker: dict) -> None:
    state = {
        "status": "running",
        "round": 1,
        "phase": "design",
        "phases": [{"round": 1, "phase": "design", "status": "running"}],
        "worker": worker,
    }
    (project.root / "state.json").write_text(json.dumps(state))


def test_a_worker_with_a_stale_heartbeat_and_no_process_is_dead(project, make_workers):
    stale = format_stamp(datetime.now(UTC) - timedelta(seconds=120))
    write_running_state(
        project, {"pid": dead_pid(), "host": socket.gethostname(), "heartbeat": stale}
    )
    workers = make_workers()
    status = workers.status(project.root)
    assert status["state"] == "idle"
    assert status["last_exit"]["interrupted_phase"] == "design"
    state = state_of(project)
    assert state["status"] == "stopped" and state["phases"][0]["status"] == "interrupted"
    assert "worker of this phase is gone" in state["error"]
    workers.start(project.root, "advance", mode="quick")  # and a new one may start
    assert workers.wait(project.root, timeout=60)


def test_a_fresh_heartbeat_keeps_a_worker_alive_that_cannot_be_seen(project, make_workers):
    # Another machine, or a container with its own process numbers: only the heartbeat tells.
    fresh = format_stamp(datetime.now(UTC))
    write_running_state(project, {"pid": dead_pid(), "host": "another-host", "heartbeat": fresh})
    workers = make_workers()
    status = workers.status(project.root)
    assert status["state"] == "running" and status["pid"] is None
    assert status["host"] == "another-host"
    with pytest.raises(WorkerBusy):
        workers.start(project.root, "advance")
    with pytest.raises(WorkerError, match="another machine"):
        workers.stop(project.root)
    assert state_of(project)["status"] == "running"  # left alone


def test_a_stale_heartbeat_with_somebody_elses_process_under_that_pid_is_dead(
    project, make_workers
):
    # After a reboot the pid in state.json belongs to whatever got it next.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stale = format_stamp(datetime.now(UTC) - timedelta(seconds=300))
        write_running_state(
            project, {"pid": unrelated.pid, "host": socket.gethostname(), "heartbeat": stale}
        )
        workers = make_workers()
        assert workers.status(project.root)["state"] == "idle"
        assert unrelated.poll() is None  # and it was not touched
    finally:
        unrelated.kill()
        unrelated.wait()


# ---------------------------------------------------------------------------
# The command line of a worker
# ---------------------------------------------------------------------------


def test_default_command_is_the_packages_own_cli(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    base = [sys.executable, "-P", "-m", "paper_orchestra", "research"]
    assert default_command("advance", root, {}) == [*base, "advance", str(root)]
    assert default_command(
        "advance", root, {"phase": "design", "instruction": "--phase=run is not an option here"}
    ) == [
        *base,
        "advance",
        str(root),
        "--phase=design",
        "--instruction=--phase=run is not an option here",
    ]
    assert default_command("run", root, {"experiments": ["E2", "E1"]}) == [
        *base,
        "run",
        str(root),
        "E2",
        "E1",
    ]
    assert default_command("write", root, {"template": "cvpr2025", "model": "m"}) == [
        *base,
        "write",
        str(root),
        "--template=cvpr2025",
        "--model=m",
    ]
    with pytest.raises(WorkerError):
        default_command("dance", root, {})
    # The project is given as an absolute path: it says which directory is meant wherever
    # the worker runs, and is never read as an option, whatever it is called.
    monkeypatch.chdir(tmp_path)
    assert default_command("advance", Path("research/demo"), {})[-1] == str(
        tmp_path / "research" / "demo"
    )
    assert default_command("advance", Path("-h"), {})[-1] == str(tmp_path / "-h")


# ---------------------------------------------------------------------------
# What the files of a project directory can make the server do: nothing
# ---------------------------------------------------------------------------
#
# A project directory may come from somebody else (an archive, a clone, a shared
# folder), and a process number in an old file may be anybody's by now.  The server
# signals a process only when it started it or can show it to be this project's worker.


@pytest.fixture
def victim():
    """A process of the person's that has nothing to do with any project, and a child of
    it in a session of its own.  It must be alive and untouched after every test."""
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],"
        " start_new_session=True)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True, text=True
    )
    child = int(process.stdout.readline())
    process.child = child
    yield process
    alive = process.poll() is None and process_exists(child)
    for pid in (process.pid, child):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    process.wait()
    assert alive, "the unrelated process (or its child) was signalled"


@pytest.fixture
def signals(monkeypatch):
    """Record every signal the code under test would send, and send none of them: for
    the tests whose target must not be signalled even by a mistake of the test."""
    sent: list[tuple[str, int, int]] = []
    real_kill = os.kill

    def kill(pid, signum):
        if signum == 0:
            return real_kill(pid, 0)
        sent.append(("kill", pid, signum))

    def killpg(group, signum):
        sent.append(("killpg", group, signum))

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)
    return sent


def start_token_of(pid: int) -> str | None:
    return workers_module._start_token(pid)


def plant_record(project: Project, **fields) -> None:
    (project.root / RECORD_FILE).write_text(json.dumps({"host": HOST, **fields}))


def plant_state(project: Project, worker, status: str = "running") -> None:
    state = {
        "status": status,
        "round": 1,
        "phase": "design",
        "phases": [{"round": 1, "phase": "design", "status": "running"}],
        "worker": worker,
    }
    (project.root / "state.json").write_text(json.dumps(state))


def stamp(seconds_from_now: float) -> str:
    return format_stamp(datetime.now(UTC) + timedelta(seconds=seconds_from_now))


def nothing_was_signalled(workers: Workers, project: Project) -> None:
    """Everything a person or the end of the server could do to a worker, done."""
    with contextlib.suppress(WorkerError):
        assert workers.stop(project.root)["state"] != "stopping"
    assert workers.live() == 0
    workers.shutdown(grace=0.2)
    workers_module._stop_all_at_exit()


@pytest.mark.parametrize("with_token", [False, True])
def test_a_record_that_names_somebody_elses_process_is_not_this_projects_worker(
    project, make_workers, victim, with_token
):
    # The start token of any process can be read by anyone, so a planted record may
    # carry the right one: what it cannot have is a process that works on this project.
    plant_record(
        project,
        pid=victim.pid,
        kind="advance",
        start_token=start_token_of(victim.pid) if with_token else None,
    )
    workers = make_workers(grace=0.2, stop_on_exit=True)
    status = workers.status(project.root)
    assert status["state"] == "idle" and status["pid"] is None
    assert not (project.root / RECORD_FILE).exists()  # it named nothing of this project's
    nothing_was_signalled(workers, project)
    workers.reopen()
    workers.start(project.root, "advance", mode="quick")  # and nothing stands in the way
    assert workers.wait(project.root, timeout=60)


@pytest.mark.parametrize("heartbeat", ["2099-01-01T00:00:00Z", None, "fresh"])
@pytest.mark.parametrize("host", [None, HOST])
def test_a_state_that_names_somebody_elses_process_is_not_this_projects_worker(
    project, make_workers, victim, heartbeat, host
):
    worker = {"pid": victim.pid}
    if heartbeat is not None:
        worker["heartbeat"] = stamp(0) if heartbeat == "fresh" else heartbeat
    if host is not None:
        worker["host"] = host
    plant_state(project, worker)
    workers = make_workers(grace=0.2, stop_on_exit=True)
    status = workers.status(project.root)
    assert status["state"] == "idle" and status["pid"] is None
    state = state_of(project)
    assert state["status"] == "stopped" and state["worker"] is None
    nothing_was_signalled(workers, project)
    workers.reopen()
    workers.start(project.root, "advance", mode="quick")
    assert workers.wait(project.root, timeout=60)


def test_a_process_whose_command_line_merely_holds_the_directory_name_is_not_a_worker(
    project, make_workers
):
    # After a reboot the pid of an honest, stale state.json is somebody else's; that
    # this process was started with a file called like the project proves nothing.
    lookalike = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(300)",
            "--open",
            f"/home/someone/notes/{project.root.name}.txt",
            str(project.root),
        ],
        start_new_session=True,
    )
    try:
        plant_state(project, {"pid": lookalike.pid, "host": HOST, "heartbeat": stamp(-3600)})
        workers = make_workers(grace=0.2, stop_on_exit=True)
        assert workers.status(project.root)["state"] == "idle"
        nothing_was_signalled(workers, project)
        assert lookalike.poll() is None
    finally:
        lookalike.kill()
        lookalike.wait()


@pytest.mark.parametrize("named", [1, 0, -1, -7, True, "self", "parent", 2**22, 10**30, 1.5, "12"])
def test_a_pid_that_is_no_workers_is_never_adopted_or_signalled(
    project, make_workers, signals, named
):
    pid = {"self": os.getpid(), "parent": os.getppid()}.get(named, named)
    for plant in (
        lambda: plant_record(project, pid=pid, start_token=None),
        lambda: plant_record(project, pid=pid, start_token="1", kind="advance"),
        lambda: plant_state(project, {"pid": pid, "heartbeat": "2099-01-01T00:00:00Z"}),
        lambda: plant_state(project, {"pid": pid, "host": HOST, "heartbeat": stamp(0)}),
        lambda: plant_state(project, {"pid": pid, "host": HOST, "heartbeat": stamp(-900)}),
    ):
        plant()
        workers = make_workers(grace=0.1, stop_on_exit=True)
        status = workers.status(project.root)  # does not raise, whatever the number
        assert status["state"] == "idle" and status["pid"] is None, (named, status)
        nothing_was_signalled(workers, project)
        (project.root / "state.json").unlink(missing_ok=True)
    assert signals == []


def test_what_a_process_started_is_not_looked_up_for_the_server_or_its_ancestors(signals):
    # "Everything the worker started" of pid 1 is every process of the machine, and of
    # an ancestor of the server it includes the server.
    # Something this very process started, in a session of its own, as a worker's job is:
    # what would be found by a walk from the server or from one of its ancestors.
    # It ends when its input closes: no signal can be sent in this test (see ``signals``).
    started = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        for pid in (1, 0, -1, True, os.getpid(), os.getppid(), 10**30):
            targets = workers_module._targets_of(pid)
            assert targets.groups == {} and targets.processes == {}, pid
            targets.signal(signal.SIGKILL)
        assert signals == []
        assert {os.getpid(), os.getppid()} <= workers_module._own_lineage()
    finally:
        started.stdin.close()
        started.wait(timeout=60)


HOSTILE = [
    None, True, False, 0, 1, -1, 2**22, 10**30, 1.5, float("1e300"), "", "x", "12", [], [1], {},
    {"pid": 1}, "2099-01-01T00:00:00Z",
]  # fmt: skip


def test_no_value_in_a_projects_files_makes_a_look_at_its_worker_fail(project, make_workers):
    # One project's file must not break the listing of all of them.
    workers = make_workers()
    keys = ("pid", "host", "heartbeat", "started", "kind")
    for value in HOSTILE:
        for key in keys:
            worker = {"pid": dead_pid(), "host": HOST, "heartbeat": stamp(0), key: value}
            plant_state(project, worker)
            assert workers.status(project.root)["state"] in ("idle", "running"), (key, value)
        plant_state(project, value)
        assert workers.status(project.root)["state"] == "idle", value
        (project.root / "state.json").write_text(json.dumps(value))
        assert workers.status(project.root)["state"] == "idle", value
        (project.root / "state.json").unlink()
        record_keys = ("pid", "host", "start_token", "kind", "options", "command", "started",
                       "server_pid", "server_token")  # fmt: skip
        for key in record_keys:
            record = {"pid": os.getpid(), "host": HOST, "start_token": "1", key: value}
            (project.root / RECORD_FILE).write_text(json.dumps(record))
            assert workers.status(project.root)["state"] in ("idle", "running"), (key, value)
        (project.root / RECORD_FILE).write_text(json.dumps(value))
        assert workers.status(project.root)["state"] == "idle", value
    (project.root / RECORD_FILE).write_text("{ not json")
    assert workers.status(project.root)["state"] == "idle"
    assert not (project.root / RECORD_FILE).exists()  # nothing a server wrote


def test_no_value_in_the_record_of_a_live_worker_makes_a_look_at_it_fail(project, make_workers):
    # The record of a worker that is there, with one field after the other replaced.
    pid = started(make_workers(), project)["pid"]
    record = json.loads((project.root / RECORD_FILE).read_text())
    looking = make_workers()
    for key in ("kind", "options", "command", "started", "server_pid", "server_token", "extra"):
        for value in HOSTILE:
            (project.root / RECORD_FILE).write_text(json.dumps({**record, key: value}))
            looking._entries.clear()
            status = looking.status(project.root)
            assert status["state"] == "running" and status["pid"] == pid, (key, value)
            assert status["verified"] is True and looking.live() in (0, 1)
    # What says which process it is cannot be replaced: then it is not that worker's record.
    for key in ("pid", "host", "start_token"):
        for value in HOSTILE:
            (project.root / RECORD_FILE).write_text(json.dumps({**record, key: value}))
            looking._entries.clear()
            status = looking.status(project.root)
            assert status["pid"] in (None, pid) and status["state"] in ("idle", "running")
            (project.root / RECORD_FILE).write_text(json.dumps(record))
    assert process_exists(pid)


@pytest.mark.parametrize("name", ["state.json", RECORD_FILE])
def test_a_file_nested_too_deep_to_read_is_no_file_of_a_worker(project, make_workers, name):
    # The API reads state.json through read_state for every row of the library: text that
    # the JSON reader gives up on (not with a ValueError) must not become a 500 for all.
    deep = "[" * 200_000 + "]" * 200_000
    (project.root / name).write_text('{"status": "running", "worker": ' + deep + "}")
    assert workers_module.read_state(project.root) == {}
    assert make_workers().status(project.root)["state"] == "idle"
    (project.root / name).write_text(deep)
    assert workers_module.read_state(project.root) == {}
    assert make_workers().status(project.root)["state"] == "idle"


def test_a_file_far_larger_than_any_state_is_not_taken_into_memory(project, monkeypatch):
    # The library reads state.json of every project at every look.
    with (project.root / "state.json").open("wb") as file:
        file.truncate(3 * 1024**3)  # sparse: it takes no room on disk
    taken = 0
    real_read = os.read

    def read(descriptor, count):
        nonlocal taken
        piece = real_read(descriptor, count)
        taken += len(piece)
        return piece

    monkeypatch.setattr(os, "read", read)
    assert workers_module.read_state(project.root) == {}
    assert taken == 0


def test_a_state_file_that_is_a_link_is_not_read(project, make_workers, tmp_path):
    # What state.json holds is shown in the app; a project that came from somebody else
    # must not be able to show a file of the person's that way.
    outside = tmp_path / "somebodys.json"
    outside.write_text(json.dumps({"status": "running", "secret": "OUTSIDE", "worker": None}))
    (project.root / "state.json").symlink_to(outside)
    assert workers_module.read_state(project.root) == {}
    workers = make_workers()
    assert workers.status(project.root)["state"] == "idle"
    assert json.loads(outside.read_text())["status"] == "running"  # and it is not written


# ---------------------------------------------------------------------------
# A worker this server can show to be the project's own
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="needs /proc")
def test_an_engine_started_with_a_relative_path_is_this_projects_worker_however_old_its_heartbeat(
    script, project, make_workers
):
    # `cd project; paper-orchestra research advance .`, then the laptop slept: the
    # heartbeat is old, the process is there and works on this directory.
    process = engine_from_a_terminal(
        script, project, named=".", cwd=project.root, heartbeat=datetime.now(UTC) - timedelta(1)
    )
    try:
        workers = make_workers(grace=5.0)
        status = workers.status(project.root)
        assert status["state"] == "running" and status["pid"] == process.pid
        assert status["verified"] is True and status["note"] is None
        assert state_of(project)["status"] == "running"  # not recorded as interrupted
        workers.stop(project.root)
        assert workers.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
    finally:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_an_engine_that_holds_the_projects_worker_lock_is_its_worker_however_it_was_named(
    script, project, make_workers
):
    # `paper-orchestra research advance p-0123`: the command line names no directory,
    # but the lock of this directory is held and this process has it open.
    process = engine_from_a_terminal(
        script,
        project,
        named="p-0123456789ab",
        env={"STAND_IN_ROOT": str(project.root), "STAND_IN_LOCK": "1"},
    )
    try:
        assert workers_module._lock_held(project.root) is True
        assert workers_module._has_open(process.pid, project.root / LOCK_FILE)
        assert not workers_module._has_open(os.getpid(), project.root / LOCK_FILE)
        workers = make_workers(grace=5.0)
        status = workers.status(project.root)
        assert status["state"] == "running" and status["pid"] == process.pid
        assert status["verified"] is True
        workers.stop(project.root)
        assert workers.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
        assert workers_module._lock_held(project.root) is False
    finally:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)


def test_a_worker_that_is_here_is_this_machines_whatever_name_the_file_gives_it(
    script, project, make_workers
):
    # A laptop on another network has another host name; the engine that named itself
    # under the old one is still a process of this machine, and can be stopped.
    process = engine_from_a_terminal(script, project)
    try:
        state = state_of(project)
        state["worker"]["host"] = "the-name-it-had-this-morning.local"
        (project.root / "state.json").write_text(json.dumps(state))
        workers = make_workers(grace=5.0)
        status = workers.status(project.root)
        assert status["state"] == "running" and status["pid"] == process.pid
        assert status["verified"] is True and status["host"] == HOST
        workers.stop(project.root)
        assert workers.wait(project.root, timeout=60)
        assert (project.root / "interrupted").exists()
    finally:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)


def test_the_command_line_of_a_worker_is_told_from_one_that_resembles_it(tmp_path):
    root = tmp_path / "ws" / "demo"
    root.mkdir(parents=True)
    (tmp_path / "link").symlink_to(root)
    is_worker = workers_module._worker_arguments
    names = workers_module._names_directory

    def shown(command: list[str], cwd: str | None = None) -> bool:
        arguments = is_worker(command)
        return arguments is not None and names(arguments, cwd, root)

    python = sys.executable
    assert shown([python, "-P", "-m", "paper_orchestra", "research", "advance", str(root)])
    assert shown([python, "-m", "paper_orchestra", "research", "run", str(root), "E1"])
    assert shown([python, "/venv/bin/paper-orchestra", "research", "write", str(root)])
    assert shown(["/venv/bin/paper-orchestra", "research", "advance", "--phase=design", str(root)])
    assert shown([python, "-m", "paper_orchestra", "research", "advance", "demo"], str(root.parent))
    assert shown([python, "-m", "paper_orchestra", "research", "advance", "."], str(root))
    assert shown([python, "-m", "paper_orchestra", "research", "advance", str(tmp_path / "link")])
    # Another project, another command, another program, or the name without the place.
    assert not shown([python, "-m", "paper_orchestra", "research", "advance", str(root.parent)])
    assert not shown([python, "-m", "paper_orchestra", "research", "advance", "demo"], "/tmp")
    assert not shown([python, "-m", "paper_orchestra", "research", "advance", "demo"], None)
    assert not shown([python, "-m", "paper_orchestra", "research", "status", str(root)])
    assert not shown([python, "-m", "paper_orchestra", "run", str(root)])
    assert not shown([python, "-m", "other_package", "research", "advance", str(root)])
    assert not shown([python, "victim.py", "--open", str(root)])
    assert not shown(["vim", str(root / "research" / "advance")])
    assert not shown(["research", "advance", str(root)])
    assert not shown([python, "-m", "paper_orchestra", "research", "advance", f"--x={root}"])
    assert not shown([])


# ---------------------------------------------------------------------------
# A worker this server cannot verify: shown, never signalled, and forgettable
# ---------------------------------------------------------------------------


@pytest.fixture
def lock_holder(project):
    """Some process that holds the project's worker lock without being recognisable as
    its worker: the kernel says that work is going on here, not whose."""
    code = (
        "import fcntl, os, sys, time\n"
        "lock = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o644)\n"
        "fcntl.flock(lock, fcntl.LOCK_EX)\n"
        "print('held', flush=True)\n"
        "time.sleep(300)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(project.root / LOCK_FILE)],
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    assert process.stdout.readline().strip() == "held"
    yield process
    alive = process.poll() is None
    process.kill()
    process.wait()
    assert alive, "the process that held the lock was signalled"


def test_a_worker_that_cannot_be_verified_is_shown_without_a_stop_and_can_be_forgotten(
    project, make_workers, lock_holder
):
    plant_record(project, pid=lock_holder.pid, start_token=start_token_of(lock_holder.pid))
    plant_state(project, {"pid": lock_holder.pid, "host": HOST, "heartbeat": stamp(0)})
    workers = make_workers(grace=0.2, stop_on_exit=True)
    status = workers.status(project.root)
    assert status["state"] == "running" and status["verified"] is False
    assert status["pid"] is None  # not a process this server vouches for
    assert "cannot show" in status["note"] and "forget" in status["note"]
    assert str(lock_holder.pid) in status["note"]
    assert workers.glance(project.root) == "running" and workers.live() == 0
    with pytest.raises(WorkerBusy) as busy:
        workers.start(project.root, "advance")
    # The refusal names what can be done here: such a worker has no Stop.
    assert "forget" in str(busy.value) and "stop it first" not in str(busy.value)
    assert busy.value.status["verified"] is False
    with pytest.raises(WorkerError, match="does not signal") as refused:
        workers.stop(project.root)
    assert "forget" in str(refused.value)
    assert workers.running() == []  # nothing of this server's to speak for
    workers.shutdown(grace=0.2)
    workers_module._stop_all_at_exit()
    workers.reopen()

    forgotten = workers.forget(project.root)
    assert forgotten["state"] == "idle" and forgotten["verified"] is None
    assert forgotten["last_exit"]["forgotten"] is True
    assert forgotten["last_exit"]["interrupted_phase"] == "design"
    state = state_of(project)
    assert state["status"] == "stopped" and state["worker"] is None and state["error"] is None
    assert state["phases"][0]["status"] == "interrupted"
    assert not (project.root / RECORD_FILE).exists()
    assert workers.status(project.root)["state"] == "idle"
    assert workers.forget(project.root)["state"] == "idle"  # nothing to forget is no error
    wait_for(lambda: project.root in make_workers.settled, message="its runs to be looked at")


def test_a_worker_on_another_machine_can_be_forgotten_when_its_heartbeat_lies(
    project, make_workers
):
    # A clock that runs ahead, or a file that says what it likes: the project could be
    # neither started nor stopped for as long as the heartbeat lay in the future.
    plant_state(project, {"pid": 4242, "host": "other-machine", "heartbeat": stamp(30)})
    workers = make_workers()
    status = workers.status(project.root)
    assert status["state"] == "running" and status["verified"] is False
    assert status["host"] == "other-machine" and "other-machine" in status["note"]
    with pytest.raises(WorkerError, match="another machine"):
        workers.stop(project.root)
    # The record of that machine's server is not this server's to remove.
    (project.root / RECORD_FILE).write_text(json.dumps({"pid": 4242, "host": "other-machine"}))
    assert workers.forget(project.root)["state"] == "idle"
    assert (project.root / RECORD_FILE).exists()
    assert state_of(project)["worker"] is None
    workers.start(project.root, "advance", mode="quick")
    assert workers.wait(project.root, timeout=60)

    plant_state(project, {"pid": 4242, "host": "other-machine", "heartbeat": stamp(3600)})
    assert workers.status(project.root)["state"] == "idle"  # an hour ahead is no sign of life
    assert state_of(project)["status"] == "stopped"


def test_a_worker_this_server_started_or_verified_is_stopped_not_forgotten(project, make_workers):
    workers = make_workers(grace=5.0)
    pid = started(workers, project)["pid"]
    with pytest.raises(WorkerError, match="stop it instead"):
        workers.forget(project.root)
    adopting = make_workers(grace=5.0)
    assert adopting.status(project.root)["verified"] is True
    with pytest.raises(WorkerError, match="stop it instead"):
        adopting.forget(project.root)
    assert process_exists(pid) and state_of(project)["status"] == "running"


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


def test_a_heartbeat_from_the_future_is_not_a_sign_of_life():
    fresh = workers_module._heartbeat_fresh
    here = workers_module.HEARTBEAT_AHEAD_SECONDS
    elsewhere = workers_module.HEARTBEAT_AHEAD_SECONDS_ELSEWHERE
    assert fresh({"heartbeat": stamp(-30)}, ahead=here)
    assert fresh({"heartbeat": stamp(2)}, ahead=here)  # written within the same second or so
    assert not fresh({"heartbeat": stamp(-120)}, ahead=here)
    assert not fresh({"heartbeat": stamp(30)}, ahead=here)
    assert fresh({"heartbeat": stamp(30)}, ahead=elsewhere)  # another machine's clock
    assert not fresh({"heartbeat": stamp(7200)}, ahead=elsewhere)
    assert not fresh({"heartbeat": "2099-01-01T00:00:00Z"}, ahead=elsewhere)
    for nothing in ({}, {"heartbeat": None}, {"heartbeat": 12}, {"heartbeat": "soon"}):
        assert not fresh(nothing, ahead=elsewhere)


@pytest.mark.parametrize("lock_known", [True, False])
def test_after_the_clock_was_set_back_a_killed_worker_is_dead(
    project, make_workers, monkeypatch, lock_known
):
    # The worker was killed; its last heartbeat now lies two hours in the future.  The
    # project could be neither continued nor stopped until the clock caught up.
    if not lock_known:
        monkeypatch.setattr(workers_module, "_lock_held", lambda root: None)
    plant_state(project, {"pid": dead_pid(), "host": HOST, "heartbeat": stamp(7200)})
    workers = make_workers()
    status = workers.status(project.root)
    assert status["state"] == "idle" and status["last_exit"]["interrupted_phase"] == "design"
    state = state_of(project)
    assert state["status"] == "stopped" and state["phases"][0]["status"] == "interrupted"
    assert workers.stop(project.root)["state"] == "idle"
    workers.start(project.root, "advance", mode="quick")
    assert workers.wait(project.root, timeout=60)


def test_on_this_machine_a_worker_whose_process_is_gone_is_dead_however_fresh_its_heartbeat(
    project, make_workers, monkeypatch
):
    # For its first minute a killed `research run` read as a worker without a pid that
    # "runs on another machine": it could not be stopped and nothing could be started.
    monkeypatch.setattr(workers_module, "_lock_held", lambda root: None)
    plant_state(
        project, {"pid": dead_pid(), "host": HOST, "heartbeat": stamp(0)}, status="checkpoint"
    )
    workers = make_workers()
    assert workers.status(project.root)["state"] == "idle"
    assert state_of(project)["worker"] is None
    wait_for(lambda: project.root in make_workers.settled, message="its runs to be settled")


# ---------------------------------------------------------------------------
# A copy is not the original
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("locks", [False, True])
def test_a_copy_of_a_project_does_not_show_or_stop_the_originals_worker(
    script, project, make_workers, tmp_path, monkeypatch, locks
):
    if locks:
        monkeypatch.setenv("STAND_IN_LOCK", "1")  # like the engine: the worker lock is held
    first = make_workers()
    pid = started(first, project, phase="research")["pid"]
    state = state_of(project)
    state["worker"] = {"pid": pid, "host": HOST, "started": stamp(0), "heartbeat": stamp(0)}
    (project.root / "state.json").write_text(json.dumps(state))
    copy = Project.open(shutil.copytree(project.root, tmp_path / "ws" / "a-branch"))
    assert json.loads((copy.root / RECORD_FILE).read_text())["pid"] == pid
    (copy.root / "ready").unlink()

    server = make_workers(grace=0.3, stop_on_exit=True)
    status = server.status(copy.root)
    assert status["state"] == "idle" and status["pid"] is None  # a copy reads as interrupted
    assert status["last_exit"]["interrupted_phase"] == "research"
    assert state_of(copy)["status"] == "stopped" and state_of(copy)["worker"] is None
    assert not (copy.root / RECORD_FILE).exists()
    assert server.stop(copy.root)["state"] == "idle"
    server.shutdown(grace=0.3)
    server.reopen()
    # The original is as it was: running, on record, and this server's to show.
    assert process_exists(pid) and not (project.root / "interrupted").exists()
    assert state_of(project)["status"] == "running"
    assert json.loads((project.root / RECORD_FILE).read_text())["pid"] == pid
    assert server.status(project.root)["pid"] == pid
    # The copy is a project of its own and can be worked on.
    monkeypatch.delenv("STAND_IN_LOCK", raising=False)
    server.start(copy.root, "advance", mode="quick")
    assert server.wait(copy.root, timeout=60)
    assert process_exists(pid)


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_a_copy_made_of_hard_links_shares_the_lock_file_and_still_is_not_the_original(
    script, project, make_workers, tmp_path
):
    # `cp -al`, or a backup tool that links unchanged files: the copy's lock file is the
    # original's file, held by the original's worker (which was given its project by id).
    process = engine_from_a_terminal(
        script,
        project,
        named="p-0123456789ab",
        env={"STAND_IN_ROOT": str(project.root), "STAND_IN_LOCK": "1"},
    )
    try:
        copy = tmp_path / "ws" / "linked-copy"
        shutil.copytree(project.root, copy, copy_function=os.link)
        assert os.path.samefile(copy / LOCK_FILE, project.root / LOCK_FILE)
        workers = make_workers(grace=0.3, stop_on_exit=True)
        status = workers.status(copy)
        # Work is going on under that lock, but not shown to be this directory's.
        assert status["state"] == "running" and status["verified"] is False
        assert status["pid"] is None
        with pytest.raises(WorkerError, match="does not signal"):
            workers.stop(copy)
        workers.shutdown(grace=0.3)
        assert process.poll() is None and not (project.root / "interrupted").exists()
        assert workers.status(project.root)["pid"] == process.pid  # the original's own
    finally:
        os.killpg(process.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Workers import the installed package; the log is the project's own file
# ---------------------------------------------------------------------------


def test_a_worker_does_not_import_files_of_the_directory_the_server_runs_in(tmp_path, project):
    # The default workspace is ./research, so the server usually runs in the person's
    # own checkout.  A file there named like a module (secrets.py is a common name for
    # one's settings) was imported by every worker, next to the credentials.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    proof = tmp_path / "proof.txt"
    shadow = f"open({str(proof)!r}, 'a').write('ran')\nraise RuntimeError('shadowed')\n"
    for name in ("secrets.py", "types.py", "typer.py", "json.py", "paper_orchestra.py"):
        (checkout / name).write_text(shadow)
    workers = Workers()
    try:
        command = default_command("advance", tmp_path / "ws" / "no-such-project", {})
        done = subprocess.run(
            command, cwd=checkout, env=workers._environment(), capture_output=True, text=True,
            timeout=300,
        )  # fmt: skip
        assert not proof.exists(), done.stderr[-2000:]
        assert "shadowed" not in done.stderr
        assert done.returncode != 0 and "no-such-project" in (done.stdout + done.stderr)
        assert workers._environment()["PYTHONSAFEPATH"] == "1"
    finally:
        workers.shutdown(grace=1.0)


def test_a_worker_of_a_server_run_from_an_uninstalled_checkout_finds_the_package(monkeypatch):
    # `cd src && python -m paper_orchestra ui`: the server imports the package from the
    # directory it was started in, and its workers are to run the same code.
    import paper_orchestra

    workers = Workers(environ={"PYTHONPATH": "/somewhere/else"})
    try:
        assert workers._environment()["PYTHONPATH"] == "/somewhere/else"  # installed: as it was
        source = str(Path(paper_orchestra.__file__).resolve().parent.parent)
        monkeypatch.setattr(sys, "path", [source, *sys.path])
        assert workers._environment()["PYTHONPATH"] == os.pathsep.join([source, "/somewhere/else"])
        monkeypatch.setattr(sys, "path", ["", *sys.path[1:]])
        monkeypatch.chdir(source)
        assert workers._environment()["PYTHONPATH"].split(os.pathsep)[0] == source
    finally:
        workers.shutdown(grace=1.0)


@pytest.mark.parametrize("planted", ["link", "dangling", "pipe"])
def test_a_worker_log_that_is_not_the_projects_own_file_is_not_written_to(
    project, make_workers, tmp_path, planted
):
    # With worker.log -> ~/.bashrc, pressing Continue appended the command line (with the
    # person's instruction) and everything the worker printed to that file.
    outside = tmp_path / "bashrc"
    outside.write_text("export PATH=$PATH\n")
    log = project.root / WORKER_LOG
    if planted == "link":
        log.symlink_to(outside)
    elif planted == "dangling":
        log.symlink_to(tmp_path / "created-through-the-link")
    else:
        os.mkfifo(log)
    workers = make_workers()
    with pytest.raises(WorkerError, match=r"worker's log cannot be used.*Remove") as refused:
        workers.start(project.root, "advance", instruction="Something private.")
    assert str(log) in str(refused.value)
    assert outside.read_text() == "export PATH=$PATH\n"
    assert not (tmp_path / "created-through-the-link").exists()
    assert workers.status(project.root)["state"] == "idle"
    assert not (project.root / "ready").exists() and not (project.root / RECORD_FILE).exists()
    log.unlink()
    started(workers, project)  # with the planted file gone, the worker starts
    assert "starting:" in log.read_text()


# ---------------------------------------------------------------------------
# The end of the server, when it was asked to stop its workers
# ---------------------------------------------------------------------------


def test_whose_worker_it_is_is_decided_when_the_server_ends_not_when_it_was_found(
    project, make_workers
):
    # Two servers on one workspace; the one that started the worker is killed.  The
    # other, asked to stop workers at its end, then stops what would be nobody's.
    first = make_workers()
    pid = started(first, project)["pid"]
    other_server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        record = json.loads((project.root / RECORD_FILE).read_text())
        record.update(server_pid=other_server.pid, server_token=start_token_of(other_server.pid))
        (project.root / RECORD_FILE).write_text(json.dumps(record))
        second = make_workers(grace=5.0, stop_on_exit=True)
        assert second.status(project.root)["adopted"] is True
        assert second.live() == 0  # that server is alive: its worker
    finally:
        other_server.kill()
        other_server.wait()
    assert second.live() == 1  # it is gone: the worker would be left to nobody
    second.shutdown()
    assert not process_exists(pid) and (project.root / "interrupted").exists()


def test_a_stop_that_was_asked_for_is_carried_out_before_the_server_ends(project, make_workers):
    # The person pressed Stop and was told "stopping"; the server ending right after
    # must not leave a worker that ignores the interrupt running for good.
    workers = make_workers("ignore", grace=1.0)
    pid = started(workers, project)["pid"]
    assert workers.stop(project.root)["state"] == "stopping" and workers.stopping() == 1
    assert workers.close() == []  # nothing goes on
    assert not process_exists(pid) and workers.stopping() == 0
    assert state_of(project)["status"] == "stopped"


def test_a_found_worker_that_could_not_be_ended_is_still_shown_as_running(
    script, project, make_workers, signals, monkeypatch
):
    # A worker that runs as another user cannot be signalled.  (Here: no signal is sent.)
    process = engine_from_a_terminal(script, project)
    try:
        monkeypatch.setattr(workers_module, "_KILL_WAIT_SECONDS", 0.2)
        workers = make_workers(grace=0.2)
        assert workers.stop(project.root)["state"] == "stopping"
        wait_for(lambda: workers.stopping() == 0, message="the stop to give up")
        assert {signum for _, _, signum in signals} == {signal.SIGINT, signal.SIGKILL}
        status = workers.status(project.root)
        assert status["state"] == "running" and status["pid"] == process.pid
        assert state_of(project)["status"] == "running"  # not recorded as interrupted
        assert [event["type"] for event in project.events()] == ["project.created"]
    finally:
        monkeypatch.undo()  # signals are sent again
        os.killpg(process.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# The kill after the grace period
# ---------------------------------------------------------------------------


def test_a_process_group_is_killed_only_while_a_process_seen_in_it_is_still_there():
    # The kill comes a grace period after the groups were looked up.  A group whose
    # processes have all gone may be another group under the same number by then.
    def group() -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
        )

    unrelated, own = group(), group()
    try:
        gone = dead_pid()
        targets = workers_module._Targets()
        # As if the group of a job that has ended since had been given out again.
        targets.groups = {unrelated.pid: {gone: "1"}, own.pid: {own.pid: start_token_of(own.pid)}}
        targets.signal(signal.SIGKILL)
        assert own.wait(timeout=60) == -signal.SIGKILL
        assert unrelated.poll() is None
        # Under the same pid but not the process that was seen: not the same group either.
        targets.groups = {unrelated.pid: {unrelated.pid: "not-its-start-time"}}
        targets.signal(signal.SIGKILL)
        assert unrelated.poll() is None
    finally:
        for process in (unrelated, own):
            process.kill()
            process.wait()


# ---------------------------------------------------------------------------
# The runs a dead worker left behind
# ---------------------------------------------------------------------------


def test_the_runs_of_a_worker_that_died_are_settled_and_those_of_one_that_finished_are_not(
    project, make_workers, tmp_path
):
    workers = make_workers("die")
    workers.start(project.root, "run", experiments=["E1"])
    assert workers.wait(project.root, timeout=60)
    wait_for(lambda: make_workers.settled == [project.root], message="the runs to be settled")
    assert workers.settled(project.root, timeout=60)

    for kind, mode in (("run", "quick"), ("advance", "quick"), ("write", "die")):
        workers.start(project.root, kind, mode=mode)
        assert workers.wait(project.root, timeout=60)
    # A worker that ended in order settled its own runs; a write-up has none.
    assert workers.settled(project.root, timeout=60) and make_workers.settled == [project.root]

    workers.start(project.root, "advance", mode="die")  # an engine may die in its run phase
    assert workers.wait(project.root, timeout=60)
    wait_for(lambda: len(make_workers.settled) == 2, message="the runs to be settled")


def test_a_start_waits_until_the_runs_of_the_dead_worker_are_settled(script, project):
    # The new worker would otherwise find an experiment held by the server.
    order: list[str] = []
    entered, release = threading.Event(), threading.Event()

    def settle(root):
        entered.set()
        release.wait(60)
        order.append("settled")

    def launcher(kind, root, options):
        order.append("launched")
        return worker_command(script, kind, root, options.get("mode", "die"))

    workers = Workers(launcher=launcher, settle_runs=settle)
    try:
        workers.start(project.root, "run")
        assert entered.wait(60)
        assert workers.wait(project.root, timeout=60)
        again = threading.Thread(
            target=lambda: workers.start(project.root, "run", mode="quick"), daemon=True
        )
        again.start()
        time.sleep(0.3)  # time for a start that does not wait to show itself
        assert order == ["launched"]
        release.set()
        again.join(60)
        assert order == ["launched", "settled", "launched"]
        assert workers.wait(project.root, timeout=60)
    finally:
        release.set()
        workers.shutdown(grace=1.0)


def test_a_failure_to_settle_the_runs_is_logged_and_nothing_else(project, script, caplog):
    def settle(root):
        raise RuntimeError("the experiments module is not what it was")

    workers = Workers(
        launcher=lambda kind, root, options: worker_command(script, kind, root, "die"),
        settle_runs=settle,
    )
    try:
        workers.start(project.root, "run")
        assert workers.wait(project.root, timeout=60)
        assert workers.settled(project.root, timeout=60)
        wait_for(lambda: "Could not settle the runs" in caplog.text, message="the log line")
        assert workers.status(project.root)["last_exit"]["code"] == 3
        workers.start(project.root, "run", mode="quick")  # the next worker settles them itself
        assert workers.wait(project.root, timeout=60)
    finally:
        workers.shutdown(grace=1.0)


def run_left_by_a_dead_worker(project: Project, *, backend: str = "local") -> tuple[str, Path]:
    """An approved experiment whose run a worker left at ``running`` when it was killed."""
    from paper_orchestra.research import experiments

    page = project.create_page(
        "experiment",
        "Sweep warmup length",
        "## Plan\n\nRuns main.py.\n",
        author="human",
        fields={"mode": "script"},
    )
    experiments.write_code(project, page.id, {"main.py": "print('hello')\n"}, author="human")
    for state, by in (("approved", "human"), ("queued", "system"), ("running", "system")):
        project.set_state(page.path, state, by=by)
    directory = project.experiment_dir(page.id) / "runs" / "1"
    directory.mkdir(parents=True)
    record = {
        "experiment": page.id, "number": 1, "status": "running", "backend": backend,
        "handle": None if backend == "local" else "sb-123", "started": "2026-11-15T12:00:00Z",
        "finished": None, "seconds": None, "exit_code": None, "timed_out": False,
        "metrics": {}, "artifacts": [], "error": None, "cost_usd": None,
        "code_digest": "0" * 64, "repairs": [],
    }  # fmt: skip
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "log.txt").write_text("epoch 1\n")
    return str(page.id), directory


def test_a_run_whose_worker_died_is_settled_through_the_experiments_recovery(project):
    from paper_orchestra.research import experiments

    eid, directory = run_left_by_a_dead_worker(project)
    [record] = workers_module.settle_stale_runs(project.root)
    assert record["status"] == "failed" and "worker stopped" in record["error"]
    [run] = experiments.runs(project, eid)
    assert run["status"] == "failed" and json.loads((directory / "run.json").read_text()) == run
    # Back in the queue, not started: only a person's approval made it runnable, and a
    # worker (the person's Run, or the next phase) is what runs it again.
    assert Project.open(project.root).page(eid).state == "queued"
    assert workers_module.settle_stale_runs(project.root) == []  # nothing is left


def test_a_run_that_may_still_have_its_remote_job_is_left_for_the_next_worker(project):
    from paper_orchestra.research import experiments

    eid, directory = run_left_by_a_dead_worker(project, backend="modal")
    before = (directory / "run.json").read_text()
    # Recovery would follow the job to its end, for as long as that takes: a worker's
    # part, not something the server does while it answers requests.
    assert workers_module.settle_stale_runs(project.root) == []
    assert (directory / "run.json").read_text() == before
    assert experiments.runs(project, eid)[0]["status"] == "running"
    assert Project.open(project.root).page(eid).state == "running"


def test_a_local_run_is_settled_in_a_project_that_also_has_a_remote_job(project):
    """The server settles what needs no waiting (``recover(..., follow=False)``): a run
    with a remote job no longer keeps the dead local runs of its project unsettled."""
    from paper_orchestra.research import experiments

    remote, remote_dir = run_left_by_a_dead_worker(project, backend="modal")
    local, local_dir = run_left_by_a_dead_worker(project)
    before = (remote_dir / "run.json").read_text()
    [record] = workers_module.settle_stale_runs(project.root)
    assert record["experiment"] == local and record["status"] == "failed"
    assert json.loads((local_dir / "run.json").read_text())["status"] == "failed"
    assert experiments.runs(project, local)[0]["status"] == "failed"
    assert Project.open(project.root).page(local).state == "queued"
    # The remote job may be alive: following it is a worker's part.
    assert (remote_dir / "run.json").read_text() == before
    assert Project.open(project.root).page(remote).state == "running"
    assert workers_module.settle_stale_runs(project.root) == []


def test_the_lock_of_any_file_is_looked_at_without_taking_it_from_its_holder(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / ".run.lock"
    assert workers_module.lock_held(path) is False  # no file: nobody ever held it
    with path.open("w") as holder:
        assert workers_module.lock_held(path) is False
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert workers_module.lock_held(path) is True
        assert workers_module.lock_held(path) is True  # looking did not disturb the holder
    assert workers_module.lock_held(path) is False
    (tmp_path / "a-directory").mkdir()
    assert workers_module.lock_held(tmp_path / "a-directory") is None
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    assert workers_module.lock_held(path) is None  # a link is not the lock file


# Settling a run also cleans up after its job: the local backend ends what the job
# provably left behind (processes that carry the mark of a working directory the backend
# made) and removes that directory.  A run's record lies in the project directory like
# every other file, so nothing is ended because a record names it; and the server, which
# settles at a mere look, settles only where the backend would end no process at all
# (it asks the backend: ``LocalBackend.job_processes``).


@pytest.fixture
def backend_scratch():
    """Makes working directories where the local backend makes its own (directly in the
    system's temporary directory, with its prefix), and removes them afterwards."""
    import shutil
    import tempfile

    from paper_orchestra.compute import local

    made: list[Path] = []

    def make(name: str) -> Path:
        scratch = Path(tempfile.mkdtemp(prefix=f"{local.SCRATCH_PREFIX}{name}-"))
        made.append(scratch)
        return scratch

    yield make
    for scratch in made:
        shutil.rmtree(scratch, ignore_errors=True)


on_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="what a job left running is found through /proc"
)


def name_job(directory: Path, job: dict, *, where: str) -> None:
    """Put the record of a run's job where recovery reads it: in ``run.json``, or next to
    it in ``job.json`` (where the backend writes it when the job starts)."""
    if where == "record":
        record = json.loads((directory / "run.json").read_text())
        (directory / "run.json").write_text(json.dumps({**record, "job": job}))
    else:
        (directory / "job.json").write_text(json.dumps(job))


def session_record(pid: int, scratch: str = "") -> dict:
    """What the local backend records of a job whose session ``pid`` leads."""
    from paper_orchestra.compute import local

    started = local._session_started(pid)
    assert started, "the start of the session could not be read"
    return {"session": pid, "session_started": started, "scratch": scratch}


@on_linux
@pytest.mark.parametrize("where", ["record", "job.json"])
def test_a_run_record_that_names_somebody_elses_session_ends_nothing_at_a_look(
    project, make_workers, victim, monkeypatch, where
):
    # A project directory from somebody else: an experiment at running, a run record that
    # names a session of the person's as its job, and a state that names a worker.  Every
    # number in it can be read by anybody (/proc/<pid>/stat).  A look at the library used
    # to end that session.
    from paper_orchestra.research import experiments

    monkeypatch.setattr(workers_module, "_JOB_GONE_WAIT_SECONDS", 0.3, raising=False)
    eid, directory = run_left_by_a_dead_worker(project)
    name_job(directory, session_record(victim.pid), where=where)
    before = (directory / "run.json").read_text()
    plant_state(project, {"pid": dead_pid(), "host": HOST, "heartbeat": stamp(-300)})

    workers = make_workers(settle_runs=None)  # the real settling, as the server has it
    assert workers.status(project.root)["state"] == "idle"
    assert workers.settled(project.root, timeout=60)
    assert victim.poll() is None and process_exists(victim.child)
    # Changed with the backend's proof (FIXES K): a session that a record merely names is
    # nothing the backend would end, so it holds nothing up either.  The run is settled
    # as one whose worker stopped, and the session it named is untouched.
    assert (directory / "run.json").read_text() != before
    [run] = experiments.runs(project, eid)
    assert run["status"] == "failed" and "worker stopped" in run["error"]
    assert workers_module.settle_stale_runs(project.root) == []
    assert victim.poll() is None and process_exists(victim.child)


@on_linux
def test_a_run_record_that_names_the_working_directory_of_a_live_job_ends_nothing(
    project, backend_scratch, monkeypatch
):
    # The working directories of jobs can be listed by anybody (they are made in the
    # temporary directory).  A record that names the one of a job that is running, of
    # another project, used to get that job killed and its directory removed.
    from paper_orchestra.compute import local

    monkeypatch.setattr(workers_module, "_JOB_GONE_WAIT_SECONDS", 0.3, raising=False)
    scratch = backend_scratch("another")
    (scratch / "work").mkdir(parents=True)
    (scratch / "work" / "checkpoint.bin").write_text("hours of work")
    job = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, local.JOB_ENV: str(scratch)},
        start_new_session=True,
    )
    try:
        _, directory = run_left_by_a_dead_worker(project)
        name_job(
            directory,
            {"session": 0, "session_started": "", "scratch": str(scratch)},
            where="record",
        )
        before = (directory / "run.json").read_text()
        assert workers_module.settle_stale_runs(project.root) == []
        assert job.poll() is None
        assert (scratch / "work" / "checkpoint.bin").read_text() == "hours of work"
        assert (directory / "run.json").read_text() == before
    finally:
        job.kill()
        job.wait()


@on_linux
def test_what_a_dead_job_left_behind_is_still_cleaned_up_when_its_run_is_settled(
    project, backend_scratch, tmp_path
):
    # Changed with the backend's proof (FIXES K): only a working directory where the
    # backend makes them is the backend's to remove; one that a record names anywhere
    # else is left alone, and the run's notes say so.
    from paper_orchestra.compute import local

    elsewhere = tmp_path / f"{local.SCRATCH_PREFIX}elsewhere"
    elsewhere.mkdir()
    _, other = run_left_by_a_dead_worker(project)
    name_job(other, {"session": 0, "session_started": "", "scratch": str(elsewhere)},
             where="job.json")  # fmt: skip
    [left] = workers_module.settle_stale_runs(project.root)
    assert left["status"] == "failed" and elsewhere.is_dir()

    scratch = backend_scratch("left")
    (scratch / "work").mkdir(parents=True)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    record = session_record(gone.pid, str(scratch))
    gone.wait()
    _, directory = run_left_by_a_dead_worker(project)
    name_job(directory, record, where="job.json")
    [settled] = workers_module.settle_stale_runs(project.root)
    assert settled["status"] == "failed" and "worker stopped" in settled["error"]
    assert not scratch.exists()


@on_linux
def test_a_job_that_is_just_dying_with_its_worker_is_waited_for(
    project, backend_scratch, monkeypatch
):
    # The job of a worker that was killed is killed with it, a moment later.
    from paper_orchestra.compute import local

    monkeypatch.setattr(workers_module, "_JOB_GONE_WAIT_SECONDS", 120.0, raising=False)
    scratch = backend_scratch("dying")
    dying = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1.5)"],
        env={**os.environ, local.JOB_ENV: str(scratch)},  # the mark every job carries
        start_new_session=True,
    )
    try:
        _, directory = run_left_by_a_dead_worker(project)
        name_job(directory, session_record(dying.pid, str(scratch)), where="record")
        assert local.LocalBackend.job_processes(session_record(dying.pid, str(scratch)))
        collected = threading.Thread(target=dying.wait, daemon=True)
        collected.start()  # as the system collects a job whose worker is gone
        [settled] = workers_module.settle_stale_runs(project.root)
        assert settled["status"] == "failed"
        assert dying.wait(timeout=60) == 0  # it ended by itself: the server ended nothing
    finally:
        dying.kill()
        dying.wait()


def test_a_start_waits_for_the_runs_of_a_worker_it_finds_dead_itself(script, project):
    # `research run` in a terminal was killed while no server looked.  The start that
    # notices it must not launch its worker beside the settling it has just set off: the
    # worker would find the experiment held by the server.
    order: list[str] = []
    entered, release = threading.Event(), threading.Event()

    def settle(root):
        entered.set()
        release.wait(60)
        order.append("settled")

    def launcher(kind, root, options):
        order.append("launched")
        return worker_command(script, kind, root, "quick")

    plant_state(project, {"pid": dead_pid(), "host": HOST, "heartbeat": stamp(-300)}, "checkpoint")
    workers = Workers(launcher=launcher, settle_runs=settle)
    try:
        start = threading.Thread(target=lambda: workers.start(project.root, "run"), daemon=True)
        start.start()
        assert entered.wait(60)
        time.sleep(0.3)  # time for a start that does not wait to show itself
        assert order == []
        release.set()
        start.join(60)
        assert order == ["settled", "launched"]
        assert workers.wait(project.root, timeout=60)
    finally:
        release.set()
        workers.shutdown(grace=1.0)


def test_the_worker_lock_has_the_engines_name():
    engine = pytest.importorskip("paper_orchestra.research.engine")
    assert workers_module.WORKER_LOCK_FILE == engine.WORKER_LOCK_FILE == LOCK_FILE
