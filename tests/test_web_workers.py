"""Worker processes of the web server: start, stop, death, adoption, shutdown.

Every worker here is a small Python script that behaves like the engine in one
respect (it stops on an interrupt, ignores it, dies, starts a job of its own).
"""

from __future__ import annotations

import contextlib
import json
import os
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

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and signals")

# What the stand-in worker does, by its second argument.
WORKER_SCRIPT = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

root, mode = Path(sys.argv[1]), sys.argv[2]
state_file = root / "state.json"


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


def wait_for(condition, timeout: float = 15.0, message: str = "condition") -> None:
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
    path = tmp_path / "stand_in_worker.py"
    path.write_text(WORKER_SCRIPT, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "ws" / "proj", title="Warmup", brief="Does warmup matter?")


@pytest.fixture
def make_workers(script):
    """Build supervisors whose workers are the stand-in script; all are shut down after."""
    made: list[Workers] = []

    def make(mode: str = "sleep", grace: float = 10.0) -> Workers:
        def launcher(kind, root, options):
            return [sys.executable, str(script), str(root), options.get("mode", mode)]

        workers = Workers(launcher=launcher, grace=grace)
        made.append(workers)
        return workers

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
    assert workers.wait(project.root, timeout=15)
    status = workers.status(project.root)
    assert status["state"] == "idle"
    assert status["last_exit"]["code"] == 0 and status["last_exit"]["stopped"] is False
    assert status["last_exit"]["interrupted_phase"] is None
    assert state_of(project)["status"] == "checkpoint"  # the worker's own account stands
    assert not (project.root / RECORD_FILE).exists()
    assert "done at a checkpoint" in (project.root / WORKER_LOG).read_text()
    # The next worker appends to the same log.
    workers.start(project.root, "advance")
    assert workers.wait(project.root, timeout=15)
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
        assert workers.wait(project.root, timeout=15)
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
    assert workers.wait(project.root, timeout=15)

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
    assert workers.wait(project.root, timeout=15)
    assert 0.4 <= time.monotonic() - before < 10
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
    workers = make_workers("term-only", grace=10.0)
    started(workers, project)
    workers.stop(project.root)
    assert workers.wait(project.root, timeout=8)  # well inside the grace period
    assert (project.root / "terminated").exists()
    assert state_of(project)["status"] == "stopped"


def test_stop_also_ends_what_the_worker_started_in_other_sessions(project, make_workers):
    workers = make_workers("job", grace=0.5)
    started(workers, project)
    job = int((project.root / "job.pid").read_text())
    assert process_exists(job) and os.getsid(job) == job
    workers.stop(project.root)
    assert workers.wait(project.root, timeout=15)
    # The worker itself went at the interrupt; its job ignored it and was killed.
    assert (project.root / "interrupted").exists()
    wait_for(lambda: not process_exists(job), message="the job to be killed")


def test_what_the_worker_started_after_the_interrupt_is_killed_with_it(project, make_workers):
    workers = make_workers("late-job", grace=1.0)
    pid = started(workers, project)["pid"]
    workers.stop(project.root)
    wait_for(lambda: (project.root / "job.pid").exists(), message="the late job")
    job = int((project.root / "job.pid").read_text())
    assert workers.wait(project.root, timeout=15)
    assert not process_exists(pid)
    # It was not there when the stop began; it is looked for again before the kill.
    wait_for(lambda: not process_exists(job), message="the late job to be killed")


# ---------------------------------------------------------------------------
# A worker that dies
# ---------------------------------------------------------------------------


def test_a_worker_that_dies_is_recorded_as_interrupted_and_reported(project, make_workers):
    workers = make_workers("die")
    workers.start(project.root, "advance")
    assert workers.wait(project.root, timeout=15)

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
    assert workers.wait(project.root, timeout=15)
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
        assert workers.wait(project.root, timeout=15)
        assert workers.status(project.root)["last_exit"]["code"] == 1
        assert state_of(project) == earlier  # nothing to settle: it never got that far
        error = project.events()[-1]
        assert error["type"] == "error" and "exit code 1" in error["summary"]
        assert "worker.log" in error["summary"]
        assert "No credentials are configured." in (project.root / WORKER_LOG).read_text()

        # A run or a write-up that ends the same way (the command refused, the project is
        # locked by another worker) left no account either.
        workers.start(project.root, "run", experiments=["E1"])
        assert workers.wait(project.root, timeout=15)
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
            assert workers.wait(project.root, timeout=30)
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
        assert workers.wait(project.root, timeout=15)
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
        assert workers.wait(project.root, timeout=15)
        status = workers.status(project.root)
        assert status["state"] == "idle" and status["last_exit"]["code"] == -signal.SIGKILL
        state = state_of(project)
        assert state["worker"] is None and state["status"] == "checkpoint"
        assert project.events()[-1]["type"] == "error"
        workers.start(project.root, "run", experiments=["E1"])  # nothing stands in the way
        assert workers.wait(project.root, timeout=15)
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
    assert workers.wait(project.root, timeout=15)


def test_a_server_that_exits_does_not_wait_out_a_slow_stop(project, make_workers):
    workers = make_workers("ignore", grace=60.0)
    pid = started(workers, project)["pid"]
    assert workers.stop(project.root)["state"] == "stopping"  # would take a minute
    before = time.monotonic()
    workers.shutdown(grace=0.3)
    assert time.monotonic() - before < 10
    assert not process_exists(pid)
    assert state_of(project)["status"] == "stopped"


def test_the_exit_hook_stops_workers_of_a_server_that_did_not_shut_down(project, make_workers):
    workers = make_workers()
    pid = started(workers, project)["pid"]
    workers_module._stop_all_at_exit()
    assert not process_exists(pid)
    assert (project.root / "interrupted").exists()


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
    assert new_server.wait(project.root, timeout=15)
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
    assert old_server.wait(project.root, timeout=15)
    # As the killed server left things: its record, and a state that says running.
    (project.root / RECORD_FILE).write_text(record)
    (project.root / "state.json").write_text(json.dumps(state))

    new_server = make_workers()
    status = new_server.status(project.root)
    assert status["state"] == "idle"  # known dead, however recent the heartbeat
    assert status["last_exit"]["interrupted_phase"] == "research"
    assert state_of(project)["status"] == "stopped"
    assert not (project.root / RECORD_FILE).exists()


def engine_from_a_terminal(script: Path, project: Project) -> subprocess.Popen:
    """An engine nobody's server started, as ``paper-orchestra research advance`` in a
    terminal would be: known only through ``state.json["worker"]``."""
    process = subprocess.Popen(
        [sys.executable, str(script), str(project.root), "sleep"], start_new_session=True
    )
    threading.Thread(target=process.wait, daemon=True).start()
    wait_for(lambda: (project.root / "ready").exists())
    state = state_of(project)
    state["worker"] = {
        "pid": process.pid,
        "host": socket.gethostname(),
        "started": format_stamp(datetime.now(UTC)),
        "heartbeat": format_stamp(datetime.now(UTC)),
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
    assert workers.wait(project.root, timeout=15)
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
        assert first.wait(project.root, timeout=15)
        assert (project.root / "interrupted").exists()
    finally:
        other_server.kill()
        other_server.wait()


PARENT_SCRIPT = r"""
import os, subprocess, sys
from pathlib import Path

script, root, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
# Like a shell script or a Makefile that runs the engine: same process group as the child.
child = subprocess.Popen([sys.executable, script, str(root), mode])
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
        assert workers.wait(project.root, timeout=15)
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
    assert workers.wait(project.root, timeout=15)


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


def test_default_command_is_the_packages_own_cli(tmp_path):
    root = tmp_path / "proj"
    base = [sys.executable, "-m", "paper_orchestra", "research"]
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
    # A project directory is never read as an option, whatever it is called.
    assert default_command("advance", Path("-h"), {})[-1] == "./-h"
