"""One consolidation of a wiki at a time.

Runs that share a wiki (parallel papers and the lab memory) must not
consolidate it together.  The lock is a kernel lock on a file in the wiki's
directory: the kernel releases it when its holder exits, however the holder
died, so there is nothing to time out and nothing for a waiter to take over.

Where the file system has no such locks, an exclusively created file is used
instead, and a file that has not been refreshed for a long time is taken to
belong to a holder that died.  Both are mutual exclusion on a local file
system; storage where neither works (some network volumes) needs a single
consolidating process.
"""

from __future__ import annotations

import contextlib
import errno
import os
import socket
import threading
import time
from pathlib import Path

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

LOCK_FILE = ".dream.lock"
#: Only for the file scheme: a lock not refreshed for this long belongs to a dead holder.
STALE_LOCK_SECONDS = 15 * 60
LOCK_REFRESH_SECONDS = 60
_NO_KERNEL_LOCKS = {errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}


class WikiLock:
    """Context manager; check ``acquired`` inside the block.

    Entering never raises and never waits longer than ``wait_seconds``.  When
    the lock was not acquired, ``error`` says why if the reason is anything
    other than another holder (for example a directory that cannot be written).
    """

    def __init__(self, root: Path, *, wait_seconds: float = 120.0, poll: float = 1.0) -> None:
        self.path = Path(root) / LOCK_FILE
        self.wait_seconds = wait_seconds
        self.poll = poll
        self.acquired = False
        self.error: str | None = None
        self._fd: int | None = None
        self._by_file = False
        self._stop = threading.Event()

    # -- kernel lock -----------------------------------------------------------

    def _try_kernel_lock(self) -> bool | None:
        """True: acquired.  False: not now.  None: this file system has no kernel locks."""
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as exc:
            self.error = f"the lock file {self.path} cannot be opened: {exc}"
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError as exc:
            os.close(fd)
            if exc.errno in _NO_KERNEL_LOCKS:
                return None
            self.error = f"the lock file {self.path} cannot be locked: {exc}"
            return False
        try:
            # The path must still name the file that was locked: a holder using the file
            # scheme removes its file when it finishes.
            current = os.stat(self.path).st_ino == os.fstat(fd).st_ino
        except OSError:
            current = False
        if not current:
            os.close(fd)
            return False
        with contextlib.suppress(OSError):  # who holds it, for a person looking at the file
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {socket.gethostname()}\n".encode())
        self._fd = fd
        return True

    # -- file scheme -----------------------------------------------------------

    def _stale(self) -> bool:
        try:
            return time.time() - self.path.stat().st_mtime > STALE_LOCK_SECONDS
        except OSError:
            return False

    def _try_file(self, *, retried: bool = False) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if retried or not self._stale():
                return False
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # Cannot be removed (another user's file, a read-only directory): wait as
                # for a live holder instead of trying again and again.
                return False
            return self._try_file(retried=True)
        except OSError as exc:
            self.error = f"the lock file {self.path} cannot be created: {exc}"
            return False
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()} {socket.gethostname()}\n")
        self._by_file = True
        threading.Thread(target=self._refresh, daemon=True).start()
        return True

    def _refresh(self) -> None:
        while not self._stop.wait(LOCK_REFRESH_SECONDS):
            with contextlib.suppress(OSError):
                os.utime(self.path, None)

    # -- context manager ---------------------------------------------------------

    def _try(self) -> bool:
        if fcntl is not None:
            outcome = self._try_kernel_lock()
            if outcome is not None:
                return outcome
        return self._try_file()

    def __enter__(self) -> WikiLock:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                self.acquired = self._try()
            except Exception as exc:  # entering a lock must never be what fails a run
                self.error = f"{type(exc).__name__}: {exc}"
            if self.acquired or self.error or time.monotonic() >= deadline:
                return self
            time.sleep(self.poll)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
        if self._by_file:
            with contextlib.suppress(OSError):
                self.path.unlink()
            self._by_file = False
        self.acquired = False
