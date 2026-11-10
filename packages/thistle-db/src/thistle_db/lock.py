"""Single-writer enforcement — an advisory OS lock on a lockfile.

Single-writer ingest is a universal invariant: output-file guards assume
serialized ingest. The lock is an OS-level advisory lock (``flock`` /
``msvcrt.locking``), released automatically by the OS when the process
dies, so a crash can never wedge it — which is why the ``running`` row in
``ingest_runs`` is a diagnostic, never a lock.

The lockfile lives beside the config file (``<config>.lock``), so two
deployments with different configs on one host never contend.
"""

import pathlib
import sys

__all__ = ["LockHeldError", "SingleWriterLock", "lock_path_for"]


class LockHeldError(RuntimeError):
    """Another writer holds the lock — a second ingest must not start."""


def lock_path_for(config_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(config_path) + ".lock")


class SingleWriterLock:
    """Non-blocking exclusive lock on a lockfile; context manager.

    Raises LockHeldError immediately when another process holds it. The
    lockfile itself is left in place after release (removing it on
    Windows while another process probes it races; an empty file is
    harmless).
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._handle = None

    def __enter__(self) -> "SingleWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise LockHeldError(
                f"another thistle-db writer holds {self.path} — single-writer "
                "ingest is enforced; wait for the running command to finish"
            ) from None
        self._handle = handle
        return self

    def __exit__(self, *exc_info) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
