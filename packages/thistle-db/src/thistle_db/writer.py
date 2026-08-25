"""Parallel filesystem work for output generation.

Generation is dominated by per-file syscall latency, not by computation: in
a profile of a 200k-row / 5k-object catalog across four outputs, `open` and
close accounted for 16.6s of a 38s full rebuild — 44% of the run, spent
waiting. Those waits release the GIL, so a small thread pool recovers most
of it.

The pool only ever receives **pure filesystem work**. The SQLAlchemy
`Session` is not thread-safe, so every query stays on the calling thread;
tasks are handed plain data (rows already fetched, paths already resolved)
and hand back plain results. Formatting runs inside the task rather than
before it, so one worker's CSV building overlaps another's blocking write.

Work is submitted a chunk at a time via `map`, which waits for the chunk
before returning. That keeps memory bounded (only one chunk of rows is in
flight), surfaces an exception at the chunk boundary instead of at the end
of the run, and — because each task owns a distinct path — means no two
tasks ever touch the same file.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Beyond this, more threads buy nothing: the work is one blocking syscall
# per file, and the queue depth the filesystem can usefully absorb is small.
MAX_WORKERS = 32


def resolve_workers(configured: int) -> int:
    """Thread count for a `write_workers` setting (0 = auto)."""
    if configured > 0:
        return min(configured, MAX_WORKERS)
    # I/O-bound: oversubscribe cores, since threads spend their time blocked.
    return min(MAX_WORKERS, (os.cpu_count() or 4) * 4)


class WritePool:
    """Runs filesystem tasks across a thread pool, a chunk at a time.

    With one worker it runs everything inline and never creates a thread —
    the escape hatch for debugging, and for network filesystems that handle
    concurrent writes poorly.
    """

    def __init__(self, workers: int = 0):
        self.workers = resolve_workers(workers)
        self._executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> "WritePool":
        if self.workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="thistle-write"
            )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply `fn` to every item, returning results in input order.

        Waits for the whole batch. An exception in any task propagates once
        the batch is drained, so a failure can't be silently dropped.
        """
        items = list(items)
        if self._executor is None or len(items) < 2:
            return [fn(item) for item in items]
        return list(self._executor.map(fn, items))
