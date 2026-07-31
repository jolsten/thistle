"""Interactive progress display.

A single shared rich Console on **stderr** backs both the progress bar and
loguru's console sink: rich renders log lines above the live bar instead of
letting them garble it, and stdout stays clean for command output
(``get-tle``, ``dump`` messages, etc.).

``ProgressReporter`` is a context manager that no-ops entirely when
disabled, so ``ingest``/``generate`` internals can report progress
unconditionally and crons pay zero cost. The CLI enables it only when
stderr is an interactive terminal and ``--no-progress`` was not passed.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

console = Console(stderr=True)


class ProgressReporter:
    """Bottom-anchored progress bar; every method no-ops when disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._progress: Optional[Progress] = None

    def __enter__(self) -> "ProgressReporter":
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,  # bar disappears when done; logs remain
            )
            self._progress.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def task(self, description: str, total: Optional[int] = None) -> TaskID:
        """Add a task. ``total=None`` renders an indeterminate bar with a
        running count (used when the item count isn't known upfront)."""
        if self._progress is None:
            return TaskID(-1)
        return self._progress.add_task(description, total=total)

    def advance(self, task_id: TaskID, n: int = 1) -> None:
        if self._progress is not None:
            self._progress.advance(task_id, n)

    def finish(self, task_id: TaskID) -> None:
        """Remove a completed task's bar (multi-phase commands start the
        next phase with a fresh bar)."""
        if self._progress is not None:
            self._progress.remove_task(task_id)


NO_PROGRESS = ProgressReporter(enabled=False)
"""Shared inert reporter — the default for library callers."""
