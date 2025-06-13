"""Unified progress monitoring utilities for CJE."""

from typing import Optional, Any, Iterable, TypeVar, Union, ContextManager
from contextlib import contextmanager, nullcontext
import os

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    ProgressColumn,
    TaskID,
)
from rich.console import Console
from rich.table import Table
import time

T = TypeVar("T")

# Global console for consistent output
console = Console()

# Check if progress should be disabled (e.g., in CI/CD)
DISABLE_PROGRESS = os.environ.get("CJE_DISABLE_PROGRESS", "").lower() in (
    "1",
    "true",
    "yes",
)


class RateColumn(ProgressColumn):
    """Shows the rate of progress (items/second)."""

    def render(self, task: Any) -> str:
        """Render the column."""
        if task.speed is None:
            return "[progress.percentage]0.0 it/s[/progress.percentage]"
        return f"[progress.percentage]{task.speed:.1f} it/s[/progress.percentage]"


def get_progress(
    description: str = "Processing",
    total: Optional[int] = None,
    disable: bool = False,
    transient: bool = True,
) -> Progress:
    """Create a configured Progress instance with CJE styling.

    Args:
        description: Description of the task
        total: Total number of items (None for indeterminate)
        disable: Whether to disable progress display
        transient: Whether to remove progress bar when complete

    Returns:
        Configured Progress instance
    """
    if disable or DISABLE_PROGRESS:
        # Return a dummy progress that does nothing
        class DummyProgress:
            def add_task(self, *args: Any, **kwargs: Any) -> int:
                return 0

            def update(self, *args: Any, **kwargs: Any) -> None:
                pass

            def advance(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> "DummyProgress":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        return DummyProgress()  # type: ignore

    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]

    if total is not None:
        columns.extend(
            [
                TimeRemainingColumn(),
                RateColumn(),
            ]
        )

    return Progress(
        *columns,
        console=console,
        transient=transient,
    )


def track(
    sequence: Iterable[T],
    description: str = "Processing",
    total: Optional[int] = None,
    disable: bool = False,
    transient: bool = True,
) -> Iterable[T]:
    """Track progress of an iterable, similar to tqdm.

    Args:
        sequence: Iterable to track
        description: Description of the task
        total: Total number of items (auto-detected if possible)
        disable: Whether to disable progress display
        transient: Whether to remove progress bar when complete

    Yields:
        Items from the sequence
    """
    if disable or DISABLE_PROGRESS:
        yield from sequence
        return

    # Try to get length if not provided
    if total is None:
        try:
            total = len(sequence)  # type: ignore
        except (TypeError, AttributeError):
            pass

    with get_progress(description, total, disable, transient) as progress:
        task_id = progress.add_task(description, total=total)
        for item in sequence:
            yield item
            progress.advance(task_id)


@contextmanager
def progress_context(
    description: str = "Processing",
    total: Optional[int] = None,
    disable: bool = False,
) -> Any:
    """Context manager for manual progress updates.

    Example:
        with progress_context("Training", total=epochs) as progress:
            for epoch in range(epochs):
                # ... training code ...
                progress.update(1)
    """
    if disable or DISABLE_PROGRESS:
        # Return dummy context
        class DummyContext:
            def update(self, advance: int = 1) -> None:
                pass

            def set_description(self, description: str) -> None:
                pass

        yield DummyContext()
        return

    with get_progress(description, total, disable, transient=False) as progress:
        task_id = progress.add_task(description, total=total)

        class ProgressContext:
            def update(self, advance: int = 1) -> None:
                progress.advance(task_id, advance)

            def set_description(self, description: str) -> None:
                progress.update(task_id, description=description)

        yield ProgressContext()


class ProgressMonitor:
    """Monitor multiple concurrent tasks with a single progress display."""

    def __init__(self, disable: bool = False):
        """Initialize the progress monitor."""
        self.disable = disable or DISABLE_PROGRESS
        self.progress: Optional[Progress] = None
        self.tasks: dict[str, TaskID] = {}

    def __enter__(self) -> "ProgressMonitor":
        """Enter the context."""
        if not self.disable:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                RateColumn(),
                console=console,
                transient=False,
            )
            self.progress.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        if self.progress:
            self.progress.__exit__(*args)

    def add_task(
        self, name: str, description: str, total: Optional[int] = None
    ) -> None:
        """Add a new task to monitor."""
        if self.progress:
            task_id = self.progress.add_task(description, total=total)
            self.tasks[name] = task_id

    def update(self, name: str, advance: int = 1, **kwargs: Any) -> None:
        """Update a task's progress."""
        if self.progress and name in self.tasks:
            self.progress.update(self.tasks[name], advance=advance, **kwargs)

    def set_total(self, name: str, total: int) -> None:
        """Set the total for a task."""
        if self.progress and name in self.tasks:
            self.progress.update(self.tasks[name], total=total)


def print_summary_table(
    title: str,
    data: list[dict[str, Any]],
    columns: Optional[list[str]] = None,
) -> None:
    """Print a summary table using rich.

    Args:
        title: Table title
        data: List of dictionaries with data
        columns: Column names to display (default: all keys)
    """
    if not data:
        return

    if columns is None:
        columns = list(data[0].keys())

    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    for col in columns:
        table.add_column(col)

    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, "")) for col in columns])

    console.print(table)


# ------------------------------------------------------------------
# Helper: maybe_track (conditionally wrap iterable with progress)
# ------------------------------------------------------------------


def maybe_track(
    iterable: Iterable[T],
    *,
    description: str = "Processing",
    total: Optional[int] = None,
    verbose: bool = False,
) -> Iterable[T]:
    """Return rich.track(iterable) if verbose else iterable.

    Args:
        iterable: The iterable to wrap
        description: Description for the progress bar
        total: Optional total length
        verbose: If False, returns the iterable unchanged
    """
    if not verbose:
        return iterable
    return track(iterable, description=description, total=total)


# Export main functions
__all__ = [
    "track",
    "progress_context",
    "ProgressMonitor",
    "get_progress",
    "print_summary_table",
    "console",
    "DISABLE_PROGRESS",
]
