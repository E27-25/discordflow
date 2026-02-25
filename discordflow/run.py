"""ActiveRun — context manager for a single DiscordFlow training run."""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional

from .utils import (
    COLOR_TEAL,
    COLOR_RED,
    COLOR_GREEN,
    format_kv_table,
    human_duration,
    ascii_progress,
    truncate,
)

if TYPE_CHECKING:
    from .core import DiscordFlow


class ActiveRun:
    """
    Represents a single experiment run.

    Returned by :meth:`DiscordFlow.start_run` and designed to be used as a
    context manager::

        with dflow.start_run("epoch_sweep") as run:
            run.log_metrics({"loss": 0.42}, step=1)

    On exit the run automatically posts a summary embed to Discord showing
    all logged params, final metrics, tags, and elapsed time.
    """

    def __init__(self, tracker: "DiscordFlow", run_name: str):
        self._tracker = tracker
        self.run_name = run_name
        self._start_time: float = time.time()

        # Accumulated state
        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}   # last value per key
        self._tags: Dict[str, Any] = {}
        self._steps: int = 0
        self._artifacts: list[str] = []
        self._status: str = "RUNNING"

    # ------------------------------------------------------------------
    # Logging helpers (delegate to parent tracker, also track locally)
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self._params[key] = value
        self._tracker.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters at once."""
        self._params.update(params)
        self._tracker.log_params(params)

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """Log a single metric value."""
        self._metrics[key] = value
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        self._metrics.update(metrics)
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metrics(metrics, step=step)

    def set_tag(self, key: str, value: Any) -> None:
        """Attach an arbitrary tag to this run."""
        self._tags[key] = value
        self._tracker.set_tag(key, value)

    def log_artifact(self, file_path: str) -> None:
        """Upload a file artifact."""
        self._artifacts.append(file_path)
        self._tracker.log_artifact(file_path)

    def log_text(self, text: str, filename: str = "output.txt") -> None:
        """Upload a text snippet as a file artifact."""
        self._tracker.log_text(text, filename)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._status = "FAILED"
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._tracker._post_run_summary(self, error_text=tb_text)
            # Don't suppress the exception
            return False

        self._status = "FINISHED"
        self._tracker._post_run_summary(self)
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def __repr__(self) -> str:
        return f"<ActiveRun name={self.run_name!r} status={self._status}>"
