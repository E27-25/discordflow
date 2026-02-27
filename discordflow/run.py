"""ActiveRun and ForumActiveRun — context managers for DiscordFlow runs."""

from __future__ import annotations

import os
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .core import DiscordFlow


class ActiveRun:
    """
    Context manager for a single run on a **normal** Discord channel.

    Returned by :meth:`DiscordFlow.start_run`.
    All log calls post embeds directly into the channel.

    Usage::

        with dflow.start_run("baseline") as run:
            run.log_params({"lr": 3e-4})
            run.log_metrics({"loss": 0.5}, step=1, system_metrics=["cpu", "ram"])
    """

    def __init__(self, tracker: "DiscordFlow", run_name: str):
        self._tracker  = tracker
        self.run_name  = run_name
        self._start_time = time.time()

        self._params:    Dict[str, Any] = {}
        self._metrics:   Dict[str, Any] = {}
        self._tags:      Dict[str, Any] = {}
        self._steps:     int            = 0
        self._artifacts: list           = []
        self._status:    str            = "RUNNING"

    # ------------------------------------------------------------------
    # Logging (delegates to tracker, also accumulates locally)
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value
        self._tracker.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        self._tracker.log_params(params)

    def log_metric(
        self, key: str, value: Any,
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
    ) -> None:
        self._metrics[key] = value
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metric(key, value, step=step, system_metrics=system_metrics)

    def log_metrics(
        self, metrics: Dict[str, Any],
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Log multiple metrics.

        Parameters
        ----------
        system_metrics:
            List of hardware stats to attach. Options:
            ``"cpu"``, ``"ram"``, ``"gpu"``, ``"disk"``, ``"network"``.
        """
        self._metrics.update(metrics)
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metrics(metrics, step=step, system_metrics=system_metrics)

    def set_tag(self, key: str, value: Any) -> None:
        self._tags[key] = value
        self._tracker.set_tag(key, value)

    def log_artifact(self, file_path: str) -> None:
        self._artifacts.append(file_path)
        self._tracker.log_artifact(file_path)

    def log_text(self, text: str, filename: str = "output.txt") -> None:
        self._tracker.log_text(text, filename)

    def log_figure(self, fig, title: str = "Training Graph") -> None:
        """Send a matplotlib figure as a PNG into the channel."""
        self._tracker.log_figure(fig, title=title)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._status  = "FAILED"
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._tracker._post_run_summary(self, error_text=tb_text)
            return False
        self._status = "FINISHED"
        self._tracker._post_run_summary(self)
        return False

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def __repr__(self) -> str:
        return f"<ActiveRun name={self.run_name!r} status={self._status}>"


class ForumActiveRun(ActiveRun):
    """
    Context manager for a single run on a **Discord Forum channel**.

    Returned by :meth:`DiscordFlow.start_forum_run`.
    All log calls are automatically routed into the dedicated forum thread.

    Usage::

        with dflow.start_forum_run("lora_rank_16") as run:
            run.log_params({"lr": 2e-4, "lora_rank": 16})
            run.log_metrics({"loss": 0.4}, step=1, system_metrics=["cpu", "ram", "gpu"])
            run.log_figure(fig, title="Loss Curve")
        # ← summary posted in the thread automatically
    """

    def __init__(self, tracker: "DiscordFlow", run_name: str, thread_id: str):
        super().__init__(tracker=tracker, run_name=run_name)
        self.thread_id = thread_id

    # ------------------------------------------------------------------
    # Override all logging to inject thread_id
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value
        # Params go to the thread as a plain embed
        self._tracker._send_to_thread(self.thread_id, {
            "title": "⚙️  Parameter Logged",
            "color": 0x3498DB,
            "fields": [
                {"name": str(key), "value": str(value), "inline": True},
            ],
        })

    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        fields = [{"name": str(k), "value": str(v), "inline": True} for k, v in params.items()]
        self._tracker._send_to_thread(self.thread_id, {
            "title": f"⚙️  Parameters — {len(params)} logged",
            "color": 0x3498DB,
            "fields": fields,
        })

    def log_metric(
        self, key: str, value: Any,
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
    ) -> None:
        self._metrics[key] = value
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metrics(
            {key: value}, step=step,
            system_metrics=system_metrics,
            thread_id=self.thread_id,
        )

    def log_metrics(
        self, metrics: Dict[str, Any],
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Log multiple metrics into the forum thread.

        Parameters
        ----------
        system_metrics:
            Hardware stats to attach — any of:
            ``"cpu"``, ``"ram"``, ``"gpu"``, ``"disk"``, ``"network"``.
        """
        self._metrics.update(metrics)
        if step is not None:
            self._steps = max(self._steps, step)
        self._tracker.log_metrics(
            metrics, step=step,
            system_metrics=system_metrics,
            thread_id=self.thread_id,
        )

    def set_tag(self, key: str, value: Any) -> None:
        self._tags[key] = value
        self._tracker.set_tag(key, value, thread_id=self.thread_id)

    def log_artifact(self, file_path: str) -> None:
        self._artifacts.append(file_path)
        self._tracker.log_artifact(file_path, thread_id=self.thread_id)

    def log_text(self, text: str, filename: str = "output.txt") -> None:
        self._tracker.log_text(text, filename, thread_id=self.thread_id)

    def log_figure(self, fig, title: str = "Training Graph") -> None:
        """Send a matplotlib figure as a PNG into this forum thread."""
        self._tracker.log_figure(fig, title=title, thread_id=self.thread_id)

    # ------------------------------------------------------------------
    # Context manager — summary goes into the thread
    # ------------------------------------------------------------------

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._status  = "FAILED"
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._tracker._post_run_summary(self, error_text=tb_text, thread_id=self.thread_id)
            return False
        self._status = "FINISHED"
        self._tracker._post_run_summary(self, thread_id=self.thread_id)
        return False

    def __repr__(self) -> str:
        return f"<ForumActiveRun name={self.run_name!r} thread={self.thread_id!r} status={self._status}>"
