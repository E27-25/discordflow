"""
DiscordFlow — Lightweight ML experiment tracker for Discord webhooks.

Normal channel::

    from discordflow import DiscordFlow

    dflow = DiscordFlow("WEBHOOK_URL", "MyExperiment")
    with dflow.start_run("baseline") as run:
        run.log_params({"lr": 3e-4})
        run.log_metrics({"loss": 0.5}, step=1, system_metrics=["cpu", "ram"])

Forum channel::

    dflow = DiscordFlow("FORUM_WEBHOOK_URL", "MyExperiment")
    with dflow.start_forum_run("baseline") as run:
        run.log_metrics({"loss": 0.5}, step=1, system_metrics=["cpu", "ram", "gpu"])
        run.log_figure(fig, title="Loss Curve")
    dflow.save()   # persist thread IDs across Colab restarts
"""

from .core import DiscordFlow
from .run import ActiveRun, ForumActiveRun
from .exceptions import (
    DiscordFlowError,
    WebhookError,
    ArtifactTooLargeError,
    RunNotActiveError,
)

__version__ = "0.3.3"
__author__  = "Watin Promfiy"
__license__ = "MIT"

__all__ = [
    # Main class
    "DiscordFlow",
    # Run context managers
    "ActiveRun",
    "ForumActiveRun",
    # Exceptions
    "DiscordFlowError",
    "WebhookError",
    "ArtifactTooLargeError",
    "RunNotActiveError",
    # Colab utilities (imported on demand)
    # from discordflow.colab_utils import export_session, import_session
]