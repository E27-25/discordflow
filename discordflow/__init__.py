"""
DiscordFlow — Lightweight ML experiment tracker for Discord webhooks.

Quick start::

    from discordflow import DiscordFlow

    dflow = DiscordFlow("YOUR_WEBHOOK_URL", experiment_name="my_experiment")
    dflow.log_param("lr", 3e-4)
    dflow.log_metrics({"loss": 0.42, "acc": 0.91}, step=1)

Context-manager pattern::

    with dflow.start_run("sweep_01") as run:
        run.log_params({"lr": 3e-4, "batch": 32})
        run.log_metrics({"loss": 0.3}, step=5)
"""

from .core import DiscordFlow
from .run import ActiveRun
from .exceptions import (
    DiscordFlowError,
    WebhookError,
    ArtifactTooLargeError,
    RunNotActiveError,
)

__version__ = "0.2.0"
__author__  = "DiscordFlow Contributors"
__license__ = "MIT"

__all__ = [
    "DiscordFlow",
    "ActiveRun",
    "DiscordFlowError",
    "WebhookError",
    "ArtifactTooLargeError",
    "RunNotActiveError",
]