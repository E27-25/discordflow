"""Core DiscordFlow tracker — the main entry point for users."""

from __future__ import annotations

import io
import os
import sys
import time
import datetime
from typing import Any, Dict, Optional

import requests

from .exceptions import WebhookError, ArtifactTooLargeError
from .utils import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_GOLD,
    COLOR_GRAY,
    COLOR_TEAL,
    format_kv_table,
    human_size,
    human_duration,
    ascii_progress,
    truncate,
)
from .run import ActiveRun

# Discord limits
_MAX_ARTIFACT_BYTES = 25 * 1024 * 1024  # 25 MB
_EMBED_FIELD_LIMIT  = 25                 # max fields per embed


class DiscordFlow:
    """
    Lightweight ML experiment tracker that posts to a Discord webhook.

    Parameters
    ----------
    webhook_url:
        Your Discord webhook URL.
    experiment_name:
        A human-readable name for this experiment (shown in every embed).
    dry_run:
        If ``True``, print messages to stdout instead of calling the webhook.
        Useful for local testing without a real Discord server.
    username:
        Override the webhook bot username shown in Discord.
    avatar_url:
        URL to a custom avatar image for the webhook bot.

    Examples
    --------
    >>> dflow = DiscordFlow("https://discord.com/api/webhooks/...", "ResNet_Training")
    >>> dflow.log_param("lr", 3e-4)
    >>> dflow.log_metrics({"loss": 0.42, "acc": 0.91}, step=1)
    >>> dflow.log_artifact("checkpoint.pt")

    Context-manager pattern::

        with dflow.start_run("sweep_01") as run:
            run.log_params({"lr": 3e-4, "batch": 32})
            for epoch in range(10):
                run.log_metrics({"loss": loss_val}, step=epoch)
    """

    def __init__(
        self,
        webhook_url: str,
        experiment_name: str = "Default Experiment",
        dry_run: bool = False,
        username: str = "DiscordFlow 🤖",
        avatar_url: Optional[str] = None,
    ):
        self.webhook_url    = webhook_url
        self.experiment_name = experiment_name
        self.dry_run        = dry_run
        self._username      = username
        self._avatar_url    = avatar_url
        self._active_run: Optional[ActiveRun] = None

        # Send startup embed
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        self._send_embed({
            "title": "🚀 Experiment Started",
            "description": f"**{experiment_name}**",
            "color": COLOR_GRAY,
            "footer": {"text": f"DiscordFlow • {now}"},
            "fields": [
                {"name": "Python", "value": sys.version.split()[0], "inline": True},
                {"name": "Host",   "value": os.uname().nodename if hasattr(os, "uname") else "N/A", "inline": True},
            ],
        })

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def start_run(self, run_name: str = "Run") -> ActiveRun:
        """
        Start a new named run and return an :class:`ActiveRun` context manager.

        Usage::

            with dflow.start_run("sweep_01") as run:
                run.log_metrics({"loss": 0.5}, step=1)
        """
        run = ActiveRun(tracker=self, run_name=run_name)
        self._active_run = run

        self._send_embed({
            "title": f"▶️  Run Started: `{run_name}`",
            "color": COLOR_TEAL,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
            ],
        })
        return run

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Explicitly end the current run and post a summary.

        Call this if you prefer not to use the context manager.
        """
        if self._active_run is not None:
            self._active_run._status = status
            self._post_run_summary(self._active_run)
            self._active_run = None
        else:
            self._send_embed({
                "title": "⏹  Run Ended",
                "color": COLOR_GRAY,
                "fields": [{"name": "Status", "value": status, "inline": True}],
            })

    # ------------------------------------------------------------------
    # Logging APIs
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self._send_embed({
            "title": "⚙️  Parameter Logged",
            "color": COLOR_BLUE,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
                {"name": str(key), "value": truncate(str(value), 1024), "inline": True},
            ],
        })

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters in a single embed."""
        fields = [
            {"name": str(k), "value": truncate(str(v), 1024), "inline": True}
            for k, v in list(params.items())[:_EMBED_FIELD_LIMIT]
        ]
        self._send_embed({
            "title": f"⚙️  Parameters — {len(params)} logged",
            "color": COLOR_BLUE,
            "fields": fields,
        })

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """Log a single metric value."""
        fields = [
            {"name": "Experiment", "value": self.experiment_name, "inline": True},
            {"name": str(key), "value": str(value), "inline": True},
        ]
        if step is not None:
            fields.append({"name": "Step", "value": str(step), "inline": True})

        self._send_embed({
            "title": "📈  Metric Update",
            "color": COLOR_GREEN,
            "fields": fields,
        })

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log multiple metrics in a single embed."""
        fields = [
            {"name": str(k), "value": str(v), "inline": True}
            for k, v in list(metrics.items())[:_EMBED_FIELD_LIMIT - 1]
        ]
        if step is not None:
            fields.append({"name": "Step / Epoch", "value": str(step), "inline": False})

        self._send_embed({
            "title": "📈  Metrics Update",
            "color": COLOR_GREEN,
            "fields": fields,
        })

    def set_tag(self, key: str, value: Any) -> None:
        """Attach an arbitrary tag to the run."""
        self._send_embed({
            "title": "🏷️  Tag Set",
            "color": COLOR_PURPLE,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
                {"name": str(key), "value": truncate(str(value), 1024), "inline": True},
            ],
        })

    def log_artifact(self, file_path: str) -> None:
        """
        Upload a file artifact to Discord (max 25 MB).

        Parameters
        ----------
        file_path:
            Path to the local file to upload.

        Raises
        ------
        FileNotFoundError:
            If the file does not exist.
        ArtifactTooLargeError:
            If the file exceeds Discord's 25 MB limit.
        """
        if not os.path.exists(file_path):
            print(f"[DiscordFlow] ⚠  File not found: {file_path}")
            return

        size = os.path.getsize(file_path)
        if size > _MAX_ARTIFACT_BYTES:
            raise ArtifactTooLargeError(file_path, size)

        file_name = os.path.basename(file_path)
        caption = (
            f"📁  **Artifact:** `{file_name}`  |  "
            f"`{human_size(size)}`  |  `{self.experiment_name}`"
        )

        if self.dry_run:
            print(f"[DiscordFlow][DRY RUN] Would upload artifact: {file_path} ({human_size(size)})")
            return

        with open(file_path, "rb") as fh:
            try:
                resp = requests.post(
                    self.webhook_url,
                    files={"file": (file_name, fh)},
                    data={
                        "content": caption,
                        "username": self._username,
                        **({"avatar_url": self._avatar_url} if self._avatar_url else {}),
                    },
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise WebhookError(f"Failed to upload artifact: {exc}") from exc

    def log_text(self, text: str, filename: str = "output.txt") -> None:
        """
        Upload a text snippet as a ``.txt`` file artifact.

        Parameters
        ----------
        text:
            The text content to upload.
        filename:
            Filename shown in Discord (default: ``output.txt``).
        """
        caption = f"📄  **Text artifact:** `{filename}`  |  `{self.experiment_name}`"
        if self.dry_run:
            print(f"[DiscordFlow][DRY RUN] Would upload text as: {filename}")
            print("--- content preview ---")
            print(text[:200] + ("..." if len(text) > 200 else ""))
            return

        buf = io.BytesIO(text.encode("utf-8"))
        try:
            resp = requests.post(
                self.webhook_url,
                files={"file": (filename, buf, "text/plain")},
                data={
                    "content": caption,
                    "username": self._username,
                    **({"avatar_url": self._avatar_url} if self._avatar_url else {}),
                },
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise WebhookError(f"Failed to upload text: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_embed(self, embed: dict) -> None:
        """Send an embed (or print to stdout in dry_run mode)."""
        # Always stamp experiment name in footer if not already set
        if "footer" not in embed:
            embed["footer"] = {"text": f"DiscordFlow • {self.experiment_name}"}

        if self.dry_run:
            self._dry_run_print(embed)
            return

        payload: dict = {
            "embeds": [embed],
            "username": self._username,
        }
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            # Don't crash training loops — just warn
            print(f"[DiscordFlow] ⚠  Failed to send embed: {exc}", file=sys.stderr)

    def _post_run_summary(
        self,
        run: ActiveRun,
        error_text: Optional[str] = None,
    ) -> None:
        """Post the final run-summary embed when a run ends."""
        elapsed = run.elapsed
        success = run._status == "FINISHED"

        fields: list = []

        # Params
        if run._params:
            fields.append({
                "name": "⚙️  Parameters",
                "value": truncate(format_kv_table(run._params)),
                "inline": False,
            })

        # Final metrics
        if run._metrics:
            fields.append({
                "name": "📈  Final Metrics",
                "value": truncate(format_kv_table(run._metrics)),
                "inline": False,
            })

        # Tags
        if run._tags:
            fields.append({
                "name": "🏷️  Tags",
                "value": truncate(format_kv_table(run._tags)),
                "inline": False,
            })

        # Artifacts
        if run._artifacts:
            fields.append({
                "name": "📁  Artifacts",
                "value": "\n".join(f"`{os.path.basename(a)}`" for a in run._artifacts),
                "inline": False,
            })

        # Error traceback snippet
        if error_text:
            fields.append({
                "name": "❌  Error",
                "value": truncate(f"```\n{error_text}\n```", 1024),
                "inline": False,
            })

        # Stats row
        fields.append({
            "name": "⏱️  Elapsed",
            "value": human_duration(elapsed),
            "inline": True,
        })
        if run._steps:
            fields.append({
                "name": "👣  Steps",
                "value": str(run._steps),
                "inline": True,
            })
        fields.append({
            "name": "📌  Status",
            "value": "✅ FINISHED" if success else "❌ FAILED",
            "inline": True,
        })

        self._send_embed({
            "title": f"{'✅' if success else '❌'}  Run Complete: `{run.run_name}`",
            "description": f"Experiment: **{self.experiment_name}**",
            "color": COLOR_TEAL if success else COLOR_RED,
            "fields": fields,
        })

    @staticmethod
    def _dry_run_print(embed: dict) -> None:
        """Pretty-print an embed dict to stdout."""
        title   = embed.get("title", "")
        desc    = embed.get("description", "")
        fields  = embed.get("fields", [])
        footer  = embed.get("footer", {}).get("text", "")

        lines = [
            "─" * 50,
            f"  {title}",
        ]
        if desc:
            lines.append(f"  {desc}")
        if fields:
            for f in fields:
                lines.append(f"  {f['name']}: {f['value']}")
        if footer:
            lines.append(f"  [{footer}]")
        lines.append("─" * 50)
        print("\n".join(lines))