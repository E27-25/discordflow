"""Core DiscordFlow tracker — the main entry point for users."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

from .exceptions import WebhookError, ArtifactTooLargeError
from .utils import (
    COLOR_BLUE, COLOR_GREEN, COLOR_PURPLE, COLOR_RED,
    COLOR_GOLD, COLOR_GRAY, COLOR_TEAL, COLOR_ORANGE,
    format_kv_table, human_size, human_duration,
    truncate, collect_system_metrics,
)
from .run import ActiveRun, ForumActiveRun

# Discord limits
_MAX_ARTIFACT_BYTES = 25 * 1024 * 1024  # 25 MB
_EMBED_FIELD_LIMIT  = 25


class DiscordFlow:
    """
    Lightweight ML experiment tracker that posts to a Discord webhook.

    Supports two distinct channel modes — use the correct ``start_*`` method
    for your channel type:

    * **Normal channel** → :meth:`start_run`
    * **Forum channel**  → :meth:`start_forum_run`

    Parameters
    ----------
    webhook_url:
        Your Discord webhook URL.
    experiment_name:
        Human-readable experiment name shown in every embed.
    state_file:
        Path to a JSON file used to persist ``{run_name: thread_id}`` state.
        Pass ``None`` to disable persistence (default: ``".discordflow_state.json"``).
    async_logging:
        If ``True`` (default), webhook calls are dispatched on a background
        thread so they never block your training loop.
        Call :meth:`finish` before program exit to flush pending messages.
    dry_run:
        If ``True``, print embeds to stdout instead of calling the webhook.
    username:
        Override the bot display name shown in Discord.
    avatar_url:
        URL to a custom avatar image for the bot.

    Examples
    --------
    Normal channel::

        dflow = DiscordFlow(WEBHOOK_URL, "ResNet_Experiment")
        with dflow.start_run("baseline") as run:
            run.log_params({"lr": 3e-4})
            run.log_metrics({"loss": 0.5}, step=1, system_metrics=["cpu", "ram"])

    Forum channel::

        dflow = DiscordFlow(FORUM_WEBHOOK_URL, "ResNet_Experiment")
        with dflow.start_forum_run("baseline") as run:
            run.log_metrics({"loss": 0.5}, step=1)
        # Each run lives in its own Discord forum thread ↑
    """

    def __init__(
        self,
        webhook_url: str,
        experiment_name: str = "Default Experiment",
        state_file: Optional[str] = ".discordflow_state.json",
        async_logging: bool = True,
        dry_run: bool = False,
        username: str = "DiscordFlow 🤖",
        avatar_url: Optional[str] = None,
    ):
        self.webhook_url     = webhook_url
        self.experiment_name = experiment_name
        self.state_file      = state_file
        self.dry_run         = dry_run
        self._username       = username
        self._avatar_url     = avatar_url
        self._async_logging  = async_logging
        self._executor       = ThreadPoolExecutor(max_workers=2) if async_logging else None

        # Load persisted run state {run_name: thread_id}
        self._run_state: Dict[str, str] = self._load_state()

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
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, str]:
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Persist the current ``{run_name: thread_id}`` state to a JSON file.

        Call this right after :meth:`start_forum_run` so you can resume
        threads after a Colab runtime restart.

        Parameters
        ----------
        filepath:
            Override the default ``state_file`` path for this save only.
        """
        target = filepath or self.state_file
        if not target:
            print("[DiscordFlow] ⚠  No state_file configured — state not saved.")
            return
        with open(target, "w") as f:
            json.dump(self._run_state, f, indent=2)
        abs_path = os.path.abspath(target)
        print(f"💾 [DiscordFlow] Session state saved to: {abs_path}")

    # ------------------------------------------------------------------
    # Normal channel — start_run()
    # ------------------------------------------------------------------

    def start_run(self, run_name: str = "Run") -> ActiveRun:
        """
        Start a named run on a **normal text/announcement channel**.

        Posts all embeds (params, metrics, summary) directly into the channel.
        Use :meth:`start_forum_run` if your webhook points to a Forum channel.

        Parameters
        ----------
        run_name:
            Label for this run, shown in all embeds.

        Returns
        -------
        ActiveRun
            Use as a context manager — auto-posts a summary on exit.

        Raises
        ------
        DiscordFlowModeError
            If called on a webhook that belongs to a Forum channel.
            Switch to :meth:`start_forum_run` in that case.

        Example
        -------
        >>> with dflow.start_run("sweep_01") as run:
        ...     run.log_params({"lr": 3e-4})
        ...     run.log_metrics({"loss": 0.5}, step=1)
        """
        run = ActiveRun(tracker=self, run_name=run_name)
        self._send_embed({
            "title": f"▶️  Run Started: `{run_name}`",
            "color": COLOR_TEAL,
            "description": (
                "📌 **Channel mode:** Normal channel\n"
                "Logs will be posted directly into this channel."
            ),
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
            ],
        })
        return run

    # ------------------------------------------------------------------
    # Forum channel — start_forum_run()
    # ------------------------------------------------------------------

    def start_forum_run(
        self,
        run_name: str = "Run",
        description: str = "New training run.",
    ) -> "ForumActiveRun":
        """
        Start a named run on a **Discord Forum channel**.

        Creates a new forum thread for this run on first call; on subsequent
        calls with the same ``run_name`` (e.g. after a Colab restart) it
        **resumes the existing thread** if the thread ID is in the saved state.

        Use :meth:`start_run` if your webhook points to a normal text channel.

        Parameters
        ----------
        run_name:
            Label for this run — also used as the forum thread name.
        description:
            Short description shown in the thread's first post.

        Returns
        -------
        ForumActiveRun
            Use as a context manager — auto-posts a summary on exit.

        Raises
        ------
        WebhookError
            If Discord returns an error. A common cause is using a normal-
            channel webhook URL — switch to :meth:`start_run` in that case.

        Example
        -------
        >>> with dflow.start_forum_run("lora_rank_16", "LoRA sweep r=16") as run:
        ...     run.log_metrics({"loss": 0.4}, step=1)
        # A dedicated forum thread is created/resumed automatically ↑
        """
        # --- Resume existing thread if we have its ID ---
        if run_name in self._run_state:
            thread_id = self._run_state[run_name]
            print(f"🔄 [DiscordFlow] Resuming forum thread '{run_name}' (ID: {thread_id})")
            run = ForumActiveRun(tracker=self, run_name=run_name, thread_id=thread_id)
            self._send_to_thread(thread_id, {
                "title": f"🔄 Run Resumed: `{run_name}`",
                "color": COLOR_TEAL,
                "description": f"Experiment: **{self.experiment_name}**",
            })
            return run

        # --- Create a new forum thread ---
        if self.dry_run:
            fake_thread_id = "DRY_RUN_THREAD_0000"
            print(f"[DiscordFlow][DRY RUN] Would create forum thread: '{run_name}' → {fake_thread_id}")
            self._run_state[run_name] = fake_thread_id
            return ForumActiveRun(tracker=self, run_name=run_name, thread_id=fake_thread_id)

        url = f"{self.webhook_url}?wait=true"
        embed = {
            "title": f"🚀 Run Initialized: `{run_name}`",
            "description": description,
            "color": COLOR_TEAL,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
                {"name": "Mode", "value": "📋 Forum Thread", "inline": True},
            ],
        }
        payload = {
            "thread_name": run_name,
            "username": self._username,
            "embeds": [embed],
        }
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        try:
            resp = requests.post(url, json=payload, timeout=15)
        except requests.exceptions.RequestException as exc:
            raise WebhookError(
                f"Failed to create forum thread for '{run_name}': {exc}\n"
                "👉 If your webhook is a normal channel, use start_run() instead."
            ) from exc

        if resp.status_code not in (200, 201):
            raise WebhookError(
                f"Discord returned {resp.status_code} when creating forum thread.\n"
                f"Response: {resp.text}\n"
                "👉 If your webhook is a normal channel, use start_run() instead."
            )

        data      = resp.json()
        thread_id = str(data.get("channel_id", ""))
        if not thread_id:
            raise WebhookError(f"Discord response missing channel_id: {data}")

        self._run_state[run_name] = thread_id
        print(f"✅ [DiscordFlow] Forum thread created: '{run_name}' → thread {thread_id}")
        print("⚠️  Call logger.save() to persist this thread ID across restarts!")

        return ForumActiveRun(tracker=self, run_name=run_name, thread_id=thread_id)

    def resume_run(self, run_name: str, thread_id: str) -> None:
        """
        Manually link a run name to an existing forum thread ID.

        Use this if you know the thread ID but don't have a saved state file.

        Parameters
        ----------
        run_name:
            The run name to link.
        thread_id:
            The Discord forum thread channel ID.

        Example
        -------
        >>> dflow.resume_run("lora_rank_16", "1234567890")
        >>> dflow.save()   # persist so you don't need to do this again
        """
        self._run_state[run_name] = str(thread_id)
        print(f"🔗 [DiscordFlow] Manually linked '{run_name}' → thread {thread_id}")
        print("⚠️  Call logger.save() to persist this!")

    # ------------------------------------------------------------------
    # Logging APIs (work for both modes)
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self._send_embed({
            "title": "⚙️  Parameter Logged",
            "color": COLOR_BLUE,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
                {"name": str(key), "value": truncate(str(value)), "inline": True},
            ],
        })

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters in a single embed."""
        fields = [
            {"name": str(k), "value": truncate(str(v)), "inline": True}
            for k, v in list(params.items())[:_EMBED_FIELD_LIMIT]
        ]
        self._send_embed({
            "title": f"⚙️  Parameters — {len(params)} logged",
            "color": COLOR_BLUE,
            "fields": fields,
        })

    def log_metric(
        self,
        key: str,
        value: Any,
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
    ) -> None:
        """Log a single metric value."""
        fields = [
            {"name": "Experiment", "value": self.experiment_name, "inline": True},
            {"name": str(key), "value": str(value), "inline": True},
        ]
        if step is not None:
            fields.append({"name": "Step", "value": str(step), "inline": True})
        if system_metrics:
            fields += self._system_metric_fields(system_metrics)

        self._send_embed({"title": "📈  Metric Update", "color": COLOR_GREEN, "fields": fields})

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        system_metrics: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """
        Log multiple metrics in a single embed.

        Parameters
        ----------
        metrics:
            Dict of ``{metric_name: value}`` to log.
        step:
            Current training step or epoch.
        system_metrics:
            List of hardware stats to include. Any combination of:
            ``"cpu"``, ``"ram"``, ``"gpu"``, ``"disk"``, ``"network"``.

            Examples::

                # Only CPU and RAM
                run.log_metrics({"loss": 0.4}, step=1, system_metrics=["cpu", "ram"])

                # All available hardware stats
                run.log_metrics({"loss": 0.4}, step=1, system_metrics=["cpu", "ram", "gpu", "disk", "network"])
        thread_id:
            Forum thread ID (set automatically by ForumActiveRun).
        """
        fields = [
            {"name": str(k), "value": str(v), "inline": True}
            for k, v in list(metrics.items())[:_EMBED_FIELD_LIMIT - 2]
        ]
        if step is not None:
            fields.append({"name": "Step / Epoch", "value": str(step), "inline": False})
        if system_metrics:
            fields += self._system_metric_fields(system_metrics)

        embed = {"title": "📈  Metrics Update", "color": COLOR_GREEN, "fields": fields}

        if thread_id:
            self._send_to_thread(thread_id, embed)
        else:
            self._send_embed(embed)

    def set_tag(self, key: str, value: Any, thread_id: Optional[str] = None) -> None:
        """Attach an arbitrary tag to the run."""
        embed = {
            "title": "🏷️  Tag Set",
            "color": COLOR_PURPLE,
            "fields": [
                {"name": "Experiment", "value": self.experiment_name, "inline": True},
                {"name": str(key), "value": truncate(str(value)), "inline": True},
            ],
        }
        if thread_id:
            self._send_to_thread(thread_id, embed)
        else:
            self._send_embed(embed)

    def log_artifact(self, file_path: str, thread_id: Optional[str] = None) -> None:
        """Upload a file artifact (max 25 MB)."""
        if not os.path.exists(file_path):
            print(f"[DiscordFlow] ⚠  File not found: {file_path}")
            return

        size = os.path.getsize(file_path)
        if size > _MAX_ARTIFACT_BYTES:
            raise ArtifactTooLargeError(file_path, size)

        file_name = os.path.basename(file_path)
        caption   = f"📁  **Artifact:** `{file_name}`  |  `{human_size(size)}`  |  `{self.experiment_name}`"

        if self.dry_run:
            print(f"[DiscordFlow][DRY RUN] Would upload artifact: {file_path} ({human_size(size)})")
            return

        target_url = (
            f"{self.webhook_url}?thread_id={thread_id}" if thread_id else self.webhook_url
        )
        with open(file_path, "rb") as fh:
            self._post_with_file(target_url, caption, file_name, fh)

    def log_text(self, text: str, filename: str = "output.txt", thread_id: Optional[str] = None) -> None:
        """Upload a text snippet as a file artifact."""
        caption    = f"📄  **Text artifact:** `{filename}`  |  `{self.experiment_name}`"
        target_url = (
            f"{self.webhook_url}?thread_id={thread_id}" if thread_id else self.webhook_url
        )

        if self.dry_run:
            print(f"[DiscordFlow][DRY RUN] Would upload text as: {filename}")
            print(text[:200] + ("..." if len(text) > 200 else ""))
            return

        buf = io.BytesIO(text.encode("utf-8"))
        self._post_with_file(target_url, caption, filename, buf)

    def log_figure(
        self,
        fig,
        title: str = "Training Graph",
        thread_id: Optional[str] = None,
    ) -> None:
        """
        Send a ``matplotlib`` figure as a PNG attachment.

        Parameters
        ----------
        fig:
            A ``matplotlib.figure.Figure`` instance.
        title:
            Title shown in the Discord embed above the image.
        thread_id:
            Forum thread ID (set automatically by ForumActiveRun).

        Example
        -------
        >>> fig, ax = plt.subplots()
        >>> ax.plot(losses)
        >>> run.log_figure(fig, title="Loss Curve")
        """
        target_url = (
            f"{self.webhook_url}?thread_id={thread_id}" if thread_id else self.webhook_url
        )

        if self.dry_run:
            print(f"[DiscordFlow][DRY RUN] Would upload figure: '{title}' as plot.png")
            return

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)

        embed        = {"title": f"📈 {title}", "color": COLOR_GREEN, "image": {"url": "attachment://plot.png"}}
        payload_json = {"username": self._username, "embeds": [embed]}
        if self._avatar_url:
            payload_json["avatar_url"] = self._avatar_url

        data  = {"payload_json": __import__("json").dumps(payload_json)}
        files = {"file": ("plot.png", buf, "image/png")}

        def _do_post():
            try:
                resp = requests.post(target_url, data=data, files=files, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                print(f"[DiscordFlow] ⚠  Failed to upload figure: {exc}", file=sys.stderr)

        if self._async_logging and self._executor:
            self._executor.submit(_do_post)
        else:
            _do_post()

    def finish(self) -> None:
        """
        Flush all pending async webhook calls and shut down the background executor.

        Always call this at the end of your script / notebook to ensure
        all metrics have been delivered to Discord before exit.

        Example
        -------
        >>> logger.finish()
        ✅ [DiscordFlow] All logs synced.
        """
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        print("✅ [DiscordFlow] All logs synced.")

    def end_run(self, status: str = "FINISHED") -> None:
        """Explicitly close the current run and post a status embed."""
        self._send_embed({
            "title": "⏹  Run Ended",
            "color": COLOR_GRAY,
            "fields": [{"name": "Status", "value": status, "inline": True}],
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_metric_fields(self, metrics: List[str]) -> list:
        """Collect hardware stats and return as Discord embed fields."""
        hw = collect_system_metrics(metrics)
        return [
            {"name": label, "value": truncate(str(val), 256), "inline": True}
            for label, val in hw.items()
        ]

    def _send_embed(self, embed: dict, thread_id: Optional[str] = None) -> None:
        """Send an embed to the channel (or a specific thread).

        Automatically retries on 429 Too Many Requests, respecting the
        ``Retry-After`` header from Discord (up to 3 retries).
        """
        if "footer" not in embed:
            embed["footer"] = {"text": f"DiscordFlow • {self.experiment_name}"}

        if self.dry_run:
            self._dry_run_print(embed)
            return

        url     = f"{self.webhook_url}?thread_id={thread_id}" if thread_id else self.webhook_url
        payload = {"embeds": [embed], "username": self._username}
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        def _post():
            for attempt in range(4):   # up to 3 retries
                try:
                    resp = requests.post(url, json=payload, timeout=10)
                    if resp.status_code == 429:
                        # Respect Discord's Retry-After (seconds, possibly float)
                        retry_after = float(resp.headers.get("Retry-After", 2.0))
                        print(
                            f"[DiscordFlow] ⏳  Rate-limited — waiting {retry_after:.1f}s "
                            f"(attempt {attempt + 1}/3)",
                            file=sys.stderr,
                        )
                        time.sleep(retry_after + 0.1)
                        continue   # retry
                    resp.raise_for_status()
                    return  # success
                except requests.exceptions.RequestException as exc:
                    if attempt == 3:
                        print(f"[DiscordFlow] ⚠  Failed to send embed: {exc}", file=sys.stderr)
                    else:
                        time.sleep(0.5 * (attempt + 1))

        if self._async_logging and self._executor:
            self._executor.submit(_post)
        else:
            _post()

    def _send_to_thread(self, thread_id: str, embed: dict) -> None:
        """Send an embed into a specific forum thread."""
        self._send_embed(embed, thread_id=thread_id)

    def _post_with_file(self, url: str, caption: str, filename: str, file_obj) -> None:
        """Upload a file to Discord, retrying on 429."""
        payload = {"content": caption, "username": self._username}
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url
        for attempt in range(4):
            try:
                resp = requests.post(
                    url, files={"file": (filename, file_obj)}, data=payload, timeout=30
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    print(
                        f"[DiscordFlow] ⏳  Rate-limited on file upload — waiting {retry_after:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(retry_after + 0.1)
                    file_obj.seek(0)   # rewind before retry
                    continue
                resp.raise_for_status()
                return  # success
            except requests.exceptions.RequestException as exc:
                if attempt == 3:
                    raise WebhookError(f"Failed to upload file: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))

    def _post_run_summary(
        self, run: "ActiveRun", error_text: Optional[str] = None, thread_id: Optional[str] = None
    ) -> None:
        """Post a run-summary embed when a run ends."""
        elapsed = run.elapsed
        success = run._status == "FINISHED"
        fields: list = []

        if run._params:
            fields.append({"name": "⚙️  Parameters", "value": truncate(format_kv_table(run._params)), "inline": False})
        if run._metrics:
            fields.append({"name": "📈  Final Metrics", "value": truncate(format_kv_table(run._metrics)), "inline": False})
        if run._tags:
            fields.append({"name": "🏷️  Tags", "value": truncate(format_kv_table(run._tags)), "inline": False})
        if run._artifacts:
            fields.append({"name": "📁  Artifacts", "value": "\n".join(f"`{os.path.basename(a)}`" for a in run._artifacts), "inline": False})
        if error_text:
            fields.append({"name": "❌  Error", "value": truncate(f"```\n{error_text}\n```"), "inline": False})

        fields.append({"name": "⏱️  Elapsed", "value": human_duration(elapsed), "inline": True})
        if run._steps:
            fields.append({"name": "👣  Steps", "value": str(run._steps), "inline": True})
        fields.append({"name": "📌  Status", "value": "✅ FINISHED" if success else "❌ FAILED", "inline": True})

        embed = {
            "title": f"{'✅' if success else '❌'}  Run Complete: `{run.run_name}`",
            "description": f"Experiment: **{self.experiment_name}**",
            "color": COLOR_TEAL if success else COLOR_RED,
            "fields": fields,
        }
        if thread_id:
            self._send_to_thread(thread_id, embed)
        else:
            self._send_embed(embed)

    @staticmethod
    def _dry_run_print(embed: dict) -> None:
        title  = embed.get("title", "")
        desc   = embed.get("description", "")
        fields = embed.get("fields", [])
        footer = embed.get("footer", {}).get("text", "")
        lines  = ["─" * 50, f"  {title}"]
        if desc:
            lines.append(f"  {desc}")
        for f in fields:
            lines.append(f"  {f['name']}: {f['value']}")
        if footer:
            lines.append(f"  [{footer}]")
        lines.append("─" * 50)
        print("\n".join(lines))