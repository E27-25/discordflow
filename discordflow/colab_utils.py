"""
discordflow.colab_utils — Google Colab session management helpers.

These utilities let you save and restore DiscordFlow's run state (the
``{run_name: thread_id}`` mapping) across Colab runtime restarts, using a
ZIP file that you can download to your local machine and re-upload when needed.

Usage::

    from discordflow.colab_utils import export_session, import_session

    # After training — download backup
    export_session(logger)

    # On a fresh Colab runtime — restore state
    import_session(logger)
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discordflow.core import DiscordFlow


def export_session(
    logger: "DiscordFlow",
    zip_name: str = "discordflow_backup.zip",
) -> None:
    """
    Save the logger's current run state to a JSON file, zip it, and
    download the ZIP to your local machine (Google Colab only).

    Parameters
    ----------
    logger:
        The :class:`~discordflow.DiscordFlow` instance whose state to export.
    zip_name:
        Filename for the downloaded ZIP (default: ``discordflow_backup.zip``).

    Example
    -------
    >>> from discordflow.colab_utils import export_session
    >>> export_session(logger)
    💾 Saved state to .discordflow_state.json
    📦 Created discordflow_backup.zip. Downloading...
    """
    # First, save the current state
    logger.save()

    state_file = logger.state_file
    if not os.path.exists(state_file):
        print(f"⚠️  No state file found at '{os.path.abspath(state_file)}'.")
        return

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(state_file, arcname=os.path.basename(state_file))

    print(f"📦 Created {zip_name}. Downloading...")

    try:
        from google.colab import files as colab_files  # type: ignore
        colab_files.download(zip_name)
    except ImportError:
        abs_path = os.path.abspath(zip_name)
        print(f"ℹ️  Not running in Colab. ZIP saved locally at: {abs_path}")


def import_session(
    logger: "DiscordFlow",
) -> None:
    """
    Prompt a file upload in Google Colab, extract the ZIP, and restore
    the logger's run state from the contained JSON file.

    Parameters
    ----------
    logger:
        The :class:`~discordflow.DiscordFlow` instance to restore into.

    Example
    -------
    >>> from discordflow.colab_utils import import_session
    >>> import_session(logger)
    📤 Please upload your 'discordflow_backup.zip'...
    ✅ Session restored! 2 run(s) loaded.
    """
    try:
        from google.colab import files as colab_files  # type: ignore
    except ImportError:
        print("⚠️  import_session() only works inside Google Colab.")
        return

    print("📤 Please upload your 'discordflow_backup.zip'...")
    uploaded = colab_files.upload()

    for filename in uploaded:
        if filename.endswith(".zip"):
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(filename, "r") as zf:
                zf.extractall(".")

            # Reload state into logger
            state_file = logger.state_file
            if os.path.exists(state_file):
                with open(state_file) as f:
                    logger._run_state = json.load(f)
                count = len(logger._run_state)
                print(f"✅ Session restored! {count} run(s) loaded.")
                for name, tid in logger._run_state.items():
                    print(f"   • '{name}' → thread {tid}")
            else:
                print("⚠️  ZIP extracted but no state file found inside.")
            return

    print("⚠️  No .zip file was uploaded.")
