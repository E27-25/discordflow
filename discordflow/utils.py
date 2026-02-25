"""Formatting utilities for DiscordFlow embeds and output."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Discord embed colour palette
# ---------------------------------------------------------------------------
COLOR_BLUE   = 0x3498DB   # params
COLOR_GREEN  = 0x57F287   # metrics
COLOR_PURPLE = 0x9B59B6   # tags
COLOR_RED    = 0xED4245   # errors / failed run
COLOR_GOLD   = 0xFEE75C   # artifacts
COLOR_GRAY   = 0x95A5A6   # info / run start
COLOR_TEAL   = 0x1ABC9C   # run finished successfully


# ---------------------------------------------------------------------------
# Human-readable helpers
# ---------------------------------------------------------------------------

def human_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (e.g. '1.2 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def human_duration(seconds: float) -> str:
    """Convert seconds to a compact human-readable duration."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {sec}s"


# ---------------------------------------------------------------------------
# ASCII / text widgets
# ---------------------------------------------------------------------------

def ascii_progress(value: float, total: float, width: int = 20) -> str:
    """Return a filled ASCII progress bar, e.g.  ████████████░░░░░░░░  60%"""
    if total == 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / total))
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(ratio * 100)
    return f"{bar}  {pct}%"


def format_kv_table(data: dict, max_value_len: int = 50) -> str:
    """
    Format a dict as a compact ``key: value`` code block suitable for a
    Discord embed field value (max 1024 chars per Discord limits).
    """
    lines = []
    for k, v in data.items():
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len - 3] + "..."
        lines.append(f"{k}: {v_str}")
    return "```\n" + "\n".join(lines) + "\n```"


def truncate(text: str, limit: int = 1024) -> str:
    """Truncate text to Discord embed field value limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
