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
COLOR_ORANGE = 0xE67E22   # system metrics


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


# ---------------------------------------------------------------------------
# System metrics collector
# ---------------------------------------------------------------------------

# Valid metric keys that users can request
VALID_SYSTEM_METRICS = {"cpu", "ram", "gpu", "disk", "network"}

def collect_system_metrics(metrics: list) -> dict:
    """
    Collect the requested hardware metrics.

    Parameters
    ----------
    metrics:
        A list of metric names to collect. Valid values:
        ``"cpu"``, ``"ram"``, ``"gpu"``, ``"disk"``, ``"network"``.

    Returns
    -------
    dict
        Mapping of metric label → human-readable value string.

    Examples
    --------
    >>> collect_system_metrics(["cpu", "ram"])
    {'🖥️ CPU': '24.3%', '🧠 RAM': '61.2% (9.8 GB used)'}
    >>> collect_system_metrics(["cpu", "ram", "gpu"])
    {'🖥️ CPU': '24.3%', '🧠 RAM': '61.2% (9.8 GB used)', '🎮 GPU': '72% util | 8.1/24.0 GB VRAM'}
    """
    try:
        import psutil
    except ImportError:
        return {"⚠️ System Metrics": "Install psutil: `pip install discordflow[system]`"}

    result = {}
    unknown = [m for m in metrics if m not in VALID_SYSTEM_METRICS]
    if unknown:
        result["⚠️ Unknown metrics"] = ", ".join(unknown)

    for key in metrics:
        key = key.lower()

        if key == "cpu":
            try:
                pct = psutil.cpu_percent(interval=0.1)
                freq = psutil.cpu_freq()
                freq_str = f" @ {freq.current:.0f} MHz" if freq else ""
                result["🖥️ CPU"] = f"{pct}%{freq_str}"
            except Exception as e:
                result["🖥️ CPU"] = f"Error: {e}"

        elif key == "ram":
            try:
                vm = psutil.virtual_memory()
                used_gb  = vm.used  / 1024**3
                total_gb = vm.total / 1024**3
                result["🧠 RAM"] = f"{vm.percent}% ({used_gb:.1f}/{total_gb:.1f} GB)"
            except Exception as e:
                result["🧠 RAM"] = f"Error: {e}"

        elif key == "gpu":
            try:
                import pynvml
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                gpu_parts = []
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    name   = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode()
                    used_gb  = mem.used  / 1024**3
                    total_gb = mem.total / 1024**3
                    gpu_parts.append(
                        f"[{name}] {util.gpu}% util | {used_gb:.1f}/{total_gb:.1f} GB VRAM"
                    )
                pynvml.nvmlShutdown()
                result["🎮 GPU"] = "\n".join(gpu_parts)
            except ImportError:
                result["🎮 GPU"] = "pynvml not installed (`pip install discordflow[gpu]`)"
            except Exception as e:
                result["🎮 GPU"] = f"Not available ({e})"

        elif key == "disk":
            try:
                disk = psutil.disk_usage("/")
                used_gb  = disk.used  / 1024**3
                total_gb = disk.total / 1024**3
                result["💾 Disk"] = f"{disk.percent}% ({used_gb:.1f}/{total_gb:.1f} GB)"
            except Exception as e:
                result["💾 Disk"] = f"Error: {e}"

        elif key == "network":
            try:
                net = psutil.net_io_counters()
                sent_mb = net.bytes_sent / 1024**2
                recv_mb = net.bytes_recv / 1024**2
                result["🌐 Network"] = f"↑ {sent_mb:.1f} MB  ↓ {recv_mb:.1f} MB"
            except Exception as e:
                result["🌐 Network"] = f"Error: {e}"

    return result


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
