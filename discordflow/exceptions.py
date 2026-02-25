class DiscordFlowError(Exception):
    """Base exception for DiscordFlow errors."""
    pass


class WebhookError(DiscordFlowError):
    """Raised when a Discord webhook request fails."""
    pass


class ArtifactTooLargeError(DiscordFlowError):
    """Raised when an artifact exceeds Discord's 25 MB file size limit."""

    MAX_SIZE_MB = 25

    def __init__(self, file_path: str, size_bytes: int):
        size_mb = size_bytes / (1024 * 1024)
        super().__init__(
            f"Artifact '{file_path}' is {size_mb:.1f} MB, "
            f"which exceeds Discord's {self.MAX_SIZE_MB} MB limit."
        )
        self.file_path = file_path
        self.size_bytes = size_bytes


class RunNotActiveError(DiscordFlowError):
    """Raised when an operation is attempted with no active run."""

    def __init__(self):
        super().__init__(
            "No active run. Start a run with dflow.start_run() first, "
            "or use DiscordFlow directly for run-free logging."
        )
