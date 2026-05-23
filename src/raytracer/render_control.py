from __future__ import annotations


class RenderCancelled(RuntimeError):
    """Raised when a caller requests render cancellation between work chunks."""

