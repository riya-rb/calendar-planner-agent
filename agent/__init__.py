"""Calendar planner agent package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Event",
    "Task",
    "UserPreferences",
    "load_calendar",
    "load_ground_truth",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import models

        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
