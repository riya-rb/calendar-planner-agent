"""Simple scheduling helpers for calendar events."""

from __future__ import annotations

from typing import Dict, Iterable, List


def find_conflicts(events: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return events that overlap with the previous sorted event."""
    sorted_events = sorted(events, key=lambda event: event["start"])
    conflicts: List[Dict[str, object]] = []

    for previous, current in zip(sorted_events, sorted_events[1:]):
        if current["start"] < previous["end"]:
            conflicts.append(current)

    return conflicts

