"""Utilities for turning raw event data into normalized records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List


def parse_events(raw_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize event dictionaries into a consistent shape."""
    parsed_events: List[Dict[str, Any]] = []

    for event in raw_events:
        parsed_events.append(
            {
                "title": event["title"],
                "start": datetime.fromisoformat(event["start"]),
                "end": datetime.fromisoformat(event["end"]),
                "priority": event.get("priority", "medium"),
            }
        )

    return parsed_events

