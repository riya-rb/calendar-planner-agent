"""Reminder helpers for scheduled events."""

from __future__ import annotations

from typing import Dict


def build_reminder(event: Dict[str, object]) -> str:
    """Create a simple human-readable reminder string."""
    return f"Reminder: {event['title']} starts at {event['start']}"

