"""Core data models and data-loading helpers for the calendar planner agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from dateutil.parser import isoparse


@dataclass
class Event:
    id: str
    title: str
    start: datetime
    end: datetime
    recurring: bool


@dataclass
class Task:
    title: str
    deadline: datetime
    duration_hours: float
    priority: str
    preferred_time: Optional[str]


@dataclass
class UserPreferences:
    work_start_hour: int = 8
    work_end_hour: int = 22
    sleep_end_hour: int = 7
    preferred_time: str = "morning"


def load_calendar(path: str) -> List[Event]:
    """Load calendar events from a JSON file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        raw_events: List[dict[str, Any]] = json.load(file_obj)

    return [
        Event(
            id=event["id"],
            title=event["title"],
            start=isoparse(event["start"]),
            end=isoparse(event["end"]),
            recurring=event["recurring"],
        )
        for event in raw_events
    ]


def load_ground_truth(path: str) -> list[Any]:
    """Load ground-truth scheduling scenarios from a JSON file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, SYNTHETIC_CALENDAR_PATH

    calendar_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    ground_truth_cases = load_ground_truth(str(GROUND_TRUTH_PATH))

    print(f"Loaded {len(calendar_events)} calendar events from {SYNTHETIC_CALENDAR_PATH.name}.")
    print(f"Loaded {len(ground_truth_cases)} ground-truth scenarios from {GROUND_TRUTH_PATH.name}.")
