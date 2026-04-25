"""Scheduling engine for calendar planning tasks."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple

from agent.models import Event, Task, UserPreferences, load_calendar
from config import SYNTHETIC_CALENDAR_PATH

LOGGER = logging.getLogger(__name__)

SLOT_STEP = timedelta(minutes=30)
PREFERRED_START_HOURS = {
    "morning": 8,
    "afternoon": 13,
    "evening": 17,
}


def get_events(events: List[Event], date_range: Tuple[datetime, datetime]) -> List[Event]:
    """Return events that overlap the provided date range, sorted by start time."""
    range_start, range_end = date_range
    matching_events = [
        event
        for event in events
        if not (event.end < range_start or event.start > range_end)
    ]
    return sorted(matching_events, key=lambda event: event.start)


def detect_conflict(events: List[Event], start: datetime, end: datetime) -> bool:
    """Return True if any existing event overlaps with the proposed interval."""
    return any(not (end <= event.start or start >= event.end) for event in events)


def check_availability(
    events: List[Event],
    target_date: datetime,
    duration_hours: float,
    prefs: UserPreferences,
) -> Optional[Tuple[datetime, datetime]]:
    """Find the first available slot on a given day using 30-minute increments."""
    duration = timedelta(hours=duration_hours)
    target_day = target_date.date()
    day_events = get_events(
        events,
        (
            datetime.combine(target_day, time.min),
            datetime.combine(target_day, time.max),
        ),
    )

    search_start_hour = _get_search_start_hour(prefs)
    day_start = datetime.combine(target_day, time(hour=max(prefs.work_start_hour, search_start_hour)))
    day_end = datetime.combine(target_day, time(hour=prefs.work_end_hour))

    current_start = day_start
    while current_start + duration <= day_end:
        current_end = current_start + duration
        if not detect_conflict(day_events, current_start, current_end):
            return current_start, current_end
        current_start += SLOT_STEP

    return None


def schedule_event(events: List[Event], task: Task, prefs: UserPreferences) -> Optional[Event]:
    """Schedule a task between today and its deadline using simple priority rules."""
    today = datetime.now().date()
    deadline_day = task.deadline.date()

    if deadline_day < today:
        LOGGER.warning("Could not schedule task %r: deadline %s has already passed.", task.title, deadline_day)
        return None

    effective_prefs = _merge_task_preferences(task, prefs)
    candidate_days = _build_candidate_days(today, deadline_day, task.priority)

    for candidate_day in candidate_days:
        slot = check_availability(
            events=events,
            target_date=datetime.combine(candidate_day, time.min),
            duration_hours=task.duration_hours,
            prefs=effective_prefs,
        )
        if slot is None:
            continue

        slot_start, slot_end = slot
        if slot_end > task.deadline:
            continue

        scheduled_event = Event(
            id=_generate_event_id(events),
            title=task.title,
            start=slot_start,
            end=slot_end,
            recurring=False,
        )
        events.append(scheduled_event)
        return scheduled_event

    LOGGER.warning("Could not schedule task %r before deadline %s.", task.title, task.deadline.isoformat())
    return None


def baseline_fcfs_scheduler(
    tasks: List[Task],
    events: List[Event],
    prefs: UserPreferences,
) -> List[Event]:
    """Schedule tasks in arrival order without any priority-based date ordering."""
    scheduled_events: List[Event] = []
    today = datetime.now().date()

    for task in tasks:
        effective_prefs = _merge_task_preferences(task, prefs)
        scheduled_event: Optional[Event] = None

        for candidate_day in _daterange(today, task.deadline.date()):
            slot = check_availability(
                events=events,
                target_date=datetime.combine(candidate_day, time.min),
                duration_hours=task.duration_hours,
                prefs=effective_prefs,
            )
            if slot is None:
                continue

            slot_start, slot_end = slot
            if slot_end > task.deadline:
                continue

            scheduled_event = Event(
                id=_generate_event_id(events),
                title=task.title,
                start=slot_start,
                end=slot_end,
                recurring=False,
            )
            events.append(scheduled_event)
            scheduled_events.append(scheduled_event)
            break

        if scheduled_event is None:
            LOGGER.warning("FCFS scheduler could not place task %r before deadline %s.", task.title, task.deadline)

    return scheduled_events


def _get_search_start_hour(prefs: UserPreferences) -> int:
    preferred_time = prefs.preferred_time if prefs.preferred_time in PREFERRED_START_HOURS else None
    if preferred_time is None:
        return prefs.work_start_hour
    return PREFERRED_START_HOURS[preferred_time]


def _merge_task_preferences(task: Task, prefs: UserPreferences) -> UserPreferences:
    if task.preferred_time is None:
        return prefs
    return replace(prefs, preferred_time=task.preferred_time)


def _build_candidate_days(start_day: date, end_day: date, priority: str) -> List[date]:
    days = list(_daterange(start_day, end_day))
    if priority.lower() == "high":
        return list(reversed(days))
    return days


def _daterange(start_day: date, end_day: date) -> List[date]:
    if end_day < start_day:
        return []
    total_days = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(total_days + 1)]


def _generate_event_id(events: List[Event]) -> str:
    used_ids = {event.id for event in events}
    next_index = len(events) + 1

    while True:
        candidate = f"scheduled_{next_index}"
        if candidate not in used_ids:
            return candidate
        next_index += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scheduled_calendar = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    preferences = UserPreferences()
    test_tasks = [
        Task(
            title="Finish problem set",
            deadline=datetime(2026, 4, 27, 20, 0),
            duration_hours=2.0,
            priority="high",
            preferred_time="morning",
        ),
        Task(
            title="Prepare research summary",
            deadline=datetime(2026, 4, 28, 18, 0),
            duration_hours=1.5,
            priority="medium",
            preferred_time=None,
        ),
        Task(
            title="Plan presentation outline",
            deadline=datetime(2026, 4, 29, 21, 0),
            duration_hours=1.0,
            priority="low",
            preferred_time="evening",
        ),
    ]

    for task in test_tasks:
        result = schedule_event(scheduled_calendar, task, preferences)
        if result is None:
            print(f"Unable to schedule: {task.title}")
        else:
            print(
                f"{result.title}: "
                f"{result.start.isoformat()} -> {result.end.isoformat()} "
                f"(id={result.id})"
            )
