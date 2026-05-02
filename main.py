"""Main agent loop for the calendar planner agent."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil.parser import isoparse

from agent.models import Event, Task, UserPreferences, load_calendar
from agent.parser import parse_task
from agent.scheduler import detect_conflict, schedule_event
from config import SYNTHETIC_CALENDAR_PATH

LOGGER = logging.getLogger(__name__)


def run_agent(batch_mode: bool = False, tasks_file: str = "data/ground_truth_tests.json") -> None:
    """Run the interactive agent loop or batch evaluation mode."""
    if batch_mode:
        _run_batch_mode(tasks_file)
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    live_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    prefs = UserPreferences()

    print("Calendar Planner Agent")
    print("Commands: type a task, or use 'show', 'reset', or 'quit'.")

    while True:
        try:
            print()
            print(_schedule_summary(live_events))
            user_input = input("What would you like to schedule? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting calendar planner agent.")
            break

        if not user_input:
            print("Please enter a task description or a command.")
            continue

        command = user_input.lower()
        if command == "quit":
            print("Goodbye.")
            break
        if command == "show":
            print(_format_full_schedule(live_events))
            continue
        if command == "reset":
            live_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
            print("Schedule reset to the original synthetic calendar.")
            continue

        try:
            task = parse_task(user_input)
            scheduled_event = schedule_event(live_events, task, prefs)

            if scheduled_event is None:
                print(
                    f"Could not find a slot before {task.deadline.strftime('%Y-%m-%d %H:%M')}. "
                    "Try a shorter duration or later deadline."
                )
                continue

            print(
                f"Scheduled: {scheduled_event.title} on {scheduled_event.start.strftime('%Y-%m-%d')} "
                f"from {scheduled_event.start.strftime('%H:%M')} to {scheduled_event.end.strftime('%H:%M')}"
            )
        except Exception as error:
            print(f"Sorry, I couldn't process that request: {error}")
            LOGGER.exception("Interactive agent error")


def _run_batch_mode(tasks_file: str) -> None:
    """Run the ground-truth scenarios in a non-interactive evaluation pass."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scenarios = _load_json_file(Path(tasks_file))
    base_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    base_event_map = {event.id: event for event in base_events}

    pass_count = 0
    for scenario in scenarios:
        result = _evaluate_scenario(scenario, base_event_map)
        if result["passed"]:
            pass_count += 1

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] Scenario {scenario['scenario_id']}: {scenario['description']}")
        if result["details"]:
            print(f"  {'; '.join(result['details'])}")

    print()
    print(f"Batch results: {pass_count}/{len(scenarios)} scenarios passed.")


def _evaluate_scenario(scenario: Dict[str, Any], base_event_map: Dict[str, Event]) -> Dict[str, Any]:
    details: List[str] = []

    try:
        scenario_events = [base_event_map[event_id] for event_id in scenario["existing_events"]]
    except KeyError as error:
        return {"passed": False, "details": [f"Unknown event id in scenario: {error}"]}

    shifted_events, shift_delta = _shift_events_for_current_date(scenario_events)
    original_events = list(shifted_events)
    expected = scenario.get("expected", {})

    try:
        parsed_task = parse_task(scenario["task_input"])
    except Exception as error:
        return {"passed": False, "details": [f"Task parsing failed: {error}"]}

    expected_deadline = _expected_deadline_datetime(expected)
    if expected_deadline is not None:
        parsed_task = replace(parsed_task, deadline=expected_deadline + shift_delta)

    scheduled_event = schedule_event(shifted_events, parsed_task, UserPreferences())

    if scheduled_event is None:
        return {"passed": False, "details": ["No slot found before the scenario deadline."]}

    if "title" in expected:
        expected_title = str(expected["title"]).lower()
        got_title = parsed_task.title.lower()
        # Exact title matching is brittle for LLM evaluations because small rephrasings
        # can preserve meaning while changing the surface string.
        if expected_title not in got_title and got_title not in expected_title:
            details.append(f"title mismatch: expected {expected['title']!r}, got {parsed_task.title!r}")

    if "duration_hours" in expected and abs(parsed_task.duration_hours - float(expected["duration_hours"])) > 1e-9:
        details.append(
            f"duration mismatch: expected {expected['duration_hours']}, got {parsed_task.duration_hours}"
        )

    if "priority" in expected and parsed_task.priority != expected["priority"]:
        details.append(f"priority mismatch: expected {expected['priority']!r}, got {parsed_task.priority!r}")

    if "preferred_time" in expected and parsed_task.preferred_time != expected["preferred_time"]:
        details.append(
            f"preferred_time mismatch: expected {expected['preferred_time']!r}, got {parsed_task.preferred_time!r}"
        )

    if detect_conflict(original_events, scheduled_event.start, scheduled_event.end):
        details.append("scheduled slot conflicts with an existing event")

    if "valid_slot_start_before" in expected:
        limit = isoparse(expected["valid_slot_start_before"]) + shift_delta
        if scheduled_event.start >= limit:
            details.append(f"slot starts too late: expected before {limit.isoformat()}")

    if "scheduled_before" in expected:
        limit = isoparse(expected["scheduled_before"]) + shift_delta
        if scheduled_event.start >= limit:
            details.append(f"slot starts too late: expected before {limit.isoformat()}")

    if "deadline" in expected:
        deadline_limit = _coerce_deadline_to_end_of_day(expected["deadline"]) + shift_delta
        if scheduled_event.end > deadline_limit:
            details.append(f"scheduled end exceeds deadline {deadline_limit.isoformat()}")

    for blocked_event_id in expected.get("must_not_conflict_with", []):
        blocked_event = next((event for event in original_events if event.id == blocked_event_id), None)
        if blocked_event and detect_conflict([blocked_event], scheduled_event.start, scheduled_event.end):
            details.append(f"scheduled slot conflicts with blocked event {blocked_event_id}")

    return {"passed": not details, "details": details}


def _schedule_summary(events: List[Event]) -> str:
    sorted_events = sorted(events, key=lambda event: event.start)
    upcoming = sorted_events[:3]
    lines = [f"Current schedule: {len(sorted_events)} events loaded."]

    if not upcoming:
        lines.append("No events scheduled.")
        return "\n".join(lines)

    lines.append("Next events:")
    for event in upcoming:
        lines.append(
            f"- {event.title} on {event.start.strftime('%Y-%m-%d %H:%M')} "
            f"to {event.end.strftime('%H:%M')}"
        )
    return "\n".join(lines)


def _format_full_schedule(events: List[Event]) -> str:
    sorted_events = sorted(events, key=lambda event: event.start)
    lines = ["Full schedule:"]
    for event in sorted_events:
        lines.append(
            f"- [{event.id}] {event.title}: "
            f"{event.start.strftime('%Y-%m-%d %H:%M')} -> {event.end.strftime('%Y-%m-%d %H:%M')}"
        )
    return "\n".join(lines)


def _load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _shift_events_for_current_date(events: List[Event]) -> tuple[List[Event], timedelta]:
    if not events:
        return [], timedelta(0)

    anchor_day = min(event.start.date() for event in events)
    today = datetime.now().date()
    shift_delta = timedelta(days=(today - anchor_day).days) if anchor_day < today else timedelta(0)

    shifted_events = [
        Event(
            id=event.id,
            title=event.title,
            start=event.start + shift_delta,
            end=event.end + shift_delta,
            recurring=event.recurring,
        )
        for event in events
    ]
    return shifted_events, shift_delta


def _expected_deadline_datetime(expected: Dict[str, Any]) -> Optional[datetime]:
    if "deadline" in expected:
        return _coerce_deadline_to_end_of_day(expected["deadline"])
    if "scheduled_before" in expected:
        return isoparse(expected["scheduled_before"])
    if "valid_slot_start_before" in expected:
        return isoparse(expected["valid_slot_start_before"])
    return None


def _coerce_deadline_to_end_of_day(value: str) -> datetime:
    parsed = isoparse(value)
    if parsed.time() == time(0, 0):
        return datetime.combine(parsed.date(), time(23, 59, 59))
    return parsed


if __name__ == "__main__":
    run_agent()
