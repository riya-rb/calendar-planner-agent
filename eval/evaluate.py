"""Evaluation script for the calendar planner agent."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil.parser import isoparse

from agent.models import Event, UserPreferences, load_calendar, load_ground_truth
from agent.parser import parse_task_strategy_a, parse_task_strategy_b, parse_tasks_batch
from agent.scheduler import baseline_fcfs_scheduler, detect_conflict, schedule_event
from config import GROUND_TRUTH_PATH, SYNTHETIC_CALENDAR_PATH

RESULTS_PATH = Path("eval/results.json")
STRATEGY_RESULTS_PATH = Path("eval/strategy_comparison.json")
PREFERENCE_WINDOWS = {
    "morning": (8, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
}


def run_evaluation() -> Dict[str, Any]:
    """Run all evaluation scenarios and return aggregate metrics."""
    scenarios = load_ground_truth(str(GROUND_TRUTH_PATH))
    base_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    base_event_map = {event.id: event for event in base_events}
    prefs = UserPreferences()

    task_inputs = [scenario["task_input"] for scenario in scenarios]
    parsed_tasks = parse_tasks_batch(task_inputs)
    if len(parsed_tasks) != len(scenarios):
        raise ValueError(
            "Evaluation aborted: parse_tasks_batch did not return one parsed task per scenario."
        )

    agent_rows: List[Dict[str, Any]] = []
    fcfs_rows: List[Dict[str, Any]] = []

    for scenario, parsed_task in zip(scenarios, parsed_tasks):
        agent_rows.append(_evaluate_single_scenario(scenario, parsed_task, base_event_map, prefs, use_fcfs=False))
        fcfs_rows.append(_evaluate_single_scenario(scenario, parsed_task, base_event_map, prefs, use_fcfs=True))

    results = {
        "timestamp": datetime.now().isoformat(),
        "scenario_count": len(scenarios),
        "agent": _aggregate_metrics(agent_rows),
        "fcfs_baseline": _aggregate_metrics(fcfs_rows),
        "scenario_details": {
            "agent": agent_rows,
            "fcfs_baseline": fcfs_rows,
        },
    }
    strategy_comparison = _run_strategy_comparison(scenarios)
    results["strategy_comparison"] = strategy_comparison

    _print_summary_table(results)
    _save_results(results)
    _save_strategy_comparison(strategy_comparison)
    return results


def _evaluate_single_scenario(
    scenario: Dict[str, Any],
    parsed_task: Any,
    base_event_map: Dict[str, Event],
    prefs: UserPreferences,
    use_fcfs: bool,
) -> Dict[str, Any]:
    expected = scenario.get("expected", {})
    scenario_events = [deepcopy(base_event_map[event_id]) for event_id in scenario["existing_events"]]
    shifted_events, shift_delta = _shift_events_for_current_date(scenario_events)
    original_events = [deepcopy(event) for event in shifted_events]

    task = deepcopy(parsed_task)
    expected_deadline = _expected_deadline_datetime(expected)
    if expected_deadline is not None:
        task.deadline = expected_deadline + shift_delta

    scheduled_event = _schedule_task(shifted_events, task, prefs, use_fcfs)
    duration_match = False
    conflict_free = False
    before_deadline = False
    preference_match: Optional[bool] = None

    if scheduled_event is not None:
        duration_match = abs(_duration_hours(scheduled_event) - float(expected.get("duration_hours", task.duration_hours))) < 1e-9
        conflict_free = _conflict_free(scheduled_event, original_events, expected.get("must_not_conflict_with", []))
        before_deadline = scheduled_event.end <= _effective_deadline(task.deadline)

        expected_preference = expected.get("preferred_time")
        if expected_preference is not None:
            preference_match = _matches_preference_window(scheduled_event, expected_preference)

    scheduling_pass = scheduled_event is not None and duration_match and conflict_free and before_deadline

    return {
        "scenario_id": scenario["scenario_id"],
        "scheduled": scheduled_event is not None,
        "scheduled_event": _serialize_event(scheduled_event),
        "scheduling_pass": scheduling_pass,
        "conflict_free": conflict_free,
        "before_deadline": before_deadline,
        "duration_match": duration_match,
        "preference_applicable": preference_match is not None,
        "preference_match": preference_match,
        "mode": "fcfs_baseline" if use_fcfs else "agent",
    }


def _schedule_task(
    events: List[Event],
    task: Any,
    prefs: UserPreferences,
    use_fcfs: bool,
) -> Optional[Event]:
    if use_fcfs:
        scheduled_events = baseline_fcfs_scheduler([task], events, prefs)
        return scheduled_events[0] if scheduled_events else None
    return schedule_event(events, task, prefs)


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    accuracy_passes = sum(1 for row in rows if row["scheduling_pass"])
    conflict_free_passes = sum(1 for row in rows if row["conflict_free"])

    preference_rows = [row for row in rows if row["preference_applicable"]]
    preference_matches = sum(1 for row in preference_rows if row["preference_match"])

    return {
        "scheduling_accuracy": _pct(accuracy_passes, total),
        "preference_satisfaction": _pct(preference_matches, len(preference_rows)),
        "conflict_free_rate": _pct(conflict_free_passes, total),
        "counts": {
            "accuracy_passes": accuracy_passes,
            "total": total,
            "preference_matches": preference_matches,
            "preference_total": len(preference_rows),
            "conflict_free_passes": conflict_free_passes,
        },
    }


def _print_summary_table(results: Dict[str, Any]) -> None:
    agent = results["agent"]
    fcfs = results["fcfs_baseline"]

    print("Metric                    | Agent  | FCFS baseline")
    print("--------------------------------------------------")
    print(
        f"{'Scheduling accuracy':<25} | "
        f"{_fmt_pct(agent['scheduling_accuracy']):<6} | "
        f"{_fmt_pct(fcfs['scheduling_accuracy'])}"
    )
    print(
        f"{'Preference satisfaction':<25} | "
        f"{_fmt_pct(agent['preference_satisfaction']):<6} | "
        f"{_fmt_pct(fcfs['preference_satisfaction'])}"
    )
    print(
        f"{'Conflict-free rate':<25} | "
        f"{_fmt_pct(agent['conflict_free_rate']):<6} | "
        f"{_fmt_pct(fcfs['conflict_free_rate'])}"
    )


def _save_results(results: Dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _run_strategy_comparison(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    def parser_accuracy(parsed_task: Any, expected: Dict[str, Any]) -> Dict[str, bool]:
        expected_deadline_raw = expected.get("deadline")
        deadline_match = False
        if expected_deadline_raw is not None:
            expected_deadline = _coerce_deadline_to_end_of_day(expected_deadline_raw)
            deadline_delta_days = abs((parsed_task.deadline - expected_deadline).total_seconds()) / 86400
            deadline_match = deadline_delta_days <= 1

        duration_match = abs(parsed_task.duration_hours - float(expected.get("duration_hours", 0.0))) <= 0.25

        expected_priority = expected.get("priority")
        acceptable_priorities = expected.get("valid_slot_constraints", {}).get("acceptable_priorities")
        if acceptable_priorities:
            priority_match = parsed_task.priority in acceptable_priorities
        else:
            priority_match = parsed_task.priority == expected_priority

        preferred_time_match = parsed_task.preferred_time == expected.get("preferred_time")

        overall_match = deadline_match and duration_match and priority_match and preferred_time_match
        return {
            "priority_match": priority_match,
            "duration_match": duration_match,
            "deadline_match": deadline_match,
            "preferred_time_match": preferred_time_match,
            "overall_match": overall_match,
        }

    def evaluate_strategy(
        strategy_name: str,
        parse_fn: Any,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        ambiguous_rows: List[Dict[str, Any]] = []

        for scenario in scenarios:
            expected = scenario.get("expected", {})
            try:
                parsed_task = parse_fn(scenario["task_input"])
                accuracy = parser_accuracy(parsed_task, expected)
                row = {
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario.get("category"),
                    "parsed": {
                        "title": parsed_task.title,
                        "deadline": parsed_task.deadline.isoformat(),
                        "duration_hours": parsed_task.duration_hours,
                        "priority": parsed_task.priority,
                        "preferred_time": parsed_task.preferred_time,
                    },
                    **accuracy,
                    "error": None,
                }
            except Exception as error:
                row = {
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario.get("category"),
                    "parsed": None,
                    "priority_match": False,
                    "duration_match": False,
                    "deadline_match": False,
                    "preferred_time_match": False,
                    "overall_match": False,
                    "error": str(error),
                }

            rows.append(row)
            if scenario.get("category") == "Parser stress tests":
                ambiguous_rows.append(row)

        return {
            "name": strategy_name,
            "priority_accuracy": _pct(sum(1 for row in rows if row["priority_match"]), len(rows)),
            "duration_accuracy": _pct(sum(1 for row in rows if row["duration_match"]), len(rows)),
            "deadline_accuracy": _pct(sum(1 for row in rows if row["deadline_match"]), len(rows)),
            "overall_accuracy": _pct(sum(1 for row in rows if row["overall_match"]), len(rows)),
            "ambiguous_input_accuracy": _pct(
                sum(1 for row in ambiguous_rows if row["overall_match"]),
                len(ambiguous_rows),
            ),
            "rows": rows,
        }

    strategy_a = evaluate_strategy("Strategy A", parse_task_strategy_a)
    strategy_b = evaluate_strategy("Strategy B", parse_task_strategy_b)

    print()
    print("Parser strategy comparison")
    print("Field          | Strategy A | Strategy B")
    print("----------------------------------------")
    print(
        f"{'Priority acc.':<14} | "
        f"{_fmt_pct(strategy_a['priority_accuracy']):<10} | "
        f"{_fmt_pct(strategy_b['priority_accuracy'])}"
    )
    print(
        f"{'Duration acc.':<14} | "
        f"{_fmt_pct(strategy_a['duration_accuracy']):<10} | "
        f"{_fmt_pct(strategy_b['duration_accuracy'])}"
    )
    print(
        f"{'Deadline acc.':<14} | "
        f"{_fmt_pct(strategy_a['deadline_accuracy']):<10} | "
        f"{_fmt_pct(strategy_b['deadline_accuracy'])}"
    )
    print(
        f"{'Overall':<14} | "
        f"{_fmt_pct(strategy_a['overall_accuracy']):<10} | "
        f"{_fmt_pct(strategy_b['overall_accuracy'])}"
    )
    print(
        f"{'Ambiguous cat.':<14} | "
        f"{_fmt_pct(strategy_a['ambiguous_input_accuracy']):<10} | "
        f"{_fmt_pct(strategy_b['ambiguous_input_accuracy'])}"
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "scenario_count": len(scenarios),
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
    }


def _save_strategy_comparison(results: Dict[str, Any]) -> None:
    STRATEGY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _shift_events_for_current_date(events: List[Event]) -> Tuple[List[Event], timedelta]:
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


def _effective_deadline(deadline: datetime) -> datetime:
    if deadline.time() == time(0, 0):
        return datetime.combine(deadline.date(), time(23, 59, 59))
    return deadline


def _duration_hours(event: Event) -> float:
    return (event.end - event.start).total_seconds() / 3600


def _conflict_free(event: Event, original_events: List[Event], blocked_ids: List[str]) -> bool:
    for blocked_id in blocked_ids:
        blocked_event = next((candidate for candidate in original_events if candidate.id == blocked_id), None)
        if blocked_event is not None and detect_conflict([blocked_event], event.start, event.end):
            return False
    return True


def _matches_preference_window(event: Event, preferred_time: str) -> bool:
    if preferred_time not in PREFERENCE_WINDOWS:
        return False

    start_hour, end_hour = PREFERENCE_WINDOWS[preferred_time]
    slot_time = event.start.time()
    window_start = time(start_hour, 0)
    window_end = time(end_hour, 0)
    return window_start <= slot_time < window_end


def _serialize_event(event: Optional[Event]) -> Optional[Dict[str, Any]]:
    if event is None:
        return None

    return {
        "id": event.id,
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "recurring": event.recurring,
    }


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


if __name__ == "__main__":
    run_evaluation()
