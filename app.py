"""Streamlit front-end for the calendar planner agent."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dateutil.parser import isoparse

from agent.models import Event, UserPreferences, load_calendar, load_ground_truth
from agent.parser import parse_task, parse_tasks_batch
from agent.scheduler import baseline_fcfs_scheduler, detect_conflict, schedule_event
from config import GROUND_TRUTH_PATH, SYNTHETIC_CALENDAR_PATH

PREFERENCE_WINDOWS = {
    "morning": (8, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
}


def _load_initial_events() -> List[Event]:
    return load_calendar(str(SYNTHETIC_CALENDAR_PATH))


def _ensure_session_state() -> None:
    if "live_events" not in st.session_state:
        st.session_state.live_events = _load_initial_events()
    if "last_parsed_task" not in st.session_state:
        st.session_state.last_parsed_task = None
    if "last_scheduled_event" not in st.session_state:
        st.session_state.last_scheduled_event = None
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None


def _event_rows(events: List[Event]) -> List[Dict[str, str]]:
    sorted_events = sorted(events, key=lambda event: event.start)
    return [
        {
            "Title": event.title,
            "Date": event.start.strftime("%Y-%m-%d"),
            "Start": event.start.strftime("%H:%M"),
            "End": event.end.strftime("%H:%M"),
        }
        for event in sorted_events
    ]


def _render_schedule_tab() -> None:
    st.subheader("Schedule a Task")
    task_input = st.text_input(
        "Enter a task in natural language",
        placeholder="Finish ML assignment by Friday, 2 hours, high priority, prefer mornings",
    )

    schedule_clicked = st.button("Schedule", type="primary")
    if schedule_clicked:
        if not task_input.strip():
            st.error("Please enter a task description before scheduling.")
        else:
            try:
                with st.spinner("Parsing task with the LLM..."):
                    parsed_task = parse_task(task_input)
                st.session_state.last_parsed_task = parsed_task
            except Exception as error:
                st.session_state.last_parsed_task = None
                st.error(f"Sorry, I couldn't parse that task: {error}")
            else:
                try:
                    scheduled_event = schedule_event(
                        st.session_state.live_events,
                        parsed_task,
                        UserPreferences(),
                    )
                    st.session_state.last_scheduled_event = scheduled_event
                except Exception as error:
                    st.session_state.last_scheduled_event = None
                    st.error(f"Sorry, something went wrong while scheduling: {error}")
                else:
                    if scheduled_event is None:
                        st.error("No slot was found before the deadline.")
                    else:
                        st.success(
                            "Scheduled: "
                            f"{scheduled_event.title} on {scheduled_event.start.strftime('%Y-%m-%d')} "
                            f"from {scheduled_event.start.strftime('%H:%M')} "
                            f"to {scheduled_event.end.strftime('%H:%M')}"
                        )

    if st.session_state.last_parsed_task is not None:
        task = st.session_state.last_parsed_task
        st.info(
            "\n".join(
                [
                    f"Title: {task.title}",
                    f"Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}",
                    f"Duration (hours): {task.duration_hours}",
                    f"Priority: {task.priority}",
                    f"Preferred time: {task.preferred_time or 'None'}",
                ]
            )
        )

    if st.button("Reset calendar"):
        st.session_state.live_events = _load_initial_events()
        st.session_state.last_parsed_task = None
        st.session_state.last_scheduled_event = None
        st.success("Calendar reset to the synthetic baseline.")

    st.markdown("### Current calendar")
    st.dataframe(_event_rows(st.session_state.live_events), use_container_width=True)


def _render_evaluation_tab() -> None:
    st.subheader("Evaluation")

    if st.button("Run evaluation"):
        try:
            with st.spinner("Loading scenarios and running evaluation..."):
                scenarios = load_ground_truth(str(GROUND_TRUTH_PATH))
                progress = st.progress(0)
                progress.progress(10)

                task_inputs = [scenario["task_input"] for scenario in scenarios]
                parsed_tasks = parse_tasks_batch(task_inputs)
                progress.progress(40)

                parsed_by_input = {task_input: parsed for task_input, parsed in zip(task_inputs, parsed_tasks)}
                task_records = []
                for index, scenario in enumerate(scenarios):
                    parsed_task_obj = parsed_by_input.get(scenario["task_input"])
                    if parsed_task_obj is None:
                        try:
                            parsed_task_obj = parse_task(scenario["task_input"])
                        except Exception as error:
                            task_records.append(
                                {
                                    "scenario": scenario,
                                    "parsed_task": None,
                                    "parse_error": str(error),
                                }
                            )
                            progress.progress(40 + int(((index + 1) / max(len(scenarios), 1)) * 20))
                            continue

                    task_records.append(
                        {
                            "scenario": scenario,
                            "parsed_task": parsed_task_obj,
                            "parse_error": None,
                        }
                    )
                    progress.progress(40 + int(((index + 1) / max(len(scenarios), 1)) * 20))

                agent_rows: List[Dict[str, Any]] = []
                fcfs_rows: List[Dict[str, Any]] = []

                for index, record in enumerate(task_records):
                    if record["parsed_task"] is None:
                        failure_row = _parse_failure_row(record["scenario"], record["parse_error"] or "Unknown parse error")
                        agent_rows.append(failure_row)
                        fcfs_rows.append(dict(failure_row, mode="fcfs_baseline"))
                    else:
                        agent_rows.append(
                            _evaluate_single_scenario(
                                record["scenario"],
                                record["parsed_task"],
                                use_fcfs=False,
                            )
                        )
                        fcfs_rows.append(
                            _evaluate_single_scenario(
                                record["scenario"],
                                record["parsed_task"],
                                use_fcfs=True,
                            )
                        )
                    progress.progress(60 + int(((index + 1) / max(len(task_records), 1)) * 40))

                progress.progress(100)
                st.session_state.evaluation_results = {
                    "agent": _aggregate_metrics(agent_rows),
                    "fcfs_baseline": _aggregate_metrics(fcfs_rows),
                    "scenario_details": {
                        "agent": agent_rows,
                        "fcfs_baseline": fcfs_rows,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as error:
            st.error(f"Sorry, evaluation could not be completed: {error}")

    results = st.session_state.evaluation_results
    if not results:
        st.caption("Run evaluation to compare the agent against the FCFS baseline.")
        return

    st.markdown("### Results")
    st.table(
        [
            {
                "Metric": "Scheduling accuracy",
                "Agent": _fmt_pct(results["agent"]["scheduling_accuracy"]),
                "FCFS baseline": _fmt_pct(results["fcfs_baseline"]["scheduling_accuracy"]),
            },
            {
                "Metric": "Preference satisfaction",
                "Agent": _fmt_pct(results["agent"]["preference_satisfaction"]),
                "FCFS baseline": _fmt_pct(results["fcfs_baseline"]["preference_satisfaction"]),
            },
            {
                "Metric": "Conflict-free rate",
                "Agent": _fmt_pct(results["agent"]["conflict_free_rate"]),
                "FCFS baseline": _fmt_pct(results["fcfs_baseline"]["conflict_free_rate"]),
            },
        ]
    )

    st.markdown("### Metric interpretation")
    st.write(
        "Scheduling accuracy measures how often the scheduled slot satisfied the required duration, "
        "avoided blocked events, and finished before the deadline."
    )
    st.write(
        "Preference satisfaction measures how often the scheduled start time landed inside the expected "
        "morning, afternoon, or evening window when a preference was specified."
    )
    st.write(
        "Conflict-free rate measures how often the final scheduled slot avoided the scenario's blocked events, "
        "regardless of the other evaluation checks."
    )

    with st.expander("Individual scenario results", expanded=False):
        for mode_label, rows in (
            ("Agent", results["scenario_details"]["agent"]),
            ("FCFS baseline", results["scenario_details"]["fcfs_baseline"]),
        ):
            st.markdown(f"#### {mode_label}")
            for row in rows:
                status = "PASS" if row["scheduling_pass"] else "FAIL"
                details = []
                if row.get("parse_error"):
                    details.append(f"parse error: {row['parse_error']}")
                else:
                    details.append(f"conflict_free={row['conflict_free']}")
                    details.append(f"before_deadline={row['before_deadline']}")
                    details.append(f"duration_match={row['duration_match']}")
                    if row["preference_applicable"]:
                        details.append(f"preference_match={row['preference_match']}")
                st.write(f"Scenario {row['scenario_id']}: {status} — " + ", ".join(details))


def _parse_failure_row(scenario: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    return {
        "scenario_id": scenario["scenario_id"],
        "scheduled": False,
        "scheduled_event": None,
        "scheduling_pass": False,
        "conflict_free": False,
        "before_deadline": False,
        "duration_match": False,
        "preference_applicable": False,
        "preference_match": None,
        "mode": "agent",
        "parse_error": error_message,
    }


def _evaluate_single_scenario(
    scenario: Dict[str, Any],
    parsed_task: Any,
    use_fcfs: bool,
) -> Dict[str, Any]:
    base_events = load_calendar(str(SYNTHETIC_CALENDAR_PATH))
    base_event_map = {event.id: event for event in base_events}
    prefs = UserPreferences()
    expected = scenario.get("expected", {})
    scenario_events = [deepcopy(base_event_map[event_id]) for event_id in scenario["existing_events"]]
    shifted_events, shift_delta = _shift_events_for_current_date(scenario_events)
    original_events = [deepcopy(event) for event in shifted_events]

    task = deepcopy(parsed_task)
    expected_deadline = _expected_deadline_datetime(expected)
    if expected_deadline is not None:
        task.deadline = expected_deadline + shift_delta

    try:
        scheduled_event = _schedule_task(shifted_events, task, prefs, use_fcfs)
    except Exception as error:
        return {
            "scenario_id": scenario["scenario_id"],
            "scheduled": False,
            "scheduled_event": None,
            "scheduling_pass": False,
            "conflict_free": False,
            "before_deadline": False,
            "duration_match": False,
            "preference_applicable": False,
            "preference_match": None,
            "mode": "fcfs_baseline" if use_fcfs else "agent",
            "parse_error": str(error),
        }

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

    return {
        "scenario_id": scenario["scenario_id"],
        "scheduled": scheduled_event is not None,
        "scheduled_event": _serialize_event(scheduled_event),
        "scheduling_pass": scheduled_event is not None and duration_match and conflict_free and before_deadline,
        "conflict_free": conflict_free,
        "before_deadline": before_deadline,
        "duration_match": duration_match,
        "preference_applicable": preference_match is not None,
        "preference_match": preference_match,
        "mode": "fcfs_baseline" if use_fcfs else "agent",
        "parse_error": None,
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


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(rows)
    preference_rows = [row for row in rows if row["preference_applicable"]]
    return {
        "scheduling_accuracy": _pct(sum(1 for row in rows if row["scheduling_pass"]), total),
        "preference_satisfaction": _pct(
            sum(1 for row in preference_rows if row["preference_match"]),
            len(preference_rows),
        ),
        "conflict_free_rate": _pct(sum(1 for row in rows if row["conflict_free"]), total),
    }


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
    return time(start_hour, 0) <= event.start.time() < time(end_hour, 0)


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


def main() -> None:
    st.set_page_config(page_title="Calendar Planner Agent", page_icon="calendar", layout="wide")
    st.title("Calendar Planner Agent")
    st.caption("Natural-language task scheduling with live calendar updates and evaluation metrics.")

    _ensure_session_state()
    schedule_tab, evaluation_tab = st.tabs(["Schedule a Task", "Evaluation"])

    with schedule_tab:
        _render_schedule_tab()

    with evaluation_tab:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
