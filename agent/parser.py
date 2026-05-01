"""LLM-backed parsing utilities for calendar planning tasks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from dateutil.parser import isoparse
from dotenv import load_dotenv

from agent.models import Task

load_dotenv()

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "llama-3.3-70b-versatile"
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_PREFERRED_TIMES = {"morning", "afternoon", "evening", None}

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


def parse_task(user_input: str) -> Task:
    """Parse a single natural-language task description into a Task."""
    return parse_task_strategy_a(user_input)


def parse_task_strategy_a(user_input: str) -> Task:
    """Strategy A: single-prompt JSON extraction."""
    if not user_input.strip():
        raise ValueError("parse_task_strategy_a failed: user_input must not be empty.")

    client = _build_groq_client()
    initial_response = _request_task_json(client, user_input, _build_strategy_a_system_prompt())

    try:
        payload = _parse_json_response(initial_response)
        return _task_from_payload(payload)
    except ValueError as first_error:
        repaired_response = _request_json_correction(
            client,
            user_input,
            initial_response,
            first_error,
            _build_strategy_a_system_prompt(),
        )

        try:
            payload = _parse_json_response(repaired_response)
            return _task_from_payload(payload)
        except ValueError as second_error:
            raise ValueError(
                "parse_task_strategy_a failed after retry: "
                f"could not convert model output into a valid Task. "
                f"Initial error: {first_error}. Retry error: {second_error}."
            ) from second_error


def parse_task_strategy_b(user_input: str) -> Task:
    """Strategy B: guided step-by-step reasoning with JSON-only final output."""
    if not user_input.strip():
        raise ValueError("parse_task_strategy_b failed: user_input must not be empty.")

    client = _build_groq_client()
    initial_response = _request_task_json(client, user_input, _build_strategy_b_system_prompt())

    try:
        payload = _parse_json_response(initial_response)
        return _task_from_payload(payload)
    except ValueError as first_error:
        repaired_response = _request_json_correction(
            client,
            user_input,
            initial_response,
            first_error,
            _build_strategy_b_system_prompt(),
        )

        try:
            payload = _parse_json_response(repaired_response)
            return _task_from_payload(payload)
        except ValueError as second_error:
            raise ValueError(
                "parse_task_strategy_b failed after retry: "
                f"could not convert model output into a valid Task. "
                f"Initial error: {first_error}. Retry error: {second_error}."
            ) from second_error


def parse_tasks_batch(inputs: List[str]) -> List[Task]:
    """Parse a batch of task descriptions, logging failures and continuing."""
    parsed_tasks: List[Task] = []

    for index, user_input in enumerate(inputs, start=1):
        try:
            parsed_tasks.append(parse_task(user_input))
        except ValueError as error:
            LOGGER.error("Failed to parse task %s: %s | input=%r", index, error, user_input)

    return parsed_tasks


def _build_groq_client() -> Any:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("parse_task failed: GROQ_API_KEY is not set in the environment.")

    try:
        from groq import Groq
    except ImportError as error:
        raise ValueError(
            "parse_task failed: the 'groq' package is not installed. Install it with 'pip install groq'."
        ) from error

    return Groq(api_key=api_key)


def _request_task_json(client: Any, user_input: str, system_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    return _extract_message_content(completion)


def _request_json_correction(
    client: Any,
    user_input: str,
    invalid_response: str,
    error: Exception,
    system_prompt: str,
) -> str:
    correction_prompt = (
        "Your previous response was invalid.\n"
        f"Original user input: {user_input}\n"
        f"Validation error: {error}\n"
        "Return a corrected response as exactly one valid JSON object with keys "
        "title, deadline, duration_hours, priority, preferred_time.\n"
        "Do not include markdown, comments, or extra text."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": invalid_response},
            {"role": "user", "content": correction_prompt},
        ],
        temperature=0,
    )
    return _extract_message_content(completion)


def _extract_message_content(completion: Any) -> str:
    content = completion.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned an empty response.")
    return content.strip()


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model did not return valid JSON: {error.msg}") from error

    if not isinstance(payload, dict):
        raise ValueError("Model JSON response must be an object.")

    return payload


def _build_strategy_a_system_prompt() -> str:
    current_date_iso = datetime.now().date().isoformat()
    return f"""
You extract structured task information for a calendar planner.
Today's date is {current_date_iso}.

Return exactly one valid JSON object and nothing else.
Do not use markdown fences.
Do not add commentary, explanations, or extra keys.

Required JSON schema:
{{
  "title": "string",
  "deadline": "YYYY-MM-DD",
  "duration_hours": 1.5,
  "priority": "high" | "medium" | "low",
  "preferred_time": "morning" | "afternoon" | "evening" | null
}}

Rules:
- Resolve relative dates like "Friday", "tomorrow", and "end of week" using today's date.
- Normalize urgent wording to priority "high".
- Convert vague durations like "a couple hours" to a reasonable float.
- If no preferred time is stated, set preferred_time to null.
- The output must be valid JSON parseable by Python's json.loads.
""".strip()


def _build_strategy_b_system_prompt() -> str:
    current_date_iso = datetime.now().date().isoformat()
    return f"""
You extract structured task information for a calendar planner.
Today's date is {current_date_iso}.

Reason through the task internally using this sequence:
Step 1: Identify the task title.
Step 2: Identify the deadline and convert any relative dates to YYYY-MM-DD.
Step 3: Estimate the duration in hours as a float.
Step 4: Map urgency to one of high, medium, or low.
Step 5: Map any time preference to morning, afternoon, evening, or null.

Do not reveal your step-by-step reasoning.
Return exactly one valid JSON object and nothing else.
Do not use markdown fences.
Do not add commentary, explanations, or extra keys.

Required JSON schema:
{{
  "title": "string",
  "deadline": "YYYY-MM-DD",
  "duration_hours": 1.5,
  "priority": "high" | "medium" | "low",
  "preferred_time": "morning" | "afternoon" | "evening" | null
}}

Rules:
- Resolve relative dates like "Friday", "tomorrow", "before the weekend", and "end of week" using today's date.
- Normalize urgent wording to priority "high".
- Convert vague durations like "a couple hours" to a reasonable float.
- If no preferred time is stated, set preferred_time to null.
- The output must be valid JSON parseable by Python's json.loads.
""".strip()


def _task_from_payload(payload: Dict[str, Any]) -> Task:
    required_keys = {"title", "deadline", "duration_hours", "priority", "preferred_time"}
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

    title = payload["title"]
    deadline_raw = payload["deadline"]
    duration_raw = payload["duration_hours"]
    priority = payload["priority"]
    preferred_time = payload["preferred_time"]

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Field 'title' must be a non-empty string.")

    if not isinstance(deadline_raw, str) or not deadline_raw.strip():
        raise ValueError("Field 'deadline' must be a non-empty ISO date string.")

    try:
        deadline = isoparse(deadline_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Field 'deadline' must be a valid ISO date string.") from error

    try:
        duration_hours = float(duration_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Field 'duration_hours' must be numeric.") from error

    if duration_hours <= 0:
        raise ValueError("Field 'duration_hours' must be greater than zero.")

    if not isinstance(priority, str):
        raise ValueError("Field 'priority' must be a string.")
    priority = priority.lower()
    if priority not in VALID_PRIORITIES:
        raise ValueError("Field 'priority' must be one of: high, medium, low.")

    if preferred_time is not None and not isinstance(preferred_time, str):
        raise ValueError("Field 'preferred_time' must be a string or null.")
    if isinstance(preferred_time, str):
        preferred_time = preferred_time.lower()
    if preferred_time not in VALID_PREFERRED_TIMES:
        raise ValueError("Field 'preferred_time' must be morning, afternoon, evening, or null.")

    return Task(
        title=title.strip(),
        deadline=deadline,
        duration_hours=duration_hours,
        priority=priority,
        preferred_time=preferred_time,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_inputs = [
        "Finish the literature review due Friday, should take 2 hours, medium priority, prefer morning.",
        "Draft my project check-in by end of week, maybe 90 minutes, low priority, evening is best.",
        "Prepare slides for the lab update due 2026-04-30, takes 1.5 hours.",
        "Urgent: submit internship application tomorrow, 45 minutes.",
        "Work on my final project write-up due next Tuesday, a couple hours, high priority, prefer afternoons.",
    ]

    for sample in sample_inputs:
        try:
            parsed_task = parse_task(sample)
            print(asdict(parsed_task))
        except ValueError as error:
            print(f"Failed to parse sample input {sample!r}: {error}")
