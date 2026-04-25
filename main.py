"""Entry point for the calendar planner agent."""

from __future__ import annotations

import json
from pathlib import Path

from agent.parser import parse_events
from agent.scheduler import find_conflicts


def main() -> None:
    sample_path = Path("data/sample_events.json")
    raw_events = json.loads(sample_path.read_text())
    events = parse_events(raw_events)
    conflicts = find_conflicts(events)

    print(f"Loaded {len(events)} events.")
    print(f"Found {len(conflicts)} conflicting events.")


if __name__ == "__main__":
    main()

