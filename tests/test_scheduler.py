from datetime import datetime

from agent.scheduler import find_conflicts


def test_find_conflicts_returns_overlapping_events():
    events = [
        {
            "title": "Deep work",
            "start": datetime(2026, 4, 20, 9, 0),
            "end": datetime(2026, 4, 20, 10, 0),
        },
        {
            "title": "Client call",
            "start": datetime(2026, 4, 20, 9, 30),
            "end": datetime(2026, 4, 20, 10, 30),
        },
    ]

    conflicts = find_conflicts(events)

    assert len(conflicts) == 1
    assert conflicts[0]["title"] == "Client call"

