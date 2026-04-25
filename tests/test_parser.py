from agent.parser import parse_events


def test_parse_events_returns_datetime_objects():
    events = [
        {
            "title": "Standup",
            "start": "2026-04-20T09:00:00",
            "end": "2026-04-20T09:15:00",
            "priority": "high",
        }
    ]

    parsed = parse_events(events)

    assert parsed[0]["title"] == "Standup"
    assert parsed[0]["start"].isoformat() == "2026-04-20T09:00:00"
    assert parsed[0]["end"].isoformat() == "2026-04-20T09:15:00"
    assert parsed[0]["priority"] == "high"

