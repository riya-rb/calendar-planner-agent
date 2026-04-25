"""Project configuration values."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_CALENDAR_PATH = DATA_DIR / "synthetic_calendar.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth_tests.json"
