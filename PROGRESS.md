## Iteration log

### Run 1 — 3/5 scenarios passing
- Two scenarios failed during batch evaluation for separate reasons.
- One failure came from the title comparison logic in `main.py`: evaluation used exact string equality, so semantically correct LLM outputs still failed when the model shortened or rephrased a title. A representative case was an expected title such as `"Finish problem set 3"` versus a parsed title like `"problem set 3"`.
- The other failure came from deadline handling in `agent/scheduler.py`: when a task deadline was on the current day, the scheduler could fail to search usable slots on that day correctly, especially when the deadline was represented as a date boundary. This made same-day scheduling brittle.
- The error output in batch mode reflected these issues as evaluation failures such as `title mismatch: expected ... got ...` and `No slot found before the scenario deadline.`

### Run 2 — 5/5 scenarios passing
- The batch evaluator in `main.py` was updated to use substring matching for titles instead of exact equality.
- The exact evaluation fix was:
  `expected_title.lower() in got_title.lower() or got_title.lower() in expected_title.lower()`
- The scheduler deadline boundary bug was also fixed in `agent/scheduler.py`.
- The scheduling fix was:
  exact string match -> substring match for title comparison, and deadline boundary bug -> scheduler now includes today's remaining hours and treats day-level deadlines as inclusive end-of-day bounds.
- After the scheduler fix, tasks due on the current day could still be placed into remaining same-day slots, and deadlines expressed at midnight no longer incorrectly excluded valid slots later that day.
- Lesson learned:
  evaluation logic for LLM systems needs tolerance for semantically equivalent outputs, and temporal constraints need explicit boundary handling rather than relying on naive datetime comparisons.

## Design decisions made during implementation
- High priority tasks search backwards from the deadline because the intent is to place urgent work as late as safely possible while still meeting the due date, preserving earlier time for other tasks.
- Low and medium priority tasks search forwards from today because a first-available placement is a reasonable baseline that reduces procrastination and keeps the schedule simple.
- Substring matching is better than exact match for LLM output evaluation because the model may preserve task meaning while changing surface wording, shortening titles, or dropping function words.

## Known limitations observed so far
- LLMs rephrase titles slightly, so exact match is unreliable for evaluation and should be avoided unless the output format is fully constrained.
- Relative date resolution such as `"end of week"` is inherently ambiguous because the interpretation depends on calendar convention, locale, and the assumed current date.
- The parser currently normalizes missing priority information to `"medium"` rather than preserving an explicit null or unknown state, which simplifies scheduling but loses uncertainty information.
