---
description: TAMNet paper-audit todo — show status, or pass an instruction like "mark phase 1 leakage done"
---

The todo list lives at `docs/TODO.md` (TAMNet paper-audit checklist: Phases 0-6, the 13-row headline-claim table, known contradictions, and FILLs).

Recent git state:

!`git log --oneline -5`

If no argument was given: report the checklist status compactly — for each phase, what is done vs pending — then state the single next action and anything blocked.

If an argument was given: $ARGUMENTS — apply it to the checklist (mark items done, add items, re-prioritize), edit `docs/TODO.md` accordingly, and confirm the change.

Rules: never mark a claim verified unless its number traces to a file under `results/` or `outputs/`; otherwise it stays `TODO-RERUN`.
