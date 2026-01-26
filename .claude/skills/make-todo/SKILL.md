---
name: make-todo
description: Convert requirements into an agent-executable task backlog.
---

# ROLE
Technical planner. Convert requirements into an exhaustive agent-executable task backlog.

# INPUT
A feature request, PRD, or unstructured development idea.

# OUTPUT
A `TODO.md` file with:

1. **Completion Criteria:** Checkboxes for the overall goals.
2. **Tasks:** Atomic units of work.

# TASK FORMAT

#### TASK-XXX: [Short Title]
- **Status:** [TODO|IN_PROGRESS|BLOCKED|DONE]
- **Done When:** [Exact command that returns exit code 0 on success]
- **Context:** [Files likely involved—agent uses this + claude.md to orient]
- **Notes:** [Leave empty; agent fills this with errors/observations]

# RULES
1. Tasks must be completable in a single focused session.
2. Every task MUST have a concrete, scriptable `Done When`.
3. Do NOT include research or investigation tasks—those are human tasks.
4. Order tasks by likely dependency, but note that the agent may reorder.
5. If the input is vague, state assumptions before the task list.
6. Prefer tasks that result in a passing test over "code exists."
7. If a task spans multiple files, list all relevant files in `Context`.