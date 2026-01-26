---
name: ralph-singleton
description: Study SPEC.md, pick a task from TODO.md, implement it fully following the workflow, commit when done.
---

# ROLE
Autonomous development agent. Pick the highest priority task from TODO.md and implement it completely following project conventions.

# INPUT
None required. Agent self-directs by reading SPEC.md and TODO.md.

# OUTPUT
1. Completed implementation of one task from TODO.md
2. Updated TODO.md with task status changes
3. Updated DEVLOG.md with implementation notes and learnings
4. Git commit with descriptive message
5. New TODO tasks if gaps in SPEC are identified

# WORKFLOW

## 1. Study Context
- Read SPEC.md to understand project requirements and architecture
- Read CLAUDE.md to understand workflow and conventions
- Read TODO.md to see all available tasks

## 2. Select Task
- Pick the highest priority task with status TODO or IN_PROGRESS
- Prefer tasks from earlier phases (Phase 1 before Phase 2)
- Skip BLOCKED tasks
- Update task status to IN_PROGRESS in TODO.md

## 3. Implement Task
- Follow the "Done When" criteria exactly
- Adhere to conventions in CLAUDE.md:
  - Use Pydantic for configuration
  - Use Polars for data processing
  - Use Rich for CLI output
  - Include error handling in all code
  - Use aggregated logging (not piecemeal)
- Write clean, type-hinted code
- Add docstrings (Google style) to all public functions
- Stay within task scope - don't expand features

## 4. Verify Completion
- Run the "Done When" command from the task
- Ensure exit code is 0 (success)
- Run linting: `uv run ruff check src/`
- Fix any issues found

## 5. Document Changes
- Update TODO.md:
  - Change task status to DONE
  - Add notes about implementation decisions
  - Add blocking information if relevant
- Append to DEVLOG.md:
  - Date and task ID
  - What was implemented
  - Key decisions made
  - Any learnings or gotchas
  - Link to commit

## 6. Identify Spec Gaps
- If you discover missing requirements or inconsistencies
- Add new tasks to TODO.md backlog
- Note the gap in DEVLOG.md
- Include reference to source of the gap

## 7. Commit
- Stage relevant files (implementation + TODO.md + DEVLOG.md)
- Write descriptive commit message:
  - Start with task ID: "TASK-XXX: Brief description"
  - Include what was implemented
  - End with: "Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
- Commit to current branch (migration/ghcnh-parquet-architecture)

# RULES
1. Work on ONLY ONE task at a time until completely done
2. Do NOT start a new task until current one is verified and committed
3. If blocked, update task status to BLOCKED with reason in Notes
4. After 2 failed attempts, stop and report findings in task Notes
5. Never skip the DEVLOG.md update
6. Never skip the commit step
7. Ensure "Done When" command passes before marking DONE
8. If task requires changes to SPEC.md, note this in DEVLOG and create a TODO task for spec update
9. Use TODO.md format exactly (don't change structure)
10. When stuck, ask human for clarification rather than making speculative assumptions

# CONFIDENCE MARKERS
Include in task Notes when making assumptions:
- **KNOWN:** Fact from SPEC.md, documentation, or verified by test
- **INFERRED:** Reasonable assumption based on code patterns
- **SPECULATIVE:** Guess that may be wrong

# ERROR HANDLING
- If "Done When" command fails, capture error output in task Notes
- If linting fails, fix issues before committing
- If unsure about implementation approach, note alternatives in task Notes
- If task reveals it's blocked by another task, update status and dependencies

# COMPLETION SIGNAL
When task is complete, output:
```
✓ TASK-XXX: [Title]
  - Implementation: [brief summary]
  - Files changed: [list]
  - Verified: [Done When command]
  - Committed: [commit hash]
  - DEVLOG updated
```
