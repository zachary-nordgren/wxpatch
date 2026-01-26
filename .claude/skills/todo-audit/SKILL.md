---
name: todo-audit
description: Audit TODO.md to verify completed items are actually done, update SPEC if code has drifted.
---

# ROLE
Quality assurance auditor. Verify that tasks marked DONE are actually complete and that SPEC.md reflects current implementation.

# INPUT
None required. Agent reads TODO.md, SPEC.md, source code, and git history.

# OUTPUT
1. Audit report showing verification results for each DONE task
2. New TODO tasks for any incomplete items
3. Updated SPEC.md if code has necessarily drifted from specification
4. Summary of findings

# WORKFLOW

## 1. Collect DONE Tasks
- Read TODO.md
- Extract all tasks with status DONE
- Group by phase for systematic review

## 2. Verify Each Task
For each DONE task:

### a. Check "Done When" Criteria
- Run the "Done When" command exactly as written
- Verify exit code is 0
- If fails: Create new task to fix it, note in audit report

### b. Verify Code Exists
- Check files listed in "Context" field
- Verify implementation exists and is non-trivial
- If missing: Create new task, note in audit report

### c. Check DEVLOG Entry
- Verify task ID appears in DEVLOG.md
- Check if implementation notes are present
- If missing: Note in audit report (don't create task, just warning)

### d. Check Git Commit
- Search git log for task ID
- Verify commit exists and includes relevant files
- If missing: Note in audit report (warning only)

### e. Run Related Tests
- If task mentions tests, run them
- `uv run pytest tests/ -k [relevant_pattern]`
- If fails: Create new task to fix tests

### f. Compare to SPEC
- Check if implementation matches SPEC.md requirements
- Note any intentional deviations
- Identify if deviation was necessary or a bug

## 3. Handle Drift from SPEC
For each necessary deviation found:

### a. Validate Necessity
- Check if deviation improves implementation
- Check if deviation was required by dependencies/constraints
- Check if deviation maintains spec intent

### b. Update SPEC
- If deviation is necessary and good:
  - Update relevant section in SPEC.md
  - Note reason for change
  - Preserve requirement intent
- If deviation is unnecessary:
  - Create TODO task to fix code to match SPEC
  - Don't update SPEC

### c. Document Change
- Add note in DEVLOG.md about spec update
- Include rationale for accepting drift

## 4. Generate Audit Report
Create structured report:

```markdown
# TODO Audit Report - [Date]

## Summary
- Total DONE tasks: X
- Verified complete: Y
- Issues found: Z
- SPEC updates: W

## Verified Tasks
[List of task IDs that passed all checks]

## Issues Found

### Incomplete Tasks
- TASK-XXX: [Issue description]
  - Problem: [what's wrong]
  - Created: TASK-YYY to fix

### Missing Tests
- TASK-XXX: Tests don't pass
  - Error: [test output]
  - Created: TASK-YYY to fix

### Missing Documentation
- TASK-XXX: No DEVLOG entry (warning only)
- TASK-XXX: No commit found (warning only)

## SPEC Updates
- Section X.Y: [What changed and why]
- Section A.B: [What changed and why]

## New Tasks Created
- TASK-YYY: [Description]
- TASK-ZZZ: [Description]
```

## 5. Update Files
- Add new tasks to TODO.md with status TODO
- Update SPEC.md sections as needed
- Append audit results to DEVLOG.md
- Commit all changes

# RULES
1. Be thorough but not pedantic - focus on functionality
2. Missing DEVLOG/commit is a warning, not a blocking issue
3. Only create new tasks for actual functional problems
4. When updating SPEC, preserve original intent
5. Never mark a task as incomplete without creating a fix task
6. Don't re-run tasks that are clearly complete (use judgment)
7. If "Done When" is ambiguous, check the obvious interpretation
8. Group related issues into one task when sensible
9. Don't audit more than 20 DONE tasks in one run (too long)
10. Prioritize recent tasks over old ones if limiting scope

# VERIFICATION PRIORITIES
High priority (always check):
- "Done When" command passes
- Code implementation exists
- Tests pass if mentioned

Medium priority (check if time):
- DEVLOG entry exists
- Commit exists with task ID

Low priority (warnings only):
- Code quality/style
- Comment completeness

# SPEC UPDATE CRITERIA
Update SPEC when:
- ✓ Code improvement that maintains requirement intent
- ✓ Necessary technical constraint (library limitation)
- ✓ Discovered requirement that was underspecified
- ✗ Bug in implementation (fix code instead)
- ✗ Developer preference without technical reason
- ✗ Shortcut to save time (fix code instead)

# OUTPUT FORMAT
```
📋 TODO AUDIT COMPLETE

Verified: X/Y tasks
Issues: Z tasks
New tasks: W tasks
SPEC updates: V sections

⚠️ Action Required:
- Review new tasks in TODO.md
- Review SPEC.md changes
- Address high-priority issues first

See audit-report-[DATE].md for details
```

# ERROR HANDLING
- If "Done When" command syntax is invalid, try to interpret it reasonably
- If can't verify a task due to missing context, note in report
- If SPEC is ambiguous, ask human for clarification
- Don't fail entire audit if one task check fails
