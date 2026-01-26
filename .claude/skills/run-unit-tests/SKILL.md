---
name: run-unit-tests
description: Run all unit tests, create TODO tasks for failures, split into reasonable chunks.
---

# ROLE
Test runner and failure analyst. Execute test suite, categorize failures, create actionable fix tasks.

# INPUT
None required. Agent discovers and runs all tests via pytest.

# OUTPUT
1. Test execution report with pass/fail breakdown
2. New TODO tasks for each distinct failure category
3. Grouped failures into reasonable fix chunks
4. Priority assigned to each failure task

# WORKFLOW

## 1. Discover Tests
- Run: `uv run pytest tests/ --collect-only`
- Count total tests
- Identify test modules and their purposes

## 2. Run Full Test Suite
- Execute: `uv run pytest tests/ -v --tb=short`
- Capture all output
- Note overall pass/fail counts
- Record execution time

## 3. Analyze Failures
For each failing test:

### a. Categorize Failure Type
- **Import Error:** Missing dependency or module
- **Assertion Error:** Logic/behavior mismatch
- **Type Error:** mypy-like type mismatch at runtime
- **File Not Found:** Missing test data or fixture
- **Timeout:** Performance issue
- **Setup/Teardown:** Fixture or test infrastructure issue

### b. Identify Root Cause
- Read test code to understand intent
- Check error message and traceback
- Identify if it's:
  - Missing implementation
  - Wrong implementation
  - Test bug (test itself is wrong)
  - Environmental issue (missing file, etc.)

### c. Group Related Failures
- Group by module (e.g., all `test_masking.py` failures)
- Group by root cause (e.g., all "missing MICE implementation")
- Group by priority/complexity
- Don't create one task per test - cluster logically

## 4. Create TODO Tasks
For each failure group:

### Task Format
```markdown
#### TASK-XXX: Fix [module/component] test failures
- **Status:** TODO
- **Priority:** [HIGH|MEDIUM|LOW]
- **Done When:** `uv run pytest tests/[specific_pattern] -v` exits 0
- **Context:**
  - Test file: tests/test_[module].py
  - Source file: src/weather_imputation/[module].py
  - Related: TASK-YYY (if dependent)
- **Failures:**
  - `test_function_name`: [brief error description]
  - `test_other_function`: [brief error description]
- **Root Cause:** [Missing implementation | Bug in X | Test infrastructure]
- **Notes:** [Leave empty for agent]
```

### Priority Assignment
- **HIGH:** Blocking core functionality (data loading, model training)
- **MEDIUM:** Important but not blocking (metrics, analysis)
- **LOW:** Nice-to-have (edge cases, optional features)

## 5. Group by Complexity
Split large failure groups if:
- More than 5 related test failures → split into 2-3 tasks
- Mix of quick fixes and complex work → separate tasks
- Different components involved → separate tasks

Reasonable task size:
- 1-5 test failures per task (most common)
- Single focused fix (implementation of one function)
- Completable in one work session

## 6. Generate Report
Create structured test report:

```markdown
# Test Execution Report - [Date]

## Summary
- **Total Tests:** X
- **Passed:** Y (Z%)
- **Failed:** W
- **Skipped:** V
- **Execution Time:** T seconds

## Pass Rate by Module
- tests/test_data/: X/Y (Z%)
- tests/test_models/: X/Y (Z%)
- tests/test_evaluation/: X/Y (Z%)

## Failure Breakdown

### Critical (HIGH Priority) - N failures
1. **[Module/Component]** - M tests
   - Root cause: [description]
   - Created: TASK-XXX

### Important (MEDIUM Priority) - N failures
1. **[Module/Component]** - M tests
   - Root cause: [description]
   - Created: TASK-YYY

### Optional (LOW Priority) - N failures
1. **[Module/Component]** - M tests
   - Root cause: [description]
   - Created: TASK-ZZZ

## New Tasks Created
- TASK-XXX: Fix data loader tests (HIGH)
- TASK-YYY: Fix SAITS model tests (MEDIUM)
- TASK-ZZZ: Fix edge case tests (LOW)

## Recommendations
1. [Suggested order to tackle tasks]
2. [Any patterns or systemic issues noticed]
3. [Suggested test infrastructure improvements]
```

## 7. Update TODO.md
- Add new tasks to TODO.md
- Place in appropriate phase section
- Order by priority within phase
- Link related tasks with "Related:" or "Blocked by:"

## 8. Update DEVLOG.md
- Append test report summary
- Note test coverage status
- Document any patterns in failures

# RULES
1. Don't create duplicate tasks - check TODO.md first
2. Group intelligently - not too granular, not too broad
3. Include actual error messages in task descriptions (truncated if long)
4. Assign priority based on functionality, not test count
5. If all tests pass, just report success (no new tasks)
6. Skip tests marked with `@pytest.mark.skip` in statistics
7. Re-run flaky tests once before marking as failure
8. Include pytest command that will verify fix in "Done When"
9. Don't overwhelm TODO with 50 tasks - max 10-15 tasks total
10. If too many failures, create umbrella tasks

# TASK GROUPING STRATEGIES

### Strategy 1: By Module
Group all failures in same test module:
- `tests/test_masking.py` → one task
- `tests/test_normalization.py` → one task

### Strategy 2: By Feature
Group by feature being tested:
- All MCAR masking tests → one task
- All z-score normalization tests → one task

### Strategy 3: By Root Cause
Group by underlying issue:
- All "missing MICE implementation" → one task
- All "incorrect tensor shapes" → one task

### Strategy 4: By Priority/Size
- Quick wins (simple fixes) → one task
- Complex implementations → separate tasks each

**Choose strategy based on failure patterns.**

# EXAMPLE TASK
```markdown
#### TASK-042: Fix data masking test failures
- **Status:** TODO
- **Priority:** HIGH
- **Done When:** `uv run pytest tests/test_data/test_masking.py -v` exits 0
- **Context:**
  - Test file: tests/test_data/test_masking.py
  - Source: src/weather_imputation/data/masking.py
  - Related: TASK-015 (implement masking strategies)
- **Failures:**
  - `test_mcar_masking_preserves_distribution`: AssertionError on marginal stats
  - `test_mar_masking_temperature_dependency`: KeyError 'dew_point_temperature'
  - `test_mnar_masking_extreme_values`: NotImplementedError in _identify_extremes
- **Root Cause:** Missing MNAR strategy implementation, MAR has incorrect variable references
- **Notes:** [Leave empty]
```

# OUTPUT FORMAT
```
🧪 TEST SUITE EXECUTION COMPLETE

Total: X tests
Passed: Y (Z%)
Failed: W

New tasks created: N
  High priority: X
  Medium priority: Y
  Low priority: Z

📊 See test-report-[DATE].md for details
📝 TODO.md updated with fix tasks
```

# ERROR HANDLING
- If pytest fails to run (import errors), create one HIGH priority task to fix test infrastructure
- If all tests fail due to one root cause (e.g., missing dependency), create one task, don't spam TODO
- If can't determine priority, default to MEDIUM
- If unsure how to group, prefer more specific tasks over fewer giant tasks
