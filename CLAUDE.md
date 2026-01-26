# Agent Context

## Role
Development agent under human supervision. Propose implementations; human executes and approves.

## Standard Workflow

**CRITICAL: Follow this workflow strictly for all tasks:**

1. **Study SPEC.md** - Understand project goals and architecture
2. **Pick ONE task from TODO.md** - Select highest priority TODO or IN_PROGRESS task
3. **Update task to IN_PROGRESS** - Mark task status before starting work
4. **Implement task fully** - Work until done-when criteria passes OR give up after 2 failed attempts
5. **Update TODO.md** - Mark task DONE/BLOCKED/FAILED with notes
6. **Update DEVLOG.md** - Append dated entry with changes and decisions
7. **Commit to main** - Descriptive message with "Co-Authored-By: Claude Sonnet 4.5"

**Do not work on multiple tasks simultaneously. Complete one task before moving to the next.**

## Output Format

For implementation tasks, always include:
1. **Assumptions** - with confidence level: `KNOWN | INFERRED | SPECULATIVE`
2. **Files to create/modify** - specific paths
3. **Verification steps** - how to test the implementation
4. **Done-when criteria** - testable success conditions

STOP and ask if you'd need a SPECULATIVE assumption about external APIs/libraries.

## Key Constraints

- Include error handling in all code
- Use event logging by aggregating data into log structures, emit full log with context (not piecemeal)
- Stay within task scope - don't expand unilaterally
- Run `uv run ruff check src/` before committing

## When Stuck

After 2 failed attempts on the same problem, STOP and report what you've learned. Don't continue failing.

## Project Resources

- **SPEC.md** - Project specification and architecture *(start here)*
- **TODO.md** - Complete task backlog with priorities and status
- **DEVLOG.md** - Development log with decisions and version history
- **README.md** - Usage guide, command reference, data schemas, directory structure
- **pyproject.toml** - Dependencies and tool configuration
- **docs/ghcnh_DOCUMENTATION.pdf** - Official NOAA data format documentation
- **docs/papers/** - Research papers relevant to the project

**Note:** When researching or implementing features, if you find relevant papers from arXiv (e.g., SAITS, CSDI, imputation methods), download them and add them to `docs/papers/` for future reference.

## Quick Reference

**Tech Stack:** Python 3.10+, Polars, PyTorch, Marimo notebooks, Typer CLI, uv build tool

**Key Commands:**
```bash
# Download data
uv run python src/scripts/ghcnh_downloader.py

# Compute/clean metadata
uv run python src/scripts/compute_metadata.py compute
uv run python src/scripts/clean_metadata.py clean

# Notebooks
uv run marimo edit notebooks/01_station_exploration.py

# Code quality
uv run ruff check src/
uv run mypy src/
```

See **README.md** for detailed command options, architecture, and data schemas.
