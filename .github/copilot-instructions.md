# Your purpose

You are a general purpose agent for long-running agentic tasks. Use the tools and skills available to you eagerly. Delegate to sub-agents when a task requires heavy reasoning or context (research, code-writing, large refactors) so the main agent preserves a clear high-level picture. Context management is paramount.

# Style

## Infer Intent, Don't Follow Instructions Literally

Treat user instructions as signals of intent, not exact specifications. Prioritize what the user is trying to achieve rather than asking for clarification at every detail. 

- Fill gaps intelligently. When a request is underspecified, make a reasonable assumption and proceed.
- Correct for obvious errors. If an instruction contains a mistake or contradicts a stated goal, address the underlying intent and note what you changed.
- Recognise the X-Y problem. If the requested approach seems roundabout, name the underlying goal and address that instead.
- Consider the broader context. A single instruction exists within a larger conversation — resolve conflicts with established goals thoughtfully rather than treating each message in isolation.

## Do not's

- Call done prematurely, only call done when the task is truly complete and verified. The only times to call done early is to ask important questions or ask for a review on a significant decision point. 

- Source control is strictly the responsibility of the user, not agents. You should not touch source control.

# Skills

Skills live in `.agents/skills/`. Each skill is a directory containing a `SKILL.md` with instructions for a specific kind of task (e.g. `test-driven-development`, `systematic-debugging`, `writing-plans`, `quantitative-research`).

Available skills are injected into your system prompt under `<available_skills>`. Load one via the `skill` tool.

- Before starting any task, scan available skills and load matching ones.
- When transitioning to a new sub-task, re-evaluate and load skills relevant to it.
- Load eagerly. Skills encode hard-won workflow knowledge; loading is cheap, skipping is expensive.

# Repository layout

Polyglot monorepo for a systematic NZX investment strategy.

- `java/` — Data ingestion. Fetches ~14 sources (Yahoo Finance, NZ macro stats, etc.) and produces `data/data_long.csv`. Entry point `java/src/main/java/lazic/Main.java`. See `java/docs/ARCHITECTURE.md`.
- `python/` — Three packages in a `uv` workspace:
  - `python/common/` (`skuld_common`) — shared contract types and validation. No deps on the other two.
  - `python/research/` (`skuld_research`) — research, backtesting, factor signals, walk-forward. Produces frozen spec files.
  - `python/portfolio/` (`skuld_portfolio`) — production recommendation app. Consumes a frozen spec + Sharesies CSV.
  - `python/tests/` — shared pytest suite for all three packages.
- `docs/` — Cross-cutting docs, plans (`docs/plans/`), specs (`docs/specs/`).
- `data/` — Generated CSV artefacts (gitignored except `.gitkeep`).
- `.agents/skills/` — Skill definitions loaded via the `skill` tool.

# Philosophy

Production-ready systematic investment strategy for the NZX. Design principles:

- DRY (don't repeat yourself).
- SSOT (single source of truth).
- SoC (separation of concerns).
- Minimize bloat and technical debt. Bias to simplicity and clarity. Delete dead code.
- Split files over 500 lines into submodules.
- Prefer low nesting depth.
- Gather context before acting. Uninformed decisions are dangerous.
- Follow language-specific conventions and best practices.
- Consider scalability and maintainability, but not at the expense of simplicity. Refactor when a pattern will be cumbersome to extend.
- Use the right tool for the job. Don't reinvent the wheel; use existing libraries unless there is a compelling reason not to.

# Documentation requirements

When creating any new reusable system, component, or pattern:

1. Document it in the most fitting place and add a link under the Documentation index at the bottom of this file. Mandatory — other agents need to discover it.
2. Other agents must be able to discover and reuse your work.
3. Undocumented systems lead to duplicated effort and inconsistency.
4. Keep docs concise and high-level. Implementation details belong in code comments and docstrings. Documentation explains purpose and usage, not internals.

## Where docs live

- Cross-cutting / project-wide: `docs/<TOPIC>.md` (e.g. `docs/APPLICATION.md`, `docs/DATA_PIPELINE.md`).
- Language- or module-specific: alongside the code (e.g. `java/docs/ARCHITECTURE.md`).
- Plans (multi-step implementation roadmaps): `docs/plans/YYYY-MM-DD-<slug>.md`.
- Specs (frozen requirements / acceptance criteria for a milestone): `docs/specs/YYYY-MM-DD-<slug>.md`.

Date prefix on plans and specs is the authored date in ISO format. See the `writing-plans` skill before authoring a new plan.

# Tests and feedback loops

Tests verify changes. Write them with intent to find edge cases that would fail.

Before calling a task done, ask: is there a feedback loop I can use to verify the change? Tests, or review by another agent. If a feedback loop exists, use it before declaring done.

When tests fail or feedback is negative, understand why first. Modify the test or feedback loop only if investigation shows the issue is with them, not the code. Bias heavily on changing code rather than tests — tests verify code, not the other way around. See the `verification-before-completion` skill before claiming work is complete.

## Running tests

Python tests use pytest via uv from `python/`:

- `uv run pytest` — full suite.
- `uv run pytest -m "not slow"` — skip slow tests.
- `uv run pytest tests/test_pit_loader.py::test_name` — single test.

Lint and type-check from `python/`:

- `uv run ruff check .`
- `uv run pyright`

Java builds with Maven from `java/`:

- `mvn -q compile`
- `mvn -q exec:java`

# Documentation index

Keep current. When adding, moving, or deleting a doc, update the corresponding entry.

Project and architecture:

- `docs/APPLICATION.md` — High-level Python application structure: the three packages (`skuld_common`, `skuld_research`, `skuld_portfolio`), data flow, research-vs-production split, statistical gating. Start here for orientation.
- `docs/DATA_PIPELINE.md` — Raw data quality characteristics, preprocessing stages, contract types passed between pipeline stages. Required reading before touching data code.
- `docs/DATA_ANALYSIS.md` — Schema and content reference for `data/data_long.csv` and `data/source_legend.csv` (Java ingestion output).
- `docs/specs/2026-04-29-raw-data-analysis-workflow.md` — Specification for a reusable raw-data analysis workflow over `data/data_long.csv`, including coverage, sparsity, temporal behavior, anomaly detection, and leakage heuristics.
- `java/docs/ARCHITECTURE.md` — Java ingestion architecture: `IngestManager`, `DataSourceBase`, parallel source execution, CSV output.
- `java/docs/DATA_SOURCES.md` — Per-source reference: which fields each of the ~14 data sources produces.

Domain reference:

- `docs/sharesies-pricing.md` — Sharesies NZ subscription plans and per-trade fee structure. Used by the execution planner's fee-cliff optimisation.

Plans and specs (historical records, dated):

- `docs/plans/` — Implementation plans authored before multi-step work.
- `docs/specs/` — Frozen requirements and acceptance criteria for milestones.
