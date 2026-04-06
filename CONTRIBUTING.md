# Contributing

## Ground Rules

- Keep the simulator modular. New features should fit the existing `drone`, `task`, `sensor`, `reward`, and `reset` boundaries rather than bypassing them.
- Avoid adding fallback or mock behavior unless it is explicitly requested. Unsupported features should fail clearly.
- Keep source files small and focused. Files larger than roughly 300 LOC are discouraged.
- Preserve the civilian research and developer-use orientation of the project.

## Development Setup

```bash
python -m pip install -e . --no-build-isolation
pytest
python scripts/smoke_test.py --config configs/tasks/hover.toml
```

If you are working in a shared Python environment, prefer a dedicated virtualenv. Upstream PufferLib packaging can influence the installed NumPy version during `pip install`.

Legacy in-repo training still depends on `pufferlib<4`. Use the dedicated PufferLib 4 export/train scripts when working against the upstream `4.0` branch.

## Making Changes

1. Open an issue or discussion first for non-trivial simulator, API, or training changes.
2. Keep commits scoped and named clearly, for example `feat(native): add new task` or `fix(training): correct batch geometry`.
3. Add or update tests when changing:
   - native physics behavior
   - reward logic
   - config parsing
   - public Python interfaces
4. Update `README.md` and `docs/architecture.md` when changing repo structure or user-facing commands.

## Pull Requests

- Include a short problem statement and the design intent.
- List the commands you ran locally.
- Call out any follow-up work or known limitations.
- Keep PRs reviewable. Prefer incremental changes over large mixed refactors.

## Reporting Issues

When filing a bug, include:

- the config file used
- the exact command
- platform and Python version
- whether the failure happens during build, reset, stepping, rollout, or training
- a minimal reproduction if possible
