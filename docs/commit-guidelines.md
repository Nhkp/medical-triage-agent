# Commit guidelines

## Principles

- A commit must be independent: it should be understandable, reviewable, and revertable
  on its own.
- A commit covers one small task.
- Do not mix refactoring, fixes, documentation, and feature changes in the same commit.
- Do not include generated files, local caches, secrets, or temporary data.
- Run the checks before pushing:

```bash
make check
```

## Message format

Messages follow Conventional Commits:

```text
type(scope): description courte
```

The `scope` is optional, but recommended when it clarifies the affected area.

Common types:

- `feat`: new feature.
- `fix`: bug fix.
- `docs`: documentation only.
- `test`: adding or changing tests.
- `refactor`: internal change without new behavior.
- `chore`: maintenance, configuration, tooling.
- `ci`: CI/CD pipeline.
- `build`: packaging or dependencies.

Exemples:

```text
docs: add commit guidelines
ci: add coverage check
chore(pyproject): configure ruff and mypy
test: add package import smoke test
```

## Before committing

- Check that the diff matches a single intent.
- Split into several commits if the diff tells several stories.
- Review `git diff --staged`.
- Fix issues and run `git add` again if a hook modifies files.
- Install hooks with `make hooks` when working locally.
