# Git workflow

## Branches

- `main` must stay stable.
- Use short-lived branches named after the intent, for example `ci/dependabot`.
- Prefer several small branches over one large branch.

## Commits

- One commit covers one small task.
- Each commit must be independently understandable and revertable.
- Commit messages must follow Conventional Commits.
- Split the work when a diff mixes unrelated changes.

## Pull requests

- Keep pull requests focused on one intent.
- Run `make check` before opening or updating a pull request.
- Use the pull request template checklist.
- Merge only after CI passes.

## Release notes

Update `CHANGELOG.md` when a change affects users, commands, architecture, or project
policy.
