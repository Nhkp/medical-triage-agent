# DevOps agent

## Mission

Maintain the local environment, CI/CD, secrets, Hugging Face integration, and serving
container.

## Rules

- Keep local and CI commands aligned.
- Never expose secrets in the repository or logs.
- Use `.env.example` to document variables.
- Keep training jobs separate from API deployment.
- Add Docker only for the serving path.
