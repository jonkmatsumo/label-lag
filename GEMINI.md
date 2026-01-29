# GEMINI.md

This file provides context and guidelines for Gemini CLI when working with this repository.

## Project Overview

**Label Lag** is a synthetic data generation and fraud detection pipeline. It consists of a Python-based backend for data generation and inference, a Node.js/Fastify Backend-for-Frontend (BFF), and a React-based web UI.

### Key Components
- **Root (`/`)**: Python backend using `uv`, FastAPI, and SQLAlchemy. Handles data generation, ML training, and inference.
- **BFF (`/bff`)**: Node.js/Fastify proxy layer providing type-safe APIs to the frontend.
- **Web (`/web`)**: React/Vite frontend using TanStack Query and Material Design principles.
- **Infrastructure**: Docker Compose managed via `Makefile`.

## Development Workflow

### Package Management
- **Python**: Uses `uv`. **Do not use pip directly.**
  - Install: `uv sync --all-extras`
  - Run: `uv run <command>`
- **Node.js**: Uses `npm` in `bff/` and `web/` directories.

### Common Commands (Prefer `Makefile`)

| Task | Command | Description |
|------|---------|-------------|
| **Install** | `make install` | Installs Python (`uv`) and sets up hooks. |
| **Test (Py)** | `make test` | Runs Python tests with coverage (`pytest`). |
| **Lint (Py)** | `make lint` | Checks Python code with `ruff`. |
| **Format (Py)** | `make lint-fix` | Auto-formats Python code. |
| **Infra** | `make infra-up` | Starts Postgres, MinIO, MLflow. |
| **Start App** | `make app-up` | Starts API, BFF, and Web in Docker. |
| **Reset** | `make reset-all` | Destructive reset of all data and containers. |

### Component-Specific Commands

**BFF (`/bff`)**:
- Test: `npm test` (Vitest)
- Dev: `npm run dev`

**Web (`/web`)**:
- Test: `npm run test:e2e` (Playwright)
- Dev: `npm run dev`

## Code Style & Conventions

### Python
- **Formatter/Linter**: `ruff` is the authority.
  - Line length: 88
  - Rules: E, F, I, N, W, UP
- **Typing**: Use standard Python type hints.
- **Testing**: `pytest` is the runner. Place tests in `tests/`.

### TypeScript / JavaScript
- **Web**: React + Vite. Use functional components and hooks.
- **BFF**: Fastify + TypeScript.
- **Linting**: ESLint is configured in both directories.

### Commit Messages
Follow **Conventional Commits** strictly:
`type(scope): description`

- **Types**: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`
- **Scopes**: `web`, `bff`, `api`, `infra`, `e2e`
- **Description**: Lowercase, imperative mood, no period at end.

**Example**:
```text
feat(web): implement drift monitoring panel in model lab
test(bff): add contract tests for analytics endpoints
fix(api): correct window function for velocity feature
```

## Critical Rules
1.  **Respect `.env`**: Configuration comes from environment variables.
2.  **Point-in-Time Correctness**: When touching feature engineering code, ensure no future leakage.
3.  **Docker First**: Assume services run in Docker. Use `make` commands to interact with the stack.
4.  **No Leaky Abstractions**: Keep `evaluation_metadata` separate from training features.
