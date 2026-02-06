# Repo Layout Refactor 2026-02

This document tracks the migration of the repository structure to top-level language directories.

## Folder Mapping

| Old Content                    | New Path              | Description                    |
| ------------------------------ | --------------------- | ------------------------------ |
| `src/`                         | `python/src/`         | Python application code        |
| `tests/`                       | `python/tests/`       | Python tests                   |
| `web/`                         | `node/ui/`            | React Frontend                 |
| `bff/`                         | `node/bff/`           | Backend-for-Frontend (Node.js) |
| `src/services/analytics-crud`  | `go/analytics-crud`   | Go Service: Analytics CRUD     |
| `src/services/inference-gateway`| `go/inference-gateway`| Go Service: Inference Gateway  |

## Checklist of Updates

- [x] Dockerfiles & Compose
    - [x] `docker-compose.app.yml`
    - [x] `Dockerfile` (API)
    - [x] `src/services/*/Dockerfile`
    - [x] `bff/Dockerfile`
    - [ ] `config/docker/*.Dockerfile`
- [ ] CI/CD
    - `.github/workflows/ci.yml` (paths, working-dirs)
- [ ] Tooling
    - `Makefile`
    - `pyproject.toml`
    - `package.json` (scripts, workspaces if added)
    - `go.mod` (module paths if changed - unlikely)

## Notes

- This refactor does not change runtime behavior or APIs.
- Go services retain their module isolation.
