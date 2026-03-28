# Releasing

This repository uses a Python-first release flow.

## Release Authority

The only release authority is:

- `python-semantic-release`

Version source of truth:

- `pyproject.toml`

Releases are cut from:

- `main`

## Version Bump Rules

- `feat:` -> minor bump
- `fix:` -> patch bump
- `perf:` -> patch bump
- `feat!:` / `fix!:` / `BREAKING CHANGE:` -> major bump

Examples:

```text
feat: add new retrieval workflow
fix: correct docker image tag handling
feat!: replace legacy release system
```

## Release Outputs

Each real release can produce:

- a git tag like `v1.48.2`
- updated `CHANGELOG.md`
- updated version in `pyproject.toml`
- a GitHub release
- a PyPI package release
- compose bundle assets attached to the GitHub release

## Docker Releases

Docker image builds are driven directly by the main release workflow after a successful semantic release.

Release flow:

1. `python-semantic-release` computes and creates the release tag
2. Python package artifacts are built and published
3. compose bundles are attached to the GitHub release
4. Docker images are built and pushed in the same workflow using the released tag

This avoids the GitHub Actions limitation where tags created by `GITHUB_TOKEN` do not reliably trigger downstream workflows.

The Docker workflow builds and pushes:

- `talk2aiagents4pharma`
- `talk2biomodels`
- `talk2scholars`
- `talk2knowledgegraphs`

CPU/GPU variants are preserved where configured by the workflow.

The standalone `docker_build.yml` workflow remains available for manual or tag-based rebuilds, but normal releases no longer depend on it being triggered by a separate workflow event.

## Compose Bundles

Compose bundles are packaged during the main release workflow and uploaded as release assets.

They do not own the GitHub release body and do not append large repeated release text.

## Local Verification

Before changing release behavior:

```bash
uv sync --extra dev
uv run semantic-release --version
uv run semantic-release version --help
uv build
uv run twine check dist/*
```
