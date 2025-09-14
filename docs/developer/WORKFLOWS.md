# GitHub Workflows Documentation

This document provides a comprehensive overview of all GitHub Actions workflows in the AIAgents4Pharma repository, detailing their purpose, triggers, and functionality.

## Overview

Our CI/CD pipeline uses **UV** for fast, reliable dependency management across all workflows. All workflows are designed to be efficient, secure, and provide comprehensive quality assurance.

## Workflow Categories

### 🧪 Testing Workflows
- [`tests_talk2aiagents4pharma.yml`](#tests-talk2aiagents4pharma)
- [`tests_talk2biomodels.yml`](#tests-talk2biomodels)
- [`tests_talk2knowledgegraphs.yml`](#tests-talk2knowledgegraphs)
- [`tests_talk2scholars.yml`](#tests-talk2scholars)
- [`tests_talk2cells.yml`](#tests-talk2cells)

### 🔒 Security & Quality
- [`security_audit.yml`](#security-audit)
- [`sonarcloud.yml`](#sonarcloud-analysis)

### 🐳 Build & Deploy
- [`docker_build.yml`](#docker-build)
- [`docker_compose_release.yml`](#docker-compose-release)
- [`release.yml`](#release)

### 📚 Documentation
- [`mkdocs_deploy.yml`](#mkdocs-deploy)

---

## Testing Workflows

### Tests Talk2Scholars

**File:** `tests_talk2scholars.yml`

**Purpose:** Comprehensive testing and quality checks for the Talk2Scholars component

**Triggers:**
- Pull requests to `main` with changes to:
  - `aiagents4pharma/talk2scholars/**`
  - `pyproject.toml`
  - `uv.lock`
- Manual workflow dispatch

**Jobs:**

#### 1. Code Quality Checks
- **Runner:** Ubuntu Latest
- **Dependencies:** UV sync with frozen lockfile
- **Checks:**
  - Pylint analysis with specific disabled rules
  - Ruff linting for code style
  - Bandit security scanning

#### 2. Cross-Platform Testing Matrix
- **Strategy:** Fail-fast disabled for comprehensive testing
- **Matrix:**
  - OS: Ubuntu Latest, macOS 15, Windows Latest
  - Python: 3.12
- **Steps:**
  - UV dependency installation
  - Test execution with coverage
  - Coverage reporting and XML generation
  - **100% coverage requirement** (builds fail if not met)
  - Codecov upload (Ubuntu only)

**Environment Variables:**
```yaml
OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
ZOTERO_API_KEY: ${{ secrets.ZOTERO_API_KEY }}
ZOTERO_USER_ID: ${{ secrets.ZOTERO_USER_ID }}
```

---

## Security & Quality Workflows

### Security Audit

**File:** `security_audit.yml`

**Purpose:** Focused dependency security monitoring and vulnerability detection

**Triggers:**
- Weekly schedule (Sundays at 2 AM UTC)
- Push to `main` affecting `pyproject.toml` or `uv.lock`
- Pull requests to `main` affecting dependency files
- Manual workflow dispatch

**Jobs:**

#### 1. Dependency Security Scan
- **Tools:** pip-audit, safety
- **Outputs:** JSON and Markdown reports
- **Features:** Continues on error to allow other jobs

#### 2. SARIF Upload & Processing
- **Purpose:** Integration with GitHub Security tab
- **Process:**
  - Downloads all security reports
  - Processes vulnerability findings
  - Uploads consolidated reports
- **Focus:** Dependency vulnerabilities only

#### 3. Security Summary Generation
- **Output:** Markdown summary with vulnerability counts
- **Purpose:** Weekly dependency vulnerability monitoring
- **Note:** Code security handled by test workflows via bandit

#### 4. SonarCloud Security Integration
- **Trigger:** Push to main branch only
- **Features:**
  - Downloads security reports
  - Generates coverage for SonarCloud
  - Performs dependency-focused security analysis

**Key Features:**
- ✅ Focused dependency vulnerability monitoring
- ✅ Weekly automated security scanning schedule
- ✅ Streamlined workflow for reliable security checks
- ✅ Integration with GitHub Security dashboard

### SonarCloud Analysis

**File:** `sonarcloud.yml`

**Purpose:** Advanced code quality analysis and technical debt tracking

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Manual workflow dispatch

**Environment Variables:**
```yaml
OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # Required for test execution
```

**Process:**
1. **Setup:** UV dependency installation with dev dependencies
2. **Testing:** Full test suite with coverage generation
3. **Analysis:** Pylint JSON output with standard disable flags
4. **Upload:** SonarCloud analysis with all reports

**Key Features:**
- ✅ OPENAI_API_KEY environment for test compatibility
- ✅ Pylint analysis with standard disable flags: `--disable=R0801,R0902,W0221,W0122`
- ✅ Comprehensive code quality analysis
- ✅ Streamlined security and quality reporting

**Artifacts:**
- Coverage XML
- Pylint JSON report
- 30-day retention period

---

## Build & Deploy Workflows

### Docker Build

**File:** `docker_build.yml`

**Purpose:** Build and push Docker images for all agents

**Features:**
- Multi-stage builds for optimized image sizes
- Pinned base images (no `latest` tags)
- Separate CPU and GPU variants
- Health check implementations
- Push to Docker Hub registry

### Docker Compose Release

**File:** `docker_compose_release.yml`

**Purpose:** Release management for Docker Compose configurations

**Features:**
- Separate compose files for CPU and GPU deployments
- Production-ready configurations
- Version tagging and release automation

### Package Build

**File:** `package_build.yml`

**Purpose:** Python package validation and testing

**Triggers:**
- Pull requests to `main` with changes to:
  - `pyproject.toml`
  - `uv.lock`
  - `aiagents4pharma/**`
- Manual workflow dispatch

**Features:**
- **VCS Versioning:** Uses git tags via hatch-vcs for version management
- **Quality Checks:** Ruff linting and Bandit security scanning
- **Cross-Platform Testing:** Ubuntu, macOS 15, Windows Latest
- **Package Validation:** Builds wheel/sdist and tests installation
- **Dependency Resolution:** Fixed PyArrow compatibility issues with uv

**Key Features:**
- ✅ VCS git tag versioning with hatch-vcs
- ✅ Compatible dependencies with `pyarrow>=14.0.0` and `datasets>=4.0.0`
- ✅ Cross-platform virtual environment handling (Windows/Unix)
- ✅ Uses latest available git tag for PR testing

### Release

**File:** `release.yml`

**Purpose:** Automated release management with semantic versioning

**Triggers:**
- Push to `main` branch with changes to:
  - `aiagents4pharma/**`
  - `pyproject.toml`
  - `uv.lock`
- Manual workflow dispatch

**Features:**
- **Semantic Release:** Automatic version bumping based on commit messages
- **Auto-Tagging:** Creates git tags automatically (feat:/fix:/BREAKING CHANGE:)
- **Quality Gates:** Ruff linting and Bandit security before release
- **PyPI Publishing:** Automated package distribution
- **GitHub Releases:** Auto-generated with changelogs

**Semantic Release Convention:**
```bash
feat: new feature     → Minor version bump (1.0.0 → 1.1.0)
fix: bug fix         → Patch version bump (1.0.0 → 1.0.1)
BREAKING CHANGE:     → Major version bump (1.0.0 → 2.0.0)
```

**Key Features:**
- ✅ Semantic-release automation with conventional commits
- ✅ Modern uv dependency management for fast builds
- ✅ Triggers on main branch pushes for continuous deployment
- ✅ Auto-creates tags and triggers publishing pipeline

---

## Documentation Workflows

### MkDocs Deploy

**File:** `mkdocs-deploy.yml`

**Purpose:** Automated documentation deployment to GitHub Pages

**Triggers:**
- Push to `main` branch with changes to:
  - `docs/**`
  - `mkdocs.yml`
  - `aiagents4pharma/**`
  - `pyproject.toml`
- Manual workflow dispatch

**Features:**
- **Modern UV Setup:** Uses project dependencies instead of manual pip installs
- **Jupyter Integration:** Notebook rendering with mkdocs-jupyter
- **Material Theme:** Modern styling with mkdocs-material
- **Auto-Deploy:** GitHub Pages deployment with force push
- **Smart Triggers:** Only runs when documentation-related files change

**Key Features:**
- ✅ Modern UV dependency management with `uv sync --frozen`
- ✅ Uses project dependencies from pyproject.toml
- ✅ Streamlined deployment process
- ✅ Path-based triggers for efficient builds

---

## Workflow Architecture Principles

### 1. **UV-First Approach**
All workflows use UV for fast, reliable dependency management:
```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4

- name: Set up Python
  run: uv python install 3.12

- name: Install dependencies
  run: uv sync --frozen --extra dev
```

### 2. **Security-First Design**
- Comprehensive security scanning in test workflows
- Weekly dependency vulnerability monitoring
- Integration with GitHub Security dashboard
- Bandit skips for legitimate ML/data science patterns

### 3. **Quality Assurance**
- Multi-platform testing matrices (Ubuntu, macOS, Windows)
- 100% code coverage requirements with flexible LLM test handling
- Multiple linting tools (ruff, pylint) with consistent disable flags
- Cross-platform compatibility (Windows PowerShell vs Unix bash)

### 4. **Dependency Management & Reliability**
- **Streamlined approach** for simplicity and reliability
- **Dependency conflict resolution** via uv (PyArrow/datasets compatibility)
- **Frozen lockfile usage** for reproducible builds
- **VCS versioning** with hatch-vcs for automatic version management

### 5. **Semantic Release Automation**
- **Automatic tagging** based on commit message conventions
- **Version bumping** following semantic versioning rules
- **Auto-publishing** to PyPI on main branch merges
- **Changelog generation** and GitHub releases

### 6. **Integration & Reporting**
- SonarCloud integration for advanced analysis
- Codecov for coverage tracking
- GitHub Security tab integration
- Automated artifact management

## Environment Variables & Secrets

### Required Secrets
```yaml
OPENAI_API_KEY          # OpenAI API access (test execution)
ZOTERO_API_KEY          # Zotero integration (Talk2Scholars)
ZOTERO_USER_ID          # Zotero user identification
CODECOV_TOKEN           # Coverage reporting
SONAR_TOKEN             # SonarCloud analysis
PYPI_API_TOKEN          # PyPI publishing (semantic-release)
GITHUB_TOKEN            # GitHub API access (auto-provided)
```

### Security Best Practices
- All secrets managed through GitHub Secrets
- No hardcoded credentials in workflows
- Minimal permission scopes
- Secure artifact handling

## Monitoring & Troubleshooting

### Workflow Status Badges
Add these badges to your README for real-time status monitoring:

```markdown
[![Security Audit](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/security_audit.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/security_audit.yml)
[![SonarCloud](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/sonarcloud.yml)
```

### Common Issues & Solutions

#### Dependency Conflicts
If you encounter dependency issues:
```bash
# Use uv to update conflicting packages
uv add "package>=new-version"
uv lock --upgrade
```

#### LLM Test Failures
Some tests may fail in CI due to non-deterministic LLM responses:
- Tests include fallback logic for access failures
- Uses flexible keyword matching instead of exact strings
- Reports "unable to access" as valid responses

#### Coverage Threshold Failures
The workflows enforce 100% coverage. To handle this:
1. Add proper tests for uncovered code
2. Use coverage exclusions in `pyproject.toml` for legitimate cases
3. Check for import-time code paths not triggered in CI

#### Cross-Platform Issues
For Windows/Unix compatibility:
- Use `shell: bash` for consistent shell behavior
- Handle path differences (`.venv/Scripts/activate` vs `.venv/bin/activate`)
- Avoid PowerShell-specific commands

## Maintenance

### Regular Updates
- **Dependencies:** Dependabot manages updates automatically
- **Actions:** Update action versions quarterly
- **Python:** Update matrix versions as new releases become available

### Performance Optimization
- **Streamlined Design:** Simplified approach for reliability over speed
- **Parallelization:** Jobs run in parallel where possible
- **Dependency Resolution:** UV handles conflicts automatically
- **Resource Usage:** Optimized for GitHub Actions limits

## Migration Summary

This documentation reflects the complete migration from pip-based to UV-based workflows:

### ✅ **Successfully Modernized:**
- All test workflows with cross-platform compatibility
- Semantic release automation (restored from old system)
- Dependency conflict resolution (PyArrow/datasets)
- Security audit streamlining
- Package build validation
- Documentation deployment

### ✅ **Key Improvements:**
- Faster dependency installation with UV
- Automatic version management with VCS
- Streamlined workflow architecture
- Better error handling for LLM tests
- Cross-platform virtual environment support

This workflow architecture provides comprehensive quality assurance, security scanning, and deployment automation with modern tooling while maintaining full automation capabilities.
