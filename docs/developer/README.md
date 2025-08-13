# AIAgents4Pharma Developer Guide

This guide covers the complete development setup, tooling, and workflow for AIAgents4Pharma project.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Development Tools](#development-tools)
5. [Code Quality & Security](#code-quality--security)
6. [Dependency Management](#dependency-management)
7. [Testing](#testing)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Docker & Deployment](#docker--deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (modern Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
cd AIAgents4Pharma

# 2. Install dependencies (creates virtual environment automatically)
uv sync --extra dev

# 3. Set up pre-commit hooks (optional but recommended)
uv run pre-commit install

# 4. Set up API keys
export OPENAI_API_KEY=sk-...
export NVIDIA_API_KEY=nvapi-...
export ZOTERO_API_KEY=...
export ZOTERO_USER_ID=...

# 5. Test installation
uv run python -c "import aiagents4pharma; print('✅ Installation successful!')"
```

---

## 🛠 Development Environment Setup

### Modern Python Stack

This project uses a modern Python development stack:

- **📦 uv**: Ultra-fast Python package manager and dependency resolver
- **🏗️ hatchling**: Modern build backend (PEP 621 compliant)
- **📝 pyproject.toml**: Single source of truth for project configuration
- **🔒 uv.lock**: Reproducible dependency resolution

### Why uv over pip/conda?

- **10-100x faster** than pip for dependency resolution
- **Automatic virtual environment management**
- **Built-in lock file support** for reproducible builds
- **Better dependency conflict resolution**
- **Native pyproject.toml support**

---

## 📂 Project Structure

```
AIAgents4Pharma/
├── aiagents4pharma/           # Main package
│   ├── talk2biomodels/        # Systems biology agent
│   ├── talk2knowledgegraphs/  # Knowledge graph agent
│   ├── talk2scholars/         # Scientific literature agent
│   ├── talk2cells/            # Single cell analysis agent
│   └── talk2aiagents4pharma/  # Meta-agent (orchestrator)
├── app/                       # Streamlit applications
├── docs/                      # Documentation
├── pyproject.toml            # Project configuration (dependencies, tools)
├── uv.lock                   # Lock file for reproducible builds
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
└── release_version.txt       # Version file
```

---

## 🔧 Development Tools

### Code Quality Tools

All tools are configured in `pyproject.toml` and run automatically via pre-commit:

#### 🎨 **Black** - Code Formatting
```bash
# Format all code
uv run black .

# Check formatting without applying
uv run black --check .
```

#### ⚡ **Ruff** - Fast Linting & Import Sorting
```bash
# Lint and auto-fix issues
uv run ruff check --fix .

# Check only (no fixes)
uv run ruff check .

# Format imports and code style
uv run ruff format .
```

#### 🔍 **MyPy** - Static Type Checking
```bash
# Type check the main package
uv run mypy aiagents4pharma/

# Type check everything
uv run mypy .
```

### Security Tools

#### 🛡️ **Bandit** - Security Vulnerability Scanner
```bash
# Scan for security issues
uv run bandit -r aiagents4pharma/

# Generate detailed report
uv run bandit -r aiagents4pharma/ -f json -o security-report.json
```

#### 🔒 **Dependency Vulnerability Scanning**
```bash
# Scan dependencies for known vulnerabilities
uv run pip-audit

# Alternative scanner
uv run safety check

# Scan with detailed output
uv run pip-audit --desc --format=json
```

---

## 🔄 Code Quality & Security

### Pre-commit Hooks

Pre-commit runs automatically before every commit to ensure code quality:

```bash
# Install hooks (one-time setup)
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run black
uv run pre-commit run ruff
uv run pre-commit run mypy
```

### What runs on each commit:
1. **Black** - Formats code
2. **Ruff** - Lints and fixes imports
3. **MyPy** - Type checking
4. **Bandit** - Security scanning
5. **Safety** - Dependency vulnerability check
6. **General checks** - Trailing whitespace, YAML validation, etc.

### Bypassing Pre-commit (Emergency Only)
```bash
# Skip pre-commit hooks (not recommended)
git commit --no-verify -m "emergency fix"
```

---

## 📦 Dependency Management

### Adding Dependencies

```bash
# Add runtime dependency
uv add numpy>=1.24.0

# Add development dependency
uv add --group dev pytest>=7.0.0

# Add optional dependency group
uv add --optional ml torch>=2.0.0
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv add package_name@latest

# Update dev dependencies
uv sync --extra dev --upgrade
```

### Lock File Management

```bash
# Generate/update lock file
uv lock

# Install from lock file (production)
uv sync --frozen

# Install with development tools
uv sync --extra dev --frozen
```

### Dependency Groups

- **Main**: Core runtime dependencies
- **Dev**: Development tools (black, ruff, mypy, etc.)
- **Optional**: Feature-specific dependencies

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=aiagents4pharma

# Run specific test file
uv run pytest aiagents4pharma/talk2biomodels/tests/test_api.py

# Run integration tests only
uv run pytest -m integration
```

### Test Categories

- **Unit tests**: Fast, isolated tests
- **Integration tests**: Cross-component tests (marked with `@pytest.mark.integration`)

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

The project uses GitHub Actions for automated testing and deployment:

```yaml
# .github/workflows/ci.yml (example)
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync --extra dev
      - run: uv run pytest
      - run: uv run pip-audit
```

### Manual CI Commands

```bash
# Run the same checks as CI locally
uv run pytest                    # Tests
uv run pip-audit                # Security scan
uv run bandit -r aiagents4pharma/ # Security scan
uv run mypy aiagents4pharma/     # Type checking
```

---

## 🐳 Docker & Deployment

### Building Docker Images

Each agent has its own Dockerfile:

```bash
# Build specific agent
docker build -f aiagents4pharma/talk2scholars/Dockerfile -t talk2scholars .

# Build all with docker-compose
docker-compose build
```

### Production Deployment

```bash
# Install production dependencies only
uv sync --frozen --no-dev

# Build production package
uv build

# Install built package
pip install dist/aiagents4pharma-*.whl
```

---

## 🚨 Security Best Practices

### Regular Security Scans

```bash
# Weekly security scan
uv run pip-audit
uv run safety check
uv run bandit -r aiagents4pharma/

# Check for outdated packages with vulnerabilities
uv run pip-audit --desc
```

### Dependency Updates

- **Dependabot** automatically creates PRs for security updates
- **Pre-commit hooks** catch vulnerabilities before commit
- **CI pipeline** blocks PRs with security issues

### API Key Management

```bash
# Set environment variables (never commit these!)
export OPENAI_API_KEY=sk-...
export NVIDIA_API_KEY=nvapi-...

# Use .env file for local development (add to .gitignore!)
echo "OPENAI_API_KEY=sk-..." >> .env
```

---

## 🛠 Common Development Tasks

### Starting Development

```bash
# 1. Activate environment and install dependencies
uv sync --extra dev

# 2. Run pre-commit setup
uv run pre-commit install

# 3. Start coding!
```

### Before Committing

```bash
# 1. Run quality checks
uv run ruff check --fix .
uv run black .
uv run mypy aiagents4pharma/

# 2. Run tests
uv run pytest

# 3. Security scan
uv run pip-audit

# 4. Commit (pre-commit will run automatically)
git add .
git commit -m "your message"
```

### Adding a New Agent

1. Create new directory: `aiagents4pharma/talk2newagent/`
2. Add dependencies to `pyproject.toml`
3. Update package configuration
4. Add tests and documentation
5. Update Docker configuration

---

## 🐛 Troubleshooting

### Common Issues

#### Dependency Conflicts
```bash
# Clear cache and reinstall
rm -rf .venv uv.lock
uv sync --extra dev
```

#### Pre-commit Issues
```bash
# Reinstall hooks
uv run pre-commit uninstall
uv run pre-commit install

# Update hook versions
uv run pre-commit autoupdate
```

#### Import Errors
```bash
# Verify installation
uv run python -c "import aiagents4pharma; print('OK')"

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

#### Type Checking Errors
```bash
# Install missing type stubs
uv add --group dev types-requests types-PyYAML

# Run with verbose output
uv run mypy --verbose aiagents4pharma/
```

### Performance Issues

```bash
# Profile dependency resolution
uv sync --extra dev --verbose

# Check lock file
uv lock --verbose
```

---

## 📚 Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Black Code Style](https://black.readthedocs.io/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [MyPy Configuration](https://mypy.readthedocs.io/en/stable/config_file.html)
- [Pre-commit Hooks](https://pre-commit.com/)

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Setup** development environment: `uv sync --extra dev`
4. **Install** pre-commit: `uv run pre-commit install`
5. **Make** changes and ensure all checks pass
6. **Commit** with descriptive message
7. **Push** to your fork and create Pull Request

All contributions are automatically scanned for:
- Code formatting and style
- Type safety
- Security vulnerabilities
- Test coverage

---

**Happy coding! 🚀**
