# Testing, Linting, and Code Coverage with UV

This guide covers how to run tests, perform linting, and generate code coverage reports using UV after the migration from setuptools to modern UV-based dependency management.

## Quick Reference

```bash
# Install dependencies (production)
uv sync --frozen

# Install with development tools
uv sync --extra dev --frozen

# Run all tests
uv run pytest

# Run tests with coverage
uv run coverage run -m pytest
uv run coverage report

# Run linting
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues
uv run pylint aiagents4pharma/

# Run security scanning
uv run bandit -r aiagents4pharma/
uv run pip-audit

# Format code
uv run ruff format .

# Pre-commit hooks (runs all checks at once)
uv run pre-commit run --all-files

# Run individual pre-commit checks
uv run pre-commit run ruff --all-files         # Linting only
uv run pre-commit run ruff-format --all-files  # Formatting only
uv run pre-commit run bandit --all-files       # Security only
uv run pre-commit run pip-audit --all-files    # Vulnerabilities only
```

---

## Setup & Installation

### 1. Install UV

Follow the [official UV installation guide](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip
pip install uv
```

### 2. Install Dependencies

```bash
# Clone repository
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
cd AIAgents4Pharma

# Sync dependencies (creates .venv and installs everything)
uv sync --frozen

# Install with development dependencies (recommended)
uv sync --extra dev --frozen
```

### 3. System Prerequisites

**macOS/Linux users:**

```bash
# macOS
brew install libmagic

# Linux (Ubuntu/Debian)
sudo apt-get install libmagic1

# Linux (RHEL/CentOS)
sudo yum install file-libs
```

**Windows users:** libmagic is bundled automatically with python-magic.

---

## Package Management with UV

### Adding New Dependencies

#### Add Runtime Dependencies

```bash
# Add a new package to main dependencies
uv add "package-name"

# Add with specific version constraint
uv add "package-name>=1.0.0,<2.0.0"

# Add with exact version
uv add "package-name==1.2.3"

# Add from specific index
uv add "package-name" --index-url https://pypi.org/simple/
```

#### Add Development Dependencies

```bash
# Add to development dependencies
uv add --dev "pytest-mock"

# Add to specific extra group
uv add --extra dev "pip-audit==2.9.0"
```

#### Add from Git Repository

```bash
# Install from Git repository
uv add "git+https://github.com/user/repo.git"

# Install from specific branch/tag
uv add "git+https://github.com/user/repo.git@main"
uv add "git+https://github.com/user/repo.git@v1.0.0"
```

### Removing Dependencies

```bash
# Remove a package
uv remove "package-name"

# Remove development dependency
uv remove --dev "package-name"
```

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv add "package-name" --upgrade

# Update to latest compatible versions
uv sync --upgrade
```

### Managing Virtual Environments

```bash
# Create virtual environment (automatic with uv sync)
uv venv

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Install current project in development mode
uv pip install -e .

# List installed packages
uv pip list

# Show package information
uv pip show "package-name"
```

### Lock File Management

```bash
# Generate/update uv.lock
uv lock

# Install from lock file (exact versions)
uv sync --frozen

# Check for dependency conflicts
uv pip check
```

---

## Testing

### Running Tests

#### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest aiagents4pharma/talk2scholars/tests/test_main_agent.py

# Run specific test function
uv run pytest aiagents4pharma/talk2scholars/tests/test_main_agent.py::test_specific_function

# Run tests matching a pattern
uv run pytest -k "test_pdf"
```

#### Component-Specific Testing

```bash
# Test individual agents
uv run pytest aiagents4pharma/talk2scholars/tests/
uv run pytest aiagents4pharma/talk2biomodels/tests/
uv run pytest aiagents4pharma/talk2knowledgegraphs/tests/
uv run pytest aiagents4pharma/talk2aiagents4pharma/tests/
uv run pytest aiagents4pharma/talk2cells/tests/
```

### Submodule-Specific Quality Checks

#### Pylint for Individual Submodules

```bash
# Run pylint on specific submodules (configuration in pyproject.toml)
uv run pylint aiagents4pharma/talk2scholars/
uv run pylint aiagents4pharma/talk2biomodels/
uv run pylint aiagents4pharma/talk2knowledgegraphs/
uv run pylint aiagents4pharma/talk2aiagents4pharma/
uv run pylint aiagents4pharma/talk2cells/
```

#### Coverage for Individual Submodules

```bash
# Run coverage on specific submodules
uv run coverage run --include="aiagents4pharma/talk2scholars/*" -m pytest --cache-clear aiagents4pharma/talk2scholars/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2biomodels/*" -m pytest --cache-clear aiagents4pharma/talk2biomodels/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2knowledgegraphs/*" -m pytest --cache-clear aiagents4pharma/talk2knowledgegraphs/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2aiagents4pharma/*" -m pytest --cache-clear aiagents4pharma/talk2aiagents4pharma/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2cells/*" -m pytest --cache-clear aiagents4pharma/talk2cells/tests/ && uv run coverage report
```

#### Pre-commit Hooks for Specific Files/Directories

```bash
# Run pre-commit on specific submodule
uv run pre-commit run --files aiagents4pharma/talk2scholars/*.py
uv run pre-commit run --files aiagents4pharma/talk2biomodels/*.py
uv run pre-commit run --files aiagents4pharma/talk2knowledgegraphs/*.py

# Run specific hook on specific submodule
uv run pre-commit run ruff --files aiagents4pharma/talk2scholars/*.py
uv run pre-commit run bandit --files aiagents4pharma/talk2knowledgegraphs/*.py
```

#### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_default_fixture_loop_scope = "function"
markers = ["integration: marks tests as integration tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

#### Environment Variables for Testing

```bash
# Required for certain tests
export OPENAI_API_KEY="your-openai-key"
export ZOTERO_API_KEY="your-zotero-key"  # For Talk2Scholars tests
export ZOTERO_USER_ID="your-zotero-id"   # For Talk2Scholars tests

# Optional for LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-key"
```

---

## Code Coverage

### Basic Coverage Commands

#### Generate Coverage Reports

```bash
# Run tests with coverage
uv run coverage run -m pytest

# Show coverage report in terminal
uv run coverage report

# Generate HTML coverage report
uv run coverage html

# Generate XML coverage report (for CI/CD)
uv run coverage xml
```

#### Advanced Coverage Usage

**Component-Specific Coverage:**

```bash
# Coverage for specific component (standard configuration)
# Note: Use quotes on macOS/zsh shell, GitHub Actions CI doesn't require them
uv run coverage run --include="aiagents4pharma/talk2scholars/*" -m pytest --cache-clear aiagents4pharma/talk2scholars/tests/ && uv run coverage report

# Run coverage on each subfolder individually (local development - use quotes for macOS/zsh)
uv run coverage run --include="aiagents4pharma/talk2scholars/*" -m pytest --cache-clear aiagents4pharma/talk2scholars/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2biomodels/*" -m pytest --cache-clear aiagents4pharma/talk2biomodels/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2knowledgegraphs/*" -m pytest --cache-clear aiagents4pharma/talk2knowledgegraphs/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2aiagents4pharma/*" -m pytest --cache-clear aiagents4pharma/talk2aiagents4pharma/tests/ && uv run coverage report
uv run coverage run --include="aiagents4pharma/talk2cells/*" -m pytest --cache-clear aiagents4pharma/talk2cells/tests/ && uv run coverage report

# Alternative format for CI environments (without quotes)
# uv run coverage run --include=aiagents4pharma/talk2scholars/* -m pytest --cache-clear aiagents4pharma/talk2scholars/tests/

# Coverage with source specification (all components)
uv run coverage run --source=aiagents4pharma -m pytest

# Show missing lines
uv run coverage report -m

# Show coverage by file with line numbers
uv run coverage report --show-missing
```

**Coverage Configuration** (`pyproject.toml`):

```toml
[tool.coverage.run]
source = ["aiagents4pharma"]
omit = [
    "*/tests/*",
    "*/test_*",
    "docs/*",
    "app/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

[tool.coverage.xml]
output = "coverage.xml"
```

#### Coverage Thresholds

**Check Coverage Threshold:**

```bash
# Check if coverage meets minimum threshold
COVERAGE=$(uv run coverage report -m | awk 'END {print int($NF)}')
if [[ $COVERAGE -lt 80 ]]; then
    echo "Coverage is below 80%"
    exit 1
fi
```

**HTML Reports:** Open `htmlcov/index.html` after running `uv run coverage html` to see detailed coverage information with highlighted missing lines.

---

## Linting & Code Quality

### Ruff (Fast Python Linter)

#### Basic Ruff Commands

```bash
# Check all files
uv run ruff check .

# Check specific directory
uv run ruff check aiagents4pharma/

# Fix auto-fixable issues
uv run ruff check --fix .

# Check with specific rules
uv run ruff check --select E,W,F .

# Format code
uv run ruff format .
```

#### Ruff Configuration (`pyproject.toml`)

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
```

### Pylint (Comprehensive Static Analysis)

#### Basic Pylint Commands

```bash
# Run pylint on entire codebase
uv run pylint aiagents4pharma/

# Run on specific component
uv run pylint aiagents4pharma/talk2scholars/

# Standard configuration (disabled rules configured in pyproject.toml)
uv run pylint aiagents4pharma/

# Run on each subfolder individually
uv run pylint aiagents4pharma/talk2scholars/
uv run pylint aiagents4pharma/talk2biomodels/
uv run pylint aiagents4pharma/talk2knowledgegraphs/
uv run pylint aiagents4pharma/talk2cells/

# Generate JSON report
uv run pylint aiagents4pharma/ --output-format=json --reports=no > pylint-report.json

# Show only errors and warnings
uv run pylint aiagents4pharma/ --errors-only
```

#### Pylint Configuration (`pyproject.toml`)

```toml
[tool.pylint.MASTER]
extension-pkg-allow-list = ["pcst_fast"]

[tool.pylint.messages_control]
disable = ["R0801", "R0902", "W0221", "W0122"]
```

---

## Security Scanning

### Bandit (Security Linter)

#### Basic Bandit Commands

```bash
# Scan entire codebase
uv run bandit -r aiagents4pharma/

# Use configuration from pyproject.toml
uv run bandit -c pyproject.toml -r aiagents4pharma/

# Generate JSON report
uv run bandit -c pyproject.toml -f json -o bandit-report.json -r aiagents4pharma/

# Show only high severity issues
uv run bandit -r aiagents4pharma/ -ll
```

#### Bandit Configuration (`pyproject.toml`)

```toml
[tool.bandit]
exclude_dirs = ["tests", "test_*"]
skips = ["B101", "B601"]

[tool.bandit.assert_used]
skips = ["*_test.py", "*/test_*.py"]
```

### Vulnerability Scanning

#### pip-audit (Dependency Vulnerability Scanner)

```bash
# Basic vulnerability scan
uv run pip-audit

# Generate reports
uv run pip-audit --desc --format=json --output=audit-report.json
uv run pip-audit --desc --format=markdown --output=audit-report.md

# Scan specific requirements
uv run pip-audit --desc --format=table
```

#### safety (Alternative Vulnerability Scanner)

```bash
# Basic safety check
uv run safety check

# JSON output
uv run safety check --json --output safety-report.json

# Check specific files
uv run safety check --file=uv.lock
```

---

## Type Checking (MyPy)

### Current Status

MyPy is currently **disabled** in pre-commit due to 1,121 type annotation errors that require dedicated cleanup effort.

#### Manual MyPy Execution

```bash
# Run mypy (will show many errors)
uv run mypy aiagents4pharma/

# Run with lenient settings
uv run mypy aiagents4pharma/ --ignore-missing-imports --show-error-codes

# Check specific component
uv run mypy aiagents4pharma/talk2scholars/ --ignore-missing-imports
```

#### MyPy Configuration (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "setuptools.*",
    "pkg_resources.*",
]
ignore_missing_imports = true
```

---

## Pre-commit Hooks

### Installation & Setup

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff

# Update hooks to latest versions
uv run pre-commit autoupdate
```

### Pre-commit Configuration (`.pre-commit-config.yaml`)

The repository includes comprehensive pre-commit hooks:

- **Ruff**: Fast linting and code formatting
- **Bandit**: Security vulnerability scanning
- **General hooks**: Trailing whitespace, YAML validation, large file checks, etc.
- **pip-audit**: Dependency vulnerability scanning

### Individual Pre-commit Commands

```bash
# Run all hooks
uv run pre-commit run --all-files

# Run specific hooks
uv run pre-commit run ruff --all-files         # Linting with auto-fix
uv run pre-commit run ruff-format --all-files  # Code formatting
uv run pre-commit run bandit --all-files       # Security scanning
uv run pre-commit run pip-audit --all-files    # Dependency vulnerabilities

# General quality checks
uv run pre-commit run trailing-whitespace --all-files
uv run pre-commit run end-of-file-fixer --all-files
uv run pre-commit run check-yaml --all-files
uv run pre-commit run check-added-large-files --all-files

# Run on specific files
uv run pre-commit run ruff --files aiagents4pharma/talk2scholars/main.py
uv run pre-commit run bandit --files aiagents4pharma/talk2knowledgegraphs/tools/*.py
```

### Bypassing Hooks (Emergency Use)

```bash
# Skip all hooks for urgent commits
git commit -m "urgent fix" --no-verify

# Skip specific hooks
SKIP=ruff git commit -m "work in progress"
```

---

## Automation & CI Integration

### GitHub Actions Integration

All these tools are integrated into our CI/CD pipeline:

```bash
# Simulate CI locally
uv run pytest                                    # Tests
uv run coverage run -m pytest && coverage report # Coverage
uv run ruff check .                              # Linting
uv run pylint aiagents4pharma/                   # Static analysis
uv run bandit -c pyproject.toml -r aiagents4pharma/ # Security
uv run pip-audit                                 # Vulnerabilities
```

### Local Development Workflow

```bash
# 1. Make changes to code
git checkout -b feature/my-feature

# 2. Run tests and quality checks
uv run pytest
uv run coverage run -m pytest && uv run coverage report
uv run ruff check --fix .

# 3. Run pre-commit hooks
uv run pre-commit run --all-files

# 4. Commit changes
git add .
git commit -m "feat: add new feature"

# 5. Push and create PR
git push origin feature/my-feature
```

---

## Performance & Optimization

### UV Advantages

- **Speed**: UV is 10-100x faster than pip
- **Reliability**: Deterministic dependency resolution
- **Caching**: Smart caching reduces repeat installations
- **Lock files**: `uv.lock` ensures reproducible environments

### Development Tips

#### Faster Test Iteration

```bash
# Run tests with pytest-xdist for parallel execution
uv add --dev pytest-xdist
uv run pytest -n auto

# Use pytest cache for faster re-runs
uv run pytest --cache-clear  # Clear cache when needed
uv run pytest               # Reuse cache for speed
```

#### Efficient Coverage

```bash
# Run coverage only on changed files
git diff --name-only | grep "\.py$" | xargs uv run coverage run -m pytest

# Use coverage's fast mode
uv run coverage run --parallel-mode -m pytest
uv run coverage combine
```

---

## Troubleshooting

### Common Issues

#### Dependency Conflicts

```bash
# Fix dependency conflicts using uv
uv add "package>=new-version"
uv lock --upgrade

# Example: Fix PyArrow compatibility
uv add "pyarrow>=14.0.0"
uv add "datasets>=4.0.0"

# Clear and reinstall if needed
rm -rf .venv
uv sync --extra dev --frozen
```

#### Import Errors

```bash
# Ensure you're in the right environment
uv run python -c "import sys; print(sys.executable)"

# Check installed packages
uv pip list
```

#### Permission Issues (Linux/macOS)

```bash
# Fix ownership if needed
sudo chown -R $USER:$USER .venv

# Alternative: use --user flag
uv sync --user
```

#### Windows-Specific Issues

```powershell
# Use PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Use forward slashes in paths
uv run pytest aiagents4pharma/talk2scholars/tests/
```

### Getting Help

- **UV Documentation**: https://docs.astral.sh/uv/
- **pytest Documentation**: https://docs.pytest.org/
- **Repository Issues**: https://github.com/VirtualPatientEngine/AIAgents4Pharma/issues

This guide provides a comprehensive overview of testing, linting, and code coverage using UV. The modern toolchain ensures fast, reliable, and secure development practices.
