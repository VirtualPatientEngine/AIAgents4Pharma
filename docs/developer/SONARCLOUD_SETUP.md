# SonarCloud Integration Setup Guide

## Overview

SonarCloud provides advanced code quality analysis, security scanning, and technical debt tracking for the AIAgents4Pharma project. This document outlines the setup and configuration process.

## Setup Steps

### 1. SonarCloud Account Setup

1. **Create SonarCloud Account**: Go to [sonarcloud.io](https://sonarcloud.io) and sign in with GitHub
2. **Import Repository**: Import the `VirtualPatientEngine/AIAgents4Pharma` repository
3. **Generate Token**: Go to Account → Security → Generate new token

### 2. GitHub Repository Configuration

Add the following secrets to your GitHub repository settings:

```bash
# Repository Settings → Secrets and variables → Actions
SONAR_TOKEN=your_sonarcloud_token_here
```

### 3. SonarCloud Project Configuration

The project is configured with:

- **Project Key**: `VirtualPatientEngine_AIAgents4Pharma`
- **Organization**: `virtualpatientengine`
- **Quality Gate**: Uses SonarCloud's default quality gate

### 4. Analysis Configuration

The analysis includes:

#### Code Quality Metrics
- **Code Coverage**: Minimum 80% coverage required
- **Duplicated Code**: Less than 3% duplication
- **Maintainability Rating**: A rating required
- **Reliability Rating**: A rating required
- **Security Rating**: A rating required

#### Security Analysis
- **Security Hotspots**: Automatic detection of security issues
- **Vulnerabilities**: OWASP Top 10 compliance checking
- **Code Smells**: Anti-pattern detection

#### Technical Debt
- **Maintainability**: Technical debt ratio tracking
- **Code Complexity**: Cyclomatic complexity analysis
- **Test Coverage**: Line and branch coverage analysis

## Workflow Integration

### Automatic Analysis
The SonarCloud workflow runs on:
- **Pull requests to `main`**
- **Pushes to `main`**
- **Manual trigger** via `workflow_dispatch`

### Current CI/CD Approach
- **Self-contained**: SonarCloud generates its own coverage, pylint, and bandit reports
- **Deterministic**: One workflow produces the full analysis input set
- **No workflow cascade**: Analysis no longer depends on downloading artifacts from other workflows
- **Ubuntu-only execution**: SonarCloud uses one Linux job to produce consistent Python reports

### Workflow Architecture
```text
PR/Push → SonarCloud Analysis
         ├─ Install uv + Python 3.12
         ├─ Generate bandit JSON for each module
         ├─ Generate pylint JSON for each module
         ├─ Run module tests with coverage XML output
         └─ Upload all reports to SonarCloud in one scan
```

### Workflow Triggers
- **pull_request**: Runs on PRs targeting `main`
- **push**: Runs on changes merged to `main`
- **workflow_dispatch**: Manual trigger for on-demand analysis

### Quality Gates
- **Coverage** is only one input
- **Security Hotspots** are separate from bandit and can fail the gate independently
- **Duplication on New Code** is a separate metric and is not affected by local pytest coverage

## Reports Generated

### Generated During Sonar Workflow
1. **Coverage XML reports**: One per module
2. **PyLint JSON reports**: One per module
3. **Bandit JSON reports**: One per module

### SonarCloud Outputs
4. **SonarCloud Dashboard**: Available on sonarcloud.io with comprehensive analysis
5. **PR Decoration**: Quality gate feedback on pull requests
6. **Quality Gate Status**: Pass/fail based on SonarCloud rules

## Quality Standards

### Coverage Requirements
- **Overall Coverage**: ≥ 80%
- **New Code Coverage**: ≥ 90%
- **Duplicated Lines**: < 3%

### Security Standards
- **Security Rating**: A (no vulnerabilities)
- **Security Hotspots**: All reviewed
- **Bandit Issues**: Critical issues must be resolved
- **Dependency Security**: pip-audit and safety scans clean
- **File Upload Security**: Streamlit uploads validated and secure

### Maintainability
- **Maintainability Rating**: A
- **Code Smells**: < 10 per 1000 lines
- **Technical Debt**: < 5% of total development time

## Local Analysis

Run SonarCloud analysis locally:

```bash
# Install SonarScanner on macOS
brew install sonar-scanner

# Run analysis
sonar-scanner \
  -Dsonar.projectKey=VirtualPatientEngine_AIAgents4Pharma \
  -Dsonar.organization=virtualpatientengine \
  -Dsonar.sources=aiagents4pharma \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.login=your_token_here
```

## Troubleshooting

### Common Issues

1. **Coverage Not Detected**
   - Ensure the SonarCloud workflow generated the expected `coverage-*.xml` files
   - Check report paths in `sonar-project.properties`

2. **Quality Gate Failure**
   - Coverage and pylint can be green while the gate still fails on:
     - Security Hotspots
     - Duplication on New Code
   - Review the SonarCloud dashboard for the exact files flagged

3. **Unexpected Duplication or Hotspots**
   - Verify non-source files are excluded from SonarCloud analysis
   - The repo intentionally excludes `.venv`, `.egg-info`, `.env`, Dockerfiles, install docs, and boilerplate state/config modules

4. **Token Issues**
   - Regenerate SONAR_TOKEN in SonarCloud
   - Update GitHub repository secrets

### Configuration Files

- `pyproject.toml`:
  - SonarCloud project settings (`[tool.sonarcloud]`)
  - Pylint configuration (`[tool.pylint.messages_control]`)
  - Coverage settings (`[tool.coverage.*]`)
  - Bandit security configuration (`[tool.bandit]`)
- `.github/workflows/sonarcloud.yml`: Self-contained SonarCloud workflow
- `sonar-project.properties`: SonarCloud source, report, and exclusion rules

## Benefits

### For Developers
- **Real-time Feedback**: Immediate code quality feedback
- **Security Awareness**: Automated security vulnerability detection
- **Best Practices**: Enforcement of coding standards

### For Project
- **Quality Assurance**: Consistent code quality across contributors
- **Technical Debt Management**: Tracking and reduction of technical debt
- **Security Compliance**: Continuous security monitoring

### For Users
- **Reliability**: Higher code quality leads to fewer bugs
- **Security**: Enhanced security through automated scanning
- **Maintainability**: Easier to contribute to well-maintained codebase
