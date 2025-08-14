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
- **Push to main branch**: Full analysis
- **Pull requests**: Differential analysis
- **Manual trigger**: On-demand analysis

### Quality Gates
- **PR Decoration**: Automatic PR comments with quality gate status
- **Failed Quality Gate**: Blocks merge if quality standards aren't met
- **Coverage**: Requires minimum coverage on new code

## Reports Generated

1. **Code Coverage Report**: `coverage.xml`
2. **PyLint Analysis**: `pylint-report.json`
3. **Security Scan**: `bandit-report.json`
4. **Dependency Security**: `pip-audit` and `safety` reports
5. **SonarCloud Dashboard**: Available on sonarcloud.io
6. **SARIF Reports**: Uploaded to GitHub Security tab

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
# Install SonarScanner
npm install -g sonarqube-scanner

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
   - Ensure `coverage.xml` is generated
   - Check file paths in `sonar-project.properties`

2. **Quality Gate Failure**
   - Review SonarCloud dashboard for specific issues
   - Address code smells and security hotspots

3. **Token Issues**
   - Regenerate SONAR_TOKEN in SonarCloud
   - Update GitHub repository secrets

### Configuration Files

- `sonar-project.properties`: Main SonarCloud configuration
- `pyproject.toml`: Coverage and tool integration settings
- `.github/workflows/sonarcloud.yml`: CI/CD workflow

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
