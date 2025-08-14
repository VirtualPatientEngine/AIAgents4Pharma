# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test coverage across all modules
- Modern CI/CD pipeline with UV package manager
- SonarCloud integration for code quality monitoring
- Automated dependency updates with Dependabot
- Security scanning with bandit and pip-audit
- Code formatting with black and ruff
- Type checking with mypy
- Comprehensive input validation for file uploads
- Multi-stage Docker builds with health checks
- Automated semantic release workflow
- GitHub issue and PR templates
- Auto-assignment of PR authors

### Changed
- Migrated from pip to UV for dependency management
- Updated to use pyproject.toml as single source of truth
- Improved documentation structure and organization
- Enhanced Docker configurations for production readiness

### Security
- Added comprehensive file upload validation
- Implemented MIME type checking and file size limits
- Added security scanning to CI pipeline
- Enhanced input sanitization

---

*This changelog is automatically updated by semantic-release.*
