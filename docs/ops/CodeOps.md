---
hide:
  - navigation
  - toc
---

<img src='../assets/VPE.png' height="50" width="50"></img>

# <font color=black>CodeOps</font>

> <font color=black>ℹ️</font><br>
> **Date** 2025-09-04<br>
> **Version** 2.0<br>

1. [Overview](#overview)
2. [GitHub Repo Organization](#github-repo-organization)
3. [GitHub templates for PRs, features and bugs](#github-templates-for-prs-features-and-bugs)
4. [Coding practices](#coding-practices)
5. [Testing locally](#testing-locally)
6. [Keywords in commit messages](#keywords-in-commit-messages)
7. [Resources](#resources)

<hr>

## Overview

Welcome to Team VPE’s CodeOps document!

This guide is maintained by Team VPE (Virtual Patient Engine). Learn more about the team and our mission here: https://bmedx.com/research-teams/artificial-intelligence/team-vpe/

This public-first repository encourages open collaboration while adhering to quality, security, and release standards suitable for production use.

This document serves as a guide to our team's approach to managing code. Your insights and feedback are highly encouraged 😊. Please provide feedback via [GitHub Issues](https://github.com/VirtualPatientEngine/AIAgents4Pharma/issues). Thanks 😊.

This guide will cover the following topics relevant for our CodeOps:.

🗸 **GitHub repo organization**: About the organization of the repository on GitHub.<br>
🗸 **GitHub templates**: Explains the predefined templates that are provided in a standardized format to provide details on proposed changes, new features, or bugs.<br>
🗸 **Coding Practices**: Outlines the coding standards that we strive to follow to ensure the quality, maintainability, and consistency of our codebase.<br>
🗸 **Testing locally**: Steps to run unit tests locally before committing your code, thereby reducing workflow failures on GitHub Actions prior to commit.<br>
🗸 **Keywords in Commit Messages**: Specific keywords in commit messages that can trigger release on GitHub Actions.<br>
🗸 **Resources for Further Reading**: Additional reading material.<br>

## GitHub Repo Organization

This repository is public-first and modular. The main Python package lives under `aiagents4pharma/`, with Streamlit apps in `app/`, and all tooling configured via `pyproject.toml` and managed by UV. Key paths:

| Folder or File                                | Description                                                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `.github/`                                    | CI/CD workflows and issue/PR templates                                                                                            |
| `aiagents4pharma/`                            | Main Python package with agents (`talk2biomodels`, `talk2knowledgegraphs`, `talk2scholars`, `talk2cells`, `talk2aiagents4pharma`) |
| `app/`                                        | Streamlit applications and UI utilities (e.g., secure file upload)                                                                |
| `docs/`                                       | MkDocs documentation (developer, ops, user docs)                                                                                  |
| `.github/workflows/`                          | GitHub Actions workflows (tests, security, SonarCloud, docker, release, docs)                                                     |
| `.gitignore`                                  | Ignore rules for Git                                                                                                              |
| `pyproject.toml`                              | Project config (PEP 621), tools (ruff, pylint, bandit, coverage, mypy)                                                            |
| `uv.lock`                                     | Locked dependencies for reproducible builds                                                                                       |
| `.pre-commit-config.yaml`                     | Pre-commit hooks (ruff, bandit, pip-audit, etc.)                                                                                  |
| `mkdocs.yml`                                  | Documentation site configuration                                                                                                  |
| `.coveragerc` or coverage in `pyproject.toml` | Coverage configuration used locally/CI                                                                                            |
| `sonar-project.properties`                    | SonarCloud project configuration                                                                                                  |
| `README.md`                                   | Project overview and contributor quickstart                                                                                       |
| `CONTRIBUTING.md`                             | Contribution guidelines                                                                                                           |
| `package.json`                                | Semantic-release tooling (node_modules not committed)                                                                             |
| `LICENSE`                                     | Project license                                                                                                                   |

## GitHub templates for PRs, features, and bugs

We have created three essential templates: the Pull Request (PR) template, the Feature Request template, and the Bug Report template (all present in the .github/ folder). Each serves a distinct purpose in streamlining our workflow and ensuring effective communication among team members and contributors.

1. The **PR template** serves as a structured guide for anyone submitting a pull request. It outlines essential details such as the purpose of the changes, any associated issues or feature requests, testing instructions, and any necessary documentation updates. By adhering to this template, contributors provide comprehensive context, making the review process smoother and more efficient.
2. The **Feature Request** template offers a standardized format for suggesting new functionalities or enhancements to our projects. It prompts users to describe the desired feature, its rationale, and any potential challenges or considerations.
3. The **Bug Report** template assists users in reporting issues or bugs encountered within our projects. It encourages clear and concise descriptions of the problem, including steps to reproduce, expected behavior, and any relevant screenshots or error messages.

Upon opening an issue on GitHub, users are prompted to select the appropriate template based on their specific needs—whether it's a bug report, or a feature request. Likewise, when initiating a pull request, the PR template automatically loads, guiding contributors through the necessary steps to ensure thorough documentation and review of their proposed changes.

## Coding practices

### Branching model

1. [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
2. How to use branches:
   1. Tagged branches (on VPE account) for releases.
   2. Main branch (on VPE account) to start all Feature/Fix branches (on private account).
   3. Merge of Feature/Fix branches (on private account) into Main (on VPE account) following a successful PR.
   4. Tag the Main (on VPE accounts) for a new release.

### Unit Tests

1. Each class/method should have a [unit test](https://en.wikipedia.org/wiki/Unit_testing)
2. The tests must cover at least the following:
   1. Unit testing: see [PyTest](https://docs.pytest.org/en/7.4.x/)
   2. Linting: see [Ruff](https://docs.astral.sh/ruff/) and [Pylint](https://pypi.org/project/pylint/)
   3. Code coverage: see [Coverage](https://coverage.readthedocs.io/en/7.3.2/)
   4. Works on Linux, Windows, and Mac machines

_NOTE: All tests must be written in the tests/ folder (this is where pytest will search by default)_

\*Pro-tips:

- Use the GitHub co-pilot to write docstrings (though not always accurate).
- Install PyLint on VS code to spot the linting errors on the fly.\*

### PR policies

1. Number of approving reviewers on a PR: >= 1
2. Passing unit testing (pytest)
3. Passing linting (pylint)
4. Passing coverage (coverage)

### Documentation of classes, methods, and APIs

Use [MkDocs](https://www.mkdocs.org/). Refer to the DevOps guide for more details.

### Best practices in Python

1. Choose your preferred Python version, but ensure your repository's environment passes tests on Windows, macOS, and Linux (you should be able to test that via GitHub actions).
2. Coding style is enforced via **Ruff**; prefer NumPy-style docstrings where applicable ([Style guide](https://numpydoc.readthedocs.io/en/latest/format.html))
3. Use modules and packages (add **init**.py)
4. One Class per script
5. Separate packages for utilities/helper functions
6. Import the module, not the function (call the function by accessing the module in the code)

```
# bad practice
import module1.pkg
result = pkg()
```

```
# good practice
import module1
result = module1.pkg()
# You always know where the function is coming from
# Avoids polluting the global name space
```

7. Readable code >> efficient code
8. Use type hinting whenever possible

```
def greeting(name: str) - > str:
  return "Hello "+name
```

9. Use List comprehension whenever possible (but don't forget the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle))

```
list = [x*2 for x in array if x > 1]
```

10. Docstring for methods and classes

```
def sum(a: int, b: int) -> int:
    """
    Function to return sum of 2 integers

    Args:
        a: first number
        b: second number

    Returns:
        int: sum of 2 integers
    """
    return (a + b)
```

_Pro-tip: Use co-pilot to automatically write a docstring for the methods/classes (though not always accurate)_

11. Examples: Jupyter notebook with the following
    1. Clear API calls to the sources of all data to run the analysis.
    2. Record of all analyses and figure generation routines.
    3. Documentation of the analysis so that another team member could reproduce the results in your absence.

 

## Testing locally

To streamline our development process and save time, we've implemented a CI/CD pipeline that includes automated testing through GitHub Actions (see the DevOps document for details). Essentially, each time code is pushed to GitHub, a TESTS workflow is triggered to test the code automatically. However, running these tests on GitHub Actions can be time-consuming. To optimize efficiency and catch issues early, **it's recommended to run the tests locally before committing changes to GitHub**. This involves executing pytest, pylint, and coverage tests (which are the core of the TESTS workflow) locally to ensure code quality and test coverage meet our standards. Below are the commands to execute these tests locally:

### pytest:

**Job**: test scripts in the tests/ folder<br>
**Passing-criteria**: pass all the tests<br>

```
uv run pytest
```

_Note: Running pytest without any options can sometimes execute all the python files, including unintended ones. To avoid this, you can specify the folders you want to test. For example, running_

```
uv run pytest tests/
```

_will execute pytest only on the tests/ folder. It is important to ensure that pytest is run on at least the `aiagents4pharma/` and `app/` folders. Additionally, if you choose to run pytest on specific folders while testing locally, mirror the same in CI where appropriate (see DevOps workflows)._

### ruff and pylint:

**Action**: lint/format and static analysis<br>
**Passing-criteria**: No errors; warnings per `pyproject.toml` configuration<br>

```
uv run ruff check --fix .
uv run ruff format .

uv run pylint aiagents4pharma/
```

_Notes:_

- Primary style/import checks use **ruff** (fast). Deep analysis uses **pylint** with disables configured in `pyproject.toml`.
- If rule exceptions are needed, update `pyproject.toml` so local and CI behavior match (see DevOps workflows). Avoid ad-hoc CLI disables.

### coverage:

**Job**: measure test coverage and enforce thresholds in CI<br>
**Passing-criteria**: As enforced by CI (per-workflow coverage and SonarCloud gates)

```
uv run coverage run -m pytest
uv run coverage report -m
```

_Notes: Component-specific coverage commands are available; coverage settings live in `pyproject.toml` and CI consumes XML artifacts._

### MkDocs:

**Job**: Hosts the documentation locally<br>
**Passing-criteria**: Manual assessment<br>

```
uv run mkdocs serve
```

_NOTE: Please refer to the "Unit Tests" subsection within the "Coding Practices" section for further details._

## Keywords in commit messages

We use [semantic-release](https://github.com/semantic-release/semantic-release) software to automate the process of versioning and releasing code on GitHub. It operates by analyzing commit messages to determine the type of changes made in the codebase. Following the Semantic Versioning convention, commonly known as SemVer, version numbers are structured as MAJOR.MINOR.PATCH. The MAJOR version is incremented for incompatible changes in the code, MINOR for feature additions, and PATCH for backward-compatible bug fixes. This automated approach ensures that version increments are consistent and meaningful, aiding developers and users in understanding the impact of updates.

**Job: Bump up the release (MAJOR.MINOR.PATCH) based on the commit message**

### “feat:”

will bump up the minor version (MINOR)

```
git commit –m “feat: add a new feature”
```

### “fix:”

will bump up the patch version (PATCH)

```
git commit –m “fix: fix bug”
```

### “feat:” or “fix:” followed by “BREAKING CHANGE:”

will bump up the major version (MAJOR)

```
git commit –m “feat: add a new feature
BREAKING CHANGE: update several features”
```

#### Notes:

1. The first line of the commit message must begin with the keyword “feat” or “fix” and the last line with "BREAKING_CHANGE" to prompt a major version bump.
2. Use conventional commits when merging feature branches into `main`. This ensures `release.yml` runs in GitHub Actions.
3. **The obligation of bumping up the major version lies with the reviewer** when changes are breaking (use `BREAKING CHANGE:` in body).

### “chore”

triggers no action.

```
git commit –m “chore: add new example in the folder
```

_Please specify conventional commit keywords when merging a pull request into `main` that will lead to a release. See the DevOps guide for PR practices and workflow details._

## Resources

- Outline of working with GitHub for collaborative projects [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow).
  Read more about: i. [GitHub Actions](https://docs.github.com/en/actions/using-workflows) workflows and ii. [Semantic versioning](https://semver.org/)
