---
hide:
  - navigation
  - toc
---

<img src='../VPE.png' height="50" width="50"></img>

# <font color=black>CodeOps</font>
> <font color=black>ℹ️</font><br>
> **Date** 2024-11-02<br>
> **Version** 1.0<br>

1. [Overview](#overview)
2. [GitHub Repo Types](#github-repo-types)
3. [GitHub templates for PRs, features and bugs](#github-templates-for-prs-features-and-bugs)
4. [Coding practices](#coding-practices)
5. [Testing locally](#testing-locally)
6. [Keywords in commit messages](#keywords-in-commit-messages)
7. [Resources](#resources)

<hr>

## Overview
Welcome to Team VPE’s CodeOps document!

This document serves as a guide to our team's approach to managing code. Your insights and feedback are highly encouraged 😊. Please provide feedback via [GitHub Issues](https://github.com/VirtualPatientEngine/AIAgents4Pharma/issues). Thanks 😊.

This guide will cover the following topics relevant for our CodeOps:.

🗸 **GitHub repo organization**: About the organization of the repository on GitHub.<br>
🗸 **GitHub templates**: Explains the predefined templates that are provided in a standardized format to provide details on proposed changes, new features, or bugs.<br>
🗸 **Coding Practices**: Outlines the coding standards that we strive to follow to ensure the quality, maintainability, and consistency of our codebase.<br>
🗸 **Testing locally**: Steps to run unit tests locally before committing your code, thereby reducing workflow failures on GitHub Actions prior to commit.<br>
🗸 **Keywords in Commit Messages**: Specific keywords in commit messages that can trigger release on GitHub Actions.<br>
🗸 **Resources for Further Reading**: Additional reading material.<br>

## GitHub Repo Organization
This repository is intended to be public facing, encouraging easy collaboration, and sharing within the wider community. It follows a modular structure that allows developers to focus on individual AI Agent modules or compose a complete application using all or some of the available AI Agents. Our folder structure is as follows:

| Folder or File | Description |
| -------------- | ----------- |
| .gitignore | Path to files and folders to be ignored |
| .github/ | Workflows for continuous integration (CI) and templates (Bug/Feature/PR) |
| app/ | Where the application code sits |
| app/\<frontend or backend\>/src/ | Code related to the client web UI or server backend, respectively |
| app/\<frontend or backend\>/tests/ | All pytests for the frontend or backend |
| app/\<frontend or backend\>/docs/ | All documentation for the frontend or backend |
| app/\<frontend or backend\>/pyproject.toml | List all the packages required for the front-end or backend |
| app/\<frontend or backend\>/LICENSE | If differing for the frontend or backend compared to the LICENSE of the repo |
| app/\<frontend or backend\>/README.md | Description of the frontend or backend |
| agents/ | Where the AI agents code sits |
| agents/\<agent\>/src/ |Code related to a particular <agent> |
| agents/\<agent\>/src/models/ | All code that is specific to defining the decision making of a particular \<agent\> |
| agents/\<agent\>/src/tools/ | All code that is specific to defining functionality of tools available to a particular \<agent\> |
| agents/\<agent\>/src/prompts/ | Prompts that are specific to a particular \<agent\> |
| agents/\<agent\>/tests/ | All pytests for a particular \<agent\> |
| agents/\<agent\>/docs/ | All documentation for a particular \<agent\> |
| agents/\<agent\>/examples/ | Notebooks exemplifying how to use a particular \<agent\> |
| agents/\<agent\>/pyproject.toml | Python installation script for a particular \<agent\> |
| agents/\<agent\>/LICENSE | If differing for a particular \<agent\> compared to the LICENSE of the repo |
| agents/\<agent\>/README.md | Description of a particular \<agent\> |
| docs/ | Where you write .md files for MkDocs for the repository website |
| env/ | Dockerfiles or scripts for setting up a virtual environment for development or deployment |
| node_modules | Packages required by sematic-release (do not modify/delete) |
| pyproject.toml | Build script for the repository following [PEP 518](https://peps.python.org/pep-0518/) |
| LICENSE | You know what it means |
| README.md | Description of your repo |
| CONTRIBUTING.md | Contributing guidelines |
| *.yml Files | that come with semantic-release and MkDocs |
| *.json | They are pre-configured |
| *.js | Modify them based on your need |

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
    2. Linting: see [Pylint](https://pypi.org/project/pylint/)
    3. Code coverage: see [Coverage](https://coverage.readthedocs.io/en/7.3.2/)
    4. Works on Linux, Windows, and Mac machines

*NOTE: All tests must be written in the tests/ folder (this is where pytest will search by default)*

*Pro-tips:
  - Use the GitHub co-pilot to write docstrings (though not always accurate).
  - Install PyLint on VS code to spot the linting errors on the fly.*

### PR policies
1. Number of approving reviewers on a PR: >= 1
2. Passing unit testing (pytest)
3. Passing linting (pylint)
4. Passing coverage (coverage)

### Documentation of classes, methods, and APIs
Use [MkDocs](https://www.mkdocs.org/). Refer to the DevOps guide for more details.

### Best practices in Python
1. Choose your preferred Python version, but ensure your repository's environment passes tests on Windows, macOS, and Linux (you should be able to test that via GitHub actions).
2. Coding style -> Numpy style ([Style guide](https://numpydoc.readthedocs.io/en/latest/format.html)) for Python using Flake8
3. Use modules and packages (add __init__.py)
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

*Pro-tip: Use co-pilot to automatically write a docstring for the methods/classes (though not always accurate)*

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
pytest
```

*Note: Running pytest without any options can sometimes execute all the python files, including unintended ones. To avoid this, you can specify the folders you want to test. For example, running*

```
pytest tests/
```

*will execute pytest only on the tests/ folder. It is important to ensure that pytest is run on at least the app/ and agents/ folders. Additionally, if you choose to run pytest on specific folders while testing locally, you must also do the same in the workflow files for GitHub actions (see the section on Automated workflows in the DevOps guide).*

### pylint:
**Action**: lint all *.py scripts in the specified folder<br>
**Passing-criteria**: code rating must be **10.00/10.00**<br>

```
pylint app/
pylint agents/
```

*Note: If you want to disable a particular warning, use the disable option in pylint.
For example, running*

```
pylint --disable=R0801,W0613 app/
pylint --disable=R0801,W0613 agents/
```

*will ignore the warnings with codes [R0801](https://pylint.readthedocs.io/en/stable/user_guide/messages/refactor/duplicate-code.html) and [W0613](https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/unused-argument.html). Choose to disable warnings wisely. Additionally, if you choose to disable a warning while testing locally, you must also disable it in the workflow files for GitHub Actions (see the section on Automated workflows in the DevOps guide). We have already disabled a few warnings. Please look at the [tests.yml](https://github.com/VirtualPatientEngine/AIAgents4Pharma/main/.github/workflows/tests.yml) to know the warnings we have currently disabled.*

### coverage:
**Job**: makes sure every method is called at least once in the tests/ folder<br>
**Passing-criteria**: 100% score<br>

```
coverage run –m pytest agents app
coverage report –m
```

*Note: Lines to be excluded should be specified in the file .coveragerc*

### MkDocs:
**Job**: Hosts the documentation locally<br>
**Passing-criteria**: Manual assessment<br>

```
mkdocs serve
```

*NOTE: Please refer to the "Unit Tests" subsection within the "Coding Practices" section for further details.*

## Keywords in commit messages
We use `python-semantic-release` to automate versioning and GitHub releases. It analyzes conventional commit messages and applies Semantic Versioning (`MAJOR.MINOR.PATCH`) based on the highest-impact change since the last tag.

**Job: Bump up the release (MAJOR.MINOR.PATCH) based on the commit message**

### `feat:`
will bump up the minor version (MINOR)

```
git commit –m “feat: add a new feature”
```

### `fix:`
will bump up the patch version (PATCH)

```
git commit –m “fix: fix bug”
```

### `feat!:` / `fix!:` or `BREAKING CHANGE:`
will bump up the major version (MAJOR)

```
git commit –m “feat: add a new feature
BREAKING CHANGE: update several features”
```

#### Notes:
1. Releases are cut from `main`.
2. The merge or squash commit message on `main` must carry the intended conventional-commit prefix.
3. Breaking changes should be marked explicitly with `!` or `BREAKING CHANGE:`.

### `chore:`
triggers no action.

```
git commit –m “chore: add new example in the folder
```

*Please note it is mandatory to use conventional commit keywords in the final commit message that lands on `main` if you expect a semantic version bump.*

## Resources
- Outline of working with GitHub for collaborative projects [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow).
Read more about: i. [GitHub Actions](https://docs.github.com/en/actions/using-workflows) workflows and ii. [Semantic versioning](https://semver.org/)
