# For authors

## Description
Please:
1. Provide a summary of the modifications made and any associated issue (if applicable).
2. Include relevant context and motivation for the changes.
3. If this relates to a change in any website's frontend, kindly attach a screenshot of the adjustment from your localhost.
4. List any dependencies necessary for implementing this change.

## Fixes # (issue) Mention the issue number.

## Type of change
Please delete options that are not relevant.
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## How Has This Been Tested?
Please describe the tests you conducted to verify your changes. These may involve creating new test scripts or updating existing ones.
- [ ] Added new test(s) in the ```tests``` folder
- [ ] Added new function(s) to an existing test(s) (e.g.: ```tests/testX.py```)
- [ ] No new tests added (Please explain the rationale in this case)

## Checklist
- [ ] My code follows the style guidelines mentioned in the Code/DevOps guides
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation (e.g. MkDocs)
- [ ] My changes generate no new warnings
- [ ] I have added or updated tests (in the ```tests``` folder) that prove my fix is effective or that my feature works
- [ ] New and existing tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

# For reviewers

## Checklist pre-approval
- [ ] Is there enough documentation?
- [ ] If a new feature has been added, or a bug fixed, has a test been added to confirm good behavior?
- [ ] Does the test(s) successfully test edge/corner cases?
- [ ] Does the PR pass the tests? (if the repository has continuous integration)

## Checklist post-approval
- [ ] Does this PR merge ```develop``` into ```main```? If so, please make sure to add a prefix (feat/fix/chore) and/or a suffix BREAKING CHANGE (if it's a major release) to your commit message.
- [ ] Does this PR close an issue? If so, please make sure to descriptively close this issue when the PR is merged.

## Checklist post-merge
- [ ] When you approve of the PR, merge and close it (Read this [article](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github) to know about different merge methods on GitHub)
- [ ] Did this PR merge ```develop``` into ```main``` and is it suppose to run an automated release workflow (if applicable)? If so, please make sure to check under the "Actions" tab to see if the workflow has been initiated, and return later to verify that it has completed successfully.
