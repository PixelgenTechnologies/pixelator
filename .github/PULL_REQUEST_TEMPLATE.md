<!--
# Pixelgen Technologies pull request template

Many thanks for contributing to it if you think it could be improved. This is a self-improving process!

Please fill in the appropriate checklist below (delete whatever is not relevant).
These are the most common things requested on pull requests (PRs).

Remember that PRs should be made against the dev branch, unless you're preparing a pipeline release.

Learn more about contributing: [CONTRIBUTING.md](https://github.com/PixelgenTechnologies/pixelator/blob/main/CONTRIBUTING.md)
-->

## Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce. Please also list any relevant details for your test configuration

- [ ] Test A
- [ ] Test B

**Test Configuration**:
* Firmware version:
* Hardware:
* Toolchain:
* SDK:

## PR checklist:

- [ ] This comment contains a description of changes (with reason).
- [ ] My code follows the style guidelines of this project
- [ ] Make sure your code lints, is well-formatted and type checks
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] If you've added a new stage - have you followed the pipeline CLI conventions in the [USAGE guide](../USAGE.md)
- [ ] [README.md](./README.md) is updated
- [ ] If a new tool or package is included, update poetry.lock, [the conda recipe](../conda-recipe/pixelator/meta.yaml) and [cited it properly](../CITATIONS.md)
- [ ] Include any new [authors/contributors](../AUTHORS.md)
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
- [ ] I have checked my code and documentation and corrected any misspellings
- [ ] Usage Documentation is updated
- [ ] If you are doing a [release](../RELEASING.md#Releasing), or a significant change to the code, update [CHANGELOG.md](../CHANGELOG.md)
