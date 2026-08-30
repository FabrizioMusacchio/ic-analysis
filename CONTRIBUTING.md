# How to Contribute

Thank you for your interest in contributing to the IntelliCage Place Learning
Toolkit. This project welcomes improvements to the code, documentation, tests,
synthetic example data, and analysis workflows.

The goal of the toolkit is to provide a robust, transparent, and reproducible
Python interface for analyzing place learning experiments conducted in
IntelliCage.

## Before You Start

Please check the GitHub issue tracker to see whether your idea, bug report, or
enhancement has already been discussed:

[https://github.com/FabrizioMusacchio/ic-placelearning/issues](https://github.com/FabrizioMusacchio/ic-placelearning/issues)

For small fixes such as typos or minor documentation improvements, opening a
pull request directly is fine.

For larger changes, please open or comment on an issue with:

- what you would like to change or add
- why it is useful for IntelliCage place-learning analysis
- expected input data structure and output behavior
- edge cases, assumptions, or testing ideas

## Development Environment

The package requires Python 3.10 or newer. Python 3.12 is recommended for
development.

```bash
git clone https://github.com/FabrizioMusacchio/ic-placelearning.git
cd ic-placelearning

conda create -n ic_placelearning python=3.12 -y
conda activate ic_placelearning

pip install -e ".[dev]"
```

## Making Changes

All code contributions should be submitted as pull requests against the `main`
branch.

A recommended workflow:

1. Create a feature branch.
2. Implement a focused change.
3. Add or update tests for user-visible behavior.
4. Run the test suite locally.
5. Open a pull request with a concise title and explanation.

Clear docstrings are expected for new public functions. Please document
assumptions about IntelliCage export structure, phase naming, timing alignment,
or behavioral metric definitions when they affect interpretation.

## Testing

Run the full test suite with:

```bash
pytest
```

The public package should maintain at least 75% coverage. Tests should be small
and self-contained. Use the synthetic dataset generator or minimal generated
tables instead of real experimental data.

## Analysis Policy Decisions

Changes to metric definitions should be explicit, documented, and tested. In
particular, please be careful with:

- phase-number and phase-name mapping
- assigned-corner logic for phase 3 and phase 4
- rewarded correct-corner definitions
- awake/sleep and mouse-day alignment
- responder thresholds and onset calculations
- statistical model assumptions and fallback behavior

Pull requests that change these semantics should explain how old and new
outputs differ.

## Data Policy

Do not add private or unpublished experimental IntelliCage datasets to the
repository. Public examples should use synthetic data or small anonymized test
fixtures that do not expose animal-level experimental records.

Large generated result folders should not be committed. They should be
reproducible from scripts and synthetic input data.

## Commit Conventions

Clear commit messages help keep the project history readable. Prefixes inspired
by Conventional Commits are encouraged:

- `feat:` new functionality
- `fix:` bug fixes
- `docs:` documentation changes
- `refactor:` internal code restructuring without behavior changes
- `test:` adding or modifying tests
- `chore:` maintenance tasks or tooling updates

Example:

```text
test: cover aligned phase-window metrics
```

## License and Contributions

By submitting a pull request, you agree that your contributions will be released
under the project license as specified in the repository.
