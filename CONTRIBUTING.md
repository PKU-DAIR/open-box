# Contributing to This Project
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

### Run Test Locally
**Before submitting a pull request**, please make sure the test suite passes.
You may need to modify or add new test cases in `test/`.

To install the test dependencies, run `pip install "openbox[test]"` or `pip install -r requirements/dev/test.txt`.

To run test locally:
```bash
pytest -rap test
```

To see output of each test case:
```bash
pytest -rap --durations=20 --verbose --capture=tee-sys test
```

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

We provide beginner-friendly issue templates to help you get started.
Try [filling an issue](https://github.com/PKU-DAIR/open-box/issues)!

## License
By contributing to this project, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
