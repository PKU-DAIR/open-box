# Run Test

OpenBox uses `pytest` as the test framework.
Tests are automatically run by GitHub Actions when a commit is pushed (to master branch) or a pull request is opened.
Tests can also be run locally.

## For pytest

See `[tool.pytest.ini_options]` in `pyproject.toml`.
**Options that are not specified in command line are read from this file.**

Pytest documentation: https://docs.pytest.org/

Pytest will run all files of the form `test_*.py` or `*_test.py` in `testpath` directories (`test/` in this repo). From those files, collect test items:
+ `test` prefixed test functions or methods outside of class.
+ `test` prefixed test functions or methods inside `Test` prefixed test classes (without an `__init__` method). Methods decorated with `@staticmethod` and `@classmethods` are also considered.

Test discovery rules: https://docs.pytest.org/en/8.0.x/explanation/goodpractices.html#test-discovery


Example command:
```bash
pytest -rap test
pytest -h  # show help
```

Use `-r[chars]` to show extra test summary info as specified by chars:
```
(f)ailed, (E)rror, (s)kipped, (x)failed, (X)passed,
(p)assed, (P)assed with output, (a)ll except passed
(p/P), or (A)ll. (w)arnings are enabled by default (see
--disable-warnings), 'N' can be used to reset the list.
(default: 'fE').
```


## For GitHub Actions

See `.github/workflows/test.yml` for the configuration of GitHub Actions.

Documentation: https://docs.github.com/actions

An example repository with GitHub Actions for testing:
https://github.com/jhj0411jhj/test_action


## Code coverage

codecov: https://app.codecov.io

openbox codecov: https://app.codecov.io/gh/PKU-DAIR/open-box

To manage codecov of OpenBox, you need to be a member of PKU-DAIR organization.

Codecov report is automatically generated and uploaded.


## TODO: pylint

TODO: use pylint (and flake8, ...) to check code quality.
