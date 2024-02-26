# Releases

<font color=#FF0000>Maintainers are suggested to read the entire content of this document.</font>

## Release a new version on GitHub

To release a new version on GitHub, follow the steps below:
1. Update version number in `openbox/__init__.py` and update `CHANGELOG.md`.
2. Commit and create a new git tag (e.g. v0.8.1). Push the commit and the tag to GitHub.
3. Wait and make sure tests are passed on GitHub Actions!
4. Create a new release on GitHub by clicking "Draft a new release" on <https://github.com/PKU-DAIR/open-box/releases>.
   Select the tag you just created.
   Fill in the title (e.g. v0.8.1 - Mar 8, 2023) and description (from CHANGELOG.md).
5. Double check the release information and the test results on GitHub Actions.
6. Click "Publish release". The release will trigger GitHub Actions to build and upload packages to PyPI.

## Version Control

Currently, the version number is manually updated in `openbox/__init__.py`.

If no new features are added, only bug fixes, the version number should be updated to `x.y.z+1`.

If API is changed, the version number should be updated to `x.y+1.0`.

See [PEP 440 -- Version Identification and Dependency Specification](https://www.python.org/dev/peps/pep-0440/).

**TODO:** To automatically update version number,
[setuptools_scm](https://setuptools-scm.readthedocs.io/) can be used.

You may take [BoTorch](https://github.com/pytorch/botorch) as an example.

## Python Packaging

See `pyproject.toml` and `MANIFEST.in`.

Tutorial:
* https://packaging.python.org/en/latest/tutorials/packaging-projects/
* https://setuptools.pypa.io/en/latest/userguide/quickstart.html

**Caution:** In this repo, package will be uploaded to PyPI 
by GitHub Actions triggered by publishing a release.
So, you don't need to upload the package manually.

To build and test python package locally, or upload the package manually:
```bash
python -m pip install --upgrade pip setuptools build twine

# Build
python -m build --sdist --wheel
# For quick build, you may use --no-isolation option.

# Check
tar tf dist/*.tar.gz
unzip -l dist/*.whl
twine check dist/* --strict

# Manually upload to TestPyPI
# twine upload --repository testpypi dist/*
# Manually upload to PyPI (make sure old version files are removed in dist/ !!!)
# twine upload dist/*

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ ${package_name}
```

### Tips:
* TestPyPI can be used to test package before uploading to PyPI: https://test.pypi.org/
* `distutils` is deprecated with removal planned for Python 3.12. Use `setuptools` instead.
* Deprecated: `python setup.py install`
* `setup.py` is suggested (by setuptools) to be replaced by `setup.cfg` or `pyproject.toml`: 
  https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html

### pyproject.toml:
* Configuring setuptools using pyproject.toml files:
  https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
* Configuring setuptools using setup.cfg files:
  https://setuptools.pypa.io/en/latest/userguide/declarative_config.html
* project metadata:
  https://packaging.python.org/en/latest/specifications/declaring-project-metadata/

### To include files in the package, use `MANIFEST.in`:
* https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#controlling-files-in-the-distribution
* https://packaging.python.org/en/latest/guides/using-manifest-in/

### Define entry points:
Run `python -m openbox` will execute the entry point defined in `pyproject.toml [project.scripts]`.


## PyPI

OpenBox PyPI: https://pypi.org/project/openbox/

Maintainer access of PyPI project is required to upload packages manually.

Currently, this repo uses GitHub Actions to build and upload packages to PyPI when publishing a release.
An API token has been generated on PyPI (To create a new one: https://pypi.org/manage/account/token/)
and added to GitHub Secrets (GitHub project - Settings - Security - Secrets and variables - Actions - New repository secret).

PyPI now supports Trusted Publisher Management (https://pypi.org/manage/account/publishing/).
If necessary, migrate to Trusted Publisher Management to allow publishing packages from GitHub Actions.


## Configure GitHub Actions

See `.github/workflows/publish_on_release.yml`.

API token or Trusted Publisher Management is required to upload packages to PyPI.
