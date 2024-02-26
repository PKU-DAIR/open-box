# Documentation

OpenBox uses `sphinx` to generate the documentation.
It supports `reStructuredText(.rst)` and `Markdown(.md)` as the source files.

For markdown parser, `recommonmark` is deprecated and `myst-parser` is used instead.

Documentation is hosted on `Read the Docs` and is built automatically when a commit is pushed to the master branch.

Currently, the `sphinx-hoverxref` extension requires `Read the Docs` as backend to work properly.
It cannot be used in local build. And if you want to use `GitHub Pages` to host the documentation, you need to remove this extension.


## Official Guides

+ sphinx: https://www.sphinx-doc.org/en/master/usage/quickstart.html

+ myst-parser: https://myst-parser.readthedocs.io/
  + myst-parser syntax: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#
  + To include-markdown-files-into-an-rst-file:
    https://myst-parser.readthedocs.io/en/latest/faq/index.html#include-markdown-files-into-an-rst-file
    (The `parser` option requires docutils>=0.17)
+ readthedocs: https://docs.readthedocs.io/


## Configuration

Sphinx configuration file is `docs/en/conf.py`.

Read the Docs configuration file is `.readthedocs.yaml`.
(For Chinese version, it is `docs/zh_CN/.readthedocs.yaml`.)

The requirements for building the documentation are listed in `requirements/dev/docs.txt`.

Please look through these files carefully.

## Build the Documentation Locally

Install the requirements via `pip install -r requirements/dev/docs.txt`.

Before pushing your changes, you should build the documentation locally to make sure it looks good:

```bash
cd docs/en
make html
```

Open `build/html/index.html` or type `open build/html/index.html` in the terminal to view the documentation.

To remove the build files, run `make clean` or simply delete the `build` directory.

Remember to include `build` in `.gitignore` to avoid committing the build files.


## For Read the Docs

Project page: https://readthedocs.org/projects/open-box/

Project page (Chinese version): https://readthedocs.org/projects/open-box-zh-cn/

The documentation is currently hosted on Read the Docs

### Sync with GitHub

The documentation is built automatically when a commit is pushed to the master branch.

If the documentation is not updated after a push,
please check the "Integrations" (https://readthedocs.org/dashboard/open-box/integrations/)
and "Webhooks" in the repository settings on GitHub.

If the integration cannot be added successfully,
you need to remove the Read the Docs integration and webhooks (if any),
remove the webhooks in the repository settings on GitHub,
and then manually add the webhook in GitHub.
Tutorial: https://docs.readthedocs.io/en/stable/guides/setup/git-repo-manual.html

### Multi-language Support

The documentation supports multi-language.

All source files are located in `docs/en` and `docs/zh_CN`.

We created multiple Read the Docs projects for different languages.
The English version `open-box` is the main project.
The Chinese version `open-box-zh_CN` is a translation project.
The translation project is linked to the main project in the `Translations` section of the main project settings.


## For sphinx-hoverxref

Currently, the `sphinx-hoverxref` extension requires `Read the Docs` as backend to work properly.
It cannot be used in local build. And if you want to use `GitHub Pages` to host the documentation, you need to remove this extension.

To use `sphinx-hoverxref` in `Markdown(.md)` files, instead of using `[Link Text](file path)`, use the following syntax:
```
{ref}`Link Text <directory/filename_without_suffix:title>`
```

For example:
```
{ref}`Installation Guide <installation/installation_guide:installation guide>`
```
