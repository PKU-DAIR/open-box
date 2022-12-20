# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('./../../'))

# -- Project information -----------------------------------------------------

project = 'OpenBox'
copyright = f'{date.today().year}, PKU-DAIR'
author = 'PKU-DAIR'

# The full version, including alpha/beta/rc tags
# from openbox import version as _version
# release = str(_version)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'myst_parser',          # https://myst-parser.readthedocs.io
    'sphinx_copybutton',    # https://sphinx-copybutton.readthedocs.io
    'notfound.extension',   # https://sphinx-notfound-page.readthedocs.io
]

# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
# example:
#     rst:   :ref:`design principle <overview/overview:design principle>`
#     md:    {ref}`design principle <overview/overview:design principle>`
#            or [](<overview/overview:design principle>)  (hoverxref CANNOT identify this syntax!)
extensions += ['sphinx.ext.autosectionlabel']
# Make sure the target is unique
autosectionlabel_prefix_document = True  # ref example: `dir/file:header`
autosectionlabel_maxdepth = None  # Must be None. Or failed to build change_logs


# myst_parser
# documentation: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",  # pip install linkify-it-py
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-header-anchors
myst_heading_anchors = 3  # e.g., [](../overview/overview.md#design-principle) (hoverxref CANNOT identify this syntax!)


# Show tooltip when hover on the reference. Currently, only Read the Docs is supported as backend server!
# https://sphinx-hoverxref.readthedocs.io/
extensions += ['hoverxref.extension']
hoverxref_auto_ref = True
hoverxref_role_types = {}
hoverxref_default_type = 'tooltip'  # 'modal' or 'tooltip'
# hoverxref_sphinxtabs = True
# hoverxref_mathjax = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

language = 'zh_CN'
root_doc = 'index'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = '../imgs/logo.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# default theme
# html_theme = 'alabaster'

# sphinx_rtd_theme (pip install sphinx_rtd_theme) (https://sphinx-rtd-theme.readthedocs.io/)
# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {
#     'logo_only': True,
#     'style_nav_header_background': 'black',
# }

# furo (pip install furo) (https://pradyunsg.me/furo/)
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#336790",  # "blue"
        # "color-brand-content": "#336790",  # use the original blue for links in content
        "color-inline-code-background": "#E6ECFF",  # original: #f8f9fb
    },
    "dark_css_variables": {
        "color-brand-primary": "#E5B62F",  # "yellow"
        "color-brand-content": "#E5B62F",
        "color-inline-code-background": "#393939",  # original: #1a1c1e
    },
}
