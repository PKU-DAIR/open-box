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
import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('./../../'))

# -- Project information -----------------------------------------------------

project = 'OpenBox'
copyright = '2021, Thomas (Yang) Li'
author = 'Thomas (Yang) Li'

# The full version, including alpha/beta/rc tags
release = 'beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark', 'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx_markdown_tables']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to en directory, that match files and
# directories to ignore when looking for en files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
language = 'en'
master_doc = 'index'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': 'black',
}
html_logo = '../imgs/logo.png'

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/docs/'


# app setup hook
def setup(app):
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_math': False,
        'enable_inline_math': False,
        'enable_eval_rst': True
    }, True)
    app.add_transform(AutoStructify)
