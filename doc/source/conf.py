# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
from pyvkfft.version import __authors__, __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyvkfft'
copyright = '2023- Vincent Favre-Nicolin ; ESRF-The European Synchrotron'
author = ', '.join(__authors__)
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'nbsphinx_link',
    'sphinxarg.ext'
]

templates_path = ['_templates']
exclude_patterns = []

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Do not transform "--" into "-"
smartquotes = False