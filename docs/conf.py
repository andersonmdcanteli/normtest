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
sys.path.insert(0, os.path.abspath('..'))
import normtest
from datetime import date

# -- Project information -----------------------------------------------------

year = date.today()

project = normtest.__name__
copyright = f'{year.year}, {normtest.__author__}'
author = normtest.__author__

# The full version, including alpha/beta/rc tags
release = 'Alpha ' + normtest.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    # 'sphinx_toolbox.shields',
    # 'sphinx.ext.doctest', # o problema é que ele quebra todo o output, e retorna muitos falsos erros
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'furo'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static',]

html_css_files = [
    'css\custom.css',
]

# Set up intersphinx maps
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),

        }


html_logo = "_static/logo.svg"


html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/andersoncanteli/normtest",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "website",
            "url": "https://andersonmdcanteli.github.io/",
            "icon": "fas fa-coffee",
            "type": "fontawesome",
        },
        {
            "name": "Contact",
            "url": "https://drive.google.com/file/d/15lgqKeKLskShRvSqjIhUd4QspgSpaFYM/view?usp=sharing",
            "icon": "fas fa-id-badge",
            # The default for `type` is `fontawesome` so it is not actually required in any of the above examples as it is shown here
        },
    ],

}

numpydoc_show_class_members = True # set False to remove mebers
#
