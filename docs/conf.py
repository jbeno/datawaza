# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# The directory relative to which the paths are considered is the one
# containing the conf.py file, so you should adjust the '../' to point
# to the root of your Python package.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Datawaza'
copyright = '2024, Jim Beno'
author = 'Jim Beno'
version = '0.1'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_design',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    'logo': {
        'alt-text': 'Datawaza',
        'image_light': '_static/datawaza_logo_name_white.svg',
        'image_dark': '_static/datawaza_logo_name_grey.svg'
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/jbeno/datawaza",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "favicons": [
        {
            "rel": "icon",
            "sizes": "16x16",
            "href": "datawaza_logo_favicon_16x16.png",
        },
        {
            "rel": "icon",
            "sizes": "32x32",
            "href": "datawaza_logo_favicon_32x32.png",
        },
        {
            "rel": "apple-touch-icon",
            "sizes": "180x180",
            "href": "datawaza_logo_favicon_180x180.png"
        }
    ]
}
html_css_files = [
    'css/custom.css',
]
html_sidebars = {
    'index': [],
    'explore': [],
    'model': []
}
