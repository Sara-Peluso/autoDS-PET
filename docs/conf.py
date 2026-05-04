"""Sphinx configuration for autods-pet documentation."""

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# Silence deprecation warnings from sphinx-autodoc-typehints (harmless,
# fixed in a future release of the extension).
warnings.filterwarnings("ignore", message=".*RemovedInSphinx10Warning.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sphinx_autodoc_typehints"
)

# -- Project information -----------------------------------------------------

project = "autods-pet"
author = "Sara Peluso"
copyright = "2026, Sara Peluso"

try:
    release = _version("autods-pet")
except PackageNotFoundError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Mock heavy C-extension / optional dependencies so docs build on RTD
# without installing them.
autodoc_mock_imports = [
    "SimpleITK",
    "numpy",
    "pandas",
    "pydicom",
    "rich",
    "typer",
    "openpyxl",
    "TotalSegmentator",
    "totalsegmentator",
    "tqdm",
]

# -- Napoleon (NumPy docstrings) ---------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- sphinx-autodoc-typehints ------------------------------------------------

always_use_bars_union = True
typehints_defaults = "braces"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Sara-Peluso/autoDS-PET",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}
