# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "Pixelator"
copyright = "2026, Pixelgen Technologies"
author = "Pixelgen Technologies"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "autoapi.extension",
    "sphinx_click",
]

autoapi_type = "python"
autoapi_dirs = ["../src/pixelator"]
autoapi_root = "api/generated"
autoapi_add_toctree_entry = False

autoapi_ignore = [
    "*/pixelator/mpx/*",
    "*/pixelator/mpx/**/*",
    "*/pixelator/cli/*",
    "*/pixelator/cli/**/*",
    "*/pixelator/pna/cli/*",
    "*/pixelator/pna/cli/**/*",
    "*/__pycache__/*",
    "*/__pycache__/**/*",
]

autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_member_order = "alphabetical"
autoapi_python_class_content = "both"
autoapi_python_use_implicit_namespaces = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented_params"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_init_with_doc = True

napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}
