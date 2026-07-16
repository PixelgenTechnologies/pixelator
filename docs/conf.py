# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import logging
import os

project = "Pixelator"
copyright = "2026 Pixelgen Technologies"
author = "Pixelgen Technologies"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

nitpicky = True
nitpick_ignore_regex = [
    # --- MPX excluded ---
    (r"py:.*", r"pixelator\.mpx\..*"),
    # --- Unable to resolve intersphinx inventory ---
    (r"py:.*", r"polars\..*"),
    (r"py:.*", r"cutadapt\..*"),
    (r"py:.*", r"numpy\..*"),
    (r"py:.*", r"duckdb\..*"),
    (r"py:.*", r"faiss\..*"),
    # --- Private, internal, TypeVar, protocol-base, and native targets ---
    # These come from annotations, base-class lists, or TypeVars in the source code.
    (
        r"py:.*",
        r"(Edge|Vertex|VertexSequence|AmpliconBuilder|BarcodeDemuxer"
        r"|DemuxFilenamePolicy|PipelineRunner|StatisticsClass"
        r"|NetworkxBasedVertexClustering|T|mpctx_Process)",
    ),
    (r"py:.*", r"(_SummaryStatsDict|_PartitionCandidate|_COMPONENT_BATCH_SIZE)"),
    (r"py:.*", r"pixelator_core\.PyGraphProperties"),
    (r"py:.*", r"pixelator\.types\.PathType"),
    (
        r"py:.*",
        r"pixelator\.common\.graph\.backends\.implementations\._networkx"
        r"\.NetworkXGraphBackend",
    ),
]

suppress_warnings = [
    "ref.python",
    "autoapi.python_import_resolution",
    "toc.not_included",
]

# Warnings emitted without a Sphinx type/subtype, so suppress_warnings
# cannot match them; they are suppressed via a logging filter instead.
_suppressed_warning_fragments = (
    # Import placeholders AutoAPI could not resolve.
    "Unknown type: placeholder",
    # pna_config is assigned 4 times in config_instance.py; AutoAPI
    # documents each assignment.
    "duplicate object description of pixelator.pna.config.config_instance.pna_config",
)


def _keep_warning(record):
    message = record.getMessage()
    return not any(fragment in message for fragment in _suppressed_warning_fragments)


for _logger_name in ("sphinx.autoapi._mapper", "sphinx.sphinx.domains.python"):
    logging.getLogger(_logger_name).addFilter(_keep_warning)

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

# Options for AutoAPI
autoapi_options = [
    "members",  # Display children of an object
    "show-inheritance",  # Display a list of base classes below the class signature
    "show-module-summary",  # Whether to include autosummary directives in generated module documentation.
    "undoc-members",  # Display objects that have no docstring
    "imported-members",  # For objects imported into a package, display objects imported from the same top level package or module.
    # "special-members",  # Display special objects (eg. __foo__ in Python)
    # "inherited-members",  # Display children of an object that have been inherited from a base class.
    # Note the absence of the option private-members
]

autoapi_member_order = "alphabetical"
autoapi_python_class_content = "class"
autoapi_python_use_implicit_namespaces = True
autoapi_own_page_level = "function"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

_docs_version = os.environ.get("DOCS_VERSION", "latest")
_docs_base_url = os.environ.get(
    "DOCS_BASE_URL",
    "https://PixelgenTechnologies.github.io/pixelator",
).rstrip("/")

html_baseurl = f"{_docs_base_url}/"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PixelgenTechnologies/pixelator",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["version-switcher"],
    "switcher": {
        "json_url": f"{_docs_base_url}/switcher.json",
        "version_match": _docs_version,
    },
    "logo": {
        "image_light": "_static/pixelator.svg",
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

python_maximum_signature_line_length = 40

autodoc_member_order = "alphabetical"
autodoc_typehints = "signature"
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
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "cutadapt": ("https://cutadapt.readthedocs.io/en/stable/", None),
    "dnaio": ("https://dnaio.readthedocs.io/en/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "click": ("https://click.palletsprojects.com/en/stable/", None),
    "packaging": ("https://packaging.pypa.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "requests": ("https://requests.readthedocs.io/en/stable/", None),
}
