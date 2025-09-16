# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import date

# Add path to local extension
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "PhyAGI"
author = "Microsoft"
copyright = f"{date.today().year}"

# General configuration
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = [
    ".github/**",
    ".vscode/**",
    "clusters/**",
    "docker/**",
    "scripts/**",
    "tests/**",
]
extlinks = {"github": ("https://github.com/microsoft/phyagi-sdk/tree/main/%s", "%s")}
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
html_baseurl = "https://microsoft.github.io/phyagi-sdk/"

# Options for HTML output
html_title = project
html_baseurl = "https://microsoft.github.io/phyagi"
html_theme = "sphinx_book_theme"
html_logo = "assets/logo.png"
html_favicon = "assets/favicon.ico"
html_last_updated_fmt = ""
html_static_path = ["assets"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "PhyAGI",
    },
    "repository_url": "https://github.com/microsoft/phyagi",
    "use_issues_button": True,
    "use_edit_page_button": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "use_repository_button": True,
    "show_navbar_depth": 1,
    "toc_title": "Sections",
}

# Autodoc
autodoc_default_options = {"exclude-members": "__weakref__"}
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "flash-attn",
    "lm-eval",
    "mpi4py",
    "torchao",
    "vllm",
]

# NbSphinx
nbsphinx_execute = "never"
