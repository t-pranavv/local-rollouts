# Documentation

This directory contains the documentation source files for PhyAGI project. We use **Sphinx** and **reStructuredText (RST)** to build and maintain high-quality developer and user documentation.

## Building

To build the documentation, you need to have Pandoc and Sphinx installed.

For Pandoc, install it via your system package manager or from the [official site](https://pandoc.org/install.html):

```bash
sudo apt install pandoc # for Debian/Ubuntu
# brew install pandoc # for MacOS
```

After installing Pandoc, you can install Sphinx and the necessary extensions using pip:

```bash
pip install -e ..[docs]
```

Then, you can build the documentation by running:

```bash
make clean; make html
```

The HTML files will be generated in the `_build/html/` directory, and you can open `html/index.html` in your browser to preview the documentation.

For more details on building and contributing to the documentation, refer to the [documentation guide](../docs/contributing/documentation.rst).