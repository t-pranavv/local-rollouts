Documentation
=============

The PhyAGI project welcomes contributions through the implementation of documentation files using `Sphinx <https://www.sphinx-doc.org/en/master>`_ and `RST <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_. If you are interested in contributing to the project in this way, please follow these steps:

1. **Clone repository and install Pandoc/Sphinx:**

.. code-block:: bash

    git clone git@github.com:microsoft/phyagi-sdk.git
    cd phyagi-sdk
    sudo apt install pandoc
    pip install -e .[docs]

2. **Create a new branch:**

.. code-block:: bash

    git checkout -b docs/<your_branch_name>

3. **Create an RST file:**

   - You can use the template available on :github:`docs/getting_started/quick_start` as a starting point.
   - If writing an API document, the file should be placed in the :github:`docs/api` folder.

4. **Build the documentation:**

.. code-block:: bash

    cd docs
    make html

The HTML files will be created in a ``_build`` directory. Open the ``html/index.html`` file in your browser to view the documentation.

5. **Push the changes:**

.. code-block:: bash

    git push origin docs/<your_branch_name>

After pushing the changes, create a `pull request <https://github.com/microsoft/phyagi-sdk/pulls>`_ and tag a reviewer.