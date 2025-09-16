Installation
============

.. admonition:: Requirements
    :class: note

    Python 3.10+, CUDA 12.4+ and PyTorch 2.6+.

PhyAGI can be installed in several ways, but using a virtual environment, such as `Anaconda <https://docs.conda.io/en/latest/miniconda.html>`_, is highly recommended. Virtual environments provide an isolated setup, ensuring consistency and ease of package management. Before starting the installation, ensure you have cloned the repository:

.. code-block:: bash

    git clone git@github.com:microsoft/phyagi-sdk.git  # Alternatively: https://github.com/microsoft/phyagi-sdk.git
    cd phyagi-sdk

To install PhyAGI automatically, run:

.. code-block:: bash

    ./install.sh

This script sets up CUDA, PyTorch, core functionalities and additional components, such as Hugging Face, DeepSpeed, PyTorch-Lightning, Ray, and Flash-Attention, ensuring everything works out-of-the-box.

Additional components
---------------------

To enable specific components, you can install them using the following commands:

* **With evaluation packages:**

  .. code-block:: bash

      pip install -e .[eval]

* **With Flash-Attention:**

  .. code-block:: bash

      ./scripts/tools/install/flash-attn.sh
      pip install -e .[flash-attn]

* **With Reinforcement Learning packages:**

  .. code-block:: bash

      pip install -e .[rl]
