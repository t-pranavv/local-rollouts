Build a Docker image
====================

This guide will walk you through the process of building and using a custom Docker image for PhyAGI. By utilizing Docker, you can easily manage dependencies, streamline deployment, and ensure a consistent environment across various stages of development.

Requirements
------------

Before proceeding with this tutorial, ensure that you have the following installed and set up:

* `Docker <https://docs.docker.com/get-docker>`_
* `Azure CLI <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>`_
* Active Azure account with access to the respective ACR (Azure Container Registry).

Packages
--------

The following packages **must be used** when building a new image:

* ``transformers>=4.53.0``
* ``flash-attn>=2.7.4``
* ``deepspeed>=0.16.9``

Build files
-----------

The ``docker/`` folder is divided into two environments:

* ``docker/nvidia/``: for CUDA-enabled NVIDIA GPUs.
* ``docker/amd/``: for ROCm-enabled AMD GPUs.

Each folder includes a ``Dockerfile`` and build scripts tailored to the respective backend. The ``Dockerfile`` follows the Singularity `guidelines <https://singularitydocs.azurewebsites.net/docs/tutorials/custom_images>`_ and serves as a template for building custom images. The ``build_image.sh`` script is a wrapper that loads the base, validation, and installation images.

.. tip::

   #. Modify the necessary installations, such as ``apt-get`` or ``python`` packages.
   #. Update the base image and the output tag according to your requirements.

NVIDIA (CUDA) instructions
--------------------------

Follow these steps from the root folder of ``phyagi-sdk`` to build and push your custom Docker image:

1. **Sign in to Azure and the corresponding ACR:**

   .. code-block:: bash

      az acr login -n <acr-name>

2. **Build the image:**

   .. code-block:: bash

      bash docker/nvidia/build_image.sh

   Optionally install ``vLLM`` by setting the ``--install-vllm`` flag.

3. **Push the image to the ACR:**

   .. code-block:: bash

      docker push <acr-name>.azurecr.io/<image-name>:<tag>

   Make sure to use the correct registry and image name as necessary.

AMD (ROCm) instructions
-----------------------

.. important::

   For ROCm builds, it is **crucial** to ensure that PyTorch, Flash-Attention, and related libraries are compatible. Mismatched versions can lead to training instability such as NaNs or gradient explosions.

Follow these steps from the root folder of ``phyagi-sdk`` to build and push your custom Docker image:

1. **Sign in to Azure and the corresponding ACR:**

   .. code-block:: bash

      az acr login -n <acr-name>

2. **Build the image:**

   .. code-block:: bash

      bash docker/amd/build_image.sh

3. **Push the image to the ACR:**

   .. code-block:: bash

      docker push <acr-name>.azurecr.io/<image-name>:<tag>

   Make sure to use the correct registry and image name as necessary.

Training on ROCm devices (e.g., MI250, MI300) **will be slower** than CUDA-based GPUs. Many kernels are either unoptimized or unsupported. Expect lower MFU and throughput, especially for transformer-based models.

Additional resources
--------------------

- `NVIDIA (CUDA) Docker Images <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_
- `AMD (ROCm) Docker Images <https://hub.docker.com/r/rocm>`_
- `PyTorch on ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html>`_
