Submiting jobs with Singularity
===============================

Before using or creating resources, ensure that you have completed the following prerequisites:

1. **Log in with your SC-ALT account:** Follow the `Log In with SC-ALT <./sc_alt.rst>`_ guide to log in using your SC-ALT account.

Azure Machine Learning (AML)
----------------------------

Azure Machine Learning (AML) allows you to submit and manage machine learning jobs using either the Azure CLI or the Python SDK. This direct approach is useful for custom workflows, automation, and advanced configurations.

Before you begin, make sure you have installed the Azure ML CLI extension:

.. code-block:: bash

   az extension add -n ml

We provide several example job configuration files for different tasks such as training, fine-tuning, and evaluation. Some examples include:

- `Fine-tune Phi-4 <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/aml/finetune_phi4.yaml>`_
- `Fine-tune Phi-4 (MI300x) <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/aml/finetune_phi4_mi300x.yaml>`_
- `Evaluation <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/aml/eval_evalchemy.yaml>`_
- `Evaluation (MI300x) <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/aml/eval_evalchemy_mi300x.yaml>`_

To use these job configuration files, simply run the following command with the appropriate file path:

.. code-block:: bash

   az ml job create --file <job-configuration-file>.yaml --resource-group aifrontiers --workspace-name aifrontiers_ws

Replace ``<job-configuration-file>`` with the path to your job configuration file. Be sure to check the configuration file for any additional parameters that need to be set.

Amulet
------

Amulet simplifies the process of managing experiments and submitting jobs. Before proceeding, make sure you have installed Amulet by following the `installation guide <https://amulet-docs.azurewebsites.net/main/setup.html>`_.

It is highly recommended to install ``amulet`` in a Python 3.10 virtual environment. You can do so with the following commands:

.. code-block:: bash

   conda create -n amlt python=3.10
   pip install -U "amlt>=10.9.1.dev0" --index-url https://msrpypi.azurewebsites.net/nightly/leloojoo

Before submitting jobs, you need to create a project and set the workspace:

.. code-block:: bash

   amlt project create <project-name> aifrontiers
   amlt workspace add aifrontiers_ws --resource-group aifrontiers --subscription d4fe558f-6660-4fe7-99ec-ae4716b5e03f

Answer **yes** to any prompts.

We provide several example job configuration files for different tasks such as training, fine-tuning, and evaluation. Some examples include:

- `Train Phi-1, Phi-2 or Phi-3 <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/train_phi.yaml>`_
- `Fine-tune Phi-4 <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/finetune_phi4.yaml>`__
- `Fine-tune Phi-4 (MI300x) <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/finetune_phi4_mi300x.yaml>`__
- `Fine-tune Qwen2.5-32B <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/finetune_qwen2.5.yaml>`_
- `Evaluation <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/eval_evalchemy.yaml>`__
- `Evaluation (MI300x) <https://github.com/microsoft/phyagi-sdk/blob/main/clusters/amulet/eval_evalchemy_mi300x.yaml>`__

To use these job configuration files, simply run the following command with the appropriate file path:

.. code-block:: bash

   amlt run <job-configuration-file>.yaml :<job-name> <experiment-name>

Replace ``<job-configuration-file>`` with the path to your job configuration file, ``<job-name>`` with the name of the job, and ``<experiment-name>`` with the name of your experiment. Be sure to check the configuration file for any additional parameters that need to be set.

Additional resources
--------------------

- `Singularity Documentation <https://singularitydocs.azurewebsites.net/>`_
- `Azure Machine Learning (AML) Documentation <https://learn.microsoft.com/en-us/azure/machine-learning>`_
- `Amulet Documentation <https://amulet-docs.azurewebsites.net/main/index.html>`_
- `How to Submit a Job with Singularity <https://amulet-docs.azurewebsites.net/main/tutorial.html>`_
