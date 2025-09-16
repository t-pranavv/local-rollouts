Documentation
=============

**Physics of Artificial General Intelligence (PhyAGI)** is a project focusing on training, fine-tuning, and evaluating Large Language Models (LLM) with ease and flexibility.

Contents
--------

The documentation is organized into these main sections:

**GETTING STARTED:** Quick-start guide with installation instructions.

**GUIDES:** Comprehensive explanations of core features.

**TUTORIALS:** Step-by-step instructions for beginners.

**ADVANCED TUTORIALS:** Detailed tutorials on advanced topics.

**AZURE:** Guidance for configuring and using Azure services.

**CONTRIBUTING:** Information on how to contribute to PhyAGI.

**API:** A detailed reference of all classes and functions, categorized for easy navigation.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   Installation <getting_started/installation>
   Build a Docker image <getting_started/build_docker>
   Quick start <getting_started/quick_start>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Guides

   Logging <guides/logging>
   Configuration system <guides/configuration_system>
   Command-Line Interface (CLI) <guides/cli>
   Data generation infrastructure <guides/data_generation_infrastructure>
   Llama.cpp and GGUF <guides/llama_cpp_and_gguf>
   Troubleshooting <guides/troubleshooting>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   Data-related (providers, mixers and collators) <tutorials/data_related.ipynb>
   Model architecture <tutorials/model_architecture.ipynb>
   Training <tutorials/training.ipynb>
   Supervised Fine-Tuning (SFT) <tutorials/sft.ipynb>
   Reinforcement Learning from Human Feedback (RLHF) <tutorials/rlhf.ipynb>
   Evaluation <tutorials/evaluation.ipynb>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced tutorials

   Learning rate schedulers <advanced_tutorials/lr_schedulers.ipynb>
   Optimizers <advanced_tutorials/optimizers.ipynb>
   Parameter-Efficient Techniques (PEFT) <advanced_tutorials/parameter_efficient.ipynb>
   Batch tracking <advanced_tutorials/batch_tracking>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Azure

   Log-In with SC-ALT <azure/sc_alt>
   Elevate permissions with Privileged Identity Management (PIM) <azure/pim>
   Submiting jobs with Singularity <azure/singularity>
   Azure Storage Account <azure/storage_account>
   Azure Virtual Machine (VM) <azure/virtual_machine>
   Azure Container Registry (ACR) <azure/container_registry>
   Azure App Service <azure/app_service>
   Azure Kubernetes Service (AKS) <azure/kubernetes_service>
   Microsoft Entra <azure/entra>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contributing

   First time contributor <contributing/first_time_contributor>
   Documentation <contributing/documentation>
   Tests <contributing/tests>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   Datasets <api/datasets>
   Models <api/models>
   Optimizers <api/optimizers>
   Trainers <api/trainers>
   Reinforcement Learning <api/rl>
   Evaluation <api/eval>
   Utilities <api/utils>
   CLI <api/cli>
