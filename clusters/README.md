# Job Submission for Clusters

This folder contains job configuration files for submitting machine learning tasks using Azure Machine Learning (AML) and Amulet. Before going through the guide, ensure you have [logged in with your SC-ALT account](../docs/azure/sc_alt.rst).

## Azure Machine Learning (AML)

AML enables submitting and managing machine learning jobs directly via the Azure CLI.

### Installation

Install the Azure ML CLI extension:

```bash
az extension add -n ml
```

### Submitting Jobs

Use the provided YAML files located in the `aml/` directory to submit jobs:

```bash
az ml job create --file <job-configuration-file>.yaml --resource-group aifrontiers --workspace-name aifrontiers_ws
```

Replace `<job-configuration-file>` with your desired YAML configuration file.

## Amulet

[Amulet](https://amulet-docs.azurewebsites.net) simplifies experiment management and job submissions.

### Installation

It is recommended to install Amulet in a Python 3.10 environment:

```bash
conda create -n amlt python=3.10
conda activate amlt
pip install -U "amlt>=10.9.1.dev0" --index-url https://msrpypi.azurewebsites.net/nightly/leloojoo
```

### Setup

Create a project and configure the workspace:

```bash
amlt project create <project-name> aifrontiers
amlt workspace add aifrontiers_ws --resource-group aifrontiers --subscription d4fe558f-6660-4fe7-99ec-ae4716b5e03f
```

### Submitting Jobs

Use the YAML files located in the `amulet/` directory to run jobs:

```bash
amlt run <job-configuration-file>.yaml :<job-name> <experiment-name>
```

Replace:

- `<job-configuration-file>` with your YAML configuration file.
- `<job-name>` with a descriptive name for the job.
- `<experiment-name>` with the name of your experiment.
