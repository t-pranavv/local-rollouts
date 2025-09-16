<h1 align="center">
   <img src="docs/assets/logo.png" alt="PhyAGI logo" width="96px" />
   <br />
   PhyAGI
</h1>

<div align="center">
   <b>PhyAGI</b> is a project focused on <b>training</b>, <b>fine-tuning</b>, and <b>evaluating</b> Large Language Models (LLM) with ease and flexibility.
</div>

<br />

<div align="center">
   <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
   <img src="https://img.shields.io/badge/cuda-12.4+-green.svg" alt="CUDA 12.4+">
   <img src="https://img.shields.io/badge/pytorch-2.6+-red.svg" alt="PyTorch 2.6+">
</div>

<br />

<div align="center">
   <a href="#features">Features</a> •
   <a href="#installation">Installation</a> •
   <a href="#quick-start">Quick Start</a> •
   <a href="#documentation">Documentation</a> •
   <a href="#contributing-and-support">Contributing and Support</a>
</div>

<br />

## Features

PhyAGI is a modular and scalable framework for research, supporting local development through to multi-node training and evaluation. It provides unified support for Hugging Face, PyTorch Lightning, DeepSpeed, Ray, and custom training loops.

<details>
<summary><strong>Datasets</strong></summary>

- Modular dataset providers for various task types.
- Support for:
  - Language Modeling (LM), Streaming LM, and Masked LM.
  - Supervised Fine-Tuning (SFT).
  - Reinforcement Learning (PPO, DPO, GRPO, ISFT).
- Shared-memory and iterable dataset loading.
- Efficient in-memory tokenization and pre-processing.
- Custom data collators for training and fine-tuning phases.
- Dynamic packing, special token injection, and formatting utilities.
- Dataset concatenation with random, sequential, or iterable strategies.

</details>

<details>
<summary><strong>Models</strong></summary>

- Hugging Face-compatible and PyTorch-native model support.
- Built-in architecture: <strong>MixFormer</strong>
  - Modular components:
    - Input embeddings.
    - Attention mechanisms.
    - MLPs.
    - Normalization layers.
    - Task heads.
  - Sequential block and parallel block compositions.
- Parallelism-ready model wrappers and configuration utilities.

</details>

<details>
<summary><strong>Training</strong></summary>

- Training backends:
  - Hugging Face.
  - PyTorch Lightning.
  - DeepSpeed.
- Unified interface across backends via registries, e.g., `phyagi.trainers.registry`.
- Support for:
  - ZeRO (DeepSpeed): data, context, and pipeline parallelisms.
  - FSDP (PyTorch Lightning): data, context, and tensor parallelisms.
- Custom optimizers (e.g., D-Adaptation, Dion, Lion and Muon).
- Custom learning rate schedulers (e.g., warmup, warmup with decay, warmup with decay and cooldown).
- Logging of training progress and FLOPs estimation.
- Model checkpointing, timer, and batch-level tracking.

</details>

<details>
<summary><strong>Fine-Tuning</strong></summary>

- Full fine-tuning and adapter-based tuning workflows.
- Ray-based distributed tuning with support for:
  - Interactive Supervised Fine-Tuning (ISFT).
  - Reinforcement Learning methods:
    - PPO, DPO, GRPO.
- Built-in reward manager and custom reward functions:
  - Math and code-based evaluation.
  - Phi-4-reasoning-plus reward.
- Support for vLLM rollout workers.
- Distributed actor model for scalability.

</details>

<details>
<summary><strong>Evaluation</strong></summary>

- Multi-node and distributed evaluation pipelines.
- Supports both batched text generation and log-likelihood evaluation.
- Task suite:
  - Code generation (HumanEval, MBPP).
  - Commonsense reasoning (PIQA, SIQA, HellaSwag, Winogrande).
  - Math (GSM8K, MathQA, MATH).
  - QA (NQ, TriviaQA, OpenBookQA, Race, ARC).
  - Reading comprehension (SQuAD, LAMBADA).
  - Benchmark integrations (MMLU, GLUE, SuperGLUE).
- Integration with LM-Eval and compatibility with external tools.

</details>

<details>
<summary><strong>Utilities</strong></summary>

- Configuration system: supports YAML, JSON, CLI arguments, and Python dictionaries.
- Object registry for runtime model/dataset/optimizer resolution.
- Import utilities, type safety helpers, and logging abstraction.
- Logging integrations: WandB, MLflow, TensorBoard.
- Deprecation warning system for legacy interfaces.
- CLI tools:
  - Model conversion and evaluation.
  - Ray cluster launcher.
  - Benchmarking tools (speed and FLOPs).
- Distributed checkpointing and file utilities.

</details>

## Installation

```bash
git clone git@github.com:microsoft/phyagi-sdk.git
cd phyagi-sdk
bash install.sh
```

This script sets up CUDA, PyTorch, core functionalities and additional components, ensuring everything works out-of-the-box. For more information, check the [installation guide](https://microsoft.github.io/phyagi-sdk/getting_started/installation.html).

## Quick Start

### Running on clusters

Check the in-depth guide on how to [submit a job with Singularity](https://microsoft.github.io/phyagi-sdk/azure/singularity.html).

### Running locally

Use a Linux sandbox and Visual Studio Code with the [remote development](https://code.visualstudio.com/docs/remote/remote-overview) extension.

#### Prepare datasets

If `install.sh` has been used to install the framework, `DATA_ROOT` has already been set to `/mnt/phyagi`. Nevertheless, this environment variable can be replaced to download datasets to a new folder:

```bash
echo 'export DATA_ROOT=<local_folder>' >> ~/.bashrc && source ~/.bashrc
```

Replace `<local_folder>` with the full path where you want the datasets saved.

Download some pre-tokenized datasets to your local machine:

```bash
bash scripts/tools/data/copy_datasets.sh
```

#### Training

Train a Phi-1 350M model ([training documentation](scripts/train/README.md)):

```bash
cd scripts/train
deepspeed ds_train.py configs/phi1/phi1-350m.yaml
```

#### Supervised fine-tuning

Fine-tune (with SFT) a Phi-1 350M model ([fine-tuning documentation](scripts/tune/README.md)):

```bash
cd scripts/tune
accelerate launch --num_processes 1 hf_sft_tune.py configs/hf_sft.yaml
```

#### Evaluation

Evaluate a Phi-1 350M model on common-sense reasoning tasks ([evaluation documentation](scripts/eval/README.md)):

```bash
cd scripts/eval
python eval.py configs/common_sense_reasoning.yaml microsoft/phi-1
```

## Documentation

[PhyAGI documentation](https://microsoft.github.io/phyagi-sdk) provides detailed information on how to use the framework, including guides, tutorials, and API references, covering everything from installation to advanced usage scenarios.

## Contributing and Support

We welcome issues, feature requests, and pull requests! For more information on how to contribute, refer to the [first time contributor guide](https://microsoft.github.io/phyagi-sdk/contributing/first_time_contributor.html).

*For experimental work, use [phyagi-experiments](https://github.com/microsoft/phyagi-experiments), where you can create your user folder and make any changes you like.*