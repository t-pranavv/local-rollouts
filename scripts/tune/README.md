# Fine-Tuning (with Reinforcement Learning)

This documentation covers the process of fine-tuning your model using Hugging Face and Ray.

Before diving into the details, there are some important notes:

1. When using Hugging Face, you can provide a `--config_file` argument to the Accelerate launcher. In the [`accelerate_configs`](./accelerate_configs/) directory, you can find a set of configurations that uses DeepSpeed ZeRO-{1,2,3}, FSDP, and more.

2. In the [`configs`](./configs/) directory, you can find a set of pre-defined configurations that are used to fine-tune models. For example, [`configs/hf_sft.yaml`](./configs/hf_sft.yaml) and [`configs/hf_dpo.yaml`](./configs/hf_dpo.yaml) are pre-defined to fine-tune a `microsoft/phi-1` model, but they require `dataset.data_files` to be a valid `.jsonl` file path.

3. Additional folders inside the [`configs`](./configs/) directory, e.g., [`configs/phi4`](./configs/phi4/), provide a set of cluster-ready configurations for fine-tuning models. These configurations should be used with either AzureML or Amulet files available in the top-level [`clusters`](../../clusters/) folder.

4. If needed, scripts can be launched with `-h` to see additional arguments.

## Hugging Face

### Supervised Fine-Tuning

The [`hf_sft_tune.py`](./hf_sft_tune.py) script implements a Hugging Face-based approach to fine-tune a model using SFT with support for a `.yaml` configuration file. To fine-tune your model using this method, run the following command:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml --num_processes 1 hf_sft_tune.py configs/hf_sft.yaml
```

### Direct Preference Optimization

The [`hf_dpo_tune.py`](./hf_dpo_tune.py) script implements a Hugging Face-based approach to fine-tune a model using DPO with support for a `.yaml` configuration file. To fine-tune your model using this method, run the following command:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml --num_processes 1 hf_dpo_tune.py configs/hf_dpo.yaml
```

## Ray

### Interactive Supervised Fine-Tuning

The [`ray_isft_tune.py`](./ray_isft_tune.py) script implements a Ray-based approach to fine-tune a model using ISFT with support for a `.yaml` configuration file. To fine-tune your model using this method, run the following command:

```bash
RAY_DEDUP_LOGS=0 python ray_isft_tune.py configs/ray_isft.yaml
```

### Group Relative Policy Optimization

The [`ray_grpo_tune.py`](./ray_grpo_tune.py) script implements a Ray-based approach to fine-tune a model using GRPO with support for a `.yaml` configuration file. To fine-tune your model using this method, run the following command:

```bash
RAY_DEDUP_LOGS=0 python ray_grpo_tune.py configs/ray_grpo.yaml
```

*`RAY_DEDUP_LOGS=0` is used to disable Ray's log deduplication feature, which can be helpful for debugging purposes. You can remove it if you do not need it.*