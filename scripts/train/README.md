# Training

This documentation covers the process of training your model using DeepSpeed, Hugging Face, and PyTorch Lightning.

Before diving into the details, there are some important notes:

1. In the [`configs`](./configs/) directory, you can find a set of pre-defined configurations that are used to train models. For example, [`configs/ds_mixformer-sequential-350m.yaml`](./configs/ds_mixformer-sequential-350m.yaml) provides a configuration for training a 350M-parameter model using DeepSpeed and the MixFormer architecture. The file also provides a set of comments that describes the parameters used in the training process.

2. Additional folders inside the [`configs`](./configs/) directory, e.g., [`configs/phi3`](./configs/phi3/) and [`configs/phi4`](./configs/phi4/), provide a set of cluster-ready configurations for training models. These configurations should be used with either AzureML or Amulet files available in the top-level [`clusters`](../../clusters/) folder.

3. If needed, scripts can be launched with `-h` to see additional arguments.

## DeepSpeed

The [`ds_train.py`](./ds_train.py) implements a DeepSpeed training script with support for a `.yaml` configuration file. To train your model using this method, run the following command:

```bash
deepspeed ds_train.py configs/ds_mixformer-sequential-350m.yaml
```

## Hugging Face

The [`hf_train.py`](./hf_train.py) implements a Hugging Face training script with support for a `.yaml` configuration file. To train your model using this method, run the following command:

```bash
accelerate launch --num_processes 1 hf_train.py configs/hf_mixformer-sequential-350m.yaml
```

## PyTorch Lightning

The [`pl_train.py`](./pl_train.py) implements a PyTorch Lightning training script with support for a `.yaml` configuration file. To train your model using this method, run the following command:

```bash
python pl_train.py configs/pl_mixformer-sequential-350m.yaml
```