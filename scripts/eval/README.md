# Evaluation

This documentation covers the process of evaluating your model. Before diving into the details, there are some important notes:

1. Some evaluations require additional dependencies to be installed. For example `lm-eval` is not installed by default, but it can be used when evaluation a model with `phyagi-sdk`:

    ```bash
    pip install -e .[eval]
    ```

2. In the [`configs`](./configs/) directory, you can find a set of pre-defined configurations that are used to evaluate models. For example, [`configs/common_sense_reasoning.yaml`](./configs/common_sense_reasoning.yaml) provides a configuration for evaluating a model on common sense reasoning tasks with `phyagi-sdk`.

3. Additional folders, e.g., [`evalchemy`](./evalchemy/) and [`long_context`](./long_context/), provide sets of files (scripts, readmes, configurations) to evaluate models using different packages not supported by `phyagi-sdk`.

4. If needed, scripts can be launched with `-h` to see additional arguments. For more information, please refer to the [evaluation tutorial](https://microsoft.github.io/phyagi-sdk/tutorials/evaluation.html).

### Single GPU

The [`eval.py`](./eval.py) script allows you to evaluate a model on multiple tasks, using multiple checkpoints, and save the results to a file. The script takes the following arguments:

```bash
python eval.py configs/common_sense_reasoning.yaml <pretrained_model_name_or_path>
```

### Multiple GPUs / Nodes

To evaluate a model on multiple GPUs / nodes, you can use the [`eval_distributed.py`](./eval_distributed.py) script. It takes the same arguments as `eval.py`, but it need to be launched with either `torchrun` or `python -m torch.distributed.run`. For example:

```bash
torchrun --nproc_per_node <num_gpus> eval_distributed.py configs/common_sense_reasoning.yaml <pretrained_model_name_or_path>
```
