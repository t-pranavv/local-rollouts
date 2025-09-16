Command-Line Interface (CLI)
============================

The PhyAGI Command-Line Interface (CLI) is a powerful tool that allows users to interact with PhyAGI's functionalities directly from the terminal or command prompt. This interface facilitates scripting and automation of tasks, which can be highly valuable for users who need to process large amounts of data or integrate PhyAGI's capabilities into their workflows.

Convert
-------

The ``convert`` command allows users to convert a transformer-based model from one type to another. This can be useful when you want to adapt a model to a different architecture or when you require a specific format for deployment.

.. code-block:: bash

    phyagi-cli convert <pretrained_model_name_or_path> <convert_model_type> [options]

**Arguments:**

- ``pretrained_model_name_or_path``: The path to the directory containing the pre-trained model or the name of the model if it is available through a model repository.
- ``convert_model_type``: The type of model to convert the pre-trained model into. The supported target model types are defined by the available converters in the PhyAGI system.

**Options:**

- ``--resize_embeddings``: If set, this flag will resize the embeddings of the model to match the tokenizer's vocabulary size (including added tokens).
- ``--dtype``: The data type to which the model's parameters should be converted. The default is ``float32``. This option helps in optimizing the model for specific hardware requirements or precision needs.
- ``--debug_logits``: If set, this flag will enable debugging of the model's logits (the raw outputs from the last layer before the activation function) to ensure that the conversion process has not affected the model's predictions.
- ``--debug_params``: If set, this flag will enable debugging of the model's parameters to verify that all parameters have been correctly converted and are present in the new model.
- ``--deepspeed_weights``: If set, this flag will convert the model's weights to the format used by DeepSpeed.
- ``--from_deepspeed_zero``: If set, this flag will convert the model's weights from the format used by DeepSpeed's ZeRO-{2,3}.
- ``--from_pl_fsdp``: If set, this flag will convert the model's weights from the format used by PyTorch Lightning's FSDP (Fully Sharded Data Parallel).

Evaluate
--------

The ``evaluate`` command is designed to assess the performance of pre-trained models on specified evaluation tasks. This functionality is crucial for users who wish to understand how well a model performs on a given task, such as text classification, language understanding, or any other task for which the model has been trained.

.. code-block:: bash

    phyagi-cli evaluate <pretrained_model_name_or_path> <task_name> [options]

**Arguments:**

- ``pretrained_model_name_or_path``: This is the path to the directory that contains the pre-trained model files or the name of the model if it is available in a model repository.
- ``task_name``: The name of the task on which the model is to be evaluated. The task should be one that the model is capable of performing and for which evaluation procedures and metrics are defined within the PhyAGI framework.

**Options:**

- ``--pretrained_tokenizer_name_or_path``: This optional argument specifies the path or name of the pre-trained tokenizer to use. If not provided, it defaults to the same path or name as the pre-trained model.
- ``--device_map``: This argument defines the strategy for device placement (e.g., which model parts should be placed on which device). Options include ``"auto"``, ``"balanced"``, ``"balanced_low_0"``, ``"cpu"``, ``"cuda"``, and ``"sequential"``. The default setting is ``"cuda"``, which places the model on a CUDA-capable GPU if available.
- ``--dtype``: Specifies the data type for the model during evaluation. The default is ``float32``, which is a common choice for balance between precision and computational efficiency.
- ``--batch_size``: This defines the number of examples to process simultaneously during the evaluation. The default batch size is 1.
- ``--use_amp``: When this flag is set, the command will employ automatic mixed precision (AMP) for evaluation, which can lead to faster computation on compatible hardware with potentially lower memory usage.

Speed-Benchmark
---------------

The ``speed-benchmark`` command is designed for conducting speed benchmarks of neural network architectures. It specifically measures the time taken for forward and backward passes, which are critical operations during the training and inference stages of machine learning models.

.. code-block:: bash

    phyagi-cli speed-benchmark <model_config_file_path> [options]

**Arguments:**

- ``model_config_file_path``: This is the path to the configuration file for the model. The configuration file should contain all the necessary details to instantiate the model, including the type of model and any relevant parameters.

**Options:**

- ``--device_map``: This argument specifies the device placement strategy for the model. Options include ``auto``, ``balanced``, ``balanced_low_0``, ``cpu``, ``cuda``, and ``sequential``. The default is ``cuda``, indicating that if a GPU is available, it will be used for the benchmark.
- ``--dtype``: Specifies the data type for the model during the benchmark. The default is ``float32``, a common choice for balance between precision and computational efficiency.
- ``--n_trials``: The number of trials to run for the benchmark. Each trial consists of a forward pass and an optional backward pass (for gradient computation). The default number of trials is 10.

.. note::

    Running speed benchmarks with different hardware configurations can yield varying results. Always ensure that your hardware is compatible and optimally configured for the best performance.