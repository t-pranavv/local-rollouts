Configuration system
====================

PhyAGI utilizes `OmegaConf <https://omegaconf.readthedocs.io>`_ for managing configurations, offering a structured approach for defining parameters such as datasets, model architectures, and training routines. OmegaConf's hierarchical configuration system supports variable interpolation and merging of configurations, which simplifies the management of various environments and setups.

The following sections offer an in-depth look at the `primary configuration keys <https://github.com/microsoft/phyagi-sdk/blob/main/scripts/train/configs/ds_mixformer-sequential-350m.yaml>`_ used in PhyAGI's configuration system.

Configuration keys
------------------

.. tip::

   These configuration keys define essential parameters that affect various aspects of the model training process.

* ``data_root``: Specifies the directory where the dataset is stored. This value can either be set by the ``DATA_ROOT`` environment variable or take the default value.

* ``output_root``: Defines where output files (e.g., training checkpoints) are stored. This location can be overridden by the ``OUTPUT_ROOT`` environment variable.

* ``output_dir``: The directory within ``output_root`` for saving outputs specific to a model's training.

Dataset configuration
---------------------

The ``dataset`` section plays a crucial role in specifying how data is managed during training. Below are the keys within this section:

* ``cache_dir``: Directory where the processed and cached dataset is stored.

* ``train_file``: Path to the ``.npy`` file containing the training data. If ``cache_dir`` is defined, it expects to find the training file at ``<cache_dir>/train.npy``.

* ``validation_file``: Path to the ``.npy`` file containing the validation data, typically found at ``<cache_dir>/validation.npy`` when ``cache_dir`` is set.

* ``validation_split``: Used instead of ``validation_file`` to specify a validation dataset as an integer or float:

  * Integer: The number of tokens from the training dataset to be used for validation.
  * Float: Fraction of the training data used for validation (between ``0.0`` and ``1.0``).

* ``tokenizer_file``: Path to the ``.pkl`` file containing the tokenizer configuration, typically located at ``<cache_dir>/tokenizer.pkl``.

* ``seq_len``: Sequence length for the model's input.

* ``shift_labels``: Boolean flag that determines if the labels should be shifted, commonly used in next-token prediction tasks.

* ``weight``: Assigns a relative weight to the dataset, useful when multiple datasets are in use.

* ``label``: Defines the label to index the validation dataset's logs.

* ``dataset_concat``: Specifies how multiple datasets are concatenated during training.

* ``eval_dataset_concat``: Specifies how multiple datasets are concatenated during validation.

* ``dataset_provider``: Defines the type of dataset provider, which determines how the samples are retrieved and formatted.

* ``dataset_collator``: Configuration for the dataset collator, responsible for batching and preparing data samples:

  * ``cls``: The class of the collator.
  * ``ignore_token_ids``: List of token IDs to be ignored.
  * ``ignore_token_range``: Range of token IDs to be ignored.
  * ``ignore_index``: The index to be ignored by the loss function.

Model configuration
-------------------

The ``model`` section defines the architecture and parameters of the neural network:

* ``model_type``: Defines the architecture type (e.g., ``mixformer-sequential``).

* ``vocab_size``: Total number of unique tokens in the model's vocabulary.

* ``n_positions``: Maximum number of positions (sequence length) supported by the model.

* ``n_embd``: Dimensionality of the embeddings.

* ``n_layer``: Number of layers in the model (e.g., transformer blocks).

* ``n_head``: Number of attention heads in each transformer layer.

* ``rotary_dim``: Dimensionality for rotary position embeddings used in attention mechanisms.

* ``embd_pdrop``: Dropout probability for the embedding layer.

* ``resid_pdrop``: Dropout probability for residual connections in the model layers.

* ``activation_function``: Type of activation function (e.g., ``gelu_new``).

* ``layer_norm_epsilon``: Small constant added for stability in layer normalization.

* ``initializer_range``: Range of values for initializing model parameters.

* ``pad_vocab_size_multiple``: Ensures that the vocabulary size is a multiple of this value for efficiency.

* ``use_cache``: Whether to use cache for attention mechanisms (inference-only).

* ``embd_layer``: Specifies the embedding layer type, such as ``default``.

Training configuration (DeepSpeed)
----------------------------------

.. warning::

   The configuration for training varies according to the trainer (Hugging Face, DeepSpeed, Lightning) and must be set correctly to ensure optimal performance during model training.

Training arguments include various settings that control the training process:

* ``ds_config``: Configuration for DeepSpeed, including the following sub-keys:

  * ``fp16``: Enables 16-bit floating-point precision.
  * ``optimizer``: Defines the optimizer settings (e.g., ``AdamW``).
  * ``scheduler``: Configures the learning rate scheduler.
  * ``zero_optimization``: Configures ZeRO optimizer stages.
  * ``gradient_clipping``: Defines the maximum norm for gradient clipping.

* ``do_eval``: Indicates whether to run evaluation on the validation set.

* ``num_train_epochs``: The total number of epochs for training.

* ``max_steps``: Maximum number of training steps.

* ``logging_steps``: Frequency (in steps) at which training metrics are logged.

* ``save_steps``: Defines how often to save model checkpoints.

* ``save_final_checkpoint``: Whether to save the final model checkpoint after training.

* ``seed``: Sets the random seed for reproducibility.

* ``mlflow``: Enables logging to MLflow for tracking training metrics.

* ``wandb``: Integrates with Weights & Biases for tracking experiments.

* ``gradient_clipping``: Controls gradient clipping.
