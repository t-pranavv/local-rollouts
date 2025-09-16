Quick start
===========

This quick start guide will help you get up and running with PhyAGI and demonstrate how to train a model. If you're a beginner, we recommend exploring the tutorials for a more in-depth explanation of the concepts introduced here.

Preparing the data
------------------

To start training a model with PhyAGI, the first step is preparing your data. In this example, we will use the :class:`.LMDatasetProvider` class, which offers a simple interface for loading and tokenizing datasets from the Hugging Face hub.

.. code-block:: python

    from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider

    dataset_provider = LMDatasetProvider.from_hub(
        "wikitext",
        dataset_name="wikitext-103-raw-v1",
        revision=None,
        tokenizer="microsoft/phi-2",
        validation_split=0.1,
        shuffle=True,
        use_eos_token=True,
        num_workers=1,
        cache_dir="wt103",
        seq_len=2048,
    )

:meth:`.LMDatasetProvider.from_hub()` will automatically download the dataset from the Hugging Face hub, tokenize it, and cache it in the specified ``cache_dir`` folder. This folder will contain a ``config.json`` file (dataset provider configuration) and ``.npy`` files (NumPy tokenized arrays).

Once the dataset is downloaded and tokenized, you can access the training and validation datasets:

.. code-block:: python

    train_dataset = dataset_provider.get_train_dataset()
    val_dataset = dataset_provider.get_val_dataset()

Loading the model
-----------------

With the data prepared, you can now load the model architecture you wish to use. PhyAGI supports any ``torch.nn.Module``. In this example, we will use the :class:`.MixFormerSequentialForCausalLM` architecture:

.. code-block:: python

    from phyagi.models.mixformer_sequential import MixFormerSequentialConfig, MixFormerSequentialForCausalLM

    config = MixFormerSequentialConfig(n_layer=4, n_embd=128)
    model = MixFormerSequentialForCausalLM(config)

In this example, we are using the default configuration (with the exception of the number of layers and embedding size). These parameters can easily be modified by adjusting the arguments passed to :class:`.MixFormerSequentialConfig`.

Training the model
------------------

Now that you have prepared the data and loaded the model, you are ready to train! PhyAGI provides three training classes: :class:`.DsTrainer`, :class:`.HfTrainer`, and :class:`.PlTrainer`. The first uses DeepSpeed for training, while the second uses Hugging Face's :class:`transformers.Trainer` class, and the third uses PyTorch Lightning. Each trainer has its own set of arguments and configurations, but they all follow a similar pattern.

With DeepSpeed
^^^^^^^^^^^^^^^

.. code-block:: python

    from phyagi.trainers.ds.ds_trainer import DsTrainer
    from phyagi.trainers.ds.ds_training_args import DsTrainingArguments

    training_args = DsTrainingArguments("mixformer-wt103", max_steps=1)
    trainer = DsTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

**Note:** DeepSpeed requires the ``deepspeed`` launcher. To train the model using 4 GPUs, use the following command:

.. code-block:: bash

    deepspeed --num_gpus=4 script.py

With Hugging Face
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from transformers import TrainingArguments
    from phyagi.trainers.hf.hf_trainer import HfTrainer

    training_args = TrainingArguments("mixformer-wt103", max_steps=1)
    trainer = HfTrainer(
        model,
        args=training_args,
        data_collator=LMDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

**Note:** Hugging Face requires the ``torchrun`` launcher for multi-GPU training. To train with 4 GPUs, use this command:

.. code-block:: bash

    torchrun --nproc-per-node=4 script.py

With PyTorch Lightning
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from phyagi.trainers.pl.pl_trainer import PlTrainer
    from phyagi.trainers.pl.pl_training_args import (
        PlLightningModuleArguments,
        PlTrainerArguments,
        PlTrainingArguments,
    )

    training_args = PlTrainingArguments(
        "mixformer-wt103",
        trainer=PlTrainerArguments(
            precision=16,
            max_steps=1,
        ),
        lightning_module=PlLightningModuleArguments(
            optimizer={"type": "adamw", "lr": 1.8e-3},
        ),
    )

    trainer = PlTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.fit()

**Note:** PyTorch Lightning requires the ``python`` launcher. To train the model using 4 GPUs, use the following command and it will auto-detect the number of GPUs available:

.. code-block:: bash

    python script.py

Next steps
----------

Congratulations! You have successfully trained your first model using PhyAGI. To continue your learning journey, check out our detailed guides, tutorials, and advanced topics.