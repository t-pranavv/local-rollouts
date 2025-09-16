Trainers
========

.. automodule:: phyagi.trainers.flops_utils
   :members:

.. automodule:: phyagi.trainers.trainer_utils
   :members:

DeepSpeed
---------

.. autoclass:: phyagi.trainers.ds.ds_training_args.DsTrainingArguments
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.ds.ds_trainer.DsTrainer
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.ds.ds_trainer_callback.DsTrainerCallback
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.ds.ds_trainer_callback.DsCallbackHandler
   :members:
   :special-members: __init__

Hugging Face
------------

.. autoclass:: phyagi.trainers.hf.hf_trainer.HfTrainer
   :members:
   :special-members: __init__

PyTorch Lightning
-----------------

.. autoclass:: phyagi.trainers.pl.pl_training_args.PlStrategyArguments
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_training_args.PlTrainerArguments
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_training_args.PlLightningModuleArguments
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_training_args.PlTrainingArguments
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_lightning_module.TrainingLightningModule
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_strategies.DataContextTensorParallelStrategy
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_trainer.PlTrainer
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_callbacks.MetricLogCallback
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_callbacks.OptimizerLogCallback
   :members:
   :special-members: __init__

.. autoclass:: phyagi.trainers.pl.pl_progress_bars.TQDMStepProgressBar
   :members:
   :special-members: __init__

Registry
--------

.. automodule:: phyagi.trainers.registry
   :members:
