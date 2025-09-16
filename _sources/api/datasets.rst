Datasets
========

Dataset Providers (Abstract)
----------------------------

.. autoclass:: phyagi.datasets.dataset_provider.DatasetProvider
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.dataset_provider.DatasetProviderConfig
   :members:
   :special-members: __init__

Datasets Concatenation
----------------------

.. autoclass:: phyagi.datasets.concat_dataset.WeightedConcatDataset
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.concat_dataset.SequentialWeightedConcatDataset
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.concat_dataset.WeightedConcatIterableDataset
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.concat_dataset.WeightedConcatChatDataset
   :members:
   :special-members: __init__

Pre-training
------------

Datasets Providers
^^^^^^^^^^^^^^^^^^

.. autoclass:: phyagi.datasets.train.lm.lm_dataset_provider.LMDatasetProvider
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.train.stream_lm.stream_lm_dataset_provider.StreamLMDatasetProviderConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.train.stream_lm.stream_lm_dataset_provider.StreamLMDatasetProvider
   :members:
   :special-members: __init__

Datasets
^^^^^^^^

.. autoclass:: phyagi.datasets.train.lm.lm_dataset.LMDataset
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.train.stream_lm.stream_lm_dataset.StreamLMDataset
   :members:
   :special-members: __init__

Data Collators
^^^^^^^^^^^^^^

.. autoclass:: phyagi.datasets.train.train_data_collator.LMDataCollator
   :members:
   :special-members: __init__

Reinforcement Learning
----------------------

.. automodule:: phyagi.datasets.rl.formatting_utils
   :members:

.. automodule:: phyagi.datasets.rl.packing
   :members:

.. automodule:: phyagi.datasets.rl.special_tokens
   :members:

Datasets Providers
^^^^^^^^^^^^^^^^^^

.. autoclass:: phyagi.datasets.rl.chat.chat_dataset_provider.ChatDatasetProviderConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.datasets.rl.chat.chat_dataset_provider.ChatDatasetProvider
   :members:
   :special-members: __init__

Datasets
^^^^^^^^

.. autoclass:: phyagi.datasets.rl.chat.chat_dataset.ChatDataset
   :members:
   :special-members: __init__

Data Collators
^^^^^^^^^^^^^^

.. autoclass:: phyagi.datasets.rl.rl_data_collator.RewardDataCollator
   :members:
   :special-members: __init__

Registry
--------

.. automodule:: phyagi.datasets.registry
   :members: