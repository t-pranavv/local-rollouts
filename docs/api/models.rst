Models
======

MixFormer (Sequential)
----------------------

.. autoclass:: phyagi.models.mixformer_sequential.configuration_mixformer_sequential.MixFormerSequentialConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.modeling_mixformer_sequential.MixFormerSequentialForCausalLM
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.modeling_mixformer_sequential.MixFormerSequentialForSequenceClassification
   :members:
   :special-members: __init__

.. automodule:: phyagi.models.mixformer_sequential.parallel_mixformer_sequential
   :members:

Blocks
^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.parallel.ParallelBlock
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.sequential.SequentialBlock
   :members:
   :special-members: __init__

Embeddings
^^^^^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.embeddings.common.Embedding
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.embeddings.common.PositionalEmbedding
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.embeddings.rotary.RotaryEmbedding
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.embeddings.rotary.YarnEmbedding
   :members:
   :special-members: __init__

Heads/Losses
^^^^^^^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.heads.causal_lm.CausalLMHead
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.heads.causal_lm.CausalLMLoss
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.heads.seq_cls.SequenceClassificationHead
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.heads.seq_cls.SequenceClassificationLoss
   :members:
   :special-members: __init__

Mixers
^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.mixers.mha.MHA
   :members:
   :special-members: __init__

Multi-Layer Perceptrons
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.mlps.glu.GLU
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.mlps.mlp.MLP
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.mlps.mlp.FusedMLP
   :members:
   :special-members: __init__

Normalization
^^^^^^^^^^^^^

.. autoclass:: phyagi.models.mixformer_sequential.blocks.norms.low_precision.LPLayerNorm
   :members:
   :special-members: __init__

.. autoclass:: phyagi.models.mixformer_sequential.blocks.norms.rms.RMSLayerNorm
   :members:
   :special-members: __init__

Model Conversion
----------------

.. automodule:: phyagi.models.model_convert
   :members:

Model Parallelism (Utilities)
-----------------------------

.. automodule:: phyagi.models.parallel_utils
   :members:

Registry
--------

.. automodule:: phyagi.models.registry
   :members: