Reinforcement Learning
======================

Models
------

.. autoclass:: phyagi.rl.models.actor_config.ActorConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.models.actor.Actor
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.models.reference.Reference
   :members:
   :special-members: __init__

Rewards
-------

.. autoclass:: phyagi.rl.rewards.reward.Reward
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.rewards.reward_manager.RewardManager
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.rewards.gsm8k.GSM8kReward
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.rewards.phi4rp.Phi4RPReward
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.rewards.python_code_executor.PythonCodeExecutorReward
   :members:
   :special-members: __init__

Registry
^^^^^^^^

.. automodule:: phyagi.rl.rewards.registry
   :members:

Rollout
-------

.. autoclass:: phyagi.rl.rollout.vllm_worker_config.VLLMWorkerConfig
   :members:
   :special-members: __init__

.. automodule:: phyagi.rl.rollout.vllm_worker.VLLMWorker
   :members:
   :special-members: __init__

Tuners
------

.. autoclass:: phyagi.rl.tuners.ray_worker_config.RayWorkerConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.tuners.ray_worker.RayWorker
   :members:
   :special-members: __init__

Hugging Face
^^^^^^^^^^^^

.. autoclass:: phyagi.tuners.hf.hf_tuner.HfSFTTuner
   :members:
   :special-members: __init__

.. autoclass:: phyagi.tuners.hf.hf_tuner.HfDPOTuner
   :members:
   :special-members: __init__

.. autoclass:: phyagi.tuners.hf.hf_tuner.HfGRPOTuner
   :members:
   :special-members: __init__

GRPO
^^^^

.. autoclass:: phyagi.rl.tuners.grpo.grpo_config.RayGRPOConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.tuners.grpo.grpo_worker.RayGRPOWorker
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.tuners.grpo.grpo_tuner.RayGRPOTuner
   :members:
   :special-members: __init__

ISFT
^^^^

.. autoclass:: phyagi.rl.tuners.isft.isft_config.RayISFTConfig
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.tuners.isft.isft_worker.RayISFTWorker
   :members:
   :special-members: __init__

.. autoclass:: phyagi.rl.tuners.isft.isft_tuner.RayISFTTuner
   :members:
   :special-members: __init__

Registry
^^^^^^^^

.. automodule:: phyagi.rl.tuners.registry
   :members:
