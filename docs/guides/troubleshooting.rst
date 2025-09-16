Troubleshooting
===============

NVIDIA
------

* When running experiments on older clusters, e.g., those with A100 GPUs, you may encounter compatibility issues with the latest container images optimized for H100 hardware. A common symptom of this is the error ``cuda failure 304: 'os call failed or operation not supported on this os'``. To mitigate such an issue, use an image based on an older NVIDIA base layer that is compatible with A100 environments, such as ``nvidia24.05-pytorch2.6.0-te2.1-deepspeed0.16.7-flashattn2.7.4.post1-vllm0.8.4:20250503``.

PyTorch
-------

* There's a `regression <https://github.com/pytorch/pytorch/issues/149119>`_ on PyTorch 2.6.0 and 2.7.0, where ``torch.distributed.barrier()`` creates an additional overhead on the main process, leading to an increase in allocated memory. To mitigate such an issue, you can upgrade to PyTorch 2.7.1 or later.

* Frequent allocation and deallocation of tensors can lead to memory fragmentation, which may cause out-of-memory (OOM) errors. Such behavior is more pronounced in Reinforcement Learning (RL) workloads, where there are several offloads between CPU and GPU. To mitigate such an issue, consider doing forward and backward passes using a constant batch size and sequence length, as well as using ``torch.cuda.empty_cache()`` to release unused memory.

DeepSpeed
---------

* There's a known `issue <https://github.com/deepspeedai/DeepSpeed/issues/5242>`_ with DeepSpeed ZeRO-2 and multi-node training, where losses and gradients are not properly synchronized across nodes, leading to NaNs. To mitigate such an issue, set ``overlap_comm=False`` in the DeepSpeed configuration.