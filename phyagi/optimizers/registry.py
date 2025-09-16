# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from dion import Dion, DionMixedPrecisionConfig, Muon
from torch.distributed import DeviceMesh
from torch.optim import (
    ASGD,
    SGD,
    Adadelta,
    Adagrad,
    Adamax,
    AdamW,
    Optimizer,
    RMSprop,
    Rprop,
)
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from phyagi.optimizers.dadaptation import DAdaptAdam
from phyagi.optimizers.lion import Lion
from phyagi.optimizers.schedulers.warmup_decay import (
    WarmupDecayCooldownLR,
    WarmupDecayLR,
    WarmupLR,
)

OPTIMIZERS = {
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "adamw": AdamW,
    "asgd": ASGD,
    "dadapt_adam": DAdaptAdam,
    "dion": Dion,
    "muon": Muon,
    "lion": Lion,
    "rmsprop": RMSprop,
    "rprop": Rprop,
    "sgd": SGD,
}

LR_SCHEDULERS = {
    "constant": ConstantLR,
    "cosine_annealing": CosineAnnealingLR,
    "cosine_annealing_with_warm_restart": CosineAnnealingWarmRestarts,
    "cyclic": CyclicLR,
    "exponential": ExponentialLR,
    "lambda": LambdaLR,
    "linear": LinearLR,
    "multiplicative": MultiplicativeLR,
    "multi_step": MultiStepLR,
    "one_cycle": OneCycleLR,
    "reduce_lr_on_plateau": ReduceLROnPlateau,
    "step": StepLR,
    "warmup": WarmupLR,
    "warmup_decay": WarmupDecayLR,
    "warmup_decay_cooldown": WarmupDecayCooldownLR,
}


def _get_dion_or_muon_param_groups(
    optimizer_type: str, model: torch.nn.Module, lr: float = 0.01
) -> List[Dict[str, Any]]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise ValueError("`model` must have a `config` attribute, but got None.")

    matrix_params, array_params, head_params = [], [], []

    def _is_head_param(param_name: str) -> bool:
        patterns = ["classifier", "final_layer", "head", "lm_head", "output", "prediction_head"]
        return any(p in param_name for p in patterns) or f"layers.{model_config.n_layer+1}" in param_name

    def _is_embedding_param(param_name: str) -> bool:
        patterns = [
            "embed",
            "embedding",
            "pos_embed",
            "position_embeddings",
            "token_embed",
            "tok_embeddings",
            "word_embeddings",
            "wte",
        ]
        return any(p in param_name for p in patterns)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Match against a broad set of head-like layer names
        if _is_head_param(name_lower):
            head_params.append(param)

        # Match 2D weight matrices (not embedding or head)
        elif param.ndim == 2 and not _is_embedding_param(name_lower):
            matrix_params.append(param)

        # Everything else (biases, norms, embeddings)
        else:
            array_params.append(param)

    return [
        dict(params=matrix_params, algorithm=optimizer_type),
        dict(
            params=array_params,
            algorithm="lion",
            lr=lr,
            betas=(0.95, 0.98),
            weight_decay=0,
        ),
        dict(
            params=head_params,
            algorithm="lion",
            lr=lr / math.sqrt(model_config.n_embd),
            betas=(0.95, 0.98),
            weight_decay=0,
        ),
    ]


def _get_dion_or_muon_optimizer(
    optimizer_type: str, model: torch.nn.Module, device_mesh: Optional[DeviceMesh] = None, **kwargs
) -> Dion:
    replicate_mesh, outer_shard_mesh, inner_shard_mesh = None, None, None
    if device_mesh is not None:
        outer_shard_mesh = device_mesh["data_context_parallel"]
        inner_shard_mesh = device_mesh["tensor_parallel"]
    else:
        replicate_mesh = torch.distributed.group.WORLD

    lr = kwargs.get("lr", 0.01)
    param_groups = _get_dion_or_muon_param_groups(optimizer_type, model, lr=lr)

    optimizer_cls = None
    optimizer_kwargs = {
        "lr": lr,
        "mu": kwargs.get("mu", 0.95),
        "weight_decay": kwargs.get("weight_decay", 0.01),
    }

    if optimizer_type == "dion":
        optimizer_cls = Dion
        optimizer_kwargs.update(
            {
                "replicate_mesh": replicate_mesh,
                "outer_shard_mesh": outer_shard_mesh,
                "inner_shard_mesh": inner_shard_mesh,
                "replicate_mesh_grad_sync": True,
                "rank_fraction": kwargs.get("rank_fraction", 1.0),
                "mixed_precision_config": DionMixedPrecisionConfig(momentum_dtype=torch.float32),
            }
        )
    elif optimizer_type == "muon":
        optimizer_cls = Muon
        optimizer_kwargs.update(
            {
                "distributed_mesh": outer_shard_mesh
                if outer_shard_mesh and outer_shard_mesh.size() > 1
                else replicate_mesh,
                "nesterov": True,
                "adjust_lr": "spectral_norm",
                "use_triton": False,
            }
        )

    return optimizer_cls(param_groups, **optimizer_kwargs)


def get_optimizer(model: torch.nn.Module, optimizer_type: str = "adamw", **kwargs) -> Optimizer:
    """Get an optimizer.

    Args:
        model: Model to be optimized.
        optimizer_type: Optimizer to be used.

    Returns:
        Optimizer.

    """

    if optimizer_type not in OPTIMIZERS:
        raise ValueError(f"`optimizer_type` must be one of {list(OPTIMIZERS.keys())}, but got '{optimizer_type}'.")

    # Since Dion / Muon require a different initialization, we handle them separately
    if optimizer_type in ["dion", "muon"]:
        return _get_dion_or_muon_optimizer(optimizer_type, model, **kwargs)

    return OPTIMIZERS[optimizer_type](model.parameters(), **kwargs)


def get_lr_scheduler(optimizer: Optimizer, lr_scheduler_type: str = "constant", **kwargs) -> LRScheduler:
    """Get a learning rate scheduler.

    Args:
        optimizer: Optimizer to be used.
        lr_scheduler_type: Learning rate scheduler to be used.

    Returns:
        Learning rate scheduler.

    """

    if lr_scheduler_type not in LR_SCHEDULERS:
        raise ValueError(
            f"`lr_scheduler_type` must be one of {list(LR_SCHEDULERS.keys())}, but got '{lr_scheduler_type}'."
        )

    return LR_SCHEDULERS[lr_scheduler_type](optimizer, **kwargs)
