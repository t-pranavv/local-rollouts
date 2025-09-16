# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _update_lr(param_groups: List[Dict[str, float]], lrs: List[float]) -> List[float]:
    for param_group, lr in zip(param_groups, lrs):
        if isinstance(param_group["lr"], torch.Tensor):
            param_group["lr"].fill_(lr)
        else:
            param_group["lr"] = lr
    return [group["lr"] for group in param_groups]


class WarmupLR(LRScheduler):
    """Learning rate scheduler with warmup period.

    This scheduler supports one stage:

    1. Warmup: Linearly increase learning rate from ``warmup_min_lr`` to ``warmup_max_lr``.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = "log",
        last_batch_iteration: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_min_lr: Minimum learning rate before warmup.
            warmup_max_lr: Maximum learning rate after warmup.
            warmup_num_steps: Number of steps to warm up from ``warmup_min_lr`` to ``warmup_max_lr``.
            warmup_type: Increasing function during warmup, either ``linear`` or ``log``.
            last_batch_iteration: Index of the last batch.

        """

        if warmup_type not in ["log", "linear"]:
            raise ValueError(f"`warmup_type` must be 'log' or 'linear', but got '{warmup_type}'.")

        self.optimizer = optimizer

        self._min_lrs = self._format_param(warmup_min_lr)
        self._max_lrs = self._format_param(warmup_max_lr)
        self._delta_lrs = [max_lr - min_lr for max_lr, min_lr in zip(self._max_lrs, self._min_lrs)]

        self._warmup_num_steps = max(2, warmup_num_steps)
        self._warmup_type = warmup_type
        self._inverse_log_warm_up = 1.0 / math.log(self._warmup_num_steps)

        self._last_batch_iteration = last_batch_iteration
        if self._last_batch_iteration == -1:
            self._last_lr = _update_lr(self.optimizer.param_groups, self.get_lr())

    def _format_param(self, param_value: Union[float, List[float]]) -> List[float]:
        if isinstance(param_value, (list, tuple)):
            if len(param_value) != len(self.optimizer.param_groups):
                raise ValueError(f"Expected {len(self.optimizer.param_groups)} values, but got {len(param_value)}.")
            return list(param_value)
        return [param_value] * len(self.optimizer.param_groups)

    def _get_gamma(self) -> float:
        if self._last_batch_iteration < self._warmup_num_steps:
            if self._warmup_type == "log":
                return self._inverse_log_warm_up * math.log(self._last_batch_iteration + 1)
            return self._last_batch_iteration / self._warmup_num_steps
        return 1.0

    def get_lr(self) -> List[float]:
        if self._last_batch_iteration < 0:
            return self._min_lrs

        gamma = self._get_gamma()
        return [min_lr + (delta_lr * gamma) for min_lr, delta_lr in zip(self._min_lrs, self._delta_lrs)]

    def get_last_lr(self) -> List[float]:
        if getattr(self, "_last_lr", None) is None:
            raise ValueError("`step()` must be called before `get_last_lr()`.")
        return self._last_lr

    def step(self, last_batch_iteration: Optional[int] = None) -> None:
        if last_batch_iteration is None:
            last_batch_iteration = self._last_batch_iteration + 1
        self._last_batch_iteration = last_batch_iteration

        self._last_lr = _update_lr(self.optimizer.param_groups, self.get_lr())

    def state_dict(self) -> Dict[str, int]:
        return {"last_batch_iteration": self._last_batch_iteration}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self._last_batch_iteration = state_dict["last_batch_iteration"]


class WarmupDecayLR(WarmupLR):
    """Learning rate scheduler with warmup and decay periods.

    This scheduler supports two stages:

    1. Warmup: Linearly increase learning rate from ``warmup_min_lr`` to ``warmup_max_lr``.
    2. Decay: Decay learning rate from ``warmup_max_lr`` to ``0.0``.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int = 1000,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = "log",
        last_batch_iteration: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            total_num_steps: Total number of training steps.
            warmup_min_lr: Minimum learning rate before warmup.
            warmup_max_lr: Maximum learning rate after warmup.
            warmup_num_steps: Number of steps to warm up from ``warmup_min_lr`` to ``warmup_max_lr``.
            warmup_type: Increasing function during warmup, either ``linear`` or ``log``.
            last_batch_iteration: Index of the last batch.

        """

        if total_num_steps < warmup_num_steps:
            raise ValueError(f"`total_num_steps` must be >= {warmup_num_steps}, but got {total_num_steps}.")
        self._total_num_steps = total_num_steps

        super().__init__(optimizer, warmup_min_lr, warmup_max_lr, warmup_num_steps, warmup_type, last_batch_iteration)

    def _get_gamma(self) -> float:
        if self._last_batch_iteration < self._warmup_num_steps:
            return super()._get_gamma()

        return max(
            0.0,
            (self._total_num_steps - self._last_batch_iteration)
            / max(1.0, self._total_num_steps - self._warmup_num_steps),
        )


class WarmupDecayCooldownLR(WarmupDecayLR):
    """Learning rate scheduler with warmup, decay and cooldown periods.

    This scheduler supports five stages:

    1. Warmup: Linearly increase learning rate from ``warmup_min_lr`` to ``warmup_max_lr``.
    2. Patience: Wait for ``warmup_patience_num_steps`` steps before decaying.
    3. Decay: Decay learning rate from ``warmup_max_lr`` to ``decay_min_lr``.
    4. Patience: Wait for ``decay_patience_num_steps`` steps before cooling down.
    5. Linear cooldown: Linearly decrease learning rate from ``decay_min_lr`` to ``cooldown_min_lr``.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int = 1000,
        warmup_min_lr: Union[float, List[float]] = 0.0,
        warmup_max_lr: Union[float, List[float]] = 0.001,
        warmup_num_steps: int = 1000,
        warmup_patience_num_steps: int = 0,
        warmup_type: str = "linear",
        decay_min_lr: Optional[Union[float, List[float]]] = None,
        decay_num_steps: int = 0,
        decay_patience_num_steps: int = 0,
        decay_type: str = "cosine",
        decay_cycles: float = 0.5,
        cooldown_min_lr: Optional[Union[float, List[float]]] = None,
        cooldown_num_steps: int = 0,
        last_batch_iteration: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            total_num_steps: Total number of training steps.
            warmup_min_lr: Minimum learning rate before warmup.
            warmup_max_lr: Maximum learning rate after warmup.
            warmup_num_steps: Number of steps to warm up from ``warmup_min_lr`` to ``warmup_max_lr``.
            warmup_patience_num_steps: Number of steps to wait before decaying.
            warmup_type: Increasing function during warmup, either ``linear`` or ``log``.
            decay_min_lr: Minimum learning rate after decaying.
                If ``None`` and ``decay_num_steps > 0``, the decaying learning rate is set to ``warmup_min_lr``.
                If ``decay_num_steps <= 0``, the decaying learning rate is set to ``warmup_max_lr``.
            decay_num_steps: Number of steps to decay from ``warmup_max_lr`` to ``decay_min_lr``.
            decay_patience_num_steps: Number of steps to wait before cooling down.
            decay_type: Decaying function, either ``cosine`` or ``inverse_square_root``.
            decay_cycles: Number of cosine cycles during decaying (only used when ``decay_type`` is ``cosine``).
            cooldown_min_lr: Minimum learning rate after cooldown.
                If ``None`` and ``cooldown_num_steps > 0``, the cooldown learning rate is set to ``warmup_min_lr``.
                If ``cooldown_num_steps <= 0``, the cooldown learning rate is set to ``decay_min_lr``.
            cooldown_num_steps: Number of steps to cooldown from ``decay_min_lr`` to ``cooldown_min_lr``.
            last_batch_iteration: Index of the last batch.

        """

        if not (
            total_num_steps
            >= warmup_num_steps
            + warmup_patience_num_steps
            + decay_num_steps
            + decay_patience_num_steps
            + cooldown_num_steps
        ):
            raise ValueError(
                f"`total_num_steps` must be >= {warmup_num_steps + warmup_patience_num_steps + decay_num_steps + decay_patience_num_steps + cooldown_num_steps}, but got {total_num_steps}."
            )
        if decay_type not in ["cosine", "inverse_square_root"]:
            raise ValueError(f"`decay_type` must be 'cosine' or 'inverse_square_root', but got '{decay_type}'.")

        super().__init__(
            optimizer,
            total_num_steps,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )

        self._gamma = 0.0
        self._warmup_patience_num_steps = warmup_patience_num_steps

        # If `decay_num_steps > 0`, then `decay_min_lr` must be defined,
        # otherwise, `decay_min_lr` is set to `warmup_max_lr` to prevent any decaying
        if decay_num_steps > 0:
            decay_min_lr = warmup_min_lr if decay_min_lr is None else decay_min_lr
        else:
            decay_min_lr = warmup_max_lr

        self._decay_lrs = self._format_param(decay_min_lr)
        self._decay_delta_lrs = [big - small for big, small in zip(self._max_lrs, self._decay_lrs)]
        self._decay_num_steps = decay_num_steps
        self._decay_patience_num_steps = decay_patience_num_steps
        self._decay_type = decay_type
        self._decay_cycles = decay_cycles

        # If `cooldown_num_steps > 0`, then `cooldown_min_lr` must be defined,
        # otherwise, `cooldown_min_lr` is set to `decay_min_lr` to prevent any cooldown
        if cooldown_num_steps > 0:
            cooldown_min_lr = cooldown_min_lr if cooldown_min_lr is not None else warmup_min_lr
        else:
            cooldown_min_lr = decay_min_lr

        self._cooldown_lrs = self._format_param(cooldown_min_lr)
        self._cooldown_delta_lrs = [big - small for big, small in zip(self._decay_lrs, self._cooldown_lrs)]
        self._cooldown_num_steps = cooldown_num_steps

    def _set_gamma(self) -> None:
        # Warmup
        if self._last_batch_iteration < self._warmup_num_steps:
            if self._warmup_type == "log":
                self._gamma = self._inverse_log_warm_up * math.log(self._last_batch_iteration + 1)
            elif self._warmup_type == "linear":
                self._gamma = self._last_batch_iteration / self._warmup_num_steps

        # Patience after warmup
        elif self._last_batch_iteration < self._warmup_num_steps + self._warmup_patience_num_steps:
            self._gamma = 1.0

        # Decay
        elif (
            self._last_batch_iteration
            < self._warmup_num_steps + self._warmup_patience_num_steps + self._decay_num_steps
        ):
            if self._decay_type == "cosine":
                progress = float(
                    self._last_batch_iteration - self._warmup_num_steps - self._warmup_patience_num_steps
                ) / float(max(1, self._decay_num_steps))
                self._gamma = 0.5 * (1.0 + math.cos(math.pi * float(self._decay_cycles) * 2.0 * progress))

            elif self._decay_type == "inverse_square_root":
                progress_start = self._warmup_num_steps + self._warmup_patience_num_steps
                progress_end = progress_start + self._decay_num_steps
                self._gamma = (1 / math.sqrt(self._last_batch_iteration) - 1 / math.sqrt(progress_end)) / (
                    1 / math.sqrt(progress_start) - 1 / math.sqrt(progress_end)
                )

        # Patience after decay
        elif self._last_batch_iteration < (
            self._warmup_num_steps
            + self._warmup_patience_num_steps
            + self._decay_num_steps
            + self._decay_patience_num_steps
        ):
            self._gamma = 0.0

        # Linear cooldown
        elif (
            self._last_batch_iteration
            < self._warmup_num_steps
            + self._warmup_patience_num_steps
            + self._decay_num_steps
            + self._decay_patience_num_steps
            + self._cooldown_num_steps
        ):
            self._gamma = max(
                0.0,
                float(self._total_num_steps - self._last_batch_iteration)
                / float(
                    max(
                        1.0,
                        self._total_num_steps
                        - self._warmup_num_steps
                        - self._warmup_patience_num_steps
                        - self._decay_num_steps
                        - self._decay_patience_num_steps,
                    )
                ),
            )

        # After going through all `cooldown_num_steps`, the last `gamma` is fixed
        # and the learning rate stays fixed for the remaining steps

    def get_lr(self) -> List[float]:
        if self._last_batch_iteration < 0:
            return [0.0]

        # Set current value for `gamma`
        self._set_gamma()

        # Warmup and patience after warmup
        if self._last_batch_iteration < self._warmup_num_steps + self._warmup_patience_num_steps:
            return [min_lr + (delta_lr * self._gamma) for min_lr, delta_lr in zip(self._min_lrs, self._delta_lrs)]

        # Decay and patience after decay
        if (
            self._last_batch_iteration
            < self._warmup_num_steps
            + self._warmup_patience_num_steps
            + self._decay_num_steps
            + self._decay_patience_num_steps
        ):
            return [
                decay_lr + (decay_delta_lr * self._gamma)
                for decay_lr, decay_delta_lr in zip(self._decay_lrs, self._decay_delta_lrs)
            ]

        # Linear cooldown
        return [
            cooldown_lr + (cooldown_delta_lr * self._gamma)
            for cooldown_lr, cooldown_delta_lr in zip(self._cooldown_lrs, self._cooldown_delta_lrs)
        ]
