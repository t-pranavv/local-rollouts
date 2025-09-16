# Copyright (c) lucidrains.
# Licensed under the MIT license.
# https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py

from typing import Callable, Optional, Tuple

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer.

    Reference:
        Symbolic Discovery of Optimization Algorithms.
        https://arxiv.org/abs/2302.06675.

    """

    def __init__(
        self,
        params: torch.nn.ParameterList,
        lr: float = 1e-4,
        betas: Optional[Tuple[float, float]] = None,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            weight_decay: Weight decay coefficient.

        """

        if lr <= 0.0:
            raise ValueError(f"`lr` must be > 0.0, but got {lr}.")

        betas = betas or (0.9, 0.999)
        if not all([0.0 <= beta <= 1.0 for beta in betas]):
            raise ValueError(f"`betas` must be in [0, 1], but got {betas}.")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        decay: float,
        beta1: float,
        beta2: float,
    ) -> None:
        # Step weight decay
        p.data.mul_(1 - lr * decay)

        # Weight update
        update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # Decay the momentum by running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> torch.FloatTensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group["params"]):
                grad, state = p.grad, self.state[p]
                lr, decay = group["lr"], group["weight_decay"]
                beta1, beta2 = group["betas"]

                # Initial state is the exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                self._update(p, grad, exp_avg, lr, decay, beta1, beta2)

        return loss
