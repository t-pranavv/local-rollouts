# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license.
# https://github.com/facebookresearch/dadaptation

from typing import Callable, Optional, Tuple

import torch
from torch.optim import Optimizer


def _to_real(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x.real
    return x


class DAdaptAdam(Optimizer):
    """D-Adaptation Adam optimizer.

    Reference:
        Learning-Rate-Free Learning by D-Adaptation.
        https://arxiv.org/abs/2301.07733.

    """

    def __init__(
        self,
        params: torch.nn.ParameterList,
        lr: float = 1.0,
        eps: float = 1e-8,
        d0: float = 1e-6,
        betas: Optional[Tuple[float, float]] = None,
        weight_decay: float = 0.0,
        decouple: bool = False,
        growth_rate: float = float("inf"),
    ) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            eps: Term added to the denominator to improve numerical stability.
            d0: Initial value of the gradient-based term.
            betas: Coefficients used for computing running averages of gradient and its square.
            weight_decay: Weight decay coefficient.
            decouple: Whether to decouple the weight decay from the gradient-based term.
            growth_rate: Growth rate of the gradient-based term.

        """

        if lr <= 0.0:
            raise ValueError(f"`lr` must be > 0.0, but got {lr}.")
        if eps <= 0.0:
            raise ValueError(f"`eps` must be > 0.0, but got {eps}.")
        if d0 <= 0.0:
            raise ValueError(f"`d0` must be > 0.0, but got {d0}.")

        betas = betas or (0.9, 0.999)
        if not all([0.0 <= beta <= 1.0 for beta in betas]):
            raise ValueError(f"`betas` must be in [0, 1], but got {betas}.")

        defaults = dict(
            lr=lr,
            eps=eps,
            d=d0,
            betas=betas,
            weight_decay=weight_decay,
            k=0,
            gsq_weighted=0.0,
            decouple=decouple,
            growth_rate=growth_rate,
        )

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> torch.FloatTensor:
        loss = None
        if closure is not None:
            loss = closure()

        g_sq = 0.0
        sksq_weighted = 0.0
        sk_l1 = 0.0

        lr = max(group["lr"] for group in self.param_groups)

        group = self.param_groups[0]
        gsq_weighted = group["gsq_weighted"]
        d = group["d"]
        dlr = d * lr

        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        beta1, beta2 = group["betas"]

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            group_lr = group["lr"]
            if group_lr not in [lr, 0.0]:
                raise ValueError(
                    f"Learning rates in different parameter groups are only supported for 0.0 or {lr}, but got {group_lr}."
                )

            for p in filter(lambda p: p.grad is not None, group["params"]):
                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        _to_real(p.data), memory_format=torch.preserve_format
                    ).detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad_grad = _to_real(grad * grad.conj())

                # Adam EMA updates
                if group_lr > 0:
                    exp_avg.mul_(beta1).add_(grad, alpha=dlr * (1 - beta1))
                    exp_avg_sq.mul_(beta2).add_(grad_grad, alpha=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(eps)

                    g_sq += grad_grad.div_(denom).sum().item()

                    s = state["s"]
                    s.mul_(beta2).add_(grad, alpha=dlr * (1 - beta2))
                    sksq_weighted += _to_real(s * s.conj()).div_(denom).sum().item()
                    sk_l1 += s.abs().sum().item()

        gsq_weighted = beta2 * gsq_weighted + g_sq * (dlr**2) * (1 - beta2)
        d_hat = d

        # If there is no progress, return loss as it is
        if sk_l1 == 0:
            return loss

        # If there are available gradients, `sk_l1`` > 0 (unless |g|=0)
        if lr > 0.0:
            global_sksq_weighted = sksq_weighted
            global_gsq_weighted = gsq_weighted
            global_sk_l1 = sk_l1

            d_hat = (global_sksq_weighted / (1 - beta2) - global_gsq_weighted) / global_sk_l1
            d = max(d, min(d_hat, d * growth_rate))

        for group in self.param_groups:
            group["gsq_weighted"] = gsq_weighted
            group["d"] = d

            group_lr = group["lr"]
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            for p in filter(lambda p: p.grad is not None, group["params"]):
                grad = p.grad.data

                state = self.state[p]
                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                denom = exp_avg_sq.sqrt().add_(eps)
                denom = denom.type(p.type())

                # Apply weight decay (de-scoupled variant)
                if decay != 0 and decouple and group_lr > 0:
                    p.data.add_(p.data, alpha=-decay * dlr)

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group["k"] = k + 1

        return loss
