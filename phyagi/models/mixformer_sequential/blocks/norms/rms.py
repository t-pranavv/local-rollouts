# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union

import torch

from phyagi.utils.type_utils import to_torch_dtype


class RMSLayerNorm(torch.nn.LayerNorm):
    """Root Mean Square layer normalization."""

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-05,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=True,
            device=device,
            dtype=to_torch_dtype(dtype),
        )

        # Un-register the bias parameter
        self.register_parameter("bias", None)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._rms_norm(x.float()) * self.weight
        return output.type_as(x)
