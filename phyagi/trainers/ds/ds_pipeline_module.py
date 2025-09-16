# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import torch.nn as nn
from deepspeed import PipelineModule
from deepspeed.pipe import LayerSpec


class DsPipelineModule(PipelineModule):
    """DeepSpeed pipeline module.

    This class improves :class:`deepspeed.pipe.PipelineModule` by enabling the partition function
    to accept a list as the partition method.

    Examples:
        >>> self._partition_layers([0, 4, 8])
        >>> # This will result in two stages, stage 1 with layers [0, 1, 2, 3]
        >>> # and stage 2 with layers [4, 5, 6, 7]

    """

    def _partition_layers(self, method: Union[List[int], str] = "uniform") -> None:
        """Set and print the bounds for a specific ``stage_id``.

        Args:
            method: Partition method. If a list is provided, it will use the custom
                partitioning method. Otherwise, it will use the original function.

        """

        # Original function
        if not isinstance(method, list):
            super()._partition_layers(method)
            return

        num_stages = self._topo.get_dim("pipe")
        stage_id = self._topo.get_coord(self.global_rank).pipe
        method = [int(layer) for layer in method]

        if len(method) != num_stages + 1:
            raise ValueError(f"`method` must have a length of {num_stages + 1}, but got {len(method)}.")
        if method[-1] != len(self._layer_specs) + 1:
            raise ValueError(
                f"`method` must have a last element of {len(self._layer_specs) + 1}, but got {method[-1]}."
            )

        self.parts = method
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f"stage={stage} layers={stop - start}")

                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)

                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass

                    print(f"    {idx+start:2d}: {name}")

            if self.loss_fn:
                try:
                    print(f"  loss: {self.loss_fn.__name__}")
                except AttributeError:
                    print(f"  loss: {self.loss_fn.__class__.__name__}")

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
