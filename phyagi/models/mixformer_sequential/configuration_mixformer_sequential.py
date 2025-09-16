# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks import DEFAULT_BLOCK
from phyagi.models.mixformer_sequential.blocks.mixers import DEFAULT_MIXER
from phyagi.models.mixformer_sequential.blocks.mlps import DEFAULT_MLP
from phyagi.models.mixformer_sequential.blocks.norms import DEFAULT_NORM
from phyagi.utils.config import override_nested_dict
from phyagi.utils.logging_utils import get_logger
from phyagi.version import __version__

logger = get_logger(__name__)

MIXFORMER_SEQUENTIAL_MODEL_TYPE = "mixformer-sequential"


class MixFormerSequentialConfig(PretrainedConfig):
    """MixFormer (sequential) configuration."""

    model_type = MIXFORMER_SEQUENTIAL_MODEL_TYPE
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "n_ctx": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "cross_sample_sep": "cross_sample_token_id",
    }

    default_arch = {
        "block_cls": DEFAULT_BLOCK,
        "norm": {
            "norm_cls": DEFAULT_NORM,
        },
        "mixer": {
            "mixer_cls": DEFAULT_MIXER,
        },
        "mlp": {
            "mlp_cls": DEFAULT_MLP,
        },
    }

    def __init__(
        self,
        vocab_size: int = 50304,
        n_positions: int = 2048,
        n_embd: int = 1024,
        n_layer: int = 20,
        n_inner: Optional[int] = None,
        n_head: int = 16,
        n_head_kv: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        activation_function: str = "gelu_new",
        embd_layer: str = "default",
        architecture: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_vocab_size_multiple: int = 64,
        gradient_checkpointing: bool = False,
        cross_sample_token_id: Optional[int] = None,
        cp_size: int = 1,
        tp_size: int = 1,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> None:
        self.vocab_size = int(math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.rotary_dim = rotary_dim or (n_embd // n_head)
        self.activation_function = activation_function
        self.embd_layer = embd_layer
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache
        self.cross_sample_token_id = cross_sample_token_id
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.phyagi_version = __version__

        self.architecture = self._format_architecture(architecture)

        # `architectures` is used by `transformers` to specify a list of classes related to the model
        architectures = kwargs.pop("architectures", [])

        super().__init__(tie_word_embeddings=tie_word_embeddings, architectures=architectures, **kwargs)

    def _format_architecture(
        self, arch: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        def __format_architecture(arch: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in self.default_arch.items():
                if key not in arch:
                    arch[key] = value
            return arch

        arch = arch or {}
        if not isinstance(arch, (dict, list)):
            raise TypeError(f"`architecture` must be a dictionary or list of dictionaries, but got '{type(arch)}'.")

        if isinstance(arch, list):
            for i in range(len(arch)):
                arch[i] = __format_architecture(arch[i]) if isinstance(arch[i], dict) else self.default_arch
        elif isinstance(arch, dict):
            arch = __format_architecture(arch)

        return arch

    @classmethod
    def from_dict(
        cls: MixFormerSequentialConfig, config_dict: Dict[str, Any], **kwargs
    ) -> Union[MixFormerSequentialConfig, Tuple[MixFormerSequentialConfig, Dict[str, Any]]]:
        """Constructs a :class:`MixFormerSequentialConfig` from a Python dictionary of parameters.

        Args:
            config_dict: Dictionary used to instantiate the configuration object.

        Returns:
            An instance of :class:`MixFormerSequentialConfig`, and optionally a dictionary of unused keyword arguments that were not consumed by the
                configuration's :meth:`from_dict`.

        """

        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        arch_kwargs = kwargs.pop("architecture", {})

        # Ensure `super().from_dict()` is called with `return_unused_kwargs` as True to prevent
        # `phyagi.models.modeling_utils.PreTrainedModel.from_pretrained()` from breaking
        config, kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        config.architecture = (
            override_nested_dict(config.architecture, arch_kwargs) if arch_kwargs else config.architecture
        )

        if return_unused_kwargs:
            return config, kwargs

        return config
