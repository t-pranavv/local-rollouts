# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import copy
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf


def _convert_nested_attrs_to_dict(attrs: Dict[str, Any], filter_value: Optional[Any] = None) -> Dict[str, Any]:
    def _attr_to_dict(key: str, value: Any) -> Dict[str, Any]:
        if len(key) == 1:
            return {key[0]: value}
        return _attr_to_dict(key[:-1], {key[-1]: value})

    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = _deep_update(d.get(k, {}), v)
            else:
                if v is not filter_value:
                    d[k] = v
        return d

    converted_attrs = {}
    for key, value in attrs.items():
        _deep_update(converted_attrs, _attr_to_dict(key.split("."), value))

    return converted_attrs


def _convert_list_to_dotlist(args: List[str], delimiter: Optional[str] = ",") -> List[str]:
    dotlist = []

    # Find position, name, and value of all keys (prefixed by `--`)
    keys = [
        (i, *re.match(r"--((?:\w+\.)*\w+)(?:=(\S+))?", arg).groups())
        for i, arg in enumerate(args)
        if re.search("--", arg)
    ]

    for key in keys:
        # `arg` is the first captured group
        arg = key[1]

        # `value` is the second captured group if available,
        # or the next string in the original list
        if key[2] is not None:
            value = key[2]
        else:
            value_idx = key[0] + 1
            value = args[value_idx] if value_idx < len(args) and "--" not in args[value_idx] else True

        # If the value contains the delimiter, we split it
        if isinstance(value, str) and delimiter in value:
            value = value.split(delimiter)

        dotlist.append(f"{arg}={value}")

    return dotlist


def for_resolver(obj: Any, n: int = 1, replace_char: Optional[str] = None) -> List[Any]:
    """Resolve a for-loop operation (for OmegaConf).

    Examples:
        >>> ${for:${oc.create:{cache_dir:shard_?/train.npy,seq_len:4096}},2,?}
        >>> [{"cache_dir": "shard_0/train.npy", "seq_len": 4096}, {"cache_dir": "shard_1/train.npy", "seq_len": 4096}]

    Args:
        obj: Input object.
        n: Number of times to repeat the object.
        replace_char: Character to replace in the object.

    Returns:
        List of objects.

    """

    if len(replace_char) != 1:
        raise ValueError(f"`replace_char` must be a single character, but got '{replace_char}'.")

    objs = [copy.deepcopy(obj) for _ in range(n)]
    for i, obj in enumerate(objs):
        if isinstance(obj, str):
            obj = obj.replace(replace_char, str(i))
        elif isinstance(obj, (DictConfig, dict)):
            for k, v in obj.items():
                obj[k] = v.replace(replace_char, str(i)) if isinstance(v, str) else v

    return objs


def filter_dict(input_dict: Dict[str, Any], value: Optional[Any] = None) -> Dict[str, Any]:
    """Filter a dictionary to remove ``value``.

    Args:
        input_dict: Dictionary to be filtered.
        value: Value to be removed from the dictionary.

    Returns:
        Filtered dictionary.

    """

    if not isinstance(input_dict, dict):
        raise TypeError(f"`input_dict` must be a dict, but got '{type(input_dict)}'.")

    return {k: v for k, v in input_dict.items() if v != value}


def filter_lists(*args: Tuple[List[Any], ...], value: Optional[Any] = None) -> Tuple[List[Any], ...]:
    """Filter lists by removing ``value``.

    Args:
        args: Lists to be filtered.
        value: Value to be removed from the lists.

    Returns:
        Filtered lists.

    """

    if not all(isinstance(arg, list) for arg in args):
        raise TypeError(f"Items from `args` must be lists, but got {[type(arg) for arg in args]}.")

    filtered_tuples = [t for t in zip(*args) if all(x != value for x in t)]
    filtered_lists = tuple(map(list, zip(*filtered_tuples))) if filtered_tuples else tuple([[]] * len(args))

    # If the filtered tuple has only one element, we return it as a list to avoid unpacking it
    # when calling the function
    if len(filtered_lists) == 1:
        return filtered_lists[0]

    return filtered_lists


def override_nested_dict(input_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Manually override values in ``input_dict`` with values from ``override_dict``.

    This function recursively overrides values in the input dictionary with values from the override
    dictionary. If a key in the override dictionary is a nested dictionary, it will recursively
    override the corresponding key in the input dictionary.

    Args:
        input_dict: Dictionary to override values from.
        override_dict: Dictionary containing the new values.

    Returns:
        Overridden dictionary.

    """

    if input_dict is None:
        return override_dict

    input_dict = deepcopy(input_dict)

    for key, value in override_dict.items():
        if isinstance(value, dict) and key in input_dict:
            input_dict[key] = override_nested_dict(input_dict[key], value)
        else:
            input_dict[key] = value

    return input_dict


def load_config(*configs, use_native_types: bool = True) -> Union[Dict[str, Any], DictConfig]:
    """Load a set of arguments and merge them into a unique configuration object, resolving
    from left to right.

    Each object of ``configs`` is expected to be either:

    - Path-like object representing the path to a YAML/json config file.
    - Dictionary with the configuration.
    - ``argparse.Namespace`` object.
    - List of dot list arguments (e.g., ``["--a", "1", "--b", "2"]``).

    Args:
        configs: Arguments to be resolved into a final configuration.
        use_native_types: If ``True``, configuration is loaded using ``OmegaConf.to_object()``
            and native types are preserved.

    Returns:
        Merged configuration.

    """

    loaded_configs = []
    for cfg in configs:
        # If the config is a path-like object, we parse it to find its extension
        if isinstance(cfg, (str, Path)):
            file_extension = Path(cfg).suffix

            # If the extension is .json, we load the file and use OmegaConf.create()
            if file_extension == ".json":
                with open(cfg, "r") as f:
                    loaded_configs.append(OmegaConf.create(json.load(f)))

            # If the file extension is .jsonl, we parse the lines and use OmegaConf.create()
            if file_extension == ".jsonl":
                with open(cfg, "r") as f:
                    json_lines = [json.loads(line) for line in f.readlines()]
                    loaded_configs.append(OmegaConf.create(json_lines))

            # If the extension is .yaml, we use OmegaConf.load()
            if file_extension == ".yaml":
                loaded_configs.append(OmegaConf.load(cfg))

        # If the config is a dictionary, we use OmegaConf.create()
        if isinstance(cfg, dict):
            loaded_configs.append(OmegaConf.create(cfg))

        # If the config is a list (usually from CLI), we convert it to a dot-list
        # and use OmegaConf.from_dotlist()
        if isinstance(cfg, list):
            loaded_configs.append(OmegaConf.from_dotlist(_convert_list_to_dotlist(cfg)))

        # If the config is a argparse.Namespace, we convert it to a dictionary
        # and use OmegaConf.create()
        if isinstance(cfg, argparse.Namespace):
            cfg = _convert_nested_attrs_to_dict(vars(cfg))
            loaded_configs.append(OmegaConf.create(cfg))

        if isinstance(cfg, DictConfig):
            loaded_configs.append(cfg)

    merged_config = OmegaConf.merge(*loaded_configs)

    if use_native_types:
        return OmegaConf.to_object(merged_config)

    return merged_config


def save_config(config: Union[Dict[str, Any], OmegaConf], output_path: Union[str, Path]) -> None:
    """Save a configuration object to a YAML file.

    Args:
        config: Configuration object.
        output_path: Path to save the configuration.

    """

    with open(output_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)


# Register custom resolvers for OmegaConf
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("fdiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("for", for_resolver)
