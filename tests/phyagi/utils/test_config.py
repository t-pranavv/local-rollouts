# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from phyagi.utils.config import (
    filter_dict,
    filter_lists,
    for_resolver,
    load_config,
    override_nested_dict,
    save_config,
)


@pytest.fixture()
def yaml_config(tmp_path_factory):
    content = "a:\n  b: 1\n  c: 2\nd: 3"

    p = tmp_path_factory.mktemp("data") / "config.yaml"
    p.write_text(content)

    return p


@pytest.fixture()
def json_config(tmp_path_factory):
    content = json.dumps({"a": {"b": 1, "c": 2}, "d": 3})

    p = tmp_path_factory.mktemp("data") / "config.json"
    p.write_text(content)

    return p


@pytest.fixture()
def dataclass_yaml(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sample.yaml"
    path.write_text("a: 10\nb: hello\nc: 2.71")
    return path


def test_filter_dict():
    input_dict = {"a": 1, "b": 2, "c": 3}
    expected = {"a": 1, "c": 3}
    assert filter_dict(input_dict, value=2) == expected


def test_filter_lists():
    list1 = [1, 2, 3, 4, 5]
    list2 = [6, 7, 8, 9, 10]
    expected1 = [1, 3, 4, 5]
    expected2 = [6, 8, 9, 10]

    assert filter_lists(list1, list2, value=2) == (expected1, expected2)


def test_override_nested_dict():
    input_dict = {"a": {"b": 1, "c": 2}, "d": 3}
    override_dict = {"a": {"b": 4, "c": 5}, "d": 6}
    expected = {"a": {"b": 4, "c": 5}, "d": 6}

    assert override_nested_dict(input_dict, override_dict) == expected


def test_load_config(yaml_config, json_config):
    expected = {"a": {"b": 1, "c": 2}, "d": 3}

    assert load_config(str(yaml_config)) == expected
    assert load_config(str(json_config)) == expected


def test_save_config(yaml_config):
    config = load_config(str(yaml_config))
    save_config(config, str(yaml_config))

    saved_config = load_config(str(yaml_config))
    assert config == saved_config


def test_for_resolver():
    obj = {"cache_dir": "shard_?/train.npy", "seq_len": 4096}
    assert for_resolver(obj, n=2, replace_char="?") == [
        {"cache_dir": "shard_0/train.npy", "seq_len": 4096},
        {"cache_dir": "shard_1/train.npy", "seq_len": 4096},
    ]
