# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from phyagi.utils.file_utils import (
    get_checkpoints_info,
    get_full_path,
    is_checkpoint_available,
    is_file_available,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
    validate_file_extension,
)


def test_get_checkpoints_info(tmp_path):
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-2").mkdir()

    checkpoints_info = get_checkpoints_info(tmp_path)
    assert len(checkpoints_info) == 2
    assert {
        "path": tmp_path / "checkpoint-1",
        "basename": "checkpoint-1",
        "step": 1,
    } in checkpoints_info
    assert {
        "path": tmp_path / "checkpoint-2",
        "basename": "checkpoint-2",
        "step": 2,
    } in checkpoints_info


def test_get_full_path():
    full_path = get_full_path("test_folder", create_folder=True)
    assert Path(full_path).exists()

    shutil.rmtree("test_folder", ignore_errors=True)


def test_load_json_file(tmp_path):
    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps({"key": "value"}))

    data = load_json_file(test_file)
    assert data == {"key": "value"}


def test_save_json_file(tmp_path):
    test_file = tmp_path / "test.json"
    save_json_file({"key": "value"}, test_file)

    with test_file.open("r") as f:
        data = json.load(f)
    assert data == {"key": "value"}


def test_load_jsonl_file(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps({"key": "value"}) + "\n")

    data = load_jsonl_file(test_file)
    assert data == [{"key": "value"}]


def test_save_jsonl_file(tmp_path):
    test_file = tmp_path / "test.jsonl"
    save_jsonl_file([{"key": "value"}], test_file)

    with test_file.open("r") as f:
        data = json.loads(f.readline())
    assert data == {"key": "value"}


def test_validate_file_extension():
    assert validate_file_extension("test.jpg", ".jpg") is True
    assert validate_file_extension("path/to/test.png", [".png"]) is True
    assert validate_file_extension("path/to/test.jpeg", ".jpeg") is True

    assert validate_file_extension(Path("test.jpg"), ".jpg") is True
    assert validate_file_extension(Path("path/to/test.png"), [".png"]) is True
    assert validate_file_extension(Path("path/to/test.jpeg"), ".jpeg") is True

    assert validate_file_extension("test.txt", ".jpg") is False
    assert validate_file_extension("test.pdf", [".jpg", ".png"]) is False
    assert validate_file_extension("path/to/test.gif", [".png"]) is False
    assert validate_file_extension("path/to/test.bmp", ".jpeg") is False

    assert validate_file_extension("test.JPG", ".jpg") is False
    assert validate_file_extension("test.jpg", ".JPG") is False

    assert validate_file_extension("", ".jpg") is False
    assert validate_file_extension("test.jpg", "") is False
    assert validate_file_extension("test.jpg", []) is False


def test_is_file_available():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        file_path = temp_path / "testfile.txt"
        file_path.touch()
        assert is_file_available(file_path, ".txt") == file_path

        non_existent_path = temp_path / "nonexistent.txt"
        assert is_file_available(non_existent_path, ".txt") is None

        base_file_path = temp_path / "testfile"
        expected_file = temp_path / "testfile.json"
        expected_file.touch()
        assert is_file_available(base_file_path, ".json") == expected_file

        assert is_file_available(123, ".txt") is None
        assert is_file_available(None, ".txt") is None
        assert is_file_available({}, ".txt") is None


def test_is_checkpoint_available(tmp_path):
    assert is_checkpoint_available(12345) is False
    assert is_checkpoint_available(None) is False

    nonexistent_dir = tmp_path / "nonexistent"
    assert is_checkpoint_available(nonexistent_dir) is False

    file_path = tmp_path / "somefile"
    file_path.write_text("not a directory")
    with pytest.raises(ValueError):
        is_checkpoint_available(file_path)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    assert is_checkpoint_available(ckpt_dir) is False

    latest_file = ckpt_dir / "latest"
    latest_file.write_text("42")
    assert is_checkpoint_available(ckpt_dir) is False

    step_dir = ckpt_dir / "42"
    step_dir.mkdir()
    assert is_checkpoint_available(ckpt_dir) is True
    assert is_checkpoint_available(ckpt_dir, step=42) is True
    assert is_checkpoint_available(ckpt_dir, step=100) is False
