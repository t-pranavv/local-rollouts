# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import natsort


def _find_checkpoints(
    checkpoint_dir: Union[str, Path],
    checkpoint_regex: Optional[Union[str, re.Pattern]] = "",
    n_checkpoints: Optional[int] = None,
    reverse: bool = True,
) -> List[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return []

    checkpoints = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and re.search(checkpoint_regex, path.name) is not None
    ]
    checkpoints = natsort.natsorted(checkpoints, reverse=reverse)

    # If no sub-directories are found, assume that `checkpoint_dir` is the checkpoint
    if len(checkpoints) == 0:
        checkpoints = [checkpoint_dir]

    if n_checkpoints is not None:
        checkpoints = checkpoints[:n_checkpoints]

    return checkpoints


def _get_basename(path: Union[str, Path], n_levels: int = 1) -> str:
    path_parts = Path(path).parts

    if n_levels > len(path_parts):
        return str(path)
    if n_levels <= 0:
        return ""

    return str(Path(*path_parts[-n_levels:]))


def get_checkpoints_info(
    checkpoint_dir: Union[str, Path],
    checkpoint_regex: Optional[Union[str, re.Pattern]] = "",
    n_checkpoints: Optional[int] = None,
    reverse: bool = True,
) -> List[Dict[str, Union[Path, str, int]]]:
    """Get information about available checkpoints in a given directory.

    Args:
        checkpoint_dir: Path to the directory that might contain checkpoints.
        checkpoint_regex: Regular expression to match the checkpoint folder name.
        n_checkpoints: Number of checkpoints to return.
            If ``None``, all checkpoints are returned.
        reverse: Whether to return the checkpoints in reverse order.

    Returns:
        List of dictionaries with checkpoint information.

    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return [{"path": checkpoint_dir, "basename": "", "step": 0}]

    checkpoints = _find_checkpoints(
        checkpoint_dir,
        checkpoint_regex=checkpoint_regex,
        n_checkpoints=n_checkpoints,
        reverse=reverse,
    )

    checkpoints_info = []
    for checkpoint in checkpoints:
        checkpoint_basename = _get_basename(checkpoint, n_levels=1)

        if checkpoint_basename.isdigit():
            step = int(checkpoint_basename)
        elif checkpoint_basename.split("-")[-1].isdigit():
            step = int(checkpoint_basename.split("-")[-1])
        elif checkpoint_basename.split("-")[-1] == "last":
            step = sys.maxsize
        else:
            step = 0

        checkpoints_info.append({"path": checkpoint, "basename": checkpoint_basename, "step": step})

    return checkpoints_info


def get_full_path(path: Union[str, Path], create_folder: bool = False) -> Path:
    """Get the full path to a file or folder.

    Args:
        path: Path to the file or folder.
        create_folder: Whether to create the folder if it does not exist.

    Returns:
        Full path to the file or folder.

    """

    if not path:
        raise ValueError(f"`path` must be defined, but got '{path}'.")

    full_path = Path(path).expanduser().resolve()

    if create_folder:
        if full_path.suffix:
            # If the path is a file, ensure its parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # If the path is a directory, ensure it exists
            full_path.mkdir(parents=True, exist_ok=True)

    return full_path


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file (optionally compressed with ``gzip``).

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary-based contents of the file.

    """

    file_path = Path(file_path)

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt") as f:
            return json.load(f)

    with file_path.open("r", encoding="utf8") as f:
        return json.load(f)


def save_json_file(
    obj: Dict[str, Any],
    file_path: Union[str, Path],
    create_folder: bool = False,
    mode: str = "w",
    indent: int = 4,
    sort_keys: bool = False,
) -> None:
    """Save an object to a JSON file (optionally compressed with ``gzip``).

    Args:
        obj: Object to save.
        file_path: Path to the file.
        create_folder: Whether to create the folder if it does not exist.
        mode: Mode to open the file.
        indent: Number of spaces to indent.
        sort_keys: Whether to sort the keys in the JSON file.

    """

    file_path = get_full_path(file_path, create_folder=create_folder)

    if file_path.suffix == ".gz":
        if mode != "wt":
            raise ValueError(f"`mode` must be 'wt' when saving to a compressed file, but got '{mode}'.")

        with gzip.open(file_path, mode) as f:
            json.dump(obj, f, indent=indent, sort_keys=sort_keys)
    else:
        with file_path.open(mode, encoding="utf8") as f:
            json.dump(obj, f, indent=indent, sort_keys=sort_keys)


def load_jsonl_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file (optionally compressed with ``gzip``).

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary-based contents of the file.

    """

    file_path = Path(file_path)

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt") as f:
            lines = f.readlines()
    else:
        with file_path.open("r", encoding="utf8") as f:
            lines = f.readlines()

    return [json.loads(line) for line in lines if line]


def save_jsonl_file(
    obj: List[Dict[str, Any]], file_path: Union[str, Path], create_folder: bool = False, mode: str = "w"
) -> None:
    """Save a list of objects to a JSONL file (optionally compressed with ``gzip``).

    Args:
        obj: List of objects to save.
        file_path: Path to the file.
        create_folder: Whether to create the folder if it does not exist.
        mode: Mode to open the file.

    """

    file_path = get_full_path(file_path, create_folder=create_folder)

    if file_path.suffix == ".gz":
        if mode != "wt":
            raise ValueError(f"`mode` must be 'wt' when saving to a compressed file, but got '{mode}'.")

        with gzip.open(file_path, mode) as f:
            for o in obj:
                f.write(json.dumps(o) + "\n")
    else:
        with file_path.open(mode, encoding="utf8") as f:
            for o in obj:
                f.write(json.dumps(o) + "\n")


def validate_file_extension(file_path: Any, extensions: Union[str, List[str]]) -> bool:
    """Validate whether a file has a valid extension.

    Args:
        file_path: Path to the file.
        extensions: Valid extension(s).

    Returns:
        Whether the file has a valid extension.

    """

    if not isinstance(file_path, (str, Path)):
        return False

    file_path = Path(file_path)
    extensions = extensions if isinstance(extensions, list) else [extensions]

    return file_path.suffix in extensions


def is_file_available(file_path: Any, file_extension: str = "") -> Optional[Path]:
    """Check if ``file_path`` has a corresponding ``file_extension`` file.

    Args:
        file_path: Path to the file.
        file_extension: Extension of the corresponding file.

    Returns:
        Path to the corresponding file if available, otherwise ``None``.

    """

    if not isinstance(file_path, (str, Path)):
        return None

    file_path = Path(file_path)
    file_path_with_extension = file_path.parent / (file_path.stem + file_extension)

    return file_path_with_extension if file_path_with_extension.exists() else None


def is_checkpoint_available(checkpoint_dir: Any, step: Optional[int] = None) -> bool:
    """Check if a checkpoint is available in the given directory.

    Args:
        checkpoint_dir: Path to the directory that might contain checkpoints.
        step: Step of the checkpoint.

    Returns:
        Whether a checkpoint is available.

    """

    if not isinstance(checkpoint_dir, (str, Path)):
        return False

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return False
    if not checkpoint_dir.is_dir():
        raise ValueError(f"`checkpoint_dir` must be a directory, but got '{checkpoint_dir}'.")

    if step is None:
        latest_file = checkpoint_dir / "latest"
        if not latest_file.exists():
            return False
        step = int(latest_file.read_text())

    return (checkpoint_dir / str(step)).exists()
