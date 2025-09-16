# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import difflib
from pathlib import Path
from typing import List

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from phyagi.models.registry import get_model
from phyagi.utils.config import load_config

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "scripts" / "train" / "configs"
MODEL_ARCHS_DIR = ROOT_DIR / "tests" / "scripts" / "model_archs"
ALLOWED_SUBDIRS = ["phi1", "phi1.5", "phi2", "phi3", "phi4", "qwen2.5"]


def _yaml_files(root: Path) -> List[Path]:
    yaml_files = []

    for ext in ("*.yml", "*.yaml"):
        yaml_files.extend(root.glob(ext))

    for subdir in ALLOWED_SUBDIRS:
        for ext in ("*.yml", "*.yaml"):
            yaml_files.extend((root / subdir).rglob(ext))

    return yaml_files


YAML_FILES = _yaml_files(CONFIG_DIR)


def _get_model_architecture(yaml_path: Path) -> str:
    config = load_config(yaml_path, use_native_types=False)

    # Disables any `flash_attn` configurations to avoid generating a different architecture
    mixer = OmegaConf.select(config, "model.architecture.mixer", default=None)
    if mixer is not None:
        mixer["flash_attn"] = False
        mixer["fused_dense"] = False
        OmegaConf.update(config, "model.architecture.mixer", mixer)

    mlp_cls = OmegaConf.select(config, "model.architecture.mlp.mlp_cls", default=None)
    if mlp_cls == "fused_mlp":
        OmegaConf.update(config, "model.architecture.mlp.mlp_cls", "mlp")

    norm_cls = OmegaConf.select(config, "model.architecture.norm.norm_cls", default=None)
    if norm_cls == "flash_rms":
        OmegaConf.update(config, "model.architecture.norm.norm_cls", "rms")

    try:
        config = OmegaConf.to_object(config)
    except InterpolationResolutionError:
        pytest.skip("`config` has at least one environment variable.")

    if "model" not in config:
        pytest.skip("`config` lacks `model` section.")

    if "pretrained_model_name_or_path" in config["model"] and "<" in config["model"]["pretrained_model_name_or_path"]:
        pytest.skip("`config.model` has a placeholder in `pretrained_model_name_or_path`.")

    with torch.device("meta"):
        model = get_model(**config["model"], low_cpu_mem_usage=True)

    model_lines = str(model).splitlines()
    if len(model_lines) > 60:
        return "\n".join(model_lines[:30] + ["... [truncated] ..."] + model_lines[-30:])
    return str(model)


@pytest.mark.skipif(len(YAML_FILES) == 0, reason="No YAML configuration files found.")
@pytest.mark.parametrize("yaml_path", YAML_FILES, ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_check_model_architecture_change(yaml_path: Path):
    relative_path = yaml_path.relative_to(CONFIG_DIR)
    expected_model_path = MODEL_ARCHS_DIR / relative_path.with_suffix(".model")

    current_arch = _get_model_architecture(yaml_path)

    if not expected_model_path.exists():
        pytest.fail(f"Missing reference architecture file: {expected_model_path}.")

    expected_arch = expected_model_path.read_text()

    if current_arch != expected_arch:
        print(f"\nDetected architecture change for: {relative_path}.")
        diff = difflib.unified_diff(
            expected_arch.splitlines(), current_arch.splitlines(), fromfile="expected", tofile="current", lineterm=""
        )
        print("\n".join(diff))
        pytest.fail(f"Model architecture for '{relative_path}' has changed.")


# When running the script with `python` instead of `pytest`, this function will be called
# and it will generate the architectures for all YAML files in the specified directory
if __name__ == "__main__":
    for yaml_path in YAML_FILES:
        relative_path = yaml_path.relative_to(CONFIG_DIR)
        output_path = MODEL_ARCHS_DIR / relative_path.with_suffix(".model")

        try:
            arch = _get_model_architecture(yaml_path)
        except pytest.skip.Exception:
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(arch)

    print(f"Model architecture files generated in '{MODEL_ARCHS_DIR}'.")
