# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

from phyagi.rl.models.reference import Reference


def test_reference():
    mock_logger = MagicMock()
    mock_checkpoint_manager = MagicMock()
    mock_device_mesh = MagicMock()
    mock_submesh = MagicMock()
    mock_submesh.size.return_value = 1
    mock_device_mesh.__getitem__.return_value = mock_submesh

    mock_model = MagicMock()
    mock_model.config.to_diff_dict.return_value = {"dummy_key": "dummy_value"}

    config = MagicMock()
    config.use_meta_tensor = False
    config.checkpoint_mode = "sync"
    config.dtype = "float32"
    config.model = {"param1": "val1"}

    with patch.object(Reference, "configure_model", return_value=mock_model) as mock_configure_model, patch(
        "phyagi.rl.models.reference.apply_fsdp_mixformer_sequential"
    ) as mock_apply_fsdp:

        ref = Reference(config, mock_device_mesh, mock_checkpoint_manager, mock_logger)
        assert ref.config.dtype == "float32"
        assert ref.device_mesh == mock_device_mesh
        assert ref.logger == mock_logger
        assert ref.model == mock_model
        assert ref.precision == "float32"
        assert ref.fsdp_offload is True

        mock_configure_model.assert_called_once_with(param1="val1")
        mock_apply_fsdp.assert_called_once_with(mock_model, mock_submesh, "float32", cpu_offload=True)

        assert ref.config.model["dummy_key"] == "dummy_value"
        assert ref.config.model["torch_dtype"] == "float32"
