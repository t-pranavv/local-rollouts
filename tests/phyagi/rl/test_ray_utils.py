# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from unittest.mock import MagicMock, patch

from phyagi.rl.ray_utils import chunk_equal, get_ray_logger, ray_wandb_graceful_shutdown


def test_chunk_equal():
    result = chunk_equal(list(range(10)), 3)
    assert len(result) == 3
    assert sum(len(chunk) for chunk in result) == 10
    assert result[0] == [0, 1, 2, 3]
    assert result[1] == [4, 5, 6]
    assert result[2] == [7, 8, 9]


@patch("phyagi.rl.ray_utils.get_logger")
@patch("phyagi.rl.ray_utils.WandbHandler")
def test_get_ray_logger(mock_wandb_handler, mock_get_logger):
    mock_logger = MagicMock(spec=logging.Logger)
    mock_get_logger.return_value = mock_logger

    logger = get_ray_logger("test_logger")
    assert isinstance(logger, logging.Logger)

    logger = get_ray_logger("test_logger", project="proj")
    mock_logger.addHandler.assert_any_call(mock_wandb_handler.return_value)


@patch("ray.shutdown")
@patch("wandb.run")
def test_ray_wandb_graceful_shutdown(mock_wandb_run, mock_ray_shutdown):
    mock_wandb_run.finish = MagicMock()

    @ray_wandb_graceful_shutdown
    def test_func(x):
        return x + 1

    result = test_func(1)
    assert result == 2
    mock_wandb_run.finish.assert_called_once_with(exit_code=1)
    mock_ray_shutdown.assert_called_once()
