# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys
import tempfile

from phyagi.utils.logging_utils import FORMATTER, LOCAL_RANK, _RankFilter, get_logger


def test_get_logger():
    logger_name = "test_logger"

    with tempfile.NamedTemporaryFile() as temp_file:
        logger = get_logger(logger_name, temp_file.name)
        assert logger.name == logger_name
        assert len(logger.handlers) == 2

        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[0].stream == sys.stdout
        assert logger.handlers[0].formatter == FORMATTER

        assert isinstance(logger.handlers[1], logging.handlers.TimedRotatingFileHandler)
        assert logger.handlers[1].formatter == FORMATTER

        assert isinstance(logger.filters[0], _RankFilter)
        assert logger.filters[0]._rank == LOCAL_RANK

        assert logger.propagate is False
