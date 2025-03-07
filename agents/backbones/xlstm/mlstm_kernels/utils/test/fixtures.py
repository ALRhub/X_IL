#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

# We declare this here globally to enforca that there is only one timestamp per test session
TIMESTAMP = None

TEST_OUTPUT_FOLDER = Path(__file__).parents[3] / "outputs_tests"


@pytest.fixture(scope="session")
def test_session_folder() -> Path:
    global TIMESTAMP
    if TIMESTAMP is None:
        TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    timestamp = TIMESTAMP

    test_output_folder = TEST_OUTPUT_FOLDER / timestamp

    test_output_folder.mkdir(parents=True, exist_ok=True)

    logfile = test_output_folder / "pytest.log"
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[file_handler],
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        force=True,
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(f"Logging to {logfile}")
    return test_output_folder


@pytest.fixture
def test_output_folder() -> Path:
    return TEST_OUTPUT_FOLDER / "test_data"
