import os
from pathlib import Path

import pytest

from mlflow2sql import generate


@pytest.fixture
def tmp_empty(tmp_path):
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)


@pytest.fixture
def tmp_mlflow_empty(tmp_empty):

    with generate.start_mlflow():
        yield
