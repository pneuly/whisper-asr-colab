import os
import pytest

@pytest.fixture
def data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.fixture
def audio(data_dir) -> str:
    return os.path.join(data_dir, "asr_test.m4a")