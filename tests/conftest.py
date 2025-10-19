import os
import pytest


@pytest.fixture
def data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def audio(data_dir) -> str:
    audio = os.path.join(data_dir, "asr_test.m4a")
    return audio


@pytest.fixture
def asr_segments_file(data_dir) -> str:
    return os.path.join(data_dir, "asr_test.m4a_segments.json")


@pytest.fixture
def diarized_transcript(data_dir) -> str:
    return os.path.join(data_dir, "asr_test.m4a_diarized.txt")
