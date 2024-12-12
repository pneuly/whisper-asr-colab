import os
import pytest
from numpy import ndarray
from whisper_asr_colab.audio import Audio

@pytest.fixture
def data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@pytest.fixture
def audio(data_dir) -> ndarray:
    audio = Audio(file_path=os.path.join(data_dir, "asr_test.m4a"))
    return audio.ndarray

@pytest.fixture
def asr_segments_file(data_dir) -> str:
    return os.path.join(data_dir, "asr_test.m4a_segments.json")

@pytest.fixture
def diarized_transcript(data_dir) -> str:
    return os.path.join(data_dir, "asr_test.m4a_diarized.txt")
