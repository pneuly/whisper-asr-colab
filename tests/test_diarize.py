import os
from whisper_asr_colab.diarize import diarize
from numpy import ndarray

def test_diarize(audio: ndarray):
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Hugging face token required. Set the token to env var 'HF_TOKEN'")
    diarized_segments = diarize(audio, hf_token)
    assert len(diarized_segments) > 1

