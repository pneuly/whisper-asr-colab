import os
from whisper_asr_colab.diarize import diarize

def test_diarize(audio):
    hf_token = os.getenv('HF_TOKEN')
    diarized_segments = diarize(audio, hf_token)
    assert len(diarized_segments) > 1

