import os
from whisper_asr_colab.diarize import diarize
from whisper_asr_colab.audio import Audio

def test_diarize(audio: str):
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Hugging face token required. Set the token to env var 'HF_TOKEN'")
    diarized_segments = diarize(Audio.from_path_or_url(audio).ndarray, hf_token)
    assert len(diarized_segments) > 1

