import os
from whisper_asr_colab.audio import Audio
from whisper_asr_colab.diarize import DiarizationPipeline

def test_diarize(audio: str):
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Hugging face token required. Set the token to env var 'HF_TOKEN'")
    dpipe = DiarizationPipeline(use_auth_token=hf_token)
    diarized_segments = dpipe(
        audio=Audio.from_path_or_url(audio).ndarray
    )
    assert len(diarized_segments) > 1

