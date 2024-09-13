import os
from whisper_asr_colab.diarize import diarize
from whisper_asr_colab.utils import load_segments

def test_diarize(audio, asr_segments_file):
    hf_token = os.getenv('HF_TOKEN')
    asr_segments = load_segments(asr_segments_file)
    diarized_segments = diarize(audio, asr_segments, hf_token)
    assert len(diarized_segments) > 1
    assert diarized_segments[0].text != ""
