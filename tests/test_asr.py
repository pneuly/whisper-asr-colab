import os
from whisper_asr_colab.asr import faster_whisper_transcribe

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
audio = os.path.join(data_dir, "asr_test.m4a")

def perform_asr(audio, batch_size):
    segments, info = faster_whisper_transcribe(
        model_size="tiny",
        audio=audio,
        batch_size=1,
    )
    segments = list(segments)
    assert info.all_language_probs is not None
    assert info.language == "ja"
    assert info.language_probability > 0.9
    assert len(segments) > 1
    assert segments[0].text != ""
    return (segments, info,)

def test_asr_squential(audio):
    segments, info = perform_asr(audio, 1)

def test_asr_batched(audio):
    segments, info = perform_asr(audio, 4)