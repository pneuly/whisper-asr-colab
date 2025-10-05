from whisper_asr_colab.asr.asr import faster_whisper_transcribe
from whisper_asr_colab.asr.audio import Audio
from faster_whisper import WhisperModel as FasterWhisperModel

def perform_asr(audio:str, batch_size:int):
    model = FasterWhisperModel("tiny")
    segments, info = faster_whisper_transcribe(
        model=model,
        audio=Audio.from_path_or_url(audio).ndarray,
        batch_size=batch_size,
    )
    for segment in segments:
        print(segment.text)
    assert info.all_language_probs is not None
    assert info.language == "ja"
    assert info.language_probability > 0.9
    assert len(segments) > 1
    assert segments[0].text != ""
    return (segments, info,)

def test_asr_squential(audio):
    perform_asr(audio, 1)

def test_asr_batched(audio):
    perform_asr(audio, 4)
