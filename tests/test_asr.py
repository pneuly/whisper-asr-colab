from whisper_asr_colab.asr import faster_whisper_transcribe

def perform_asr(audio:str, batch_size:int):
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