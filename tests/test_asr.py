from whisper_asr_colab.audio import Audio
from whisper_asr_colab.asr import ASRWorker

def perform_asr(audio:str, batch_size:int):
    _audio = Audio(audio)
    _audio.verify_upload = False
    worker = ASRWorker(
        audio=_audio,
        load_model_args={"model_size_or_path" : "tiny",},
        transcribe_args={"batch_size" : batch_size,}
    )
    
    worker.run()
    segments = worker.asr_segments
    for seg in segments:
        print(seg.text)
    assert len(segments) > 1
    assert segments[0].text != ""
    return (segments,)

def test_asr_squential(audio):
    perform_asr(audio, 1)

def test_asr_batched(audio):
    perform_asr(audio, 4)




