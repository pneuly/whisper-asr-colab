import os
from whisper_asr_colab.audio import Audio
from whisper_asr_colab.diarize import DiarizationWorker


def test_diarize(audio: str):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging face token required. Set the token to env var 'HF_TOKEN'"
        )
    worker = DiarizationWorker(audio=Audio(audio), hugging_face_token=hf_token)
    worker.run()
    assert len(worker.diarized_segments) > 1
