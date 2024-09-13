# Faster Whisper Imprementation on Google Colab
**whisper-asr-colab** is an implementation of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on Google Colab.
It also provides memory-efficient audio loading and diarization, features ported from [WhisperX](https://github.com/m-bain/whisperX) .

## Usage
Open [whisper_asr_colab.ipynb](https://github.com/pneuly/whisper-asr-colab/blob/main/whisper_asr_colab.ipynb) on Google Colab or use the modules as shown below.
```python
from whisper_asr_colab.worker import Worker

audio = "audiofile.m4a"
model_size = "medium"
hf_token = "your hf token"

worker = Worker(
    audio=audio,
    model_size=model_size,
    hugging_face_token=hf_token,
)

worker.run()
```
