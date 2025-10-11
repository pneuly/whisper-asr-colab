[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![ja](https://img.shields.io/badge/lang-ja-blue.svg)](README_ja.md)

# Aggregation Package for Transcription and Diarization
**Whisper-asr-colab** is an aggregation package for speech-to-text and diarization, featuring an example implementation on Google Colab.

The main functions of this package are as follows:
* **Speech-to-text (transcription)**, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* **Diarization**, powered by [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* **Online audio downloading**, powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* **Writing diarization results in docx format**, powered by [python-docx](https://github.com/python-openxml/python-docx)

## Usage
Open [whisper_asr_colab.ipynb](whisper_asr_colab.ipynb) on Google Colab or use the modules as shown below.
```python
from whisper_asr_colab.audio import Audio
from whisper_asr_colab.asr.asrworker import ASRWorker
from whisper_asr_colab.diarize.diarizationworker import DiarizationWorker
from whisper_asr_colab.docx_generator import DocxGenerator

audio = Audio("audiofile.m4a")
model_size = "turbo"
hf_token = "your hf token"

asrworker = ASRWorker(
    audio=audio,
    model_size=model_size,
)
asrworker.run()

diarizationworker = DiarizationWorker(
    audio=audio,
    hugging_face_token = hf_token,
)
diarizationworker.run()
```
