[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/pneuly/whisper-asr-colab/blob/main/README.md)
[![ja](https://img.shields.io/badge/lang-ja-blue.svg)](https://github.com/pneuly/whisper-asr-colab/blob/main/README_ja.md)

# Aggregation Package for Transcription and Diarization
**Whisper-asr-colab** is an aggregation package for speech-to-text and diarization, featuring an example implementation on Google Colab.

The main functions of this package are as follows:
* **Speech-to-text (transcription)**, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* **Diarization**, powered by [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* **Online audio downloading**, powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* **Writing diarization results in docx format**, powered by [python-docx](https://github.com/python-openxml/python-docx)

## Usage
Open [whisper_asr_colab.ipynb](https://github.com/pneuly/whisper-asr-colab/blob/main/whisper_asr_colab.ipynb) on Google Colab or use the modules as shown below.
```python
from whisper_asr_colab.worker import Worker
from whisper_asr_colab.audio import Audio

audio = "audiofile.m4a"
model_size = "turbo"
hf_token = "your hf token"

worker = Worker(
    audio=Audio.from_path_or_url(audio),
    model_size=model_size,
    hugging_face_token=hf_token,
)

worker.run()
```
