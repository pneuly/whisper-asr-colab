[![ja](https://img.shields.io/badge/lang-ja-blue.svg)](README_ja.md)
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)

# 文字起こし・話者分離の統合パッケージ
**Whisper-asr-colab**は、音声からテキストへの変換と話者分離のための統合パッケージで、Google Colabでの実装例が含まれています。

このパッケージの主な機能は以下の通りです：
* **文字起こし**：[faster-whisper](https://github.com/SYSTRAN/faster-whisper) を使用
* **話者分離**：[pyannote-audio](https://github.com/pyannote/pyannote-audio) を使用
* **音声のダウンロード**：[yt-dlp](https://github.com/yt-dlp/yt-dlp) を使用
* **話者分離結果をdocx形式で書き出し**：[python-docx](https://github.com/python-openxml/python-docx) を使用

## 使用例
Google Colab 上での実装例は、[whisper_asr_colab.ipynb](whisper_asr_colab.ipynb) にあります。
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
