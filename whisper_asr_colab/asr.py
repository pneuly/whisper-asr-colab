import time
import datetime
import torch
import numpy as np
from typing import Union
import whisperx
from faster_whisper import WhisperModel
from .audio import open_stream

def whisperx_transcribe(
        audio: Union[str, np.ndarray],
        chunk_size: int = 20,
        batch_size: int = 16,
        model_size: str = "medium",
        device: str = "",
        language: str = "ja",
        initial_prompt: str = ""
        ):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(
        whisper_arch=model_size,
        device=device,
        compute_type="default",
        asr_options={"initial_prompt" : initial_prompt},
        language=language,
    )
    audio = whisperx.load_audio(audio)
    result = model.transcribe(
        audio,
        language=language,
        print_progress=True,
        chunk_size=chunk_size,
        batch_size=batch_size,
        )
    return result


def faster_whisper_transcribe(
        audio: Union[str, np.ndarray],
        model_size: str = "medium",
        language: str = "ja",
        initial_prompt: str = "",
    ):
    model = WhisperModel(
        model_size, compute_type="default"  # default: equivalent, auto: fastest
    )
    segments = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        initial_prompt=initial_prompt,
        without_timestamps=False,
    )[0]
    segments = [seg._asdict() for seg in segments]
    return segments


def realtime_transcribe(
        url: str,
        model_size: str = "medium",
    ):
    model = WhisperModel(
        model_size, compute_type="default"  # default: equivalent, auto: fastest
    )
    process = open_stream(url)
    previous_text = ""
    buffer = b""
    fh1 = open(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
        "w",
        encoding="utf-8"
    )
    while True:
        audio_data = process.stdout.read(16000 * 2)
        if process.poll() is not None:
            _realtime_asr_loop(model, buffer, fh1, previous_text)
            break

        buffer += audio_data
        if len(buffer) >= 16000 * 2 * 30:  # 30 seconds
            #print(len(buffer))
            _realtime_asr_loop(model, buffer, fh1, previous_text)
            previous_text += "です。 ます。"
            buffer = buffer[- 16000:]  # 0.5 seconds overlap
        else:
            time.sleep(0.1)
    fh1.close()


def _realtime_asr_loop(model, data, outfh, initial_prompt=""):
    previous_text = ""
    segments, info =  model.transcribe(
        np.frombuffer(data, np.int16).astype(np.float32) / 32768.0,
        language='ja',
        #vad_filter=True,
        initial_prompt=initial_prompt
        )
    for segment in segments:
        print(segment.text)
        outfh.write(segment.text + "\n")
        outfh.flush()
        previous_text = segment.text
    return previous_text
