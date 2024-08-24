import time
import datetime
import logging
import torch
import numpy as np
from typing import Union
from whisperx import load_audio, load_model
from whisperx.asr import WhisperModel as WhisperXModel
from faster_whisper import WhisperModel as FasterWhisperModel
from .audio import open_stream

class WhisperXModelSequential(WhisperXModel):
    def generate_segment_batched(self, features: np.ndarray, tokenizer, options, encoder_output = None):
        # copied from original code
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )
        # avoid batching
        encoder_output = self.encode(features)
        (result,
            avg_logprob,
            temperature,
            compression_ratio,
        ) = self.generate_with_fallback(
            encoder_output,
            prompt,
            tokenizer,
            options,
        )
        tokens = result.sequences_ids[0]
        text = tokenizer.decode(tokens)
        return [text]

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
    model = None
    logging.debug(f"batich_size: {batch_size}")
    if batch_size < 2:
        logging.info(f"batch_size is set to less than 2. ({batch_size}). Switching to sequential mode.")
        model = WhisperXModelSequential(model_size, device)
    pipeline = load_model(
        whisper_arch=model_size,
        device=device,
        asr_options={"initial_prompt" : initial_prompt},
        language=language,
        model=model,
    )
    audio = load_audio(audio)
    result = pipeline.transcribe(
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
        device: str = "",
        language: str = "ja",
        initial_prompt: str = "",
    ):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FasterWhisperModel(
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
    model = FasterWhisperModel(
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
