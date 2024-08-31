import time
import datetime
from logging import getLogger
from numpy import ndarray, frombuffer as np_frombuffer, int16 as np_int16, float32 as np_float32
from typing import Union, Optional, Iterable
from faster_whisper import BatchedInferencePipeline, WhisperModel as FasterWhisperModel
from .audio import load_audio, open_stream

logger = getLogger(__name__)

def faster_whisper_transcribe(
    # model options
    model_size: str = "large-v3",
    device: str = "auto",

    # transcribe options
    audio: Union[str, ndarray] = "",
    language: Optional[str] = None,
    multilingual: bool = False,
    initial_prompt: Optional[Union[str, Iterable[int]]] = None,
    hotwords: Optional[str] = None,
    chunk_length: int = 30,
    batch_size: int = 16,
    prefix: Optional[str] = None,
    vad_filter: bool = True,
    log_progress: bool = False,

    # other options
    realtime: bool = False,
    ):
    logger.debug(f"batich_size: {batch_size}")
    model = FasterWhisperModel(
        model_size,
        device=device,
        compute_type="default",
    )
    if batch_size > 1: # batch mode
        batched_model = BatchedInferencePipeline(model=model)
        # load_audio prevents excessive memory use by avoiding loading large audio file all at once.
        if isinstance(audio, str):
            audio = load_audio(audio)
        segments_generator, info = batched_model.transcribe(
            audio=audio,
            language=language,
            #multilingual=multilingual,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            prefix=prefix,
            #chunk_length = chunk_length,  #  not implemented
            batch_size=batch_size,
            log_progress=log_progress,
        )
    else: # sequential mode
        logger.info(f"batch_size is set to less than 2. ({batch_size}). Using equential mode.")
        if isinstance(audio, str):
            audio = load_audio(audio)
        segments_generator, info = model.transcribe(
            audio=audio,
            language=language,
            multilingual=multilingual,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            prefix=prefix,
            condition_on_previous_text=False, # supress hallucination and repetitive text
            without_timestamps=False,
        )
    segments = []
    for segment in segments_generator:
        print(segment.text)
        segments.append(segment)
    logger.info(f"Transcribed segments:\n{segments}")
    return segments, info

def realtime_transcribe(
        url: str,
        model_size: str = "medium",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ):
    model = FasterWhisperModel(model_size)
    process = open_stream(url)
    buffer = b""
    fh1 = open(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
        "w",
        encoding="utf-8"
    )

    def _realtime_asr_loop(model, data, outfh, initial_prompt=None):
        segments, _ =  model.transcribe(
            audio=np_frombuffer(data, np_int16).astype(np_float32) / 32768.0,
            language=language,
            initial_prompt=initial_prompt
            )
        for segment in segments:
            print(segment.text)
            outfh.write(segment.text + "\n")
            outfh.flush()
            previous_text = segment.text
        return previous_text

    while True:
        audio_data = process.stdout.read(16000 * 2)
        if process.poll() is not None:
            _realtime_asr_loop(model, buffer, fh1)
            break

        buffer += audio_data
        if len(buffer) >= 16000 * 2 * 30:  # 30 seconds
            _realtime_asr_loop(model, buffer, fh1)
            buffer = buffer[- 16000:]  # 0.5 seconds overlap
        else:
            time.sleep(0.1)
    fh1.close()
