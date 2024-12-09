import time
import sys
import datetime
from logging import getLogger
from subprocess import Popen
from typing import Union, Optional, Iterable, TextIO, BinaryIO, List, Any
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel as FasterWhisperModel
import ipywidgets as widgets
from .speakersegment import SpeakerSegment


logger = getLogger(__name__)

_Type_Prompt = Optional[Union[str, Iterable[int]]]

# TODO provide AsrOptions class

def faster_whisper_transcribe(
    audio: Union[str, BinaryIO, np.ndarray],
    model: Optional[FasterWhisperModel] = None,
    language: Optional[str] = None,
    multilingual: bool = False,
    initial_prompt: _Type_Prompt = None,
    hotwords: Optional[str] = None,
    chunk_length: int = 30,
    batch_size: int = 16,
    prefix: Optional[str] = None,
    vad_filter: bool = False,
    log_progress: bool = False,
    ) -> tuple[List[SpeakerSegment], Any]:

    logger.debug(f"batich_size: {batch_size}")
    if model is None:
        model = FasterWhisperModel(
            "large-v3-turbo",
            device="auto",
            compute_type="default",
    )
    if batch_size > 1: # batch mode
        batched_model = BatchedInferencePipeline(model=model)
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
        segments.append(SpeakerSegment.from_segment(segment))
    logger.info(f"Transcribed segments:\n{segments}")
    return segments, info

def realtime_transcribe(
        process: Popen, # streaming process
        model: Optional[FasterWhisperModel] = None,
        language: Optional[str] = None,
        initial_prompt: _Type_Prompt = None,
    ) -> List[SpeakerSegment]:
    ## TODO  make choppy transcription continuous one
    if model is None:
        model = FasterWhisperModel("large-v3-turbo")
    buffer = b""
    fh1 = open(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
        "w",
        encoding="utf-8"
    )

    stop_transcribing = False
    # TODO Use logger instead of print
    def _stop_button_clicked(b):
        print(f"Stop button is clicked. {b}")
        nonlocal stop_transcribing
        stop_transcribing = True

    if 'IPython' in sys.modules:
        display = sys.modules["IPython"].display
        stop_button = widgets.Button(
            description="Stop Transcribing",
            style={'font_weight': 'bold'},
        )
        display.display(stop_button)
        stop_button.on_click(_stop_button_clicked)

    def _realtime_asr_loop(
        model: FasterWhisperModel,
        data: bytes,
        outfh: TextIO,
        initial_prompt: _Type_Prompt = None
        ) -> Iterable:
        segments, _ =  model.transcribe(
            audio=np.frombuffer(data, np.int16).astype(np.float32) / 32768.0,
            language=language,
            initial_prompt=initial_prompt
            )
        for segment in segments:
            print(segment.text)
            outfh.write(segment.text + "\n")
            outfh.flush()
        return segments

    segments = []
    while not stop_transcribing:
        if process.stdout is not None:
            audio_data = process.stdout.read(16000 * 2)
        else:
            raise ValueError("process.stdout is None. Check your subprocess initialization.")
        if process.poll() is not None:
            segments += _realtime_asr_loop(model, buffer, fh1, initial_prompt)
            break

        buffer += audio_data
        if len(buffer) >= 16000 * 2 * 30:  # 30 seconds
            segments += _realtime_asr_loop(model, buffer, fh1)
            buffer = buffer[- 16000:]  # 0.5 seconds overlap
        else:
            time.sleep(0.1)
    fh1.close()
    return [SpeakerSegment.from_segment(segment) for segment in segments]
