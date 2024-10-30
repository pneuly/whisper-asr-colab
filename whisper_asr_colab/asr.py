import time
import datetime
from logging import getLogger
from numpy import ndarray, frombuffer as np_frombuffer, int16 as np_int16, float32 as np_float32
from typing import Union, Optional, Iterable, TextIO, Any
from faster_whisper import BatchedInferencePipeline, WhisperModel as FasterWhisperModel
from IPython.display import display
import ipywidgets as widgets
from .audio import load_audio, open_stream
from .speakersegment import SpeakerSegment, SpeakerSegmentList

logger = getLogger(__name__)

def faster_whisper_transcribe(
    # model options
    model: Optional[FasterWhisperModel] = None,

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
    ) -> tuple[SpeakerSegmentList, Any]:

    logger.debug(f"batich_size: {batch_size}")
    if model is None:
        model = FasterWhisperModel(
            "large-v3-turbo",
            device="auto",
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
    segments = SpeakerSegmentList()
    for segment in segments_generator:
        print(segment.text)
        segments.append(SpeakerSegment.from_segment(segment))
    logger.info(f"Transcribed segments:\n{segments}")
    return segments, info

def realtime_transcribe(
        url: str,
        model: Optional[FasterWhisperModel] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> SpeakerSegmentList:
    segments = []
    if model is None:
        model = FasterWhisperModel("large-v3-turbo")
    process = open_stream(url)
    buffer = b""
    fh1 = open(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
        "w",
        encoding="utf-8"
    )
    stop_button = widgets.Button(
        description="Stop Transcribing",
        style={'font_weight': 'bold'},
    )
    stop_transcribing = False

    def _stop_button_clicked(b):
        print(f"Stop button is clicked. {b}")
        nonlocal stop_transcribing
        stop_transcribing = True

    display(stop_button)
    stop_button.on_click(_stop_button_clicked)

    def _realtime_asr_loop(
        model,
        data: bytes,
        outfh: TextIO,
        initial_prompt: Optional[str] = None
        ) -> SpeakerSegmentList:
        segments, _ =  model.transcribe(
            audio=np_frombuffer(data, np_int16).astype(np_float32) / 32768.0,
            language=language,
            initial_prompt=initial_prompt
            )
        for segment in segments:
            print(segment.text)
            outfh.write(segment.text + "\n")
            outfh.flush()
        return segments

    while not stop_transcribing:
        audio_data = process.stdout.read(16000 * 2)
        if process.poll() is not None:
            segments += _realtime_asr_loop(model, buffer, fh1)
            break

        buffer += audio_data
        if len(buffer) >= 16000 * 2 * 30:  # 30 seconds
            segments += _realtime_asr_loop(model, buffer, fh1)
            buffer = buffer[- 16000:]  # 0.5 seconds overlap
        else:
            time.sleep(0.1)
    fh1.close()
    return SpeakerSegmentList(*[SpeakerSegment.from_segment(segment) for segment in segments])
