import time
import sys
import datetime
from logging import getLogger, DEBUG
from subprocess import Popen
from typing import Union, Optional, Iterable, TextIO, BinaryIO, List, Any, ParamSpec
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel as FasterWhisperModel
import ipywidgets as widgets
from whisper_asr_colab.common.speakersegment import SpeakerSegment
from whisper_asr_colab.common.utils import format_timestamp

logger = getLogger(__name__)

_Type_Prompt = Optional[Union[str, Iterable[int]]]
P = ParamSpec('P')

def faster_whisper_transcribe(
        audio: Union[str, BinaryIO, np.ndarray],
        model: Optional[FasterWhisperModel] = None,
        **transcribe_args: P.kwargs
) -> tuple[List[SpeakerSegment], Any]:
    
    transcribe_args["batch_size"] = transcribe_args["batch_size"] or 1
    batch_size = transcribe_args["batch_size"]
    logger.debug(f"batch_size: {batch_size}")

    model = model or FasterWhisperModel(
                        "large-v3-turbo",
                        device="auto",
                        compute_type="default",
                        )

    if batch_size > 1: # batch mode
        batched_model = BatchedInferencePipeline(model=model)
        segments_generator, info = batched_model.transcribe(
            audio=audio,
            **transcribe_args
        )
    else: # sequential mode
        logger.info(f"batch_size is set to less than 2 (batch_size={batch_size}). Using sequential mode.")
        segments_generator, info = model.transcribe(
            audio=audio,
            condition_on_previous_text=False,
            without_timestamps=False,
            **{k: v for k, v in transcribe_args.items() if k not in {"model", "batch_size"}},
        )
    segments = []
    with open("diarization_progress.txt", "w", encoding="utf-8", buffering=1) as f:
        for segment in segments_generator:
            segment_text = f"[{format_timestamp(segment.start, '02.0f')} - {format_timestamp(segment.end, '02.0f')}] {segment.text}"
            #print(segment_text)
            f.write(segment_text + "\n")
            segments.append(SpeakerSegment.from_segment(segment))
    if logger.isEnabledFor(DEBUG):
        logger.debug(f"Transcribed segments:\n{segments}")
    return segments, info

def realtime_transcribe(
        process: Popen, # streaming process
        model: Optional[FasterWhisperModel] = None,
        language: Optional[str] = None,
        initial_prompt: _Type_Prompt = None,
    ) -> List[SpeakerSegment]:
    ## TODO: Improve real-time transcription quality.
    ## The current code handles the audio every 30 seconds, which harms transcription quality.
    ## Possible improvements:
    ## - Read audio stream and store into ndarray
    ## - Read audio chunk from ndarray and input the data into WhisperModel.generate_segments
    ## - The remaining data from the previous audio chunk should be carried over to the next chunk
    if model is None:
        model = FasterWhisperModel("large-v3-turbo")
    buffer = b""
    fh1 = open(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
        "w",
        encoding="utf-8"
    )

    stop_transcribing = False
    def _stop_button_clicked(b):
        logger.warning(f"Stop button is clicked. {b}")
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
            print(f"[{segment.start} - {segment.end}]{segment.text}")
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