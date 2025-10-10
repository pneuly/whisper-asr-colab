import time
import sys
import datetime
from logging import getLogger
from subprocess import Popen
from typing import Union, Optional, Iterable, TextIO, ParamSpec
import numpy as np
from faster_whisper import WhisperModel as FasterWhisperModel
import ipywidgets as widgets
from whisper_asr_colab.common.speakersegment import SpeakerSegment
from whisper_asr_colab.common.speakersegmentlist import SpeakerSegmentList

logger = getLogger(__name__)

_Type_Prompt = Optional[Union[str, Iterable[int]]]
P = ParamSpec('P')

def realtime_transcribe(
        process: Popen, # streaming process
        model: Optional[FasterWhisperModel] = None,
        language: Optional[str] = None,
        initial_prompt: _Type_Prompt = None,
    ) -> SpeakerSegmentList:
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
    return [SpeakerSegment(segment=segment) for segment in segments]