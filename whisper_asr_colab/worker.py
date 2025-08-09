import time
import sys
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union, Iterable, Tuple, List, DefaultDict, Any
import ffmpeg
from .docx_generator import DocxGenerator
from .audio import Audio
from .utils import str2seconds
from .speakersegment import (
    SpeakerSegment,
    assign_speakers,
    combine_same_speakers,
    write_result,
    save_segments,
    load_segments,
)

def _write_result(speaker_segments, audio_filepath=None, timestamp_offset=0.0, with_speakers=False):
    if audio_filepath:
        outfilename = audio_filepath
    else:
        outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
    return write_result(
        speaker_segments,
        outfilename,
        with_speakers,
        timestamp_offset,
    )

@dataclass
class Worker:
    # core parameters
    audio: Audio
    model_size: str = "large-v3-turbo"
    device: str = "auto"

    # transcribe options
    model = None
    language: Optional[str] = None
    multilingual: bool = False
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    hotwords: Optional[str] = None
    chunk_length: int = 30
    batch_size: int = 16
    prefix: Optional[str] = None
    vad_filter: bool = False
    log_progress: bool = False

    # other options
    diarization: bool = True
    hugging_face_token: str = ""
    password: str = ""
    realtime: bool = False
    skip_silence: bool = True  # If True, skip the leading silence of the audio
    _timestamp_offset: float = 0.0

    #result data
    asr_segments: Optional[List[SpeakerSegment]] = None  # result from whisper
    diarized_segments: Optional[List[SpeakerSegment]] = None  # result from pyannote
    elapsed_time: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def timestamp_offset(self) -> float:
        return self._timestamp_offset

    @timestamp_offset.setter
    def timestamp_offset(self, sec : Union[str, int, float]):
        if isinstance(sec, str):
            sec = str2seconds(sec)
        self._timestamp_offset = sec

    def __post_init__(self):

        self.logger = logging.getLogger(__name__)
        if isinstance(self.audio, str):
            self.audio = Audio.from_path_or_url(self.audio)
        if self.password:
            self.audio.password = self.password
        if self.realtime:
            self.logger.info("`skip_silence` is disabled since `realtime` mode is enabled.")
            self.skip_silence = False
        # Use model loading as waiting function since using time.sleep is just waste of time.
        # set_silence_skip() calls audio._load_audio(), so upload_wait_func must be set before
        # calling set_silence_skip().
        #self.audio.upload_wait_func = self.load_model
        self.audio.upload_wait_func = lambda : time.sleep(5)
        if self.skip_silence:
            self.audio.set_silence_skip()

    def load_model(self):
        from .asr import load_model as _load_model
        self.model = _load_model(
            self.model_size,
            device=self.device,
            compute_type="default",
        )

    def transcribe(self) -> List[SpeakerSegment]:
        """Wrapper for faster-whisper transcription.
        Automatically sets the inference if not explicitly specified.
        Switches between normal transcription and real-time transcription based on the value of `self.realtime`.
        """
        # Transcribe
        if self.model is None:
            self.load_model()
        if self.realtime: # realtime trascription
            self.logger.error("Real time transcription is temporarily disabled.")
        else:  # normal transcription
            segments, _ = self.call_faster_whisper_transcribe()
        self.asr_segments = segments
        return self.asr_segments


    def call_faster_whisper_transcribe(
            self,
            start_time: Union[int, float, None] = None,
            end_time: Union[int, float, None] = None,) -> Tuple[List[SpeakerSegment], Any]:
        """Used by `transcribe()` to call faster-whisper transcribe function."""
        from .asr import faster_whisper_transcribe
        if self.audio.ndarray is None:
            raise ValueError("Audio must be set in worker.audio.")
        _start = start_time if start_time else self.audio.start_time
        _end = end_time if end_time else self.audio.end_time
        self.logger.info(f"Transcribing from {_start} to {_end}")
        segments, _ = faster_whisper_transcribe(
                audio=self.audio.get_time_slice(_start, _end),
                model=self.model,
                language = self.language,
                multilingual=self.multilingual,
                initial_prompt=self.initial_prompt,
                hotwords = self.hotwords,
                prefix = self.prefix,
                vad_filter=self.vad_filter,
                #chunk_length=self.chunk_length,
                batch_size=self.batch_size,
            )
        if segments and _start:
            for item in segments:
                item.shift_time(_start)
        return segments, _


    #def extract_future(self):
    #    if self.model is None:
    #        self.load_model()
    #    #if isinstance(self.model, FasterWhisperModel):
    #    #    feature = self.model.feature_extractor(self.audio.ndarray)
    #    #    self.model.feature_extractor.__call__ = lambda waveform, padding=160, chunk_length=None: feature
    #    #else:
    #    #    raise ValueError("Model is not loaded.")

    # TODO run upload/download processes (audio, model and others) simultaneously
    # TODO minimize log output to screen
    # TODO calc execution time

    def run_and_measure(self, func, *args, **kargs):
        func_name = func.__name__
        self.logger.info(f"Excecuting {func_name}.")
        start = time.time()
        result = func(*args, **kargs)
        end = time.time()
        elapsed = end - start
        self.elapsed_time[func_name] =elapsed
        sys.stdout.flush()
        self.logger.info(f"Executed {func_name} in {elapsed:.2f}s")
        return result

    def run(self):
        """Wrapper for ASR and diarization"""
        # Isolate the ASR process from diarization process
        # because Pipeline of pyannote.audio crashes if faster whisper is called beforehand.
        self.transcribe()
        outfiles = _write_result(self.asr_segments, self.audio.file_path)
        del self.model

        print("Saving ASR result as json file.")
        save_segments(self.asr_segments, "asr_result.json")
        return outfiles


@dataclass
class Diarizer:
    audio: Audio
    # other options
    hugging_face_token: str = ""
    diarized_segments: Optional[List[SpeakerSegment]] = None
    asr_segments: Optional[List[SpeakerSegment]] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    def diarize(self, show_progress=True): # -> List[SpeakerSegment]:
        self.logger.debug("Diarizing.")
        from .diarize import DiarizationPipeline
        if self.audio.ndarray is None:
            raise ValueError("Audio must be specified in diarizer.audio.")
        dpipe = DiarizationPipeline(use_auth_token=self.hugging_face_token) ## crashes here
        print("dpipe is ready.", flush=True)
        segments = dpipe(
            audio=self.audio.ndarray,
            show_progress=show_progress,
        )
        print("diarization finished.", flush=True)
        if segments and self.audio.start_time:
            for item in segments:
                item.shift_time(self.audio.start_time)
        self.diarized_segments = segments
        return self.diarized_segments

    def integrate(self):
        if not self.asr_segments:
            self.asr_segments = load_segments("asr_result.json")
            #raise ValueError("self.asr_segments is empty.")
        if not self.diarized_segments:
            raise ValueError("self.diarized_segments is empty.")
        self.diarized_segments = assign_speakers(
            asr_segments=self.asr_segments,
            diarization_result=self.diarized_segments,
            postprocesses=(combine_same_speakers,),
        )

        result_files = []
        print("Writing diarization result.")
        diarized_txt = _write_result(
            speaker_segments=self.diarized_segments,
            audio_filepath=self.audio.file_path,
            with_speakers=True)[0]
        result_files.append(diarized_txt)

        print("Writing to docx.")
        doc = DocxGenerator()
        doc.txt_to_word(diarized_txt)
        result_files.append(doc.docfilename)

        return result_files
