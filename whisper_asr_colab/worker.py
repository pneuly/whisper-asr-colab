import time
import gc
import sys
import logging
from torch.cuda import empty_cache
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union, Iterable, Tuple, List, DefaultDict, Any
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel as FasterWhisperModel
from .docx_generator import DocxGenerator
from .audio import Audio
from .utils import download_from_colab, str2seconds
from .asr import faster_whisper_transcribe, realtime_transcribe
from .diarize import DiarizationPipeline
from .speakersegment import SpeakerSegment, assign_speakers, combine, combine_same_speakers, write_result


@dataclass
class Worker:
    # core parameters
    audio: Audio
    model_size: str = "large-v3-turbo"
    device: str = "auto"

    # transcribe options
    model: Optional[FasterWhisperModel] = None
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
        if self.skip_silence:
            self.audio.set_silence_skip()


    def transcribe(self) -> List[SpeakerSegment]:
        """Wrapper for faster-whisper transcription.
        Automatically sets the inference if not explicitly specified.
        Switches between normal transcription and real-time transcription based on the value of `self.realtime`.
        """
        # Transcribe
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )
        if self.realtime: # realtime trascription
            self.logger.info("Transcribing on the fly...")
            segments = realtime_transcribe(
                process=self.audio.live_stream,
                model=self.model,
                language = self.language,
                initial_prompt = self.initial_prompt
            )
        else:  # normal transcription
            segments, _ = self.call_faster_whisper_transcribe()
        self.asr_segments = segments
        return self.asr_segments


    def transcribe_segmented(self) -> List[SpeakerSegment]:
        """Transcribe each diarized segment separately. Called by run2()"""
        self.logger.info("Transcribing each segment...")
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )

        def _transcribe_and_add_speakers(speakerseg: SpeakerSegment) -> List[SpeakerSegment]:
            _asr_result = []
            segments, _ = self.call_faster_whisper_transcribe(
                start_time=speakerseg.start,
                end_time=speakerseg.end)
            if segments:
                segment = combine(segments)
                segment.speaker = speakerseg.speaker
                _asr_result.append(segment)
            return _asr_result

        asr_result = []
        with ThreadPoolExecutor() as executor:
            if self.diarized_segments is not None:
                combine_same_speakers(self.diarized_segments)
                procs = []
                for seg in self.diarized_segments:
                    procs.append(executor.submit(_transcribe_and_add_speakers, seg))
                executor.shutdown(wait=True)
                for proc in procs:
                    result = proc.result()
                    if result:
                        asr_result.extend(result)
        self.asr_segments = asr_result
        return self.asr_segments


    def call_faster_whisper_transcribe(
            self,
            start_time: Union[int, float, None] = None,
            end_time: Union[int, float, None] = None,) -> Tuple[List[SpeakerSegment], Any]:
        """Commonly used by `transcribe()` and `transcribe_segmented().`
        """
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


    def extract_future(self):
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )
        feature = self.model.feature_extractor(self.audio.ndarray)
        self.model.feature_extractor.__call__ = lambda waveform, padding=160, chunk_length=None: feature


    def diarize(self, show_progress=True) -> List[SpeakerSegment]:
        self.logger.debug("Diarizing.")
        if self.audio.ndarray is None:
            raise ValueError("Audio must be specified in worker.audio.")
        dpipe = DiarizationPipeline(use_auth_token=self.hugging_face_token)
        segments = dpipe(
            audio=self.audio.ndarray,
            show_progress=show_progress,
        )
        if segments and self.audio.start_time:
            for item in segments:
                item.shift_time(self.audio.start_time)
        self.diarized_segments = segments
        return self.diarized_segments

    def _write_result(self, speaker_segments, with_speakers=False):
        if isinstance(self.audio.file_path, str):
            outfilename = self.audio.file_path
        else:
            outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
        return write_result(
            speaker_segments,
            outfilename,
            with_speakers,
            self.timestamp_offset if self.timestamp_offset else 0.0
        )

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
        files_to_download = []
        self.run_and_measure(self.transcribe)
        outfiles = self.run_and_measure(self._write_result, self.asr_segments)
        files_to_download.extend(outfiles)

        del self.model
        empty_cache()
        gc.collect()

        if self.diarization:
            self.run_and_measure(self.diarize)
            if not self.asr_segments:
                raise ValueError("self.asr_segments is empty.")
            if not self.diarized_segments:
                raise ValueError("self.diarized_segments is empty.")
            self.diarized_segments = assign_speakers(
                asr_segments=self.asr_segments,
                diarization_result=self.diarized_segments,
                postprocesses=(combine_same_speakers,),
            )

            print("Writing diarization result.")
            diarized_txt = self.run_and_measure(
                self._write_result, self.diarized_segments, with_speakers=True)[0]
            files_to_download.append(diarized_txt)

            print("Writing to docx.")
            doc = DocxGenerator()
            doc.txt_to_word(diarized_txt)
            print(f"Downloading {doc.docfilename}")
            download_from_colab(doc.docfilename)
        # DL audio file
        if self.audio.url:
            files_to_download.append(self.audio.file_path)

        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)


    def run2(self):
        """Similar to run(), but diarize first and ASR for each diarized segment"""
        files_to_download = []
        print("Diarizing.")
        self.diarize()
        print("Transcribing.")
        self.transcribe_segmented()

        print("Writing ASR result.")
        outfiles = self._write_result(self.asr_segments)
        files_to_download.extend(outfiles)

        print("Writing diarizing result.")
        diarized_txt = self._write_result(self.asr_segments, with_speakers=True)[0]
        files_to_download.append(diarized_txt)

        print("Writing to docx.")
        doc = DocxGenerator()
        doc.txt_to_word(diarized_txt)
        print(f"Downloading {doc.docfilename}")
        download_from_colab(doc.docfilename)

        # DL audio file
        if not self.audio.url:
            files_to_download.append(self.audio.file_path)

        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)


    def run_parallel(self):
        """Trascribing and diarizing in parallel do not improve performace
        because they confrict GPU usage resulting in worse than sequential mode.
        So, only feature extraction (Using only CPU) and diarization are
        executed in parallel.
        """
        files_to_download = []

        print("Feature extraction and diarizing in parallel.", flush=True)
        with ThreadPoolExecutor() as executor:
            proc_diarize = executor.submit(self.run_and_measure, self.diarize)
            self.run_and_measure(self.extract_future)
            executor.shutdown(wait=True)
        empty_cache()
        self.run_and_measure(self.transcribe)
        if self.asr_segments is None:
            raise ValueError("self.asr_segments is None. Run self.transcribe.")
        else:
            self.diarized_segments = assign_speakers(
                asr_segments=self.asr_segments,
                diarization_result=proc_diarize.result(),
                postprocesses=(combine_same_speakers,),
            )

        print("Writing results.")
        outfiles = self._write_result(self.asr_segments)
        files_to_download.extend(outfiles)
        diarized_txt = self._write_result(self.diarized_segments, with_speakers=True)[0]
        files_to_download.append(diarized_txt)

        print("Writing to docx.")
        doc = DocxGenerator()
        doc.txt_to_word(diarized_txt)
        files_to_download.append(doc.docfilename)

        # Add audio file to files to download if needed
        if self.audio.url:
            files_to_download.append(self.audio.file_path)
        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)
