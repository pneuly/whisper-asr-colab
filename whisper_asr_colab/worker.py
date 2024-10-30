import os
import re
import time
from torch.cuda import empty_cache
from numpy import ndarray
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union, Iterable
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel as FasterWhisperModel
from .docx_generator import DocxGenerator
from .audio import dl_audio, trim_audio, load_audio
from .utils import download_from_colab
from .asr import faster_whisper_transcribe, realtime_transcribe
from .diarize import diarize as _diarize
from .speakersegment import SpeakerSegmentList


@dataclass
class Worker:
    # model options
    model_size: str = "large-v3-turbo"
    device: str = "auto"
    model: Optional[FasterWhisperModel] = None

    # transcribe options
    audio: Union[str, ndarray] = "" # original audio path or data
    language: Optional[str] = None
    multilingual: bool = False
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    hotwords: Optional[str] = None
    chunk_length: int = 30
    batch_size: int = 16
    prefix: Optional[str] = None
    vad_filter: bool = True
    log_progress: bool = False

    # other options
    diarization: bool = True
    hugging_face_token: str = ""
    password: str = ""
    start_time: str = ""
    end_time: str = ""
    timestamp_offset: str = ""
    realtime: bool = False

    #result data
    asr_segments: Optional[SpeakerSegmentList] = None  # result from whisper
    diarized_segments: Optional[SpeakerSegmentList] = None  # result from pyannote

    _input_audio: Union[str, ndarray] = "" # audio input to pass to whisper

    @property
    def input_audio(self) -> Union[str, ndarray]:
        return self._input_audio

    def __post_init__(self):
        # download and trim audio if needed
        if self.audio and not self.realtime:
            self._input_audio = self.audio
            if re.match(r"^(https://).+", self.audio):
                self._input_audio = dl_audio(self.audio, self.password)
            else:
                # If the file size is small, check for incomplete upload.
                filesize = os.path.getsize(self.audio)
                if filesize < 10 ** 7:  # less than 10MB
                    time.sleep(10)
                    filesize2 = os.path.getsize(self.audio)
                    if (filesize2 - filesize) > 0:
                        # File uploading seems incomplete
                        raise IOError("Upload seems incomplete. Run again after the upload is finished.")
            if self.start_time or self.end_time: # trim if specified
                self._input_audio = trim_audio(self._input_audio, self.start_time, self.end_time)


    def transcribe(self, start_time=None, end_time=None)->SpeakerSegmentList:
        # Transcribe
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )
        if self.realtime: # realtime trascription
            segments = realtime_transcribe(
                url = self.audio,
                model=self.model,
                language = self.language,
                initial_prompt = self.initial_prompt
            )
            empty_cache()
            #sys.exit(0)
        else:  # use faster-whisper
            segments, _ = self.call_faster_whisper_transcribe(start_time, end_time)
        return segments


    def transcribe_segmented(self):
        """Transcribe each diarized segment separately"""
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )

        def _transcribe_and_add_speakers(speakerseg):
            asr_result = []
            segments, _ = self.call_faster_whisper_transcribe(
                speakerseg.segment.start,
                speakerseg.segment.end,
            )
            if segments:
                segment = segments.combined
                segment.speaker = speakerseg.speaker
                asr_result.append(segment)
            return asr_result

        with ThreadPoolExecutor() as executor:
            speakersegs = self.diarized_segments.combine_same_speakers
            procs = []
            for seg in speakersegs:
                procs.append(executor.submit(_transcribe_and_add_speakers, seg))
            executor.shutdown(wait=True)
            asr_result = []
            for proc in procs:
                result = proc.result()
                if result:
                    asr_result.extend(result)
        return asr_result


    def call_faster_whisper_transcribe(self, start_time=None, end_time=None):
        segments, _ = faster_whisper_transcribe(
                audio=load_audio(
                    self.input_audio,
                    start_time=start_time,
                    end_time=end_time
                ),
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
        if segments and start_time:
            segments = [
                item.shift_segment_time(start_time)
                for item in segments
            ]
        return segments, _


    def diarize(self):
        return _diarize(
            audio=self.input_audio,
            hugging_face_token=self.hugging_face_token,
        )

    def _write_result(self, with_speakers=False):
        if isinstance(self.input_audio, str):
            outfilename = self.input_audio
        else:
            outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
        segments =  self.diarized_segments if with_speakers else self.asr_segments
        return segments.write_result(
            outfilename,
            with_speakers,
            self.timestamp_offset if self.timestamp_offset else 0.0
        )

    def run(self):
        """ASR first and diarize"""
        files_to_download = []
        print("Transcribing...")
        self.asr_segments = self.transcribe()
        print("Writing result...")
        outfiles = self._write_result(with_speakers=False)
        files_to_download.extend(outfiles)

        print("Diarizing...")
        if self.diarization:
            self.diarized_segments = self.diarize()
            print("Writing result...")
            self.diarized_segments = self.asr_segments.assign_speakers(
                diarization_result=self.diarized_segments
            )
            diarized_txt = self._write_result(with_speakers=True)[0]
            files_to_download.append(diarized_txt)

            empty_cache()

            print("Writing to docx...")
            doc = DocxGenerator()
            doc.txt_to_word(diarized_txt)
            download_from_colab(doc.docfilename)
        # DL audio file
        if not self.audio == self.input_audio:
            files_to_download.append(self.input_audio)

        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)


    def run2(self):
        """Diarize first, and ASR for each diarized segment"""
        files_to_download = []
        print("Diarizing...")
        self.diarized_segments = self.diarize()

        print("Transcribing...")
        self.asr_segments = self.transcribe_segmented()

        print("Writing asr result...")
        outfiles = self._write_result()
        files_to_download.extend(outfiles)

        print("Writing dia result...")
        diarized_txt = self._write_result(with_speakers=True)[0]
        files_to_download.append(diarized_txt)

        empty_cache()

        print("Writing to docx...")
        doc = DocxGenerator()
        doc.txt_to_word(diarized_txt)
        download_from_colab(doc.docfilename)

        # DL audio file
        if not self.audio == self.input_audio:
            files_to_download.append(self.input_audio)

        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)
