import sys
import os
import re
import time
import logging
from torch.cuda import empty_cache
from numpy import ndarray
from dataclasses import dataclass
from typing import Optional, Union, Iterable, List
from .docx_generator import DocxGenerator
from .audio import dl_audio, trim_audio
from .utils import write_asr_result as _write_asr_result, write_diarize_result as _write_diarize_result, download_from_colab
from .asr import faster_whisper_transcribe, realtime_transcribe
from .diarize import diarize as _diarize


@dataclass
class Worker:
    # model options
    model_size: str = "large-v3"
    device: str = "auto"

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
    password: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timestamp_offset: Optional[str] = None
    realtime: bool = False

    #result data
    asr_segments: Optional[List] = None
    diarized_segments: Optional[List] = None

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


    def transcribe(self):
        # Transcribe
        if self.realtime: # realtime trascription
            realtime_transcribe(
                url = self.audio,
                model_size = self.model_size,
                language = self.language,
                multilingual=self.multilingual,
                initial_prompt = self.initial_prompt
            )
            empty_cache()
            sys.exit(0)
        else:  # use faster-whisper
            self.asr_segments, _ = faster_whisper_transcribe(
                audio=self.input_audio,
                model_size=self.model_size,
                language = self.language,
                multilingual=self.multilingual,
                initial_prompt=self.initial_prompt,
                hotwords = self.hotwords,
                prefix = self.prefix,
                vad_filter=self.vad_filter,
                #chunk_length=self.chunk_length,
                batch_size=self.batch_size,
            )

    def diarize(self):
        self.diarized_segments = _diarize(
            audio=self.input_audio,
            asr_segments=self.asr_segments,
            hugging_face_token=self.hugging_face_token,
        )

    def write_asr_result(self) -> tuple[str, ...]:
        # write results to text files
        return _write_asr_result(
            os.path.basename(self.input_audio),
            self.asr_segments,
            self.timestamp_offset
        )

    def write_diarize_result(self) -> tuple[str, ...]:
        # write results to text files
        return _write_diarize_result(
            os.path.basename(self.input_audio),
            self.diarized_segments,
            self.timestamp_offset
        )

    def run(self):
        files_to_download = []
        print("Transcribing...")
        self.transcribe()
        print("Writing result...")
        outfiles = self.write_asr_result()
        files_to_download.extend(outfiles)

        print("Diarizing...")
        if self.diarization:
            self.diarize()
            print("Writing result...")
            diarized_txt = self.write_diarize_result()[0]
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
