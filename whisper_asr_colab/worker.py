import sys
import os
import re
import time
import logging
from torch.cuda import empty_cache
from numpy import ndarray
from dataclasses import dataclass
from typing import Optional, Union, Iterable
from .docx_generator import DocxGenerator
from .audio import dl_audio, trim_audio
from .utils import write_asr_result, write_diarize_result, download_from_colab
from .asr import faster_whisper_transcribe, realtime_transcribe
from .diarize import diarize


@dataclass
class Worker:
    # model options
    model_size: str = "large-v3"
    device: str = "auto"

    # transcribe options
    audio: Union[str, ndarray] = "" # original audio path or data
    language: Optional[str] = None
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    hotwords: Optional[str] = None
    chunk_length: int = 30
    batch_size: int = 16
    prefix: Optional[str] = None
    vad_filter: bool = False,
    log_progress: bool = False

    # other options
    diarization: bool = True
    hugging_face_token: str = ""
    password: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timestamp_offset: Optional[str] = None
    realtime: bool = False

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
                self._input_audio = trim_audio(self.audio, self.start_time, self.end_time)


    def transcribe(self):
        # Transcribe
        if self.realtime: # realtime trascription
            realtime_transcribe(
                url = self.audio,
                model_size = self.model_size,
                language = self.language,
                initial_prompt = self.initial_prompt
            )
            empty_cache()
            sys.exit(0)
        else:  # use faster-whisper
            asr_segments, _ = faster_whisper_transcribe(
                audio=self.input_audio,
                model_size=self.model_size,
                initial_prompt=self.initial_prompt,
                hotwords = self.hotwords,
                prefix = self.prefix,
                #chunk_length=self.chunk_length,
                batch_size=self.batch_size,
            )

        # write results to text files
        outfiles = write_asr_result(
            os.path.basename(self.input_audio),
            asr_segments,
            self.timestamp_offset
        )
        for filename in outfiles:
            download_from_colab(filename)

        if self.diarization:
            # Diarize
            segments = diarize(
                audio=self.input_audio,
                asr_segments=asr_segments,
                hugging_face_token=self.hugging_face_token,
            )
            logging.info(f"Diarized segments:\n{segments}")
            filename = write_diarize_result(
                f"{os.path.basename(self.input_audio)}",
                segments,
                self.timestamp_offset
            )
            download_from_colab(filename)

        # gc GPU RAM
        #del segments
        empty_cache()

        # Generate docx
        if self.diarization:
            doc = DocxGenerator()
            doc.txt_to_word(filename)
            download_from_colab(doc.docfilename)

        # DL audio file
        if not self.audio == self.input_audio:
            download_from_colab(self.input_audio)

    def run(self):
        self.transcribe()
