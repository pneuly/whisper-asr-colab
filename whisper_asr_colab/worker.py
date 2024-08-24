import sys
import os
import re
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from .docx_generator import DocxGenerator
from .audio import dl_audio, trim_audio
from .utils import write_asr_result, write_diarize_result, download_from_colab
from .asr import whisperx_transcribe, faster_whisper_transcribe, realtime_transcribe
from .diarize import diarize

@dataclass
class Worker:
    audio: Union[str, np.ndarray] = "" # original audio path or data
    model_size: str = "large-v3"
    diarization: bool = True
    password: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timestamp_offset: Optional[str] = None
    initial_prompt: Optional[str] = None
    realtime: bool = False
    chunk_size: int = 20
    batch_size: int = 16
    hugging_face_token: str = ""
    _input_audio: Union[str, np.ndarray] = "" # audio input to pass to whisper

    @property
    def input_audio(self) -> Union[str, np.ndarray]:
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
            realtime_transcribe(self.audio, self.model_size, self.initial_prompt)
            del self.model_size
            torch.cuda.empty_cache()
            sys.exit(0)
        elif self.diarization: # use WhisperX
            result = whisperx_transcribe(
                audio=self.input_audio,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
                model_size=self.model_size,
                initial_prompt=self.initial_prompt
                )
            segments = result["segments"]
        else:  # use faster-whisper
            segments = faster_whisper_transcribe(
                audio=self.input_audio,
                model_size=self.model_size,
                initial_prompt=self.initial_prompt
                )

        # write results to text files
        outfiles = write_asr_result(
            os.path.basename(self.input_audio),
            segments,
            self.timestamp_offset
        )
        for filename in outfiles:
            download_from_colab(filename)

        if self.diarization:
            # Diarize
            segments = diarize(
                audio=self.input_audio,
                asr_result=result,
                hugging_face_token=self.hugging_face_token,
            )
            filename = write_diarize_result(
                f"{os.path.basename(self.input_audio)}",
                segments,
                self.timestamp_offset
            )
            download_from_colab(filename)

        # gc GPU RAM
        #del segments
        torch.cuda.empty_cache()

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
