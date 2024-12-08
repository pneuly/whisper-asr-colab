import os
import re
import time
import gc
from io import BytesIO
from torch.cuda import empty_cache
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union, Iterable, BinaryIO, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel as FasterWhisperModel
from .docx_generator import DocxGenerator
from .audio import dl_audio, read_audio, get_silence_duration
from .utils import str2seconds, download_from_colab
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
    audio: Union[str, BinaryIO] = "" # original audio path or data
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
    start_time: float = 0.0
    end_time: float = 0.0
    timestamp_offset: float = 0.0
    realtime: bool = False
    skip_silence: bool = True  # If True, skip the leading silence of the audio

    #result data
    asr_segments: Optional[SpeakerSegmentList] = None  # result from whisper
    diarized_segments: Optional[SpeakerSegmentList] = None  # result from pyannote

    _input_audio: Union[str, BinaryIO] = "" # audio input to pass to whisper

    @property
    def input_audio(self) -> Union[str, BinaryIO]:
        return self._input_audio

    def __post_init__(self):
        # download audio if needed
        if self.audio and not self.realtime:
            self.start_time = str2seconds(self.start_time)
            self.end_time = str2seconds(self.end_time)
            self.timestamp_offset = str2seconds(self.timestamp_offset)
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
            #if self.start_time or self.end_time: # trim if specified
            #    print("Trimming the audio file.")
            #    self._input_audio = trim_audio(self._input_audio, self.start_time, self.end_time)
            if self.skip_silence and not self.start_time:
                sec = get_silence_duration(self._input_audio)
                if sec > 0.0:
                    print(f"Leading silence detected. Skipping {sec} seconds.")
                    self.start_time = sec


    def transcribe(self) -> SpeakerSegmentList:
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
            print("Transcribing on the fly...")
            segments = realtime_transcribe(
                url = self.audio,
                model=self.model,
                language = self.language,
                initial_prompt = self.initial_prompt
            )
        else:  # normal transcription
            print(f"Transcribing from {self.start_time if self.start_time else ''}...")
            segments, _ = self.call_faster_whisper_transcribe(
                                                    self.start_time,
                                                    self.end_time)
        return segments


    def transcribe_segmented(self) -> SpeakerSegmentList:
        """Transcribe each diarized segment separately. Called by run2()"""
        print("Transcribing each segment...")
        if self.model is None:
            self.model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                )

        def _transcribe_and_add_speakers(speakerseg):
            asr_result = SpeakerSegmentList()
            segments, _ = self.call_faster_whisper_transcribe(
                speakerseg.start,
                speakerseg.end,
            )
            if segments:
                segment = segments.combined
                segment.speaker = speakerseg.speaker
                asr_result.append(segment)
            return asr_result

        with ThreadPoolExecutor() as executor:
            speakersegs = self.diarized_segments.combine_same_speakers()
            procs = []
            for seg in speakersegs:
                procs.append(executor.submit(_transcribe_and_add_speakers, seg))
            executor.shutdown(wait=True)
            asr_result = SpeakerSegmentList()
            for proc in procs:
                result = proc.result()
                if result:
                    asr_result.extend(result)
        return asr_result


    def call_faster_whisper_transcribe(
            self,
            start_time: Optional[Union[int, float]] = None,
            end_time: Optional[Union[int, float]] = None) -> Tuple[SpeakerSegmentList, Any]:
        """Commonly used by `transcribe()` and `transcribe_segmented().`
        """
        audio = read_audio(
                    self.input_audio,
                    start_time=start_time,
                    end_time=end_time
                ).stdout
        segments, _ = faster_whisper_transcribe(
                audio=audio,
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
        del audio
        if segments and start_time:
            for item in segments:
                item.shift_time(start_time)
        return segments, _


    def diarize(self):
        print("Diarizing...")
        ## pyannote.audio cannot handle stdout file-like object correctry. So, pass the audio as BytesIO.
        audio = BytesIO(read_audio(
                    self.input_audio,
                    start_time=self.start_time,
                    end_time=self.end_time
                ).stdout.read())
        segments = _diarize(
            audio=audio,
            hugging_face_token=self.hugging_face_token,
        )
        if segments and self.start_time:
            for item in segments:
                item.shift_time(self.start_time)
        return segments

    def _write_result(self, speaker_segments, with_speakers=False):
        if isinstance(self.input_audio, str):
            outfilename = self.input_audio
        else:
            outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
        return speaker_segments.write_result(
            outfilename,
            with_speakers,
            self.timestamp_offset if self.timestamp_offset else 0.0
        )

    def run(self):
        """Wrapper for ASR and diarization"""
        files_to_download = []

        self.asr_segments = self.transcribe()

        print("Writing result...")
        outfiles = self._write_result(self.asr_segments)
        files_to_download.extend(outfiles)

        del self.model
        empty_cache()
        gc.collect()

        if self.diarization:
            self.diarized_segments = self.diarize()
            self.diarized_segments = self.asr_segments.assign_speakers(
                diarization_result=self.diarized_segments
            )

            print("Writing result...")
            diarized_txt = self._write_result(self.diarized_segments, with_speakers=True)[0]

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
        """Similar to run(), but diarize first and ASR for each diarized segment"""
        files_to_download = []

        self.diarized_segments = self.diarize()
        self.asr_segments = self.transcribe_segmented()

        print("Writing asr result...")
        outfiles = self._write_result(self.asr_segments)
        files_to_download.extend(outfiles)

        print("Writing dia result...")
        diarized_txt = self._write_result(self.asr_segments, with_speakers=True)[0]
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
