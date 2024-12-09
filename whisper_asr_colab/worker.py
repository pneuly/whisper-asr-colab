import gc
from torch.cuda import empty_cache
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union, Iterable, Tuple, List, Any
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel as FasterWhisperModel
from .docx_generator import DocxGenerator
from .audio import Audio
from .utils import download_from_colab
from .asr import faster_whisper_transcribe, realtime_transcribe
from .diarize import diarize as _diarize
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
    timestamp_offset: float = 0.0
    realtime: bool = False
    skip_silence: bool = True  # If True, skip the leading silence of the audio

    #result data
    asr_segments: Optional[List[SpeakerSegment]] = None  # result from whisper
    diarized_segments: Optional[List[SpeakerSegment]] = None  # result from pyannote


    def __post_init__(self):
        if isinstance(self.audio, str):
            self.audio = Audio.from_path_or_url(self.audio)
        if self.password:
            self.audio.password = self.password
        if self.realtime:
            print("`skip_silence` is disabled since `realtime` mode is enabled.")
            self.skip_silence = False
        if self.skip_silence:
            self.audio.set_silence_skip()

#    def get_audio_slice(self,
#                         start_time: Optional[Union[int, float]] = None,
#                         end_time: Optional[Union[int, float]] = None):
#        sr = self.model.feature_extractor.sampling_rate if self.model else 16000
#        i_start = 0 if not start_time else int(start_time * sr)
#        i_end = len(self._audio_data) if not end_time else int(end_time * sr)
#        return self._audio_data[i_start:i_end]

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
            print("Transcribing on the fly...")
            segments = realtime_transcribe(
                process=self.audio.live_stream,
                model=self.model,
                language = self.language,
                initial_prompt = self.initial_prompt
            )
        else:  # normal transcription
            segments, _ = self.call_faster_whisper_transcribe()
        return segments


    def transcribe_segmented(self) -> List[SpeakerSegment]:
        """Transcribe each diarized segment separately. Called by run2()"""
        print("Transcribing each segment...")
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
        return asr_result


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
        print(f"Transcribing from {_start} to {_end}")
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


    def diarize(self):
        print("Diarizing...")
        if self.audio.ndarray is None:
            raise ValueError("Audio must be specified in worker.audio.")
        segments = _diarize(
            audio=self.audio.ndarray,
            hugging_face_token=self.hugging_face_token,
        )
        if segments and self.audio.start_time:
            for item in segments:
                item.shift_time(self.audio.start_time)
        return segments

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

    # TODO run asr and diarization simultaneously (Threading)
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
            self.diarized_segments = assign_speakers(
                asr_segments=self.asr_segments,
                diarization_result=self.diarized_segments,
                postprocesses=(combine_same_speakers,),
            )

            print("Writing result...")
            diarized_txt = self._write_result(self.diarized_segments, with_speakers=True)[0]

            files_to_download.append(diarized_txt)

            empty_cache()

            print("Writing to docx...")
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
        print(f"Downloading {doc.docfilename}")
        download_from_colab(doc.docfilename)

        # DL audio file
        if not self.audio.url:
            files_to_download.append(self.audio.file_path)

        for file in files_to_download:
            print(f"Downloading {file}")
            download_from_colab(file)
