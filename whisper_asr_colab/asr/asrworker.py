from collections import defaultdict
from typing import DefaultDict, Optional, List, Union, BinaryIO
from faster_whisper import WhisperModel as FasterWhisperModel
try:
    from ..common.speakersegment import SpeakerSegment, write_result, save_segments
    from ..common.audio import Audio
    from .asr import faster_whisper_transcribe
except ImportError:
    from whisper_asr_colab.common.speakersegment import SpeakerSegment, write_result, save_segments
    from whisper_asr_colab.common.audio import Audio
    from whisper_asr_colab.asr.asr import faster_whisper_transcribe


class ASRWorker:
    audio: Union[str, BinaryIO, 'numpy.ndarray']
    model: Optional[FasterWhisperModel] = None
    asr_segments: Optional[List[SpeakerSegment]]
    transcribe_args: DefaultDict[str, int | int | bool]
    def __init__(
            self,
            audio: Audio,
            load_model_args: Optional[dict[str, str | int | bool]] = None,
            transcribe_args: Optional[dict[str, str | int | bool]] = None,
        ):
        self.audio = Audio(audio)

        load_model_args = defaultdict(str, load_model_args or {})
        self.model = FasterWhisperModel(
            model_size_or_path = load_model_args["model_size_or_path"] or "turbo",
            device = load_model_args["device"] or "auto",
            compute_type = load_model_args["compute_type"] or "default", 
        )
        self.transcribe_args = defaultdict(str, transcribe_args or {})

    def run(self) -> List[SpeakerSegment]:
        """Wrapper for ASR"""
        # Isolate the ASR process from diarization process
        # because Pipeline of pyannote.audio crashes if faster whisper is called beforehand.
        self.asr_segments, _ = faster_whisper_transcribe(
            audio=self.audio.ndarray,
            model=self.model,
            **self.transcribe_args
        )

        #outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfiles = write_result(self.asr_segments, self.audio.local_file_path)
        del self.model

        print("Saving ASR result as json file.")
        save_segments(self.asr_segments, "asr_result.json")
        return outfiles
