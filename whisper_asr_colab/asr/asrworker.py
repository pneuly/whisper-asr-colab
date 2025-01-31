from collections import defaultdict
from typing import DefaultDict, Optional, Union, BinaryIO, TYPE_CHECKING
from faster_whisper import WhisperModel as FasterWhisperModel
if TYPE_CHECKING:
    import numpy
try:
    from ..speakersegment import SpeakerSegmentList
    from ..audio import Audio
    from .asr import faster_whisper_transcribe
except ImportError:
    from whisper_asr_colab.speakersegment import SpeakerSegmentList
    from whisper_asr_colab.audio import Audio
    from whisper_asr_colab.asr.asr import faster_whisper_transcribe


class ASRWorker:
    audio: Union[str, BinaryIO, 'numpy.ndarray']
    model: Optional[FasterWhisperModel] = None
    asr_segments: Optional[SpeakerSegmentList]
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

    def run(self) -> dict[str, str]:
        """Wrapper for ASR"""
        # Isolate the ASR process from diarization process
        # because Pipeline of pyannote.audio crashes if faster whisper is called beforehand.
        self.asr_segments, _ = faster_whisper_transcribe(
            audio=self.audio.ndarray,
            model=self.model,
            **self.transcribe_args
        )
        #outfilename = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfiles = self.asr_segments.write_asr_result(self.audio.local_file_path) #add timestamp offset if needed
        del self.model

        print("Saving ASR result as json file.")
        self.asr_segments.save("asr_result.json")
        return outfiles
