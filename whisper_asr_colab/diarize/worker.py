from dataclasses import dataclass
from typing import Optional, List
try:
    from ..common.speakersegment import SpeakerSegment, assign_speakers, combine_same_speakers, load_segments, write_result
    from ..common.audio import Audio
    from .diarize import diarize as diarize
except ImportError:
    from whisper_asr_colab.common.speakersegment import SpeakerSegment, assign_speakers, combine_same_speakers, load_segments, write_result
    from whisper_asr_colab.common.audio import Audio
    from whisper_asr_colab.diarize.diarize import diarize

@dataclass
class DiarizationWorker:
    audio: Audio
    hugging_face_token: str = ""
    hyperargs: Optional[dict] = None
    asr_segments: Optional[List[SpeakerSegment]] = None
    diarized_segments: Optional[List[SpeakerSegment]] = None
    _integrated_segments: Optional[List[SpeakerSegment]] = None

    def run(self, show_progress=True) -> List[str]:
        if self.audio.ndarray is None:
            raise ValueError("Audio must be specified in DiarizationWorker.audio.")
        segments = diarize(
            audio=self.audio.ndarray,
            hf_token=self.hugging_face_token,
            show_progress=show_progress,
        )
        if segments and self.audio.start_time:
            for item in segments:
                item.shift_time(self.audio.start_time)

        self.diarized_segments = segments
        
        print("Writing diarization result.")
        return write_result(
            self.integrated_segments,
            self.audio.local_file_path,
            True)

    @property
    def integrated_segments(self):
        if not self._integrated_segments:
            self._integrate_with_asr()
        return self._integrated_segments
    
    def _integrate_with_asr(self, asr_json: Optional[str] = "asr_result.json"):
        if not self.asr_segments:
            self.asr_segments = load_segments(asr_json)
        if not self.diarized_segments:
            raise ValueError("self.diarized_segments is empty.")
        self._integrated_segments = assign_speakers(
            asr_segments=self.asr_segments,
            diarization_result=self.diarized_segments,
            postprocesses=(combine_same_speakers,),
        )
        return
