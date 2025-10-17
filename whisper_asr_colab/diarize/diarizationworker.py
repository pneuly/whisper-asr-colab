from dataclasses import dataclass
from typing import Optional
try:
    from ..speakersegment import SpeakerSegmentList
    from ..audio import Audio
    from .diarize import diarize
except ImportError:
    from whisper_asr_colab.speakersegment import SpeakerSegmentList
    from whisper_asr_colab.audio import Audio
    from whisper_asr_colab.diarize import diarize

@dataclass
class DiarizationWorker:
    audio: Audio
    hugging_face_token: str = ""
    hyperargs: Optional[dict] = None
    asr_segments: Optional[SpeakerSegmentList] = None
    diarized_segments: Optional[SpeakerSegmentList] = None
    _integrated_segments: Optional[SpeakerSegmentList] = None

    def run(self, show_progress=True) -> str:
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
        return self.integrated_segments.write_integrated_result(
            self.audio.local_file_path) #add timestamp_offset if needed

    @property
    def integrated_segments(self):
        if not self._integrated_segments:
            self._integrate_with_asr()
        return self._integrated_segments
    
    def _integrate_with_asr(self, asr_json: Optional[str] = "asr_result.json"):
        if not self.asr_segments:
            self.asr_segments = SpeakerSegmentList.load(asr_json)
        if not self.diarized_segments:
            raise ValueError("self.diarized_segments is empty.")
        self._integrated_segments = self.asr_segments.assign_speakers(
                    diarization_result=self.diarized_segments,
                ).combine_same_speakers()
