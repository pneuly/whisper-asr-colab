from dataclasses import dataclass
from typing import Optional, List
from whisper_asr_colab.common.speakersegment import SpeakerSegment, assign_speakers, combine_same_speakers, load_segments
from whisper_asr_colab.diarize.diarize import DiarizationPipeline

@dataclass
class DiarizationWorker:
    audio: any  # Replace 'any' with your Audio type
    hugging_face_token: str = ""
    diarized_segments: Optional[List[SpeakerSegment]] = None
    asr_segments: Optional[List[SpeakerSegment]] = None

    def run(self, show_progress=True) -> List[SpeakerSegment]:
        if self.audio.ndarray is None:
            raise ValueError("Audio must be specified in DiarizationWorker.audio.")
        dpipe = DiarizationPipeline(use_auth_token=self.hugging_face_token)
        segments = dpipe(
            audio=self.audio.ndarray,
            show_progress=show_progress,
        )
        if segments and self.audio.start_time:
            for item in segments:
                item.shift_time(self.audio.start_time)
        self.diarized_segments = segments
        return self.diarized_segments

    def integrate(self):
        if not self.asr_segments:
            self.asr_segments = load_segments("asr_result.json")
        if not self.diarized_segments:
            raise ValueError("self.diarized_segments is empty.")
        self.diarized_segments = assign_speakers(
            asr_segments=self.asr_segments,
            diarization_result=self.diarized_segments,
            postprocesses=(combine_same_speakers,),
        )
        return self.diarized_segments
