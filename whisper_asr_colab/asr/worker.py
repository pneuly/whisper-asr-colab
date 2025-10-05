from dataclasses import dataclass
from typing import Optional, List, Union, Iterable, Any
from whisper_asr_colab.common.speakersegment import SpeakerSegment
from whisper_asr_colab.asr.asr import load_model, faster_whisper_transcribe

@dataclass
class ASRWorker:
    audio: Any  # Replace with your Audio type
    model_size: str = "large-v3-turbo"
    device: str = "auto"
    language: Optional[str] = None
    multilingual: bool = False
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    hotwords: Optional[str] = None
    chunk_length: int = 30
    batch_size: int = 1
    prefix: Optional[str] = None
    vad_filter: bool = False
    log_progress: bool = False
    model: Optional[Any] = None
    asr_segments: Optional[List[SpeakerSegment]] = None

    def load_model(self):
        self.model = load_model(
            model_size=self.model_size,
            device=self.device,
            compute_type="default",
        )

    def run(self) -> List[SpeakerSegment]:
        if self.model is None:
            self.load_model()
        segments, _ = faster_whisper_transcribe(
            audio=self.audio.ndarray,
            model=self.model,
            language=self.language,
            multilingual=self.multilingual,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
            chunk_length=self.chunk_length,
            batch_size=self.batch_size,
            prefix=self.prefix,
            vad_filter=self.vad_filter,
            log_progress=self.log_progress,
        )
        self.asr_segments = segments
        return segments
