from whisper_asr_colab.asr.asr import load_model, faster_whisper_transcribe
from whisper_asr_colab.diarize.diarize import DiarizationPipeline
from whisper_asr_colab.common.speakersegment import assign_speakers, combine_same_speakers

class Pipeline:
    def __init__(self, audio, asr_config=None, diarize_config=None):
        self.audio = audio
        self.asr_config = asr_config or {}
        self.diarize_config = diarize_config or {}
        self.asr_segments = None
        self.diarized_segments = None

    def run_asr(self):
        model = load_model(**self.asr_config)
        segments, _ = faster_whisper_transcribe(
            audio=self.audio.ndarray,
            model=model,
            **{k: v for k, v in self.asr_config.items() if k not in ['model_size', 'device', 'compute_type']}
        )
        self.asr_segments = segments
        return segments

    def run_diarization(self):
        pipeline = DiarizationPipeline(**self.diarize_config)
        segments = pipeline(self.audio.ndarray)
        self.diarized_segments = segments
        return segments

    def integrate(self):
        if self.asr_segments is None or self.diarized_segments is None:
            raise ValueError("ASR and diarization must be run before integration.")
        return assign_speakers(
            asr_segments=self.asr_segments,
            diarization_result=self.diarized_segments,
            postprocesses=(combine_same_speakers,)
        )

    def run_full(self):
        self.run_asr()
        self.run_diarization()
        return self.integrate()
        return self.integrate()
