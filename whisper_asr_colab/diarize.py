import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy as torch_from_numpy
from numpy import ndarray
from typing import List, Union, Optional
from pyannote.audio import Pipeline
from .audio import load_audio
from .speakersegment import SpeakerSegment


class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: Optional[Union[str, torch_device]] = "auto",
    ):
        if device == "auto":
            device = "cuda" if cuda_is_available() else "cpu"
            logging.info(f"Using device {device}")
        if isinstance(device, str):
            device = torch_device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(
            self, audio: Union[str, ndarray], num_speakers=None, min_speakers=None, max_speakers=None, min_duration_on=2.0
            ) -> List[SpeakerSegment]:
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch_from_numpy(audio[None, :]),
            'sample_rate': 16000
        }
        #self.model.instantiate({'min_duration_on': min_duration_on})
        annotation_generator = self.model(
            audio_data,
            num_speakers = num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,)
        speakersegments = [
            SpeakerSegment(segment, speaker)
            for segment, _, speaker in annotation_generator.itertracks(yield_label=True)
        ]
        return speakersegments


def diarize(
        audio: Union[str, ndarray],
        hugging_face_token: str,
    ) -> List[SpeakerSegment]:

    diarize_model = DiarizationPipeline(use_auth_token=hugging_face_token)
    diarized_result = diarize_model(audio)
    logging.info(f"diarized_result: {diarized_result}")
    return diarized_result
