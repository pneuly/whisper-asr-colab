import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy
from typing import List, Union, Optional, BinaryIO
from numpy import ndarray
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from .speakersegment import SpeakerSegment

logger = logging.getLogger(__name__)

class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: Union[str, torch_device] = "auto",
    ):
        if device == "auto":
            device = "cuda" if cuda_is_available() else "cpu"
            logging.info(f"Using device {device}")
        if isinstance(device, str):
            device = torch_device(device)
        self.pipeline = Pipeline.from_pretrained(
            model_name, use_auth_token=use_auth_token).to(device)

    def __call__(
            self,
            audio: Union[str, ndarray, BinaryIO],
            min_duration_on=1.5,  # remove speech regions shorter than this seconds.
            min_duration_off=1.5,  # fill non-speech regions shorter than this seconds.
            ) -> List[SpeakerSegment]:
        if isinstance(audio, ndarray):
            audio_data = {"waveform": from_numpy(audio[None, :]), "sample_rate": 16000}
        #elif isinstance(audio, str):
        #    audio_data = {'uri': 'audio_stream', 'audio': BytesIO(read_audio(audio, format="wav").stdout.read())}
        else:
            audio_data = {'uri': 'audio_stream', 'audio': audio}

        self.pipeline.min_duration_on = min_duration_on
        self.pipeline.min_duration_off = min_duration_off
        speaker_segments = []
        with ProgressHook() as hook:
            for time_segment, _, speaker in self.pipeline(
                    audio_data, hook=hook,).itertracks(yield_label=True):
                speaker_segments.append(
                    SpeakerSegment(
                        start=time_segment.start,
                        end=time_segment.end,
                        speaker=speaker
                    )
                )
        return speaker_segments


def diarize(
        audio: Union[str, BinaryIO, ndarray],
        hugging_face_token: str,
    ) -> List[SpeakerSegment]:

    diarize_model = DiarizationPipeline(use_auth_token=hugging_face_token)
    diarized_result = diarize_model(audio)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"diarized_result: {diarized_result}")
    return diarized_result
