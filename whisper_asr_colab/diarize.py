import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device
from typing import Union, Optional, BinaryIO
from io import BytesIO
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from .audio import read_audio
from .speakersegment import SpeakerSegment, SpeakerSegmentList

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
        self.pipeline = Pipeline.from_pretrained(
            model_name, use_auth_token=use_auth_token).to(device)

    def __call__(
            self,
            audio: Union[str, BinaryIO],
            min_duration_on=1.5,  # remove speech regions shorter than this seconds.
            min_duration_off=1.5,  # fill non-speech regions shorter than this seconds.
            ) -> SpeakerSegmentList:
        if isinstance(audio, str):
            audio_data = {'uri': 'audio_stream', 'audio': BytesIO(read_audio(audio, format="wav").stdout.read())}
        else:
            audio_data = {'uri': 'audio_stream', 'audio': audio}
        #if isinstance(audio, ndarray):
        #    audio_data = {'uri': 'audio_uri', 'audio': audio, 'sample_rate': 16000}
        #else:
        #    audio_data = {'uri': 'audio_stream', 'audio': BytesIO(read_audio(audio, format="wav").stdout.read())}

        self.pipeline.min_duration_on = min_duration_on
        self.pipeline.min_duration_off = min_duration_off
        speaker_segments = SpeakerSegmentList()
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
        #    speaker_segments = SpeakerSegmentList(*[
        #        SpeakerSegment(
        #            start=time_segment.start,
        #            end=time_segment.end,
        #            speaker=speaker
        #        ) for time_segment, _, speaker in self.pipeline(
        #            audio_data, hook=hook,).itertracks(yield_label=True)
        #    ])
        return speaker_segments


def diarize(
        audio: Union[str, BinaryIO],
        hugging_face_token: str,
    ) -> SpeakerSegmentList:

    diarize_model = DiarizationPipeline(use_auth_token=hugging_face_token)
    diarized_result = diarize_model(audio)
    logging.info(f"diarized_result: {diarized_result}")
    return diarized_result
