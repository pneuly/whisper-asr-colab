import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy
from typing import Union, Optional, BinaryIO
from numpy import ndarray
from packaging.version import parse
from importlib.metadata import version
import contextlib
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from whisper_asr_colab.speakersegment import SpeakerSegment, SpeakerSegmentList

USE_PYANNOTE4 = parse(version("pyannote.audio")) >= parse("4.0.0")
PYANNOTE_MODEL = (
    "pyannote/speaker-diarization-community-1"
    if USE_PYANNOTE4
    else "pyannote/speaker-diarization-3.1"
)

logger = logging.getLogger(__name__)


def diarize(
    audio: Union[str, BinaryIO, ndarray],
    model_name: Optional[str] = None,
    hf_token: str = "",
    device: Optional[str] = "auto",
    show_progress: bool = True,
    hyperparams: Optional[dict] = None,
) -> SpeakerSegmentList:
    model_name = model_name or PYANNOTE_MODEL
    if device == "auto":
        device = "cuda" if cuda_is_available() else "cpu"
        logger.info(f"Auto-selected device: {device}")
    if isinstance(device, str):
        device = torch_device(device)

    pipeline = (
        Pipeline.from_pretrained(model_name, token=hf_token).to(device)
        if USE_PYANNOTE4
        else Pipeline.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    )
    if hyperparams:
        pipeline.instantiate(hyperparams)

    if isinstance(audio, ndarray):
        audio_data = {"waveform": from_numpy(audio[None, :]), "sample_rate": 16000}
    else:
        audio_data = {"uri": "audio_stream", "audio": audio}

    hook = ProgressHook() if show_progress else None
    speaker_segments = SpeakerSegmentList()
    with hook or contextlib.nullcontext():
        for turn, speaker in (
            pipeline(audio_data, hook=hook).exclusive_speaker_diarization
            if USE_PYANNOTE4
            else [
                t[0::2]
                for t in pipeline(audio_data, hook=hook).itertracks(yield_label=True)
            ]
        ):
            speaker_segments.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    id=0,
                    seek=0,
                    text="",
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=0.0,
                    no_speech_prob=0.0,
                    words=None,
                    speaker=speaker,
                )
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"diarized_result: {speaker_segments}")
    return speaker_segments
