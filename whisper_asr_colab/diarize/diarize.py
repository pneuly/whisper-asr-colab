import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy
from typing import List, Union, Optional, BinaryIO, Any
from numpy import ndarray
import contextlib
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from whisper_asr_colab.common.speakersegment import SpeakerSegment

logger = logging.getLogger(__name__)

def diarize(
        audio: Union[str, BinaryIO, ndarray],
        model_name: Optional[str] = "pyannote/speaker-diarization-community-1",
        hf_token: str = "",
        device: Optional[str] = "auto",
        show_progress: bool = True,
        hyperparams: Optional[dict] = None,
     ) -> List[SpeakerSegment]:

    if device == "auto":
        device = "cuda" if cuda_is_available() else "cpu"
        logger.info(f"Auto-selected device: {device}")
    if isinstance(device, str):
        device = torch_device(device)
             
    pipeline = Pipeline.from_pretrained(
            model_name, token=hf_token).to(device)
    if hyperparams:
         pipeline.instantiate(hyperparams)
         
    if isinstance(audio, ndarray):
        audio_data = {"waveform": from_numpy(audio[None, :]), "sample_rate": 16000}    
    else:
        audio_data = {'uri': 'audio_stream', 'audio': audio}

    hook = ProgressHook() if show_progress else None
    speaker_segments = []
    with hook or contextlib.nullcontext():
        for turn, speaker in pipeline(audio_data, hook=hook).exclusive_speaker_diarization:
            speaker_segments.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"diarized_result: {speaker_segments}")
    return speaker_segments
