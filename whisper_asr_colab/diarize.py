import logging
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy as torch_from_numpy
from numpy import ndarray, minimum as np_minimum, maximum as np_maximum
from pandas import DataFrame
from typing import List, Union, Optional, NamedTuple
from itertools import groupby
from faster_whisper.transcribe import Segment, Word
from pyannote.audio import Pipeline
from .audio import load_audio


class DiarizedSegment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None
    words: Optional[List['Word']] = None
    temperature: Optional[float] = 1.0
    speaker: Optional[str] = None

class DiarizedWord(NamedTuple):
    start: float
    end: float
    word: str
    probability: float
    speaker: Optional[str] = None

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

    def __call__(self, audio: Union[str, ndarray], num_speakers=None, min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch_from_numpy(audio[None, :]),
            'sample_rate': 16000
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df


def _fill_missing_speakers(segments: List[DiarizedSegment]) -> None:
    for i in range(1, len(segments)):
        if segments[i].speaker is None:
            segments[i] = segments[i]._replace(speaker=segments[i-1].speaker)


def _combine_same_speakers(
        segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:
    _grouped = [
        list(g) for k, g in groupby(segments, lambda x: x.speaker)
    ]
    _combined = [
        DiarizedSegment(
            id = segs[0].id,
            seek = segs[0].seek,
            start = segs[0].start,
            end = segs[-1].end,
            text = "\n".join(seg.text for seg in segs).strip(),
            tokens = [token for seg in segs for token in seg.tokens],
            speaker = segs[0].speaker,
         ) for segs in _grouped
    ]
    return _combined


def assign_speakers(
        diarize_df: DataFrame,
        asr_segments: List[Segment],
        fill_missing_speakers: bool = True,
        combine_same_speakers: bool = True,
    ) -> List[DiarizedSegment]:

    def _get_speaker(start: float, end: float) -> str:
        diarize_df['intersection'] = np_minimum(diarize_df['end'], end) - np_maximum(diarize_df['start'], start)
        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        if dia_tmp.empty:
            return None
        return dia_tmp.groupby("speaker")["intersection"].sum().idxmax()

    diarized_segs = []
    for seg in asr_segments:
        seg = DiarizedSegment(
            *seg,
            speaker=_get_speaker(seg.start, seg.end)
        )
        # assign speaker to words
        if seg.words is not None:
            for word in seg.words:
                word = DiarizedWord(
                    *word,
                    speaker=_get_speaker(word.start, word.end)
                )
        diarized_segs.append(seg)
    if fill_missing_speakers: #If speaker is None, fill with the previous speaker
        _fill_missing_speakers(diarized_segs)
    if combine_same_speakers:
        diarized_segs = _combine_same_speakers(diarized_segs)
    return diarized_segs


def diarize(
        audio: Union[str, ndarray],
        asr_segments: List[Segment],
        hugging_face_token: str,
    ) -> List[DiarizedSegment]:

    diarize_model = DiarizationPipeline(use_auth_token=hugging_face_token)
    diarized_result = diarize_model(audio)
    logging.info(f"diarized_result: {diarized_result}")
    logging.info(">>performing assign_word_speakers...")
    segments = assign_speakers(diarized_result, asr_segments)
    return segments
