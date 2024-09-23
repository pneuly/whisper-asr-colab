import logging
from collections import defaultdict
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device, from_numpy as torch_from_numpy
from numpy import ndarray
from typing import List, Union, Optional, NamedTuple
from itertools import groupby
from faster_whisper.transcribe import Segment
from pyannote.audio import Pipeline
from pyannote.core import Segment as TimeSegment
from .audio import load_audio

class Annotation(NamedTuple):
    segment: Union[Segment, TimeSegment]
    speaker: Optional[str]


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
            self, audio: Union[str, ndarray], num_speakers=None, min_speakers=None, max_speakers=None
            ) -> List[Annotation]:
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch_from_numpy(audio[None, :]),
            'sample_rate': 16000
        }
        annotation_generator = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        annotations = [
            Annotation(segment, speaker)
            for segment, _, speaker in annotation_generator.itertracks(yield_label=True)
        ]
        return annotations


def _fill_missing_speakers(annotations: List[Annotation]) -> None:
    for i in range(1, len(annotations)):
        if annotations[i].speaker is None:
            annotations[i] = annotations[i]._replace(speaker=annotations[i-1].speaker)


def _combine_same_speakers(
        annotations: List[Annotation],
    ) -> List[Annotation]:
    _grouped = [
        list(g) for k, g in groupby(annotations, lambda x: x.speaker)
    ]
    _combined = []
    for annos in _grouped:
        seg, speaker = annos[0]
        _combined.append(
            Annotation(
                Segment(
                    id = seg.id,
                    seek = seg.seek,
                    start = seg.start,
                    end = annos[-1].segment.end,
                    text = "\n".join(seg.text for seg, _ in annos).strip(),
                    tokens = [token for seg, _ in annos for token in seg.tokens],
                    temperature = seg.temperature,
                    avg_logprob = seg.avg_logprob,
                    compression_ratio = seg.compression_ratio,
                    no_speech_prob = seg.no_speech_prob,
                    words = None,
                ),
                speaker
            )
        )
    return _combined


def assign_speakers(
        annotations: List[Annotation],
        asr_segments: List[TimeSegment],
        fill_missing_speakers: bool = False,
        combine_same_speakers: bool = True,
    ) -> List[Annotation]:

    dia_segments_size = len(annotations) - 1
    i = 0
    durations = defaultdict(float)
    diarized_segs = []
    for asr_seg in asr_segments:
        while i <= dia_segments_size:
            dia_seg, speaker = annotations[i]
            if asr_seg.end < dia_seg.start:  # run out of the target segment
                break
            duration = (dia_seg & asr_seg).duration
            if duration > 0.0:
                durations[speaker] += (dia_seg & asr_seg).duration
            i += 1
        diarized_segs.append(Annotation(
            asr_seg,
            max(durations, key=durations.get, default=None)
            ))
        durations.clear()
        i -= 1
    if fill_missing_speakers: #If speaker is None, fill with the previous speaker
        _fill_missing_speakers(diarized_segs)
    if combine_same_speakers:
        diarized_segs = _combine_same_speakers(diarized_segs)
    return diarized_segs


def diarize(
        audio: Union[str, ndarray],
        asr_segments: List[Segment],
        hugging_face_token: str,
    ) -> List[TimeSegment]:

    diarize_model = DiarizationPipeline(use_auth_token=hugging_face_token)
    diarized_result = diarize_model(audio)
    logging.info(f"diarized_result: {diarized_result}")
    logging.info(">>performing assign_word_speakers...")
    segments = assign_speakers(diarized_result, asr_segments)
    return segments
