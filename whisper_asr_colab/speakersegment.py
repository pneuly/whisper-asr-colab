from __future__ import annotations
from json import dump as jsondump, load as jsonload
from itertools import groupby
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, DefaultDict, List, Iterable
from faster_whisper.transcribe import Segment
from .utils import str2seconds, format_timestamp
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class SpeakerSegment(Segment):
    id: int = field(default=0)
    seek: int = field(default=0)
    start: float = field(default=0.0)
    end: float = field(default=0.0)
    text: str = field(default="")
    tokens: List[int] = field(default_factory=list)
    temperature: Optional[float] = field(default=None)
    avg_logprob: float = field(default=0.0)
    compression_ratio: float = field(default=0.0)
    no_speech_prob: float = field(default=0.0)
    words: Optional[list] = field(default=None)
    speaker: Optional[str] = field(default=None)

    @property
    def duration(self):
        return(self.end - self.start)

    @classmethod
    def from_segment(cls, segment: Segment) -> SpeakerSegment:
        return cls(**segment.__dict__)


    def shift_time(self, offset: Union[int, float]):
        self.start += offset
        self.end += offset


    def time_segment_text(
            self,
            timestamp_offset: Optional[Union[int, float, str]] = 0.0
            ) -> str:
        """Create a segment string '[H:MM:SS.ss - H:MM:SS.ss]' from a segment."""
        if timestamp_offset is None:
            timestamp_offset = 0.0
        _offset_seconds = str2seconds(timestamp_offset) if isinstance(timestamp_offset, str) else timestamp_offset
        start = self.start + _offset_seconds
        end = self.end + _offset_seconds
        return (f"[{format_timestamp(start)} - {format_timestamp(end)}]")


    def add_timestamp(
            self,
            timestamp_offset: Optional[Union[int, float, str]] = 0.0
            ) -> str:
        """Create a string '[H:MM:SS.ss - H:MM:SS.ss] {segment.text}' from a segment."""
        return (f"{self.time_segment_text(timestamp_offset)} {self.text.strip()}")


def combine(segments: List[SpeakerSegment]) -> SpeakerSegment:
        """Combine list of SpeakerSegment into a single SpeakerSegment"""
        head_seg = segments[0]
        combined_seg = SpeakerSegment(
                id = head_seg.id,
                seek = head_seg.seek,
                start = head_seg.start, # combined
                end = segments[-1].end, # combined
                text = "\n".join(str(seg.text) for seg in segments).strip(), # combined
                #tokens = [token for seg in self for token in seg.tokens],
                tokens = [],  #strip tokens to reduce the use of memory
                temperature = head_seg.temperature,
                avg_logprob = head_seg.avg_logprob,
                compression_ratio = head_seg.compression_ratio,
                no_speech_prob = head_seg.no_speech_prob,
                words = None,
                speaker = head_seg.speaker,
            )
        return combined_seg


def combine_same_speakers(segments: List[SpeakerSegment]) -> None:
    """Combine consecutive segments that have same speaker into a segment (in place)"""
    speakersegs = []
    for k, g in groupby(segments, lambda x: x.speaker):
        speakersegs.append(combine(list(g)))
    segments[:] = speakersegs # in place


def fill_missing_speakers(segments: List[SpeakerSegment]) -> None:
    """If segment.speaker is None, fill with the speaker of the previous segment"""
    for i in range(1, len(segments)):
        if segments[i].speaker is None:
            segments[i].speaker = segments[i-1].speaker


def assign_speakers(
        asr_segments: List[SpeakerSegment],  # faster whisper asr result,
        diarization_result: List[SpeakerSegment],  # pyannote diarization result
        postprocesses : Iterable = [], # List of post process functions
    ) -> List[SpeakerSegment]:
    """Assign speakers for to ASR result based on diarization result"""

    dia_segments_size = len(diarization_result) - 1
    i = 0
    durations: DefaultDict[str, float] = defaultdict(float)
    diarized_segs = []
    for asr_seg in asr_segments:
        while i <= dia_segments_size:
            dia_seg = diarization_result[i]
            logger.debug(f"i:{i} speaker:{dia_seg.speaker} dia_seg.start:{dia_seg.start:.2f} dia_seg.end:{dia_seg.end:.2f}")
            if asr_seg.end < dia_seg.start:  # run out of the target segment
                break
            # calc overlap duration of asr and diarization
            start = max(asr_seg.start, dia_seg.start)
            end = min(asr_seg.end, dia_seg.end)
            duration = end - start
            if dia_seg.speaker and duration > 0.0:
                durations[dia_seg.speaker] += duration
            i += 1
        # assign the speaker who have longest overlap in each segment
        asr_seg.speaker = max(durations, key=lambda k : durations.get(k, 0.0), default=None)

        #logger.debug(f"{asr_seg.text}")
        #if durations:
        #    for key, value in durations.items():
        #        print(f"{key} : {value}")
        #else:
        #    logger.debug("empty durations")
        #logger.debug("\n")

        diarized_segs.append(asr_seg)
        durations.clear()
        i -= 1
    for _func in postprocesses:
        _func(diarized_segs)
    return diarized_segs


def write_result(
        segments : List[SpeakerSegment],
        outfilename: str,
        with_speakers: bool = False,
        timestamp_offset: Union[int, float, str] = 0.0
        ) -> tuple[str, ...]:
    """ write results to text files """
    _write_func = (
        write_asr_result_with_speakers if with_speakers
        else write_asr_result
    )
    return _write_func(
        segments,
        outfilename,
        timestamp_offset,
    )


def write_asr_result(
        segments : List[SpeakerSegment],
        basename: str,
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write ASR segments to files with and without timestamps.
    Returns filenames.
    """
    outfilenames = (f"{basename}.txt", f"{basename}_timestamped.txt")
    fh_text, fh_timestamped = [
        open(filename, "w", encoding="utf-8") for filename in outfilenames
    ]
    for data in segments:
        fh_text.write(data.text + "\n")
        fh_timestamped.write(data.add_timestamp(timestamp_offset) + "\n")
    fh_text.close()
    fh_timestamped.close()
    return outfilenames


def write_asr_result_with_speakers(
        segments,
        basename: str,
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write diarized transcpript to a text file."""
    outfilename = f"{basename}_diarized.txt"
    fh = open(outfilename, "w", encoding="utf-8")
    for data in segments:
        fh.write(data.time_segment_text(timestamp_offset) + " ")
        if data.speaker:
            fh.write(data.speaker + "\n")
        else:
            fh.write("\n")
        fh.write(data.text.replace(" ", "") + "\n\n")
    fh.close()
    return (outfilename,)


def save_segments(segments, jsonfile: str):
    """Save segments to a JSON file."""
    with open(jsonfile, 'w') as fh:
        jsondump([asdict(segment) for segment in segments], fh)


def load_segments(jsonfile: str) -> List[SpeakerSegment]:
    """Load segments from a JSON file."""
    with open(jsonfile, 'r') as fh:
        data_list = jsonload(fh)
    return [SpeakerSegment(**data) for data in data_list]
