from __future__ import annotations
from json import dump as jsondump, load as jsonload
from itertools import groupby
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, List
from faster_whisper.transcribe import Segment
from .utils import str2seconds, format_timestamp
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class SpeakerSegment(Segment):
    id: int = field(default=0),
    seek: int = field(default=0),
    start: float = field(default=0.0),
    end: float = field(default=0.0),
    text: str = field(default=""),
    tokens: List[int] = field(default=[]),
    temperature: float = field(default=0.0),
    avg_logprob: float = field(default=0.0),
    compression_ratio: float = field(default=0.0),
    no_speech_prob: float = field(default=0.0),
    words: list = field(default=None),
    speaker: str = field(default=None),

    @property
    def duration(self):
        return(self.end - self.start)

    @classmethod
    def from_segment(cls, segment: Segment) -> SpeakerSegment:
        return cls(**segment.__dict__)

    def append(self, item):
        super().append(item)
        return self

    def shift_time(self, offset: Union[int, float]):
        self.start += offset,
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


class SpeakerSegmentList(List):
    def __init__(self, *args):
        super().__init__(args)

    @property
    def combined(self):
            head_seg = self[0]
            combined_seg = SpeakerSegment(
                    id = head_seg.id,
                    seek = head_seg.seek,
                    start = head_seg.start,
                    end = self[-1].end,
                    text = "\n".join(seg.text for seg in self).strip(),
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


    def combine_same_speakers(self) -> SpeakerSegmentList:
        speakersegs = SpeakerSegmentList()
        for k, g in groupby(self, lambda x: x.speaker):
            speakersegs.append(SpeakerSegmentList(*list(g)).combined)
        return speakersegs


    def fill_missing_speakers(self) -> None:
        for i in range(1, len(self)):
            if self[i].speaker is None:
                self[i] = self[i]._replace(speaker=self[i-1].speaker)


    def assign_speakers(
            self,
            diarization_result: SpeakerSegmentList,  # pyannote diarization result
            fill_missing_speakers: bool = False,
            combine_same_speakers: bool = True,
        ) -> SpeakerSegmentList:
        """Assign speakers for to ASR result based on diarization result"""

        dia_segments_size = len(diarization_result) - 1
        i = 0
        durations = defaultdict(float)
        diarized_segs = SpeakerSegmentList()
        for asr_seg in self:
            while i <= dia_segments_size:
                dia_seg = diarization_result[i]
                logger.debug(f"i:{i} speaker:{dia_seg.speaker} dia_seg.start:{dia_seg.start} dia_seg.end:{dia_seg.end}")
                if asr_seg.end < dia_seg.start:  # run out of the target segment
                    break
                # calc overlap duration of asr and diarization
                start = max(asr_seg.start, dia_seg.start)
                end = min(asr_seg.end, dia_seg.end)
                duration = end - start
                if duration > 0.0:
                    durations[dia_seg.speaker] += duration
                i += 1
            # assign the speaker who have longest overlap in each segment
            asr_seg.speaker = max(durations, key=durations.get, default=None)

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
        if fill_missing_speakers: #If speaker is None, fill with the previous speaker
            diarized_segs.fill_missing_speakers()
        if combine_same_speakers:
            diarized_segs = diarized_segs.combine_same_speakers()
        return diarized_segs


    def write_result(
            self,
            outfilename: str,
            with_speakers: bool = False,
            timestamp_offset: Union[int, float, str] = 0.0
            ) -> tuple[str, ...]:
        """ write results to text files """
        _write_func = (
            self.write_asr_result_with_speakers if with_speakers
            else self.write_asr_result
        )
        return _write_func(
            outfilename,
            timestamp_offset
        )


    def write_asr_result(
            self,
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
        for data in self:
            fh_text.write(data.text + "\n")
            fh_timestamped.write(data.add_timestamp(timestamp_offset) + "\n")
        fh_text.close()
        fh_timestamped.close()
        return outfilenames


    def write_asr_result_with_speakers(
            self,
            basename: str,
            timestamp_offset: Optional[Union[int, float, str]] = 0.0
            ) -> tuple[str, ...]:
        """Write diarized transcpript to a text file."""
        outfilename = f"{basename}_diarized.txt"
        fh = open(outfilename, "w", encoding="utf-8")
        for data in self:
            fh.write(data.time_segment_text(timestamp_offset) + " ")
            if data.speaker:
                fh.write(data.speaker + "\n")
            else:
                fh.write("\n")
            fh.write(data.text.replace(" ", "") + "\n\n")
        fh.close()
        return (outfilename,)

    def save_segments(self, jsonfile: str):
        """Save segments to a JSON file."""
        with open(jsonfile, 'w') as fh:
            jsondump([asdict(segment) for segment in self], fh)


    def load_segments(jsonfile: str) -> SpeakerSegmentList:
        """Load segments from a JSON file."""
        with open(jsonfile, 'r') as fh:
            data_list = jsonload(fh)
        return SpeakerSegmentList(*[SpeakerSegment(**data) for data in data_list])
