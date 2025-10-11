from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional
from faster_whisper.transcribe import Segment
from ..common.utils import str2seconds, format_timestamp
from logging import getLogger

logger = getLogger(__name__)
@dataclass
class SpeakerSegment():
    """A segment of speech with an assigned speaker."""
    segment: Optional[Segment] = None
    speaker: Optional[str] = None

    @property
    def duration(self):
        return(self.segment.end - self.segment.start)

    def to_dict(self) -> dict:
        return {
            "segment": self.segment.__dict__ if self.segment else None,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpeakerSegment:
        seg_data = data.get("segment")
        seg = Segment(**seg_data) if seg_data else None
        return cls(segment=seg, speaker=data.get("speaker"))


    def shift_time(self, offset: Union[int, float]):
        self.segment.start += offset
        self.segment.end += offset


    def to_str(
        self,
        timestamp_offset: Union[int, float, str] = 0.0,
        with_timestamp: bool = True,
        with_speaker: bool = True,
        with_text: bool = True,
        ) -> str:
        _offset_seconds = str2seconds(timestamp_offset) if isinstance(timestamp_offset, str) else timestamp_offset
        start = self.segment.start + _offset_seconds
        end = self.segment.end + _offset_seconds
        txt = ""
        if with_timestamp:
            txt += f"[{format_timestamp(start)} - {format_timestamp(end)}] "
        if with_speaker and self.speaker:
            txt += self.speaker if self.speaker else ""
            txt += "\n"
        if with_text and self.segment and self.segment.text:
            txt += self.segment.text.strip()
        return txt
