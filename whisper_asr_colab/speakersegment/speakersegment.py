from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

from ..common.utils import format_timestamp, str2seconds

logger = getLogger(__name__)


@dataclass(kw_only=True)
class SpeakerSegment:
    """A segment of speech with an assigned speaker.
    Not inherited from faster_whisper Segment to avoid dependency.
    """

    start: float
    end: float
    id: int | None = None
    seek: int | None = None
    text: str | None = None
    tokens: list[int] | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    words: list | None = None
    temperature: float | None = None
    speaker: str | None = None

    @property
    def duration(self):
        return self.end - self.start

    def shift_time(self, offset: int | float):
        self.start += offset
        self.end += offset

    def to_str(
        self,
        timestamp_offset: int | float | str = 0.0,
        with_timestamp: bool = True,
        with_speaker: bool = True,
        with_text: bool = True,
    ) -> str:
        _offset_seconds = (
            str2seconds(timestamp_offset)
            if isinstance(timestamp_offset, str)
            else timestamp_offset
        )
        start = self.start + _offset_seconds
        end = self.end + _offset_seconds
        txt = ""
        if with_timestamp:
            txt += f"[{format_timestamp(start)} - {format_timestamp(end)}] "
        if with_speaker and self.speaker:
            txt += self.speaker if self.speaker else ""
            txt += "\n"
        if with_text and self.text:
            txt += self.text.strip()
        return txt
