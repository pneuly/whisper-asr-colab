from json import dump as jsondump, load as jsonload
from datetime import datetime
from typing import List, Optional, Union
from faster_whisper.transcribe import Segment
from pyannote.core import Segment as TimeSegment


def str2seconds(time_str: str) -> float:
    """Convert a time string to seconds."""
    for fmt in ("%H:%M:%S", "%M:%S", "%S", "%H:%M:%S.%f", "%M:%S.%f", "%S.%f"):
        try:
            return (
                datetime.strptime(time_str, fmt) - datetime(1900, 1, 1)
                ).total_seconds()
        except ValueError:
            pass
    raise ValueError(f"Error: Unable to parse time string '{time_str}'")


def format_timestamp(seconds: float) -> str:
    """Format seconds into a string 'H:MM:SS.ss'."""
    hours = seconds // 3600
    remain = seconds - (hours * 3600)
    minutes = remain // 60
    seconds = remain - (minutes * 60)
    return "{:01}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)


def combine_segments(segments: List[Union[Segment, TimeSegment]]):
        head_seg = segments[0]
        if isinstance(head_seg, Segment):
            combined_seg = Segment(
                    id = head_seg.id,
                    seek = head_seg.seek,
                    start = head_seg.start,
                    end = segments[-1].end,
                    text = "\n".join(seg.text for seg in segments).strip(),
                    tokens = [token for seg in segments for token in seg.tokens],
                    temperature = head_seg.temperature,
                    avg_logprob = head_seg.avg_logprob,
                    compression_ratio = head_seg.compression_ratio,
                    no_speech_prob = head_seg.no_speech_prob,
                    words = None,
                )
        elif isinstance(head_seg, TimeSegment):
            combined_seg = TimeSegment(
                start = head_seg.start,
                end = segments[-1].end,
            )
        else:
            raise TypeError("segments must be list of Segment or TimeSegment.")
        return combined_seg


def shift_segment_time(
        segment: Union[Segment, TimeSegment],
        offset: Union[int, float]
    ) -> Union[Segment, TimeSegment]:
    """"Shift a segment by offset"""
    if isinstance(segment, Segment):
        return Segment(
            id=segment.id,
            seek=segment.seek,
            start=segment.start + offset,
            end=segment.end + offset,
            text=segment.text,
            tokens=segment.tokens,
            temperature=segment.temperature,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
            no_speech_prob=segment.no_speech_prob,
            words=None,
        )
    if isinstance(segment, TimeSegment):
        return TimeSegment(segment.start + offset, segment.end + offset)

def time_segment_text(
        segment: Union[Segment, TimeSegment],
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> str:
    """Create a segment string '[H:MM:SS.ss - H:MM:SS.ss]' from a segment."""
    if timestamp_offset is None:
        timestamp_offset = 0.0
    _offset_seconds = str2seconds(timestamp_offset) if isinstance(timestamp_offset, str) else timestamp_offset
    start = segment.start + _offset_seconds
    end = segment.end + _offset_seconds
    return (f"[{format_timestamp(start)} - {format_timestamp(end)}]")


def add_timestamp(
        segment: Segment,
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> str:
    """Create a string '[H:MM:SS.ss - H:MM:SS.ss] {segment.text}' from a segment."""
    return (f"{time_segment_text(segment, timestamp_offset)} {segment.text.strip()}")


def save_segments(jsonfile: str, segments: List[Segment]):
    """Save segments to a JSON file."""
    with open(jsonfile, 'w') as fh:
        jsondump([segment._asdict() for segment in segments], fh)


def load_segments(jsonfile: str) -> List[Segment]:
    """Load segments from a JSON file."""
    with open(jsonfile, 'r') as fh:
        data_list = jsonload(fh)
    return [Segment(**data) for data in data_list]


def download_from_colab(filepath: str):
    """Download a file in Google Colab environment."""
    if str(get_ipython()).startswith("<google.colab."):
        from google.colab import files
        files.download(filepath)
