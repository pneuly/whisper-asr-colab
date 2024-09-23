from json import dump as jsondump, load as jsonload
from datetime import datetime
from typing import List, Optional, Union
from faster_whisper.transcribe import Segment
from pyannote.core import Segment as TimeSegment
from .diarize import Annotation
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


def write_asr_result(
        basename: str,
        segments: List[Segment],
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write ASR segments to files with and without timestamps."""
    outfilenames = (f"{basename}.txt", f"{basename}_timestamped.txt")
    fh_text, fh_timestamped = [
        open(filename, "w", encoding="utf-8") for filename in outfilenames
    ]
    for segment in segments:
        fh_text.write(segment.text + "\n")
        fh_timestamped.write(add_timestamp(segment, timestamp_offset) + "\n")
    fh_text.close()
    fh_timestamped.close()
    return outfilenames

def save_segments(jsonfile: str, segments: List[Segment]):
    """Save segments to a JSON file."""
    with open(jsonfile, 'w') as fh:
        jsondump([segment._asdict() for segment in segments], fh)


def load_segments(jsonfile: str) -> List[Segment]:
    """Load segments from a JSON file."""
    with open(jsonfile, 'r') as fh:
        data_list = jsonload(fh)
    return [Segment(**data) for data in data_list]


def write_diarize_result(
        basename: str,
        annotations: List[Annotation],
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write diarized transcpript to a text file."""
    outfilename = f"{basename}_diarized.txt"
    fh = open(outfilename, "w", encoding="utf-8")
    for segment, speaker in annotations:
        fh.write(time_segment_text(segment, timestamp_offset) + " ")
        if speaker:
            fh.write(speaker + "\n")
        else:
            fh.write("\n")
        fh.write(segment.text.replace(" ", "") + "\n\n")
    fh.close()
    return (outfilename,)

def download_from_colab(filepath: str):
    """Download a file in Google Colab environment."""
    if str(get_ipython()).startswith("<google.colab."):
        from google.colab import files
        files.download(filepath)
