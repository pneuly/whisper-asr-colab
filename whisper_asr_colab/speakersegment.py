from itertools import groupby
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Optional, List
from faster_whisper.transcribe import Segment
from pyannote.core import Segment as TimeSegment
from .utils import add_timestamp, time_segment_text, combine_segments

@dataclass
class SpeakerSegment:
    segment: Union[Segment, TimeSegment]
    speaker: Optional[str]

def _combine_same_speakers(
        speakersegments: List[SpeakerSegment],
    ) -> List[SpeakerSegment]:
    _grouped = [
        list(g) for k, g in groupby(speakersegments, lambda x: x.speaker)
    ]
    _combined = []
    for speakersegs in _grouped:
        segs = [_.segment for _ in speakersegs]
        _combined.append(
            SpeakerSegment(combine_segments(segs), speakersegs[0].speaker)
        )
    return _combined


def _fill_missing_speakers(speakersegments: List[SpeakerSegment]) -> None:
    for i in range(1, len(speakersegments)):
        if speakersegments[i].speaker is None:
            speakersegments[i] = speakersegments[i]._replace(speaker=speakersegments[i-1].speaker)


def assign_speakers(
        speakersegments: List[SpeakerSegment],  # pyannote diarization result
        asr_segments: List[TimeSegment],  # Whisper transcribing result
        fill_missing_speakers: bool = False,
        combine_same_speakers: bool = True,
    ) -> List[SpeakerSegment]:
    """Assign speakers for to ASR result based on diarization result"""
    dia_segments_size = len(speakersegments) - 1
    i = 0
    durations = defaultdict(float)
    diarized_segs = []
    for asr_data in asr_segments:
        asr_seg = asr_data.segment
        while i <= dia_segments_size:
            dia_seg = speakersegments[i].segment
            speaker = speakersegments[i].speaker
            if asr_seg.end < dia_seg.start:  # run out of the target segment
                break
            # calc overlap duration of asr and diarization
            duration = (dia_seg & asr_seg).duration
            if duration > 0.0:
                durations[speaker] += (dia_seg & asr_seg).duration
            i += 1
        # assign the speaker who have longest overlap in each segment
        diarized_segs.append(SpeakerSegment(
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


def write_result(
        outfilename: str,
        speakersegments: List[SpeakerSegment],
        with_speakers: bool = False,
        timestamp_offset: Union[int, float, str] = 0.0
        ) -> tuple[str, ...]:
    """ write results to text files """
    _write_func = (
        write_asr_result_with_speakers if with_speakers
        else write_asr_result
    )
    return _write_func(
        outfilename,
        speakersegments,
        timestamp_offset
    )


def write_asr_result(
        basename: str,
        speakersegments: List[SpeakerSegment],
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write ASR segments to files with and without timestamps.
       Returns filenames.
    """
    outfilenames = (f"{basename}.txt", f"{basename}_timestamped.txt")
    fh_text, fh_timestamped = [
        open(filename, "w", encoding="utf-8") for filename in outfilenames
    ]
    for data in speakersegments:
        fh_text.write(data.segment.text + "\n")
        fh_timestamped.write(add_timestamp(data.segment, timestamp_offset) + "\n")
    fh_text.close()
    fh_timestamped.close()
    return outfilenames


def write_asr_result_with_speakers(
        basename: str,
        speakersegments: List[SpeakerSegment],
        timestamp_offset: Optional[Union[int, float, str]] = 0.0
        ) -> tuple[str, ...]:
    """Write diarized transcpript to a text file."""
    outfilename = f"{basename}_diarized.txt"
    fh = open(outfilename, "w", encoding="utf-8")
    for data in speakersegments:
        fh.write(time_segment_text(data.segment, timestamp_offset) + " ")
        if data.speaker:
            fh.write(data.speaker + "\n")
        else:
            fh.write("\n")
        fh.write(data.segment.text.replace(" ", "") + "\n\n")
    fh.close()
    return (outfilename,)
