import time
import random
from collections import defaultdict
import numpy as np
from pandas import DataFrame
from typing import List, Union, NamedTuple, Optional
#from whisper_asr_colab.diarize import DiarizedSegment
#from faster_whisper.transcribe import Segment
from pyannote.core import Segment as SimpleSegment

class Annotation(NamedTuple):
    segment: Optional[SimpleSegment]
    speaker: Optional[str]

def original_assign_speakers( #use pandas DataFrame
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    diarize_df = DataFrame(dia_segments, columns=['segment', 'speaker'])
    diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

    def _get_speaker(start: float, end: float):# -> tuple[Optional[str], Optional[str]]:
        diarize_df['intersection'] = np.minimum(diarize_df['end'], end) - np.maximum(diarize_df['start'], start)
        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        if dia_tmp.empty:
            return None, None
        result = dia_tmp.groupby("speaker")["intersection"].sum()
        return result.idxmax(), result.max() #speaker, intersection

    diarized_segs = []
    for seg in asr_segments:
        speaker, _ = _get_speaker(seg.start, seg.end)
        diarized_segs.append(Annotation(seg, speaker))
    return diarized_segs

def optimized_assign_speakers( #not using pandas DataFrame
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    #diarize_start, diarize_end, diarize_speakers = np.array(
    #    [(item[0].start, item[0].end, item[1]) for item in dia_segments], dtype=object).T
    diarize_start = np.array([item[0].start for item in dia_segments])
    diarize_end = np.array([item[0].end for item in dia_segments])
    diarize_speakers = np.array([item[1] for item in dia_segments])
    #@profile
    def _get_speaker(start: float, end: float) -> Optional[str]:
        intersection = np.minimum(diarize_end, end) - np.maximum(diarize_start, start)
        # Filter to only valid intersections
        valid_intersection = intersection > 0
        speaker_intersections = intersection[valid_intersection]
        if len(speaker_intersections) > 0:
            speakers = diarize_speakers[valid_intersection]

            # Sum the intersections by speaker
            speaker_overlap = defaultdict(float)
            for speaker, overlap in zip(speakers, speaker_intersections):
                speaker_overlap[speaker] += overlap
            # Select the speaker with the largest total overlap
            speaker = max(speaker_overlap, key=speaker_overlap.get).item()
        else:
            speaker = None
        return speaker

    diarized_segs = [
        Annotation(seg, _get_speaker(seg.start, seg.end))
        for seg in asr_segments
    ]
    return diarized_segs

def optimized_assign_speakers2( #similar to the opt1 but using np.unique
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    # Precompute start, end, and speaker arrays from diarization DataFrame
    diarize_start = np.array([item[0].start for item in dia_segments])
    diarize_end = np.array([item[0].end for item in dia_segments])
    diarize_speakers = np.array([item[1] for item in dia_segments])
    #@profile
    def _get_speaker(start: float, end: float) -> Optional[str]:
        intersection = np.minimum(diarize_end, end) - np.maximum(diarize_start, start)
        # Filter to only valid intersections
        valid_intersection = intersection > 0
        speaker_intersections = intersection[valid_intersection]
        if len(speaker_intersections) > 0:
            speakers = diarize_speakers[valid_intersection]
            # Sum the intersections by speaker
            # Elegant way. But slow
            unique_speakers, indices = np.unique(speakers, return_inverse=True)
            summed_intersections = np.bincount(indices, weights=speaker_intersections)
            max_index = np.argmax(summed_intersections)
            speaker = unique_speakers[max_index].item()
        else:
            speaker = None
        return speaker

    asr_start = np.array([seg.start for seg in asr_segments], dtype=float)
    asr_end = np.array([seg.end for seg in asr_segments], dtype=float)

    # Compute speakers for all ASR segments
    asr_speakers = np.array([
        _get_speaker(start, end)
        for start, end in zip(asr_start, asr_end)
    ])

    # Create DiarizedSegment instances with assigned speakers
    diarized_segs = [
        Annotation(SimpleSegment(start, end), speaker)
        for start, end, speaker in zip(asr_start, asr_end, asr_speakers)
    ]

    return diarized_segs

def optimized_assign_speakers3( #optimized for loop
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    dia_segments_size = len(dia_segments) - 1
    i = 0
    durations = defaultdict(float)
    diarized_segs = []
    #@profile
    for asr_seg in asr_segments:
        while i <= dia_segments_size:
            dia_seg, speaker = dia_segments[i]
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
    return diarized_segs


def optimized_assign_speakers4( #similar to opt2 but shrinks asr list
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    np_dia = np.array([
        [item[0].start for item in dia_segments], #start
        [item[0].end for item in dia_segments], #end
    ])
    diarize_speakers = np.array([item[1] for item in dia_segments])

    diarized_segs = []
    #@profile
    def _get_speaker(start: float, end: float) -> tuple[Optional[str], int]:
        max_idx = 0
        intersection = np.minimum(np_dia[1], end) - np.maximum(np_dia[0], start)
        valid_intersection = intersection > 0
        #print(valid_intersection)
        speaker_intersections = intersection[valid_intersection]
        if len(speaker_intersections) > 0:
            speakers = diarize_speakers[valid_intersection]
            # Sum the intersections by speaker
            speaker_overlap = defaultdict(float)
            for speaker, overlap in zip(speakers, speaker_intersections):
                speaker_overlap[speaker] += overlap
            # Select the speaker with the largest total overlap
            speaker = max(speaker_overlap, key=speaker_overlap.get)
            max_idx = np.where(valid_intersection)[0].max()
            #print(max_idx)
            #print(len(diarize_start))
        else:
            speaker = None
        return speaker, max_idx

    for seg in asr_segments:
        speaker, max_idx = _get_speaker(seg.start, seg.end)
        diarized_segs.append(Annotation(seg, speaker))
        #Shrink
        np_dia = np_dia[:,max_idx:]
        diarize_speakers = diarize_speakers[max_idx:]

    return diarized_segs

if __name__ == "__main__":
    # Create test data for benchmarking
    asr_mutiplyer = 2
    dia_segments_size = 500
    asr_segments_size = dia_segments_size * asr_mutiplyer
    dia_max_duration = 10
    asr_max_duration = dia_max_duration / asr_mutiplyer

    # Random segments
    def segments_generator(size:int, min_duration=0.5, max_duration=10.0)->List[SimpleSegment]:
        segments = []
        start_time = 0.0
        for i in range(size):
            duration = random.uniform(0.5, 10.0)
            segment = SimpleSegment(
                        start_time,
                        start_time + duration,
            )
            start_time += duration
            segments.append(segment)
        return segments

    # Random annotations
    def annotations_generator(segments:list, max_speakers:int=5)->List[Annotation]:
        return[
            Annotation(segment, f"SPEAKER{random.randint(0,max_speakers):02}")
            for segment in segments]

    asr_segments = segments_generator(asr_segments_size, max_duration=asr_max_duration)
    dia_segments = annotations_generator(
        segments_generator(dia_segments_size, max_duration=dia_max_duration))


    # Benchmark functions
    def bench(funcname: str):
        start_time = time.time()
        result = globals()[funcname](dia_segments, asr_segments)
        duration = time.time() - start_time
        return duration, result

    fnames = [
        "original_assign_speakers",
        "optimized_assign_speakers",
        "optimized_assign_speakers2",
        "optimized_assign_speakers3",
        #"optimized_assign_speakers4",
        ]

    resultlist = [(funcname, *bench(funcname),) for funcname in fnames]
    for funcname, duration, result in resultlist:
        print(f"{funcname}: {duration}")
        print(resultlist[0][2] == result)
        #print(result)

        for i, (a, b) in enumerate(zip(resultlist[0][2], result)):
            if a != b:
                print(f"Index {i}:\n{a}!=\n{b}\n")
            #else:
            #    print(f"Index {i}:\n{a}=\n{b}\n")


