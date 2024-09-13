import time
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional

class DiarizedSegment(NamedTuple):
    start: float
    end: float
    speaker: Optional[str] = None
    intersection: Optional[float] = None

    @property
    def duration(self):
        return(self.end - self.start)

def original_assign_speakers(
        dia_segments: List[DiarizedSegment],
        asr_segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:

    diarize_df = pd.DataFrame(dia_segments, columns=['segment', 'speaker'])
    diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

    def _get_speaker(start: float, end: float) -> Optional[str]:
        diarize_df['intersection'] = np.minimum(diarize_df['end'], end) - np.maximum(diarize_df['start'], start)
        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        if dia_tmp.empty:
            return None, None
        result = dia_tmp.groupby("speaker")["intersection"].sum()
        return result.idxmax(), result.max()

    diarized_segs = []
    for seg in asr_segments:
        speaker, isec = _get_speaker(seg.start, seg.end)
        diarized_segs.append(DiarizedSegment(seg.start, seg.end, speaker))
    return diarized_segs

def optimized_assign_speakers(
        dia_segments: List[DiarizedSegment],
        asr_segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:

    #diarize_start, diarize_end, diarize_speakers = np.array(
    #    [(item[0].start, item[0].end, item[1]) for item in dia_segments], dtype=object).T
    diarize_start = np.array([item[0].start for item in dia_segments])
    diarize_end = np.array([item[0].end for item in dia_segments])
    diarize_speakers = np.array([item[1] for item in dia_segments])

    def _get_speaker(start: float, end: float) -> Optional[str]:
        intersection = np.minimum(diarize_end, end) - np.maximum(diarize_start, start)
        valid_intersection = intersection > 0
        if valid_intersection.any():
            # Filter to only valid intersections
            speaker_intersections = intersection[valid_intersection]
            speakers = diarize_speakers[valid_intersection]

            # Sum the intersections by speaker
            speaker_overlap = defaultdict(float)
            for speaker, overlap in zip(speakers, speaker_intersections):
                speaker_overlap[speaker] += overlap
            # Select the speaker with the largest total overlap
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            speaker = None
        return speaker

    diarized_segs = [
        DiarizedSegment(seg.start, seg.end, _get_speaker(seg.start, seg.end))
        for seg in asr_segments
    ]
    return diarized_segs

def optimized_assign_speakers2(
        dia_segments: List[DiarizedSegment],
        asr_segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:

    # Precompute start, end, and speaker arrays from diarization DataFrame
    diarize_start = np.array([item[0].start for item in dia_segments])
    diarize_end = np.array([item[0].end for item in dia_segments])
    diarize_speakers = np.array([item[1] for item in dia_segments])

    def _get_speaker(start: float, end: float) -> Optional[str]:
        intersection = np.minimum(diarize_end, end) - np.maximum(diarize_start, start)
        valid_intersection = intersection > 0
        if valid_intersection.any():
            # Filter to only valid intersections
            speaker_intersections = intersection[valid_intersection]
            speakers = diarize_speakers[valid_intersection]

            # Sum the intersections by speaker
            #speaker_overlap = defaultdict(lambda: 0.0)
            #for i, speaker in enumerate(speakers):
            #    speaker_overlap[speaker] += speaker_intersections[i]
            unique_speakers, indices = np.unique(speakers, return_inverse=True)
            summed_intersections = np.zeros(unique_speakers.size)
            np.add.at(summed_intersections, indices, speaker_intersections)

            # Select the speaker with the largest total overlap
            max_index = np.argmax(summed_intersections)
            speaker = unique_speakers[max_index]
            #speaker = max(speaker_overlap, key=speaker_overlap.get)
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
        DiarizedSegment(start, end, speaker)
        for start, end, speaker in zip(asr_start, asr_end, asr_speakers)
    ]

    return diarized_segs

def optimized_assign_speakers3(
        dia_segments: List[DiarizedSegment],
        asr_segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:

    starts = [item[0].start for item in dia_segments]
    ends = [item[0].end for item in dia_segments]
    speakers = [item[1] for item in dia_segments]
    dia_segments_size = len(dia_segments) - 1

    def _get_speaker(start: float, end: float) :
        i = 0
        durations = defaultdict(float)
        while True:
            if i > dia_segments_size: #reached the end of dia_segments
                yield max(durations, key=durations.get, default=None)
                durations = defaultdict(float)
                continue
            if ends[i] < start: ## fast forward
                i += 1
                continue
            #print(f"i:{i} start:{start} end:{end} {dia_segments[i-1][0]}")
            if starts[i] > end: # run out of the target segment
                yield max(durations, key=durations.get, default=None)
                durations = defaultdict(float)
                continue
            # intersection
            durations[speakers[i]] += min(end, ends[i]) - max(start, starts[i])
            i += 1


    diarized_segs = [
        DiarizedSegment(seg.start, seg.end, next(_get_speaker(seg.start, seg.end)))
        for seg in asr_segments
    ]
    return diarized_segs

def optimized_assign_speakers4(
        dia_segments: List[DiarizedSegment],
        asr_segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:

    diarize_start = np.array([item[0].start for item in dia_segments])
    diarize_end = np.array([item[0].end for item in dia_segments])
    diarize_speakers = np.array([item[1] for item in dia_segments])
    max_idx = 0

    def _get_speaker(start: float, end: float) -> Optional[str]:
        nonlocal max_idx
        intersection = np.minimum(diarize_end, end) - np.maximum(diarize_start, start)
        valid_intersection = intersection > 0
        max_idx = np.where(valid_intersection)[0].max()
        print(valid_intersection)
        if valid_intersection.any():
            # Filter to only valid intersections
            speaker_intersections = intersection[valid_intersection]
            speakers = diarize_speakers[valid_intersection]

            # Sum the intersections by speaker
            speaker_overlap = defaultdict(float)
            for speaker, overlap in zip(speakers, speaker_intersections):
                speaker_overlap[speaker] += overlap
            # Select the speaker with the largest total overlap
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            speaker = None
        return speaker

    diarized_segs = [
        DiarizedSegment(seg.start, seg.end, _get_speaker(seg.start, seg.end))
        for seg in asr_segments
    ]
    return diarized_segs

### main
# Create test data for benchmarking
num_diarized_segments = 10
asr_mutiplyer = 3
num_asr_segments = num_diarized_segments * asr_mutiplyer

# Random segments
def segment_generator(size, min_duration=0.5, max_duration=10.0, has_speaker=False):
    segments = []
    start_time = 0
    speaker = 1
    for i in range(num_diarized_segments):
        duration = random.uniform(0.5, 10.0)
        args = [start_time, start_time + duration,]
        segment = DiarizedSegment(
                    *args,
        )
        start_time += duration
        if has_speaker:
            segments.append((segment, speaker))
            speaker += 1
        else:
            segments.append(segment)
    return segments


dia_segments = segment_generator(num_diarized_segments, 0.5, 10.0, True)
asr_segments = segment_generator(num_diarized_segments, 0.5, 5.0, False)

# Benchmark original function
start_time = time.time()
original_result = original_assign_speakers(dia_segments, asr_segments)
original_duration = time.time() - start_time


# Benchmark optimized functions
def bench(funcname: str):
    start_time = time.time()
    result = globals()[funcname](dia_segments, asr_segments)
    duration = time.time() - start_time
    return duration, result

fnames = [
    #"original_assign_speakers",
    "optimized_assign_speakers",
    "optimized_assign_speakers2",
    "optimized_assign_speakers4",
    ]

resultlist = [(funcname, *bench(funcname),) for funcname in fnames]
for funcname, duration, result in resultlist:
    print(f"{funcname}: {duration}")
    print(resultlist[0][2] == result)

#print(f"original  : {original_duration}")
#print(f"optimized : {optimized_duration}")
#print(f"optimized2: {optimized_duration2}")
#print(f"optimized3: {optimized_duration3}")
#print(original_result == optimized_result)
#print(original_result == optimized_result2)
#print(original_result == optimized_result3)
#print(original_result)
#for i, (a, b) in enumerate(zip(original_result, optimized_result3)):
#    if a != b:
#        print(f"Index {i}:\n{a}!=\n{b}\n")

