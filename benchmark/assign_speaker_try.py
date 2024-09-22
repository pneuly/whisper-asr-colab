import time
import random
from collections import defaultdict
import numpy as np
from typing import List, Union, NamedTuple, Optional
#from whisper_asr_colab.diarize import DiarizedSegment
#from faster_whisper.transcribe import Segment
from pyannote.core import Segment as SimpleSegment
from line_profiler import LineProfiler

prf = LineProfiler()

class Annotation(NamedTuple):
    segment: Optional[SimpleSegment]
    speaker: Optional[str]

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
        valid_intersection = intersection > 0
        speaker_intersections = intersection[valid_intersection]
        if len(speaker_intersections) > 0:
            speakers = diarize_speakers[valid_intersection]
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
        return speaker.item() if speaker is not None else None

    diarized_segs = [
        Annotation(seg, _get_speaker(seg.start, seg.end))
        for seg in asr_segments
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
        #print(f"loop {i}")
        while i <= dia_segments_size:
            dia_seg, speaker = dia_segments[i]
            if asr_seg.end < dia_seg.start: # run out of the target segment
                #print(f"{i} break")
                break
            # intersected duration
            duration = (dia_seg & asr_seg).duration
            #print(f"{i} duration {duration}")
            if duration > 0.0:
                durations[speaker] += duration
                #print(f"{i} intersect {speaker} {duration}")
            #else:
            #    print(f"{i} skip {duration}")
            i += 1
        diarized_segs.append(Annotation(
            asr_seg,
            max(durations, key=durations.get, default=None)
            ))
        #print(f"appended {diarized_segs[-1][0]} {diarized_segs[-1][1]}")
        durations.clear()
        i -= 1
    return diarized_segs

prf.add_function(optimized_assign_speakers3)

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

if __name__ == "__main__":
    def main():
        # Create test data for benchmarking
        asr_mutiplyer = 2
        dia_segments_size = 500
        asr_segments_size = dia_segments_size * asr_mutiplyer
        dia_max_duration = 10
        asr_max_duration = dia_max_duration / asr_mutiplyer

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
            "optimized_assign_speakers",
            "optimized_assign_speakers3",
            ]

        resultlist = [(funcname, *bench(funcname),) for funcname in fnames]
        for funcname, duration, result in resultlist:
            print(f"{funcname}: {duration}")
            if funcname == "optimized_assign_speakers3":
                print(resultlist[0][2] == result)
            #print(result)
            if funcname == "optimized_assign_speakers3":
                for i, (a, b) in enumerate(zip(resultlist[0][2], result)):
                    if a != b:
                        print(f"Index {i}:\n{a[0]} {a[1]}!=\n{b[0]} {b[1]}\n")
                    #else:
                    #    print(f"Index {i}:\n{a}=\n{b}\n")
        #for i, seg in enumerate(asr_segments):
        #    print(f"{i} {seg}")
        #print("")
        #or i, (seg, speaker) in enumerate(dia_segments):
        #    print(f"{i} {seg} {speaker}")

    prf.runcall(main)
    prf.print_stats()