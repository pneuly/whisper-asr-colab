import time
import random
from collections import defaultdict
from typing import List, NamedTuple, Optional
from line_profiler import LineProfiler
from pyannote.core import Segment as SimpleSegment
prf = LineProfiler()
class Annotation(NamedTuple):
    segment: Optional[SimpleSegment]
    speaker: Optional[str]

def optimized_assign_speakers3(
        dia_segments: List[Annotation],
        asr_segments: List[SimpleSegment],
    ) -> List[Annotation]:

    dia_segments_size = len(dia_segments) - 1
    #@profile
    def _get_speaker(asr_segments: List[SimpleSegment]) :
        i = 0
        durations = defaultdict(float)
        for asr_seg in asr_segments:
            while True:
                print(i)
                if i > dia_segments_size: #reached the end of dia_segments
                    yield Annotation(dia_seg, max(durations, key=durations.get, default=None))
                    durations = defaultdict(float)
                    continue
                dia_seg = dia_segments[i][0]
                if dia_seg.end < asr_seg.start: ## fast forward
                    i += 1
                    continue
                #print(f"i:{i} start:{start} end:{end} {dia_segments[i-1][0]}")
                if dia_seg.start > asr_seg.end: # run out of the target segment
                    yield Annotation(dia_seg, max(durations, key=durations.get, default=None))
                    durations = defaultdict(float)
                    continue
                # intersection
                durations[dia_segments[i][1]] += (dia_seg & asr_seg).duration
                i += 1
    prf.add_function(_get_speaker)

    gen = _get_speaker(asr_segments)
    diarized_segs = [
        next(gen)
    ]
    return diarized_segs

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
    # Create test data for benchmarking
    asr_mutiplyer = 2
    dia_segments_size = 10
    asr_segments_size = dia_segments_size * asr_mutiplyer
    dia_max_duration = 10
    asr_max_duration = dia_max_duration / asr_mutiplyer

    asr_segments = segments_generator(asr_segments_size, max_duration=asr_max_duration)
    dia_segments = annotations_generator(
        segments_generator(dia_segments_size, max_duration=dia_max_duration))

    def test():
        optimized_assign_speakers3(dia_segments, asr_segments)

    prf.runcall(test)
    prf.print_stats()
