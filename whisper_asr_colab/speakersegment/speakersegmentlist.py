from __future__ import annotations
from json import dump as jsondump, load as jsonload
from itertools import groupby
from collections import defaultdict
from typing import Union, Optional, DefaultDict, List
from faster_whisper.transcribe import Segment
from logging import getLogger
from .speakersegment import SpeakerSegment

logger = getLogger(__name__)

class SpeakerSegmentList(List[SpeakerSegment]):

    def combine(self) -> SpeakerSegment:
        """Combine list of SpeakerSegment into a single SpeakerSegment"""
        head_seg = self[0]
        last_seg = self[-1]
        combined_underlying_seg = Segment(
            id=head_seg.segment.id,
            seek=head_seg.segment.seek,
            start=head_seg.segment.start,
            end=last_seg.segment.end,
            text="\n".join(s.segment.text for s in self).strip(),
            tokens=[],
            temperature=head_seg.segment.temperature,
            avg_logprob=head_seg.segment.avg_logprob,
            compression_ratio=head_seg.segment.compression_ratio,
            no_speech_prob=head_seg.segment.no_speech_prob,
            words=None,
        )
        return SpeakerSegment(
            segment=combined_underlying_seg,
            speaker=head_seg.speaker,
        )


    def combine_same_speakers(self) -> None:
        """Combine consecutive segments that have same speaker into a segment"""
        speakersegs = SpeakerSegmentList()
        for _, g in groupby(self, lambda x: x.speaker):
            #speakersegs.append(combine(list(g)))
            group_list = SpeakerSegmentList(g)
            speakersegs.append(group_list.combine())
        return speakersegs


    def fill_missing_speakers(self) -> None:
        """If segment.speaker is None, fill with the speaker of the previous segment"""
        for i in range(1, len(self)):
            if self[i].speaker is None:
                self[i].speaker = self[i - 1].speaker

    def assign_speakers(
        self,
        diarization_result: SpeakerSegmentList,
    ) -> SpeakerSegmentList:
        """Assign speakers for to ASR result based on diarization result"""
        dia_segments_size = len(diarization_result) - 1
        i = 0
        durations: DefaultDict[str, float] = defaultdict(float)
        diarized_segs = SpeakerSegmentList()
        for asr_seg in self:
            while i <= dia_segments_size:
                dia_seg = diarization_result[i]
                if asr_seg.segment.end < dia_seg.segment.start:
                    break
                start = max(asr_seg.segment.start, dia_seg.segment.start)
                end = min(asr_seg.segment.end, dia_seg.segment.end)
                duration = end - start
                if dia_seg.speaker and duration > 0.0:
                    durations[dia_seg.speaker] += duration
                i += 1

            if durations:
                asr_seg.speaker = max(durations, key=durations.get)
            diarized_segs.append(asr_seg)
            durations.clear()
            if i > 0:
                i -= 1
        return diarized_segs


    def save(self, jsonfile: str):
        with open(jsonfile, "w", encoding="utf-8") as fh:
            jsondump([seg.to_dict() for seg in self], fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, jsonfile: str) -> SpeakerSegmentList:
        with open(jsonfile, "r", encoding="utf-8") as fh:
            data_list = jsonload(fh)
        return cls(SpeakerSegment.from_dict(d) for d in data_list)


    def write_asr_result(
        self,
        basename: str,
        timestamp_offset: Optional[Union[int, float, str]] = 0.0,
    ) -> dict[str, str]:
        asr_result_file = f"{basename}.txt"
        asr_result_file_timestamped = f"{basename}_timestamped.txt"
        with open(asr_result_file, "w", encoding="utf-8") as fh_text, \
             open(asr_result_file_timestamped, "w", encoding="utf-8") as fh_timestamped:
            for speaker_seg in self:
                fh_text.write(speaker_seg.to_str(with_timestamp=False, with_speaker=False) + "\n")
                fh_timestamped.write(speaker_seg.to_str(timestamp_offset, with_speaker=False) + "\n")
        return {"asr_result_file" : asr_result_file,
                "asr_result_file_timestamped" : asr_result_file_timestamped}


    def write_integrated_result(
        self,
        basename: str,
        timestamp_offset: Optional[Union[int, float, str]] = 0.0,
    ) -> str:
        outfilename = f"{basename}_integrated.txt"
        with open(outfilename, "w", encoding="utf-8") as fh:
            for speaker_seg in self:
                fh.write(speaker_seg.to_str(timestamp_offset) + "\n\n")
        return outfilename
