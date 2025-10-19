from dataclasses import asdict
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, BinaryIO, Optional, Union

from faster_whisper import BatchedInferencePipeline
from faster_whisper import WhisperModel as FasterWhisperModel

from whisper_asr_colab.speakersegment import SpeakerSegment, SpeakerSegmentList

if TYPE_CHECKING:
    import numpy

ASR_PROGRESS_FILE = "asr_progress.txt"

logger = getLogger(__name__)


def faster_whisper_transcribe(
    audio: Union[str, BinaryIO, "numpy.ndarray"],
    model: Optional[FasterWhisperModel] = None,
    **transcribe_args: Any,
) -> tuple[SpeakerSegmentList, Any]:
    batch_size = transcribe_args.pop("batch_size", 1)
    logger.debug(f"batch_size: {batch_size}")

    model = model or FasterWhisperModel(
        "large-v3-turbo",
        device="auto",
        compute_type="default",
    )

    if batch_size > 1:  # batch mode
        batched_model = BatchedInferencePipeline(model=model)
        segments_generator, info = batched_model.transcribe(
            audio=audio,
            **transcribe_args,
        )
    else:  # sequential mode
        logger.info(
            "batch_size is set to less than 2 (batch_size=%d). Using sequential mode."
            % batch_size
        )
        transcribe_args["condition_on_previous_text"] = False
        transcribe_args["without_timestamps"] = False
        segments_generator, info = model.transcribe(
            audio=audio,
            **transcribe_args,
        )
    segments = SpeakerSegmentList()
    with open(ASR_PROGRESS_FILE, "w", encoding="utf-8", buffering=1) as f:
        for segment in segments_generator:
            speaker_seg = SpeakerSegment(**asdict(segment))
            segment_text = speaker_seg.to_str(with_speaker=False)
            # print(segment_text)
            f.write(segment_text + "\n")
            segments.append(speaker_seg)
    if logger.isEnabledFor(DEBUG):
        logger.debug(f"Transcribed segments:\n{segments}")
    return segments, info
