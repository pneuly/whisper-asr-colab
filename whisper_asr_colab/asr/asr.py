from logging import getLogger, DEBUG
from typing import Union, Optional, BinaryIO, List, Any
from faster_whisper import BatchedInferencePipeline, WhisperModel as FasterWhisperModel
from whisper_asr_colab.common.speakersegment import SpeakerSegment
from whisper_asr_colab.common.utils import format_timestamp

logger = getLogger(__name__)

def faster_whisper_transcribe(
        audio: Union[str, BinaryIO, "numpy.ndarray"],
        model: Optional[FasterWhisperModel] = None,
        **transcribe_args: Any
    ) -> tuple[List[SpeakerSegment], Any]:
    
    batch_size = transcribe_args.pop("batch_size", 1)
    logger.debug(f"batch_size: {batch_size}")

    model = model or FasterWhisperModel(
                        "large-v3-turbo",
                        device="auto",
                        compute_type="default",
                        )

    if batch_size > 1: # batch mode
        batched_model = BatchedInferencePipeline(model=model)
        segments_generator, info = batched_model.transcribe(
            audio=audio,
            **transcribe_args,
        )
    else: # sequential mode
        logger.info(f"batch_size is set to less than 2 (batch_size={batch_size}). Using sequential mode.")
        transcribe_args["condition_on_previous_text"] = False
        transcribe_args["without_timestamps"] = False
        segments_generator, info = model.transcribe(
            audio=audio,
            **transcribe_args,
        )
    segments = []
    with open("diarization_progress.txt", "w", encoding="utf-8", buffering=1) as f:
        for segment in segments_generator:
            segment_text = f"[{format_timestamp(segment.start, '02.0f')} - {format_timestamp(segment.end, '02.0f')}] {segment.text}"
            #print(segment_text)
            f.write(segment_text + "\n")
            segments.append(SpeakerSegment.from_segment(segment))
    if logger.isEnabledFor(DEBUG):
        logger.debug(f"Transcribed segments:\n{segments}")
    return segments, info
