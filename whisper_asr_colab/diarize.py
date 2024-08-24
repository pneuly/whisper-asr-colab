import torch
import whisperx


def fill_missing_speakers(segments):
    prev = None
    for item in segments:
        if 'speaker' in item:
            prev = item['speaker']
        else:
            item.update({'speaker' : prev})
    return segments


def combine_same_speaker(segments):
    from itertools import groupby
    segments = fill_missing_speakers(segments)
    _grouped = [
        list(g) for k, g in groupby(segments, lambda x: x["speaker"])
    ]
    _combined = [
        {"start" : segs[0]["start"],
         "end" : segs[-1]["end"],
         "text" : "\n".join([seg["text"] for seg in segs]).strip(),
         "speaker" : segs[0]["speaker"],
         } for segs in _grouped
    ]
    return _combined


def diarize(audio, asr_result, hugging_face_token):
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hugging_face_token,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    diarized_result = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarized_result, asr_result, fill_nearest=True)
    segments = [
        {k: v for k, v in d.items() if k != 'words'}
        for d in result["segments"]]

    return combine_same_speaker(segments)
