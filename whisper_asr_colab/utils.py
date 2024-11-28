from datetime import datetime
import silero_vad
import gc
from .audio import load_audio

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


def download_from_colab(filepath: str):
    """Download a file in Google Colab environment."""
    if str(get_ipython()).startswith("<google.colab."):
        from google.colab import files
        files.download(filepath)

def get_speech_timestamps(
        audio,
        model=None,
        min_silence_duration_ms=2000,
        return_seconds=True,
    ):

    wav = load_audio(audio, data_type='torch')
    wav = wav.squeeze(0)

    if model is None:
        model = silero_vad.load_silero_vad(onnx=True)
    speech_timestamps = silero_vad.get_speech_timestamps(
        audio=wav,
        model=model,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=return_seconds,
    )

    wav.detach()
    del model, wav
    gc.collect()
    return speech_timestamps
