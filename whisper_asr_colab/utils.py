from datetime import datetime
import re

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

def sanitize_filename(filename, replacement="_"):
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    return re.sub(invalid_chars, replacement, filename)