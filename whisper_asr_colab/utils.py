import sys
import re

def str2seconds(time_str: str) -> float:
    """Convert a time string (hh:mm:ss) to seconds."""
    if not time_str:
        return 0.0
    parts = time_str.split(':')
    return sum(
        float(x) * 60 ** i for i, x
        in enumerate(parts[::-1]))


def format_timestamp(seconds: float) -> str:
    """Format seconds into a string 'H:MM:SS.ss'."""
    hours = seconds // 3600
    remain = seconds - (hours * 3600)
    minutes = remain // 60
    seconds = remain - (minutes * 60)
    return "{:01}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)


def download_from_colab(filepath: str):
    """Download a file in Google Colab environment."""
    #if str(get_ipython()).startswith("<google.colab."):
    if "google.colab" in sys.modules:
        sys.modules["google.colab"].files.download(filepath)

def sanitize_filename(filename, replacement="_"):
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    return re.sub(invalid_chars, replacement, filename)
