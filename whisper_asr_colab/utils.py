import sys
import re
from typing import Tuple

def str2seconds(time_str: str) -> float:
    """Convert a time string (hh:mm:ss) to seconds."""
    if not time_str:
        return 0.0
    parts = time_str.split(':')
    return sum(
        float(x) * 60 ** i for i, x
        in enumerate(parts[::-1]))


def seconds_to_tuple(seconds: float) -> Tuple[int, int, float]:
    _h = int(seconds // 3600)
    _remain = seconds - (_h * 3600)
    _m = int(_remain // 60)
    _s = _remain - (_m * 60)
    return _h, _m, _s


def format_timestamp(seconds: float, sec_format: str = "05.2f") -> str:
    """Format seconds into a string 'H:MM:SS.ss'."""
    _h, _m ,_s = seconds_to_tuple(seconds)
    return f"{_h:01}:{_m:02}:{_s:{sec_format}}"


def download_from_colab(filepath: str):
    """Download a file in Google Colab environment."""
    #if str(get_ipython()).startswith("<google.colab."):
    if "google.colab" in sys.modules:
        sys.modules["google.colab"].files.download(filepath)

def sanitize_filename(filename, replacement="_"):
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    return re.sub(invalid_chars, replacement, filename)
