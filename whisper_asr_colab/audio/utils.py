import sys
import os
import time
import logging
import subprocess
import numpy as np
import ffmpeg
from typing import Optional, Tuple, Callable
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

logger = logging.getLogger(__name__)


def decode_audio(audio, sampling_rate=16000) -> np.ndarray:
    """An alternative to faster_whisper.decode_audio(),
    addressing its high memory consumption."""
    _stdout = decode_audio_pipe(audio, sampling_rate).stdout
    if _stdout is None:
        raise ValueError(f"Cannot decode audio {audio}")
    return (
        np.frombuffer(_stdout.read(), np.int16).flatten().astype(np.float32) / 32768.0
    )


def decode_audio_pipe(audio: str, sampling_rate: int = 16000):
    """Returns audio as Popen instance.
    Audio can be internet url (any uri that ffmpeg can parse)."""
    return subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            audio,
            "-vn",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "-loglevel",
            "quiet",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def dl_audio(
    url: str, audio_format: Optional[str] = None, password: Optional[str] = None
):
    """Download file from Internet"""
    logger.info(f"Downloading audio from {url}")
    ydl_opts = {
        "format": "140/bestaudio/best",
        "outtmpl": "%(title)s.%(ext)s",
        "quiet": False,
        "noplaylist": True,
    }
    if audio_format:
        ydl_opts["format"] = audio_format
    if password:
        ydl_opts["videopassword"] = password
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            outfilename = ydl.prepare_filename(info)
            return outfilename
        except DownloadError as e:
            print(f"yt-dlp options used: {ydl_opts}")
            raise e


def is_uploading(
    file: str,
    wait_func: Callable = lambda x=10: time.sleep(x),
) -> Tuple[bool, float, float, float]:
    """There is no direct way to detect if the file is still uploading on Google Colab.
    So, detecting file size increase is used as a workaround."""
    filesize = os.path.getsize(file)
    wait_start = time.time()
    wait_func()
    wait_time = time.time() - wait_start
    filesize2 = os.path.getsize(file)
    if (filesize2 - filesize) > 0:
        logger.info(f"File {file} is still uploading.")
        return True, filesize, filesize2, wait_time
    return False, filesize, filesize2, wait_time


def convert_audio(input_file: str, ext: str, overwrite: bool = True):
    """Convert audio file to the specified format using ffmpeg."""
    if not input_file.endswith(ext):
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.{ext}"
        ffmpeg.run(
            ffmpeg.input(input_file).output(output_file), overwrite_output=overwrite
        )
        return output_file
    print("No conversion is needed.")
    return input_file


def subprocess_progress(cmd: list):
    # TODO: move to utils
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False
    )
    if os.name != "nt":
        import fcntl

        flag = fcntl.fcntl(p.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(p.stdout.fileno(), fcntl.F_SETFL, flag | os.O_NONBLOCK)
        while True:
            buf = p.stdout.read()
            if buf is not None:
                sys.stdout.write(buf.decode("utf-8"))
                sys.stdout.flush()
            if p.poll() is not None:
                break
            time.sleep(0.5)


## Depricated
def get_silence_duration(audio_file) -> float:
    """get silence duration at the top of the audio"""
    output = subprocess.run(
        [
            "ffmpeg",
            "-i",
            audio_file,
            "-af",
            "silencedetect=noise=-50dB:d=5",
            "-f",
            "null",
            "-",
        ],
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    result = []
    silence_duration = 0.0
    for line in output.stderr.splitlines():
        if "silencedetect" in line:
            result.append(line)
    if len(result) > 0 and "silence_start: 0" in result[0]:
        silence_duration = float(result[1].split()[-1])
    return silence_duration
