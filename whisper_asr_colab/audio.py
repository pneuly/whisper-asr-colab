import sys
import os
import time
import subprocess
import ffmpeg
import torchaudio
from io import BytesIO
from typing import Union, Optional
from numpy import frombuffer as np_frombuffer, int16 as np_int16, float32 as np_float32


def dl_audio(url: str, password: str = ""):
    """Download file from Internet"""
    # YoutubeDL class causes download errors, using external command instead
    options = ["-x", "-S", "+acodec:mp4a", "-o", "%(title)s.%(ext)s"]
    if password:
        options += ["--video-password", password]
    outfilename = subprocess.run(
        ["yt-dlp", "--print", "filename"] + options + [url],
        capture_output=True,
        text=True,
        encoding="utf-8"
    ).stdout.strip()
    subprocess_progress(["yt-dlp"] + options + [url])
    return outfilename


def trim_audio(
        audiopath: str,
        start_time: str = "",
        end_time: str = ""
    ):
    if start_time and end_time:
        input = ffmpeg.input(audiopath, ss=start_time, to=end_time)
    elif not start_time and end_time:
        input = ffmpeg.input(audiopath, to=end_time)
    else:
        input = ffmpeg.input(audiopath, ss=start_time)
    input_base, input_ext = os.path.splitext(audiopath)
    input_path = f"{input_base}_trimmed{input_ext}"
    print(f"trimming audio from {start_time} to {end_time}.")
    ffmpeg.output(input, input_path, acodec="copy", vcodec="copy").run(
            overwrite_output=True
            )
    return input_path

def read_audio(
        file: str,
        sr: int = 16000,
        format: str = "wav", # output format
        start_time: Optional[Union[int, float, str]] = None,
        end_time: Optional[Union[int, float, str]] = None
):
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", format,
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
        ]
        if start_time:
            cmd.extend(["-ss", str(start_time)])
        if end_time:
            cmd.extend(["-to", str(end_time)])
        cmd.append("-")
        return subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


def open_live_stream(url: str) -> subprocess.Popen:
    command = ["yt-dlp", "-g", url, "-x", "-S", "+acodec:mp4a"]
    audio_url = subprocess.check_output(command).decode("utf-8").strip()
    return subprocess.Popen(
        [
            "ffmpeg",
            "-i", audio_url,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-",
            "-loglevel", "quiet"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def load_audio(
        file: str,
        sr: int = 16000,
        start_time: Optional[Union[int, float, str]] = None,
        end_time: Optional[Union[int, float, str]] = None,
        data_type: str = "file-like"
    ):
    if data_type == "numpy":
        format = "s16le"
    elif data_type == "numpy" or "file-like":
        format = "wav"
    else:
        raise ValueError("data_type has to be 'numpy', 'torch', or 'file-like'")
    stream = read_audio(
        file=file,
        sr=sr,
        format=format,
        start_time=start_time,
        end_time=end_time
    )
    if data_type == "numpy":
        return np_frombuffer(stream, np_int16).flatten().astype(np_float32) / 32768.0
    elif data_type == "torch":
        out, sr = torchaudio.load(stream, format=format)
        print(f"sampling rate: {sr}")
        return out
    elif data_type == "file-like":
        return BytesIO(stream)


def subprocess_progress(cmd: list):
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
                sys.stdout.write(buf)
                sys.stdout.flush()
            if p.poll() is not None:
                break
            time.sleep(0.5)
