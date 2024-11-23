import sys
import os
import time
import subprocess
import ffmpeg
from typing import Union, Optional
from numpy import ndarray, frombuffer as np_frombuffer, int16 as np_int16, float32 as np_float32


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

def load_audio(
        file: str,
        sr: int = 16000,
        start_time: Optional[Union[int, float, str]] = None,
        end_time: Optional[Union[int, float, str]] = None
    ) -> ndarray:
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
        ]
        if start_time:
            cmd.extend(["-ss", str(start_time)])
        if end_time:
            cmd.extend(["-to", str(end_time)])
        cmd.append("-")
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np_frombuffer(out, np_int16).flatten().astype(np_float32) / 32768.0


def open_stream(url: str) -> subprocess.Popen:
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
