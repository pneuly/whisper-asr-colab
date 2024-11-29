import sys
import os
import time
import subprocess
import ffmpeg
from typing import Union, Optional
from .utils import sanitize_filename

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
        start_time: Union[str, int, float] = "",
        end_time: Union[str, int, float] = ""
    ):
    start_time = str(start_time)
    end_time = str(end_time)
    if start_time and end_time:
        input = ffmpeg.input(audiopath, ss=start_time, to=end_time)
    elif not start_time and end_time:
        input = ffmpeg.input(audiopath, to=end_time)
    else:
        input = ffmpeg.input(audiopath, ss=start_time)
    input_base, input_ext = os.path.splitext(audiopath)
    input_path = f"{input_base}_{sanitize_filename(start_time)}_{sanitize_filename(end_time)}{input_ext}"
    print(f"Trimming audio from {start_time} to {end_time}.")
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
    ) -> subprocess.Popen:
    """Read audio file with resampling and time duration
    Args:
        file (str): Audio file path
        sr (int, optional): Sampling rate. Defaults to 16000.
        format (str, optional): Output format. Defaults to "wav".
        start_time (Optional[Union[int, float, str]], optional): Audio start time. Defaults to None.
        end_time (Optional[Union[int, float, str]], optional): Audio end time. Defaults to None.
    Returns:
        subprocess.Popen
    """
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
            #"-loglevel", "quiet",
        ]
        if start_time:
            cmd.extend(["-ss", str(start_time)])
        if end_time:
            cmd.extend(["-to", str(end_time)])
        cmd.append("-")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


def open_stream(url: str) -> subprocess.Popen:
    command = ["yt-dlp", "-g", url, "-x", "-S", "+acodec:mp4a"]
    audio_url = subprocess.check_output(command).decode("utf-8").strip()
    return read_audio(file=audio_url)


def get_silence_duration(audio_file) -> float:
    """get silence duration at the top of the audio"""
    output = subprocess.run(
        [
            "ffmpeg",
            "-i", audio_file,
            "-af", "silencedetect=noise=-50dB:d=5",
            "-f", "null", "-"
        ],
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    result = []
    silence_duration = 0.0
    for line in output.stderr.splitlines():
        if "silencedetect" in line:
            result.append(line)
    if len(result) > 0 and "silence_start: 0" in result[0]:
        silence_duration = float(result[1].split()[-1])
    return silence_duration


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
