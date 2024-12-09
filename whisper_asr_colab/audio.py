import sys
import re
import os
import time
import warnings
import subprocess
import ffmpeg
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass
from .utils import sanitize_filename, str2seconds


@dataclass
class Audio:
    url: Optional[str] = None
    password: Optional[str] = None
    file_path: Optional[str] = None
    sampling_rate: int = 16000
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    verify_upload: bool = True
    _rawdata: Optional[np.ndarray] = None


    @property
    def ndarray(self) -> np.ndarray:
        if self._rawdata is None:
            _rawdata = self._load_audio()
            return _rawdata[self.start_frame:self.end_frame]
        else:
            return self._rawdata[self.start_frame:self.end_frame]

    def _load_audio(self) -> np.ndarray:
        """Load audio to self._rawdata based on self.url or self.file_path.
        Returns loaded audio data"""
        if self.file_path is None and self.url is None:
            raise ValueError("No url or file path set.")
        if self.file_path is None or not os.path.exists(self.file_path):
            self.file_path = dl_audio(self.url, self.password) # type: ignore
        print(f"Loading audio file {self.file_path}")
        if self.verify_upload and not is_upload_complete(self.file_path):
            message = f"Uploaing {self.file_path} seems incomplete. Run again after the upload is finished."
            if 'IPython' in sys.modules:
                display = sys.modules["IPython"].display
                display.display(display.Javascript(f'alert("{message}")'))
                display.display(display.HTML(f'<div style="color: red; font-size:large; font-weight: bold;">⚠️ {message}</div>'))
                warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")
                raise SystemExit
            else:
                sys.exit(message)
        _rawdata = decode_audio(self.file_path, self.sampling_rate)
        if not isinstance(_rawdata, np.ndarray):
            raise ValueError("Failed to get audio data. Check if url or file path is correctly set.")
        self._rawdata = _rawdata
        return self._rawdata

    ## TODO  def write_data()

    @property
    def live_stream(self) -> subprocess.Popen:
        if not self.url:
            raise ValueError("No url set.")
        command = ["yt-dlp", "-g", self.url, "-x", "-S", "+acodec:mp4a", "-q"]
        audio_url = subprocess.check_output(command).decode("utf-8").strip()
        return decode_audio_pipe(audio_url, 16000)


    @property
    def start_time(self) -> float:
        return (self.start_frame if self.start_frame else 0) / self.sampling_rate

    @start_time.setter
    def start_time(self, sec : Union[str, int, float]):
        if isinstance(sec, str):
            sec = str2seconds(sec)
        self.start_frame = int(sec * self.sampling_rate)

    @property
    def end_time(self) -> float:
        if isinstance(self._rawdata, np.ndarray):
            return (self.end_frame if self.end_frame else len(self._rawdata)) / self.sampling_rate
        else:
            return(0.0)

    @end_time.setter
    def end_time(self, sec : Union[str, int, float]):
        if isinstance(sec, str):
            sec = str2seconds(sec)
        self.end_frame = int(sec * self.sampling_rate)

    @staticmethod
    def from_path_or_url(source: str) -> "Audio":
        if re.match(r"^(https://).+", source):
            return Audio(url=source)
        else:
            return Audio(file_path=source)

    def set_silence_skip(
        self,
        threshold: float = 0.1,
        min_silence_duration: Union[int, float] = 5) -> Tuple[Union[int, None], Union[int, None]]:
        """Set start_frame and end_frame based on silence detection"""
        if not isinstance(self.ndarray, np.ndarray):
            raise ValueError("Cannot get self.ndarray as ndarray. Set url or file path to Audio instance.")
        audio_data = np.abs(self.ndarray)
        non_silent_indices = np.where(audio_data > threshold)[0]
        if len(non_silent_indices) == 0:
            print("Entire signal is silent!")
            return None, None
        leading = non_silent_indices[0]
        trailing = non_silent_indices[-1]
        min_frame_size = int(min_silence_duration * self.sampling_rate)
        if (not self.start_frame) and (leading > min_frame_size):
            print(f"Leading silence detected. Skipping {int(leading / self.sampling_rate)} seconds.")
            self.start_frame = int(leading)
        frame_size = len(self.ndarray)
        if (not self.end_frame) and (trailing < frame_size) and (trailing + min_frame_size < frame_size):
            trailing_sec = int((frame_size - trailing)/ self.sampling_rate)
            print(f"Trailing silence detected. Skipping the last {trailing_sec} seconds.")
            self.end_frame = int(trailing) + 1
        return self.start_frame, self.end_frame


    def get_time_slice(
            self,
            start_sec:Union[int, float, None] = None,
            end_sec:Union[int, float, None] = None) -> np.ndarray:
        sr = self.sampling_rate
        start_frame = int(start_sec * sr) if start_sec is not None else 0.0
        end_frame = int(end_sec * sr) if end_sec is not None else len(self.ndarray)
        if self._rawdata is None:
            _rawdata = self._load_audio()
            return _rawdata[start_frame:end_frame]
        else:
            return self._rawdata[start_frame:end_frame]


def decode_audio(audio, sampling_rate=16000) -> np.ndarray:
    """An alternative to faster_whisper.decode_audio(),
    addressing its high memory consumption."""
    _stdout = decode_audio_pipe(audio, sampling_rate).stdout
    if _stdout is None:
        raise ValueError(f"Cannot decode audio {audio}")
    return np.frombuffer(
        _stdout.read(),
        np.int16
        ).flatten().astype(np.float32) / 32768.0

def decode_audio_pipe(audio: str, sampling_rate: int = 16000):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-i", audio,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sampling_rate),
            "-loglevel", "quiet",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def dl_audio(url: str, password: Optional[str] = None):
    """Download file from Internet"""
    print(f"Downloading audio from {url}")
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


def is_upload_complete(file: str, threshold: int = 10000000) -> bool:
    """Check if the file upload is complete when the file size is
    below the threshold (bytes). Default threshold is 10MB"""
    filesize = os.path.getsize(file)
    if filesize < threshold:
        time.sleep(10)
        filesize2 = os.path.getsize(file)
        if (filesize2 - filesize) > 0:
        # File uploading seems incomplete
            return False
    return True


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


## Depricated
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
