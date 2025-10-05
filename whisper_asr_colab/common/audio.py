import sys
import re
import os
import time
import logging
import subprocess
import numpy as np
import ffmpeg
from typing import Union, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from yt_dlp import YoutubeDL
from .utils import str2seconds

logger = logging.getLogger(__name__)

def default_upload_wait(x: int = 10) -> None:
    time.sleep(x)

class Audio:
    """Audio class to handle audio input from local file or internet url."""
    download_format: Optional[str] = None
    password: Optional[str] = None
    sampling_rate: int = 16000
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    verify_upload: bool = True
    upload_wait_func: Callable = default_upload_wait
    _file_path: Optional[str] = None
    _url: Optional[str] = None
    _rawdata: Optional[np.ndarray] = None

    def __init__(
        self,
        **kwargs: Any
    ):
        for attr, value in kwargs.items():
            if value is None or value == "":
                continue
            setattr(self, attr, value)


    @property
    def source(self) -> Optional[str]:
        return self._url if self._url else self._file_path
    
    @source.setter
    def source(self, source: str):
        if source.startswith("https://"):
            self._url = source
            self._file_path = None
        else:
            self._file_path = source
            self._url = None
        self._rawdata = None

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def url(self) -> str:
        return self._url

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

    @property
    def length(self) -> float:
        return len(self.ndarray) / self.sampling_rate

    @property
    def ndarray(self) -> np.ndarray:
        if self._rawdata is None:
            _rawdata = self._load_audio()
            return _rawdata[self.start_frame:self.end_frame]
        else:
            return self._rawdata[self.start_frame:self.end_frame]

    def _load_audio(self) -> np.ndarray:
        """Load audio to self._rawdata based on self._url or self._file_path.
        Returns loaded audio data"""
        if self._file_path is None and self._url is None:
            raise ValueError("No audio source is set.")
        if self._file_path is None or not os.path.exists(self._file_path):
            logger.info(f"File path ({self._file_path}) is not set or file does not exist. Downloading audio.")
            self._file_path = dl_audio(self._url, self.download_format, self.password) # type: ignore
        logger.info(f"Loading audio file {self._file_path} ({os.path.getsize(self._file_path)/1000000:.02f}MB)")
        # Check if the uploading is finished.
        # self.upload_wait_func is used to wait for the check.
        if self.verify_upload:
            logger.info(f"Checking if uploading {self._file_path} is still uploading.")
            uploading_flag, filesize1, filesize2, wait_time = is_uploading(self._file_path, self.upload_wait_func)
            if uploading_flag:
                logger.info(f"File size increase is detected in {wait_time:.02f} seconds.")
                message = f"{self._file_path} seems still uploading. Waiting until upload finished."
                if 'IPython' in sys.modules:
                    display = sys.modules["IPython"].display
                    display.display(display.HTML(f'<div style="color: red; font-size:large; font-weight: bold;">⚠️ {message}</div>'))
                uploading_flag = True
                while uploading_flag:
                    uploading_flag, filesize1, filesize2, wait_time = is_uploading(self._file_path, lambda x=10: time.sleep(x))
                    logger.info(f"Uploaded size: {filesize2 / 1000000}MB ({(filesize2 - filesize1) / (wait_time * 1000):.01f} kB/s)")
            else:
                logger.info(f"No file size increase is detected in {wait_time:.02f} seconds.")
        _rawdata = decode_audio(self._file_path, self.sampling_rate)
        if not isinstance(_rawdata, np.ndarray):
            raise ValueError("Failed to get audio data. Check if url or file path is correctly set.")
        self._rawdata = _rawdata
        return self._rawdata


    @property
    def live_stream(self) -> subprocess.Popen:
        if not self._url:
            raise ValueError("No url set.")
        command = ["yt-dlp", "-g", self._url, "-x", "-S", "+acodec:mp4a", "-q"]
        audio_url = subprocess.check_output(command).decode("utf-8").strip()
        return decode_audio_pipe(audio_url, 16000)


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
            logger.warning("Entire audio signal is silent!")
            return None, None
        leading = non_silent_indices[0]
        trailing = non_silent_indices[-1]
        min_frame_size = int(min_silence_duration * self.sampling_rate)
        if (not self.start_frame) and (leading > min_frame_size):
            logger.info(f"Leading silence detected. Skipping {leading / self.sampling_rate} seconds.")
            self.start_frame = int(leading)
        frame_size = len(self.ndarray)
        if (not self.end_frame) and (trailing < frame_size) and (trailing + min_frame_size < frame_size):
            trailing_sec = (frame_size - trailing)/ self.sampling_rate
            logger.info(f"Trailing silence detected. Skipping the last {trailing_sec} seconds.")
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
    """Returns audio as Popen instance.
    Audio can be internet url (any uri that ffmpeg can parse)."""
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


def dl_audio(url: str, format: Optional[str] = None, password: Optional[str] = None):
    """Download file from Internet"""
    logger.info(f"Downloading audio from {url}")
    ydl_opts = {
        'format': '140/bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'quiet': False,
        'noplaylist': True,
    }
    if password:
        ydl_opts['video_password'] = password
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        outfilename = ydl.prepare_filename(info)
        return outfilename

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
            ffmpeg.input(input_file).output(output_file),
            overwrite_output=overwrite
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
                sys.stdout.write(buf.decode('utf-8'))
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
