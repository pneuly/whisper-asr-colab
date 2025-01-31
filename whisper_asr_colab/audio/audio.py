import sys
import os
import time
import logging
import subprocess
import numpy as np
from typing import Union, Optional, Tuple, Callable, Any, TypeVar
from .utils import decode_audio, decode_audio_pipe, dl_audio, is_uploading
from ..common.utils import str2seconds

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Audio')
class Audio:
    """Audio class to handle audio input from local file or internet url."""
    _source: str | T  # To read this value, use .source property
    download_format: Optional[str] = None
    password: Optional[str] = None
    sampling_rate: int = 16000
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    verify_upload: bool = True
    is_remote: Optional[bool] = None
    _local_file_path: Optional[str] = None
    _rawdata: Optional[np.ndarray] = None

    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], cls):
            # if source is an instance of Audio, do nothing.
            return args[0]
        return super().__new__(cls)
    
    def __init__(
        self,
        source: str | T,
        **kwargs: Any
    ):
        if isinstance(source, Audio):
            return
        self._source = source  # read only after instance created
        if source.startswith("https://"):
            self.is_remote = True
        else:
            self.is_remote = False
            self._local_file_path = source

        for attr, value in kwargs.items():
            if value is None or value == "":
                continue
            setattr(self, attr, value)

    def __str__(self):
        return f'{self.__class__.__name__}({", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())})'

    @property
    def source(self):
        return self._source

    @property
    def local_file_path(self) -> str:
        return self._local_file_path

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
        """Load audio to self._rawdata.
        Returns loaded audio data"""
        if self.local_file_path is None and self.is_remote:
            logger.info(f"Downloading from ({self.source}) ...")
            self._local_file_path = dl_audio(self.source, self.download_format, self.password) # type: ignore
        logger.info(f"Loading audio file {self.local_file_path} ({os.path.getsize(self.local_file_path)/1000000:.02f}MB)")
        # Check if the uploading is finished.
        if self.verify_upload:
            logger.info(f"Checking if uploading {self.local_file_path} is still uploading.")
            uploading_flag, filesize1, filesize2, wait_time = is_uploading(self.local_file_path)
            if uploading_flag:
                logger.info(f"File size increase is detected in {wait_time:.02f} seconds.")
                message = f"{self.local_file_path} seems still uploading. Waiting until upload finished."
                if 'IPython' in sys.modules:
                    display = sys.modules["IPython"].display
                    display.display(display.HTML(f'<div style="color: red; font-size:large; font-weight: bold;">⚠️ {message}</div>'))
                uploading_flag = True
                while uploading_flag:
                    uploading_flag, filesize1, filesize2, wait_time = is_uploading(self.local_file_path, lambda x=10: time.sleep(x))
                    logger.info(f"Uploaded size: {filesize2 / 1000000}MB ({(filesize2 - filesize1) / (wait_time * 1000):.01f} kB/s)")
            else:
                logger.info(f"No file size increase is detected in {wait_time:.02f} seconds.")
        _rawdata = decode_audio(self.local_file_path, self.sampling_rate)
        if not isinstance(_rawdata, np.ndarray):
            raise ValueError("Failed to get audio data. Check if url or file path is correctly set.")
        self._rawdata = _rawdata
        return self._rawdata


    @property
    def live_stream(self) -> subprocess.Popen:
        if not self.source:
            raise ValueError("No url set.")
        command = ["yt-dlp", "-g", self.source, "-x", "-S", "+acodec:mp4a", "-q"]
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
