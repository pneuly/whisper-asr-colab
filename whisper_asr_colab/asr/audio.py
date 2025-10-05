import os
import time
import requests
import numpy as np
from typing import Union, Optional, Iterable, BinaryIO
from pydub import AudioSegment
from io import BytesIO
from .utils import download_file, get_file_size, get_audio_format

class Audio:
    """Audio file handler.
    Supports local files and URL (http/https).

    Attributes:
        url (str): Audio file URL or local path.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        download_format (str): Format to download audio (for URL sources).
        password (str): Password for audio files requiring authentication.
        ndarray (np.ndarray): Audio data as numpy array.
    """

    def __init__(self, url: str):
        self.url = url
        self.start_time = 0.0
        self.end_time = 0.0
        self.download_format = "mp3"
        self.password = ""
        self.ndarray = None

    @classmethod
    def from_path_or_url(cls, path_or_url: Union[str, BinaryIO]):
        """Create Audio instance from local path or URL."""
        if isinstance(path_or_url, str):
            instance = cls(url=path_or_url)
            if instance.is_url():
                instance.download()
            else:
                instance.load_local()
            return instance
        raise ValueError("Invalid audio source.")

    def is_url(self) -> bool:
        """Check if the URL is a valid http/https URL."""
        return self.url.startswith(("http://", "https://"))

    def download(self):
        """Download audio file from URL."""
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        file_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks
        downloaded = 0
        audio_data = BytesIO()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                audio_data.write(chunk)
                downloaded += len(chunk)
                self.show_progress(downloaded, file_size)
        audio_data.seek(0)
        self.ndarray = self.audio_to_numpy(audio_data)
        self.save_local()
        print(f"\nDownloaded and saved as {self.url.split('/')[-1]}")

    def show_progress(self, downloaded: int, total: int):
        """Show download progress."""
        percent = downloaded / total * 100
        bar_length = 40
        filled_length = int(bar_length * downloaded // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(f"\r|{bar}| {percent:.2f}%", end="\r")

    def audio_to_numpy(self, audio_data: BytesIO) -> np.ndarray:
        """Convert audio data to numpy array."""
        audio_segment = AudioSegment.from_file(audio_data, format=self.download_format)
        return np.array(audio_segment.get_array_of_samples())

    def save_local(self):
        """Save downloaded audio to local file."""
        with open(self.url.split("/")[-1], "wb") as f:
            f.write(self.ndarray)

    def load_local(self):
        """Load audio data from local file."""
        file_path = self.url
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        self.ndarray = self.audio_to_numpy(file_path)

    def audio_to_numpy(self, file_path: str) -> np.ndarray:
        """Convert audio file to numpy array."""
        audio_segment = AudioSegment.from_file(file_path)
        return np.array(audio_segment.get_array_of_samples())

    def get_duration(self) -> float:
        """Get audio duration in seconds."""
        if self.ndarray is not None:
            return len(self.ndarray) / 16000.0
        raise ValueError("Audio data is not loaded.")

    def trim_silence(self, top_db=20):
        """Trim silence from the beginning and end of the audio."""
        if self.ndarray is not None:
            audio_segment = AudioSegment(
                self.ndarray.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            trimmed_audio = audio_segment.strip_silence(silence_thresh=top_db)
            self.ndarray = np.array(trimmed_audio.get_array_of_samples())
        else:
            raise ValueError("Audio data is not loaded.")

    def set_silence_skip(self):
        """Set silence skip for audio processing."""
        if self.ndarray is not None:
            self.ndarray = self.skip_silence(self.ndarray)
        else:
            raise ValueError("Audio data is not loaded.")

    def skip_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Skip silence in the audio data."""
        non_silent = np.where(np.abs(audio_data) > threshold)[0]
        if len(non_silent) == 0:
            return audio_data
        start, end = non_silent[0], non_silent[-1] + 1
        return audio_data[start:end]

    def get_time_slice(self, start_time: float, end_time: float) -> np.ndarray:
        """Get a slice of the audio data between start_time and end_time."""
        if self.ndarray is not None:
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            return self.ndarray[start_sample:end_sample]
        raise ValueError("Audio data is not loaded.")

    def __repr__(self):
        return f"<Audio url={self.url}, start_time={self.start_time}, end_time={self.end_time}>"