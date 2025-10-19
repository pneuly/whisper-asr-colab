import os
import time
import threading
from joblib import Parallel, delayed
from collections.abc import Callable
from typing import Optional, Any

DEFAULT_PROGRESS_FILE = "progress.txt"


def _tail_progress_file(stop_event: threading.Event, file_path: str):
    while not os.path.exists(file_path):
        print("Waiting for ASR to begin.")
        if stop_event.wait(timeout=5):
            break

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    stripped_line = line.strip()
                    if stripped_line:
                        print(stripped_line, flush=True)
                else:  # file not updated, wait a bit
                    time.sleep(0.5)
            print(f.readline().strip(), flush=True)  # print last line if any
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found or was deleted during tailing.")


def process_isolator(
    func: Callable, progress_file: str | None, *args: Any, **kwargs: Any
) -> Any:
    """
    Executes a function in an isolated child process using joblib's 'loky' backend.

    This is primarily used to prevent **crashes or conflicts** (e.g., between
    libraries like FasterWhisper and pyannote.audio) when initialized in the
    same namespace, particularly in Jupyter environments.

    Since 'loky' suppresses child process output, a **progress monitoring**
    workaround is employed:
    1. A separate thread **tails** the specified `progress_file`.
    2. The isolated function (`func`) is expected to write its status updates
       to this file, which the main process then prints.

    Args:
        func (Callable): The function to execute in isolation. It should write
            progress messages to the file specified by `progress_file`.
        progress_file (Optional[str]): The file path for progress communication.
            Defaults to `"progress.txt"`.
        *args (Any): Positional arguments for `func`.
        **kwargs (Any): Keyword arguments for `func`.

    Returns:
        Any: The result returned by the isolated function `func`.
    """

    stop_event = threading.Event()
    progress_file = progress_file or DEFAULT_PROGRESS_FILE

    threading.Thread(
        target=_tail_progress_file, args=(stop_event, progress_file), daemon=True
    ).start()

    result = Parallel(n_jobs=2, backend="loky", verbose=5)(
        [delayed(func)(*args, **kwargs)]
    )

    stop_event.set()
    return result[0]
