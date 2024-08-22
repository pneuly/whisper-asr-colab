import datetime
def str2seconds(time_str):
    for fmt in ("%H:%M:%S", "%M:%S", "%S", "%H:%M:%S.%f", "%M:%S.%f", "%S.%f"):
        try:
            return (
                datetime.strptime(time_str, fmt) - datetime(1900, 1, 1)
                ).total_seconds()
        except ValueError:
            pass
    print(f"Error: Unable to parse time string '{time_str}'")
    return None


def format_timestamp(seconds: float):
    # td = timedelta(seconds=seconds)
    # return f"{str(td)[:10]}"
    hours = seconds // 3600
    remain = seconds - (hours * 3600)
    minutes = remain // 60
    seconds = remain - (minutes * 60)
    return "{:01}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)


def time_segment_text(segment, timestamp_offset=None):
    _offset_seconds = str2seconds(timestamp_offset) if timestamp_offset else 0.0
    start = segment["start"] + _offset_seconds
    end = segment["end"] + _offset_seconds
    return (f"[{format_timestamp(start)} - {format_timestamp(end)}]")


def add_timestamp(segment, timestamp_offset=None):
    return (f"{time_segment_text(segment, timestamp_offset)} {segment['text'].strip()}")


def write_asr_result(basename, segments, timestamp_offset=None):
    outfilenames = (f"{basename}.txt", f"{basename}_timestamped.txt")
    filehandles = [open(filename, "w", encoding="utf-8") for filename in outfilenames]
    for segment in segments:
        filehandles[0].write(segment["text"] + "\n")
        filehandles[1].write(add_timestamp(segment, timestamp_offset) + "\n")
    filehandles[0].close()
    filehandles[1].close()
    return outfilenames


def write_diarize_result(basename, segments, timestamp_offset=None):
    outfilename = f"{basename}_diarized.txt"
    fh = open(outfilename, "w", encoding="utf-8")
    for segment in segments:
        fh.write(time_segment_text(segment, timestamp_offset) + " ")
        try:
            fh.write(segment["speaker"] + "\n")
        except:
            fh.write("\n")
        fh.write(segment["text"].replace(" ", "") + "\n\n")
    fh.close()
    return outfilename
