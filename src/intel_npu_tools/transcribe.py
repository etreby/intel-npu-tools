"""Timestamped transcription of long recordings: meetings, lectures, memos.

The microphone dictation in speech_app.py hands Whisper a clip of a few dozen
seconds and gets plain text back. That is the wrong shape for a ten-minute
meeting recording: the whole file left as one call gives the caller nothing
until every second of it has been transcribed, and the plain text has no way
to say when anything was said. This module streams the audio through the same
NPU pipeline in Whisper-sized windows instead, so a caller (or the command
line) gets timestamped segments one at a time as each window finishes, and can
render the result as WebVTT, SubRip or Markdown without any cloud service
seeing a sample of it.

The structure of the streaming is governed by one awkward fact: Whisper reads
audio through a fixed 30-second attention window. A cut placed arbitrarily
inside a word produces a hallucinated guess for the tail of that word, so
every interior cut is snapped to the quietest nearby frame, which is almost
always between utterances. ``return_timestamps`` then makes the pipeline emit
each result chunk with its own offsets, which are shifted onto the file's
timeline here. Everything except the model call itself is testable without an
NPU; the model call is behind ``whisper_pipeline`` so tests can stand in for
it.
"""

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio

# The rate Whisper's encoder was trained on, and the one decode_audio resamples
# everything to; window lengths below are defined against it.
SAMPLE_RATE = 16000

# Length of one Whisper attention window. Cutting to this or less means every
# chunk arrives at the model already inside a single window, so the timestamps
# in each result are relative to the start of the chunk we passed in.
WINDOW_SECONDS = 30.0

# An interior cut may move up to this far from its nominal position to reach
# quieter audio. Much wider and a pause at, say, 18:47 would drag a cut all the
# way to it, producing one long chunk; much narrower and a chunk frequently
# lands inside speech after all. Three seconds around the cut channel reaches
# the gap after a sentence in ordinary speech.
SEEK_SECONDS = 3.0

# Resolution of the search for that quieter frame. Shorter frames follow the
# waveform's own amplitude more than whether speech is in them; longer frames
# blur a real pause into the word before it.
FRAME_SECONDS = 0.375

# A final chunk shorter than this gets merged back by rebalancing the cut that
# precedes it. A window of a second or two is below what a typical sentence
# occupies, so cutting one out invites a hallucinated guess in both directions.
MINIMUM_FINAL_SECONDS = 8.0

# Peak-sample gate for a chunk with nothing to transcribe. The same 0.002 the
# dictation path uses on whole recordings; a chunk that never gets above it is
# room tone, and running the model on it produces invented words rather than
# nothing, which is worse.
SILENCE_LEVEL = 0.002

_CAPTION_PATTERN = re.compile(r"[\[\(][^\]\)]*[\]\)]", re.IGNORECASE)


@dataclass
class Segment:
    """One utterance, with its place on the recording's own timeline."""

    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """Everything a recording produced: its utterances plus context for them."""

    source: str
    duration: float
    language: str | None
    segments: list[Segment]

    def full_text(self) -> str:
        """The plain transcript with nothing but the words."""
        return " ".join(segment.text for segment in self.segments).strip()

    def to_srt(self) -> str:
        """SubRip: numbered blocks with comma-millisecond timestamps."""
        blocks = []
        for number, segment in enumerate(self.segments, start=1):
            blocks.append(
                f"{number}\n"
                f"{_clock(segment.start)} --> {_clock(segment.end)}\n"
                f"{segment.text}\n"
            )
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        """WebVTT: the same cues with dot-millisecond timestamps."""
        cues = [
            f"{_clock(segment.start, decimal=True)} --> {_clock(segment.end, decimal=True)}\n{segment.text}\n"
            for segment in self.segments
        ]
        header = ["WEBVTT"]
        if self.source:
            header.append(f"NOTE {self.source}")
        body = "\n\n".join(cues) if cues else ""
        return "\n".join(header) + ("\n\n" + body if body else "") + "\n"

    def to_markdown(self, with_timestamps: bool = True) -> str:
        """Readable plain text: paragraphs, timestamped at each new one.

        The word-level detail of SubRip is noise when a human reads a meeting
        transcript, so adjacent segments flow into one paragraph until silence
        interrupts them, and only the paragraph gets a marker.
        """
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        heading = [f"# Transcript — {self.source}", ""]
        language = f" · Original language: {self.language}" if self.language else ""
        heading.append(f"*Duration {minutes:02d}:{seconds:02d}{language} · Whisper on Intel NPU*")
        heading.append("")
        paragraphs = list(self.paragraphs())
        if not paragraphs:
            heading.append("*(No speech detected)*")
            return "\n".join(heading) + "\n"
        for index, (start, text) in enumerate(paragraphs):
            if index:
                heading.append("")
            if with_timestamps:
                marker = int(start // 60)
                second = int(start % 60)
                heading.append(f"**[{marker:02d}:{second:02d}]** {text}")
            else:
                heading.append(text)
        return "\n".join(heading) + "\n"

    def paragraphs(self) -> list[tuple[float, str]]:
        """Group adjacent segments into paragraphs at gaps longer than two seconds."""
        grouped: list[tuple[float, str]] = []
        current: list[str] = []
        current_start = 0.0
        previous_end = None
        for segment in self.segments:
            if previous_end is not None and segment.start - previous_end > 2.0:
                grouped.append((current_start, " ".join(current)))
                current = []
            if not current:
                current_start = segment.start
            current.append(segment.text)
            previous_end = segment.end
        if current:
            grouped.append((current_start, " ".join(current)))
        return grouped

    def to_dict(self) -> dict:
        """A compact JSON-ready form for the MCP tool and the ``--json`` flag."""
        return {
            "source": self.source,
            "duration": round(self.duration, 3),
            "language": self.language,
            "text": self.full_text(),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for s in self.segments
            ],
        }


def _clock(seconds: float, decimal: bool = False) -> str:
    """A subtitle timestamp: ``HH:MM:SS,mmm`` for SubRip, ``HH:MM:SS.mmm`` for VTT."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    whole = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds == 1000:
        # Rounding 0.9997 up must not produce :60, a timestamp no player accepts.
        milliseconds = 0
        whole += 1
        if whole == 60:
            whole = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    separator = "." if decimal else ","
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{milliseconds:03d}"


def _frame_levels(samples: np.ndarray) -> list[float]:
    """Root mean square energy of every frame_seconds-wide frame."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    full_frames, remainder = divmod(samples.size, frame)
    levels: list[float] = []
    if full_frames:
        usable = samples[: full_frames * frame].reshape(full_frames, frame)
        levels.extend(float(value) for value in np.sqrt(np.mean(np.square(usable), axis=1)))
    if remainder:
        tail = samples[full_frames * frame :]
        levels.append(float(np.sqrt(np.mean(np.square(tail)))))
    return levels


def _silent(samples: np.ndarray) -> bool:
    """True when a chunk is room tone and not worth the model's time."""
    return not samples.size or float(np.max(np.abs(samples))) < SILENCE_LEVEL


def _quietest_sample_index(samples: np.ndarray, target_seconds: float, seek_seconds: float) -> int:
    """Where to cut so that the harm done is smallest: at the least energy nearby."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    low = max(0, int((target_seconds - seek_seconds) * SAMPLE_RATE))
    high = min(samples.size, int((target_seconds + seek_seconds) * SAMPLE_RATE) + frame)
    first_frame = low // frame
    last_frame = high // frame
    window = _frame_levels(samples)[first_frame : last_frame]
    if not window:
        return low
    quietest = first_frame + int(np.argmin(window))
    # Cut in the middle of the quiet frame, not at its edge, because the pause
    # then surrounds the cut in both directions instead of ending on it.
    return min(samples.size, max(low, (quietest + 1) * frame - frame // 2))


def split_chunks(
    samples: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
    seek_seconds: float = SEEK_SECONDS,
    minimum_final_seconds: float = MINIMUM_FINAL_SECONDS,
) -> list[tuple[float, np.ndarray]]:
    """Cut a recording into Whisper-window-sized chunks at quiet points.

    Returns (start_seconds, samples) pairs whose starts are strictly increasing
    and whose concatenation is the whole recording minus nothing. Boundaries
    are carried as exact sample indices rather than rounded seconds, so
    streaming 30 chunks cannot quietly lose a fraction of a second per chunk.
    The last pair holds whatever remains, but never a sliver: a remainder too
    short compared with its own chunk is instead shared between two balanced
    halves, so both stay above the minimum and inside a single attention
    window.
    """
    rate = SAMPLE_RATE
    size = samples.size
    chunks: list[tuple[float, np.ndarray]] = []
    start = 0  # sample index, exact
    while True:
        remaining = (size - start) / rate
        if remaining <= window_seconds:
            chunks.append((start / rate, samples[start:]))
            break
        target = start / rate + window_seconds
        if remaining - window_seconds <= minimum_final_seconds:
            # Whatever followed the nominal cut would have been the one short
            # tail. Splitting what remains in two instead leaves both halves
            # transcribable, even though neither is a full window.
            target = start / rate + remaining / 2.0
        # The search window begins seek_seconds before the cut, which is
        # always past the previous boundary, so the cut advances every loop.
        cut_sample = _quietest_sample_index(samples, target, seek_seconds)
        chunks.append((start / rate, samples[start:cut_sample]))
        start = cut_sample
    return chunks


def _is_caption_noise(text: str) -> bool:
    """True when a block is nothing but bracketed filler."""
    without_captions = _CAPTION_PATTERN.sub("", text)
    return not any(character.isalnum() for character in without_captions)


def whisper_pipeline():
    """The shared NPU pipeline, resolved at call time so tests can substitute it."""
    return audio.pipeline()


def _chunk_segments(
    chunk: np.ndarray, offset: float, wav_pipeline
) -> tuple[list[Segment], str | None]:
    """Transcribe one chunk, returning its timestamped segments and language."""
    result = wav_pipeline.generate(chunk, return_timestamps=True)
    segments = []
    for item in result.chunks:
        text = str(item.text).strip()
        if not text or _is_caption_noise(text):
            continue
        start = offset + float(item.start_ts)
        end = offset + float(item.end_ts)
        segments.append(Segment(start, end, text))
    texts = [str(text).strip() for text in result.texts if str(text).strip()]
    if not segments and texts and not _is_caption_noise(texts[0]):
        # The pipeline produced text without chunk boundaries — this happens
        # when an entire window resolved to one caption. A single segment
        # spanning the chunk keeps that text discoverable; it is better than
        # dropping real words to keep the timeline neat.
        segments.append(Segment(offset, offset + chunk.size / SAMPLE_RATE, texts[0]))
    language = getattr(result, "language", None)
    return segments, str(language) if language else None


def transcribe_stream(
    source_audio: np.ndarray,
    wav_pipeline=None,
    window_seconds: float = WINDOW_SECONDS,
    seek_seconds: float = SEEK_SECONDS,
) -> Iterator[Segment]:
    """Yield timestamped segments as each chunk finishes, in speaking order.

    This is the streaming surface: a caller that renders as it goes (a progress
    bar, a live caption pane) gets its first segment around one window's
    transcribing time rather than after the whole file.
    """
    wav_pipeline = wav_pipeline or whisper_pipeline()
    for offset, chunk in split_chunks(source_audio, window_seconds, seek_seconds):
        if _silent(chunk):
            continue
        segments, _ = _chunk_segments(chunk, offset, wav_pipeline)
        yield from segments


def transcribe(path: Path, *, chunk_callback: Callable[[Transcript, float], None] | None = None) -> Transcript:
    """Transcribe a media file on the Intel NPU, returning a full transcript.

    ``chunk_callback`` receives a partially built transcript (its segments so
    far) after each completed chunk, which lets a caller report progress at the
    cadence the underlying model actually works in.
    """
    source_audio = audio.decode_audio(path)
    duration = source_audio.size / SAMPLE_RATE
    transcript = Transcript(source=path.name, duration=duration, language=None, segments=[])
    wav_pipeline = whisper_pipeline()
    for offset, chunk in split_chunks(source_audio):
        if _silent(chunk):
            continue
        segments, language = _chunk_segments(chunk, offset, wav_pipeline)
        transcript.segments.extend(segments)
        if language and not transcript.language:
            transcript.language = language
        if chunk_callback is not None:
            chunk_callback(transcript, offset + chunk.size / SAMPLE_RATE)
    return transcript


def write_files(
    transcript: Transcript,
    directory: Path,
    formats: tuple[str, ...] = ("vtt", "srt", "md"),
    plain_markdown: bool = False,
) -> list[Path]:
    """Write one file per requested format, named after the recording.

    Overwrites a matching earlier export rather than skipping one: re-running
    on the same recording after changing the model is the ordinary case, and a
    stale file quietly claiming to be the current transcript is worse than a
    fresh one replacing it.
    """
    stem = Path(transcript.source).stem or "transcript"
    renderers = {
        "vtt": transcript.to_vtt,
        "srt": transcript.to_srt,
        "md": lambda: transcript.to_markdown(not plain_markdown),
    }
    written = []
    for name in formats:
        try:
            render = renderers[name]
        except KeyError:
            raise ValueError(f"Unknown format {name!r}; expected one of {sorted(renderers)}") from None
        target = directory / f"{stem}.{name}"
        target.write_text(render(), encoding="utf-8")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files locally on the Intel NPU, with timestamps.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Audio files (WAV, MP3, M4A, FLAC, OGG, OPUS, WEBM)")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory for outputs (default: beside each input)")
    parser.add_argument(
        "--formats",
        default="vtt,srt,md",
        help="Comma-separated formats to write: vtt, srt, md (default: all)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain Markdown transcript without timestamp markers",
    )
    parser.add_argument("--json", action="store_true", help="Also print a JSON summary to stdout")
    arguments = parser.parse_args(argv)
    formats = tuple(name.strip() for name in arguments.formats.split(",") if name.strip())

    for path in arguments.files:
        path = Path(path).expanduser().resolve()
        transcript = transcribe(path)
        if not transcript.segments:
            print(f"{path.name}: nothing transcribable found", file=sys.stderr)
        directory = arguments.output_dir if arguments.output_dir else path.parent
        directory.mkdir(parents=True, exist_ok=True)
        written = write_files(transcript, directory, formats, plain_markdown=arguments.plain)
        for target in written:
            print(target)
        if arguments.json:
            print(json.dumps(transcript.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
