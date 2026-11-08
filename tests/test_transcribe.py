"""Cover the long-form transcription pipeline without hardware.

Synthetic audio and a stand-in pipeline stand in for the NPU: the chunker,
timestamp arithmetic and every export format are pure computation, so they are
tested exactly, while the model call itself is exercised by the real hardware
run recorded in the README.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from intel_npu_tools import transcribe
from intel_npu_tools.transcribe import Segment, Transcript, split_chunks


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    """An audibly-continuous band, indistinguishable to an energy gate from speech."""
    time = np.arange(int(seconds * transcribe.SAMPLE_RATE)) / transcribe.SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * transcribe.SAMPLE_RATE), dtype=np.float32)


@dataclass
class FakeChunk:
    text: str
    start_ts: float
    end_ts: float


@dataclass
class FakeResult:
    texts: list[str] = field(default_factory=list)
    chunks: list = field(default_factory=list)
    language: str | None = None


@dataclass
class FakePipeline:
    """One canned result per generate() call, recording which chunks it saw."""

    responses: list = field(default_factory=list)
    seen: list = field(default_factory=list)

    def generate(self, chunk, **kwargs):
        assert kwargs.get("return_timestamps") is True
        self.seen.append(chunk)
        return self.responses.pop(0)


# ---------------------------------------------------------------- timestamps


def test_srt_uses_comma_milliseconds_and_counts_from_one():
    transcript = Transcript("clip.wav", 2.4, "en", [Segment(0.0, 1.0, "hello"), Segment(1.5, 2.4, "again")])
    assert transcript.to_srt() == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"
        "2\n00:00:01,500 --> 00:00:02,400\nagain\n"
    )


def test_vtt_has_a_header_note_and_dot_milliseconds():
    transcript = Transcript("clip.wav", 2.4, "en", [Segment(0.0, 1.0, "hello")])
    vtt = transcript.to_vtt()
    assert vtt.startswith("WEBVTT\nNOTE clip.wav\n\n")
    assert "00:00:00.000 --> 00:00:01.000\nhello\n" in vtt


def test_vtt_and_srt_are_both_valid_with_no_segments_at_all():
    transcript = Transcript("silence.wav", 30.0, None, [])
    assert transcript.to_srt() == ""
    assert transcript.to_vtt().startswith("WEBVTT")


def test_clock_never_prints_a_rolled_up_second():
    # 0.9997 of a second rounds milliseconds to 1000; the carry must land in
    # the next unit rather than print :1000 or :60.
    assert transcribe._clock(3599.9997) == "01:00:00,000"
    assert transcribe._clock(59.9997) == "00:01:00,000"
    assert transcribe._clock(89.9997) == "00:01:30,000"


def test_clock_clamps_negative_starts():
    assert transcribe._clock(-0.5) == "00:00:00,000"


# ----------------------------------------------------------------- markdown


def test_markdown_groups_adjacent_segments_into_timestamped_paragraphs():
    transcript = Transcript(
        "meeting.wav",
        70.0,
        "en",
        [
            Segment(0.0, 5.0, "One two three."),
            Segment(5.1, 9.0, "Four five."),
            Segment(69.0, 70.9, "Later point."),
        ],
    )
    text = transcript.to_markdown()
    assert text.startswith("# Transcript — meeting.wav")
    assert "*Duration 01:10 · Original language: en · Whisper on Intel NPU*" in text
    # The two tight segments share one paragraph; the 60-second gap splits.
    assert "**[00:00]** One two three. Four five." in text
    assert "**[01:09]** Later point." in text


def test_markdown_plain_mode_drops_the_timestamp_markers():
    transcript = Transcript("meeting.wav", 30.0, None, [Segment(3.0, 6.0, "only words")])
    plain = transcript.to_markdown(with_timestamps=False)
    assert "only words" in plain
    assert "**[" not in plain


def test_markdown_reports_silence_rather_than_an_empty_body():
    transcript = Transcript("quiet.wav", 300.0, None, [])
    assert "(No speech detected)" in transcript.to_markdown()


def test_full_text_joins_everything():
    transcript = Transcript("a.wav", 5.0, None, [Segment(0, 1, "a"), Segment(1, 2, "b")])
    assert transcript.full_text() == "a b"


def test_to_dict_rounds_and_flattens():
    transcript = Transcript("a.wav", 5.0, "en", [Segment(0.5, 1.5, "hi")])
    assert transcript.to_dict() == {
        "source": "a.wav",
        "duration": 5.0,
        "language": "en",
        "text": "hi",
        "segments": [{"start": 0.5, "end": 1.5, "text": "hi"}],
    }


# ------------------------------------------------------------------ chunking


def test_chunks_cover_every_sample_without_gaps_or_overlap():
    samples = np.concatenate([
        tone(25), quiet(4), tone(25), quiet(4), tone(25), quiet(4), tone(3),
    ])
    chunks = split_chunks(samples)
    assert np.array_equal(np.concatenate([chunk for _, chunk in chunks]), samples)
    starts = [start for start, _ in chunks]
    assert starts == sorted(starts)
    assert chunks[0][0] == 0.0


def test_cuts_snap_to_silence_when_a_pause_is_nearby():
    # Speech 0–27.5, a real pause 27.5–30.5, speech again. Interior cuts that
    # chase the 30-second window must fall inside a pause, not the word before it.
    samples = np.concatenate([tone(27.5), quiet(3.0), tone(27.0), quiet(3.0), tone(28.0), quiet(6.5)])
    pauses = [(27.5, 30.5), (57.5, 60.5)]
    position = 0.0
    cut_inside_pause = 0
    for _, chunk in split_chunks(samples)[:-1]:
        position += chunk.size / transcribe.SAMPLE_RATE
        cut_inside_pause += any(low <= position <= high for low, high in pauses)
    # Only the first two cuts can reach a pause: the third is the rebalanced
    # split, which sits mid-speech by construction. Two snapped cuts is the
    # claim.
    assert cut_inside_pause >= 2


def test_a_short_remainder_is_rebalanced_not_left_thin():
    # 61 seconds would otherwise cut as 30 + 30 + 1; the one-second sliver is
    # exactly what the minimum exists to prevent.
    samples = np.concatenate([tone(28), quiet(2.4), tone(28), quiet(2.6)])
    lengths = [chunk.size / transcribe.SAMPLE_RATE for _, chunk in split_chunks(samples)]
    assert abs(sum(lengths) - 61.0) < 0.2
    assert all(length >= transcribe.MINIMUM_FINAL_SECONDS for length in lengths[1:])


def test_short_recordings_stay_one_chunk():
    samples = np.concatenate([tone(6), quiet(0.5)])
    chunks = split_chunks(samples)
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0
    assert chunks[0][1].size == samples.size


def test_quietest_frame_is_found_in_the_search_band():
    samples = np.concatenate([tone(20, 0.5), quiet(4), tone(20, 0.5)])
    # A target at 24 seconds has a search band 21–27, and only the silence in
    # front of the second tone (20–24) sits inside it with zero energy.
    cut = transcribe._quietest_sample_index(samples, 24, transcribe.SEEK_SECONDS)
    assert 20.0 <= cut / transcribe.SAMPLE_RATE <= 24.0


# ------------------------------------------------------------------- gating


def test_chunk_timestamps_are_shifted_onto_the_file_timeline():
    # The offset math is tested directly, because where the stream's own cuts
    # land depends on the audio; global offsets are offset + model timestamp.
    result = FakeResult(language="en")
    result.chunks = [FakeChunk("a", 1.0, 3.5), FakeChunk("b", 4.0, 6.0), FakeChunk("[BLANK_AUDIO]", 6.0, 7.0)]
    result.texts = ["a b"]
    segments, language = transcribe._chunk_segments(tone(40), 25.0, FakePipeline(responses=[result]))
    assert [(s.start, s.end, s.text) for s in segments] == [(26.0, 28.5, "a"), (29.0, 31.0, "b")]
    assert language == "en"


def test_a_text_only_result_becomes_one_spanning_segment():
    result = FakeResult(texts=["whole window words"], language="en")
    chunk = np.concatenate([tone(30), quiet(0.1)])
    segments, _ = transcribe._chunk_segments(chunk, 60.0, FakePipeline(responses=[result]))
    assert len(segments) == 1
    assert segments[0].text == "whole window words"
    assert segments[0].start == 60.0
    assert segments[0].end == pytest.approx(60.0 + chunk.size / transcribe.SAMPLE_RATE, abs=0.01)


def test_caption_noise_is_dropped():
    assert transcribe._is_caption_noise("[BLANK_AUDIO]")
    assert transcribe._is_caption_noise("[Music]")
    assert transcribe._is_caption_noise("(typing)")
    assert not transcribe._is_caption_noise("[Music] around them")
    assert not transcribe._is_caption_noise("Hello there.")


def test_bracket_captions_are_removed_but_real_words_are_kept():
    caption = FakeResult(texts=["[BLANK_AUDIO]"], chunks=[FakeChunk("[Music]", 0, 1)])
    real = FakeResult(texts=["real words"], chunks=[FakeChunk("real words", 0, 2), FakeChunk("[market]", 2, 3)])
    pipeline = FakePipeline(responses=[caption, real])
    samples = np.concatenate([tone(20), quiet(0.1)])
    segments = list(transcribe.transcribe_stream(np.concatenate([samples, samples]), wav_pipeline=pipeline))
    assert [s.text for s in segments] == ["real words"]


def test_a_silent_leading_recording_is_never_sent_to_the_model():
    # Enough silence ahead that the first chunk is room tone through and
    # through; the tone behind it is short enough to be the final chunk.
    samples = np.concatenate([quiet(29.0), tone(14)])  # 43 s total
    result = FakeResult(texts=["after the pause"], chunks=[FakeChunk("after the pause", 0.2, 2.2)], language="en")
    pipeline = FakePipeline(responses=[result])
    segments = list(transcribe.transcribe_stream(samples, wav_pipeline=pipeline))
    assert len(pipeline.seen) == 1
    assert [s.text for s in segments] == ["after the pause"]
    # The single call happened on the trailing tone, after the 29 seconds of
    # silence — its starting offset sits near the pause, not near zero.
    assert segments[0].start > 27.0


# ------------------------------------------------------------- file pipeline


@pytest.fixture
def fake_hardware(monkeypatch):
    """Install the stand-in pipeline and a decode step that needs no ffmpeg."""

    samples = tone(45)  # splits as 30 + 15, a two-chunk recording

    def install(responses):
        pipeline = FakePipeline(responses=list(responses))
        monkeypatch.setattr(transcribe, "whisper_pipeline", lambda: pipeline)
        monkeypatch.setattr(transcribe.audio, "decode_audio", lambda path: samples)
        return pipeline

    return install


def pipeline_result(text="spoken", language="en"):
    return FakeResult(
        texts=[text],
        chunks=[FakeChunk(text, 0.25, 2.75)],
        language=language,
    )


MEETING = Path("/somewhere/meeting.wav")


def test_transcribe_records_source_language_and_progress(fake_hardware):
    fake_hardware([pipeline_result("a phrase"), pipeline_result("b phrase")])
    progress = []
    transcript = transcribe.transcribe(MEETING, chunk_callback=lambda t, at: progress.append((t, at)))
    assert transcript.source == "meeting.wav"
    assert transcript.duration == 45.0
    assert transcript.language == "en"
    assert [s.text for s in transcript.segments] == ["a phrase", "b phrase"]
    # Every later offset sits further out, and progress advanced chunk-wise
    # (the second call arrives after chunk one finished) rather than all at once.
    offsets = [at for _, at in progress]
    assert offsets == sorted(offsets)
    assert 20.0 < progress[1][1] <= 45.0


def test_transcribe_passes_timestamps_down_to_the_model(fake_hardware):
    pipeline = fake_hardware([pipeline_result(), pipeline_result()])
    transcribe.transcribe(MEETING)
    assert len(pipeline.seen) == 2


def test_write_files_produces_every_requested_format(fake_hardware, tmp_path):
    fake_hardware([pipeline_result("hello"), pipeline_result("hello")])
    transcript = transcribe.transcribe(MEETING)
    written = transcribe.write_files(transcript, tmp_path)
    assert [path.name for path in written] == ["meeting.vtt", "meeting.srt", "meeting.md"]
    assert "hello" in (tmp_path / "meeting.srt").read_text(encoding="utf-8")


def test_write_files_rejects_an_unknown_format(fake_hardware, tmp_path):
    fake_hardware([pipeline_result(), pipeline_result()])
    with pytest.raises(ValueError, match="Unknown format"):
        transcribe.write_files(transcribe.transcribe(MEETING), tmp_path, formats=("wav",))


def test_main_writes_formats_and_json(fake_hardware, tmp_path, capsys):
    fake_hardware([pipeline_result("final words"), pipeline_result("final words")])
    output = tmp_path / "out"
    assert transcribe.main(["--output-dir", str(output), "--json", str(MEETING)]) == 0
    printed = capsys.readouterr().out.strip().splitlines()
    names = [Path(line).name for line in printed[:-1]]
    assert names == ["meeting.vtt", "meeting.srt", "meeting.md"]
    summary = json.loads(printed[-1])
    assert summary["source"] == "meeting.wav"
    assert summary["segments"][0]["text"] == "final words"
    assert summary["segments"][0]["start"] == pytest.approx(0.25, abs=0.01)
    # Files really landed, not just printouts of where they would go.
    assert (output / "meeting.md").exists()


def test_main_plain_mode_suppresses_timestamp_markers(fake_hardware, tmp_path):
    fake_hardware([pipeline_result("words"), pipeline_result("words")])
    output = tmp_path / "out"
    assert transcribe.main(["--plain", "--output-dir", str(output), str(MEETING)]) == 0
    markdown = (output / "meeting.md").read_text(encoding="utf-8")
    assert "**[" not in markdown
    assert "words" in markdown
