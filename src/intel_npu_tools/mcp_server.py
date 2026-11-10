import subprocess
import tempfile
from pathlib import Path

from mcp.server import MCPServer

from .audio import transcribe_file
from .context_filter import filter_context
from .ocr import capture_current_monitor, extract_text, structured_text
from .paths import WHISPER_MODEL
from .semantic import SemanticIndex
from .transcribe import Transcript, transcribe as transcribe_recording


mcp = MCPServer(
    "intel-npu-tools",
    title="Intel Arrow Lake NPU Tools",
    description="Private local speech, OCR, and semantic search using Intel AI Boost.",
)

semantic = SemanticIndex()

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def local_file(value: str, suffixes: tuple[str, ...]) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return path


@mcp.tool()
def npu_status() -> dict:
    """Report OpenVINO devices and whether Intel AI Boost is available."""
    import openvino as ov

    core = ov.Core()
    devices = {device: core.get_property(device, "FULL_DEVICE_NAME") for device in core.available_devices}
    return {"npu_available": "NPU" in devices, "devices": devices, "whisper_model": str(WHISPER_MODEL)}


@mcp.tool()
def transcribe_audio(audio_path: str, formats: str | None = None) -> str | dict:
    """Transcribe a local audio file on the Intel NPU.

    With no formats argument this returns the plain transcript text, exactly
    as before. Pass formats="vtt,srt,md" (any mix of those) and the language,
    duration and every timestamped segment come back with the rendered WebVTT,
    SubRip and Markdown documents — useful for meetings and lectures, where
    the transcript has to say when each thing was said.
    """
    path = local_file(audio_path, (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".opus"))
    if not formats:
        return transcribe_file(path)
    requested = {name.strip() for name in formats.split(",") if name.strip()}
    transcript: Transcript = transcribe_recording(path)
    rendered = {
        "vtt": transcript.to_vtt,
        "srt": transcript.to_srt,
        "md": lambda: transcript.to_markdown(),
        "json": transcript.to_dict,
    }
    unknown = sorted(requested - set(rendered))
    if unknown:
        raise ValueError(f"Unknown format(s) {unknown}; expected any of {sorted(rendered)}")
    return {name: render() for name, render in sorted(rendered.items()) if name in requested}


@mcp.tool()
def record_and_transcribe(seconds: int = 10) -> str:
    """Record the default microphone for 1-60 seconds and transcribe on the NPU."""
    seconds = max(1, min(int(seconds), 60))
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
        subprocess.run(["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", "--sample-count", str(16000 * seconds), audio.name], check=True, timeout=seconds + 10)
        return transcribe_file(Path(audio.name))


@mcp.tool()
def ocr_image(image_path: str) -> dict:
    """Extract English and Arabic text from a local image using NPU text models."""
    return extract_text(local_file(image_path, IMAGE_SUFFIXES))


@mcp.tool()
def ocr_current_monitor() -> dict:
    """Capture the current monitor and extract its text."""
    with tempfile.NamedTemporaryFile(suffix=".png") as image:
        return extract_text(capture_current_monitor(Path(image.name)))


@mcp.tool()
def screen_to_text(
    image_path: str | None = None, detail: str = "lines", min_confidence: int = 40
) -> dict:
    """Read a screenshot as compact structured text instead of sending an image.

    A screenshot costs a vision model roughly 1,600 to 4,800 image tokens. This
    returns the same screen as a few hundred tokens of text in reading order,
    with a bounding box per line, so you can see what is on screen and where to
    click without spending them. Omit image_path to capture the current monitor.

    detail is "lines" (default: text plus one box per line), "text" (cheapest,
    reading order only), or "words" (a box per word — on a dense screen this can
    cost MORE tokens than the screenshot did, and the reply says so when it has).

    Read these limits before relying on it. The text comes from Tesseract OCR, so
    it is a best-effort transcription and not a user-interface tree: it cannot
    report widget types, enabled or checked state, focus, scroll position, or
    anything off-screen, and it misreads small or low-contrast text. **If the
    target is a web page, use Playwright's accessibility tree instead** — that is
    already structured text, it is exact, it includes content scrolled out of
    view, and it costs no model tokens to produce. This tool is for surfaces with
    no such tree: native desktop applications, remote desktops, canvas and WebGL,
    video frames, and scanned documents.
    """
    if image_path:
        return structured_text(local_file(image_path, IMAGE_SUFFIXES), detail, min_confidence)
    with tempfile.NamedTemporaryFile(suffix=".png") as image:
        return structured_text(capture_current_monitor(Path(image.name)), detail, min_confidence)


@mcp.tool()
def semantic_index(path: str, include_office: bool = True) -> dict:
    """Index a local text or office document or directory for private semantic search on the Intel NPU."""
    return semantic.index(path, include_office=include_office)


@mcp.tool()
def semantic_search(
    query: str, limit: int = 5, root: str | None = None, rerank: bool | None = None
) -> list[dict]:
    """Search indexed local files by meaning using Intel NPU embeddings.

    When the optional reranker model is installed, the top passages are re-scored
    by a cross-encoder and each hit gains a `rerank_score`. That score is an
    unbounded logit, not a cosine, so do not compare it against `score`. Pass
    rerank=false to skip the extra second of work.
    """
    return semantic.search(query, limit, root, rerank)


@mcp.tool()
def context_filter(path: str, query: str, limit: int = 8, context_lines: int = 0) -> dict:
    """Extract only the parts of a large local text file that are relevant to a question.

    Use this instead of reading a big build log, test output, diff, or data file
    into your context. Write the output to a file first, then filter it:

        make 2>&1 | tee /tmp/build.log
        context_filter("/tmp/build.log", "why did the linker step fail")

    Returns spans copied verbatim from the file with exact line numbers, never a
    summary, so quoted text and line numbers can be cited. It also reports how
    much was dropped and how close the best dropped chunk scored, so you can tell
    when to widen the query or raise limit rather than assuming full coverage.

    Chunks are embedded on the Intel NPU and ranked by cosine similarity; nothing
    is written to the semantic index and nothing leaves this machine. This is not
    a substitute for grep: when you know the exact string to look for, grep is
    faster, free, and exact. Files under 4 KB and over 256 KB are rejected.
    """
    return filter_context(path, query, limit, context_lines)


@mcp.tool()
def semantic_index_status() -> dict:
    """Report the local semantic index database, roots, file count, and chunk count."""
    return semantic.status()


@mcp.tool()
def open_speech_app() -> str:
    """Open the interactive speech-to-text desktop application."""
    subprocess.Popen(["intel-npu-speech"], start_new_session=True)
    return "Opened Intel NPU Speech to Text."


@mcp.tool()
def open_ocr_selector() -> str:
    """Open the rectangular screenshot OCR selector."""
    subprocess.Popen(["intel-npu-ocr"], start_new_session=True)
    return "Opened Intel NPU Screenshot OCR."


def main():
    mcp.run(transport="stdio")
