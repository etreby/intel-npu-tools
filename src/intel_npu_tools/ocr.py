import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .layout import (
    WordToken,
    detect_layout_blocks,
    group_words_into_lines,
    parse_tsv_words,
    render_markdown,
    tsv_to_markdown,
)
from .paths import ocr_model
from .runtime import npu_properties


DETECTOR = "horizontal-text-detection-0001"
RECOGNIZER = "text-recognition-0014"

_models = None
_lock = threading.RLock()


def ocr_models():
    """Compile both OCR models once per process.

    They were previously compiled on every npu_ocr() call, so a single
    screenshot paid the full driver compilation twice before a pixel was read.
    Compiles on NPU when present; gracefully falls back to CPU on hosts/VMs
    without an NPU.
    """
    global _models
    with _lock:
        if _models is None:
            import openvino as ov

            for name in (DETECTOR, RECOGNIZER):
                if not ocr_model(name).exists():
                    raise FileNotFoundError(
                        f"OCR model not found at {ocr_model(name)}; run scripts/download-models.py"
                    )
            core = ov.Core()
            device = "NPU" if "NPU" in core.available_devices else "CPU"
            properties = npu_properties() if device == "NPU" else {}
            _models = (
                core.compile_model(ocr_model(DETECTOR), device, properties),
                core.compile_model(ocr_model(RECOGNIZER), device, properties),
            )
        return _models


def npu_ocr(image: np.ndarray) -> tuple[str, int]:
    # A CompiledModel reuses one implicit infer request, so two concurrent MCP
    # calls against the shared models would race. The NPU runs one inference at
    # a time regardless (OPTIMAL_NUMBER_OF_INFER_REQUESTS is 1), so serialising
    # here costs nothing that the device was not already serialising.
    with _lock:
        detector, recognizer = ocr_models()
        return _run_ocr(image, detector, recognizer)


def _run_ocr(image: np.ndarray, detector, recognizer) -> tuple[str, int]:
    words, boxes_count = _run_ocr_tokens(image, detector, recognizer)
    return " ".join(item[2] for item in words), boxes_count


def _run_ocr_tokens(image: np.ndarray, detector, recognizer) -> tuple[list[tuple[int, int, str, float]], int]:
    import cv2

    _, _, det_h, det_w = detector.input(0).shape
    resized = cv2.resize(image, (int(det_w), int(det_h)))
    tensor = np.expand_dims(resized.transpose(2, 0, 1), 0)
    try:
        output = detector.output("boxes")
    except RuntimeError:
        output = detector.output(0)
    boxes = detector([tensor])[output]
    boxes = [box for box in boxes if not np.all(box == 0) and float(box[-1]) >= 0.30]
    _, _, rec_h, rec_w = recognizer.input(0).shape
    ratio_x, ratio_y = image.shape[1] / resized.shape[1], image.shape[0] / resized.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alphabet = "#1234567890abcdefghijklmnopqrstuvwxyz"
    words: list[tuple[int, int, str, float]] = []
    for box in boxes:
        x1, y1 = max(0, int(box[0] * ratio_x)), max(0, int(box[1] * ratio_y))
        x2, y2 = min(image.shape[1], int(box[2] * ratio_x)), min(image.shape[0], int(box[3] * ratio_y))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = cv2.resize(gray[y1:y2, x1:x2], (int(rec_w), int(rec_h))).reshape(1, 1, int(rec_h), int(rec_w))
        scores = np.squeeze(recognizer([crop])[recognizer.output(0)])
        characters = (alphabet[row.argmax()] for row in scores)
        confidence = float(np.mean([row.max() for row in scores])) * 100.0
        word = "".join(character for character in characters if character != "#")
        if word:
            words.append((y1, x1, word, confidence))
    words.sort(key=lambda item: (item[0] // 30, item[1]))
    return words, len(boxes)


def npu_word_tokens(image: np.ndarray, page: int = 1) -> list[WordToken]:
    """Extract WordToken objects using the OpenVINO NPU detection backbone."""
    with _lock:
        detector, recognizer = ocr_models()
        words, _ = _run_ocr_tokens(image, detector, recognizer)

    tokens: list[WordToken] = []
    for y1, x1, word, conf in words:
        # Approximate box dimensions based on character width and line height
        w_px = max(20, len(word) * 10)
        h_px = 20
        tokens.append(WordToken(
            text=word,
            bbox=(x1, y1, x1 + w_px, y1 + h_px),
            conf=conf,
            page=page,
        ))
    return tokens


def tesseract_text(path: Path) -> str:
    """Read layout, punctuation, and non-Latin scripts. Returns "" when unavailable."""
    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "eng+ara", "--psm", "6"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def capture_current_monitor(target: Path) -> Path:
    """Screenshot the screen into target, whatever desktop this is."""
    from .desktop import capture

    return capture(target, region=False)


def tesseract_tsv(path: Path) -> str:
    """Word-level OCR with layout structure. Returns "" when unavailable.

    Page segmentation mode 3 rather than the 6 used by tesseract_text above.
    Mode 3 detects columns and keeps them apart.
    """
    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "eng+ara", "--psm", "3", "tsv"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


# The same rough rule context_filter.py uses, and just as approximate.
CHARS_PER_TOKEN = 4
IMAGE_TOKEN_DIVISOR = 750
IMAGE_TOKEN_CAP = 4784


def _parse_tsv(tsv: str, min_confidence: int) -> tuple[list[dict], dict, int, int]:
    """Group Tesseract's word rows into lines using its own layout analysis."""
    rows = [line.split("\t") for line in tsv.splitlines()]
    if not rows or rows[0][:1] != ["level"]:
        return [], {}, 0, 0
    header = rows[0]
    index = {name: position for position, name in enumerate(header)}
    size: dict = {}
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    words = 0
    pages = 0
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        level = row[index["level"]]
        left, top = int(row[index["left"]]), int(row[index["top"]])
        width, height = int(row[index["width"]]), int(row[index["height"]])
        if level == "1":
            pages += 1
            if not size:
                size = {"width": width, "height": height}
            continue
        if level != "5":
            continue
        text = row[index["text"]].strip()
        try:
            confidence = float(row[index["conf"]])
        except ValueError:
            continue
        if not text or confidence < min_confidence:
            continue
        words += 1
        key = (
            row[index["page_num"]],
            row[index["block_num"]],
            row[index["par_num"]],
            row[index["line_num"]],
        )
        grouped.setdefault(key, []).append({
            "text": text,
            "conf": confidence,
            "bbox": [left, top, left + width, top + height],
        })

    lines = []
    for key in sorted(grouped, key=lambda item: tuple(int(part) for part in item)):
        members = grouped[key]
        boxes = [member["bbox"] for member in members]
        lines.append({
            "text": " ".join(member["text"] for member in members),
            "bbox": [
                min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes),
            ],
            "conf": round(sum(member["conf"] for member in members) / len(members)),
            "page": int(key[0]),
            "words": members,
        })
    return lines, size, words, pages


def structured_text(path: Path, detail: str = "lines", min_confidence: int = 40) -> dict:
    """Render a screenshot as compact structured text instead of an image."""
    if detail not in ("text", "lines", "words"):
        raise ValueError(f"detail must be text, lines, or words; got {detail!r}")
    min_confidence = max(0, min(int(min_confidence), 100))

    tsv = tesseract_tsv(path)
    if not tsv:
        raise RuntimeError(
            "Tesseract produced no output. It provides this tool's text; install it with "
            "apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara."
        )
    lines, size, words, pages = _parse_tsv(tsv, min_confidence)

    reading_order = "\n".join(line["text"] for line in lines)
    payload: dict = {"lines": []}
    if detail == "text":
        payload = {"text": reading_order}
    elif detail == "lines":
        payload = {
            "lines": [
                {"text": line["text"], "bbox": line["bbox"], "conf": line["conf"],
                 **({"page": line["page"]} if pages > 1 else {})}
                for line in lines
            ]
        }
    else:
        payload = {
            "lines": [
                {
                    "text": line["text"],
                    "bbox": line["bbox"],
                    "conf": line["conf"],
                    **({"page": line["page"]} if pages > 1 else {}),
                    "words": [
                        {"text": word["text"], "bbox": word["bbox"], "conf": round(word["conf"])}
                        for word in line["words"]
                    ],
                }
                for line in lines
            ]
        }

    rendered = len(json.dumps(payload, ensure_ascii=False))
    pixels = size.get("width", 0) * size.get("height", 0)
    image_tokens = min(IMAGE_TOKEN_CAP, pixels // IMAGE_TOKEN_DIVISOR) if pixels else 0
    text_tokens = rendered // CHARS_PER_TOKEN
    low_confidence = sum(1 for line in lines if line["conf"] < 60)

    warnings = [
        "Text is OCR output, not a user-interface tree. It cannot report widget type, "
        "enabled or checked state, focus, scroll position, or anything off-screen."
    ]
    if low_confidence:
        warnings.append(f"{low_confidence} of {len(lines)} lines scored below 60 and may be misread.")
    if detail == "words" and text_tokens > image_tokens > 0:
        warnings.append(
            "This reply is larger than the screenshot would have been. Use detail='lines' "
            "or 'text' unless per-word boxes are genuinely needed."
        )
    if not lines:
        warnings.append("No text cleared the confidence threshold; try lowering min_confidence.")
    if pages > 1:
        warnings.append(
            f"This file has {pages} pages. Each line carries its page number, and every "
            "bounding box is relative to that page rather than to the file."
        )

    return {
        "source": str(path),
        "engine": "tesseract (eng+ara)",
        "detail": detail,
        "image": size,
        **payload,
        "stats": {
            "lines": len(lines),
            "words": words,
            "pages": pages,
            "estimated_tokens": text_tokens,
            "estimated_image_tokens_avoided": image_tokens,
        },
        "estimates": (
            "Token counts are approximate. Image cost is estimated as pixels divided by 750, "
            f"capped at {IMAGE_TOKEN_CAP}, which is roughly how current vision models charge."
        ),
        "warnings": warnings,
    }


def extract_text(path: Path) -> dict:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    try:
        npu_text, regions = npu_ocr(image)
    except Exception:
        npu_text, regions = "", 0
    full_text = tesseract_text(path)
    return {"text": full_text or npu_text or "(No text detected)", "npu_regions": regions, "npu_text": npu_text}


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    """Render all pages of a PDF into PNG images in output_dir using pdftoppm.

    Executes in tens of milliseconds per page.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    prefix = output_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RuntimeError(f"pdftoppm failed to render PDF '{pdf_path}': {exc}")

    # pdftoppm names files like page-1.png or page-01.png
    images = sorted(output_dir.glob("page-*.png"), key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)])
    return images


import re


def pdftotext_tsv(pdf_path: Path, page: int | None = None) -> str:
    """Extract word-level TSV layout from a PDF using poppler pdftotext."""
    cmd = ["pdftotext", "-tsv"]
    if page is not None:
        cmd.extend(["-f", str(page), "-l", str(page)])
    cmd.extend([str(pdf_path), "-"])
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        return res.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def ocr_document_to_markdown(path: Path, min_confidence: int = 40) -> tuple[str, dict[str, Any]]:
    """Scan documents (PDF or image) to structured markdown with multi-column table extraction.

    Supports local NPU acceleration or Tesseract layout backbone.
    Returns (markdown_text, metadata).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    start_time = time.perf_counter()
    is_pdf = path.suffix.lower() == ".pdf"
    rendered_pages: list[Path] = []
    temp_dir = None

    try:
        if is_pdf:
            temp_dir = tempfile.TemporaryDirectory(prefix="npu-ocr-pdf-")
            rendered_pages = render_pdf_to_images(path, Path(temp_dir.name))
            if not rendered_pages:
                raise RuntimeError(f"No pages could be extracted from PDF: {path}")
        else:
            rendered_pages = [path]

        page_markdowns: list[str] = []
        total_words = 0
        total_tables = 0

        for page_idx, page_image in enumerate(rendered_pages, start=1):
            words: list[WordToken] = []

            # 1. Try Tesseract TSV first for rich layout
            tsv = tesseract_tsv(page_image)
            if tsv:
                words = parse_tsv_words(tsv, min_confidence=min_confidence)

            # 2. If Tesseract produced no output or is missing, try OpenVINO NPU detection
            if not words:
                try:
                    import cv2
                    img = cv2.imread(str(page_image))
                    if img is not None:
                        words = npu_word_tokens(img, page=page_idx)
                except Exception:
                    pass

            # 3. For PDFs where OCR engines were empty, fall back to pdftotext TSV
            if not words and is_pdf:
                pdf_tsv = pdftotext_tsv(path, page=page_idx)
                if pdf_tsv:
                    words = parse_tsv_words(pdf_tsv, min_confidence=min_confidence)

            if words:
                total_words += len(words)
                lines = group_words_into_lines(words)
                blocks = detect_layout_blocks(lines)
                table_count = sum(1 for b in blocks if b.get("type") == "table")
                total_tables += table_count
                page_md = render_markdown(blocks)
            else:
                page_md = "(No text detected on page)"

            if is_pdf and len(rendered_pages) > 1:
                page_markdowns.append(f"<!-- Page {page_idx} -->\n\n{page_md}")
            else:
                page_markdowns.append(page_md)

        combined_md = "\n\n".join(page_markdowns).strip() + "\n"
        elapsed = time.perf_counter() - start_time
        sec_per_page = elapsed / max(1, len(rendered_pages))

        metadata = {
            "source": str(path),
            "pages": len(rendered_pages),
            "words": total_words,
            "tables": total_tables,
            "total_seconds": round(elapsed, 4),
            "seconds_per_page": round(sec_per_page, 4),
            "engine": "tesseract_openvino_npu",
        }
        return combined_md, metadata

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
