"""Document layout analysis and GitHub-flavored markdown table extraction.

Transforms OCR word tokens and bounding boxes (from Tesseract TSV, OpenVINO
NPU detection backbone, or PaddleOCR) into clean, structured Markdown with
multi-column table detection and formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WordToken:
    text: str
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)
    conf: float = 100.0
    page: int = 1

    @property
    def left(self) -> int:
        return self.bbox[0]

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


@dataclass
class Cell:
    text: str
    bbox: tuple[int, int, int, int]
    conf: float = 100.0
    words: list[WordToken] = field(default_factory=list)

    @property
    def left(self) -> int:
        return self.bbox[0]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def x_center(self) -> float:
        return (self.left + self.right) / 2.0


@dataclass
class Line:
    words: list[WordToken]
    bbox: tuple[int, int, int, int]
    page: int = 1
    cells: list[Cell] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def left(self) -> int:
        return self.bbox[0]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]
    bbox: tuple[int, int, int, int]
    page: int = 1

    def to_markdown(self) -> str:
        """Render table as GitHub-flavored Markdown."""
        if not self.headers and not self.rows:
            return ""

        headers = [re.sub(r"\s+", " ", h.strip()).replace("|", "\\|") for h in self.headers]
        num_cols = len(headers) if headers else (len(self.rows[0]) if self.rows else 0)
        if num_cols == 0:
            return ""

        # Normalize rows to have exact num_cols
        clean_rows = []
        for r in self.rows:
            padded = [re.sub(r"\s+", " ", cell.strip()).replace("|", "\\|") for cell in r]
            if len(padded) < num_cols:
                padded.extend([""] * (num_cols - len(padded)))
            else:
                padded = padded[:num_cols]
            clean_rows.append(padded)

        # Calculate max width per column for alignment
        col_widths = [max(len(headers[i]) if i < len(headers) else 3, 3) for i in range(num_cols)]
        for r in clean_rows:
            for i in range(num_cols):
                col_widths[i] = max(col_widths[i], len(r[i]))

        lines = []
        # Header row
        hdr_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
        sep_line = "| " + " | ".join((":" + "-" * max(col_widths[i] - 1, 2)) for i in range(num_cols)) + " |"
        lines.append(hdr_line)
        lines.append(sep_line)

        # Body rows
        for r in clean_rows:
            row_line = "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
            lines.append(row_line)

        return "\n".join(lines)


def group_words_into_lines(words: list[WordToken], y_tolerance: int = 8) -> list[Line]:
    """Group word tokens into reading-order lines based on page and vertical coordinates."""
    if not words:
        return []

    # Sort primarily by page, then roughly by top Y
    sorted_words = sorted(words, key=lambda w: (w.page, w.top, w.left))
    pages: dict[int, list[WordToken]] = {}
    for w in sorted_words:
        pages.setdefault(w.page, []).append(w)

    all_lines: list[Line] = []

    for page_num, p_words in pages.items():
        current_lines: list[list[WordToken]] = []
        for w in p_words:
            placed = False
            for line_words in current_lines:
                # Calculate vertical overlap with existing line
                line_top = min(lw.top for lw in line_words)
                line_bottom = max(lw.bottom for lw in line_words)
                line_height = max(1, line_bottom - line_top)
                w_mid = (w.top + w.bottom) / 2.0

                if line_top - y_tolerance <= w_mid <= line_bottom + y_tolerance:
                    line_words.append(w)
                    placed = True
                    break

            if not placed:
                current_lines.append([w])

        # Sort lines by top coordinate
        for line_words in current_lines:
            line_words.sort(key=lambda lw: lw.left)
            bbox = (
                min(lw.left for lw in line_words),
                min(lw.top for lw in line_words),
                max(lw.right for lw in line_words),
                max(lw.bottom for lw in line_words),
            )
            line = Line(words=line_words, bbox=bbox, page=page_num)
            segment_line_cells(line)
            all_lines.append(line)

    all_lines.sort(key=lambda l: (l.page, l.top, l.left))
    return all_lines


def segment_line_cells(line: Line, min_gap_px: int = 22) -> list[Cell]:
    """Segment words within a single line into distinct horizontal cells/columns.

    Adjacent words with small spacing are grouped into a single cell phrase
    (e.g. "Monthly Rent" or "$1,500.00"). Large horizontal gaps delineate columns.
    """
    if not line.words:
        line.cells = []
        return []

    # Calculate average character width in this line
    total_chars = sum(len(w.text) for w in line.words)
    total_word_width = sum(w.width for w in line.words)
    avg_char_w = max(4.0, total_word_width / max(1, total_chars))
    col_gap_threshold = max(min_gap_px, int(avg_char_w * 2.8))

    cells: list[Cell] = []
    current_cell_words: list[WordToken] = [line.words[0]]

    for w in line.words[1:]:
        prev_w = current_cell_words[-1]
        gap = w.left - prev_w.right

        if gap >= col_gap_threshold:
            # New cell boundary
            c_bbox = (
                min(cw.left for cw in current_cell_words),
                min(cw.top for cw in current_cell_words),
                max(cw.right for cw in current_cell_words),
                max(cw.bottom for cw in current_cell_words),
            )
            c_text = " ".join(cw.text for cw in current_cell_words)
            c_conf = sum(cw.conf for cw in current_cell_words) / len(current_cell_words)
            cells.append(Cell(text=c_text, bbox=c_bbox, conf=c_conf, words=list(current_cell_words)))
            current_cell_words = [w]
        else:
            current_cell_words.append(w)

    if current_cell_words:
        c_bbox = (
            min(cw.left for cw in current_cell_words),
            min(cw.top for cw in current_cell_words),
            max(cw.right for cw in current_cell_words),
            max(cw.bottom for cw in current_cell_words),
        )
        c_text = " ".join(cw.text for cw in current_cell_words)
        c_conf = sum(cw.conf for cw in current_cell_words) / len(current_cell_words)
        cells.append(Cell(text=c_text, bbox=c_bbox, conf=c_conf, words=list(current_cell_words)))

    line.cells = cells
    return cells


def _cols_overlap(c1: Cell, c2: Cell, tolerance: int = 15) -> bool:
    """Check if two cells in different lines share column horizontal bounds."""
    overlap = min(c1.right + tolerance, c2.right + tolerance) - max(c1.left - tolerance, c2.left - tolerance)
    return overlap > 0


def _match_row_to_columns(line: Line, col_spans: list[tuple[int, int]]) -> list[str]:
    """Map cells in a line to established global table column intervals."""
    row_values = [""] * len(col_spans)
    for cell in line.cells:
        best_col = -1
        best_overlap = -1
        for idx, (col_left, col_right) in enumerate(col_spans):
            overlap = min(cell.right, col_right) - max(cell.left, col_left)
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = idx

        if best_col == -1 or best_overlap <= 0:
            # Fall back to closest column center
            cell_mid = cell.x_center
            best_dist = float("inf")
            for idx, (col_left, col_right) in enumerate(col_spans):
                col_mid = (col_left + col_right) / 2.0
                dist = abs(cell_mid - col_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_col = idx

        if 0 <= best_col < len(row_values):
            existing = row_values[best_col]
            row_values[best_col] = (existing + " " + cell.text).strip() if existing else cell.text

    return row_values


def detect_layout_blocks(lines: list[Line]) -> list[dict[str, Any]]:
    """Segment lines into structured layout blocks: tables, headings, key-values, and paragraphs."""
    blocks: list[dict[str, Any]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Check for multi-column table candidate sequence
        if len(line.cells) >= 2:
            table_lines: list[Line] = [line]
            j = i + 1

            # Greedily absorb consecutive lines that align with column structure
            while j < n:
                next_line = lines[j]
                if next_line.page != line.page:
                    break

                # If vertical gap is too large (> 3x line height), table ended
                v_gap = next_line.top - table_lines[-1].bottom
                if v_gap > max(35, table_lines[-1].height * 3):
                    break

                # Check if next_line has multiple cells or aligns with existing columns
                matching_cols = 0
                for c_next in next_line.cells:
                    if any(_cols_overlap(c_next, c_prev) for prev in table_lines for c_prev in prev.cells):
                        matching_cols += 1

                if (len(next_line.cells) >= 2 and matching_cols >= 1) or matching_cols >= 2:
                    table_lines.append(next_line)
                    j += 1
                elif len(next_line.cells) == 1 and matching_cols == 1 and len(table_lines) >= 2:
                    # Allow single-cell row in table (e.g. Subtotal or spanned note)
                    table_lines.append(next_line)
                    j += 1
                else:
                    break

            # If we found at least 2 rows with multiple columns, construct a TableBlock
            if len(table_lines) >= 2:
                # Derive global column intervals by clustering cell X ranges
                all_cells = [c for tl in table_lines for c in tl.cells]
                # Sort unique left coordinates
                all_cells.sort(key=lambda c: c.left)

                # Cluster overlapping X intervals
                col_clusters: list[list[Cell]] = []
                for cell in all_cells:
                    matched_cluster = False
                    for cluster in col_clusters:
                        cluster_left = min(c.left for c in cluster)
                        cluster_right = max(c.right for c in cluster)
                        # Check overlap with cluster
                        overlap = min(cell.right, cluster_right) - max(cell.left, cluster_left)
                        if overlap > 0 or abs(cell.left - cluster_left) < 30:
                            cluster.append(cell)
                            matched_cluster = True
                            break
                    if not matched_cluster:
                        col_clusters.append([cell])

                # Sort clusters by average left position
                col_clusters.sort(key=lambda cluster: sum(c.left for c in cluster) / len(cluster))

                col_spans: list[tuple[int, int]] = []
                for cluster in col_clusters:
                    col_spans.append((
                        min(c.left for c in cluster),
                        max(c.right for c in cluster),
                    ))

                # If at least 2 distinct columns identified
                if len(col_spans) >= 2:
                    headers = _match_row_to_columns(table_lines[0], col_spans)
                    # If any header column is blank, give it a clean fallback
                    headers = [h if h else f"Column {idx + 1}" for idx, h in enumerate(headers)]

                    rows: list[list[str]] = []
                    for tl in table_lines[1:]:
                        row_vals = _match_row_to_columns(tl, col_spans)
                        rows.append(row_vals)

                    t_bbox = (
                        min(tl.left for tl in table_lines),
                        min(tl.top for tl in table_lines),
                        max(tl.right for tl in table_lines),
                        max(tl.bottom for tl in table_lines),
                    )
                    table_block = TableBlock(headers=headers, rows=rows, bbox=t_bbox, page=line.page)
                    blocks.append({
                        "type": "table",
                        "block": table_block,
                        "markdown": table_block.to_markdown(),
                        "page": line.page,
                    })
                    i = j
                    continue

        # Single line / Non-table block processing
        text = line.text.strip()
        if not text:
            i += 1
            continue

        # Check for Heading (e.g. ALL CAPS title, short line, or numbered section)
        is_upper = text.isupper() and len(text) > 3 and not re.search(r"\d{3,}", text)
        is_numbered_section = bool(re.match(r"^(Section|\d+\.|\d+\))\s+[A-Z]", text))

        if (is_upper and len(text) < 70) or is_numbered_section:
            level = 1 if len(text) < 35 and is_upper else 2
            blocks.append({
                "type": "heading",
                "level": level,
                "text": text,
                "markdown": f"{'#' * level} {text}",
                "page": line.page,
            })
        elif ":" in text and not text.startswith("http"):
            parts = text.split(":", 1)
            key, val = parts[0].strip(), parts[1].strip()
            if 1 < len(key) < 30 and val:
                blocks.append({
                    "type": "key_value",
                    "key": key,
                    "value": val,
                    "markdown": f"**{key}:** {val}",
                    "page": line.page,
                })
            else:
                blocks.append({
                    "type": "paragraph",
                    "text": text,
                    "markdown": text,
                    "page": line.page,
                })
        else:
            blocks.append({
                "type": "paragraph",
                "text": text,
                "markdown": text,
                "page": line.page,
            })

        i += 1

    return blocks


def render_markdown(blocks: list[dict[str, Any]]) -> str:
    """Render layout blocks into clean GitHub-flavored markdown."""
    if not blocks:
        return ""

    out_lines: list[str] = []
    current_page = None

    for b in blocks:
        page = b.get("page", 1)
        if current_page is not None and page != current_page:
            out_lines.append(f"\n<!-- Page {page} -->\n")
        current_page = page

        b_type = b.get("type")
        md = b.get("markdown", "")
        if b_type == "table":
            out_lines.append(md)
            out_lines.append("")
        elif b_type == "heading":
            out_lines.append(md)
            out_lines.append("")
        elif b_type == "key_value":
            out_lines.append(md)
        elif b_type == "paragraph":
            out_lines.append(md)
            out_lines.append("")

    return "\n".join(out_lines).strip() + "\n"


def parse_tsv_words(tsv: str, min_confidence: int = 40) -> list[WordToken]:
    """Parse Tesseract TSV output into a list of WordToken objects."""
    rows = [line.split("\t") for line in tsv.splitlines()]
    if not rows or rows[0][:1] != ["level"]:
        return []

    header = rows[0]
    index = {name: position for position, name in enumerate(header)}
    words: list[WordToken] = []

    for row in rows[1:]:
        if len(row) != len(header):
            continue
        level = row[index["level"]]
        if level != "5":
            continue

        text = row[index["text"]].strip()
        try:
            confidence = float(row[index["conf"]])
        except ValueError:
            continue

        if not text or confidence < min_confidence:
            continue

        left = int(float(row[index["left"]]))
        top = int(float(row[index["top"]]))
        width = int(float(row[index["width"]]))
        height = int(float(row[index["height"]]))
        page = int(float(row[index["page_num"]])) if "page_num" in index else 1

        words.append(WordToken(
            text=text,
            bbox=(left, top, left + width, top + height),
            conf=confidence,
            page=page,
        ))

    return words


def tsv_to_markdown(tsv: str, min_confidence: int = 40) -> str:
    """Directly convert Tesseract TSV output into GitHub-flavored markdown with table extraction."""
    words = parse_tsv_words(tsv, min_confidence=min_confidence)
    if not words:
        return ""
    lines = group_words_into_lines(words)
    blocks = detect_layout_blocks(lines)
    return render_markdown(blocks)
