"""Batch Document OCR & Layout Detection CLI and Interactive App.

Scans PDF documents and images (e.g. property leases, receipts, invoices) to
structured GitHub-flavored markdown with multi-column table extraction.
Supports local NPU acceleration.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .desktop import capture, copy_text
from .ocr import extract_text, ocr_document_to_markdown


def result_window(result: dict):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Intel NPU Screenshot OCR")
    root.geometry("760x500")
    root.configure(bg="#181a1f")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#181a1f")
    style.configure("TLabel", background="#181a1f", foreground="#eef1f6")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=f"Intel NPU detected {result.get('npu_regions', 0)} text region(s)").pack(anchor="w", pady=(0, 10))
    box = tk.Text(frame, wrap="word", font=("Sans", 13), bg="#22252b", fg="#f4f6fa", insertbackground="white")
    box.pack(fill="both", expand=True)
    box.insert("1.0", result.get("text", ""))
    ttk.Button(frame, text="Copy text", command=lambda: copy_text(box.get("1.0", "end").strip())).pack(anchor="w", pady=(10, 0))
    root.mainloop()


def interactive_screenshot_mode():
    """Fallback interactive desktop screenshot OCR mode."""
    import tkinter as tk
    from tkinter import messagebox

    with tempfile.NamedTemporaryFile(prefix="intel-npu-ocr-", suffix=".png") as image:
        try:
            capture(Path(image.name), region=True)
        except Exception as exc:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Screenshot failed", str(exc))
            return
        if not Path(image.name).stat().st_size:
            return
        try:
            result = extract_text(Path(image.name))
            copy_text(result["text"])
            result_window(result)
        except Exception as exc:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("OCR failed", str(exc))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="intel-npu-ocr",
        description="Scan documents (PDF or image) to structured markdown with multi-column table extraction using local NPU acceleration.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more PDF or image files to scan (e.g. lease.pdf, receipt.png)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output markdown file path (defaults to stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "tables"],
        default="markdown",
        help="Output format: markdown (default), json, or tables (extract only tables)",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=40,
        help="Minimum OCR confidence score to accept (0-100, default: 40)",
    )
    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Print execution benchmark and latency metrics (e.g. seconds per page)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress status and timing messages on stderr",
    )
    return parser.parse_args(argv)


def filter_only_tables(markdown_text: str) -> str:
    """Extract only the markdown table segments from rendered markdown."""
    lines = markdown_text.splitlines()
    table_lines: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                table_lines.append("")
                in_table = False

    return "\n".join(table_lines).strip() + "\n"


def main(argv=None):
    args = parse_args(argv)

    # If no files specified: check for GUI environment
    if not args.files:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if has_display and (argv is None or len(sys.argv) == 1):
            interactive_screenshot_mode()
            return
        else:
            parse_args(["--help"])
            return

    all_outputs: list[str] = []
    all_metadata: list[dict] = []

    for file_path_str in args.files:
        file_path = Path(file_path_str)
        if not file_path.is_file():
            sys.stderr.write(f"Error: file not found: {file_path}\n")
            continue

        try:
            md_text, meta = ocr_document_to_markdown(file_path, min_confidence=args.min_confidence)
        except Exception as exc:
            sys.stderr.write(f"Error processing {file_path}: {exc}\n")
            continue

        all_metadata.append(meta)

        if args.benchmark and not args.quiet:
            sys.stderr.write(
                f"[intel-npu-ocr] {file_path.name}: {meta['pages']} page(s) in {meta['total_seconds']:.2f}s "
                f"({meta['seconds_per_page']:.2f}s/page, {meta['tables']} table(s))\n"
            )

        if args.format == "tables":
            md_text = filter_only_tables(md_text)

        all_outputs.append(md_text)

    if args.format == "json":
        result_payload = {
            "documents": all_metadata,
            "content": "\n\n".join(all_outputs),
        }
        output_data = json.dumps(result_payload, indent=2)
    else:
        output_data = "\n\n".join(all_outputs)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(output_data, encoding="utf-8")
        if not args.quiet:
            sys.stderr.write(f"[intel-npu-ocr] Wrote output to {args.output}\n")
    else:
        sys.stdout.write(output_data)


if __name__ == "__main__":
    main()
