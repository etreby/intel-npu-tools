# Intel NPU Tools for Linux

An unofficial, community-maintained toolkit that makes the integrated **Intel AI Boost NPU** in Arrow Lake processors useful on Linux. It provides private semantic search, local speech transcription, screenshot OCR, hardware verification, and twelve MCP tools that AI agents can call.

[![Validate](https://github.com/etreby/intel-npu-tools/actions/workflows/validate.yml/badge.svg)](https://github.com/etreby/intel-npu-tools/actions/workflows/validate.yml)
[![Release](https://img.shields.io/github/v/release/etreby/intel-npu-tools)](https://github.com/etreby/intel-npu-tools/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> This project is not affiliated with, sponsored by, or endorsed by Intel Corporation. Intel, Intel Core, OpenVINO, and Intel AI Boost are trademarks of their respective owners.

## Why this exists

Linux can expose an Arrow Lake NPU as `/dev/accel/accel0`, but applications still need Intel's Level Zero NPU user-mode driver, OpenVINO, compatible models, permissions, and integration code. This repository assembles those pieces into useful desktop and agent workflows.

The NPU is a good fit for efficient background inference. It does **not** replace a discrete GPU for model training, large language models, or image generation.

## Features

- **Speech to text:** multilingual Whisper Base INT8 runs locally on the NPU. Whisper Small is available for noisier rooms; see the [roadmap](docs/ROADMAP.md) for the accuracy and latency trade. Whole recordings get timestamped output: `intel-npu-transcribe` turns a meeting recording into VTT, SRT and Markdown in seconds.
- **Screenshot OCR:** select a region and copy recognized English or Arabic text.
- **Private semantic search:** index local documents, logs, and source code with Qwen3-Embedding 0.6B INT8, then retrieve passages by meaning.
- **Smaller agent context:** `context_filter` returns only the lines of a large log or file that answer a question, verbatim and with line numbers, and `screen_to_text` renders a screen as a few hundred tokens of structured text rather than a multi-thousand-token image.
- **MCP server:** Codex, Claude, Gemini CLI, AGY/Antigravity CLI, Hermes, Antigravity IDE, OpenCode, and other MCP clients can use all twelve tools.
- **Hardware diagnostics:** report every OpenVINO device and confirm `Intel(R) AI Boost` is available.
- **Reversible installation:** user applications and models are isolated under `~/.local`; the uninstaller deliberately preserves system drivers.

## Supported hardware and software

The initial tested target is:

- Intel Arrow Lake integrated NPU 3720, PCI ID `8086:ad1d`
- Intel Core Ultra 200-series desktop processors, including Core Ultra 9 285K
- Ubuntu 24.04 or an Ubuntu 24.04-compatible distribution
- Linux kernel 6.8 or newer with `intel_vpu`
- KDE Plasma, GNOME, COSMIC, or a wlroots compositor; X11 works too. Screenshots use whichever of spectacle, gnome-screenshot, cosmic-screenshot, grim, maim or scrot is installed, and the clipboard uses wl-clipboard, xclip or xsel.

OpenVINO officially identifies Arrow Lake's NPU 3720 by PCI ID `0xAD1D`. Other Intel NPU generations may work with code changes, but the bundled driver safety check intentionally refuses unknown PCI IDs.

## What it looks like

The control panel is where every feature can be tried and every setting changed.

![Voice tab: record from the microphone or transcribe a file, on the NPU](docs/images/panel-voice.png)

Search indexes a folder and retrieves passages by meaning, and `context_filter`
returns only the lines of one large file that answer a question.

![Search tab: semantic search across an indexed folder](docs/images/panel-search.png)

Settings that used to be environment variables are editable and persistent. Each
option states what it measured, so the trade is visible before it is made.

![Config tab: speech model, compiled-model caching, and turbo](docs/images/panel-config.png)

The status tab reports the device, the driver, which models are installed, and
what the desktop session can actually do — useful when a screenshot or the
clipboard is not working.

![Status tab: OpenVINO devices, NPU driver and compiler, installed models, desktop capabilities](docs/images/panel-status.png)

Regenerate these with `scripts/capture-screenshots.sh`. It runs each window on a
virtual display, so the images do not depend on anyone's wallpaper or theme and
stay reproducible as the interface changes.

## Install from a package

Native packages install the toolkit system-wide. They deliberately contain no
models and no OpenVINO runtime: those are large, are redistributed under their
own licences, and OpenVINO is not in any distribution archive. Each user runs
`intel-npu-tools-setup` once afterwards to build their own environment and
download the models, which also keeps package installation off the network.

```bash
./scripts/build-deb.sh                                  # Debian, Ubuntu, Pop!_OS
cd packaging && makepkg -si                             # Arch
rpmbuild -bb packaging/intel-npu-tools.spec \
         --define "_projectdir $PWD"                    # Fedora (openSUSE and EL untested)
```

All three install the same tree from `scripts/stage-package.sh`, so they cannot
drift apart. Then, as your own user:

```bash
intel-npu-tools-setup            # add --with-reranker and --with-whisper-small if wanted
```

## Quick installation

```bash
git clone https://github.com/etreby/intel-npu-tools.git
cd intel-npu-tools
./install.sh --with-driver
```

`--with-driver` installs Intel's signed Ubuntu 24.04 NPU user-mode packages and firmware, adds the current user to `render`, creates an isolated Python environment, downloads the models from their official upstream locations, and installs the desktop and MCP tools.

Log out and back in after the first driver installation, then verify:

```bash
intel-npu-info
```

Expected output includes:

```json
"NPU": "Intel(R) AI Boost"
```

If the driver is already installed, omit `--with-driver`:

```bash
./install.sh
```

To skip automatic MCP client registration:

```bash
./install.sh --without-mcp
```

The model download is approximately 800 MB in total, including Whisper, OCR, and the roughly 600 MB embedding model. Two optional models are excluded by default: `--with-whisper-small` (~250 MB, better in noise) and `--with-reranker` (~300 MB, sharper search results).

## Semantic search in 30 seconds

```bash
intel-npu-search index ~/Projects/my-project
intel-npu-search search "Where is authentication configured?"
intel-npu-search status
```

Indexing is incremental and remains local. See the [semantic-search guide](docs/SEMANTIC_SEARCH.md) for supported files, performance, privacy boundaries, and customization.

## Desktop usage

Launch these applications from the desktop menu:

- **Intel NPU Speech to Text:** click Start, speak, then Stop and transcribe. The result is copied to the clipboard.
- **Intel NPU Screenshot OCR:** select a rectangular region. Recognized text is displayed and copied.
- **Intel NPU Control Panel:** try every feature, change settings, and see what the NPU and the desktop session can actually do. Run `intel-npu-panel`.

On KDE, `install.sh` registers two global shortcuts:

- `Meta+F9` — Speech to Text
- `Meta+Alt+O` — Screenshot OCR

`Meta` is normally the Windows-logo key. **The shortcuts start working after your next login**, because KDE's shortcut daemon reads its configuration once at session start; restarting it during an install would briefly drop every other shortcut on the system. If you had already bound either application to a key of your own, the installer leaves your binding alone, and it refuses to write a key another component already owns rather than registering one KDE would silently discard. Speech uses `Meta+F9` rather than the `Meta+Alt+S` of earlier versions, because KDE's accessibility component binds `Meta+Alt+S` to "Toggle Screen Reader On and Off" by default, so that shortcut could never have worked. On a desktop without KDE's configuration tools the registration is skipped and the applications are launched from the desktop menu instead.

## Transcribing recordings

Dictation in the Speech to Text app answers *what was said*. `intel-npu-transcribe` answers the question a meeting or lecture recording raises afterwards: *what was said, and when*. It reads a file, streams the audio through the NPU in Whisper-sized windows cut at the nearest pause, and returns timestamped segments, writing WebVTT, SubRip and Markdown renderings next to it:

```bash
intel-npu-transcribe meeting.wav
# writes meeting.vtt, meeting.srt and meeting.md in the same directory
intel-npu-transcribe podcast.opus -o ~/Documents/transcripts --formats srt,md
```

Each Markdown paragraph carries the time it begins, and the header records the detected language and duration, so the transcript reads like notes rather than a teleprompter script. Timestamps come from the model itself (`return_timestamps`), shifted onto the file's timeline by the chunker, and chunks with no speech in them are skipped rather than transcribed, which is what keeps an empty conference room from becoming invented dialogue. The file formats are whatever ffmpeg reads: WAV, MP3, M4A, FLAC, OGG, OPUS and WEBM all work.

Measured on Arrow Lake with Whisper Base: a 10.1 minute WAV transcribed in 12.3 seconds of wall time — about thirty times faster than real time — at 3.3 percent word error rate against a reference script read by a natural-voice synthesizer, with the machine's discrete GPUs untouched throughout. Whisper Small can be selected for noisier rooms the same way as for dictation: `INTEL_NPU_TOOLS_WHISPER_MODEL=whisper-small-int8-ov`.

MCP agents get the same outputs from the `transcribe_audio` tool by passing `formats="vtt,srt,md"`; the reply then contains the language, duration, every timestamped segment, and the rendered documents.

## AI agent and MCP usage

The local stdio MCP command is:

```bash
intel-npu-mcp
```

It exposes:

| Tool | Purpose |
| --- | --- |
| `npu_status` | Verify OpenVINO and list available devices |
| `transcribe_audio` | Transcribe a local audio file on the NPU |
| `record_and_transcribe` | Record the default microphone for a bounded duration |
| `ocr_image` | Extract English/Arabic text from an image |
| `ocr_current_monitor` | Capture and OCR the current monitor |
| `screen_to_text` | Read a screen as structured text instead of an image |
| `semantic_index` | Incrementally index a text file or directory on the NPU |
| `semantic_search` | Retrieve ranked local passages by meaning |
| `context_filter` | Return only the parts of a large file that answer a question |
| `semantic_index_status` | Show indexed roots, files, chunks, and database path |
| `open_speech_app` | Open the interactive speech application |
| `open_ocr_selector` | Open interactive region OCR |

Example agent requests:

```text
Use intel-npu-tools to transcribe ~/recording.m4a.
Use the NPU to OCR ~/Pictures/error.png.
Record my microphone for 15 seconds and transcribe it.
Read the text currently visible on my monitor.
Index ~/Projects/my-project, then find where authentication is configured.
Search my indexed documents for the Windows boot recovery procedure.
```

Manual Codex registration:

```bash
codex mcp add intel-npu-tools -- "$HOME/.local/bin/intel-npu-mcp"
```

Manual Claude Code registration:

```bash
claude mcp add --scope user intel-npu-tools -- "$HOME/.local/bin/intel-npu-mcp"
```

Manual Hermes registration:

```bash
hermes mcp add intel-npu-tools --command "$HOME/.local/bin/intel-npu-mcp"
```

Manual Gemini CLI registration:

```bash
gemini mcp add --scope user intel-npu-tools "$HOME/.local/bin/intel-npu-mcp"
```

AGY/Antigravity CLI reads global servers from `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "intel-npu-tools": {
      "command": "/home/YOUR_USER/.local/bin/intel-npu-mcp",
      "args": []
    }
  }
}
```

OpenCode reads global configuration from `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "intel-npu-tools": {
      "type": "local",
      "command": ["/home/YOUR_USER/.local/bin/intel-npu-mcp"],
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

Any MCP client can use this stdio configuration:

```json
{
  "mcpServers": {
    "intel-npu-tools": {
      "command": "/home/YOUR_USER/.local/bin/intel-npu-mcp",
      "args": []
    }
  }
}
```

## How the NPU is utilized

```text
Microphone/audio ──> Whisper Base INT8 ──> OpenVINO GenAI ──> Intel NPU
Long audio file ──> pause-aligned chunks ──> the same Whisper pipeline ──> VTT/SRT/Markdown
Screenshot/image ──┬─> text detector + recognizer ──> OpenVINO ──> Intel NPU
                   └─> Tesseract (preferred for layout, punctuation, Arabic)
Local text ──> chunks ──> Qwen3 Embedding INT8 ──> Intel NPU ──> SQLite vectors
AI agent ──> local stdio MCP server ──> the same NPU pipelines
```

All inference targets `NPU` explicitly. The included tools do not silently redirect workloads to a discrete GPU. This allows an NVIDIA or Intel GPU to remain available for gaming, rendering, Ollama, or larger AI workloads.

## Privacy

- No network server is started.
- MCP communication uses a local child process over stdin/stdout.
- Audio, screenshots, indexed text, and embeddings are processed and stored locally.
- Network access is needed only during installation to download software and models.
- Temporary recordings and screenshots are deleted after processing.

## Documentation

- [Semantic search, customization, and measured performance](docs/SEMANTIC_SEARCH.md)
- [Troubleshooting and debugging](docs/TROUBLESHOOTING.md)
- [Building agent skills](docs/BUILDING_SKILLS.md)
- [Ready-to-copy local-knowledge skill](examples/skills/intel-npu-local-knowledge/SKILL.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Troubleshooting

### `/dev/accel/accel0` is missing

```bash
lspci -nn | grep -i ad1d
lsmod | grep intel_vpu
journalctl -k -b | grep -i intel_vpu
```

Update the kernel/firmware for your distribution before replacing unrelated graphics drivers.

### NPU exists but OpenVINO shows only CPU/GPU

```bash
groups
ls -l /dev/accel/accel0
```

The user must belong to `render`. Log out and back in after group changes.

### Speech produces no text

Check the default PipeWire microphone:

```bash
pactl get-default-source
pw-record --rate 16000 --channels 1 /tmp/microphone-test.wav
```

### OCR limitations

Both engines run on every image. The NPU detects text regions and recognizes them with Intel's compact model, which covers only lowercase Latin letters and digits. Tesseract reads punctuation, layout, English, and Arabic, so its output is used as the returned text whenever it produces any; the NPU result is returned when Tesseract is missing, fails, or finds nothing. `npu_regions` and `npu_text` are always reported separately so you can see what the NPU contributed. Stylized fonts and very small text may remain imperfect.

`install.sh` installs Tesseract. Without it, OCR still works but falls back to the NPU-only text.

For NPU compilation, semantic-search, and MCP diagnostics, use the complete [debugging guide](docs/TROUBLESHOOTING.md).

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m compileall -q src
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and proposed areas of work.

## Uninstall

```bash
./uninstall.sh
```

The uninstaller removes user applications, models, and MCP registrations. It intentionally preserves system-level NPU firmware and drivers.

## Upstream components and documentation

- [Intel Linux NPU Driver](https://github.com/intel/linux-npu-driver)
- [OpenVINO NPU plugin](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu)
- [OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [OpenVINO Whisper Base INT8](https://huggingface.co/OpenVINO/whisper-base-int8-ov)
- [OpenVINO Qwen3 Embedding 0.6B INT8](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov)
- [Open Model Zoo OCR tutorial](https://docs.openvino.ai/2024/notebooks/optical-character-recognition-with-output.html)

## License

Project code is released under the [MIT License](LICENSE). Downloaded drivers, models, runtimes, and trademarks remain under their respective upstream licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
