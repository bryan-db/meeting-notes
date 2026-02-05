# stt-cli

Real-time speech-to-text CLI with meeting transcription, speaker diarization, and interactive TUI. Powered by Kyutai STT with Metal acceleration on macOS.

## Features

- **Real-time transcription** from microphone or system audio
- **Dual-source meeting mode** - captures your voice (mic) + remote participants (system audio) simultaneously
- **Interactive TUI** with live transcript, markers, and notes
- **Screenshot capture** - press `s` to capture screen regions, or drag-and-drop images
- **Speaker diarization** - automatically identifies different speakers using pyannote
- **AI summaries** - generates meeting summaries using Claude API
- **JSONL event log** - structured output for LLM consumption (`tail -f events.jsonl | jq`)
- **Privacy-focused** - audio is processed in memory only, never written to disk

## Installation

### Prerequisites

- macOS (Metal GPU acceleration)
- Rust toolchain
- Python 3 with pyannote.audio (for diarization)

### Build

```bash
cargo build --release
```

### Diarization Setup

```bash
pip install pyannote.audio torch

# Set HuggingFace token (get from https://huggingface.co/settings/tokens)
# Must accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
export HUGGINGFACE_TOKEN=hf_xxxxx
```

### Summary Setup (Optional)

```bash
# Set Anthropic API key for AI summaries
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

## Usage

### Meeting Mode (Default)

Record meetings with TUI, automatic speaker diarization, and AI summary:

```bash
# Start meeting - creates ~/Documents/meetings/MOTOR/
stt meeting -o "MOTOR"

# Auto-generated folder name
stt meeting
# → ~/Documents/meetings/meeting_20260205_143000/

# Specify full path
stt meeting -o /path/to/session

# Disable diarization
stt meeting -o "MOTOR" --no-diarize

# Disable AI summary
stt meeting -o "MOTOR" --no-summary

# Plain text mode (no TUI)
stt meeting -o "MOTOR" --no-tui

# Specify microphone
stt meeting -o "MOTOR" --mic "Anker"

# List available audio devices
stt meeting --list-devices
```

### TUI Keybindings

| Key | Action |
|-----|--------|
| `s` | Screenshot (interactive region select) |
| `S` | Screenshot (window select) |
| `m` | Add marker (type label, Enter to submit) |
| `n` | Add note (type text, Enter to submit) |
| `j/k` or `↓/↑` | Scroll transcript |
| `G` | Jump to bottom (enable auto-scroll) |
| `g` | Jump to top |
| `q` | Quit |

**Drag-and-drop**: Drag an image file onto the terminal to import it.

### File Transcription

Transcribe audio/video files:

```bash
stt file recording.m4a
stt file meeting.wav -o transcript.md
stt file video.mp4 --format json --words
```

### Real-time Listen Mode

Simple single-source transcription:

```bash
stt listen
stt listen --device "MacBook Pro Microphone"
```

## Session Folder Structure

```
~/Documents/meetings/MOTOR/
├── CONTEXT.md        # (Optional) Meeting context from calendar agent
├── events.jsonl      # Transcript + events (source of truth)
├── SUMMARY.md        # AI-generated meeting summary
└── screenshots/
    ├── 001.png
    ├── 002.png
    └── ...
```

**Note**: Audio is processed in memory and never saved to disk. This is intentional for privacy - only the text transcript is persisted.

If a `CONTEXT.md` file exists in the session folder (e.g., created by a calendar agent with meeting details, attendees, agenda), it will be included in the summarization prompt for better context.

## JSONL Event Format

Each line is a JSON object. LLMs can consume this with `tail -f events.jsonl`:

```jsonl
{"type":"session_start","id":1,"ts":"2026-02-05T14:30:00Z","session_id":"mtg_20260205_143000"}
{"type":"segment","id":2,"ts":"2026-02-05T14:30:05Z","src":"mic","text":"Good morning","start_ms":5000,"end_ms":7000}
{"type":"segment","id":3,"ts":"2026-02-05T14:30:08Z","src":"sys","speaker":"SPEAKER_00","text":"Hi, let's get started","start_ms":8000,"end_ms":10000}
{"type":"marker","id":4,"ts":"2026-02-05T14:30:15Z","offset_ms":15000,"label":"ACTION_ITEM"}
{"type":"manual","id":5,"ts":"2026-02-05T14:30:20Z","offset_ms":20000,"text":"Follow up with DevOps"}
{"type":"screenshot","id":6,"ts":"2026-02-05T14:30:25Z","offset_ms":25000,"filename":"001.png"}
{"type":"session_end","id":7,"ts":"2026-02-05T15:00:00Z","duration_ms":1800000}
```

### Event Types

| Type | Description |
|------|-------------|
| `session_start` | Meeting began |
| `segment` | Transcribed speech (`src`: mic/sys, `speaker`: after diarization) |
| `marker` | User-inserted marker (ACTION_ITEM, DECISION, etc.) |
| `manual` | User-inserted note |
| `screenshot` | Captured or imported image |
| `session_end` | Meeting ended |

## Audio Sources

- **mic**: Your microphone input (your voice)
- **sys**: System audio via ScreenCaptureKit (remote meeting participants)

For system audio capture, the app uses macOS ScreenCaptureKit which requires Screen Recording permission.

## Options

```
--cpu              Use CPU instead of Metal GPU
--model <MODEL>    HuggingFace model [default: kyutai/stt-2.6b-en-candle]
--no-tui           Disable TUI, use plain text output
--no-diarize       Skip automatic speaker diarization
--no-summary       Skip AI summary generation
--hf-token <TOKEN> HuggingFace token for pyannote (or set HUGGINGFACE_TOKEN env)
--anthropic-key    Anthropic API key for summaries (or set ANTHROPIC_API_KEY env)
```

## Privacy

Audio is kept in memory during the meeting for real-time transcription and speaker diarization. When the meeting ends:

1. Audio is piped directly to pyannote for speaker identification
2. Speaker labels are added to the transcript
3. Audio buffers are dropped (freed from memory)
4. **No audio files are ever written to disk**

Only the text transcript (events.jsonl), screenshots, and summary are persisted.

## Tips

- **Watch live**: `tail -f events.jsonl | jq -c .`
- **Filter by speaker**: `jq 'select(.speaker == "SPEAKER_00")' events.jsonl`
- **Extract action items**: `jq 'select(.type == "marker" and .label == "ACTION_ITEM")' events.jsonl`
- **Get full transcript**: `jq -r 'select(.type == "segment") | "[\(.src)] \(.text)"' events.jsonl`

## Memory Usage

Audio buffers use approximately:
- ~48 KB/second per source (24kHz mono, 32-bit float)
- ~170 MB/hour per source
- ~340 MB/hour for dual-source (mic + system)

A typical 1-hour meeting uses ~350 MB of RAM for audio buffers.
