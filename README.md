# Meeting Notes CLI

Real-time meeting transcription with speaker diarization and AI-generated summaries. Built for macOS with Whisper Turbo + WeSpeaker.

## Features

- **Real-time transcription** using Whisper Turbo (whisper-large-v3-turbo)
- **Speaker diarization** with WeSpeaker embeddings (pure Rust, no Python)
- **Dual-source capture** - mic (you) + system audio (remote participants)
- **Interactive TUI** with live transcript, markers, and notes
- **AI summaries** with action item extraction using Claude
- **Screenshot capture** during meetings
- **Auto-downloading models** (~500MB, downloaded on first run)

## Quick Start

```bash
# Build
cargo build --release

# Install to PATH (optional)
sudo ln -sf $(pwd)/target/release/stt /usr/local/bin/stt

# Record a meeting
stt meeting -o "weekly-sync"

# Models download automatically on first run
```

## Installation for LLMs/Agents

To install this tool, run:

```bash
# Clone the repository
git clone https://github.com/databricks-field-eng/meeting-notes.git
cd meeting-notes

# Build
cargo build --release

# Add to PATH
sudo ln -sf $(pwd)/target/release/stt /usr/local/bin/stt

# Set API key for summaries
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

Models (~500MB) download automatically on first run to `~/Library/Application Support/stt-cli/models/`.

## Usage

### Meeting Mode

```bash
# Start meeting with TUI
stt meeting -o "project-standup"

# Auto-generated folder name
stt meeting
# → ~/Documents/meetings/meeting_20250205_143000/

# List audio devices
stt meeting --list-devices

# Skip diarization or summary
stt meeting -o "quick-call" --no-diarize --no-summary
```

### TUI Keybindings

| Key | Action |
|-----|--------|
| `q` | Quit and generate summary |
| `m` | Add marker (ACTION_ITEM, DECISION, etc.) |
| `n` | Add manual note |
| `s` | Screenshot (region select) |
| `S` | Screenshot (window select) |
| `↑/↓` | Scroll transcript |
| `PgUp/PgDn` | Page scroll |

### Listen Mode

Simple real-time transcription:

```bash
stt listen
stt listen --device "AirPods"
```

### File Transcription

```bash
stt file recording.wav
stt file meeting.m4a --diarize
```

## Meeting Folder Structure

```
~/Documents/meetings/weekly-sync/
├── CONTEXT.md        # Meeting context (improves summary quality)
├── PROMPT.md         # Custom summary prompt (optional)
├── events.jsonl      # Raw transcript + events
├── SUMMARY.md        # AI-generated summary with action items
└── screenshots/
    └── 001.png
```

### CONTEXT.md

Create this **before** the meeting starts for better summaries:

```markdown
# Meeting Context

## Meeting Title
Weekly Engineering Sync

## Attendees
- Alice (Engineering Lead)
- Bob (Backend)
- Charlie (Frontend)

## Purpose
Review sprint progress and blockers
```

### PROMPT.md (Optional)

Override the default summary prompt:

```markdown
Focus on technical decisions and action items only.
Skip status updates and small talk.
Use bullet points, no prose.
```

## Claude Code Skill

A skill is included for Claude Code agents at `plugin/skills/meeting-recorder/SKILL.md`.

The skill enables agents to:
1. Check calendar for current meeting
2. Create meeting folder with attendees
3. Write CONTEXT.md from calendar details
4. Run stt meeting in foreground
5. Report summary location when done

**Trigger phrases**: "record meeting", "transcribe meeting", "meeting notes"

## Requirements

- **macOS** (uses ScreenCaptureKit for system audio)
- **Screen Recording permission** (System Settings > Privacy > Screen Recording)
- **ANTHROPIC_API_KEY** for AI summaries (optional)

## Privacy

- Audio is processed **in memory only** - never written to disk
- Transcription uses local ONNX models (Whisper + WeSpeaker)
- Only text transcript and screenshots are saved
- Summary generation requires Anthropic API (optional)

## Technical Details

- **STT**: Whisper Turbo (whisper-large-v3-turbo, int8 quantized)
- **Speaker Embeddings**: WeSpeaker ResNet293-LM (0.45% EER)
- **Sample Rate**: 16kHz (Whisper requirement)
- **Transcription**: Periodic (every 5 seconds of audio)
- **Clustering**: Cosine similarity with 0.6 threshold
