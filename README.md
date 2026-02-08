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

## Installation

### For LLM Agents

Follow these steps in order. Check each prerequisite before proceeding.

#### Step 1: Install Rust (if not installed)

```bash
# Check if cargo exists
which cargo

# If not found, install Rust via Homebrew
brew install rust

# Or via rustup (alternative)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Step 2: Clone and Build

```bash
git clone https://github.com/bryan-db/meeting-notes.git ~/meeting-notes
cd ~/meeting-notes
cargo build --release
```

#### Step 3: Install to PATH

```bash
# Option A: Using cargo install (recommended)
cargo install --path .

# Option B: Manual symlink (requires sudo)
sudo ln -sf ~/meeting-notes/target/release/stt /usr/local/bin/stt

# Option C: User-local install (no sudo)
mkdir -p ~/.local/bin
ln -sf ~/meeting-notes/target/release/stt ~/.local/bin/stt
# Add to PATH if not already: export PATH="$HOME/.local/bin:$PATH"
```

#### Step 4: Verify Installation

```bash
stt --help
```

#### Step 5: Grant Permissions (macOS)

The app needs Screen Recording permission for system audio capture:
1. Run `stt meeting` once (it will fail but trigger the permission prompt)
2. Go to **System Settings > Privacy & Security > Screen Recording**
3. Enable permission for the terminal app you're using

#### Step 6: Set API Key (Optional, for AI summaries)

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxx
# Add to ~/.zshrc or ~/.bashrc to persist
```

### Quick Start (for humans)

```bash
brew install rust
git clone https://github.com/bryan-db/meeting-notes.git ~/meeting-notes
cd ~/meeting-notes
cargo install --path .
stt meeting -o "my-meeting"
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
| `j/↓` | Scroll down |
| `k/↑` | Scroll up |
| `G` | Jump to bottom |
| `g` | Jump to top |

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

- **STT**: Whisper Turbo (whisper-large-v3-turbo, int8 quantized) via sherpa-rs
- **Speaker Embeddings**: WeSpeaker ResNet293-LM (0.45% EER)
- **Sample Rate**: 16kHz mono (Whisper requirement), resampled from device native rate
- **Segmentation**: Fixed 5-second buffered chunks (no VAD). Audio is accumulated and sent to Whisper on a timer regardless of speech activity. Silent chunks produce empty transcripts that are discarded.
- **Diarization**: Post-meeting only. WeSpeaker extracts embeddings per segment, clustered by cosine similarity (0.6 threshold).
- **Dual capture**: Mic via cpal, system audio via ScreenCaptureKit. Each source is transcribed independently.
