# Meeting Recorder Plugin

Record and transcribe meetings with automatic transcription, speaker diarization, and AI-generated summaries with action items.

## Skills

### meeting-recorder

Record meetings with calendar integration for automatic folder naming.

**Triggers:** "record meeting", "transcribe meeting", "meeting notes", "capture meeting"

**Features:**
- Automatic meeting folder creation based on calendar events
- Real-time transcription using Whisper Turbo via sherpa-rs (5-second buffered chunks, no VAD)
- Post-meeting speaker diarization with WeSpeaker embeddings
- AI-generated summaries with action items (Claude API)
- Support for custom summarization prompts (PROMPT.md)
- Screenshot capture and drag-and-drop image import during meetings

## Installation

```bash
# Clone and build
git clone https://github.com/bryan-db/meeting-notes.git ~/meeting-notes
cd ~/meeting-notes
cargo build --release

# Add to PATH
sudo ln -sf $(pwd)/target/release/stt /usr/local/bin/stt

# Set API key for summaries
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

Models (~500MB) auto-download on first run.

## Requirements

- macOS (uses ScreenCaptureKit for system audio)
- Screen Recording permission (System Settings > Privacy)
- ANTHROPIC_API_KEY for AI summaries
- Google Calendar access for meeting detection (optional)

## Output Structure

```
~/Documents/meetings/YYYY-MM-DD_meeting-name/
├── CONTEXT.md          # Meeting context (pre-meeting)
├── PROMPT.md           # Custom prompt (optional)
├── events.jsonl        # Raw transcription
├── SUMMARY.md          # AI summary with action items
└── screenshots/        # Captured screenshots
```
