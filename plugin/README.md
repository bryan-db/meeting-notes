# Meeting Recorder Plugin

Record and transcribe meetings with automatic transcription, speaker diarization, and AI-generated summaries with action items.

## Skills

### meeting-recorder

Record meetings with calendar integration for automatic folder naming.

**Triggers:** "record meeting", "transcribe meeting", "meeting notes", "capture meeting"

**Features:**
- Automatic meeting folder creation based on calendar events
- Real-time transcription using Whisper Turbo
- Speaker diarization (identifies different speakers)
- AI-generated summaries with action items
- Support for custom summarization prompts
- Screenshot capture during meetings

## Installation

1. Build the stt-cli binary:
   ```bash
   cd ~/Projects/phase0/utils/meeting-notes/stt-cli
   cargo build --release
   ```

2. Symlink to PATH (optional):
   ```bash
   sudo ln -sf ~/Projects/phase0/utils/meeting-notes/stt-cli/target/release/stt /usr/local/bin/stt
   ```

3. Models auto-download on first run (~500MB)

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
