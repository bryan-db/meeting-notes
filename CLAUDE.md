# Meeting Notes STT CLI

Rust CLI for real-time meeting transcription with speaker diarization and AI summaries.

## Build & Run

```bash
brew install opus pkg-config
cargo build --release

# Run from project root
./target/release/stt --help
./target/release/stt meeting -o "my-meeting"
./target/release/stt listen
./target/release/stt file recording.wav --diarize
```

Models (~500MB) auto-download on first run to `~/Library/Application Support/stt-cli/models/`.

## Architecture

### Source Layout

```
src/
  main.rs          # CLI entry, clap args, WhisperTranscriber, SpeakerEmbedder,
                   #   SpeakerClusterer, file transcription, summary generation
  events.rs        # MeetingEvent enum and serialization
  app_event.rs     # AppEvent enum for TUI event loop
  jsonl_writer.rs  # Append-only JSONL writer for events
  meeting/
    mod.rs         # ScreenCaptureKit system audio, mic capture, audio routing
    tui_mode.rs    # Orchestrates TUI meeting: audio threads + UI + post-processing
  tui/
    mod.rs         # Re-exports
    app.rs         # TuiApp state (transcript, markers, scroll, mode)
    ui.rs          # Ratatui rendering (transcript pane, status bar, input)
    input.rs       # Keyboard handling (q/m/n/s/S, scrolling)
```

### Key Dependencies

- **sherpa-rs** - Whisper Turbo (STT) + WeSpeaker (speaker embeddings), ONNX runtime
- **screencapturekit** - macOS system audio capture (remote meeting participants)
- **cpal** - Microphone input (your voice)
- **ratatui/crossterm** - Terminal UI
- **clap** - CLI argument parsing

### Data Flow

```
Mic (cpal) ──────────► 16kHz mono ──► Whisper ──► transcript segments
                                                        │
System Audio (SCK) ──► 16kHz mono ──► Whisper ──► transcript segments
                                          │             │
                                          ▼             ▼
                                    events.jsonl    TUI display
                                          │
                                    ┌─────┴─────┐
                                    ▼           ▼
                              Diarization   Claude Summary
                            (WeSpeaker)    (Anthropic API)
```

### Session Output

Meetings save to `~/Documents/meetings/<name>/`:

```
events.jsonl    # Raw transcript + markers + notes (append-only)
SUMMARY.md      # Claude-generated summary with action items
CONTEXT.md      # Pre-meeting context (user-created, improves summaries)
PROMPT.md       # Custom summary prompt override (optional)
screenshots/    # Captured during meeting
```

## Conventions

- Audio is processed **in-memory only** - never written to disk as audio files
- All STT uses 16kHz mono f32 samples (Whisper requirement)
- Speaker diarization runs post-meeting on system audio using cosine similarity (threshold 0.6)
- Summary uses Claude API via curl subprocess (no Rust HTTP client dependency)
- Summary model: `claude-sonnet-4-20250514`
- Events file is JSONL format with types: `segment`, `marker`, `manual`, `screenshot`

## macOS Permissions

- **Screen Recording** required for ScreenCaptureKit system audio capture
- Grant to your terminal app in System Settings > Privacy & Security > Screen Recording
- Permission issues produce "Device not configured (os error 6)" - reset with `tccutil reset ScreenCapture` if needed

## Lessons Learned

- ScreenCaptureKit permissions are flaky - permission resets don't always help
- Meeting apps (Meet, Zoom) take exclusive mic control - dual capture works around this by using SCK for system audio + cpal for mic separately
- Post-meeting transcription/diarization is more reliable than real-time
- sherpa-rs Whisper Turbo (int8 quantized) balances speed and accuracy well
- Keep audio in memory to avoid disk I/O and privacy concerns
