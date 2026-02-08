# Meeting Notes STT CLI

Rust CLI for real-time meeting transcription with speaker diarization and AI summaries.
Uses a **two-pass architecture**: Kyutai STT streaming for real-time display, Whisper Large V3 for accurate post-session transcription.

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

Models auto-download on first run (~3.5GB total) to `~/Library/Application Support/stt-cli/models/`.
Kyutai STT downloads from HuggingFace hub on first use.

## Architecture

### Two-Pass Transcription

**Pass 1 (real-time, during meeting):** Kyutai STT 2.6B streaming via moshi/candle with Metal GPU. True streaming model - no chunking artifacts, ~160-200ms latency, 24kHz input. For TUI visual feedback only.

**Pass 2 (post-session, after 'q'):** Whisper Large V3 via sherpa-rs/ONNX for accurate batch transcription of full audio kept in memory. Replaces draft segments in events.jsonl, preserving markers/notes/screenshots.

### Source Layout

```
src/
  main.rs          # CLI entry, clap args, WhisperTranscriber, SpeakerEmbedder,
                   #   SpeakerClusterer, retranscription, file transcription, summary
  kyutai.rs        # Kyutai STT 2.6B streaming (moshi/candle, Metal GPU)
  phrase_buffer.rs # Accumulates streaming words into natural phrases (800ms silence)
  events.rs        # MeetingEvent enum and serialization
  jsonl_writer.rs  # Append-only JSONL writer for events
  meeting/
    mod.rs         # Module exports
    tui_mode.rs    # TUI meeting: dual audio capture, Kyutai streaming, resampling
  tui/
    mod.rs         # Re-exports
    app.rs         # TuiApp state (transcript, markers, scroll, mode)
    ui.rs          # Ratatui rendering (transcript pane, status bar, input)
    input.rs       # Keyboard handling (q/m/n/s/S, scrolling)
```

### Key Dependencies

- **moshi/candle** - Kyutai STT 2.6B streaming (real-time, Metal GPU)
- **sherpa-rs** - Whisper Large V3 / Turbo (post-session STT) + WeSpeaker (speaker embeddings)
- **hf-hub** - HuggingFace model download for Kyutai
- **screencapturekit** - macOS system audio capture (remote meeting participants)
- **cpal** - Microphone input (your voice)
- **ratatui/crossterm** - Terminal UI
- **clap** - CLI argument parsing

### Data Flow

```
                        ┌─► 24kHz ──► Kyutai STT ──► draft segments ──► TUI display
Mic (cpal) ─► native ──┤                                   │
                        └─► 16kHz storage ──────────────────┼──► Whisper Large V3
                                                            │    (post-session)
                        ┌─► 24kHz ──► Kyutai STT ──► draft segments ──► TUI display
Sys (SCK) ──► 48kHz ───┤                                   │
                        └─► 16kHz storage ──────────────────┘
                                 │
                                 ▼
                        events.jsonl (draft)
                                 │ ──► q pressed ──► events.draft.jsonl (backup)
                                 │
                                 ▼
                        events.jsonl (Whisper Large V3 retranscription)
                                 │
                        ┌────────┴────────┐
                        ▼                 ▼
                  Diarization       Claude Summary
                  (WeSpeaker)      (Anthropic API)
```

### Session Output

Meetings save to `~/Documents/meetings/<name>/`:

```
events.jsonl       # Final transcript (Whisper Large V3) + markers + notes
events.draft.jsonl # Backup of draft Kyutai transcript (before retranscription)
SUMMARY.md         # Claude-generated summary with action items
CONTEXT.md         # Pre-meeting context (user-created, improves summaries)
PROMPT.md          # Custom summary prompt override (optional)
screenshots/       # Captured during meeting
```

## Conventions

- Audio is processed **in-memory only** - never written to disk as audio files
- Dual sample rates: 24kHz for Kyutai real-time, 16kHz for Whisper post-session storage
- Speaker diarization runs post-meeting on system audio using cosine similarity (threshold 0.6)
- Summary uses Claude API via curl subprocess (no Rust HTTP client dependency)
- Summary model: `claude-sonnet-4-20250514`
- Events file is JSONL format with types: `segment`, `marker`, `manual`, `screenshot`
- Kyutai gracefully degrades: if model unavailable, records audio only for post-session retranscription

## macOS Permissions

- **Screen Recording** required for ScreenCaptureKit system audio capture
- Grant to your terminal app in System Settings > Privacy & Security > Screen Recording
- Permission issues produce "Device not configured (os error 6)" - reset with `tccutil reset ScreenCapture` if needed

## Lessons Learned

- ScreenCaptureKit permissions are flaky - permission resets don't always help
- Meeting apps (Meet, Zoom) take exclusive mic control - dual capture works around this by using SCK for system audio + cpal for mic separately
- Kyutai STT 2.6B provides excellent real-time streaming (~6.4% WER, 160-200ms latency)
- Whisper Large V3 provides high accuracy for post-session batch (~7.4% WER)
- Two-pass architecture: fast streaming for UX, accurate batch for final output
- Keep audio in memory to avoid disk I/O and privacy concerns
