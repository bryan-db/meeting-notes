---
name: meeting-recorder
description: Record and transcribe meetings with automatic folder naming from calendar, AI summaries, and action item extraction
---

# Meeting Recorder Skill

Record meetings with real-time transcription, speaker diarization, and AI-generated summaries.

## Prerequisites

- `stt` binary in PATH (or at `~/meeting-notes/target/release/stt`)
- ANTHROPIC_API_KEY environment variable
- Screen Recording permission granted

## Installation (if not installed)

```bash
git clone https://github.com/bryan-db/meeting-notes.git ~/meeting-notes
cd ~/meeting-notes
cargo build --release
sudo ln -sf $(pwd)/target/release/stt /usr/local/bin/stt
```

Models (~500MB) auto-download on first run.

## Instructions

### 1. Check for Current Meeting (Optional)

If the user has Google Calendar integration, use the google-calendar skill to find the current or upcoming meeting:

```
Look for meetings starting within the next 15 minutes or currently in progress.
Extract: title, attendees, description
```

If no calendar access, ask the user for the meeting name and attendees.

### 2. Create Meeting Folder

Create folder at `~/Documents/meetings/` with format:
```
YYYY-MM-DD_<sanitized-title>_<attendees>/
```

Sanitize names:
- Lowercase, replace spaces with hyphens
- Remove special characters except hyphens
- Truncate to 50 chars max per component
- Example: `2025-02-05_weekly-sync_alice-bob/`

```bash
mkdir -p ~/Documents/meetings/<folder-name>
```

### 3. Create CONTEXT.md (IMPORTANT)

**Always create this file before starting the recording.** It provides context for better summaries.

Write to `<meeting-folder>/CONTEXT.md`:

```markdown
# Meeting Context

## Meeting Title
[Title from calendar or user]

## Date
[Current date/time]

## Attendees
- [List attendees, one per line]

## Purpose
[From calendar description or user input]

## Agenda/Topics
- [Key topics to discuss]

## Background
[Any relevant context]
```

### 4. Create PROMPT.md (Optional)

If the user wants custom summarization, create `<meeting-folder>/PROMPT.md`:

```markdown
[Custom instructions for summarization]

Examples:
- "Focus on technical decisions and architecture"
- "Emphasize customer feedback and action items"
- "Summarize as bullet points, no prose"
```

### 5. Start Recording

Run the stt command in the **foreground** (the user needs to see the TUI):

```bash
stt meeting --output ~/Documents/meetings/<folder-name>
```

**Tell the user:**
- Press `q` to end the meeting and generate summary
- Press `m` to add markers (ACTION_ITEM, DECISION, QUESTION)
- Press `n` to add manual notes
- Press `s` to capture screenshots

### 6. After Meeting Ends

When the user presses `q`, the tool automatically:
1. Runs speaker diarization
2. Generates AI summary with action items
3. Saves SUMMARY.md

Report the results:
```
Meeting notes saved to: ~/Documents/meetings/<folder-name>/

Files created:
- SUMMARY.md - AI summary with action items
- events.jsonl - Full transcript
- CONTEXT.md - Meeting context

Key sections in SUMMARY.md:
- Overview
- Key Discussion Points
- Decisions Made
- Action Items (with owners and due dates)
- Next Steps
```

## Example Flows

### With Calendar
```
User: "Record my meeting"

1. Use google-calendar skill to get current meeting
2. Found: "API Design Review" with alice@co.com, bob@co.com
3. Create: ~/Documents/meetings/2025-02-05_api-design-review_alice-bob/
4. Write CONTEXT.md with calendar details
5. Run: stt meeting --output <folder>
6. User presses q when done
7. Report: "Summary saved to SUMMARY.md"
```

### Manual
```
User: "Record meeting with sales team about Q1 planning"

1. Create: ~/Documents/meetings/2025-02-05_q1-planning_sales-team/
2. Write CONTEXT.md with provided context
3. Run: stt meeting --output <folder>
```

### Custom Prompt
```
User: "Record standup, just list blockers and action items"

1. Create meeting folder
2. Write CONTEXT.md
3. Write PROMPT.md with: "Focus only on blockers and action items. Use bullet points. Skip status updates and small talk."
4. Run: stt meeting --output <folder>
```

## TUI Keyboard Reference

| Key | Action |
|-----|--------|
| q | Quit and process meeting |
| m | Add marker (ACTION_ITEM, DECISION, etc.) |
| n | Add manual note |
| s | Screenshot (interactive selection) |
| S | Screenshot (window selection) |
| Up/Down | Scroll transcript |
| PgUp/PgDn | Page scroll |

## Troubleshooting

- **Models not found**: They auto-download on first run (~500MB)
- **No audio**: Grant Screen Recording permission in System Settings
- **Summary failed**: Check ANTHROPIC_API_KEY is set
- **0 speakers**: Ensure there's actual speech in system audio
