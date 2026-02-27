// JSONL event types for meeting transcription
// These are persisted to disk and can be consumed by external tools (e.g., `tail -f | jq`)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Audio source identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSource {
    Mic,
    Sys,
}

impl std::fmt::Display for AudioSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioSource::Mic => write!(f, "MIC"),
            AudioSource::Sys => write!(f, "SYS"),
        }
    }
}

/// Event types that are written to the JSONL file
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Session started
    SessionStart {
        id: u64,
        ts: DateTime<Utc>,
        session_id: String,
    },

    /// Transcribed audio segment
    Segment {
        id: u64,
        ts: DateTime<Utc>,
        src: AudioSource,
        text: String,
        start_ms: u64,
        end_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker: Option<String>,
    },

    /// User-inserted marker (e.g., ACTION_ITEM, DECISION)
    Marker {
        id: u64,
        ts: DateTime<Utc>,
        offset_ms: u64,
        label: String,
    },

    /// User-inserted manual note
    Manual {
        id: u64,
        ts: DateTime<Utc>,
        offset_ms: u64,
        text: String,
    },

    /// Session ended
    SessionEnd {
        id: u64,
        ts: DateTime<Utc>,
        duration_ms: u64,
    },

    /// Screenshot captured
    Screenshot {
        id: u64,
        ts: DateTime<Utc>,
        offset_ms: u64,
        filename: String,
    },
}

impl Event {
    /// Get the offset in milliseconds from session start (for display ordering)
    pub fn offset_ms(&self) -> u64 {
        match self {
            Event::SessionStart { .. } => 0,
            Event::Segment { start_ms, .. } => *start_ms,
            Event::Marker { offset_ms, .. } => *offset_ms,
            Event::Manual { offset_ms, .. } => *offset_ms,
            Event::Screenshot { offset_ms, .. } => *offset_ms,
            Event::SessionEnd { duration_ms, .. } => *duration_ms,
        }
    }

    /// Format event for TUI display
    pub fn display_line(&self) -> String {
        match self {
            Event::SessionStart { session_id, .. } => {
                format!("─── Session: {} ───", session_id)
            }
            Event::Segment { src, text, start_ms, .. } => {
                let ts = format_ms(*start_ms);
                format!("[{}] [{}] {}", ts, src, text)
            }
            Event::Marker { offset_ms, label, .. } => {
                let ts = format_ms(*offset_ms);
                format!("[{}] [MRK] 📌 {}", ts, label)
            }
            Event::Manual { offset_ms, text, .. } => {
                let ts = format_ms(*offset_ms);
                format!("[{}] [NOTE] ✏️  {}", ts, text)
            }
            Event::SessionEnd { duration_ms, .. } => {
                let ts = format_ms(*duration_ms);
                format!("─── Session ended ({}) ───", ts)
            }
            Event::Screenshot { offset_ms, filename, .. } => {
                let ts = format_ms(*offset_ms);
                format!("[{}] [IMG] 📷 {}", ts, filename)
            }
        }
    }
}

fn format_ms(ms: u64) -> String {
    let secs = ms / 1000;
    let mins = secs / 60;
    let secs = secs % 60;
    format!("{:02}:{:02}", mins, secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = Event::Segment {
            id: 1,
            ts: Utc::now(),
            src: AudioSource::Mic,
            text: "Hello world".to_string(),
            start_ms: 5000,
            end_ms: 7500,
            speaker: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"segment\""));
        assert!(json.contains("\"src\":\"mic\""));
    }
}
