// Atomic JSONL writer for event logging
// Writes each event as a single line, flushing immediately for `tail -f` compatibility

use crate::events::Event;
use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct JsonlWriter {
    writer: BufWriter<File>,
    next_id: u64,
}

impl JsonlWriter {
    /// Create a new JSONL writer, creating or appending to the file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            next_id: 1,
        })
    }

    /// Get the next event ID
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Write an event to the file, returning the event with its ID
    pub fn write(&mut self, event: &Event) -> Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()?; // Flush immediately for tail -f
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::AudioSource;
    use chrono::Utc;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_jsonl_writer() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_events.jsonl");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        {
            let mut writer = JsonlWriter::new(&path).unwrap();

            let event = Event::SessionStart {
                id: writer.next_id(),
                ts: Utc::now(),
                session_id: "test_session".to_string(),
            };
            writer.write(&event).unwrap();

            let event = Event::Segment {
                id: writer.next_id(),
                ts: Utc::now(),
                src: AudioSource::Mic,
                text: "Hello".to_string(),
                start_ms: 1000,
                end_ms: 2000,
            };
            writer.write(&event).unwrap();
        }

        // Read back and verify
        let file = File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("session_start"));
        assert!(lines[1].contains("segment"));

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
