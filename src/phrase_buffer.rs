// PhraseBuffer: accumulates streaming words into natural phrases
// Words are buffered until a silence gap (800ms default) is detected,
// then flushed as a complete phrase for display in the TUI.

use std::time::{Duration, Instant};

pub struct PhraseBuffer {
    words: Vec<String>,
    first_word_time: Option<Instant>,
    last_word_time: Option<Instant>,
    silence_threshold: Duration,
}

impl PhraseBuffer {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            first_word_time: None,
            last_word_time: None,
            silence_threshold: Duration::from_millis(800),
        }
    }

    /// Add a word to the buffer
    pub fn add_word(&mut self, word: String, time: Instant) {
        if self.first_word_time.is_none() {
            self.first_word_time = Some(time);
        }
        self.last_word_time = Some(time);
        self.words.push(word);
    }

    /// Check if the buffer should be flushed (silence gap exceeded)
    pub fn should_flush(&self) -> bool {
        if let Some(last) = self.last_word_time {
            !self.words.is_empty() && last.elapsed() >= self.silence_threshold
        } else {
            false
        }
    }

    /// How long since the last word was added (in milliseconds)
    pub fn quiet_for_ms(&self) -> u64 {
        self.last_word_time
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(u64::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Flush the buffer and return (first_word_time, text)
    /// Returns None if buffer is empty
    pub fn flush(&mut self) -> Option<(Instant, String)> {
        if self.words.is_empty() {
            return None;
        }

        // Join with space - Kyutai STT outputs individual word tokens without separators
        let text = self.words.join(" ");
        let time = self.first_word_time.take().unwrap();
        self.words.clear();
        self.last_word_time = None;

        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some((time, trimmed))
        }
    }

    /// Simple flush returning just text
    pub fn flush_simple(&mut self) -> Option<String> {
        self.flush().map(|(_, text)| text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_buffer() {
        let buf = PhraseBuffer::new();
        assert!(buf.is_empty());
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_add_and_flush() {
        let mut buf = PhraseBuffer::new();
        let now = Instant::now();

        buf.add_word("Hello".to_string(), now);
        buf.add_word("world".to_string(), now);

        assert!(!buf.is_empty());

        let result = buf.flush();
        assert!(result.is_some());
        let (_, text) = result.unwrap();
        assert_eq!(text, "Hello world");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_flush_empty_after_flush() {
        let mut buf = PhraseBuffer::new();
        buf.add_word("test".to_string(), Instant::now());
        buf.flush();
        assert!(buf.flush().is_none());
    }

    #[test]
    fn test_whitespace_only_flush() {
        let mut buf = PhraseBuffer::new();
        buf.add_word(" ".to_string(), Instant::now());
        buf.add_word("  ".to_string(), Instant::now());
        assert!(buf.flush().is_none());
    }
}
