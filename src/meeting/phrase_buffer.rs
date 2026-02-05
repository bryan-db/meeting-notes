// Phrase buffer for accumulating words into sentences

use std::time::{Duration, Instant};

/// Accumulates words until a pause is detected, then flushes as a complete phrase
pub struct PhraseBuffer {
    words: Vec<String>,
    start_time: Option<Duration>,
    last_word_time: Instant,
}

impl PhraseBuffer {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            start_time: None,
            last_word_time: Instant::now(),
        }
    }

    /// Add a word to the buffer
    pub fn add_word(&mut self, word: String, elapsed: Duration) {
        if self.start_time.is_none() {
            self.start_time = Some(elapsed);
        }
        self.words.push(word);
        self.last_word_time = Instant::now();
    }

    /// Check if the buffer should be flushed (pause detected)
    pub fn should_flush(&self, pause_threshold_ms: u128) -> bool {
        !self.words.is_empty() && self.last_word_time.elapsed().as_millis() > pause_threshold_ms
    }

    /// Check if the buffer has been quiet for a given duration
    pub fn quiet_for_ms(&self) -> u128 {
        self.last_word_time.elapsed().as_millis()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Flush the buffer, returning (start_time, end_time, text) if non-empty
    pub fn flush(&mut self, current_elapsed: Duration) -> Option<(Duration, Duration, String)> {
        if self.words.is_empty() {
            return None;
        }

        let text = self.words.join(" ");
        let start = self.start_time.take().unwrap_or_default();
        self.words.clear();

        Some((start, current_elapsed, text))
    }

    /// Flush the buffer, returning (start_time, text) if non-empty
    /// Use this when end_time is not needed
    pub fn flush_simple(&mut self) -> Option<(Duration, String)> {
        if self.words.is_empty() {
            return None;
        }

        let text = self.words.join(" ");
        let start = self.start_time.take().unwrap_or_default();
        self.words.clear();

        Some((start, text))
    }
}

impl Default for PhraseBuffer {
    fn default() -> Self {
        Self::new()
    }
}
