// Internal application events for thread communication

use crate::events::AudioSource;

/// Internal events passed between threads
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum AppEvent {
    /// Transcribed segment from audio thread
    Transcript {
        source: AudioSource,
        text: String,
        start_ms: u64,
        end_ms: u64,
    },

    /// Request to quit the application
    Quit,
}
