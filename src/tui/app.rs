// TUI application state

use crate::events::Event;
use std::path::PathBuf;
use std::time::Instant;

/// Input mode for the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// Normal mode - scrolling, shortcuts
    Normal,
    /// Entering a marker label
    Marker,
    /// Entering a manual note
    Note,
}

/// TUI application state
pub struct App {
    /// All events (sorted by offset_ms for display)
    pub events: Vec<Event>,

    /// Current input mode
    pub input_mode: InputMode,

    /// Text buffer for marker/note input
    pub input_buffer: String,

    /// Paste buffer for detecting dragged file paths
    pub paste_buffer: String,

    /// Last paste activity time (for timeout)
    pub last_paste_time: Instant,

    /// Scroll position (0 = top)
    pub scroll: usize,

    /// Whether auto-scroll is enabled (jump to bottom on new events)
    pub auto_scroll: bool,

    /// Session start time for elapsed calculation
    pub session_start: Instant,

    /// Mic enabled status
    pub mic_on: bool,

    /// System audio enabled status
    pub sys_on: bool,

    /// Whether the app should quit
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            paste_buffer: String::new(),
            last_paste_time: Instant::now(),
            scroll: 0,
            auto_scroll: true,
            session_start: Instant::now(),
            mic_on: true,
            sys_on: true,
            should_quit: false,
        }
    }

    /// Add character to paste buffer, returns detected image path if complete
    pub fn add_paste_char(&mut self, c: char) -> Option<PathBuf> {
        // Reset buffer if too much time passed (500ms = typing, not paste)
        if self.last_paste_time.elapsed().as_millis() > 500 {
            self.paste_buffer.clear();
        }
        self.last_paste_time = Instant::now();
        self.paste_buffer.push(c);

        // Check if buffer looks like a complete image path
        self.check_image_path()
    }

    /// Check if paste buffer contains a complete image path
    fn check_image_path(&mut self) -> Option<PathBuf> {
        let trimmed = self.paste_buffer.trim();

        // Must start with / or ~ (absolute path)
        if !trimmed.starts_with('/') && !trimmed.starts_with('~') {
            return None;
        }

        // Check for image extensions
        let lower = trimmed.to_lowercase();
        let is_image = lower.ends_with(".png")
            || lower.ends_with(".jpg")
            || lower.ends_with(".jpeg")
            || lower.ends_with(".gif")
            || lower.ends_with(".webp")
            || lower.ends_with(".heic")
            || lower.ends_with(".bmp");

        if is_image {
            let path = if trimmed.starts_with("~/") {
                // Expand ~/ to home directory
                if let Ok(home) = std::env::var("HOME") {
                    PathBuf::from(home).join(trimmed.strip_prefix("~/").unwrap())
                } else {
                    PathBuf::from(trimmed)
                }
            } else {
                PathBuf::from(trimmed)
            };

            // Clear buffer and return path
            self.paste_buffer.clear();
            Some(path)
        } else {
            None
        }
    }

    /// Clear paste buffer (call on command keys)
    pub fn clear_paste_buffer(&mut self) {
        self.paste_buffer.clear();
    }

    /// Add an event and maintain sort order by offset_ms
    pub fn add_event(&mut self, event: Event) {
        let offset = event.offset_ms();
        // Use < for stable insertion order (events with same timestamp keep insertion order)
        let pos = self.events.partition_point(|e| e.offset_ms() < offset);
        self.events.insert(pos, event);

        // Auto-scroll to bottom if enabled
        if self.auto_scroll {
            self.scroll_to_bottom();
        }
    }

    /// Get elapsed time since session start in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.session_start.elapsed().as_millis() as u64
    }

    /// Scroll up by one line
    pub fn scroll_up(&mut self) {
        if self.scroll > 0 {
            self.scroll -= 1;
            self.auto_scroll = false;
        }
    }

    /// Scroll down by one line
    pub fn scroll_down(&mut self, visible_height: usize) {
        let max_scroll = self.events.len().saturating_sub(visible_height);
        if self.scroll < max_scroll {
            self.scroll += 1;
        }
    }

    /// Scroll to bottom and enable auto-scroll
    pub fn scroll_to_bottom(&mut self) {
        self.scroll = self.events.len();
        self.auto_scroll = true;
    }

    /// Enter marker input mode
    pub fn enter_marker_mode(&mut self) {
        self.input_mode = InputMode::Marker;
        self.input_buffer.clear();
    }

    /// Enter note input mode
    pub fn enter_note_mode(&mut self) {
        self.input_mode = InputMode::Note;
        self.input_buffer.clear();
    }

    /// Exit input mode and return to normal
    pub fn exit_input_mode(&mut self) {
        self.input_mode = InputMode::Normal;
        self.input_buffer.clear();
    }

    /// Submit the current input and return the text
    pub fn submit_input(&mut self) -> Option<String> {
        if self.input_buffer.is_empty() {
            self.exit_input_mode();
            return None;
        }

        let text = std::mem::take(&mut self.input_buffer);
        self.input_mode = InputMode::Normal;
        Some(text)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
