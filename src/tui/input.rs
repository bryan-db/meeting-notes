// TUI key handling

use crate::tui::app::{App, InputMode};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::path::PathBuf;

/// Result of handling a key event
pub enum KeyAction {
    /// No action needed
    None,
    /// Submit a marker with the given label
    SubmitMarker(String),
    /// Submit a note with the given text
    SubmitNote(String),
    /// Capture a screenshot (interactive region)
    CaptureScreenshot,
    /// Capture a screenshot (window select)
    CaptureWindowScreenshot,
    /// Import an image from a detected file path (drag-and-drop)
    ImportImage(PathBuf),
    /// Request to quit
    Quit,
}

/// Handle a key event, returning an action if needed
pub fn handle_key(app: &mut App, key: KeyEvent, visible_height: usize) -> KeyAction {
    // Ctrl+C always quits
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.should_quit = true;
        return KeyAction::Quit;
    }

    match app.input_mode {
        InputMode::Normal => handle_normal_mode(app, key, visible_height),
        InputMode::Marker => handle_input_mode(app, key, true),
        InputMode::Note => handle_input_mode(app, key, false),
    }
}

fn handle_normal_mode(app: &mut App, key: KeyEvent, visible_height: usize) -> KeyAction {
    match key.code {
        KeyCode::Char('q') => {
            app.clear_paste_buffer();
            app.should_quit = true;
            KeyAction::Quit
        }
        KeyCode::Char('m') => {
            app.clear_paste_buffer();
            app.enter_marker_mode();
            KeyAction::None
        }
        KeyCode::Char('n') => {
            app.clear_paste_buffer();
            app.enter_note_mode();
            KeyAction::None
        }
        KeyCode::Char('s') => {
            app.clear_paste_buffer();
            KeyAction::CaptureScreenshot
        }
        KeyCode::Char('S') => {
            app.clear_paste_buffer();
            KeyAction::CaptureWindowScreenshot
        }
        KeyCode::Down => {
            app.clear_paste_buffer();
            app.scroll_down(visible_height);
            KeyAction::None
        }
        KeyCode::Up => {
            app.clear_paste_buffer();
            app.scroll_up();
            KeyAction::None
        }
        KeyCode::PageDown => {
            app.clear_paste_buffer();
            for _ in 0..visible_height {
                app.scroll_down(visible_height);
            }
            KeyAction::None
        }
        KeyCode::PageUp => {
            app.clear_paste_buffer();
            for _ in 0..visible_height {
                app.scroll_up();
            }
            KeyAction::None
        }
        KeyCode::Char('G') => {
            app.clear_paste_buffer();
            app.scroll_to_bottom();
            KeyAction::None
        }
        KeyCode::Char('g') => {
            app.clear_paste_buffer();
            app.scroll = 0;
            app.auto_scroll = false;
            KeyAction::None
        }
        // Any other character: add to paste buffer for drag-and-drop detection
        KeyCode::Char(c) => {
            if let Some(path) = app.add_paste_char(c) {
                KeyAction::ImportImage(path)
            } else {
                KeyAction::None
            }
        }
        _ => {
            app.clear_paste_buffer();
            KeyAction::None
        }
    }
}

fn handle_input_mode(app: &mut App, key: KeyEvent, is_marker: bool) -> KeyAction {
    match key.code {
        KeyCode::Esc => {
            app.exit_input_mode();
            KeyAction::None
        }
        KeyCode::Enter => {
            if let Some(text) = app.submit_input() {
                if is_marker {
                    KeyAction::SubmitMarker(text)
                } else {
                    KeyAction::SubmitNote(text)
                }
            } else {
                KeyAction::None
            }
        }
        KeyCode::Backspace => {
            app.input_buffer.pop();
            KeyAction::None
        }
        KeyCode::Char(c) => {
            app.input_buffer.push(c);
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}
