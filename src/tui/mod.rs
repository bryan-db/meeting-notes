// TUI module for interactive meeting transcription

mod app;
mod input;
mod ui;

pub use app::App;
pub use input::{handle_key, KeyAction};
pub use ui::draw;
