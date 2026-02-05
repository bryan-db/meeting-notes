// Meeting mode module

mod phrase_buffer;
mod tui_mode;

pub use phrase_buffer::PhraseBuffer;
pub use tui_mode::{run_tui_meeting, RecordedAudio};
