// TUI rendering with Ratatui

use crate::tui::app::{App, InputMode};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Draw the TUI
pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // Transcript
            Constraint::Length(3), // Input/help area
            Constraint::Length(1), // Status bar
        ])
        .split(frame.area());

    draw_transcript(frame, app, chunks[0]);
    draw_input_area(frame, app, chunks[1]);
    draw_status_bar(frame, app, chunks[2]);
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let visible_height = area.height.saturating_sub(2) as usize; // Account for borders

    // Calculate visible range
    let total_events = app.events.len();
    let start = if app.auto_scroll {
        total_events.saturating_sub(visible_height)
    } else {
        app.scroll.min(total_events.saturating_sub(visible_height))
    };
    let end = (start + visible_height).min(total_events);

    let items: Vec<ListItem> = app.events[start..end]
        .iter()
        .map(|event| {
            let line = event.display_line();
            let style = match event {
                crate::events::Event::SessionStart { .. } |
                crate::events::Event::SessionEnd { .. } => {
                    Style::default().fg(Color::DarkGray)
                }
                crate::events::Event::Segment { src, .. } => {
                    match src {
                        crate::events::AudioSource::Mic => Style::default().fg(Color::Cyan),
                        crate::events::AudioSource::Sys => Style::default().fg(Color::Green),
                    }
                }
                crate::events::Event::Marker { .. } => {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                }
                crate::events::Event::Manual { .. } => {
                    Style::default().fg(Color::Magenta)
                }
                crate::events::Event::Screenshot { .. } => {
                    Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)
                }
            };
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();

    let title = format!(" Transcript ({} events) ", total_events);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title);

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn draw_input_area(frame: &mut Frame, app: &App, area: Rect) {
    let (title, content) = match app.input_mode {
        InputMode::Normal => {
            let help = " 's' screenshot | 'm' marker | 'n' note | '↑↓' scroll | 'q' quit ";
            ("Help", help.to_string())
        }
        InputMode::Marker => {
            let content = format!("Label: {}_", app.input_buffer);
            ("Marker (Enter to submit, Esc to cancel)", content)
        }
        InputMode::Note => {
            let content = format!("Note: {}_", app.input_buffer);
            ("Note (Enter to submit, Esc to cancel)", content)
        }
    };

    let style = match app.input_mode {
        InputMode::Normal => Style::default(),
        InputMode::Marker => Style::default().fg(Color::Yellow),
        InputMode::Note => Style::default().fg(Color::Magenta),
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(style);

    let paragraph = Paragraph::new(content)
        .style(style)
        .block(block);

    frame.render_widget(paragraph, area);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let elapsed = app.elapsed_ms();
    let mins = elapsed / 60000;
    let secs = (elapsed % 60000) / 1000;

    let mic_status = if app.mic_on { "🎤 ON" } else { "🎤 OFF" };
    let sys_status = if app.sys_on { "🔊 ON" } else { "🔊 OFF" };

    let status = format!(
        " {} | {} | ⏱ {:02}:{:02} | Events: {} ",
        mic_status,
        sys_status,
        mins,
        secs,
        app.events.len()
    );

    let scroll_indicator = if app.auto_scroll {
        "[AUTO]".to_string()
    } else {
        format!("[{}/{}]", app.scroll + 1, app.events.len().max(1))
    };

    let line = Line::from(vec![
        Span::styled(status, Style::default().fg(Color::White)),
        Span::raw(" "),
        Span::styled(scroll_indicator.as_str(), Style::default().fg(Color::DarkGray)),
    ]);

    let paragraph = Paragraph::new(line)
        .style(Style::default().bg(Color::DarkGray));

    frame.render_widget(paragraph, area);
}
