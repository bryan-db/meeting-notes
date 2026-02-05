// Integrated TUI meeting mode with JSONL logging

use crate::app_event::AppEvent;
use crate::events::{AudioSource, Event};
use crate::jsonl_writer::JsonlWriter;
use crate::meeting::PhraseBuffer;
use crate::tui::{draw, handle_key, App, KeyAction};

use anyhow::Result;
use candle_core::Device;
use chrono::Utc;
use crossterm::{
    event::{self, Event as CrosstermEvent, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io::{self, Stdout};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::process::Command;

/// Audio data collected during transcription (kept in memory, never written to disk)
#[derive(Debug, Default)]
pub struct RecordedAudio {
    pub mic_samples: Vec<f32>,
    pub sys_samples: Vec<f32>,
}

impl RecordedAudio {
    /// Duration in seconds based on 24kHz sample rate
    pub fn duration_secs(&self) -> f64 {
        let max_samples = self.mic_samples.len().max(self.sys_samples.len());
        max_samples as f64 / 24000.0
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        (self.mic_samples.len() + self.sys_samples.len()) * std::mem::size_of::<f32>()
    }
}

/// Guard to restore terminal state on drop (handles panics)
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

// Re-use types from main
use crate::{SimpleResampler, SystemAudioHandler, Transcriber, ctrlc_handler, to_mono};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration,
        content_filter::SCContentFilter,
        output_type::SCStreamOutputType,
        sc_stream::SCStream,
    },
};

/// Run meeting mode with TUI
/// Returns RecordedAudio containing mic and system audio samples (in memory)
pub fn run_tui_meeting(
    model_repo: &str,
    device: &Device,
    mic_device: Option<&str>,
    session_folder: &Path,
) -> Result<RecordedAudio> {
    // Create session folder structure
    std::fs::create_dir_all(session_folder)?;
    let screenshots_dir = session_folder.join("screenshots");
    std::fs::create_dir_all(&screenshots_dir)?;

    let events_path = session_folder.join("events.jsonl");

    // Initialize terminal
    enable_raw_mode()?;
    let _guard = TerminalGuard; // Ensures cleanup even on panic

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the main loop
    let recorded_audio = run_meeting_loop(
        &mut terminal,
        model_repo,
        device,
        mic_device,
        &events_path,
        &screenshots_dir,
    )?;

    // Restore terminal (guard will also run on drop, but explicit cleanup is cleaner)
    terminal.show_cursor()?;
    // Guard handles disable_raw_mode and LeaveAlternateScreen on drop

    Ok(recorded_audio)
}

fn run_meeting_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    model_repo: &str,
    device: &Device,
    mic_device: Option<&str>,
    events_path: &Path,
    screenshots_dir: &Path,
) -> Result<RecordedAudio> {
    let host = cpal::default_host();

    // Find mic device
    let mic = if let Some(name) = mic_device {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
    } else {
        host.input_devices()?
            .find(|d| {
                d.name()
                    .map(|n| n.contains("Microphone") || n.contains("Anker") || n.contains("Insta360"))
                    .unwrap_or(false)
            })
            .or_else(|| host.default_input_device())
    };

    let mic_dev = mic.ok_or_else(|| anyhow::anyhow!("No microphone found. Use --mic to specify."))?;
    let _mic_name = mic_dev.name().unwrap_or_default();

    // Get mic config
    let mic_config = mic_dev
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No config for mic"))?
        .with_max_sample_rate();

    let mic_rate = mic_config.sample_rate().0;
    let mic_channels = mic_config.channels() as usize;
    let sys_rate = 48000u32;

    // Initialize JSONL writer
    let mut writer = JsonlWriter::new(events_path)?;

    // Screenshot counter
    let mut screenshot_count: u32 = 0;
    let screenshots_dir = screenshots_dir.to_path_buf();

    // Generate session ID
    let session_id = format!("mtg_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    // Initialize TUI app
    let mut app = App::new();

    // Write session start
    let start_event = Event::SessionStart {
        id: writer.next_id(),
        ts: Utc::now(),
        session_id: session_id.clone(),
    };
    writer.write(&start_event)?;
    app.add_event(start_event);

    // Load transcriber
    let mut transcriber = Transcriber::load_batched(model_repo, device, 2)?;

    // Audio channels
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<f32>>();
    let (sys_tx, sys_rx) = mpsc::channel::<Vec<f32>>();

    // App event channel (for transcripts and key events)
    let (event_tx, event_rx) = mpsc::channel::<AppEvent>();

    let running = Arc::new(AtomicBool::new(true));
    ctrlc_handler(running.clone());

    // Setup ScreenCaptureKit
    let content = SCShareableContent::get()
        .map_err(|e| anyhow::anyhow!("Failed to get shareable content: {:?}", e))?;
    let displays = content.displays();
    let display = displays
        .first()
        .ok_or_else(|| anyhow::anyhow!("No display found"))?;

    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();

    let config = SCStreamConfiguration::default()
        .with_captures_audio(true)
        .with_excludes_current_process_audio(true)
        .with_sample_rate(sys_rate as i32)
        .with_channel_count(1)
        .with_width(2)
        .with_height(2);

    let mut sc_stream = SCStream::new(&filter, &config);
    sc_stream.add_output_handler(SystemAudioHandler { tx: sys_tx }, SCStreamOutputType::Audio);
    sc_stream
        .start_capture()
        .map_err(|e| anyhow::anyhow!("Failed to start ScreenCaptureKit: {:?}", e))?;

    // Build mic stream
    let mic_channels_clone = mic_channels;
    let mic_stream = mic_dev.build_input_stream(
        &mic_config.into(),
        move |data: &[f32], _: &_| {
            let mono = to_mono(data, mic_channels_clone);
            let _ = mic_tx.send(mono);
        },
        |err| eprintln!("Mic error: {}", err),
        None,
    )?;
    mic_stream.play()?;

    // Spawn transcription thread
    let running_clone = running.clone();
    let event_tx_clone = event_tx.clone();
    let transcription_handle = std::thread::spawn(move || {
        run_transcription_loop(
            &mut transcriber,
            mic_rx,
            sys_rx,
            mic_rate,
            sys_rate,
            event_tx_clone,
            running_clone,
        )
    });

    // Main event loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    while running.load(Ordering::SeqCst) && !app.should_quit {
        // Draw UI
        terminal.draw(|f| draw(f, &app))?;

        // Poll for crossterm events
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let CrosstermEvent::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    let visible_height = terminal.size()?.height.saturating_sub(6) as usize;
                    match handle_key(&mut app, key, visible_height) {
                        KeyAction::Quit => {
                            running.store(false, Ordering::SeqCst);
                        }
                        KeyAction::SubmitMarker(label) => {
                            let event = Event::Marker {
                                id: writer.next_id(),
                                ts: Utc::now(),
                                offset_ms: app.elapsed_ms(),
                                label,
                            };
                            writer.write(&event)?;
                            app.add_event(event);
                        }
                        KeyAction::SubmitNote(text) => {
                            let event = Event::Manual {
                                id: writer.next_id(),
                                ts: Utc::now(),
                                offset_ms: app.elapsed_ms(),
                                text,
                            };
                            writer.write(&event)?;
                            app.add_event(event);
                        }
                        KeyAction::CaptureScreenshot => {
                            // Temporarily leave raw mode for screencapture
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

                            screenshot_count += 1;
                            let filename = format!("{:03}.png", screenshot_count);
                            let filepath = screenshots_dir.join(&filename);

                            let status = Command::new("screencapture")
                                .arg("-i") // Interactive selection
                                .arg(&filepath)
                                .status();

                            // Restore terminal
                            enable_raw_mode()?;
                            execute!(terminal.backend_mut(), EnterAlternateScreen)?;
                            terminal.clear()?;

                            if status.is_ok() && filepath.exists() {
                                let event = Event::Screenshot {
                                    id: writer.next_id(),
                                    ts: Utc::now(),
                                    offset_ms: app.elapsed_ms(),
                                    filename,
                                };
                                writer.write(&event)?;
                                app.add_event(event);
                            }
                        }
                        KeyAction::CaptureWindowScreenshot => {
                            // Temporarily leave raw mode for screencapture
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

                            screenshot_count += 1;
                            let filename = format!("{:03}.png", screenshot_count);
                            let filepath = screenshots_dir.join(&filename);

                            let status = Command::new("screencapture")
                                .arg("-W") // Window selection
                                .arg(&filepath)
                                .status();

                            // Restore terminal
                            enable_raw_mode()?;
                            execute!(terminal.backend_mut(), EnterAlternateScreen)?;
                            terminal.clear()?;

                            if status.is_ok() && filepath.exists() {
                                let event = Event::Screenshot {
                                    id: writer.next_id(),
                                    ts: Utc::now(),
                                    offset_ms: app.elapsed_ms(),
                                    filename,
                                };
                                writer.write(&event)?;
                                app.add_event(event);
                            }
                        }
                        KeyAction::ImportImage(source_path) => {
                            if source_path.exists() {
                                screenshot_count += 1;
                                let ext = source_path
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .unwrap_or("png");
                                let filename = format!("{:03}.{}", screenshot_count, ext);
                                let dest_path = screenshots_dir.join(&filename);

                                if std::fs::copy(&source_path, &dest_path).is_ok() {
                                    let event = Event::Screenshot {
                                        id: writer.next_id(),
                                        ts: Utc::now(),
                                        offset_ms: app.elapsed_ms(),
                                        filename,
                                    };
                                    writer.write(&event)?;
                                    app.add_event(event);
                                }
                            }
                        }
                        KeyAction::None => {}
                    }
                }
            }
        }

        // Process transcription events
        while let Ok(app_event) = event_rx.try_recv() {
            match app_event {
                AppEvent::Transcript {
                    source,
                    text,
                    start_ms,
                    end_ms,
                } => {
                    let event = Event::Segment {
                        id: writer.next_id(),
                        ts: Utc::now(),
                        src: source,
                        text,
                        start_ms,
                        end_ms,
                    };
                    writer.write(&event)?;
                    app.add_event(event);
                }
                AppEvent::Quit => {
                    running.store(false, Ordering::SeqCst);
                }
            }
        }

        // Tick
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Cleanup
    running.store(false, Ordering::SeqCst);
    sc_stream.stop_capture().ok();

    // Wait for transcription thread and get recorded audio
    let recorded_audio = transcription_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Transcription thread panicked"))?;

    // Write session end
    let end_event = Event::SessionEnd {
        id: writer.next_id(),
        ts: Utc::now(),
        duration_ms: app.elapsed_ms(),
    };
    writer.write(&end_event)?;

    // Return recorded audio (kept in memory for diarization, never written to disk)
    Ok(recorded_audio)
}

fn run_transcription_loop(
    transcriber: &mut Transcriber,
    mic_rx: mpsc::Receiver<Vec<f32>>,
    sys_rx: mpsc::Receiver<Vec<f32>>,
    mic_rate: u32,
    sys_rate: u32,
    event_tx: mpsc::Sender<AppEvent>,
    running: Arc<AtomicBool>,
) -> RecordedAudio {
    let mut mic_resampler = if mic_rate != 24000 {
        Some(SimpleResampler::new(mic_rate, 24000))
    } else {
        None
    };
    let mut sys_resampler = if sys_rate != 24000 {
        Some(SimpleResampler::new(sys_rate, 24000))
    } else {
        None
    };

    let mut mic_buffer = Vec::new();
    let mut sys_buffer = Vec::new();
    let chunk_size = 1920;
    let start_time = Instant::now();

    // Accumulate all audio for saving
    let mut all_mic_samples = Vec::new();
    let mut all_sys_samples = Vec::new();

    let sources = [AudioSource::Mic, AudioSource::Sys];
    let mut phrase_buffers = [PhraseBuffer::new(), PhraseBuffer::new()];
    let pause_threshold_ms = 800;

    while running.load(Ordering::SeqCst) {
        // Collect audio
        while let Ok(samples) = mic_rx.try_recv() {
            let resampled = if let Some(ref mut rs) = mic_resampler {
                rs.process(&samples)
            } else {
                samples
            };
            mic_buffer.extend(resampled.iter().copied());
            all_mic_samples.extend(resampled);
        }

        while let Ok(samples) = sys_rx.try_recv() {
            let resampled = if let Some(ref mut rs) = sys_resampler {
                rs.process(&samples)
            } else {
                samples
            };
            sys_buffer.extend(resampled.iter().copied());
            all_sys_samples.extend(resampled);
        }

        // Process chunks
        while mic_buffer.len() >= chunk_size && sys_buffer.len() >= chunk_size {
            let mic_chunk: Vec<f32> = mic_buffer.drain(..chunk_size).collect();
            let sys_chunk: Vec<f32> = sys_buffer.drain(..chunk_size).collect();

            match transcriber.process_batched_chunks(&[&mic_chunk, &sys_chunk]) {
                Ok(results) => {
                    let elapsed = start_time.elapsed();
                    for (batch_idx, word) in results {
                        if batch_idx < phrase_buffers.len() {
                            // Flush other source if quiet
                            let other_idx = 1 - batch_idx;
                            if !phrase_buffers[other_idx].is_empty()
                                && phrase_buffers[other_idx].quiet_for_ms() > 300
                            {
                                if let Some((start, end, text)) =
                                    phrase_buffers[other_idx].flush(elapsed)
                                {
                                    let _ = event_tx.send(AppEvent::Transcript {
                                        source: sources[other_idx],
                                        text,
                                        start_ms: start.as_millis() as u64,
                                        end_ms: end.as_millis() as u64,
                                    });
                                }
                            }
                            phrase_buffers[batch_idx].add_word(word, elapsed);
                        }
                    }
                }
                Err(_) => {}
            }
        }

        // Flush paused phrases
        let elapsed = start_time.elapsed();
        for (idx, buffer) in phrase_buffers.iter_mut().enumerate() {
            if buffer.should_flush(pause_threshold_ms) {
                if let Some((start, end, text)) = buffer.flush(elapsed) {
                    let _ = event_tx.send(AppEvent::Transcript {
                        source: sources[idx],
                        text,
                        start_ms: start.as_millis() as u64,
                        end_ms: end.as_millis() as u64,
                    });
                }
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    // Flush remaining
    let elapsed = start_time.elapsed();
    for (idx, buffer) in phrase_buffers.iter_mut().enumerate() {
        if let Some((start, end, text)) = buffer.flush(elapsed) {
            let _ = event_tx.send(AppEvent::Transcript {
                source: sources[idx],
                text,
                start_ms: start.as_millis() as u64,
                end_ms: end.as_millis() as u64,
            });
        }
    }

    RecordedAudio {
        mic_samples: all_mic_samples,
        sys_samples: all_sys_samples,
    }
}
