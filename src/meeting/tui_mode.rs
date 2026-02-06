// Integrated TUI meeting mode with JSONL logging
// Uses Whisper for periodic transcription

use crate::events::{AudioSource, Event};
use crate::jsonl_writer::JsonlWriter;
use crate::tui::{draw, handle_key, App, KeyAction};
use crate::WhisperTranscriber;

use anyhow::Result;
use chrono::Utc;
use crossterm::{
    event::{self, Event as CrosstermEvent, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::process::Command;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use screencapturekit::{
    cm::CMSampleBuffer,
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration,
        content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait,
        output_type::SCStreamOutputType,
        sc_stream::SCStream,
    },
};

/// Audio data collected during transcription (kept in memory, never written to disk)
#[derive(Debug, Default)]
pub struct RecordedAudio {
    pub mic_samples: Vec<f32>,
    pub sys_samples: Vec<f32>,
}

impl RecordedAudio {
    /// Duration in seconds based on 16kHz sample rate
    pub fn duration_secs(&self) -> f64 {
        let max_samples = self.mic_samples.len().max(self.sys_samples.len());
        max_samples as f64 / 16000.0
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

/// ScreenCaptureKit audio handler
pub struct SystemAudioHandler {
    pub tx: mpsc::Sender<Vec<f32>>,
}

impl SCStreamOutputTrait for SystemAudioHandler {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type == SCStreamOutputType::Audio {
            if let Some(audio_list) = sample_buffer.audio_buffer_list() {
                let mut idx = 0;
                while let Some(buffer) = audio_list.buffer(idx) {
                    let data = buffer.data();
                    if !data.is_empty() {
                        let samples: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        let _ = self.tx.send(samples);
                    }
                    idx += 1;
                }
            }
        }
    }
}

/// Simple linear resampler
pub struct SimpleResampler {
    ratio: f64,
}

impl SimpleResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            ratio: source_rate as f64 / target_rate as f64,
        }
    }

    pub fn process(&self, samples: &[f32]) -> Vec<f32> {
        let output_len = (samples.len() as f64 / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * self.ratio;
            let src_floor = src_idx.floor() as usize;
            let src_ceil = (src_floor + 1).min(samples.len() - 1);
            let frac = src_idx - src_floor as f64;

            let sample = if src_floor < samples.len() {
                let s1 = samples[src_floor];
                let s2 = samples[src_ceil];
                s1 + (s2 - s1) * frac as f32
            } else {
                0.0
            };
            output.push(sample);
        }
        output
    }
}

fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        samples.to_vec()
    } else {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    }
}

pub fn ctrlc_handler(running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT])
            .expect("Failed to create signal handler");
        if signals.forever().next().is_some() {
            running.store(false, Ordering::SeqCst);
        }
    });
}

/// Run meeting mode with TUI using sherpa-rs Whisper
/// Returns RecordedAudio containing mic and system audio samples (in memory)
pub fn run_tui_meeting_sherpa(
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
    let _guard = TerminalGuard;

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the main loop
    let recorded_audio = run_meeting_loop_sherpa(
        &mut terminal,
        mic_device,
        &events_path,
        &screenshots_dir,
    )?;

    terminal.show_cursor()?;

    Ok(recorded_audio)
}

fn run_meeting_loop_sherpa(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mic_device: Option<&str>,
    events_path: &Path,
    screenshots_dir: &Path,
) -> Result<RecordedAudio> {
    let host = cpal::default_host();

    // Find mic device
    let mic_dev = if let Some(name) = mic_device {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("Mic '{}' not found", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?
    };

    // Get mic config
    let mic_config = mic_dev
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No config for mic"))?
        .with_max_sample_rate();

    let mic_rate = mic_config.sample_rate().0;
    let mic_channels = mic_config.channels() as usize;
    let sys_rate = 48000u32;
    let target_rate = 16000u32; // Whisper expects 16kHz

    // Initialize JSONL writer
    let mut writer = JsonlWriter::new(events_path)?;

    let screenshots_dir = screenshots_dir.to_path_buf();
    let mut screenshot_count: u32 = 0;

    let session_id = format!("mtg_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    let mut app = App::new();

    // Write session start
    let start_event = Event::SessionStart {
        id: writer.next_id(),
        ts: Utc::now(),
        session_id: session_id.clone(),
    };
    writer.write(&start_event)?;
    app.add_event(start_event);

    // Audio channels
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<f32>>();
    let (sys_tx, sys_rx) = mpsc::channel::<Vec<f32>>();

    // Transcript channel
    let (transcript_tx, transcript_rx) = mpsc::channel::<(AudioSource, String, u64, u64)>();

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

    // Spawn audio collection and transcription thread
    let running_clone = running.clone();
    let transcription_handle = std::thread::spawn(move || {
        run_audio_collection(
            mic_rx,
            sys_rx,
            mic_rate,
            sys_rate,
            target_rate,
            transcript_tx,
            running_clone,
        )
    });

    // Main event loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    while running.load(Ordering::SeqCst) && !app.should_quit {
        terminal.draw(|f| draw(f, &app))?;

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
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

                            screenshot_count += 1;
                            let filename = format!("{:03}.png", screenshot_count);
                            let filepath = screenshots_dir.join(&filename);

                            let status = Command::new("screencapture")
                                .arg("-i")
                                .arg(&filepath)
                                .status();

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
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

                            screenshot_count += 1;
                            let filename = format!("{:03}.png", screenshot_count);
                            let filepath = screenshots_dir.join(&filename);

                            let status = Command::new("screencapture")
                                .arg("-W")
                                .arg(&filepath)
                                .status();

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
        while let Ok((source, text, start_ms, end_ms)) = transcript_rx.try_recv() {
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

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Cleanup
    running.store(false, Ordering::SeqCst);
    sc_stream.stop_capture().ok();

    let recorded_audio = transcription_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))?;

    // Write session end
    let end_event = Event::SessionEnd {
        id: writer.next_id(),
        ts: Utc::now(),
        duration_ms: app.elapsed_ms(),
    };
    writer.write(&end_event)?;

    Ok(recorded_audio)
}

/// Collect audio and run periodic transcription
fn run_audio_collection(
    mic_rx: mpsc::Receiver<Vec<f32>>,
    sys_rx: mpsc::Receiver<Vec<f32>>,
    mic_rate: u32,
    sys_rate: u32,
    target_rate: u32,
    transcript_tx: mpsc::Sender<(AudioSource, String, u64, u64)>,
    running: Arc<AtomicBool>,
) -> RecordedAudio {
    let mic_resampler = SimpleResampler::new(mic_rate, target_rate);
    let sys_resampler = SimpleResampler::new(sys_rate, target_rate);

    let mut all_mic_samples = Vec::new();
    let mut all_sys_samples = Vec::new();

    // Buffers for periodic transcription
    let mut mic_transcribe_buffer = Vec::new();
    let mut sys_transcribe_buffer = Vec::new();

    let start_time = Instant::now();
    let mut last_transcribe = Instant::now();
    let transcribe_interval = Duration::from_secs(5); // Transcribe every 5 seconds
    let min_samples = target_rate as usize * 2; // Minimum 2 seconds of audio

    // Try to load Whisper (optional - may fail if models not found)
    let models_dir = find_models_dir_internal();
    let mut transcriber: Option<WhisperTranscriber> = models_dir
        .as_ref()
        .and_then(|dir| {
            WhisperTranscriber::new(&dir.join("sherpa-onnx-whisper-turbo")).ok()
        });

    if transcriber.is_none() {
        eprintln!("Note: Whisper models not found, recording audio only");
    }

    while running.load(Ordering::SeqCst) {
        // Collect mic audio
        while let Ok(samples) = mic_rx.try_recv() {
            let resampled = mic_resampler.process(&samples);
            all_mic_samples.extend(resampled.iter().copied());
            mic_transcribe_buffer.extend(resampled);
        }

        // Collect system audio
        while let Ok(samples) = sys_rx.try_recv() {
            let resampled = sys_resampler.process(&samples);
            all_sys_samples.extend(resampled.iter().copied());
            sys_transcribe_buffer.extend(resampled);
        }

        // Periodic transcription
        if last_transcribe.elapsed() >= transcribe_interval {
            if let Some(ref mut whisper) = transcriber {
                let elapsed_ms = start_time.elapsed().as_millis() as u64;

                // Transcribe mic buffer
                if mic_transcribe_buffer.len() >= min_samples {
                    let start_ms = elapsed_ms.saturating_sub(
                        (mic_transcribe_buffer.len() as u64 * 1000) / target_rate as u64
                    );

                    if let Ok(result) = whisper.transcribe(&mic_transcribe_buffer, target_rate) {
                        let text = result.text.trim().to_string();
                        if !text.is_empty() {
                            let _ = transcript_tx.send((AudioSource::Mic, text, start_ms, elapsed_ms));
                        }
                    }
                    mic_transcribe_buffer.clear();
                }

                // Transcribe system buffer
                if sys_transcribe_buffer.len() >= min_samples {
                    let start_ms = elapsed_ms.saturating_sub(
                        (sys_transcribe_buffer.len() as u64 * 1000) / target_rate as u64
                    );

                    if let Ok(result) = whisper.transcribe(&sys_transcribe_buffer, target_rate) {
                        let text = result.text.trim().to_string();
                        if !text.is_empty() {
                            let _ = transcript_tx.send((AudioSource::Sys, text, start_ms, elapsed_ms));
                        }
                    }
                    sys_transcribe_buffer.clear();
                }
            }

            last_transcribe = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    // Final transcription of remaining audio
    if let Some(ref mut whisper) = transcriber {
        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        if mic_transcribe_buffer.len() >= min_samples / 2 {
            if let Ok(result) = whisper.transcribe(&mic_transcribe_buffer, target_rate) {
                let text = result.text.trim().to_string();
                if !text.is_empty() {
                    let start_ms = elapsed_ms.saturating_sub(
                        (mic_transcribe_buffer.len() as u64 * 1000) / target_rate as u64
                    );
                    let _ = transcript_tx.send((AudioSource::Mic, text, start_ms, elapsed_ms));
                }
            }
        }

        if sys_transcribe_buffer.len() >= min_samples / 2 {
            if let Ok(result) = whisper.transcribe(&sys_transcribe_buffer, target_rate) {
                let text = result.text.trim().to_string();
                if !text.is_empty() {
                    let start_ms = elapsed_ms.saturating_sub(
                        (sys_transcribe_buffer.len() as u64 * 1000) / target_rate as u64
                    );
                    let _ = transcript_tx.send((AudioSource::Sys, text, start_ms, elapsed_ms));
                }
            }
        }
    }

    RecordedAudio {
        mic_samples: all_mic_samples,
        sys_samples: all_sys_samples,
    }
}

fn find_models_dir_internal() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("models"),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("models")))
            .unwrap_or_default(),
        dirs::data_dir()
            .map(|d| d.join("stt-cli").join("models"))
            .unwrap_or_default(),
    ];

    for dir in &candidates {
        if dir.join("sherpa-onnx-whisper-turbo").exists() {
            return Some(dir.clone());
        }
    }
    None
}
