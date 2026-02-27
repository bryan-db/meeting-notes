// Integrated TUI meeting mode with JSONL logging
// Pass 1: Kyutai STT streaming for real-time transcription
// Audio stored at 16kHz for post-session Whisper Large V3 retranscription

use crate::events::{AudioSource, Event};
use crate::jsonl_writer::JsonlWriter;
use crate::kyutai::{self, KyutaiTranscriber};
use crate::phrase_buffer::PhraseBuffer;
use crate::tui::{draw, handle_key, App, KeyAction};

use anyhow::Result;
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
/// Stored at 16kHz for Whisper Large V3 post-session retranscription
#[derive(Debug, Default)]
pub struct RecordedAudio {
    pub mic_samples: Vec<f32>,  // 16kHz mono
    pub sys_samples: Vec<f32>,  // 16kHz mono
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
        if samples.is_empty() {
            return Vec::new();
        }
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

pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
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

/// Run meeting mode with TUI using Kyutai STT streaming (Pass 1)
/// Returns RecordedAudio containing mic and system audio at 16kHz for post-processing
pub fn run_tui_meeting(
    mic_device: Option<&str>,
    session_folder: &Path,
    cpu: bool,
    meeting_title: Option<&str>,
) -> Result<RecordedAudio> {
    // Create session folder structure
    std::fs::create_dir_all(session_folder)?;
    let screenshots_dir = session_folder.join("screenshots");
    std::fs::create_dir_all(&screenshots_dir)?;

    let events_path = session_folder.join("events.jsonl");

    // Load Kyutai STT BEFORE entering TUI
    eprint!("Loading Kyutai STT model...");
    let kyutai_stt: Option<KyutaiTranscriber> = {
        // Redirect stderr to /dev/null during model loading to suppress
        // internal moshi/candle/hf-hub messages that would pollute the TUI
        use std::os::unix::io::AsRawFd;
        let stderr_fd = std::io::stderr().as_raw_fd();
        let saved_stderr = unsafe { libc::dup(stderr_fd) };
        let devnull = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .ok();
        if let Some(ref dn) = devnull {
            unsafe { libc::dup2(dn.as_raw_fd(), stderr_fd) };
        }

        let result = kyutai::get_device(cpu).ok().and_then(|dev| {
            KyutaiTranscriber::load(kyutai::MODEL_REPO, &dev, 2).ok()
        });

        // Restore stderr
        if saved_stderr >= 0 {
            unsafe { libc::dup2(saved_stderr, stderr_fd) };
            unsafe { libc::close(saved_stderr) };
        }

        result
    };

    if kyutai_stt.is_some() {
        eprintln!(" ready.");
    } else {
        eprintln!(" not available (will retranscribe post-session).");
    }

    // Clear screen before entering alternate screen to prevent any leaked output
    execute!(io::stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::All))?;

    // Initialize terminal (AFTER model loading)
    enable_raw_mode()?;
    let _guard = TerminalGuard;

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the main loop
    let recorded_audio = run_meeting_loop(
        &mut terminal,
        mic_device,
        &events_path,
        &screenshots_dir,
        kyutai_stt,
        meeting_title,
    )?;

    terminal.show_cursor()?;

    Ok(recorded_audio)
}

fn run_meeting_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mic_device: Option<&str>,
    events_path: &Path,
    screenshots_dir: &Path,
    kyutai_stt: Option<KyutaiTranscriber>,
    meeting_title: Option<&str>,
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

    // Initialize JSONL writer
    let mut writer = JsonlWriter::new(events_path)?;

    let screenshots_dir = screenshots_dir.to_path_buf();
    let mut screenshot_count: u32 = 0;

    let session_id = format!("mtg_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    let mut app = App::new(meeting_title.map(String::from));

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

    // Transcript channel (from transcription thread to TUI)
    let (transcript_tx, transcript_rx) = mpsc::channel::<(AudioSource, String, u64, u64)>();

    let running = Arc::new(AtomicBool::new(true));
    let paused = Arc::new(AtomicBool::new(false));
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
        |_err| { /* Can't write to stderr while TUI is active */ },
        None,
    )?;
    mic_stream.play()?;

    // Spawn audio collection and transcription thread
    let running_clone = running.clone();
    let paused_clone = paused.clone();
    let transcription_handle = std::thread::spawn(move || {
        run_audio_collection(
            mic_rx,
            sys_rx,
            mic_rate,
            sys_rate,
            transcript_tx,
            running_clone,
            paused_clone,
            kyutai_stt,
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
                        KeyAction::TogglePause => {
                            paused.store(app.paused, Ordering::SeqCst);
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
                speaker: None,
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

    // Drain any final transcript events sent during shutdown
    while let Ok((source, text, start_ms, end_ms)) = transcript_rx.try_recv() {
        let event = Event::Segment {
            id: writer.next_id(),
            ts: Utc::now(),
            src: source,
            text,
            start_ms,
            end_ms,
            speaker: None,
        };
        writer.write(&event)?;
        app.add_event(event);
    }

    // Write session end
    let end_event = Event::SessionEnd {
        id: writer.next_id(),
        ts: Utc::now(),
        duration_ms: app.elapsed_ms(),
    };
    writer.write(&end_event)?;

    Ok(recorded_audio)
}

/// Collect audio and run Kyutai STT streaming transcription
/// Audio is resampled to both 24kHz (for Kyutai real-time) and 16kHz (for post-session Whisper)
fn run_audio_collection(
    mic_rx: mpsc::Receiver<Vec<f32>>,
    sys_rx: mpsc::Receiver<Vec<f32>>,
    mic_rate: u32,
    sys_rate: u32,
    transcript_tx: mpsc::Sender<(AudioSource, String, u64, u64)>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    mut kyutai_stt: Option<KyutaiTranscriber>,
) -> RecordedAudio {
    // Resamplers: native rate → 16kHz (for storage) and → 24kHz (for Kyutai)
    let mic_resampler_16k = SimpleResampler::new(mic_rate, 16000);
    let sys_resampler_16k = SimpleResampler::new(sys_rate, 16000);
    let mic_resampler_24k = SimpleResampler::new(mic_rate, kyutai::SAMPLE_RATE);
    let sys_resampler_24k = SimpleResampler::new(sys_rate, kyutai::SAMPLE_RATE);

    // 16kHz storage for Whisper post-processing
    let mut all_mic_16k = Vec::new();
    let mut all_sys_16k = Vec::new();

    // 24kHz buffers for Kyutai chunk processing
    let mut mic_chunk_buffer = Vec::new();
    let mut sys_chunk_buffer = Vec::new();

    // Phrase buffers for readable TUI output
    let mut mic_phrase = PhraseBuffer::new();
    let mut sys_phrase = PhraseBuffer::new();

    let start_time = Instant::now();

    while running.load(Ordering::SeqCst) {
        // When paused, drain channels to prevent backpressure but discard audio
        if paused.load(Ordering::SeqCst) {
            while mic_rx.try_recv().is_ok() {}
            while sys_rx.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Collect mic audio
        while let Ok(samples) = mic_rx.try_recv() {
            // Store at 16kHz for post-processing
            let r16 = mic_resampler_16k.process(&samples);
            all_mic_16k.extend(&r16);

            // Buffer at 24kHz for Kyutai
            if kyutai_stt.is_some() {
                let r24 = mic_resampler_24k.process(&samples);
                mic_chunk_buffer.extend(r24);
            }
        }

        // Collect system audio
        while let Ok(samples) = sys_rx.try_recv() {
            let r16 = sys_resampler_16k.process(&samples);
            all_sys_16k.extend(&r16);

            if kyutai_stt.is_some() {
                let r24 = sys_resampler_24k.process(&samples);
                sys_chunk_buffer.extend(r24);
            }
        }

        // Process chunks with Kyutai (batched: mic + sys simultaneously)
        // When one source is silent, pad it with zeros so the other keeps transcribing
        if let Some(ref mut kyutai) = kyutai_stt {
            while mic_chunk_buffer.len() >= kyutai::CHUNK_SIZE
                || sys_chunk_buffer.len() >= kyutai::CHUNK_SIZE
            {
                let mic_chunk: Vec<f32> = if mic_chunk_buffer.len() >= kyutai::CHUNK_SIZE {
                    mic_chunk_buffer.drain(..kyutai::CHUNK_SIZE).collect()
                } else {
                    vec![0.0; kyutai::CHUNK_SIZE] // silence pad
                };
                let sys_chunk: Vec<f32> = if sys_chunk_buffer.len() >= kyutai::CHUNK_SIZE {
                    sys_chunk_buffer.drain(..kyutai::CHUNK_SIZE).collect()
                } else {
                    vec![0.0; kyutai::CHUNK_SIZE] // silence pad
                };

                match kyutai.process_batched_chunks(&[&mic_chunk, &sys_chunk]) {
                    Ok(results) => {
                        let now = Instant::now();
                        let elapsed_ms = start_time.elapsed().as_millis() as u64;

                        for (batch_idx, word) in results {
                            match batch_idx {
                                0 => {
                                    // Check if other source should flush first (turn-taking)
                                    if !sys_phrase.is_empty() && sys_phrase.quiet_for_ms() > 300 {
                                        if let Some((phrase_time, text)) = sys_phrase.flush() {
                                            let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
                                            let _ = transcript_tx.send((
                                                AudioSource::Sys, text,
                                                phrase_ms.min(elapsed_ms), elapsed_ms,
                                            ));
                                        }
                                    }
                                    mic_phrase.add_word(word, now);
                                }
                                1 => {
                                    if !mic_phrase.is_empty() && mic_phrase.quiet_for_ms() > 300 {
                                        if let Some((phrase_time, text)) = mic_phrase.flush() {
                                            let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
                                            let _ = transcript_tx.send((
                                                AudioSource::Mic, text,
                                                phrase_ms.min(elapsed_ms), elapsed_ms,
                                            ));
                                        }
                                    }
                                    sys_phrase.add_word(word, now);
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(_) => {
                        // Silently ignore - can't write to stderr while TUI is active
                    }
                }
            }

            // Flush completed phrases (silence threshold exceeded)
            let elapsed_ms = start_time.elapsed().as_millis() as u64;

            if mic_phrase.should_flush() {
                if let Some((phrase_time, text)) = mic_phrase.flush() {
                    let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
                    let _ = transcript_tx.send((
                        AudioSource::Mic, text,
                        phrase_ms.min(elapsed_ms), elapsed_ms,
                    ));
                }
            }

            if sys_phrase.should_flush() {
                if let Some((phrase_time, text)) = sys_phrase.flush() {
                    let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
                    let _ = transcript_tx.send((
                        AudioSource::Sys, text,
                        phrase_ms.min(elapsed_ms), elapsed_ms,
                    ));
                }
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    // Flush any remaining phrases
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    if let Some((phrase_time, text)) = mic_phrase.flush() {
        let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
        let _ = transcript_tx.send((AudioSource::Mic, text, phrase_ms.min(elapsed_ms), elapsed_ms));
    }
    if let Some((phrase_time, text)) = sys_phrase.flush() {
        let phrase_ms = phrase_time.duration_since(start_time).as_millis() as u64;
        let _ = transcript_tx.send((AudioSource::Sys, text, phrase_ms.min(elapsed_ms), elapsed_ms));
    }

    RecordedAudio {
        mic_samples: all_mic_16k,
        sys_samples: all_sys_16k,
    }
}

