// Meeting Notes STT CLI - Two-Pass Transcription
// Pass 1: Kyutai STT streaming (real-time, during meeting)
// Pass 2: Whisper Large V3 batch (post-session, accurate)

mod events;
mod jsonl_writer;
mod kyutai;
mod meeting;
mod phrase_buffer;
mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use meeting::tui_mode::{SimpleResampler, to_mono};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(name = "stt")]
#[command(about = "Meeting transcription: Kyutai STT streaming + Whisper Large V3 batch")]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Use CPU instead of Metal GPU (for Kyutai STT)
    #[arg(long, global = true)]
    cpu: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Transcribe an audio file (Whisper Large V3)
    File {
        /// Audio input file (wav, mp3, etc.)
        input: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Add speaker diarization
        #[arg(long)]
        diarize: bool,
    },
    /// Real-time transcription from microphone (Kyutai STT streaming)
    Listen {
        /// Input device name (default: system default)
        #[arg(short, long)]
        device: Option<String>,

        /// List available input devices and exit
        #[arg(long)]
        list_devices: bool,
    },
    /// Meeting mode: dual-source capture with TUI
    Meeting {
        /// Microphone input device (your voice)
        #[arg(long)]
        mic: Option<String>,

        /// List available devices and exit
        #[arg(long)]
        list_devices: bool,

        /// Disable interactive TUI mode
        #[arg(long)]
        no_tui: bool,

        /// Session name or path. Names go to ~/Documents/meetings/
        #[arg(short, long)]
        output: Option<String>,

        /// Skip speaker diarization
        #[arg(long)]
        no_diarize: bool,

        /// Skip AI summary generation
        #[arg(long)]
        no_summary: bool,

        /// Skip Whisper Large V3 retranscription (keep draft Kyutai segments)
        #[arg(long)]
        no_retranscribe: bool,

        /// Anthropic API key for summaries (or set ANTHROPIC_API_KEY env)
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        anthropic_key: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Whisper Transcribers (sherpa-rs)
// ---------------------------------------------------------------------------

/// Whisper model variant
#[derive(Clone, Copy)]
enum WhisperModel {
    Turbo,
    LargeV3,
}

impl WhisperModel {
    fn file_prefix(&self) -> &str {
        match self {
            WhisperModel::Turbo => "turbo",
            WhisperModel::LargeV3 => "large-v3",
        }
    }
}

/// Whisper transcriber (supports both Turbo and Large V3 via model variant)
pub struct WhisperTranscriber {
    recognizer: sherpa_rs::whisper::WhisperRecognizer,
}

impl WhisperTranscriber {
    fn new(model_dir: &Path, model: WhisperModel) -> Result<Self> {
        let prefix = model.file_prefix();
        let config = sherpa_rs::whisper::WhisperConfig {
            encoder: model_dir.join(format!("{}-encoder.int8.onnx", prefix)).to_string_lossy().into(),
            decoder: model_dir.join(format!("{}-decoder.int8.onnx", prefix)).to_string_lossy().into(),
            tokens: model_dir.join(format!("{}-tokens.txt", prefix)).to_string_lossy().into(),
            language: "en".into(),
            provider: Some("cpu".into()),
            num_threads: Some(4),
            ..Default::default()
        };

        let recognizer = sherpa_rs::whisper::WhisperRecognizer::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper {:?} recognizer: {:?}", prefix, e))?;

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<TranscriptResult> {
        transcribe_with_recognizer(&mut self.recognizer, samples, sample_rate)
    }
}

/// Get available system memory in bytes (macOS)
fn available_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        let ret = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut u64 as *mut _,
                &mut len as *mut usize,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 { size } else { 16 * 1024 * 1024 * 1024 } // default 16GB
    }
    #[cfg(not(target_os = "macos"))]
    { 16 * 1024 * 1024 * 1024 }
}

/// Calculate number of parallel Whisper workers based on available memory.
/// Large V3 uses ~3GB per instance, Turbo ~400MB.
fn whisper_worker_count(model: &WhisperModel) -> usize {
    let mem = available_memory_bytes();
    let reserved = 4u64 * 1024 * 1024 * 1024; // reserve 4GB for OS + other
    let available = mem.saturating_sub(reserved);
    let per_instance = match model {
        WhisperModel::LargeV3 => 3u64 * 1024 * 1024 * 1024,
        WhisperModel::Turbo => 400 * 1024 * 1024,
    };
    let workers = (available / per_instance).clamp(1, 8) as usize;
    // Also cap by CPU cores
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    workers.min(cores)
}

/// Shared transcription logic for any Whisper model.
/// Automatically chunks audio >25s to work around sherpa-rs 30s limit.
/// Parallelizes across multiple recognizer instances for long audio.
fn transcribe_with_recognizer(
    recognizer: &mut sherpa_rs::whisper::WhisperRecognizer,
    samples: &[f32],
    sample_rate: u32,
) -> Result<TranscriptResult> {
    let max_chunk_samples = (25.0 * sample_rate as f32) as usize;

    if samples.len() <= max_chunk_samples {
        return transcribe_single_chunk(recognizer, samples, sample_rate, 0.0);
    }

    // For long audio, delegate to parallel transcription
    // We need the model config from the existing recognizer to create more instances.
    // Fall back to sequential if we can't determine the model.
    transcribe_chunks_sequential(recognizer, samples, sample_rate)
}

/// Sequential chunk transcription (fallback)
fn transcribe_chunks_sequential(
    recognizer: &mut sherpa_rs::whisper::WhisperRecognizer,
    samples: &[f32],
    sample_rate: u32,
) -> Result<TranscriptResult> {
    let max_chunk_samples = (25.0 * sample_rate as f32) as usize;
    let step_samples = (20.0 * sample_rate as f32) as usize;
    let total_chunks = samples.len().div_ceil(step_samples);
    let mut all_text = String::new();
    let mut all_segments = Vec::new();
    let mut offset = 0usize;

    eprintln!("  Splitting into {} chunks (25s each, 5s overlap)...", total_chunks);

    while offset < samples.len() {
        let end = (offset + max_chunk_samples).min(samples.len());
        let chunk = &samples[offset..end];
        let chunk_start = offset as f32 / sample_rate as f32;
        let chunk_end = ((offset + step_samples).min(samples.len())) as f32 / sample_rate as f32;

        let chunk_num = offset / step_samples + 1;
        eprint!("  Chunk {}/{} ({:.0}s-{:.0}s)... ", chunk_num, total_chunks, chunk_start, chunk_start + chunk.len() as f32 / sample_rate as f32);

        match transcribe_single_chunk(recognizer, chunk, sample_rate, chunk_start) {
            Ok(result) => {
                eprintln!("{} chars", result.text.len());
                let text = result.text.trim().to_string();
                if !text.is_empty() {
                    if !all_text.is_empty() {
                        all_text.push(' ');
                    }
                    all_text.push_str(&text);
                    all_segments.push(TranscriptSegment {
                        start: chunk_start,
                        end: chunk_end,
                        text,
                    });
                }
            }
            Err(e) => {
                eprintln!("failed: {}", e);
            }
        }

        offset += step_samples;
    }

    Ok(TranscriptResult {
        text: all_text,
        segments: all_segments,
    })
}

/// Parallel chunk transcription: creates multiple WhisperRecognizer instances
/// and distributes chunks across threads.
fn transcribe_parallel(
    model_dir: &Path,
    model: WhisperModel,
    samples: &[f32],
    sample_rate: u32,
) -> Result<TranscriptResult> {
    let max_chunk_samples = (25.0 * sample_rate as f32) as usize;
    let step_samples = (20.0 * sample_rate as f32) as usize;

    // Build list of chunks
    let mut chunks: Vec<(usize, usize)> = Vec::new(); // (offset, end)
    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + max_chunk_samples).min(samples.len());
        chunks.push((offset, end));
        offset += step_samples;
    }

    let n_workers = whisper_worker_count(&model).min(chunks.len());
    eprintln!(
        "  {} chunks, {} parallel workers ({:.0}GB RAM available)",
        chunks.len(),
        n_workers,
        available_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    if n_workers <= 1 {
        // Fall back to sequential with a single recognizer
        let mut t = WhisperTranscriber::new(model_dir, model)?;
        return transcribe_chunks_sequential(&mut t.recognizer, samples, sample_rate);
    }

    // Partition chunks into per-worker batches (round-robin for balanced load)
    let mut worker_chunks: Vec<Vec<usize>> = vec![Vec::new(); n_workers];
    for (i, _) in chunks.iter().enumerate() {
        worker_chunks[i % n_workers].push(i);
    }

    // Process in parallel using scoped threads
    let total_chunks = chunks.len();
    let chunks_ref = &chunks;
    let results: Vec<Option<(usize, f32, f32, String)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();

        for (worker_id, chunk_indices) in worker_chunks.into_iter().enumerate() {
            let model_dir = model_dir.to_path_buf();
            let model_variant = model;

            let handle = scope.spawn(move || {
                let mut results = Vec::new();
                let mut recognizer = match WhisperTranscriber::new(&model_dir, model_variant) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("  Worker {} failed to load model: {}", worker_id, e);
                        return results;
                    }
                };

                for &chunk_idx in &chunk_indices {
                    let (off, end) = chunks_ref[chunk_idx];
                    let chunk = &samples[off..end];
                    let chunk_start = off as f32 / sample_rate as f32;
                    let chunk_end = ((off + step_samples).min(samples.len())) as f32 / sample_rate as f32;

                    match transcribe_single_chunk(&mut recognizer.recognizer, chunk, sample_rate, chunk_start) {
                        Ok(result) => {
                            let text = result.text.trim().to_string();
                            if !text.is_empty() {
                                results.push(Some((chunk_idx, chunk_start, chunk_end, text)));
                            } else {
                                results.push(None);
                            }
                        }
                        Err(_) => {
                            results.push(None);
                        }
                    }

                    eprint!("\r  Transcribed {}/{} chunks", chunk_idx + 1, total_chunks);
                }

                results
            });

            handles.push(handle);
        }

        let mut all_results: Vec<Option<(usize, f32, f32, String)>> = Vec::new();
        for handle in handles {
            all_results.extend(handle.join().unwrap_or_default());
        }
        all_results
    });

    eprintln!();

    // Sort results by chunk index and merge
    let mut sorted: Vec<(usize, f32, f32, String)> = results.into_iter().flatten().collect();
    sorted.sort_by_key(|(idx, _, _, _)| *idx);

    let mut all_text = String::new();
    let mut all_segments = Vec::new();

    for (_, chunk_start, chunk_end, text) in sorted {
        if !all_text.is_empty() {
            all_text.push(' ');
        }
        all_text.push_str(&text);
        all_segments.push(TranscriptSegment {
            start: chunk_start,
            end: chunk_end,
            text,
        });
    }

    Ok(TranscriptResult {
        text: all_text,
        segments: all_segments,
    })
}

fn transcribe_single_chunk(
    recognizer: &mut sherpa_rs::whisper::WhisperRecognizer,
    samples: &[f32],
    sample_rate: u32,
    time_offset: f32,
) -> Result<TranscriptResult> {
    let result = recognizer.transcribe(sample_rate, samples);

    let mut segments = Vec::new();
    let n = result.timestamps.len().min(result.tokens.len());
    for i in 0..n {
        let start = result.timestamps[i] + time_offset;
        let end = if i + 1 < n {
            result.timestamps[i + 1] + time_offset
        } else {
            start + 0.5
        };
        segments.push(TranscriptSegment {
            start,
            end,
            text: result.tokens[i].clone(),
        });
    }

    Ok(TranscriptResult {
        text: result.text,
        segments,
    })
}

#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

// ---------------------------------------------------------------------------
// Speaker Diarization (sherpa-rs pyannote + 3dspeaker)
// ---------------------------------------------------------------------------


type WhisperFn = Box<dyn FnMut(&[f32], u32) -> Result<TranscriptResult>>;

// ---------------------------------------------------------------------------
// Audio Device Listing
// ---------------------------------------------------------------------------

fn list_audio_devices() {
    let host = cpal::default_host();
    let default_input = host.default_input_device().and_then(|d| d.name().ok());

    println!("INPUT devices (for --mic):");
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                let is_default = default_input.as_ref().is_some_and(|n| n == &name);
                let marker = if is_default { " (default)" } else { "" };
                println!("  - {}{}", name, marker);
            }
        }
    }
    println!("\nSystem audio is captured via ScreenCaptureKit (no device selection needed)");
}

// ---------------------------------------------------------------------------
// Model Management
// ---------------------------------------------------------------------------

/// Find or download model directory
fn find_models_dir() -> Result<PathBuf> {
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

    // Find existing models dir or use default data dir
    let models_dir = candidates
        .iter()
        .find(|dir| {
            dir.join("sherpa-onnx-whisper-large-v3").exists()
                || dir.join("sherpa-onnx-whisper-turbo").exists()
        })
        .cloned()
        .or_else(|| dirs::data_dir().map(|d| d.join("stt-cli").join("models")))
        .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

    // Download any missing models (idempotent - skips existing ones)
    download_models(&models_dir)?;
    Ok(models_dir)
}

/// Download and extract a tar.bz2 model archive
fn download_tar_model(models_dir: &Path, name: &str, url: &str, message: &str) -> Result<()> {
    use std::process::Command;

    let model_dir = models_dir.join(name);
    if model_dir.exists() {
        return Ok(());
    }

    eprintln!("{}", message);
    let tar_file = models_dir.join(format!("{}.tar.bz2", name));

    let status = Command::new("curl")
        .args(["-L", "-o"])
        .arg(&tar_file)
        .arg(url)
        .arg("--progress-bar")
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to download {}", name);
    }

    eprintln!("Extracting...");
    let status = Command::new("tar")
        .args(["xjf"])
        .arg(&tar_file)
        .current_dir(models_dir)
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to extract {}", name);
    }

    std::fs::remove_file(&tar_file).ok();
    eprintln!("✓ {} ready", name);
    Ok(())
}

/// Download required models
fn download_models(models_dir: &Path) -> Result<()> {
    use std::process::Command;

    std::fs::create_dir_all(models_dir)?;

    download_tar_model(
        models_dir,
        "sherpa-onnx-whisper-large-v3",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-large-v3.tar.bz2",
        "Downloading Whisper Large V3 model (~3GB, high accuracy)...",
    )?;

    download_tar_model(
        models_dir,
        "sherpa-onnx-whisper-turbo",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
        "Downloading Whisper Turbo model (~400MB, fallback)...",
    )?;

    // Diarization: pyannote segmentation model (tar archive)
    download_tar_model(
        models_dir,
        "sherpa-onnx-pyannote-segmentation-3-0",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "Downloading pyannote segmentation model (~5MB, speaker diarization)...",
    )?;

    // Diarization: 3dspeaker embedding model (single file)
    let speaker_model = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");
    if !speaker_model.exists() {
        eprintln!("Downloading 3dspeaker embedding model (~27MB, speaker diarization)...");
        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&speaker_model)
            .arg("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
            .arg("--progress-bar")
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to download 3dspeaker model");
        }
        eprintln!("✓ 3dspeaker embedding model ready");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Session Folder
// ---------------------------------------------------------------------------

fn resolve_session_folder(output: Option<String>) -> Result<PathBuf> {
    let meetings_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents")
        .join("meetings");

    let folder = match output {
        Some(name) if name.contains('/') => PathBuf::from(name),
        Some(name) => meetings_dir.join(name),
        None => {
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
            meetings_dir.join(format!("meeting_{}", timestamp))
        }
    };

    if let Some(parent) = folder.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(folder)
}

/// Load best available Whisper model (prefers Large V3, falls back to Turbo)
/// Returns (transcriber_fn, model_name) or None if no model found
fn load_best_whisper(models_dir: &Path, context: &str) -> Result<Option<(WhisperFn, &'static str)>> {
    let large_v3_dir = models_dir.join("sherpa-onnx-whisper-large-v3");
    let turbo_dir = models_dir.join("sherpa-onnx-whisper-turbo");

    if large_v3_dir.exists() {
        eprintln!("Loading Whisper Large V3{}...", if context.is_empty() { "".to_string() } else { format!(" for {}", context) });
        let mut t = WhisperTranscriber::new(&large_v3_dir, WhisperModel::LargeV3)?;
        Ok(Some((Box::new(move |s, r| t.transcribe(s, r)), "Whisper Large V3")))
    } else if turbo_dir.exists() {
        eprintln!("Loading Whisper Turbo{}...", if context.is_empty() { "".to_string() } else { format!(" for {}", context) });
        let mut t = WhisperTranscriber::new(&turbo_dir, WhisperModel::Turbo)?;
        Ok(Some((Box::new(move |s, r| t.transcribe(s, r)), "Whisper Turbo")))
    } else {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// File Transcription (Whisper Large V3)
// ---------------------------------------------------------------------------

fn transcribe_file(input: &Path, diarize: bool) -> Result<String> {
    let models_dir = find_models_dir()?;

    let (mut transcriber, _model_name) = load_best_whisper(&models_dir, "")?
        .ok_or_else(|| anyhow::anyhow!("No Whisper model found. Run `stt meeting` to auto-download models."))?;

    eprintln!("Loading audio: {}", input.display());
    let (samples, sample_rate) = sherpa_rs::read_audio_file(input.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("Failed to read audio: {:?}", e))?;

    if sample_rate != 16000 {
        anyhow::bail!(
            "Audio must be 16kHz (got {}Hz). Convert with: ffmpeg -i input.wav -ar 16000 output.wav",
            sample_rate
        );
    }

    eprintln!("Transcribing {:.1}s of audio...", samples.len() as f32 / 16000.0);
    let result = transcriber(&samples, sample_rate)?;

    if !diarize || result.segments.is_empty() {
        return Ok(result.text);
    }

    // Diarize using pyannote segmentation + 3dspeaker embeddings
    eprintln!("Running speaker diarization...");
    let segmentation_model = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0").join("model.onnx");
    let embedding_model = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");

    if !segmentation_model.exists() || !embedding_model.exists() {
        eprintln!("Diarization models not found. Run `stt meeting` to auto-download.");
        return Ok(result.text);
    }

    let diarize_config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: None,
        ..Default::default()
    };
    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        segmentation_model.to_str().unwrap(),
        embedding_model.to_str().unwrap(),
        diarize_config,
    ).map_err(|e| anyhow::anyhow!("Failed to create diarizer: {:?}", e))?;

    let speaker_segments = diarizer.compute(samples, None)
        .map_err(|e| anyhow::anyhow!("Diarization failed: {:?}", e))?;

    // Align Whisper text onto speaker timeline
    let mut output = String::new();
    for seg in &result.segments {
        let seg_mid = (seg.start + seg.end) / 2.0;
        let speaker = speaker_segments.iter()
            .find(|s| seg_mid >= s.start && seg_mid < s.end)
            .map(|s| format!("speaker_{:02}", s.speaker))
            .unwrap_or_default();

        if speaker.is_empty() {
            output.push_str(&format!("[{:.2}] {}\n", seg.start, seg.text));
        } else {
            output.push_str(&format!("[{:.2}] [{}] {}\n", seg.start, speaker, seg.text));
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Post-Session Retranscription (Whisper Large V3)
// ---------------------------------------------------------------------------

/// Mux mic and system audio into mono, retranscribe with Whisper, and annotate source per segment.
/// Replaces draft Kyutai segments in events.jsonl with accurate Whisper segments.
fn retranscribe_with_large_v3(
    session_dir: &Path,
    mic_samples: &[f32],
    sys_samples: &[f32],
) -> Result<()> {
    let models_dir = find_models_dir()?;

    // Determine best model for parallel transcription
    let large_v3_dir = models_dir.join("sherpa-onnx-whisper-large-v3");
    let turbo_dir = models_dir.join("sherpa-onnx-whisper-turbo");

    let (model_dir, model_variant, model_name) = if large_v3_dir.exists() {
        (large_v3_dir, WhisperModel::LargeV3, "Whisper Large V3")
    } else if turbo_dir.exists() {
        (turbo_dir, WhisperModel::Turbo, "Whisper Turbo")
    } else {
        eprintln!("No Whisper model found, skipping retranscription");
        return Ok(());
    };
    eprintln!("Using {} for retranscription", model_name);

    let events_path = session_dir.join("events.jsonl");
    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    // Backup draft events
    let draft_path = session_dir.join("events.draft.jsonl");
    std::fs::copy(&events_path, &draft_path)?;
    eprintln!("Draft transcript backed up to events.draft.jsonl");

    // Mux mic + sys into mono: (mic + sys) / 2, handling different lengths
    let has_mic = !mic_samples.is_empty();
    let has_sys = !sys_samples.is_empty();
    let mux_len = mic_samples.len().max(sys_samples.len());

    if mux_len == 0 {
        eprintln!("No audio to retranscribe");
        return Ok(());
    }

    let muxed: Vec<f32> = if has_mic && has_sys {
        eprintln!("Muxing mic + system audio ({:.1}s)...", mux_len as f32 / 16000.0);
        (0..mux_len)
            .map(|i| {
                let m = if i < mic_samples.len() { mic_samples[i] } else { 0.0 };
                let s = if i < sys_samples.len() { sys_samples[i] } else { 0.0 };
                (m + s) * 0.5
            })
            .collect()
    } else if has_mic {
        eprintln!("Using mic audio only ({:.1}s)...", mic_samples.len() as f32 / 16000.0);
        mic_samples.to_vec()
    } else {
        eprintln!("Using system audio only ({:.1}s)...", sys_samples.len() as f32 / 16000.0);
        sys_samples.to_vec()
    };

    let duration = muxed.len() as f32 / 16000.0;
    eprintln!("Retranscribing {:.1}s of audio...", duration);

    // Read all events, separate segments from non-segments
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut non_segment_events: Vec<serde_json::Value> = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
            if event["type"] != "segment" {
                non_segment_events.push(event);
            }
        }
    }

    let mut new_segments: Vec<serde_json::Value> = Vec::new();
    let max_existing_id = non_segment_events
        .iter()
        .filter_map(|e| e.get("id").and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(0);
    let mut next_id: u64 = max_existing_id + 1;

    // Parallel Whisper pass on muxed audio
    match transcribe_parallel(&model_dir, model_variant, &muxed, 16000) {
        Ok(result) => {
            for seg in &result.segments {
                let seg_text = seg.text.trim();
                if seg_text.is_empty() {
                    continue;
                }

                let start_ms = (seg.start * 1000.0) as u64;
                let end_ms = (seg.end * 1000.0) as u64;

                // Annotate source: compute energy ratio from original channels
                let src = if has_mic && has_sys {
                    let start_sample = (start_ms as usize) * 16;
                    let end_sample = ((end_ms as usize) * 16).min(mux_len);
                    if start_sample < end_sample {
                        let mic_energy: f32 = mic_samples.get(start_sample..end_sample.min(mic_samples.len()))
                            .map(|s| s.iter().map(|x| x * x).sum())
                            .unwrap_or(0.0);
                        let sys_energy: f32 = sys_samples.get(start_sample..end_sample.min(sys_samples.len()))
                            .map(|s| s.iter().map(|x| x * x).sum())
                            .unwrap_or(0.0);
                        if mic_energy > sys_energy * 2.0 { "local" } else { "remote" }
                    } else {
                        "unknown"
                    }
                } else if has_mic {
                    "local"
                } else {
                    "remote"
                };

                new_segments.push(serde_json::json!({
                    "type": "segment",
                    "id": next_id,
                    "ts": chrono::Utc::now().to_rfc3339(),
                    "src": src,
                    "text": seg_text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }));
                next_id += 1;
            }
            eprintln!("  {} segments, {} characters", result.segments.len(), result.text.len());
        }
        Err(e) => eprintln!("  Retranscription failed: {}", e),
    }

    // Merge: non-segment events + new segments, sorted by time
    let mut all_events = non_segment_events;
    all_events.extend(new_segments);

    // Sort by start_ms/offset_ms
    all_events.sort_by_key(|e| {
        e.get("start_ms")
            .or_else(|| e.get("offset_ms"))
            .or_else(|| e.get("duration_ms"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    });

    // Write updated events
    let updated_content: String = all_events
        .iter()
        .map(|e| serde_json::to_string(e).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    std::fs::write(&events_path, updated_content)?;

    eprintln!("✓ Retranscribed with {} (draft saved to events.draft.jsonl)", model_name);

    Ok(())
}

// ---------------------------------------------------------------------------
// Post-Meeting Diarization
// ---------------------------------------------------------------------------

/// Run speaker diarization on muxed audio using pyannote + 3dspeaker.
/// Aligns speaker labels onto existing Whisper transcript segments in events.jsonl.
fn run_diarization_from_memory(
    session_dir: &Path,
    mic_samples: &[f32],
    sys_samples: &[f32],
) -> Result<()> {
    let events_path = session_dir.join("events.jsonl");

    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    // Mux mic + sys for diarization
    let has_mic = !mic_samples.is_empty();
    let has_sys = !sys_samples.is_empty();
    let mux_len = mic_samples.len().max(sys_samples.len());

    if mux_len == 0 {
        eprintln!("No audio captured, skipping diarization");
        return Ok(());
    }

    let muxed: Vec<f32> = if has_mic && has_sys {
        (0..mux_len)
            .map(|i| {
                let m = if i < mic_samples.len() { mic_samples[i] } else { 0.0 };
                let s = if i < sys_samples.len() { sys_samples[i] } else { 0.0 };
                (m + s) * 0.5
            })
            .collect()
    } else if has_mic {
        mic_samples.to_vec()
    } else {
        sys_samples.to_vec()
    };

    let models_dir = find_models_dir()?;
    let segmentation_model = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0").join("model.onnx");
    let embedding_model = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");

    if !segmentation_model.exists() || !embedding_model.exists() {
        eprintln!("Diarization models not found, skipping diarization");
        return Ok(());
    }

    let duration_secs = muxed.len() as f64 / 16000.0;
    eprintln!(
        "Running speaker diarization on {:.1}s of audio...",
        duration_secs
    );

    let diarize_config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: None,
        ..Default::default()
    };
    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        segmentation_model.to_str().unwrap(),
        embedding_model.to_str().unwrap(),
        diarize_config,
    ).map_err(|e| anyhow::anyhow!("Failed to create diarizer: {:?}", e))?;

    let progress_callback = |n_done: i32, n_total: i32| -> i32 {
        let pct = if n_total > 0 { 100 * n_done / n_total } else { 0 };
        eprint!("\r  Diarizing... {}%", pct);
        0
    };

    let speaker_segments = diarizer.compute(muxed, Some(Box::new(progress_callback)))
        .map_err(|e| anyhow::anyhow!("Diarization failed: {:?}", e))?;
    eprintln!();

    // Collect unique speakers
    let mut speakers: Vec<i32> = speaker_segments.iter().map(|s| s.speaker).collect();
    speakers.sort();
    speakers.dedup();
    eprintln!("  {} speakers detected", speakers.len());

    // Align speaker labels onto Whisper transcript segments
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut updated_events = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut event: serde_json::Value = serde_json::from_str(line)?;

        if event["type"] == "segment" {
            if let (Some(start_ms), Some(end_ms)) = (event["start_ms"].as_u64(), event["end_ms"].as_u64()) {
                let seg_mid_secs = (start_ms + end_ms) as f32 / 2000.0;

                if let Some(ds) = speaker_segments.iter()
                    .find(|s| seg_mid_secs >= s.start && seg_mid_secs < s.end)
                {
                    let label = format!("speaker_{:02}", ds.speaker);
                    event["speaker"] = serde_json::Value::String(label);
                }
            }
        }

        updated_events.push(serde_json::to_string(&event)?);
    }

    std::fs::write(&events_path, updated_events.join("\n") + "\n")?;
    eprintln!("✓ Speaker labels written to events.jsonl");

    Ok(())
}




// ---------------------------------------------------------------------------
// Summary Generation (Claude API)
// ---------------------------------------------------------------------------

const DEFAULT_SUMMARY_PROMPT: &str = r#"Please provide a comprehensive summary of this meeting transcript.

## Required Sections

### 1. Overview
Brief description of the meeting purpose and participants.

### 2. Key Discussion Points
Main topics discussed with relevant details.

### 3. Decisions Made
Any decisions that were reached during the meeting.

### 4. Action Items
**IMPORTANT**: Extract ALL action items, tasks, and commitments mentioned. For each action item include:
- [ ] **Task description** - Owner (if mentioned) - Due date (if mentioned)

Look for phrases like "I'll do", "we need to", "can you", "let's", "action item", "TODO", "follow up", etc.

### 5. Next Steps
Any planned follow-up meetings, deadlines, or milestones.

### 6. Notable Quotes
Important statements worth highlighting (optional, only if significant).

Format the output in clean Markdown."#;

fn generate_summary(session_dir: &Path, anthropic_key: Option<&str>) -> Result<()> {
    let events_path = session_dir.join("events.jsonl");
    let context_path = session_dir.join("CONTEXT.md");
    let prompt_path = session_dir.join("PROMPT.md");

    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    let api_key = anthropic_key
        .map(|s| s.to_string())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

    let api_key = api_key.ok_or_else(|| {
        anyhow::anyhow!(
            "Anthropic API key required for summary. Set --anthropic-key or ANTHROPIC_API_KEY env var."
        )
    })?;

    let context = if context_path.exists() {
        eprintln!("Using meeting context from CONTEXT.md");
        Some(std::fs::read_to_string(&context_path)?)
    } else {
        None
    };

    let custom_prompt = if prompt_path.exists() {
        eprintln!("Using custom prompt from PROMPT.md");
        Some(std::fs::read_to_string(&prompt_path)?)
    } else {
        None
    };

    let events_content = std::fs::read_to_string(&events_path)?;
    let mut transcript_lines = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
            if event["type"] == "segment" {
                let speaker = event["speaker"]
                    .as_str()
                    .unwrap_or(if event["src"] == "mic" { "ME" } else { "REMOTE" });
                let text = event["text"].as_str().unwrap_or("");
                transcript_lines.push(format!("[{}]: {}", speaker, text));
            } else if event["type"] == "marker" {
                let label = event["label"].as_str().unwrap_or("");
                transcript_lines.push(format!("[MARKER]: {}", label));
            } else if event["type"] == "manual" {
                let text = event["text"].as_str().unwrap_or("");
                transcript_lines.push(format!("[NOTE]: {}", text));
            }
        }
    }

    let transcript = transcript_lines.join("\n");
    let base_prompt = custom_prompt.as_deref().unwrap_or(DEFAULT_SUMMARY_PROMPT);

    let prompt = if let Some(ctx) = context {
        format!(
            "{}\n\n## Meeting Context\n{}\n\n## Transcript\n{}",
            base_prompt, ctx, transcript
        )
    } else {
        format!("{}\n\n## Transcript\n{}", base_prompt, transcript)
    };

    let request_body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    });

    let output = std::process::Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg("https://api.anthropic.com/v1/messages")
        .arg("-H")
        .arg(format!("x-api-key: {}", api_key))
        .arg("-H")
        .arg("anthropic-version: 2023-06-01")
        .arg("-H")
        .arg("content-type: application/json")
        .arg("-d")
        .arg(request_body.to_string())
        .output()?;

    if !output.status.success() {
        anyhow::bail!(
            "API call failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let response: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    if let Some(error) = response.get("error") {
        anyhow::bail!("API error: {}", error);
    }

    let summary = response["content"][0]["text"]
        .as_str()
        .unwrap_or("No summary generated");

    let summary_path = session_dir.join("SUMMARY.md");
    std::fs::write(&summary_path, summary)?;

    eprintln!("✓ Summary written to: {}", summary_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Listen Mode (Kyutai STT Streaming)
// ---------------------------------------------------------------------------

fn run_listen_mode(device_name: Option<&str>, cpu: bool) -> Result<()> {
    let device = kyutai::get_device(cpu)?;

    eprintln!("Loading Kyutai STT streaming model...");
    let mut transcriber = kyutai::KyutaiTranscriber::load(kyutai::MODEL_REPO, &device, 1)?;

    let host = cpal::default_host();

    let audio_device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("Device '{}' not found", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?
    };

    let dev_name = audio_device.name().unwrap_or_else(|_| "Unknown".into());
    eprintln!("Using device: {}", dev_name);

    let config = audio_device
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No supported config"))?
        .with_max_sample_rate();

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    eprintln!("Sample rate: {}Hz, {} channels", sample_rate, channels);
    eprintln!("Press Ctrl+C to stop\n");

    let resampler = SimpleResampler::new(sample_rate, kyutai::SAMPLE_RATE);

    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();

    let stream = audio_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            let mono = to_mono(data, channels);
            let _ = tx.send(mono);
        },
        |err| eprintln!("Audio error: {}", err),
        None,
    )?;

    stream.play()?;

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT])
            .expect("Failed to create signal handler");
        if signals.forever().next().is_some() {
            running_clone.store(false, Ordering::SeqCst);
        }
    });

    let mut audio_buffer = Vec::new();
    let mut phrase_buf = phrase_buffer::PhraseBuffer::new();

    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(samples) => {
                let resampled = resampler.process(&samples);
                audio_buffer.extend(resampled);

                // Process complete chunks
                while audio_buffer.len() >= kyutai::CHUNK_SIZE {
                    let chunk: Vec<f32> = audio_buffer.drain(..kyutai::CHUNK_SIZE).collect();

                    match transcriber.process_chunk(&chunk) {
                        Ok(words) => {
                            let now = Instant::now();
                            for word in words {
                                phrase_buf.add_word(word, now);
                            }
                        }
                        Err(e) => eprintln!("\n[Transcription error: {}]", e),
                    }
                }

                // Flush completed phrases
                if phrase_buf.should_flush() {
                    if let Some(text) = phrase_buf.flush_simple() {
                        print!("{} ", text);
                        std::io::stdout().flush().ok();
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Check for phrase flush on timeout too
                if phrase_buf.should_flush() {
                    if let Some(text) = phrase_buf.flush_simple() {
                        print!("{} ", text);
                        std::io::stdout().flush().ok();
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush remaining
    if let Some(text) = phrase_buf.flush_simple() {
        print!("{} ", text);
    }

    println!("\n\nStopped.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::File {
            input,
            output,
            diarize,
        } => {
            let result = transcribe_file(&input, diarize)?;

            if let Some(path) = output {
                std::fs::write(&path, &result)?;
                eprintln!("Written to: {}", path.display());
            } else {
                print!("{}", result);
            }
        }
        Commands::Listen {
            device,
            list_devices,
        } => {
            if list_devices {
                list_audio_devices();
                return Ok(());
            }
            run_listen_mode(device.as_deref(), args.cpu)?;
        }
        Commands::Meeting {
            mic,
            list_devices,
            no_tui,
            output,
            no_diarize,
            no_summary,
            no_retranscribe,
            anthropic_key,
        } => {
            if list_devices {
                list_audio_devices();
                return Ok(());
            }

            let session_folder = resolve_session_folder(output)?;

            if no_tui {
                eprintln!("Non-TUI meeting mode not yet implemented");
                return Ok(());
            }

            // Pass 1: Real-time TUI meeting mode (Kyutai STT streaming)
            let recorded_audio =
                meeting::run_tui_meeting(mic.as_deref(), &session_folder, args.cpu)?;

            eprintln!(
                "\nRecorded {:.1}s of audio ({:.1} MB in memory)",
                recorded_audio.duration_secs(),
                recorded_audio.memory_bytes() as f64 / 1_000_000.0
            );

            // Post-processing pipeline
            eprintln!("\n--- Post-processing ---\n");

            // Pass 2: Retranscribe with Whisper Large V3
            if !no_retranscribe {
                eprintln!("Pass 2: Retranscribing with Whisper Large V3...");
                if let Err(e) = retranscribe_with_large_v3(
                    &session_folder,
                    &recorded_audio.mic_samples,
                    &recorded_audio.sys_samples,
                ) {
                    eprintln!("Retranscription failed: {}", e);
                }
            }

            // Speaker diarization
            if !no_diarize {
                eprintln!("\nRunning speaker diarization...");
                if let Err(e) = run_diarization_from_memory(
                    &session_folder,
                    &recorded_audio.mic_samples,
                    &recorded_audio.sys_samples,
                ) {
                    eprintln!("Diarization failed: {}", e);
                }
            }

            // AI summary
            if !no_summary {
                eprintln!("\nGenerating AI summary...");
                if let Err(e) = generate_summary(&session_folder, anthropic_key.as_deref()) {
                    eprintln!("Summary failed: {}", e);
                }
            }

            eprintln!("\n--- Session complete ---");
            eprintln!("Folder: {}", session_folder.display());
        }
    }

    Ok(())
}
