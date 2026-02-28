// Meeting Notes STT CLI - Two-Pass Transcription
// Pass 1: Kyutai STT streaming (real-time, during meeting)
// Pass 2: Whisper Turbo batch via whisper.cpp Metal GPU (post-session, accurate)

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
#[command(about = "Meeting transcription: Kyutai STT streaming + Whisper Turbo batch (Metal GPU)")]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Use CPU instead of Metal GPU (for Kyutai STT)
    #[arg(long, global = true)]
    cpu: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Transcribe an audio file (Whisper Turbo, Metal GPU)
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

        /// Skip Whisper retranscription (keep draft Kyutai segments)
        #[arg(long)]
        no_retranscribe: bool,

        /// Anthropic API key for summaries (or set ANTHROPIC_API_KEY env)
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        anthropic_key: Option<String>,

        /// Meeting title displayed in the TUI header
        #[arg(long)]
        meeting_title: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// JSONL Helpers
// ---------------------------------------------------------------------------

/// Read a JSONL events file into a Vec of raw JSON values, skipping blank lines.
fn read_events_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    let content = std::fs::read_to_string(path)?;
    let events = content.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect();
    Ok(events)
}

// ---------------------------------------------------------------------------
// Whisper Transcription (whisper-rs / whisper.cpp with Metal GPU)
// ---------------------------------------------------------------------------

/// Transcribe audio using whisper-rs (whisper.cpp with Metal acceleration).
/// Audio must be 16kHz mono f32 samples. No manual chunking needed —
/// whisper.cpp handles long audio internally with its own segmentation.
fn whisper_transcribe(model_path: &Path, samples: &[f32], n_threads: i32) -> Result<TranscriptResult> {
    use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid model path"))?,
        WhisperContextParameters::default(),
    ).map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;

    let mut state = ctx.create_state()
        .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_n_threads(n_threads);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(true);

    state.full(params, samples)
        .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))?;

    let n_segments = state.full_n_segments();

    let mut all_text = String::new();
    let mut segments = Vec::new();

    for i in 0..n_segments {
        let seg = match state.get_segment(i) {
            Some(s) => s,
            None => continue,
        };
        let text = seg.to_str_lossy()
            .map_err(|e| anyhow::anyhow!("Failed to get segment text: {}", e))?;
        let t0 = seg.start_timestamp();
        let t1 = seg.end_timestamp();

        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if !all_text.is_empty() {
                all_text.push(' ');
            }
            all_text.push_str(trimmed);
            segments.push(TranscriptSegment {
                start: t0 as f32 / 100.0,
                end: t1 as f32 / 100.0,
                text: trimmed.to_string(),
            });
        }
    }

    Ok(TranscriptResult { text: all_text, segments })
}

/// Find the GGML Whisper model file (prefers Turbo, falls back to Large V3)
fn find_whisper_model(models_dir: &Path) -> Option<(PathBuf, &'static str)> {
    let turbo = models_dir.join("ggml-large-v3-turbo.bin");
    if turbo.exists() {
        return Some((turbo, "Whisper Turbo"));
    }
    let large_v3 = models_dir.join("ggml-large-v3.bin");
    if large_v3.exists() {
        return Some((large_v3, "Whisper Large V3"));
    }
    None
}

/// Default thread count for Whisper decoder (encoder runs on Metal GPU)
fn whisper_thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8) as i32
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
            dir.join("ggml-large-v3-turbo.bin").exists()
                || dir.join("ggml-large-v3.bin").exists()
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
    let tar_filename = format!("{}.tar.bz2", name);
    let status = Command::new("tar")
        .args(["xjf"])
        .arg(&tar_filename)
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

    // Whisper Turbo GGML model (whisper.cpp format, Metal GPU accelerated)
    let whisper_model = models_dir.join("ggml-large-v3-turbo.bin");
    if !whisper_model.exists() {
        eprintln!("Downloading Whisper Turbo model (~809MB, Metal GPU accelerated)...");
        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&whisper_model)
            .arg("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin")
            .arg("--progress-bar")
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to download Whisper Turbo model");
        }
        eprintln!("✓ Whisper Turbo model ready");
    }

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

// ---------------------------------------------------------------------------
// File Transcription (Whisper Turbo via whisper-rs, Metal GPU)
// ---------------------------------------------------------------------------

fn transcribe_file(input: &Path, diarize: bool) -> Result<String> {
    let models_dir = find_models_dir()?;

    let (model_path, model_name) = find_whisper_model(&models_dir)
        .ok_or_else(|| anyhow::anyhow!("No Whisper model found. Run `stt meeting` to auto-download models."))?;

    eprintln!("Loading audio: {}", input.display());
    let input_str = input.to_str()
        .ok_or_else(|| anyhow::anyhow!("Audio file path is not valid UTF-8: {}", input.display()))?;
    let (samples, sample_rate) = sherpa_rs::read_audio_file(input_str)
        .map_err(|e| anyhow::anyhow!("Failed to read audio: {:?}", e))?;

    if sample_rate != 16000 {
        anyhow::bail!(
            "Audio must be 16kHz (got {}Hz). Convert with: ffmpeg -i input.wav -ar 16000 output.wav",
            sample_rate
        );
    }

    let duration = samples.len() as f32 / 16000.0;
    eprintln!("Transcribing {:.1}s of audio with {} (Metal GPU)...", duration, model_name);
    let start_time = Instant::now();
    let n_threads = whisper_thread_count();
    let result = whisper_transcribe(&model_path, &samples, n_threads)?;
    let elapsed = start_time.elapsed();
    eprintln!("  Transcribed in {:.1}s ({:.1}x realtime)", elapsed.as_secs_f64(), duration as f64 / elapsed.as_secs_f64());

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
    let seg_str = segmentation_model.to_str()
        .ok_or_else(|| anyhow::anyhow!("Segmentation model path is not valid UTF-8"))?;
    let emb_str = embedding_model.to_str()
        .ok_or_else(|| anyhow::anyhow!("Embedding model path is not valid UTF-8"))?;
    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        seg_str,
        emb_str,
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
// Audio Muxing
// ---------------------------------------------------------------------------

/// Mux mic and system audio into mono: (mic + sys) / 2, handling different lengths.
/// Returns None if both channels are empty.
fn mux_audio(mic_samples: &[f32], sys_samples: &[f32]) -> Option<Vec<f32>> {
    let has_mic = !mic_samples.is_empty();
    let has_sys = !sys_samples.is_empty();
    let mux_len = mic_samples.len().max(sys_samples.len());

    if mux_len == 0 {
        return None;
    }

    let muxed = if has_mic && has_sys {
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

    Some(muxed)
}

// ---------------------------------------------------------------------------
// Post-Session Retranscription (Whisper Turbo via whisper-rs, Metal GPU)
// ---------------------------------------------------------------------------

/// Retranscribe with Whisper and annotate source per segment.
/// Replaces draft Kyutai segments in events.jsonl with accurate Whisper segments.
/// Uses whisper-rs (whisper.cpp) with Metal GPU -- single pass, no manual chunking needed.
fn retranscribe_with_whisper(
    session_dir: &Path,
    mic_samples: &[f32],
    sys_samples: &[f32],
) -> Result<()> {
    let models_dir = find_models_dir()?;

    let (model_path, model_name) = match find_whisper_model(&models_dir) {
        Some(m) => m,
        None => {
            eprintln!("No Whisper model found, skipping retranscription");
            return Ok(());
        }
    };
    eprintln!("Using {} for retranscription (Metal GPU)", model_name);

    let events_path = session_dir.join("events.jsonl");
    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    let has_mic = !mic_samples.is_empty();
    let has_sys = !sys_samples.is_empty();

    let muxed = match mux_audio(mic_samples, sys_samples) {
        Some(m) => m,
        None => {
            eprintln!("No audio to retranscribe");
            return Ok(());
        }
    };

    let duration = muxed.len() as f32 / 16000.0;
    eprintln!("Retranscribing {:.1}s of audio...", duration);

    // Read all events, separate segments from non-segments
    let all_events = read_events_jsonl(&events_path)?;
    let non_segment_events: Vec<serde_json::Value> = all_events
        .into_iter()
        .filter(|e| e["type"] != "segment")
        .collect();

    let mut new_segments: Vec<serde_json::Value> = Vec::new();
    let max_existing_id = non_segment_events
        .iter()
        .filter_map(|e| e.get("id").and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(0);
    let mut next_id: u64 = max_existing_id + 1;

    // Single-pass Whisper transcription with Metal GPU
    let n_threads = whisper_thread_count();
    let start_time = Instant::now();
    match whisper_transcribe(&model_path, &muxed, n_threads) {
        Ok(result) => {
            let elapsed = start_time.elapsed();
            eprintln!(
                "  Transcribed in {:.1}s ({:.1}x realtime)",
                elapsed.as_secs_f64(),
                duration as f64 / elapsed.as_secs_f64()
            );

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
                    let end_sample = ((end_ms as usize) * 16).min(muxed.len());
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

    eprintln!("✓ Retranscribed with {} (Metal GPU) → events.jsonl", model_name);

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

    let muxed = match mux_audio(mic_samples, sys_samples) {
        Some(m) => m,
        None => {
            eprintln!("No audio captured, skipping diarization");
            return Ok(());
        }
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
    let seg_str = segmentation_model.to_str()
        .ok_or_else(|| anyhow::anyhow!("Segmentation model path is not valid UTF-8"))?;
    let emb_str = embedding_model.to_str()
        .ok_or_else(|| anyhow::anyhow!("Embedding model path is not valid UTF-8"))?;
    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        seg_str,
        emb_str,
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
    let mut events = read_events_jsonl(&events_path)?;

    for event in &mut events {
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
    }

    let updated: Vec<String> = events.iter()
        .map(|e| serde_json::to_string(e).unwrap_or_default())
        .collect();
    std::fs::write(&events_path, updated.join("\n") + "\n")?;
    eprintln!("✓ Speaker labels written to events.jsonl");

    Ok(())
}

// ---------------------------------------------------------------------------
// Summary Generation (Claude API)
// ---------------------------------------------------------------------------

const SUMMARY_MODEL: &str = "claude-sonnet-4-20250514";

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

fn generate_summary(
    events_path: &Path,
    summary_path: &Path,
    context_path: &Path,
    prompt_path: &Path,
    anthropic_key: Option<&str>,
) -> Result<()> {
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

    let events = read_events_jsonl(events_path)?;
    let mut transcript_lines = Vec::new();

    for event in &events {
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
        "model": SUMMARY_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    });

    // Use curl --config to pass headers via stdin so the API key
    // doesn't appear in the process list (visible via `ps aux`).
    let curl_config = format!(
        "header = \"x-api-key: {}\"\n\
         header = \"anthropic-version: 2023-06-01\"\n\
         header = \"content-type: application/json\"",
        api_key
    );

    let mut child = std::process::Command::new("curl")
        .args(["-s", "-X", "POST"])
        .arg("https://api.anthropic.com/v1/messages")
        .args(["--config", "-"])
        .arg("-d")
        .arg(request_body.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(curl_config.as_bytes())?;
    }

    let output = child.wait_with_output()?;

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

    std::fs::write(summary_path, summary)?;

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
    // If launched without a TTY (e.g. double-click from Finder), relaunch inside Terminal.
    // The osascript call is attributed to STT.app (this binary) not "bash".
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 && std::env::args().len() == 1 {
        let exe = std::env::current_exe()?;
        // Escape backslashes and double quotes to prevent command injection
        // in the AppleScript string literal.
        let exe_escaped = exe.display().to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        let script = format!(
            "tell application \"Terminal\"\n  activate\n  do script \"{}\"\nend tell",
            exe_escaped
        );
        std::process::Command::new("osascript")
            .args(["-e", &script])
            .status()?;
        return Ok(());
    }

    let args = Args::parse();

    let command = args.command.unwrap_or(Commands::Meeting {
        mic: None,
        list_devices: false,
        no_tui: false,
        output: None,
        no_diarize: false,
        no_summary: false,
        no_retranscribe: false,
        anthropic_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        meeting_title: None,
    });

    match command {
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
            meeting_title,
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
                meeting::run_tui_meeting(mic.as_deref(), &session_folder, args.cpu, meeting_title.as_deref())?;

            eprintln!(
                "\nRecorded {:.1}s of audio ({:.1} MB in memory)",
                recorded_audio.duration_secs(),
                recorded_audio.memory_bytes() as f64 / 1_000_000.0
            );

            // Post-processing pipeline
            eprintln!("\n--- Post-processing ---\n");

            let events_path = session_folder.join("events.jsonl");
            let draft_path = session_folder.join("events.draft.jsonl");
            let context_path = session_folder.join("CONTEXT.md");
            let prompt_path = session_folder.join("PROMPT.md");

            // Step 1: Back up the live (Kyutai) transcript
            if events_path.exists() {
                std::fs::copy(&events_path, &draft_path)?;
                eprintln!("Live transcript saved to events.draft.jsonl");
            }

            // Step 2: Summarize the live transcript (for A/B comparison)
            if !no_summary {
                eprintln!("\nGenerating summary from live transcript (draft)...");
                let draft_summary = session_folder.join("SUMMARY.draft.md");
                if let Err(e) = generate_summary(
                    &draft_path, &draft_summary, &context_path, &prompt_path,
                    anthropic_key.as_deref(),
                ) {
                    eprintln!("Draft summary failed: {}", e);
                }
            }

            // Step 3: Retranscribe with Whisper Turbo (overwrites events.jsonl)
            if !no_retranscribe {
                eprintln!("\nPass 2: Retranscribing with Whisper (Metal GPU)...");
                if let Err(e) = retranscribe_with_whisper(
                    &session_folder,
                    &recorded_audio.mic_samples,
                    &recorded_audio.sys_samples,
                ) {
                    eprintln!("Retranscription failed: {}", e);
                }
            }

            // Step 4: Speaker diarization (on events.jsonl, whichever version it is)
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

            // Step 5: Summarize the final transcript (Whisper + diarization)
            if !no_summary {
                eprintln!("\nGenerating summary from final transcript...");
                let final_summary = session_folder.join("SUMMARY.md");
                if let Err(e) = generate_summary(
                    &events_path, &final_summary, &context_path, &prompt_path,
                    anthropic_key.as_deref(),
                ) {
                    eprintln!("Summary failed: {}", e);
                }
            }

            eprintln!("\n--- Session complete ---");
            eprintln!("Folder: {}", session_folder.display());
        }
    }

    Ok(())
}
