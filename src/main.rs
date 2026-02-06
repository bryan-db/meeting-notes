// Meeting Notes STT CLI - Whisper + Speaker Diarization
// Uses sherpa-rs for STT and speaker embedding

mod app_event;
mod events;
mod jsonl_writer;
mod meeting;
mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(name = "stt")]
#[command(about = "Meeting transcription with speaker diarization (Whisper + WeSpeaker)")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Transcribe an audio file
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
    /// Real-time transcription from microphone
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

        /// Anthropic API key for summaries (or set ANTHROPIC_API_KEY env)
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        anthropic_key: Option<String>,
    },
}

/// Whisper-based transcriber using sherpa-rs
pub struct WhisperTranscriber {
    recognizer: sherpa_rs::whisper::WhisperRecognizer,
}

impl WhisperTranscriber {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let config = sherpa_rs::whisper::WhisperConfig {
            encoder: model_dir.join("turbo-encoder.int8.onnx").to_string_lossy().into(),
            decoder: model_dir.join("turbo-decoder.int8.onnx").to_string_lossy().into(),
            tokens: model_dir.join("turbo-tokens.txt").to_string_lossy().into(),
            language: "en".into(),
            provider: Some("cpu".into()), // or "coreml" on macOS
            num_threads: Some(4),
            ..Default::default()
        };

        let recognizer = sherpa_rs::whisper::WhisperRecognizer::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper recognizer: {:?}", e))?;

        Ok(Self { recognizer })
    }

    pub fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<TranscriptResult> {
        let result = self.recognizer.transcribe(sample_rate, samples);

        // Build segments from parallel timestamps and tokens arrays
        let mut segments = Vec::new();
        let n = result.timestamps.len().min(result.tokens.len());
        for i in 0..n {
            let start = result.timestamps[i];
            let end = if i + 1 < n {
                result.timestamps[i + 1]
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

/// Speaker embedding extractor using sherpa-rs
pub struct SpeakerEmbedder {
    extractor: sherpa_rs::speaker_id::EmbeddingExtractor,
}

impl SpeakerEmbedder {
    pub fn new(model_path: &Path) -> Result<Self> {
        let config = sherpa_rs::speaker_id::ExtractorConfig {
            model: model_path.to_string_lossy().into(),
            ..Default::default()
        };

        let extractor = sherpa_rs::speaker_id::EmbeddingExtractor::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create speaker extractor: {:?}", e))?;

        Ok(Self { extractor })
    }

    pub fn compute_embedding(&mut self, samples: Vec<f32>, sample_rate: u32) -> Result<Vec<f32>> {
        self.extractor
            .compute_speaker_embedding(samples, sample_rate)
            .map_err(|e| anyhow::anyhow!("Failed to compute embedding: {:?}", e))
    }
}

/// Simple speaker clustering using cosine similarity
pub struct SpeakerClusterer {
    embeddings: Vec<(String, Vec<f32>)>, // (speaker_id, embedding)
    threshold: f32,
    next_speaker_id: usize,
}

impl SpeakerClusterer {
    pub fn new(threshold: f32) -> Self {
        Self {
            embeddings: Vec::new(),
            threshold,
            next_speaker_id: 0,
        }
    }

    pub fn identify_speaker(&mut self, embedding: Vec<f32>) -> String {
        // Find best matching speaker
        let mut best_match: Option<(usize, f32)> = None;

        for (idx, (_, known_emb)) in self.embeddings.iter().enumerate() {
            let similarity = cosine_similarity(&embedding, known_emb);
            if similarity > self.threshold {
                if let Some((_, best_sim)) = best_match {
                    if similarity > best_sim {
                        best_match = Some((idx, similarity));
                    }
                } else {
                    best_match = Some((idx, similarity));
                }
            }
        }

        if let Some((idx, _)) = best_match {
            self.embeddings[idx].0.clone()
        } else {
            // New speaker
            let speaker_id = format!("SPEAKER_{:02}", self.next_speaker_id);
            self.next_speaker_id += 1;
            self.embeddings.push((speaker_id.clone(), embedding));
            speaker_id
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn list_audio_devices() {
    use cpal::traits::{DeviceTrait, HostTrait};

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

    // Check if models exist
    for dir in &candidates {
        if dir.join("sherpa-onnx-whisper-turbo").exists() {
            return Ok(dir.clone());
        }
    }

    // Models not found - download to data dir
    let models_dir = dirs::data_dir()
        .map(|d| d.join("stt-cli").join("models"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

    download_models(&models_dir)?;
    Ok(models_dir)
}

/// Download required models
fn download_models(models_dir: &Path) -> Result<()> {
    use std::process::Command;

    std::fs::create_dir_all(models_dir)?;

    let whisper_dir = models_dir.join("sherpa-onnx-whisper-turbo");
    if !whisper_dir.exists() {
        eprintln!("Downloading Whisper Turbo model (~400MB)...");

        let tar_file = models_dir.join("sherpa-onnx-whisper-turbo.tar.bz2");
        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2";

        // Download with curl
        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&tar_file)
            .arg(url)
            .arg("--progress-bar")
            .status()?;

        if !status.success() {
            anyhow::bail!("Failed to download Whisper model");
        }

        // Extract
        eprintln!("Extracting...");
        let status = Command::new("tar")
            .args(["xjf"])
            .arg(&tar_file)
            .current_dir(models_dir)
            .status()?;

        if !status.success() {
            anyhow::bail!("Failed to extract Whisper model");
        }

        // Cleanup tar file
        std::fs::remove_file(&tar_file).ok();
        eprintln!("✓ Whisper Turbo model ready");
    }

    let speaker_model = models_dir.join("wespeaker_en_voxceleb_resnet293_LM.onnx");
    if !speaker_model.exists() {
        eprintln!("Downloading speaker embedding model (~109MB)...");

        let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet293_LM.onnx";

        let status = Command::new("curl")
            .args(["-L", "-o"])
            .arg(&speaker_model)
            .arg(url)
            .arg("--progress-bar")
            .status()?;

        if !status.success() {
            anyhow::bail!("Failed to download speaker model");
        }

        eprintln!("✓ Speaker embedding model ready");
    }

    Ok(())
}

/// Resolve session folder
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

pub fn ctrlc_handler(running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT])
            .expect("Failed to create signal handler");
        if signals.forever().next().is_some() {
            running.store(false, Ordering::SeqCst);
        }
    });
}

/// Transcribe a file with optional diarization
fn transcribe_file(input: &Path, diarize: bool) -> Result<String> {
    let models_dir = find_models_dir()?;

    eprintln!("Loading Whisper Turbo...");
    let mut transcriber = WhisperTranscriber::new(&models_dir.join("sherpa-onnx-whisper-turbo"))?;

    eprintln!("Loading audio: {}", input.display());
    let (samples, sample_rate) = sherpa_rs::read_audio_file(input.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("Failed to read audio: {:?}", e))?;

    if sample_rate != 16000 {
        anyhow::bail!("Audio must be 16kHz (got {}Hz). Convert with: ffmpeg -i input.wav -ar 16000 output.wav", sample_rate);
    }

    eprintln!("Transcribing {:.1}s of audio...", samples.len() as f32 / 16000.0);
    let result = transcriber.transcribe(&samples, sample_rate as u32)?;

    if !diarize || result.segments.is_empty() {
        return Ok(result.text);
    }

    // Diarize: extract speaker embeddings for each segment
    eprintln!("Running speaker diarization...");
    let embedding_model = models_dir.join("wespeaker_en_voxceleb_resnet293_LM.onnx");
    if !embedding_model.exists() {
        anyhow::bail!(
            "Speaker model not found. Download with:\n\
             curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet293_LM.onnx"
        );
    }

    let mut embedder = SpeakerEmbedder::new(&embedding_model)?;
    let mut clusterer = SpeakerClusterer::new(0.6);

    let mut output = String::new();
    for seg in &result.segments {
        let start_sample = (seg.start * 16000.0) as usize;
        let end_sample = (seg.end * 16000.0) as usize;
        let end_sample = end_sample.min(samples.len());

        if start_sample >= end_sample || end_sample - start_sample < 1600 {
            // Skip very short segments (<100ms)
            output.push_str(&format!("[{:.2}] {}\n", seg.start, seg.text));
            continue;
        }

        let segment_samples = samples[start_sample..end_sample].to_vec();
        match embedder.compute_embedding(segment_samples, 16000) {
            Ok(embedding) => {
                let speaker = clusterer.identify_speaker(embedding);
                output.push_str(&format!("[{:.2}] [{}] {}\n", seg.start, speaker, seg.text));
            }
            Err(_) => {
                output.push_str(&format!("[{:.2}] {}\n", seg.start, seg.text));
            }
        }
    }

    Ok(output)
}

/// Run post-meeting diarization on recorded audio
fn run_diarization_from_memory(
    session_dir: &Path,
    sys_samples: &[f32],
    _sample_rate: u32,
) -> Result<()> {
    let events_path = session_dir.join("events.jsonl");

    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    if sys_samples.is_empty() {
        eprintln!("No system audio captured, skipping diarization");
        return Ok(());
    }

    let models_dir = find_models_dir()?;
    let embedding_model = models_dir.join("wespeaker_en_voxceleb_resnet293_LM.onnx");

    if !embedding_model.exists() {
        eprintln!("Speaker embedding model not found, skipping diarization");
        return Ok(());
    }

    let duration_secs = sys_samples.len() as f64 / 16000.0;
    eprintln!(
        "Running speaker diarization on {:.1}s of audio...",
        duration_secs
    );

    let mut embedder = SpeakerEmbedder::new(&embedding_model)?;
    let mut clusterer = SpeakerClusterer::new(0.6);

    // Read events and identify speakers for system audio segments
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut updated_events = Vec::new();
    let mut sys_segment_count = 0;
    let mut processed_count = 0;

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut event: serde_json::Value = serde_json::from_str(line)?;

        // Process system audio segments
        if event["type"] == "segment" && event["src"] == "sys" {
            sys_segment_count += 1;
            let start_ms = event["start_ms"].as_u64().unwrap_or(0);
            let end_ms = event["end_ms"].as_u64().unwrap_or(0);

            // Convert to samples (16kHz)
            let start_sample = (start_ms as usize) * 16;
            let end_sample = (end_ms as usize) * 16;
            let end_sample = end_sample.min(sys_samples.len());

            if start_sample < end_sample && end_sample - start_sample >= 1600 {
                let segment_samples = sys_samples[start_sample..end_sample].to_vec();

                match embedder.compute_embedding(segment_samples, 16000) {
                    Ok(embedding) => {
                        let speaker = clusterer.identify_speaker(embedding);
                        event["speaker"] = serde_json::Value::String(speaker);
                        processed_count += 1;
                    }
                    Err(e) => {
                        eprintln!("  Embedding failed for segment {}-{}ms: {}", start_ms, end_ms, e);
                    }
                }
            }
        }

        updated_events.push(serde_json::to_string(&event)?);
    }

    eprintln!("Found {} system segments, processed {}", sys_segment_count, processed_count);

    // Write updated events
    let backup_path = session_dir.join("events.jsonl.bak");
    std::fs::rename(&events_path, &backup_path)?;
    std::fs::write(&events_path, updated_events.join("\n") + "\n")?;

    eprintln!("✓ Updated {} (backup: events.jsonl.bak)", events_path.display());

    // Print speaker summary
    eprintln!("\nSpeakers detected: {}", clusterer.embeddings.len());
    for (speaker, _) in &clusterer.embeddings {
        eprintln!("  - {}", speaker);
    }

    Ok(())
}

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

    // Read optional context file
    let context = if context_path.exists() {
        eprintln!("Using meeting context from CONTEXT.md");
        Some(std::fs::read_to_string(&context_path)?)
    } else {
        None
    };

    // Read optional custom prompt
    let custom_prompt = if prompt_path.exists() {
        eprintln!("Using custom prompt from PROMPT.md");
        Some(std::fs::read_to_string(&prompt_path)?)
    } else {
        None
    };

    // Build transcript
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
        format!(
            "{}\n\n## Transcript\n{}",
            base_prompt, transcript
        )
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
        anyhow::bail!("API call failed: {}", String::from_utf8_lossy(&output.stderr));
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

/// Simple linear resampler for listen mode
struct SimpleResampler {
    ratio: f64,
}

impl SimpleResampler {
    fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            ratio: source_rate as f64 / target_rate as f64,
        }
    }

    fn process(&self, samples: &[f32]) -> Vec<f32> {
        if (self.ratio - 1.0).abs() < 0.001 {
            return samples.to_vec();
        }
        let output_len = (samples.len() as f64 / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * self.ratio;
            let src_floor = src_idx.floor() as usize;
            let src_ceil = (src_floor + 1).min(samples.len().saturating_sub(1));
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

/// Run real-time listen mode with periodic transcription
fn run_listen_mode(device_name: Option<&str>) -> Result<()> {
    let models_dir = find_models_dir()?;

    eprintln!("Loading Whisper Turbo...");
    let transcriber = Arc::new(Mutex::new(
        WhisperTranscriber::new(&models_dir.join("sherpa-onnx-whisper-turbo"))?
    ));

    let host = cpal::default_host();

    // Find device
    let device = if let Some(name) = device_name {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("Device '{}' not found", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?
    };

    let device_name = device.name().unwrap_or_else(|_| "Unknown".into());
    eprintln!("Using device: {}", device_name);

    let config = device
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No supported config"))?
        .with_max_sample_rate();

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let target_rate = 16000u32;

    eprintln!("Sample rate: {}Hz, {} channels", sample_rate, channels);
    eprintln!("Press Ctrl+C to stop\n");

    let resampler = SimpleResampler::new(sample_rate, target_rate);

    // Audio buffer
    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buffer_clone = buffer.clone();

    // Build input stream
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            let mono = to_mono(data, channels);
            let mut buf = buffer_clone.lock().unwrap();
            buf.extend(mono);
        },
        |err| eprintln!("Audio error: {}", err),
        None,
    )?;

    stream.play()?;

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT])
            .expect("Failed to create signal handler");
        if signals.forever().next().is_some() {
            running_clone.store(false, Ordering::SeqCst);
        }
    });

    let transcribe_interval = Duration::from_secs(3);
    let min_samples = (sample_rate as usize) * 1; // Minimum 1 second
    let mut last_transcribe = Instant::now();

    while running.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));

        if last_transcribe.elapsed() >= transcribe_interval {
            let samples: Vec<f32> = {
                let mut buf = buffer.lock().unwrap();
                if buf.len() < min_samples {
                    continue;
                }
                std::mem::take(&mut *buf)
            };

            // Resample to 16kHz
            let resampled = resampler.process(&samples);

            // Transcribe
            if let Ok(mut t) = transcriber.lock() {
                match t.transcribe(&resampled, target_rate) {
                    Ok(result) => {
                        let text = result.text.trim();
                        if !text.is_empty() {
                            print!("{} ", text);
                            std::io::stdout().flush().ok();
                        }
                    }
                    Err(e) => eprintln!("\n[Transcription error: {}]", e),
                }
            }

            last_transcribe = Instant::now();
        }
    }

    println!("\n\nStopped.");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::File { input, output, diarize } => {
            let result = transcribe_file(&input, diarize)?;

            if let Some(path) = output {
                std::fs::write(&path, &result)?;
                eprintln!("Written to: {}", path.display());
            } else {
                print!("{}", result);
            }
        }
        Commands::Listen { device, list_devices } => {
            if list_devices {
                list_audio_devices();
                return Ok(());
            }
            run_listen_mode(device.as_deref())?;
        }
        Commands::Meeting {
            mic,
            list_devices,
            no_tui,
            output,
            no_diarize,
            no_summary,
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

            // Run TUI meeting mode
            let recorded_audio =
                meeting::run_tui_meeting_sherpa(mic.as_deref(), &session_folder)?;

            eprintln!(
                "\nRecorded {:.1}s of audio ({:.1} MB in memory)",
                recorded_audio.duration_secs(),
                recorded_audio.memory_bytes() as f64 / 1_000_000.0
            );

            // Post-processing
            eprintln!("\n--- Post-processing ---\n");

            if !no_diarize {
                eprintln!("Running speaker diarization...");
                if let Err(e) = run_diarization_from_memory(
                    &session_folder,
                    &recorded_audio.sys_samples,
                    16000,
                ) {
                    eprintln!("Diarization failed: {}", e);
                }
            }

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
