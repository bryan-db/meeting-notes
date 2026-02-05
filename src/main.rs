// Meeting Notes STT CLI - Kyutai STT with Metal acceleration
// Supports both file transcription and real-time mic input

mod app_event;
mod events;
mod jsonl_writer;
mod meeting;
mod tui;

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;


// ScreenCaptureKit for system audio
use screencapturekit::{
    cm::CMSampleBuffer,
    shareable_content::SCShareableContent,
    stream::{
        content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait,
        output_type::SCStreamOutputType,
        configuration::SCStreamConfiguration,
        sc_stream::SCStream,
    },
};

#[derive(Debug, Parser)]
#[command(name = "stt")]
#[command(about = "Transcribe audio using Kyutai STT (Metal-accelerated)")]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Use CPU instead of Metal GPU
    #[arg(long, global = true)]
    cpu: bool,

    /// Model repository on HuggingFace
    #[arg(long, global = true, default_value = "kyutai/stt-2.6b-en-candle")]
    model: String,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Transcribe an audio file
    File {
        /// Audio input file (wav, mp3, ogg, m4a, etc.)
        input: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format: markdown, plain, json
        #[arg(short, long, default_value = "markdown")]
        format: String,

        /// Include word-level timestamps
        #[arg(long)]
        words: bool,
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
    /// Run speaker diarization on a recorded meeting (requires pyannote)
    Diarize {
        /// Session folder containing audio_sys.wav and events.jsonl
        session: PathBuf,

        /// HuggingFace token for pyannote (or set HUGGINGFACE_TOKEN env var)
        #[arg(long)]
        hf_token: Option<String>,
    },

    /// Dual-source: capture mic (you) + system audio (meeting) simultaneously
    Meeting {
        /// Microphone input device (your voice)
        #[arg(long)]
        mic: Option<String>,

        /// System audio loopback input (e.g. "BlackHole" - captures meeting audio)
        #[arg(long)]
        system: Option<String>,

        /// List available devices and exit
        #[arg(long)]
        list_devices: bool,

        /// Disable interactive TUI mode (use plain text output)
        #[arg(long)]
        no_tui: bool,

        /// Session name or full path. Names without '/' go to ~/Documents/meetings/
        #[arg(short, long)]
        output: Option<String>,

        /// Skip speaker diarization after session ends
        #[arg(long)]
        no_diarize: bool,

        /// Skip AI summary generation after session ends
        #[arg(long)]
        no_summary: bool,

        /// HuggingFace token for pyannote (or set HUGGINGFACE_TOKEN env var)
        #[arg(long, env = "HUGGINGFACE_TOKEN")]
        hf_token: Option<String>,

        /// Anthropic API key for summary (or set ANTHROPIC_API_KEY env var)
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        anthropic_key: Option<String>,
    },
}

fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        eprintln!("Using Metal GPU");
        Ok(Device::new_metal(0)?)
    } else if candle_core::utils::cuda_is_available() {
        eprintln!("Using CUDA GPU");
        Ok(Device::new_cuda(0)?)
    } else {
        eprintln!("Using CPU");
        Ok(Device::Cpu)
    }
}

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct TranscriptWord {
    text: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TranscriptSegment {
    text: String,
    start: f64,
    end: f64,
    words: Vec<TranscriptWord>,
}

pub struct Transcriber {
    state: moshi::asr::State,
    tokenizer: sentencepiece::SentencePieceProcessor,
    config: Config,
    device: Device,
}

impl Transcriber {
    pub fn load(model_repo: &str, device: &Device) -> Result<Self> {
        Self::load_batched(model_repo, device, 1)
    }

    pub fn load_batched(model_repo: &str, device: &Device, batch_size: usize) -> Result<Self> {
        eprintln!("Loading model from: {} (batch_size={})", model_repo, batch_size);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_repo.to_string());

        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get("model.safetensors")?;
        let mimi_file = repo.get(&config.mimi_name)?;

        let tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, device)?
        };
        let lm = moshi::lm::LmModel::new(
            &config.model_config(),
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb),
        )?;

        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), device)?;
        let asr_delay = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(batch_size, asr_delay, 0., audio_tokenizer, lm)?;

        Ok(Transcriber {
            state,
            tokenizer,
            config,
            device: device.clone(),
        })
    }

    fn transcribe_file(&mut self, pcm: Vec<f32>) -> Result<Vec<TranscriptSegment>> {
        let mut pcm = pcm;

        // Add silence prefix
        if self.config.stt_config.audio_silence_prefix_seconds > 0.0 {
            let silence_len = (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
            pcm.splice(0..0, vec![0.0; silence_len]);
        }

        // Add suffix to flush
        let suffix = (self.config.stt_config.audio_delay_seconds * 24000.0) as usize;
        pcm.resize(pcm.len() + suffix + 24000, 0.0);

        let mut segments: Vec<TranscriptSegment> = Vec::new();
        let mut current_words: Vec<TranscriptWord> = Vec::new();
        let mut pending_word: Option<(String, f64)> = None;

        for chunk in pcm.chunks(1920) {
            let pcm_tensor = Tensor::new(chunk, &self.device)?.reshape((1, 1, ()))?;
            let msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for msg in msgs {
                match msg {
                    moshi::asr::AsrMsg::Word { tokens, start_time, .. } => {
                        if let Some((text, start)) = pending_word.take() {
                            current_words.push(TranscriptWord {
                                text,
                                start,
                                end: start_time,
                            });
                        }
                        let word = self.tokenizer.decode_piece_ids(&tokens).unwrap_or_default();
                        pending_word = Some((word, start_time));
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        if let Some((text, start)) = pending_word.take() {
                            current_words.push(TranscriptWord {
                                text,
                                start,
                                end: stop_time,
                            });
                        }
                    }
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        if prs[2][0] > 0.5 && !current_words.is_empty() {
                            let text: String = current_words.iter()
                                .map(|w| w.text.as_str())
                                .collect::<Vec<_>>()
                                .join(" ");
                            let start = current_words.first().map(|w| w.start).unwrap_or(0.0);
                            let end = current_words.last().map(|w| w.end).unwrap_or(0.0);
                            segments.push(TranscriptSegment {
                                text: text.trim().to_string(),
                                start,
                                end,
                                words: std::mem::take(&mut current_words),
                            });
                        }
                    }
                }
            }
        }

        // Flush remaining
        if let Some((text, start)) = pending_word {
            current_words.push(TranscriptWord { text, start, end: start + 0.5 });
        }
        if !current_words.is_empty() {
            let text: String = current_words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
            let start = current_words.first().map(|w| w.start).unwrap_or(0.0);
            let end = current_words.last().map(|w| w.end).unwrap_or(0.0);
            segments.push(TranscriptSegment {
                text: text.trim().to_string(),
                start,
                end,
                words: current_words,
            });
        }

        Ok(segments)
    }

    /// Process a chunk of audio in real-time, returns any new words
    pub fn process_chunk(&mut self, pcm: &[f32]) -> Result<Vec<String>> {
        let pcm_tensor = Tensor::new(pcm, &self.device)?.reshape((1, 1, ()))?;
        let msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        let mut words = Vec::new();
        for msg in msgs {
            if let moshi::asr::AsrMsg::Word { tokens, .. } = msg {
                let word = self.tokenizer.decode_piece_ids(&tokens).unwrap_or_default();
                words.push(word);
            }
        }
        Ok(words)
    }

    /// Process batched chunks from multiple sources simultaneously
    /// Returns (stream_index, word) pairs for each transcribed word
    pub fn process_batched_chunks(&mut self, chunks: &[&[f32]]) -> Result<Vec<(usize, String)>> {
        let batch_size = chunks.len();
        let chunk_len = chunks[0].len();

        // Stack all chunks into a single batched tensor: (batch, 1, samples)
        let mut batched_data = Vec::with_capacity(batch_size * chunk_len);
        for chunk in chunks {
            batched_data.extend_from_slice(chunk);
        }

        let pcm_tensor = Tensor::new(&batched_data[..], &self.device)?
            .reshape((batch_size, 1, chunk_len))?;

        let msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        let mut results = Vec::new();
        for msg in msgs {
            if let moshi::asr::AsrMsg::Word { tokens, batch_idx, .. } = msg {
                let word = self.tokenizer.decode_piece_ids(&tokens).unwrap_or_default();
                results.push((batch_idx, word));
            }
        }
        Ok(results)
    }
}

fn format_timestamp(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    } else {
        format!("{:02}:{:02}", mins, secs)
    }
}

fn output_markdown(segments: &[TranscriptSegment], words: bool) -> String {
    let mut out = String::new();
    out.push_str("# Transcript\n\n");
    out.push_str(&format!("*Generated: {}*\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M")));
    out.push_str("---\n\n");
    for seg in segments {
        let ts = format_timestamp(seg.start);
        if words && !seg.words.is_empty() {
            out.push_str(&format!("**[{}]**\n", ts));
            for word in &seg.words {
                out.push_str(&format!("  [{:.2}s] {}\n", word.start, word.text));
            }
            out.push('\n');
        } else {
            out.push_str(&format!("[{}] {}\n\n", ts, seg.text));
        }
    }
    out
}

fn output_plain(segments: &[TranscriptSegment]) -> String {
    segments.iter()
        .map(|s| format!("[{}] {}", format_timestamp(s.start), s.text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn output_json(segments: &[TranscriptSegment]) -> String {
    serde_json::to_string_pretty(segments).unwrap_or_default()
}

fn list_audio_devices() {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();

    let default_input = host.default_input_device().and_then(|d| d.name().ok());
    let default_output = host.default_output_device().and_then(|d| d.name().ok());

    println!("INPUT devices (for --mic):");
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                let is_default = default_input.as_ref().map(|n| n == &name).unwrap_or(false);
                let marker = if is_default { " (default)" } else { "" };
                println!("  - {}{}", name, marker);
            }
        }
    }

    println!("\nOUTPUT devices (for --system, use BlackHole to capture):");
    if let Ok(devices) = host.output_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                let is_default = default_output.as_ref().map(|n| n == &name).unwrap_or(false);
                let marker = if is_default { " (default)" } else { "" };
                println!("  - {}{}", name, marker);
            }
        }
    }

    println!("\n[Meeting mode setup]");
    println!("  --mic: Use a microphone input (your voice)");
    println!("  --system: Use BlackHole input (captures system/meeting audio)");
    println!("  Tip: Route system audio to Multi-Output (speakers + BlackHole)");
}

fn run_realtime(model_repo: &str, device: &Device, input_device: Option<&str>) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // Find input device
    let audio_device = if let Some(name) = input_device {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("Device '{}' not found", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?
    };

    let device_name = audio_device.name().unwrap_or_else(|_| "Unknown".to_string());
    eprintln!("Using input device: {}", device_name);

    // Get config - accept any channel count, we'll convert to mono
    let supported_config = audio_device
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No suitable input config"))?
        .with_max_sample_rate();

    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;
    let sample_format = supported_config.sample_format();

    eprintln!("Audio config: {}Hz, {} channels, {:?}", sample_rate, channels, sample_format);

    // Load model
    let mut transcriber = Transcriber::load(model_repo, device)?;

    // Channel for audio data
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    // Running flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Handle Ctrl+C
    ctrlc_handler(running.clone());

    eprintln!("\n🎤 Listening... (Ctrl+C to stop)\n");

    // Build and start stream
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            audio_device.build_input_stream(
                &supported_config.into(),
                move |data: &[f32], _: &_| {
                    let mono = to_mono(data, channels);
                    let _ = tx.send(mono);
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            audio_device.build_input_stream(
                &supported_config.into(),
                move |data: &[i16], _: &_| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mono = to_mono(&floats, channels);
                    let _ = tx.send(mono);
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )?
        }
        _ => anyhow::bail!("Unsupported sample format: {:?}", sample_format),
    };

    stream.play()?;

    // Resampler for converting to 24kHz
    let mut resampler = if sample_rate != 24000 {
        Some(SimpleResampler::new(sample_rate, 24000))
    } else {
        None
    };

    // Process loop
    let mut audio_buffer = Vec::new();
    let chunk_size = 1920; // ~80ms at 24kHz

    while running_clone.load(Ordering::SeqCst) {
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(samples) => {
                let resampled = if let Some(ref mut rs) = resampler {
                    rs.process(&samples)
                } else {
                    samples
                };
                audio_buffer.extend(resampled);

                // Process complete chunks
                while audio_buffer.len() >= chunk_size {
                    let chunk: Vec<f32> = audio_buffer.drain(..chunk_size).collect();

                    match transcriber.process_chunk(&chunk) {
                        Ok(words) => {
                            for word in words {
                                print!("{} ", word);
                                io::stdout().flush().ok();
                            }
                        }
                        Err(e) => eprintln!("\nTranscription error: {}", e),
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    println!("\n\n✓ Stopped");
    Ok(())
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

pub struct SimpleResampler {
    ratio: f64,
}

impl SimpleResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            ratio: source_rate as f64 / target_rate as f64,
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
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

pub fn ctrlc_handler(running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT])
            .expect("Failed to create signal handler");
        if signals.forever().next().is_some() {
            running.store(false, Ordering::SeqCst);
        }
    });
}


// ScreenCaptureKit audio handler
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
                        // SCK outputs 32-bit float audio
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


fn run_meeting_mode(
    model_repo: &str,
    device: &Device,
    mic_device: Option<&str>,
    _system_device: Option<&str>, // Ignored - using audiotee for system audio
) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // Find mic device - use specified device or system default
    let mic_device = if let Some(name) = mic_device {
        host.input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("Mic '{}' not found. Use --list-devices to see available.", name))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device. Use --mic to specify."))?
    };
    let mic_name = mic_device.name().unwrap_or_else(|_| "Unknown".into());
    eprintln!("Mic device: {}", mic_name);
    eprintln!("System audio: ScreenCaptureKit");

    // Get mic config
    let mic_config = mic_device
        .supported_input_configs()?
        .max_by_key(|c| c.max_sample_rate().0)
        .ok_or_else(|| anyhow::anyhow!("No config for mic"))?
        .with_max_sample_rate();

    let mic_rate = mic_config.sample_rate().0;
    let mic_channels = mic_config.channels() as usize;
    let sys_rate = 48000u32; // SCK default

    eprintln!("Mic: {}Hz, {} ch | System: {}Hz (SCK)", mic_rate, mic_channels, sys_rate);

    // Load single transcriber with batch_size=2 for both streams
    eprintln!("Loading model with batch_size=2...");
    let mut transcriber = Transcriber::load_batched(model_repo, device, 2)?;

    // Channels for audio data
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<f32>>();
    let (sys_tx, sys_rx) = mpsc::channel::<Vec<f32>>();

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    ctrlc_handler(running.clone());

    // Setup ScreenCaptureKit for system audio
    eprintln!("Initializing ScreenCaptureKit...");
    let content = SCShareableContent::get()
        .map_err(|e| anyhow::anyhow!("Failed to get shareable content: {:?}", e))?;
    let displays = content.displays();
    let display = displays.first()
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
    sc_stream.add_output_handler(
        SystemAudioHandler { tx: sys_tx },
        SCStreamOutputType::Audio,
    );

    sc_stream.start_capture()
        .map_err(|e| anyhow::anyhow!("Failed to start ScreenCaptureKit: {:?}", e))?;

    eprintln!("\n🎤 Meeting mode: Listening to mic + system audio... (Ctrl+C to stop)\n");

    // Build mic stream (cpal)
    let mic_channels_clone = mic_channels;
    let mic_stream = mic_device.build_input_stream(
        &mic_config.into(),
        move |data: &[f32], _: &_| {
            let mono = to_mono(data, mic_channels_clone);
            let _ = mic_tx.send(mono);
        },
        |err| eprintln!("Mic error: {}", err),
        None,
    )?;

    mic_stream.play()?;

    // Resamplers
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
    let start_time = std::time::Instant::now();

    // Source labels for batched output (batch_idx 0 = mic, batch_idx 1 = system)
    let source_labels = ["mic", "system"];

    // Phrase buffers - accumulate words until pause detected
    let mut phrase_buffers = [meeting::PhraseBuffer::new(), meeting::PhraseBuffer::new()];
    let pause_threshold_ms = 800; // Flush after 800ms of silence

    while running_clone.load(Ordering::SeqCst) {
        // Collect audio from both sources
        while let Ok(samples) = mic_rx.try_recv() {
            let resampled = if let Some(ref mut rs) = mic_resampler {
                rs.process(&samples)
            } else {
                samples
            };
            mic_buffer.extend(resampled);
        }

        while let Ok(samples) = sys_rx.try_recv() {
            let resampled = if let Some(ref mut rs) = sys_resampler {
                rs.process(&samples)
            } else {
                samples
            };
            sys_buffer.extend(resampled);
        }

        // Process in sync: both buffers must have enough data
        while mic_buffer.len() >= chunk_size && sys_buffer.len() >= chunk_size {
            let mic_chunk: Vec<f32> = mic_buffer.drain(..chunk_size).collect();
            let sys_chunk: Vec<f32> = sys_buffer.drain(..chunk_size).collect();

            match transcriber.process_batched_chunks(&[&mic_chunk, &sys_chunk]) {
                Ok(results) => {
                    let elapsed = start_time.elapsed();
                    for (batch_idx, word) in results {
                        if batch_idx < phrase_buffers.len() {
                            // When one source speaks after the other was quiet,
                            // flush the other source first (captures turn-taking)
                            let other_idx = 1 - batch_idx;
                            let other_quiet_ms = phrase_buffers[other_idx].quiet_for_ms();
                            // Only flush if other source has been quiet for 300ms+
                            // This prevents flushing during overlapping speech
                            if !phrase_buffers[other_idx].is_empty() && other_quiet_ms > 300 {
                                if let Some((ts, text)) = phrase_buffers[other_idx].flush_simple() {
                                    let source = source_labels.get(other_idx).unwrap_or(&"?");
                                    println!("[{}] [{}] {}", format_elapsed(ts), source, text);
                                }
                            }
                            phrase_buffers[batch_idx].add_word(word, elapsed);
                        }
                    }
                }
                Err(e) => eprintln!("\nTranscription error: {}", e),
            }
        }

        // Check for pauses and flush completed phrases
        for (idx, buffer) in phrase_buffers.iter_mut().enumerate() {
            if buffer.should_flush(pause_threshold_ms) {
                if let Some((ts, text)) = buffer.flush_simple() {
                    let source = source_labels.get(idx).unwrap_or(&"?");
                    println!("[{}] [{}] {}", format_elapsed(ts), source, text);
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Flush any remaining phrases
    for (idx, buffer) in phrase_buffers.iter_mut().enumerate() {
        if let Some((ts, text)) = buffer.flush_simple() {
            let source = source_labels.get(idx).unwrap_or(&"?");
            println!("[{}] [{}] {}", format_elapsed(ts), source, text);
        }
    }

    // Stop ScreenCaptureKit
    sc_stream.stop_capture().ok();

    println!("\n✓ Meeting ended");
    Ok(())
}

fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let secs = secs % 60;
    let millis = d.subsec_millis();
    format!("{:02}:{:02}.{:03}", mins, secs, millis)
}

/// Resolve session folder from user input
/// - Full path (contains '/'): use as-is
/// - Name only: put in ~/Documents/meetings/
/// - None: generate timestamped name in ~/Documents/meetings/
fn resolve_session_folder(output: Option<String>) -> Result<PathBuf> {
    let meetings_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents")
        .join("meetings");

    let folder = match output {
        Some(name) if name.contains('/') => {
            // Full path provided
            PathBuf::from(name)
        }
        Some(name) => {
            // Just a name, put in default directory
            meetings_dir.join(name)
        }
        None => {
            // Generate timestamped name
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
            meetings_dir.join(format!("meeting_{}", timestamp))
        }
    };

    // Ensure parent directory exists
    if let Some(parent) = folder.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(folder)
}

fn run_diarization(session_dir: &Path, hf_token: Option<&str>) -> Result<()> {
    let audio_path = session_dir.join("audio_sys.wav");
    let events_path = session_dir.join("events.jsonl");

    if !audio_path.exists() {
        anyhow::bail!("Audio file not found: {}", audio_path.display());
    }
    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    // Get HF token from arg or env
    let token = hf_token
        .map(|s| s.to_string())
        .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok())
        .or_else(|| std::env::var("HF_TOKEN").ok());

    let token = token.ok_or_else(|| {
        anyhow::anyhow!(
            "HuggingFace token required for pyannote. Set --hf-token or HUGGINGFACE_TOKEN env var.\n\
             Get token at: https://huggingface.co/settings/tokens\n\
             Accept pyannote terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    })?;

    eprintln!("Running speaker diarization on: {}", audio_path.display());
    eprintln!("This may take a while...");

    // Python script for diarization
    let python_script = format!(
        r#"
import json
import sys
from pyannote.audio import Pipeline

# Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="{token}"
)

# Run diarization
diarization = pipeline("{audio_path}")

# Output as JSON lines: start, end, speaker
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(json.dumps({{
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    }}))
"#,
        token = token,
        audio_path = audio_path.display()
    );

    // Run Python
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Diarization failed:\n{}", stderr);
    }

    // Parse diarization results
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut diarization_segments: Vec<(f64, f64, String)> = Vec::new();

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(start), Some(end), Some(speaker)) = (
                v["start"].as_f64(),
                v["end"].as_f64(),
                v["speaker"].as_str(),
            ) {
                diarization_segments.push((start, end, speaker.to_string()));
            }
        }
    }

    eprintln!("Found {} speaker segments", diarization_segments.len());

    // Read existing events
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut updated_events = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut event: serde_json::Value = serde_json::from_str(line)?;

        // Only process system audio segments
        if event["type"] == "segment" && event["src"] == "sys" {
            let start_ms = event["start_ms"].as_u64().unwrap_or(0);
            let end_ms = event["end_ms"].as_u64().unwrap_or(0);
            let start_sec = start_ms as f64 / 1000.0;
            let end_sec = end_ms as f64 / 1000.0;
            let mid_sec = (start_sec + end_sec) / 2.0;

            // Find best matching speaker (by midpoint)
            let speaker = diarization_segments
                .iter()
                .find(|(s, e, _)| mid_sec >= *s && mid_sec <= *e)
                .map(|(_, _, spk)| spk.clone())
                .unwrap_or_else(|| "UNKNOWN".to_string());

            event["speaker"] = serde_json::Value::String(speaker);
        }

        updated_events.push(serde_json::to_string(&event)?);
    }

    // Write updated events
    let backup_path = session_dir.join("events.jsonl.bak");
    std::fs::rename(&events_path, &backup_path)?;
    std::fs::write(&events_path, updated_events.join("\n") + "\n")?;

    eprintln!("✓ Updated {} (backup: events.jsonl.bak)", events_path.display());

    // Print speaker summary
    let mut speaker_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (_, _, speaker) in &diarization_segments {
        *speaker_counts.entry(speaker.clone()).or_insert(0) += 1;
    }
    eprintln!("\nSpeakers detected:");
    for (speaker, count) in speaker_counts {
        eprintln!("  {}: {} segments", speaker, count);
    }

    Ok(())
}

fn generate_summary(session_dir: &Path, anthropic_key: Option<&str>) -> Result<()> {
    let events_path = session_dir.join("events.jsonl");
    let context_path = session_dir.join("CONTEXT.md");

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

    // Read optional context file (from calendar agent)
    let context = if context_path.exists() {
        let content = std::fs::read_to_string(&context_path)?;
        eprintln!("Using meeting context from CONTEXT.md");
        Some(content)
    } else {
        None
    };

    // Read events and build transcript
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut transcript_lines = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
            if event["type"] == "segment" {
                let speaker = event["speaker"].as_str().unwrap_or(
                    if event["src"] == "mic" { "ME" } else { "REMOTE" }
                );
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

    // Build prompt with optional context
    let prompt = if let Some(ctx) = context {
        format!(
            "Please provide a comprehensive summary of this meeting transcript. Include:\n\
            1. **Overview**: Brief description of the meeting purpose and participants\n\
            2. **Key Discussion Points**: Main topics discussed\n\
            3. **Decisions Made**: Any decisions that were reached\n\
            4. **Action Items**: Tasks assigned or next steps identified\n\
            5. **Notable Quotes**: Important statements worth highlighting\n\n\
            Format the summary in Markdown.\n\n\
            MEETING CONTEXT:\n{}\n\n\
            TRANSCRIPT:\n{}", ctx, transcript
        )
    } else {
        format!(
            "Please provide a comprehensive summary of this meeting transcript. Include:\n\
            1. **Overview**: Brief description of the meeting purpose and participants\n\
            2. **Key Discussion Points**: Main topics discussed\n\
            3. **Decisions Made**: Any decisions that were reached\n\
            4. **Action Items**: Tasks assigned or next steps identified\n\
            5. **Notable Quotes**: Important statements worth highlighting\n\n\
            Format the summary in Markdown.\n\n\
            TRANSCRIPT:\n{}", transcript
        )
    };

    // Call Claude API
    let request_body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let output = std::process::Command::new("curl")
        .arg("-s")
        .arg("-X").arg("POST")
        .arg("https://api.anthropic.com/v1/messages")
        .arg("-H").arg(format!("x-api-key: {}", api_key))
        .arg("-H").arg("anthropic-version: 2023-06-01")
        .arg("-H").arg("content-type: application/json")
        .arg("-d").arg(request_body.to_string())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("API call failed: {}", stderr);
    }

    let response: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    if let Some(error) = response.get("error") {
        anyhow::bail!("API error: {}", error);
    }

    let summary = response["content"][0]["text"]
        .as_str()
        .unwrap_or("No summary generated");

    // Write summary
    let summary_path = session_dir.join("SUMMARY.md");
    std::fs::write(&summary_path, summary)?;

    eprintln!("✓ Summary written to: {}", summary_path.display());
    Ok(())
}

/// Run diarization from in-memory audio samples (no files written to disk)
fn run_diarization_from_memory(
    session_dir: &Path,
    sys_samples: &[f32],
    hf_token: Option<&str>,
) -> Result<()> {
    // Check for required binaries
    if std::process::Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_err()
    {
        anyhow::bail!(
            "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        );
    }

    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_err()
    {
        anyhow::bail!(
            "ffmpeg not found. Install with: brew install ffmpeg"
        );
    }

    let events_path = session_dir.join("events.jsonl");

    if !events_path.exists() {
        anyhow::bail!("Events file not found: {}", events_path.display());
    }

    if sys_samples.is_empty() {
        eprintln!("No system audio captured, skipping diarization");
        return Ok(());
    }

    // Get HF token from arg, env, or prompt
    let token = hf_token
        .map(|s| s.to_string())
        .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok())
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| {
            // Prompt user for token
            eprintln!("HuggingFace token required for speaker diarization.");
            eprintln!("Get token at: https://huggingface.co/settings/tokens");
            eprintln!("Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1");
            eprint!("\nEnter HF token (or press Enter to skip): ");
            io::stderr().flush().ok();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_ok() {
                let trimmed = input.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            None
        });

    let token = match token {
        Some(t) => t,
        None => {
            eprintln!("Skipping diarization (no token provided)");
            return Ok(());
        }
    };

    let duration_secs = sys_samples.len() as f64 / 24000.0;
    eprintln!(
        "Running speaker diarization on {:.1}s of audio ({:.1} MB in memory)...",
        duration_secs,
        (sys_samples.len() * 4) as f64 / 1_000_000.0
    );

    // Python script that reads f32 samples from stdin
    // Uses uv run with inline script dependencies
    let python_script = format!(
        r#"# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0,<2.8",
#     "torchaudio",
#     "pyannote.audio>=3.3,<4.0",
# ]
# ///

import json
import sys
import struct
import time

print("[diarize] Loading torch...", file=sys.stderr, flush=True)
import torch

# PyTorch 2.6+ changed weights_only default - allowlist required classes BEFORE importing pyannote
if hasattr(torch.serialization, 'add_safe_globals'):
    from omegaconf import ListConfig, DictConfig
    torch.serialization.add_safe_globals([ListConfig, DictConfig])

print("[diarize] Loading pyannote...", file=sys.stderr, flush=True)
from pyannote.audio import Pipeline
import torchaudio.functional as F

# Read raw f32 samples from stdin (24kHz from stt-cli)
print("[diarize] Reading audio from stdin...", file=sys.stderr, flush=True)
audio_bytes = sys.stdin.buffer.read()
num_samples = len(audio_bytes) // 4
print(f"[diarize] Read {{num_samples}} samples ({{len(audio_bytes) / 1_000_000:.1f}} MB)", file=sys.stderr, flush=True)

samples = struct.unpack(f'{{num_samples}}f', audio_bytes)

# Convert to torch tensor: (1, num_samples) for mono
waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
input_sample_rate = 24000
duration_sec = num_samples / input_sample_rate
print(f"[diarize] Audio duration: {{duration_sec:.1f}}s @ {{input_sample_rate}}Hz", file=sys.stderr, flush=True)

# Resample to 16kHz (pyannote requirement)
target_sample_rate = 16000
if input_sample_rate != target_sample_rate:
    print(f"[diarize] Resampling {{input_sample_rate}}Hz -> {{target_sample_rate}}Hz...", file=sys.stderr, flush=True)
    waveform = F.resample(waveform, input_sample_rate, target_sample_rate)

# Load pipeline
print("[diarize] Loading pyannote pipeline (may download models on first run)...", file=sys.stderr, flush=True)
start = time.time()
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="{token}"
)
print(f"[diarize] Pipeline loaded in {{time.time() - start:.1f}}s", file=sys.stderr, flush=True)

# Run diarization with in-memory audio
print("[diarize] Running diarization...", file=sys.stderr, flush=True)
start = time.time()
diarization = pipeline({{"waveform": waveform, "sample_rate": target_sample_rate}})
print(f"[diarize] Diarization completed in {{time.time() - start:.1f}}s", file=sys.stderr, flush=True)

# Output as JSON lines: start, end, speaker
segments = list(diarization.itertracks(yield_label=True))
print(f"[diarize] Found {{len(segments)}} speaker segments", file=sys.stderr, flush=True)

for turn, _, speaker in segments:
    print(json.dumps({{
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    }}))
"#,
        token = token
    );

    // Convert f32 samples to bytes for piping
    let audio_bytes: Vec<u8> = sys_samples
        .iter()
        .flat_map(|&s| s.to_le_bytes())
        .collect();

    // Write script to temp file (uv run needs a file for inline metadata)
    let temp_dir = std::env::temp_dir();
    let script_path = temp_dir.join("stt_diarize.py");
    std::fs::write(&script_path, &python_script)?;

    // Run with uv (handles dependencies automatically)
    use std::process::{Command, Stdio};
    use std::io::{BufRead, BufReader};

    let mut child = Command::new("uv")
        .arg("run")
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Write audio to stdin in a separate thread to avoid deadlock
    let stdin = child.stdin.take().expect("Failed to open stdin");
    let write_thread = std::thread::spawn(move || {
        use std::io::Write;
        let mut stdin = stdin;
        if let Err(e) = stdin.write_all(&audio_bytes) {
            eprintln!("Warning: Failed to write all audio data: {}", e);
        }
        // stdin is dropped here, closing the pipe
    });

    // Stream stderr in real-time (for progress logs)
    let stderr = child.stderr.take().expect("Failed to open stderr");
    let stderr_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut all_lines = Vec::new();
        let mut in_torchcodec_warning = false;

        for line in reader.lines().map_while(Result::ok) {
            all_lines.push(line.clone());

            // Skip torchcodec warning block (it's noise - we use in-memory audio)
            if line.contains("torchcodec is not installed") {
                in_torchcodec_warning = true;
                continue;
            }
            if in_torchcodec_warning {
                if line.starts_with("[diarize]") {
                    in_torchcodec_warning = false;
                } else {
                    continue;
                }
            }

            // Print our progress markers
            if line.starts_with("[diarize]") {
                eprintln!("{}", line);
            }
        }
        all_lines
    });

    // Read stdout (diarization results)
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let stdout_reader = BufReader::new(stdout);
    let stdout_lines: Vec<String> = stdout_reader.lines().map_while(Result::ok).collect();

    // Wait for everything to complete
    let status = child.wait()?;
    let _ = write_thread.join();
    let error_lines = stderr_thread.join().unwrap_or_default();

    // Clean up temp script
    let _ = std::fs::remove_file(&script_path);

    if !status.success() {
        // On failure, show full stderr
        eprintln!("Full error output:\n{}", error_lines.join("\n"));
        anyhow::bail!("Diarization failed (exit code: {:?})", status.code());
    }

    // Parse diarization results from stdout
    let mut diarization_segments: Vec<(f64, f64, String)> = Vec::new();

    for line in stdout_lines.iter() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(start), Some(end), Some(speaker)) = (
                v["start"].as_f64(),
                v["end"].as_f64(),
                v["speaker"].as_str(),
            ) {
                diarization_segments.push((start, end, speaker.to_string()));
            }
        }
    }

    eprintln!("[diarize] Updating JSONL with {} speaker segments", diarization_segments.len());

    // Read existing events
    let events_content = std::fs::read_to_string(&events_path)?;
    let mut updated_events = Vec::new();

    for line in events_content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut event: serde_json::Value = serde_json::from_str(line)?;

        // Only process system audio segments
        if event["type"] == "segment" && event["src"] == "sys" {
            let start_ms = event["start_ms"].as_u64().unwrap_or(0);
            let end_ms = event["end_ms"].as_u64().unwrap_or(0);
            let start_sec = start_ms as f64 / 1000.0;
            let end_sec = end_ms as f64 / 1000.0;
            let mid_sec = (start_sec + end_sec) / 2.0;

            // Find best matching speaker (by midpoint)
            let speaker = diarization_segments
                .iter()
                .find(|(s, e, _)| mid_sec >= *s && mid_sec <= *e)
                .map(|(_, _, spk)| spk.clone())
                .unwrap_or_else(|| "UNKNOWN".to_string());

            event["speaker"] = serde_json::Value::String(speaker);
        }

        updated_events.push(serde_json::to_string(&event)?);
    }

    // Write updated events
    let backup_path = session_dir.join("events.jsonl.bak");
    std::fs::rename(&events_path, &backup_path)?;
    std::fs::write(&events_path, updated_events.join("\n") + "\n")?;

    eprintln!("✓ Updated {} (backup: events.jsonl.bak)", events_path.display());

    // Print speaker summary
    let mut speaker_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (_, _, speaker) in &diarization_segments {
        *speaker_counts.entry(speaker.clone()).or_insert(0) += 1;
    }
    eprintln!("\nSpeakers detected:");
    for (speaker, count) in speaker_counts {
        eprintln!("  {}: {} segments", speaker, count);
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = get_device(args.cpu)?;

    match args.command {
        Commands::File { input, output, format, words } => {
            eprintln!("Loading audio: {}", input.display());
            let (pcm, sample_rate) = kaudio::pcm_decode(input.to_str().unwrap())?;
            let pcm = if sample_rate != 24_000 {
                eprintln!("Resampling from {}Hz to 24000Hz", sample_rate);
                kaudio::resample(&pcm, sample_rate as usize, 24_000)?
            } else {
                pcm
            };

            let duration_secs = pcm.len() as f64 / 24000.0;
            eprintln!("Audio duration: {:.1}s", duration_secs);

            let mut transcriber = Transcriber::load(&args.model, &device)?;

            eprintln!("Transcribing...");
            let segments = transcriber.transcribe_file(pcm)?;
            eprintln!("Found {} segments", segments.len());

            let out_text = match format.as_str() {
                "markdown" | "md" => output_markdown(&segments, words),
                "plain" | "txt" => output_plain(&segments),
                "json" => output_json(&segments),
                _ => output_markdown(&segments, words),
            };

            if let Some(path) = output {
                std::fs::write(&path, &out_text)?;
                eprintln!("Written to: {}", path.display());
            } else {
                print!("{}", out_text);
            }
        }
        Commands::Listen { device: input_device, list_devices } => {
            if list_devices {
                list_audio_devices();
                return Ok(());
            }
            run_realtime(&args.model, &device, input_device.as_deref())?;
        }
        Commands::Diarize { session, hf_token } => {
            run_diarization(&session, hf_token.as_deref())?;
        }
        Commands::Meeting { mic, system, list_devices, no_tui, output, no_diarize, no_summary, hf_token, anthropic_key } => {
            if list_devices {
                list_audio_devices();
                return Ok(());
            }

            // Resolve session folder
            let session_folder = resolve_session_folder(output)?;

            if no_tui {
                run_meeting_mode(&args.model, &device, mic.as_deref(), system.as_deref())?;
            } else {
                // Run meeting and get recorded audio (in memory)
                let recorded_audio = meeting::run_tui_meeting(&args.model, &device, mic.as_deref(), &session_folder)?;

                eprintln!(
                    "\nRecorded {:.1}s of audio ({:.1} MB in memory)",
                    recorded_audio.duration_secs(),
                    recorded_audio.memory_bytes() as f64 / 1_000_000.0
                );

                // Post-processing pipeline
                eprintln!("\n--- Post-processing ---\n");

                // 1. Run diarization (from memory - no audio files written to disk)
                if !no_diarize {
                    eprintln!("Running speaker diarization...");
                    if let Err(e) = run_diarization_from_memory(
                        &session_folder,
                        &recorded_audio.sys_samples,
                        hf_token.as_deref(),
                    ) {
                        eprintln!("Diarization failed: {}", e);
                    }
                }

                // 2. Generate summary
                if !no_summary {
                    eprintln!("\nGenerating AI summary...");
                    if let Err(e) = generate_summary(&session_folder, anthropic_key.as_deref()) {
                        eprintln!("Summary failed: {}", e);
                    }
                }

                // Audio is dropped here - never written to disk
                eprintln!("\n--- Session complete ---");
                eprintln!("Folder: {}", session_folder.display());
                eprintln!("(Audio was processed in memory and not saved to disk)");
            }
        }
    }

    Ok(())
}
