// Kyutai STT 2.6B streaming transcriber
// True streaming ASR via moshi/candle with Metal GPU acceleration.
// Operates at 24kHz, 1920-sample chunks (80ms per step), ~160-200ms latency.

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Chunk size for Kyutai STT: 1920 samples = 80ms at 24kHz
pub const CHUNK_SIZE: usize = 1920;

/// Sample rate required by Kyutai STT
pub const SAMPLE_RATE: u32 = 24000;

/// HuggingFace model repo for Kyutai STT
pub const MODEL_REPO: &str = "kyutai/stt-2.6b-en-candle";

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
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

pub struct KyutaiTranscriber {
    state: moshi::asr::State,
    tokenizer: sentencepiece::SentencePieceProcessor,
    device: Device,
}

impl KyutaiTranscriber {
    /// Load model from HuggingFace hub
    pub fn load(model_repo: &str, device: &Device, batch_size: usize) -> Result<Self> {

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

        let mimi_str = mimi_file.to_str()
            .ok_or_else(|| anyhow::anyhow!("Mimi model path is not valid UTF-8"))?;
        let audio_tokenizer = moshi::mimi::load(mimi_str, Some(32), device)?;
        let asr_delay = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(batch_size, asr_delay, 0., audio_tokenizer, lm)?;

        Ok(KyutaiTranscriber {
            state,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Process a single chunk of audio (1920 samples at 24kHz = 80ms)
    /// Returns any new words recognized
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

    /// Process batched chunks from multiple sources simultaneously (e.g., mic + system)
    /// Returns (stream_index, word) pairs
    pub fn process_batched_chunks(&mut self, chunks: &[&[f32]]) -> Result<Vec<(usize, String)>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = chunks.len();
        let chunk_len = chunks[0].len();

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

    /// Transcribe a full audio file (for stt file command)
    /// Audio must be 24kHz mono f32
    #[allow(dead_code)]
    pub fn transcribe_file(&mut self, mut pcm: Vec<f32>) -> Result<Vec<FileSegment>> {
        // Add delay suffix to flush the model
        let suffix_len = (1.0 * SAMPLE_RATE as f64) as usize; // 1 second of silence
        pcm.resize(pcm.len() + suffix_len, 0.0);

        let mut segments: Vec<FileSegment> = Vec::new();
        let mut current_words: Vec<(String, f64)> = Vec::new();
        let mut pending_word: Option<(String, f64)> = None;

        for chunk in pcm.chunks(CHUNK_SIZE) {
            // Zero-pad final chunk if shorter than CHUNK_SIZE (model requires exact size)
            let padded;
            let chunk = if chunk.len() < CHUNK_SIZE {
                padded = {
                    let mut buf = vec![0.0f32; CHUNK_SIZE];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    buf
                };
                &padded[..]
            } else {
                chunk
            };
            let pcm_tensor = Tensor::new(chunk, &self.device)?.reshape((1, 1, ()))?;
            let msgs = self.state.step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for msg in msgs {
                match msg {
                    moshi::asr::AsrMsg::Word { tokens, start_time, .. } => {
                        if let Some((text, start)) = pending_word.take() {
                            current_words.push((text, start));
                        }
                        let word = self.tokenizer.decode_piece_ids(&tokens).unwrap_or_default();
                        pending_word = Some((word, start_time));
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        if let Some((text, start)) = pending_word.take() {
                            current_words.push((text, start));
                            // EndWord signals sentence boundary
                            if !current_words.is_empty() {
                                let text: String = current_words.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>().join("");
                                let start = current_words.first().map(|(_, s)| *s).unwrap_or(0.0);
                                let end = stop_time;
                                let trimmed = text.trim().to_string();
                                if !trimmed.is_empty() {
                                    segments.push(FileSegment { text: trimmed, start, end });
                                }
                                current_words.clear();
                            }
                        }
                    }
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        // Check for sentence boundary probability
                        if prs[2][0] > 0.5 && !current_words.is_empty() {
                            let text: String = current_words.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>().join("");
                            let start = current_words.first().map(|(_, s)| *s).unwrap_or(0.0);
                            let end = current_words.last().map(|(_, s)| *s).unwrap_or(0.0);
                            let trimmed = text.trim().to_string();
                            if !trimmed.is_empty() {
                                segments.push(FileSegment { text: trimmed, start, end });
                            }
                            current_words.clear();
                        }
                    }
                }
            }
        }

        // Flush remaining words
        if let Some((text, start)) = pending_word {
            current_words.push((text, start));
        }
        if !current_words.is_empty() {
            let text: String = current_words.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>().join("");
            let start = current_words.first().map(|(_, s)| *s).unwrap_or(0.0);
            let end = current_words.last().map(|(_, s)| *s).unwrap_or(start + 0.5);
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                segments.push(FileSegment { text: trimmed, start, end });
            }
        }

        Ok(segments)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FileSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Get the appropriate candle Device (Metal > CUDA > CPU)
pub fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
}
