# myANE

**Reclaiming Apple's most underutilized AI accelerator for general-purpose LLM inference.**

Every Apple Silicon device ships with a dedicated Neural Engine — a matrix-multiply accelerator capable of 15–38 TOPS, depending on the chip. In most devices, it sits nearly idle. Apple uses it narrowly: Siri's speech recognition, camera computational photography, on-device text prediction, and a handful of Core ML classification tasks. These workloads are bursty and lightweight — they might occupy the ANE for milliseconds at a time, leaving the vast majority of its compute cycles unused.

## The underutilization problem

The Neural Engine is present in every iPhone (A11+), iPad, Mac, Apple Watch (S9+), Apple TV (A15+), and Vision Pro. Across hundreds of millions of devices, this is an enormous pool of dormant AI compute:

| Device class | ANE TOPS | Typical ANE utilization | Primary ANE workloads |
|---|---|---|---|
| iPhone / iPad | 15–38 | < 5% average | Camera pipeline, Siri, keyboard prediction |
| MacBook / Mac Studio | 15–38 | < 2% average | Spotlight suggestions, photo search |
| Apple Watch | ~5 | < 1% | Health sensor processing |
| Apple TV | ~15 | ~0% | Essentially unused |
| Vision Pro | ~35 | < 3% | Hand tracking, spatial audio |

On Macs specifically, the situation is striking. The ANE delivers comparable TOPS to the GPU, yet Apple's own ML frameworks (Core ML, Create ML) default to GPU execution for most models. MLX — Apple's official ML framework — uses the GPU exclusively and has no ANE backend at all. The Neural Engine on Mac is effectively a stranded asset.

**Why does this matter now?** The AI era demands concurrent inference. Users want a primary LLM for conversation while background tasks (classification, entity extraction, summarization, search optimization) run simultaneously. On Apple Silicon, the GPU is monopolized by the main model — MLX's Metal backend enforces single-threaded GPU access. The ANE, running on a completely independent data path with zero GPU contention, is the natural solution. But Apple provides no supported path to use it for LLM inference.

## What myANE does

myANE runs LLM inference on the Apple Neural Engine, turning a dormant accelerator into a concurrent AI coprocessor. It operates fully independently from the GPU, enabling parallel inference alongside GPU-based models.

The project uses two approaches:
- **Private ANE API** — Direct access to `AppleNeuralEngine.framework` for maximum control over model compilation and execution (reverse-engineered, undocumented)
- **CoreML hybrid pipeline** — Groups of transformer layers compiled as CoreML models targeting `CPU_AND_NEURAL_ENGINE`, with attention computed on CPU. Achieves 2–5× higher throughput than the private API path through LUT6 quantization and chunked execution

```
GPU (MLX)              ANE (myANE daemon)
─────────              ───────────────────
Main chat model        Utility model (0.8B–4B)
 ↕ PriorityLock        ↕ Unix socket (no lock)
 ↕ Serialized           ↕ Fully concurrent
```

**Use cases for concurrent ANE inference:**
- **Pre-generation pipeline** — prompt classification, entity extraction, search query optimization on ANE while GPU prepares the response
- **Post-generation tasks** — tagging, topic extraction, objective tracking fire on ANE without blocking user interaction
- **Background processing** — document classification, summarization run continuously with zero GPU impact
- **Real-time input analysis** — analyze prompts and show entity hints while the user is still typing

The daemon mode keeps the model resident on ANE, providing ~1–2s latency for utility calls that take 10–35s through the GPU queue.

## Performance

| Model | Backend | Size | tok/s |
|-------|---------|------|-------|
| Qwen3.5-0.8B | CoreML hybrid | 356 MB (LUT6) | **47** |
| Qwen3.5-4B | CoreML hybrid | 2.9 GB (LUT6) | **18** |
| Qwen3.5-0.8B | Private ANE | 1.7 GB (FP16) | 27 |
| Qwen3.5-4B | Private ANE | 9.3 GB (FP16) | 8.9 |

The **hybrid pipeline** fuses groups of layers into large CoreML chunks running on ANE, with CPU handling only the full attention core — 8 cross-device boundaries vs 96+ in per-layer. LUT6 quantization reduces model size by ~3.5×. Key optimizations:
- **ANE-native RMSNorm**: `[x, -x] → LayerNorm → slice` exploits ANE's dedicated LayerNorm hardware instead of manual `pow(2).mean().rsqrt()`
- **In-model top-k LM head**: Argmax runs inside the CoreML model on ANE, returning only 32 candidate tokens instead of 248K floats — cuts ANE→CPU transfer by 99.99%
- **Batch embedding**: Vectorized token embedding lookup for prefill sequences
- **Single-chunk LM head**: Full 262144-row projection in one CoreML op (was 40% of compute, now 18%)

## Supported Models

- Qwen3 (dense)
- Qwen3.5 (dense + DeltaNet hybrid, text-only)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

![image](assets/image.png)

Download a supported model (e.g. `Qwen3-0.6B` or `Qwen3.5-0.8B` in BF16 safetensors format), then:

```bash
# Single-shot generation
./build/ane-lm generate --model /path/to/Qwen3.5-0.8B --prompt "Hello"

# Interactive chat
./build/ane-lm chat --model /path/to/Qwen3.5-0.8B

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane-lm convert --model /path/to/Qwen3.5-0.8B

# Persistent daemon
./build/ane-lm serve --model /path/to/Qwen3.5-0.8B --socket /tmp/ane.sock
```

### Daemon mode

```bash
# Start daemon
./build/ane-lm serve --model /path/to/Qwen3.5-0.8B --socket /tmp/ane.sock

# Request (JSON-lines over Unix socket)
echo '{"id":"1","prompt":"Hello","max_tokens":50,"temperature":0.6}' | nc -U /tmp/ane.sock
# Response: {"id":"1","text":"...","gen_tokens":50,"gen_tps":27.5}
```

### CoreML hybrid export

```bash
# Export Qwen3.5-4B as chunked CoreML models with LUT6 quantization
python scripts/export_hybrid_chunks.py \
    --model models/Qwen3.5-4B \
    --output models/Qwen3.5-4B-hybrid-v2 \
    --lm-head

# Serve the hybrid model
python scripts/serve_hybrid.py --port 8000
```

### Options

```
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--no-ane-cache       Disable persistent ANE compile cache
--socket <path>      Unix socket path (serve mode)
-v, --verbose        Show detailed initialization info
```

## Key Technical Details

### Multi-shard SafeTensors loading
Models split across multiple `.safetensors` files are loaded automatically via `model.safetensors.index.json`.

### Weight prefix auto-detection
Automatically detects weight naming conventions (`model.language_model.`, `model.`, or bare) by probing for `embed_tokens.weight`.

### Asymmetric key/value head support
Supports models where `num_key_heads != num_value_heads` (e.g. Qwen3.5-4B DeltaNet layers with 16 key heads and 32 value heads).

### Private API limitations
Quantized MIL ops (`constexpr_lut_to_dense`, `constexpr_affine_dequantize`, `constexpr_blockwise_shift_scale`) are rejected by the private ANE API. The MIL parser only supports basic runtime ops. Quantized inference requires the CoreML public API path.

### ANE FP16 lesson
Only fuse matmul-heavy ops for ANE. Non-matmul ops (residual adds, RMSNorm, SiLU) need FP32 — keep on CPU. Fusing everything into one ANE graph causes garbled output and is slower.

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) — Original project: LLM inference on Apple Neural Engine via private APIs
- [Anemll/anemll](https://github.com/Anemll/anemll) — HuggingFace→CoreML conversion pipeline; RMSNorm-as-LayerNorm and in-model argmax techniques
- [maderix/ANE](https://github.com/maderix/ANE) — Training neural networks on ANE via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++

## License

MIT License — see [LICENSE](LICENSE) for details.
