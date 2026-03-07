# ANE-LM

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs.

Fork of [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) with extensions for multi-shard models, asymmetric GQA, and daemon mode for concurrent utility inference.

## What's New

### Multi-Shard SafeTensors Loading
Models split across multiple `.safetensors` files (e.g. `model-00001-of-00002.safetensors`) are loaded automatically via `model.safetensors.index.json`. Falls back to single-file loading if no index exists.

### Weight Prefix Auto-Detection
Automatically detects weight naming conventions (`model.language_model.`, `model.`, or bare) by probing for `embed_tokens.weight`. No manual configuration needed.

### Asymmetric Key/Value Head Support
Supports models where `num_key_heads != num_value_heads` (e.g. Qwen3.5-4B DeltaNet layers with 16 key heads and 32 value heads). Implements GQA-style key head grouping in the SSM loop.

### Daemon Mode (`serve`)
Persistent daemon process that keeps the model loaded on ANE and accepts requests over a Unix socket. Eliminates cold-start latency for repeated inference calls.

```bash
# Start daemon
./build/ane-lm serve --model /path/to/Qwen3.5-0.8B --socket /tmp/ane.sock

# Request (JSON-lines over Unix socket)
echo '{"id":"1","prompt":"Hello","max_tokens":50,"temperature":0.6}' | nc -U /tmp/ane.sock
# Response: {"id":"1","text":"...","gen_tokens":50,"gen_tps":27.5}
```

### Quantization Probes
Tested `constexpr_lut_to_dense`, `constexpr_affine_dequantize`, and `constexpr_blockwise_shift_scale` MIL ops. All rejected by the private ANE API (`InvalidMILProgram`). The private API MIL parser only supports basic runtime ops (`const`, `conv`, `add`, `mul`). Quantized inference requires the CoreML public API path.

## Motivation: MLX GPU Contention

[MLX](https://github.com/ml-explore/mlx) is the standard framework for LLM inference on Apple Silicon, but it has a fundamental threading constraint: **Metal GPU operations must execute on the main thread**. All MLX compute — matrix multiplications, attention, sampling — is dispatched through a single Metal command queue bound to the main thread. This means:

- You cannot run two MLX models concurrently. A second `mx.eval()` call blocks until the first completes.
- `asyncio.to_thread()` does not help — Metal calls from a background thread will deadlock or crash.
- Applications that need a primary chat model *and* auxiliary utility calls (classification, routing, entity extraction) must serialize all GPU work through a single lock, adding seconds of latency per request.

In practice, this forces a painful tradeoff: either run utility tasks on the same GPU with a priority lock (adding 10-22s of pre-generation delay while the utility model holds the GPU), or skip utility tasks entirely.

### ANE as a Parallel Accelerator

The Apple Neural Engine is a separate hardware accelerator with its own memory bus and compute pipeline. It shares unified memory with the GPU but has **zero contention** — ANE and GPU operate fully independently. This makes it ideal for offloading lightweight utility inference while the GPU runs the primary model:

```
GPU (MLX)          ANE (ANE-LM daemon)
─────────          ───────────────────
Main chat model    Utility model (0.8B-4B)
 ↕ PriorityLock    ↕ Unix socket (no lock needed)
 ↕ Blocked          ↕ Concurrent
```

**Use cases for concurrent ANE inference:**
- **Pre-generation pipeline** — prompt classification, entity extraction, search query optimization all run on ANE while the GPU prepares the chat response
- **Post-generation tasks** — conversation tagging, topic extraction, objective tracking fire on ANE without interrupting the next user interaction
- **Background processing** — document classification, summarization, and entity extraction run continuously on ANE with zero impact on chat latency
- **Real-time input analysis** — classify prompts, predict required sources, and show entity chips while the user is still typing

The daemon mode (`serve`) keeps the model resident on ANE, eliminating cold-start overhead and providing ~1-2s latency for utility calls that previously took 10-35s through the GPU queue.

## Supported Models

- Qwen3 (dense)
- Qwen3.5 (dense, text-only)

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

### Performance

| Model | Backend | Size | tok/s |
|-------|---------|------|-------|
| Qwen3.5-0.8B | CoreML per-layer | 356 MB (LUT6) | **47** |
| Qwen3.5-4B | Hybrid chunked | 2.7 GB (LUT6) | **21** |
| Qwen3.5-0.8B | Private ANE | 1.7 GB (FP16) | 27 |
| Qwen3.5-4B | Private ANE | 9.3 GB (FP16) | 8.9 |

The **hybrid pipeline** fuses groups of layers into large CoreML chunks running on ANE, with CPU handling only the full attention core — 8 cross-device boundaries vs 96+ in per-layer. LUT6 quantization reduces model size by ~3.5×. The ANE operates independently from the GPU, enabling concurrent inference with GPU-based models (e.g. MLX).

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) - Original project: LLM inference on Apple Neural Engine via private APIs
- [maderix/ANE](https://github.com/maderix/ANE) - Training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++

## License

MIT License - see [LICENSE](LICENSE) for details.
