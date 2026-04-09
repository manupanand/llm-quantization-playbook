# llm-quantization-playbook

> Production-grade LLM quantization and inference deployment — repeatable pipelines for FP8 and NVFP4 precision targets across NVIDIA GPU architectures.

---

## What This Is

A hands-on, battle-tested collection of scripts and configurations for quantizing large language models and serving them via OpenAI-compatible APIs. Built and validated on real hardware through real failures — not adapted from documentation.

This repo covers the full pipeline: model download → quantization → export → containerized inference server → API validation. Each step includes the exact commands, the known failure modes, and the fixes.

Starting point: **Gemma 4 31B** (multimodal, text + vision) quantized to NVFP4+FP8 using NVIDIA ModelOpt, served via vLLM on Grace Blackwell ARM infrastructure — same architecture as **AWS EC2 p6e-gb200** (Grace ARM CPU + Blackwell GPU + unified NVLink-C2C memory).

More model families and precision targets being added continuously.

---

## Hardware & Precision Matrix

| GPU | Architecture | CUDA SM | AWS Instance | Recommended Format | Status |
|-----|-------------|---------|-------------|-------------------|--------|
| L4 | Ada Lovelace | SM89 | g5, g6 | FP8 | ✅ Production ready |
| A100 | Ampere | SM80 | p4d | INT8 / BF16 | ✅ Production ready |
| H100 | Hopper | SM90 | p5, p5en | FP8 | ✅ Production ready |
| B200 | Blackwell | SM100 | p6-b200 | FP8 / NVFP4 | ✅ FP8 stable |
| GB200 (Grace Blackwell) | Blackwell | SM120 | **p6e-gb200** | FP8 / NVFP4* | ✅ FP8 stable · NVFP4 in progress |

> **p6e-gb200 architecture:** AWS EC2 p6e-gb200 UltraServers pair NVIDIA Grace ARM CPUs with Blackwell GPUs connected via NVLink-C2C — creating a unified CPU+GPU memory pool. This repo is validated on equivalent Grace Blackwell hardware (SM120, aarch64, CUDA 13.0).

> **NVFP4 note:** Requires Blackwell silicon. CUTLASS FP4 GEMM kernel support for SM120 is maturing in the open-source stack — FP8 is the recommended production path on Grace Blackwell today. NVFP4 quantization and export works; inference kernel support is the active gap being tracked upstream in vLLM and FlashInfer.

---

## Repository Structure

```
llm-quantization-playbook/
├── README.md
├── python-scripts/
│   ├── quantize_gemma4_nvfp4.py       # Iteration 1 — baseline NVFP4 attempt
│   ├── quantize_gemma4_nvfp4_2.py     # Iteration 2 — vision layer exclusions
│   └── quantize_gemma4_nvfp4_4.py     # Iteration 4 — hybrid NVFP4+FP8, production config
└── shell-scripts/
    ├── run.sh                          # vLLM container launch
    ├── gemma.sh                        # Model-specific serve config
    ├── fix.sh                          # Patch scripts (index remapping, config fixes)
    └── test.sh                         # curl-based API validation suite
```

---

## Pipeline Overview

```
HuggingFace Model (BF16)
        │
        ▼
  Environment Setup
  nvidia-modelopt · PyTorch · uv venv
        │
        ▼
  Calibration + Quantization
  NVFP4 weights + FP8 activations (text layers)
  BF16 fallback (vision encoder — shape constraints)
        │
        ▼
  HF Checkpoint Export
  model.safetensors · hf_quant_config.json
        │
        ▼
  Containerized Inference Server
  vllm/vllm-openai · OpenAI-compatible API
  /v1/chat/completions · streaming · vision
        │
        ▼
  Validation
  Text · multi-turn · image · streaming · throughput
```

---

## Gemma 4 31B — Results

| Metric | BF16 Original | FP8 Quantized | NVFP4 Quantized |
|--------|--------------|---------------|-----------------|
| Checkpoint size | 59 GB | ~35 GB | **19.6 GB** |
| GPU memory at load | ~110 GB | ~92 GB | ~36 GB (est.) |
| Quantization time | — | ~35 mins | ~35 mins (512 samples) |
| Text inference | ✅ | ✅ | ⏳ kernel pending |
| Vision inference | ✅ | ✅ | ⏳ kernel pending |
| Streaming | ✅ | ✅ | ⏳ kernel pending |
| Multi-turn context | ✅ | ✅ | ⏳ kernel pending |

> NVFP4 achieves 3x compression from BF16 (59GB → 19.6GB). End-to-end inference is blocked pending SM120 CUTLASS FP4 kernel support landing in vLLM/FlashInfer. Tracking upstream.

---

## Quantization — Key Lessons

These are the failure modes that cost real time. Documented so you don't repeat them.

**1. Vision encoder weight shapes break NVFP4 block quantization**
Gemma 4's vision MLP uses `[4304, 1152]` weight shapes. `4304 % 32 = 16` — not divisible by NVFP4's block size of 32. These layers require FP8 or BF16 fallback. The quantizer inserts them silently; the failure surfaces only at export time.

**2. Vision quantizers need image calibration data**
Passing text-only prompts during calibration leaves vision quantizers uncalibrated (`_amax` missing on export). Either pass multimodal calibration batches or explicitly disable vision quantization. Mixing the two without image data causes export to fail.

**3. `hf_quant_config.json` algo string must match inference engine exactly**
ModelOpt exports `W4A8_NVFP4_FP8`. vLLM 0.19.0 expects `NVFP4`. One string mismatch — hours of debugging. Always check `QUANT_ALGOS` in your target inference engine before export.

**4. `model.safetensors.index.json` after single-shard export**
The index file copied from the BF16 source references `model-00001-of-00002.safetensors`. The quantized export produces a single `model.safetensors`. Remap all entries in the index or vLLM fails with `Cannot find any model weights`.

**5. CUTLASS FP4 GEMM is SM-specific**
SM120 (Grace Blackwell / p6e-gb200) support for FlashInfer CUTLASS FP4 kernels is not yet stable in vLLM 0.19.0. `Failed to run cutlass FP4 gemm on sm120/sm121` is a known open issue. FP8 is the production path today.

**6. PyTorch on aarch64 + CUDA 13.0**
Standard `pip install torch --index-url https://download.pytorch.org/whl/cu128` fails on aarch64 — pypi.nvidia.com times out on ARM wheels. Use `https://pypi.org/simple` with PyTorch 2.11.0 which ships native cu130 support for ARM Grace CPU targets.

**7. Use the right vLLM image for Grace Blackwell**
`vllm/vllm-openai:latest` is built for x86. On aarch64 + CUDA 13.0 use `vllm/vllm-openai:latest-cu130-ubuntu2404` — the arm64-native image with correct CUDA 13.0 bindings.

---

## Quick Start

### 1. Environment

```bash
# Create isolated venv with uv
uv venv /models/quant-env --python 3.11
source /models/quant-env/bin/activate

# Install PyTorch (aarch64 + CUDA 13.0 — use pypi.org not whl/cu128)
uv pip install torch==2.11.0 torchvision \
  --index-url https://pypi.org/simple

# Install ModelOpt
uv pip install "nvidia-modelopt" \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple

# Install HF dependencies
uv pip install transformers accelerate datasets sentencepiece safetensors
```

### 2. Download Model

```bash
hf auth login
hf download google/gemma-4-31B-it \
  --local-dir /models/gemma4-31b-bf16 \
  --max-workers 8
```

### 3. Quantize

```bash
nohup /models/quant-env/bin/python3 python-scripts/quantize_gemma4_nvfp4_4.py \
  > /tmp/quant.log 2>&1 &

tail -f /tmp/quant.log
```

### 4. Serve (FP8 — stable on Grace Blackwell today)

```bash
docker run -d \
  --gpus all --runtime nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  --volume /models:/models \
  -e TRANSFORMERS_OFFLINE=1 \
  --name vllm-gemma4 \
  vllm/vllm-openai:latest-cu130-ubuntu2404 \
  /models/gemma4-31b-bf16 \
  --served-model-name gemma4-31b \
  --quantization fp8 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --enforce-eager \
  --host 0.0.0.0 --port 8000
```

### 5. Test

```bash
# Health check
curl http://localhost:8000/health

# Text inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-31b","messages":[{"role":"user","content":"Explain attention mechanisms in 2 sentences."}],"max_tokens":150}'

# Vision inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gemma4-31b",
    "messages":[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
      {"type":"text","text":"What is in this image?"}
    ]}],
    "max_tokens":100
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-31b","messages":[{"role":"user","content":"Write a haiku about quantization."}],"max_tokens":60,"stream":true}'
```

---

## Precision Format Reference

| Format | Weight bits | Activation bits | Hardware | Notes |
|--------|------------|----------------|----------|-------|
| BF16 | 16 | 16 | Ampere+ | Baseline, no compression |
| FP8 (E4M3) | 8 | 8 | Ada / Hopper / Blackwell | Recommended for L4, H100, p6e-gb200 |
| NVFP4 | 4 | 8 (FP8) | Blackwell only | Maximum compression — SM120 kernel support maturing |
| INT8 | 8 | 8 | All | Legacy path |

> Avoid for vLLM pipelines: GPTQ, GGUF, NF4 (bitsandbytes). These do not integrate cleanly with vLLM's quantization backend for Blackwell targets.

---

## Prerequisites

- Docker 24.x+
- NVIDIA Container Toolkit
- GPU driver ≥ 570 (Blackwell / Grace Blackwell) · ≥ 535 (H100/L4)
- CUDA 12.4+ (13.0 for Grace Blackwell ARM / p6e-gb200)
- HuggingFace account with access to target model
- Python 3.11 · uv

---

## AWS Instance Reference

| Instance | GPU | CPU | Unified Memory | Best For |
|----------|-----|-----|----------------|----------|
| `p6-b200` | 8x B200 | Intel Xeon | No (HBM only) | Large-scale training |
| `p6-b300` | 8x B300 Ultra | Intel Xeon | No (HBM only) | Large-scale training |
| `p6e-gb200` | GB200 NVL72 | **Grace ARM** | **Yes (NVLink-C2C)** | Training + inference, unified memory |
| `g7e` | RTX PRO 6000 | Intel Xeon | No | Inference, graphics |

> This repo is validated on **Grace Blackwell** (SM120, aarch64, CUDA 13.0) — architecturally equivalent to `p6e-gb200`. Scripts and container configs apply directly to that instance type.

---

## Roadmap

- [ ] Llama 3.x FP8 pipeline
- [ ] Qwen2.5 NVFP4 pipeline
- [ ] Mistral / Mixtral MoE FP8
- [ ] DeepSeek-R1 quantization
- [ ] Multi-node tensor parallel config
- [ ] Prometheus + Grafana monitoring stack
- [ ] NVFP4 inference on SM120 (tracking upstream vLLM/FlashInfer)

---

## Contributing

Open an issue before submitting a PR for significant changes. All scripts should be tested against the hardware and container versions documented above.

---

## License

Apache 2.0

---

## Disclaimer

Not affiliated with or endorsed by NVIDIA Corporation, AWS, Google, or any model vendor. All tooling is subject to its own upstream licensing terms.
