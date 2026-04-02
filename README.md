# llm-quantization-playbook

**Production-grade LLM quantization and deployment pipeline for NVIDIA GPU infrastructure.**  
Covers FP8 (L4, H100) and NVFP4 (GB10 / DGX Spark) precision targets using TensorRT-LLM and NVIDIA ModelOpt.

---

## Overview

This repository provides a structured, repeatable pipeline for quantizing HuggingFace large language models and deploying them as high-performance inference servers on NVIDIA GPU hardware.

It is designed for infrastructure and ML platform engineers responsible for delivering production-grade model serving at scale. Each step is documented with exact commands, configuration references, and hardware-specific guidance.

---

## Supported Hardware & Precision Matrix

| GPU | Architecture | CUDA SM | Recommended Precision | Production Ready |
|-----|-------------|---------|----------------------|-----------------|
| L4 | Ada Lovelace | SM89 | FP8 | ✅ |
| A100 | Ampere | SM80 | INT8 / BF16 | ✅ |
| H100 | Hopper | SM90 | FP8 | ✅ |
| GB10 — DGX Spark | Blackwell | SM100 | NVFP4 | ✅ |

> **Note:** NVFP4 requires Blackwell silicon (SM100). It has no hardware acceleration on prior architectures. TensorRT-LLM will reject or silently fall back to FP16 if deployed on incompatible hardware.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SOURCE MODEL                         │
│             HuggingFace (gated or public)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               ENVIRONMENT SETUP                         │
│   NGC Container · CUDA Drivers · ModelOpt · TRT-LLM     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  QUANTIZATION                           │
│   NVIDIA ModelOpt — FP8 (L4/H100) · NVFP4 (GB10)       │
│   Calibration dataset · Weight + activation scaling     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               ENGINE COMPILATION                        │
│   trtllm-build — target SM · KV cache dtype             │
│   CUDA graphs · Paged attention · Tensor parallelism    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              INFERENCE SERVER                           │
│   trtllm-serve — OpenAI-compatible API                  │
│   Continuous batching · Streaming · Health endpoints    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           VALIDATION & BENCHMARKING                     │
│   Accuracy regression · Latency · Throughput (tok/s)   │
└─────────────────────────────────────────────────────────┘
```

---

## Dev-to-Production Promotion Strategy

This repository is structured around a two-stage workflow. Engineers validate the full pipeline on an L4 development node using FP8 precision, then promote to GB10 production hardware by changing two flags.

```
Development (L4 — FP8)            Production (GB10 — NVFP4)
──────────────────────────         ──────────────────────────
quantize_fp8.py              →     quantize_nvfp4.py
--use_fp8_context_fmha       →     --use_fp4_context_fmha
All other steps identical    →     All other steps identical
```

The quantized checkpoint format (safetensors) is consistent across both targets. Only the precision flag and engine compilation step differ. This approach minimises risk and ensures pipeline correctness before production deployment.

---

## Repository Structure

```
llm-quantization-playbook/
├── README.md
├── docs/
│   ├── 01-environment-setup.md       # NGC container, drivers, dependencies
│   ├── 02-model-download.md          # HuggingFace CLI, gated model access
│   ├── 03-quantization.md            # ModelOpt FP8 and NVFP4 workflows
│   ├── 04-engine-build.md            # trtllm-build flags per GPU target
│   ├── 05-inference-server.md        # trtllm-serve, API, batching config
│   ├── 06-benchmarking.md            # Latency, throughput, accuracy validation
│   └── 07-troubleshooting.md         # Common errors and resolutions
├── scripts/
│   ├── quantize_fp8.py               # L4 / H100 quantization
│   ├── quantize_nvfp4.py             # GB10 / Blackwell quantization
│   ├── build_engine.sh               # trtllm-build wrapper
│   └── serve.sh                      # trtllm-serve launcher
├── configs/
│   ├── l4_fp8.json                   # L4 inference config
│   ├── h100_fp8.json                 # H100 inference config
│   └── gb10_nvfp4.json               # GB10 / DGX Spark inference config
└── docker/
    └── Dockerfile.trtllm             # Reproducible build environment
```

---

## Precision Format Reference

| Format | Weight Bits | Activation Bits | Hardware | Use Case |
|--------|------------|----------------|----------|----------|
| FP16 | 16 | 16 | All | Baseline, no compression |
| BF16 | 16 | 16 | Ampere+ | Training stability |
| FP8 (E4M3) | 8 | 8 | Ada, Hopper, Blackwell | Recommended for L4 / H100 |
| INT8 | 8 | 8 | All | Legacy quantization |
| NVFP4 | 4 | 8 | Blackwell only | Maximum efficiency on GB10 |

> **Formats to avoid in TensorRT-LLM pipelines:** GPTQ, GGUF, NF4 (bitsandbytes). These formats are incompatible with TRT engine compilation and will not yield hardware-accelerated inference.

---

## Prerequisites

- **Docker** 24.x or later
- **NVIDIA Container Toolkit** (nvidia-docker2)
- **GPU Driver** ≥ 535 (L4 / H100) · ≥ 570 (GB10)
- **CUDA** 12.4+
- **NGC Account** — [ngc.nvidia.com](https://ngc.nvidia.com)
- **HuggingFace Account** with access to target model

---

## Compatibility

| Component | Minimum Version |
|-----------|----------------|
| TensorRT-LLM | 0.17.0 |
| NVIDIA ModelOpt | 0.19.0 |
| TensorRT | 10.x |
| CUDA | 12.4 |
| NGC Container | `nvcr.io/nvidia/tensorrt_llm:latest` |

---

## Toolchain

| Tool | Role |
|------|------|
| [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Quantization — FP8 / NVFP4 calibration |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Engine compilation and inference runtime |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | Optional production serving layer |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Source model format |

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes. Ensure all scripts are tested against the NGC container specified in `docker/Dockerfile.trtllm`.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE) for details.

---

## Disclaimer

This repository is not affiliated with or endorsed by NVIDIA Corporation. TensorRT-LLM, ModelOpt, and related tooling are subject to NVIDIA's own licensing terms.
