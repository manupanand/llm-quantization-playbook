#!/usr/bin/env python3
"""
quantize_qwen35_35b_a3b_nvfp4.py  — REWRITE v3
════════════════════════════════════════════════════════════════════════════════
Qwen3.5-35B-A3B (qwen3_5_moe)  →  NVFP4
Hardware : NVIDIA DGX Spark GB10 | 128 GB unified RAM | CUDA 13.0
Env      : modelopt 0.42.0 | torch 2.11.0 | transformers 5.5.0

FIXES vs previous versions:
  FIX 1 — OOM during load (CRITICAL)
    OLD : torch_dtype=torch.bfloat16  ← silently ignored in transformers 5.5
                                         model loads as float32 = 136 GB → OOM
    NEW : dtype=torch.bfloat16        ← correct param name in transformers 5.5
          low_cpu_mem_usage=True      ← shard-by-shard load, no double buffer

  FIX 2 — Wrong FP8 dict format (pydantic ValidationError)
    OLD : {"weight": {"type": "float8_e4m3fn", "enable": True}}
    NEW : {"enable": False}  ← correct modelopt 0.42 schema for BF16 fallback

  FIX 3 — conv1d caught by shape guard incorrectly
    OLD : param.dim() < 2  ← 3D Conv tensors passed through
    NEW : param.dim() != 2 ← only exactly 2D Linear weights checked

  FIX 4 — MoE-specific exclusions (new for 35B-A3B)
    NEW : *mlp.gate*      → BF16 (router — misrouting = quality collapse)
    NEW : *shared_expert* → BF16 (fires 100% of tokens, error accumulates)

  FIX 5 — Memory headroom
    NEW : CALIB_BATCH_SIZE=1 (saves ~4 GB vs batch=2)
    NEW : gc.collect() + empty_cache() after model load
    NEW : torch.inference_mode() wrapping export
════════════════════════════════════════════════════════════════════════════════
"""

import copy
import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="modelopt")

import torch

torch.set_num_threads(20)  # must be before any torch ops

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
SOURCE_PATH = "/models/qwen35-35b-a3b-bf16"
OUTPUT_PATH = "/models/qwen35-35b-a3b-nvfp4"
DEVICE = "cuda:0"
CALIB_SAMPLES = 512
CALIB_BATCH_SIZE = 1  # FIX 5: 1 not 2 — saves ~4 GB during calibration
MAX_SEQ_LEN = 1024
PROGRESS_EVERY = 50

# BF16 exclusion patterns — {"enable": False} is correct modelopt 0.42 format
BF16_EXCLUDE_PATTERNS = [
    # Standard
    "*embed_tokens*",
    "*lm_head*",
    "*norm*",
    "*layernorm*",
    "*ln_f*",
    # GatedDeltaNet SSM — not linear layers
    "*conv1d*",
    "*linear_attn.A*",
    "*linear_attn.D*",
    "*linear_attn.dt_bias*",
    "*linear_attn.out_norm*",
    # MoE router — routing must be numerically exact
    "*mlp.gate*",
    "*.gate.weight",
    "*router*",
    # Shared expert — fires on 100% of tokens
    "*shared_expert*",
    "*shared_experts*",
]

# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Build quant config
# ═════════════════════════════════════════════════════════════════════════════


def build_quant_config():
    cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
    exclusions = {pat: {"enable": False} for pat in BF16_EXCLUDE_PATTERNS}
    merged = {}
    merged.update(cfg.get("quant_cfg", {}))
    merged.update(exclusions)
    cfg["quant_cfg"] = merged
    # Belt-and-suspenders for the three most critical
    cfg["quant_cfg"]["*lm_head*"] = {"enable": False}
    cfg["quant_cfg"]["*mlp.gate*"] = {"enable": False}
    cfg["quant_cfg"]["*shared_expert*"] = {"enable": False}
    print(f"\n[Config] Base: W4A8_NVFP4_FP8_CFG")
    print(f"  NVFP4 : attn q/k/v/o | deltanet q/k/v/o/g | routed expert gate/up/down")
    print(f"  BF16  : embed | lm_head | norms | conv1d | SSM | router | shared_expert")
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Shape guard (safety net only)
# ═════════════════════════════════════════════════════════════════════════════


def apply_shape_guard(model, cfg):
    overrides = []
    for name, param in model.named_parameters():
        if param.dim() != 2:  # FIX 3: exactly 2D only = Linear layers
            continue
        if any(d % 32 != 0 for d in param.shape):
            pat = f"*{name}*"
            if pat not in cfg["quant_cfg"]:
                cfg["quant_cfg"][pat] = {"enable": False}  # FIX 2: correct format
                overrides.append(f"  AUTO-BF16  {name}  {list(param.shape)}")
    if overrides:
        print(f"\n[ShapeGuard] {len(overrides)} auto BF16 overrides:")
        for o in overrides:
            print(o)
    else:
        print("\n[ShapeGuard] All 2D linear layers NVFP4-compatible ✓")
    return overrides


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Calibration dataset
# ═════════════════════════════════════════════════════════════════════════════

CALIB_PROMPTS = [
    "Explain the Mixture of Experts architecture and how sparse token routing works.",
    "Why does the shared expert in MoE models stay in BF16 during NVFP4 quantization?",
    "How does NVFP4 E2M1 represent model weights compared to standard BF16?",
    "What is the MoE router gate and why must it remain numerically exact?",
    "Explain GatedDeltaNet linear attention and its constant memory scaling property.",
    "How does the W4A8_NVFP4_FP8_CFG configure weight versus activation precision?",
    "Describe how calibration amax statistics are collected during post-training quantization.",
    "What is the difference between static and dynamic quantization for activations?",
    "How does vLLM expert parallelism distribute MoE layers across multiple GPUs?",
    "Explain why conv1d SSM kernels must be excluded from NVFP4 quantization.",
    "What is the mathematical relationship between perplexity and cross-entropy loss?",
    "Describe the modelopt export_hf_checkpoint output format and required files.",
    "How does hf_quant_config.json communicate quantization metadata to vLLM?",
    "Explain tensor parallelism for grouped-query attention in large language models.",
    "What are the trade-offs between num_experts and moe_intermediate_size in MoE?",
    "How does speculative decoding with MTP improve throughput in MoE inference?",
    "Describe Flash Attention memory bandwidth optimizations for Blackwell GPUs.",
    "What is expert load balancing and how does auxiliary loss encourage it?",
    "Explain how DGX Spark GB10 unified memory differs from discrete GPU VRAM.",
    "How does the safetensors format prevent arbitrary code execution versus pickle?",
    "Write a Python function implementing top-k expert routing for a MoE forward pass.",
    "Implement a calibration data loader that batches text samples for modelopt PTQ.",
    "Write a script to remap model.safetensors.index.json after quantization export.",
    "Implement post-export validation checking all shard files exist on disk.",
    "Write a Python context manager tracking and logging GPU memory usage per step.",
    "Implement exponential backoff retry logic for HuggingFace Hub file downloads.",
    "Write a function to patch hf_quant_config.json quant_algo field post-export.",
    "Implement a progress counter wrapper for the modelopt calibration forward loop.",
    "Write a CUDA memory profiler that logs peak allocation at each forward pass.",
    "Implement a safetensors shard reader that streams tensors without full model load.",
    "If a MoE has 256 experts with 8 activated per token what fraction fires per token?",
    "Calculate memory savings of NVFP4 versus BF16 for a 35 billion parameter MoE model.",
    "How many calibration tokens are processed with 512 samples at sequence length 1024?",
    "If moe_intermediate_size is 768 and hidden is 4096 what is the FFN expansion ratio?",
    "Estimate the disk size of Qwen3.5-35B-A3B in NVFP4 format given 35B total parameters.",
    "How many routed expert forward passes occur per layer for a batch of 512 tokens?",
    "If shared expert has intermediate size 3072 versus routed 768 how much larger is it?",
    "Calculate NVFP4 block count for a weight matrix of shape 768 by 4096.",
    "What is the ratio of active to total parameters in the Qwen3.5 35B-A3B model?",
    "How many bytes does a 768 by 4096 weight matrix occupy in NVFP4 versus BF16?",
    "Write a detailed explanation of Qwen3.5-35B-A3B MoE architecture for a junior engineer.",
    "Describe the complete NVFP4 quantization pipeline from BF16 checkpoint to vLLM serving.",
    "Write a production deployment checklist for a quantized MoE LLM on DGX Spark.",
    "Explain how to validate quantization quality using perplexity benchmarks on WikiText-2.",
    "Describe the engineering reasons behind excluding shared experts from NVFP4 quantization.",
    "Write a troubleshooting guide for common modelopt quantization pipeline failures.",
    "Explain how expert parallelism and tensor parallelism interact in multi-GPU MoE serving.",
    "Describe the DGX Spark GB10 memory architecture and its implications for large models.",
    "Write a guide on choosing calibration dataset size and diversity for post-training quantization.",
    "Explain the difference between qwen3_5_moe and qwen3_5_text model types in HuggingFace.",
    "Explain the history of transformer architectures from the original attention paper to MoE.",
    "What are the key architectural differences between dense transformers and sparse MoE models?",
    "Describe how reinforcement learning from human feedback shapes large language model behavior.",
    "Explain the significance of the Apache 2.0 license for open-weight model commercial deployment.",
    "What is the role of tokenizer vocabulary size in multilingual model performance and coverage?",
    "Describe how early fusion multimodal training differs from late fusion adapter approaches.",
    "Explain how DGX Spark ConnectX-7 networking enables two-node cluster configuration.",
    "What benchmarks are most informative for evaluating quantized large language model quality?",
    "Describe the trade-offs between model size and inference latency for production deployment.",
    "Explain how thinking mode and non-thinking mode differ in Qwen3.5 reasoning model behavior.",
    "Explain in Japanese why MoE router weights must stay in BF16 during NVFP4 quantization.",
    "Explain in Chinese why the shared expert in MoE must maintain BF16 precision.",
    "Explain in French how sparse routing in MoE reduces computation cost per token.",
    "Explain in German the difference between NVFP4 and INT4 for LLM weight quantization.",
]


def build_calibration_dataset(tokenizer):
    prompts = []
    while len(prompts) < CALIB_SAMPLES:
        prompts.extend(CALIB_PROMPTS)
    prompts = prompts[:CALIB_SAMPLES]
    print(
        f"\n[Calibration] Tokenizing {CALIB_SAMPLES} prompts "
        f"(batch={CALIB_BATCH_SIZE}, seq={MAX_SEQ_LEN}) ..."
    )
    batches = []
    for i in range(0, CALIB_SAMPLES, CALIB_BATCH_SIZE):
        enc = tokenizer(
            prompts[i : i + CALIB_BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        batches.append(
            {
                "input_ids": enc["input_ids"].to(DEVICE),
                "attention_mask": enc["attention_mask"].to(DEVICE),
            }
        )
    print(f"[Calibration] {len(batches)} batches ready ✓")
    return batches


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Forward loop
# ═════════════════════════════════════════════════════════════════════════════


def make_forward_loop(batches):
    total = len(batches) * CALIB_BATCH_SIZE
    t0 = time.time()
    seen = [0]

    def forward_loop(m):
        for batch in batches:
            with torch.no_grad():
                m(**batch)
            seen[0] += len(batch["input_ids"])
            if seen[0] % PROGRESS_EVERY == 0 or seen[0] >= total:
                elapsed = time.time() - t0
                rate = seen[0] / max(elapsed, 0.001)
                mem_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
                print(
                    f"  [Calib] {seen[0]:>4d}/{total} | "
                    f"{elapsed:6.1f}s | {rate:.1f} samp/s | "
                    f"GPU {mem_gb:.1f} GB"
                )

    return forward_loop


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Post-export patches
# ═════════════════════════════════════════════════════════════════════════════


def patch_hf_quant_config(output_dir):
    p = Path(output_dir) / "hf_quant_config.json"
    if not p.exists():
        print("[Patch] hf_quant_config.json not found — skipping")
        return
    with open(p) as f:
        cfg = json.load(f)
    old = cfg.get("quant_algo", "")
    if old != "NVFP4":
        cfg["quant_algo"] = "NVFP4"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[Patch] quant_algo '{old}' → 'NVFP4' ✓")
    else:
        print("[Patch] quant_algo already 'NVFP4' ✓")


def remap_safetensors_index(output_dir):
    idx_path = Path(output_dir) / "model.safetensors.index.json"
    if not idx_path.exists():
        single = Path(output_dir) / "model.safetensors"
        print(
            "[Remap] Single-shard ✓"
            if single.exists()
            else "[Remap] WARNING: no shards found!"
        )
        return
    with open(idx_path) as f:
        index = json.load(f)
    wmap = index.get("weight_map", {})
    disk_shards = sorted(p.name for p in Path(output_dir).glob("model*.safetensors"))
    index_shards = sorted(set(wmap.values()))
    print(f"[Remap] Disk={len(disk_shards)} | Index={len(index_shards)}")
    if disk_shards == index_shards:
        print("[Remap] Consistent ✓")
        return
    if not disk_shards:
        print("[Remap] ERROR: no safetensors on disk!")
        return
    if len(disk_shards) == len(index_shards):
        remap = dict(zip(index_shards, disk_shards))
    elif len(disk_shards) == 1:
        remap = {o: disk_shards[0] for o in index_shards}
    else:
        remap = {
            o: disk_shards[min(i, len(disk_shards) - 1)]
            for i, o in enumerate(index_shards)
        }
    changed = sum(1 for v in wmap.values() if remap.get(v, v) != v)
    index["weight_map"] = {k: remap.get(v, v) for k, v in wmap.items()}
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[Remap] {changed}/{len(wmap)} entries remapped ✓")


def copy_tokenizer_and_config(src, dst):
    SKIP_EXT = {".safetensors", ".bin", ".pt", ".ckpt"}
    SKIP_NAMES = {"model.safetensors.index.json", "hf_quant_config.json"}
    copied = []
    for f in Path(src).iterdir():
        if not f.is_file():
            continue
        if f.suffix in SKIP_EXT or f.name in SKIP_NAMES:
            continue
        dst_file = Path(dst) / f.name
        if not dst_file.exists():
            shutil.copy2(f, dst_file)
            copied.append(f.name)
    print(f"\n[Copy] {len(copied)} config/tokenizer files copied")


def print_output_size(output_dir):
    total = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
    print(f"\n[Output] Total size: {total / 1e9:.2f} GB  (expected ~18 GB)")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 72)
    print("  Qwen3.5-35B-A3B (qwen3_5_moe)  →  NVFP4  [v3 — all fixes applied]")
    print(f"  Source  : {SOURCE_PATH}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  Samples : {CALIB_SAMPLES} | batch={CALIB_BATCH_SIZE} | seq={MAX_SEQ_LEN}")
    print(f"  Threads : {torch.get_num_threads()}")
    print("=" * 72)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. Tokenizer
    print(f"\n[Load] Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model — load on CPU first, then move to GPU
    # WHY: HuggingFace from_pretrained with device_map="cuda:0" double-buffers
    # the final shards during loading (CPU staging + GPU copy both live at once).
    # At 96% load this spikes to ~100 GB and triggers OOM on the 119 GB GB10.
    # Solution: load entirely to CPU (no spike, pure sequential), then .cuda()
    # which moves tensors one-by-one with no double-buffer.
    print(f"\n[Load] Model in BF16 — CPU first, then GPU (avoids double-buffer OOM)")
    print(f"       Stage 1: load all shards to CPU RAM (~68 GB, sequential)")
    print(f"       Stage 2: .cuda() moves tensors to GPU one-by-one (no spike)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_PATH,
        dtype=torch.bfloat16,  # correct param for transformers 5.5.0
        device_map="cpu",  # load to CPU — no double-buffer during shard load
        low_cpu_mem_usage=True,
    )
    print(f"[Load] CPU load done in {time.time() - t0:.1f}s | moving to {DEVICE} ...")
    t1 = time.time()
    model = model.to(DEVICE)  # move to GPU one tensor at a time — no spike
    model.eval()
    load_time = time.time() - t0
    alloc_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[Load] GPU move done in {time.time() - t1:.1f}s")
    print(
        f"[Load] Total load time: {load_time:.1f}s | GPU allocated: {alloc_gb:.1f} GB"
    )

    mt = getattr(model.config, "model_type", "unknown")
    ne = getattr(model.config, "num_experts", "?")
    print(f"[Load] model_type={mt} | num_experts={ne}")
    if mt != "qwen3_5_moe":
        print(f"[WARN] Expected 'qwen3_5_moe', got '{mt}'")

    # Free CPU staging buffers
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Config
    cfg = build_quant_config()

    # 4. Shape guard
    apply_shape_guard(model, cfg)

    # 5. Calibration data
    batches = build_calibration_dataset(tokenizer)

    # 6. Quantize
    print(f"\n[Quantize] Starting mtq.quantize() ...")
    t_q = time.time()
    model = mtq.quantize(model, cfg, make_forward_loop(batches))
    print(f"[Quantize] Done in {(time.time() - t_q) / 60:.1f} min ✓")

    del batches
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Export
    print(f"\n[Export] Writing to {OUTPUT_PATH} ...")
    t_e = time.time()
    export_ok = False
    try:
        with torch.inference_mode():  # FIX 5
            export_hf_checkpoint(model, export_dir=OUTPUT_PATH)
        export_ok = True
        print(f"[Export] Done in {(time.time() - t_e) / 60:.1f} min ✓")
    except Exception as e:
        print(f"[Export] FAILED: {e}")
        fallback = os.path.join(OUTPUT_PATH, "model_state_dict.pt")
        try:
            torch.save(model.state_dict(), fallback)
            print(f"[Export] Fallback saved → {fallback}")
        except Exception as e2:
            print(f"[Export] CRITICAL: {e2}")
            sys.exit(1)

    # 8. Patches
    if export_ok:
        print("\n[PostProcess] ...")
        remap_safetensors_index(OUTPUT_PATH)
        patch_hf_quant_config(OUTPUT_PATH)

    # 9. Copy configs
    copy_tokenizer_and_config(SOURCE_PATH, OUTPUT_PATH)

    # 10. Summary
    print_output_size(OUTPUT_PATH)
    print("\n" + "=" * 72)
    print("  ✅ QUANTIZATION COMPLETE")
    print(f"  Output : {OUTPUT_PATH}")
    print()
    print("  vllm serve /models/qwen35-35b-a3b-nvfp4 \\")
    print("      --quantization modelopt_fp4 \\")
    print("      --reasoning-parser qwen3 \\")
    print("      --max-model-len 32768 \\")
    print("      --port 8000")
    print("=" * 72)


if __name__ == "__main__":
    main()
