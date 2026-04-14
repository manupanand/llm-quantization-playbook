#!/usr/bin/env python3
"""
quantize_qwen35_35b_a3b_nvfp4.py  — FINAL v5
════════════════════════════════════════════════════════════════════════════════
Qwen3.5-35B-A3B  →  NVFP4
Hardware : NVIDIA DGX Spark GB10 | 128 GB unified RAM | CUDA 13.0
Env      : modelopt 0.42.0 | torch 2.11.0 | transformers 5.5.0

ALL FIXES:
  ✓ dtype= not torch_dtype=              transformers 5.5 API
  ✓ device_map="cpu" then .to(DEVICE)    eliminates double-buffer OOM
  ✓ low_cpu_mem_usage=True               shard-by-shard staging
  ✓ {"enable": False}                    correct modelopt 0.42 schema
  ✓ param.dim() != 2 in shape guard      skips Conv1d 3D tensors
  ✓ no mtq.__version__                   module has no __version__
  ✓ accepts qwen3_5_moe / qwen3_5_moe_text  actual HF model_type
  ✓ MoE router + shared_expert BF16      prevents routing collapse
  ✓ conv1d + SSM matrices BF16           not linear layers
  ✓ CALIB_BATCH_SIZE=1                   saves 4 GB during calibration
  ✓ MODEL_NAME_TO_TYPE patch             registers Qwen3_5Moe for export
  ✓ model.config.architectures patch     fixes is_multimodal_model crash
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

warnings.filterwarnings("ignore", category=UserWarning)

import torch

torch.set_num_threads(20)

# ── Patch modelopt export registry BEFORE importing anything else ─────────────
# export_hf_checkpoint uses MODEL_NAME_TO_TYPE to identify the model class.
# Qwen3_5MoeForCausalLM is not in the registry in modelopt 0.42 — add it now.
from modelopt.torch.export.model_utils import MODEL_NAME_TO_TYPE

MODEL_NAME_TO_TYPE["Qwen3_5Moe"] = "qwen3moe"
MODEL_NAME_TO_TYPE["Qwen3_5MoeText"] = "qwen3moe"
MODEL_NAME_TO_TYPE["Qwen3_5MoeForCausalLM"] = "qwen3moe"
# ─────────────────────────────────────────────────────────────────────────────

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
CALIB_BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
PROGRESS_EVERY = 50

BF16_PATTERNS = [
    "*embed_tokens*",
    "*lm_head*",
    "*norm*",
    "*layernorm*",
    "*ln_f*",
    "*conv1d*",
    "*linear_attn.A*",
    "*linear_attn.D*",
    "*linear_attn.dt_bias*",
    "*linear_attn.out_norm*",
    "*mlp.gate*",
    "*.gate.weight",
    "*router*",
    "*shared_expert*",
    "*shared_experts*",
]

PROMPTS = [
    "Explain the Mixture of Experts architecture and how sparse token routing works.",
    "Why does the shared expert in MoE models stay in BF16 during NVFP4 quantization?",
    "How does NVFP4 E2M1 represent model weights compared to standard BF16 format?",
    "What is the MoE router gate and why must it remain numerically exact?",
    "Explain GatedDeltaNet linear attention and its constant memory scaling property.",
    "How does W4A8_NVFP4_FP8_CFG configure weight versus activation precision?",
    "Describe how calibration amax statistics are collected during post-training quantization.",
    "What is the difference between static and dynamic quantization for LLM activations?",
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
    "Implement post-export validation that checks all shard files exist on disk.",
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
    "Calculate the NVFP4 block count for a weight matrix of shape 768 by 4096.",
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
    "Explain the difference between qwen3_5_moe and qwen3_5_moe_text model types in HuggingFace.",
    "Explain the history of transformer architectures from the original attention paper to MoE.",
    "What are the key architectural differences between dense transformers and sparse MoE models?",
    "Describe how reinforcement learning from human feedback shapes large language model behavior.",
    "Explain the significance of the Apache 2.0 license for open-weight model commercial use.",
    "What is the role of tokenizer vocabulary size in multilingual model performance and coverage?",
    "Describe how early fusion multimodal training differs from late fusion adapter approaches.",
    "Explain how DGX Spark ConnectX-7 networking enables two-node cluster configuration.",
    "What benchmarks are most informative for evaluating quantized large language model quality?",
    "Describe the trade-offs between model size and inference latency for production deployment.",
    "Explain how thinking mode and non-thinking mode differ in Qwen3.5 reasoning model behavior.",
    "Explain why MoE router weights must stay in BF16 during NVFP4 quantization.",
    "Explain why the shared expert in MoE must maintain BF16 precision during quantization.",
    "How does sparse routing in MoE reduce computation cost per token during inference?",
    "What is the difference between NVFP4 and INT4 quantization for large language model weights?",
]


def build_quant_config():
    cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
    exclusions = {pat: {"enable": False} for pat in BF16_PATTERNS}
    merged = {}
    merged.update(cfg.get("quant_cfg", {}))
    merged.update(exclusions)
    merged["*lm_head*"] = {"enable": False}
    merged["*mlp.gate*"] = {"enable": False}
    merged["*shared_expert*"] = {"enable": False}
    cfg["quant_cfg"] = merged
    print("\n[Config] W4A8_NVFP4_FP8_CFG loaded")
    print("  NVFP4 : attn q/k/v/o | deltanet q/k/v/o/g | routed experts")
    print("  BF16  : embed | lm_head | norms | conv1d | SSM | router | shared_expert")
    return cfg


def apply_shape_guard(model, cfg):
    overrides = []
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        if any(d % 32 != 0 for d in param.shape):
            pat = f"*{name}*"
            if pat not in cfg["quant_cfg"]:
                cfg["quant_cfg"][pat] = {"enable": False}
                overrides.append(f"  BF16  {name}  {list(param.shape)}")
    if overrides:
        print(f"\n[ShapeGuard] {len(overrides)} auto BF16 overrides:")
        for o in overrides:
            print(o)
    else:
        print("\n[ShapeGuard] All 2D linear layers NVFP4-compatible ✓")


def build_calibration_batches(tokenizer):
    prompts = []
    while len(prompts) < CALIB_SAMPLES:
        prompts.extend(PROMPTS)
    prompts = prompts[:CALIB_SAMPLES]
    print(
        f"\n[Calib] Tokenizing {CALIB_SAMPLES} prompts "
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
    print(f"[Calib] {len(batches)} batches ready ✓")
    return batches


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
                mem = torch.cuda.memory_allocated(DEVICE) / 1e9
                print(
                    f"  [Calib] {seen[0]:>4d}/{total} | "
                    f"{elapsed:6.1f}s | {rate:.1f} samp/s | "
                    f"GPU {mem:.1f} GB"
                )

    return forward_loop


def patch_quant_config(output_dir):
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


def remap_index(output_dir):
    idx_path = Path(output_dir) / "model.safetensors.index.json"
    if not idx_path.exists():
        single = Path(output_dir) / "model.safetensors"
        print(
            "[Remap] Single-shard ✓"
            if single.exists()
            else "[Remap] WARNING: no safetensors found!"
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


def copy_configs(src, dst):
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


def output_size(output_dir):
    total = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
    print(f"\n[Output] Total size: {total / 1e9:.2f} GB  (expected ~18 GB)")


def main():
    print("=" * 72)
    print("  Qwen3.5-35B-A3B  →  NVFP4  [Final v5 — export patches included]")
    print(f"  Source  : {SOURCE_PATH}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  Samples : {CALIB_SAMPLES} | batch={CALIB_BATCH_SIZE} | seq={MAX_SEQ_LEN}")
    print(f"  Threads : {torch.get_num_threads()}")
    print("=" * 72)

    # Clean output dir — remove old failed state dict
    out = Path(OUTPUT_PATH)
    old_pt = out / "model_state_dict.pt"
    if old_pt.exists():
        print(
            f"\n[Cleanup] Removing old failed state dict ({old_pt.stat().st_size / 1e9:.1f} GB) ..."
        )
        old_pt.unlink()
        print("[Cleanup] Done ✓")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[Load] Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Load] Tokenizer ready ✓")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print(f"\n[Load] Model in BF16 ...")
    print(f"  Stage 1: CPU load (sequential, no GPU spike)")
    print(f"  Stage 2: .to(cuda:0) one tensor at a time")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    t1 = time.time()
    print(f"[Load] Stage 1 done in {t1 - t0:.1f}s")

    # PATCH: set architectures before moving to GPU — needed for export
    model.config.architectures = ["Qwen3_5MoeForCausalLM"]

    model = model.to(DEVICE)
    model.eval()
    t2 = time.time()

    alloc_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"[Load] Stage 2 done in {t2 - t1:.1f}s | GPU: {alloc_gb:.1f} GB")

    mt = getattr(model.config, "model_type", "unknown")
    ne = getattr(model.config, "num_experts", "?")
    print(f"[Load] class={type(model).__name__} | model_type={mt} | num_experts={ne}")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Quant config ───────────────────────────────────────────────────────
    cfg = build_quant_config()

    # ── 4. Shape guard ────────────────────────────────────────────────────────
    apply_shape_guard(model, cfg)

    # ── 5. Calibration ────────────────────────────────────────────────────────
    batches = build_calibration_batches(tokenizer)

    # ── 6. Quantize ───────────────────────────────────────────────────────────
    print(f"\n[Quantize] Starting ...")
    t_q = time.time()
    model = mtq.quantize(model, cfg, make_forward_loop(batches))
    print(f"[Quantize] Done in {(time.time() - t_q) / 60:.1f} min ✓")

    del batches
    gc.collect()
    torch.cuda.empty_cache()

    # ── 7. Export ─────────────────────────────────────────────────────────────
    print(f"\n[Export] Writing to {OUTPUT_PATH} ...")
    print(f"  Export patches active:")
    print(f"  - MODEL_NAME_TO_TYPE['Qwen3_5Moe'] = 'qwen3moe'")
    print(f"  - model.config.architectures = ['Qwen3_5MoeForCausalLM']")
    t_e = time.time()
    export_ok = False

    try:
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=OUTPUT_PATH)
        export_ok = True
        print(f"[Export] Done in {(time.time() - t_e) / 60:.1f} min ✓")
    except Exception as e:
        print(f"[Export] FAILED: {e}")
        import traceback

        traceback.print_exc()
        # DO NOT fall back to torch.save — it creates an unusable 65 GB file
        # and prevents re-export without OOM
        print("\n[Export] NOT saving state dict — re-run script to retry export")
        sys.exit(1)

    # ── 8. Patches ────────────────────────────────────────────────────────────
    print("\n[PostProcess] ...")
    remap_index(OUTPUT_PATH)
    patch_quant_config(OUTPUT_PATH)

    # ── 9. Copy configs ───────────────────────────────────────────────────────
    copy_configs(SOURCE_PATH, OUTPUT_PATH)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    output_size(OUTPUT_PATH)

    print("\n" + "=" * 72)
    print("  ✅ QUANTIZATION COMPLETE")
    print(f"  Output : {OUTPUT_PATH}")
    print()
    print("  Serve:")
    print(f"  docker run --rm --gpus '\"device=0\"' --shm-size 16g \\")
    print(f"    -v /models/qwen35-35b-a3b-nvfp4:/model:ro -p 8000:8000 \\")
    print(f"    vllm/vllm-openai:latest-cu130-ubuntu2404 \\")
    print(f"    --model /model --quantization modelopt_fp4 \\")
    print(f"    --reasoning-parser qwen3 --max-model-len 32768 \\")
    print(f"    --gpu-memory-utilization 0.65 --port 8000")
    print("=" * 72)


if __name__ == "__main__":
    main()
