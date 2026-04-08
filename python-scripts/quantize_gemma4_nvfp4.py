
"""
NVFP4 PTQ for Gemma 4 31B on DGX Spark GB10 (aarch64 + Blackwell)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from pathlib import Path

MODEL_PATH  = "/models/gemma4-31b-bf16"
OUTPUT_PATH = "/models/gemma4-31b-nvfp4"
CALIB_SAMPLES = 512
BATCH_SIZE    = 2
MAX_SEQ_LEN   = 1024

print("=" * 60)
print("Step 1/4: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Step 2/4: Loading model in BF16 (~60GB, takes 3-5 mins)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"  Model loaded on: {next(model.parameters()).device}")

# --- Calibration data (no internet needed) ---
CALIB_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "What are the causes of the French Revolution?",
    "Summarize the key differences between TCP and UDP.",
    "How does backpropagation work in neural networks?",
] * (CALIB_SAMPLES // 5)

def calib_loop():
    print("Step 3/4: Running calibration...")
    for i in range(0, len(CALIB_PROMPTS), BATCH_SIZE):
        batch = CALIB_PROMPTS[i:i+BATCH_SIZE]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            model(**tokens)
        if i % 50 == 0:
            print(f"  Calibrated {i}/{len(CALIB_PROMPTS)} samples...")

# NVFP4 weights + FP8 activations — optimal for Blackwell GB10
print("Step 3/4: Quantizing to NVFP4...")
mtq.quantize(model, mtq.W4A8_NVFP4_FP8_CFG, forward_loop=calib_loop)
print("  Quantization complete.")

print("Step 4/4: Exporting HF-compatible checkpoint...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
export_hf_checkpoint(model, export_dir=OUTPUT_PATH)

# Copy tokenizer files
import shutil
for f in ["tokenizer.json", "tokenizer_config.json",
          "config.json", "generation_config.json",
          "chat_template.jinja", "processor_config.json"]:
    src = Path(MODEL_PATH) / f
    if src.exists():
        shutil.copy(src, Path(OUTPUT_PATH) / f)
        print(f"  Copied: {f}")

print("=" * 60)
print(f"✅ Done. Quantized model at: {OUTPUT_PATH}")
import os
total = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob("*") if f.is_file())
print(f"   Size: {total/1e9:.1f} GB  (expected ~16-20GB vs 59GB BF16)")

