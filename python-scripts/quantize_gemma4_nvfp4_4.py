
"""
Gemma 4 31B NVFP4+FP8 Quantization for DGX Spark GB10 (Blackwell)
- Text layers (60 layers, ~57GB): NVFP4 weights + FP8 activations
- Vision tower (~2GB):            BF16 — requires image data for calibration
- Norms/embeddings:               BF16
Memory result: ~18GB vs 59GB BF16
"""
import torch
import copy
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from pathlib import Path
import shutil

# Pin all CPU threads upfront
torch.set_num_threads(20)
torch.set_num_interop_threads(20)
os.environ["OMP_NUM_THREADS"] = "20"

MODEL_PATH  = "/models/gemma4-31b-bf16"
OUTPUT_PATH = "/models/gemma4-31b-nvfp4"
CALIB_SAMPLES = 512
BATCH_SIZE    = 2
MAX_SEQ_LEN   = 1024

print("=" * 60)
print("Step 1/4: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Step 2/4: Loading model in BF16 (force cuda:0, no CPU offload)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",       # force all layers on GPU, no CPU offload
    low_cpu_mem_usage=True,
)
model.eval()
print(f"  Model loaded on: {next(model.parameters()).device}")

# Verify nothing on CPU
cpu_params = [(n, p.device) for n, p in model.named_parameters() if p.device.type == "cpu"]
if cpu_params:
    print(f"  ⚠️  WARNING: {len(cpu_params)} params still on CPU!")
else:
    print("  ✅ All parameters on GPU.")

CALIB_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "What are the causes of the French Revolution?",
    "Summarize the key differences between TCP and UDP.",
    "How does backpropagation work in neural networks?",
    "Describe the process of photosynthesis.",
    "What is the role of the mitochondria in a cell?",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a SQL query to find duplicate rows in a table.",
    "What are the main principles of object-oriented programming?",
] * (CALIB_SAMPLES // 10)

def calib_loop():
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

quant_cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
quant_cfg["quant_cfg"].update({
    # Vision tower → BF16 (needs image tokens for calibration, text-only here)
    "*vision_tower*":              {"enable": False},
    "*embed_vision*":              {"enable": False},
    # Non-quantizable by nature
    "*position_embedding_table*":  {"enable": False},
    "*q_norm*":                    {"enable": False},
    "*k_norm*":                    {"enable": False},
    "*embed_tokens*":              {"enable": False},
    "*layernorm*":                 {"enable": False},
    "*layer_norm*":                {"enable": False},
})

print("\nStep 3/4: Quantizing...")
print("  Text layers  → NVFP4 + FP8  (~57GB → ~15GB)")
print("  Vision tower → BF16 disabled (~2GB stays ~2GB)")
print("  Norms/embeds → BF16 disabled")

mtq.quantize(model, quant_cfg, forward_loop=calib_loop)
print("  ✅ Quantization complete.")

print("\nStep 4/4: Exporting HF checkpoint...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

try:
    export_hf_checkpoint(model, export_dir=OUTPUT_PATH)
    print("  ✅ HF checkpoint exported successfully.")
except Exception as e:
    print(f"  ❌ HF export failed: {e}")
    print("  Saving modelopt state dict as fallback...")
    torch.save(model.state_dict(), f"{OUTPUT_PATH}/model_nvfp4_state_dict.pt")

for f in [
    "tokenizer.json", "tokenizer_config.json",
    "config.json", "generation_config.json",
    "chat_template.jinja", "processor_config.json",
    "model.safetensors.index.json"
]:
    src = Path(MODEL_PATH) / f
    if src.exists():
        shutil.copy(src, Path(OUTPUT_PATH) / f)
        print(f"  Copied: {f}")

print("=" * 60)
total = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob("*") if f.is_file())
print(f"✅ Done. Size: {total/1e9:.1f} GB  (expected ~18GB)")

