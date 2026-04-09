
"""
Gemma 4 31B NVFP4+FP8 Quantization for DGX Spark GB10 (Blackwell)
- All linear layers (text + vision): NVFP4 weights + FP8 activations
- Non-linear (position_embedding_table, norms, lm_head): BF16 (disabled)
Block size = 32, all vision shapes divisible: 1152%32=0, 4304%32=0
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from pathlib import Path
import shutil
import copy

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

# --- Calibration prompts ---
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

# --- Build config from base, add targeted exclusions ---
quant_cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)

# Only exclude truly non-quantizable layers:
# - position_embedding_table: 3D tensor [2, 10240, 1152]
# - q_norm/k_norm: 1D vectors [72]
# - embed_tokens: vocabulary embedding
# Everything else (text + vision linear) stays NVFP4+FP8
quant_cfg["quant_cfg"].update({
    "*position_embedding_table*": {"enable": False},
    "*q_norm*":                   {"enable": False},
    "*k_norm*":                   {"enable": False},
    "*embed_tokens*":             {"enable": False},
    "*layernorm*":                {"enable": False},
    "*layer_norm*":               {"enable": False},
})

print("Step 3/4: Quantizing (all linear layers NVFP4+FP8, norms/embeds BF16)...")
mtq.quantize(model, quant_cfg, forward_loop=calib_loop)
print("  Quantization complete.")

print("Step 4/4: Exporting checkpoint...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

try:
    export_hf_checkpoint(model, export_dir=OUTPUT_PATH)
    print("  HF checkpoint exported successfully.")
except Exception as e:
    print(f"  HF export failed: {e}")
    print("  Falling back to torch.save (modelopt format)...")
    save_path = f"{OUTPUT_PATH}/model_nvfp4_state_dict.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")

# Copy tokenizer + config files
print("  Copying tokenizer and config files...")
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
print(f"✅ Done. Output: {OUTPUT_PATH}")
total = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob("*") if f.is_file())
print(f"   Size: {total/1e9:.1f} GB  (expected ~18-22GB vs 59GB BF16)")

