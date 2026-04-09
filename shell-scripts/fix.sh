cat > /models/gemma4-31b-nvfp4/hf_quant_config.json << 'EOF'
{
    "producer": {
        "name": "modelopt",
        "version": "0.42.0"
    },
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": null,
        "group_size": 32,
        "exclude_modules": [
            "lm_head",
            "model.embed_vision*",
            "model.vision_tower*"
        ]
    }
}
EOF

cat /models/gemma4-31b-nvfp4/hf_quant_config.json

python3 << 'EOF'
import json

with open('/models/gemma4-31b-nvfp4/model.safetensors.index.json') as f:
    idx = json.load(f)

# Remap all shard references to the single output file
for key in idx['weight_map']:
    idx['weight_map'][key] = 'model.safetensors'

with open('/models/gemma4-31b-nvfp4/model.safetensors.index.json', 'w') as f:
    json.dump(idx, f, indent=2)

# Verify
files = set(idx['weight_map'].values())
print(f'Index now references: {files}')
print(f'Total weights mapped: {len(idx["weight_map"])}')
EOF

