#!/bin/bash

#make directory and download
sudo mkdir -p /models/gemma4-31b-bf16
sudo chown $USER:$USER /models/gemma4-31b-bf16


# login to hugging face
hf auth login
# check for model access
hf models list --search "google/gemma-4-31b-it" | head -5

# download the models
hf download google/gemma-4-31B-it \
  --local-dir /models/gemma4-31b-bf16 \
  --max-workers 8

nohup hf download google/gemma-4-31B-it \
  --local-dir /models/gemma4-31b-bf16 \
  --max-workers 8 \
  > /tmp/gemma4-download.log 2>&1 &

# check
cat /models/gemma4-31b-bf16/config.json | python3 -m json.tool | grep -E "model_type|num_hidden|num_attention|hidden_size|vocab"
ls -lh /models/gemma4-31b-bf16/
# check what is available for quantization
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('Memory:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"
python3 -c "import modelopt; print('modelopt:', modelopt.__version__)"

# setup with UV
uv venv /models/quant-env --python 3.11
source /models/quant-env/bin/activate

# install torch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# install torch form pypi
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple

# install modelopt
uv pip install "nvidia-modelopt[torch]" \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple

# dependency
uv pip install transformers accelerate datasets sentencepiece protobuf
#check
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/models/gemma4-31b-bf16')
print('Tokenizer OK:', tok.__class__.__name__)
print('Vocab size:', tok.vocab_size)
"

/models/quant-env/bin/python3 -c "
import modelopt.torch.quantization as mtq
configs = [x for x in dir(mtq) if 'NVFP4' in x or 'FP4' in x]
print('Available FP4 configs:')
for c in configs:
    print(' ', c)
"


# run
nohup /models/quant-env/bin/python3 /models/quantize_gemma4_nvfp4.py \
  > /tmp/quant_gemma4.log 2>&1 &

echo "Quantization PID: $!"
tail -f /tmp/quant_gemma4.log

