# Install docker container for Tensort_llm 
#

docker pull nvcr.io/nvidia/tensorrt_llm:0.17.0-release

docker pull ghcr.io/nvidia/tensorrt-llm/tensorrt-llm:0.17.0




# Install without container tensor rt  llm 

sudo apt-get update && sudo apt-get install -y \
  git \
  git-lfs \
  python3-pip \
  python3-dev \
  build-essential \
  cmake \
  libopenmpi-dev \
  openmpi-bin \
  wget \
  curl

# Using vertual environment
uv venv trtllm-env --python 3.11
echo "trtllm-env/" >> .gitignore

# install setup tools in virtual environment
uv pip install --upgrade pip setuptools wheel

# check which available tensor rt LLM
uv pip install tensorrt_llm== \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match 2>&1 | head -30

# check
 curl -s https://pypi.nvidia.com/tensorrt-llm/ | grep -o 'tensorrt_llm-[^"]*cp311[^"]*\.whl' | sort

# 3.11->3.12
deactivate
rm -rf trtllm-env
uv venv trtllm-env --python 3.12
source trtllm-env/bin/activate

# install  tensor rt llm 
uv pip install tensorrt_llm==0.21.0 \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match

# check 
python3 -c "import tensorrt_llm; print('TRT-LLM version:', tensorrt_llm.__version__)"

# check parallel compute if not install
sudo apt-get install -y libopenmpi-dev openmpi-bin

# down grade

uv pip install onnx==1.16.2 --index-strategy unsafe-best-match

uv pip install cuda-python==12.6.2 --index-strategy unsafe-best-match

# verify cuda import
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# model opt
python3 -c "import modelopt; print('ModelOpt version:', modelopt.__version__)"
