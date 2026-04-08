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

