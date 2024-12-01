# Step 1: Download the model weights 
# python -m venv .LLM601
source LLM601/bin/activate
pip install --upgrade huggingface_hub
pip install llama-stack # Install the Llama CLI
export LLAMA_STACK_CONFIG_DIR="/projectnb/ece601/24FallA2Group16" # This avoid dowloading the weights to the home dir
# Apply for you personal custom URL to download the model: https://www.llama.com/llama-downloads/
# huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --include "original/*" --local-dir Llama-3.2-11B-Vision-Instruct
# llama model download --source meta --model-id Llama3.2-11B-Vision-Instruct
llama model download --source meta --model-id Llama3.2-3B
llama model download --source meta --model-id Llama3.1-8B-Instruct
llama model verify-download --model-id Llama3.1-8B-Instruct. # verify the model

# GPU requirements: https://llamaimodel.com/requirements-3-2/

# Step 2 Convert the source to the huggingface
# python3 -m venv hf-convertor
source hf-convertor/bin/activate
# git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
pip install torch tiktoken blobfile accelerate
python3 src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir "/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.2-3B" --output_dir "/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.2-3B-huggingface" --model_size 3B --llama_version 3.2
python3 src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir "/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.1-8B-Instruct" --output_dir "/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.1-8B-Instruct-huggingface" --model_size 8B --llama_version 3.1


python ./llama_test_script.py

# Step 3: Finetune the model
## Install stable PyTorch, torchvision, torchao stable releases
pip install torch torchvision torchao
pip install torchtune
## Config the training scrip, see the file "my_custom_config.yaml". Then run:
tune run --nproc_per_node 2 full_finetune_distributed --config ./my_custom_config.yaml
# tune cp llama3_1/8B_lora_single_device ./lora_8b.yaml
tune run lora_finetune_single_device --config ./lora_8b.yaml


tune run --nproc_per_node 2 full_finetune_distributed --config ./fig_gen_full.yaml

## Step 4
sd token from huggingface: 

