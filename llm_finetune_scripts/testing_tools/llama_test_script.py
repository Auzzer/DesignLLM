import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./checkpoints/Llama3.1-8B-Instruct-huggingface"
model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device_map="auto",
)

sequences = pipeline(
    "Help me to generate a design of the bedroom \n",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
)
for seq in sequences:
    print(f"{seq['generated_text']}")