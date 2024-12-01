import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model directory
model_dir = "/projectnb/ece601/24FallA2Group16/checkpoints//Llama3.1-8B-Instruct-huggingface"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize pipeline with text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Continuous interaction loop
while True:
    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Ending conversation.")
        break

    # Generate a response from the model based on the current input
    sequences = pipeline(
        user_input,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_length=512,  # You can adjust this as needed
    )

    # Print the generated response
    for seq in sequences:
        print(f"Model: {seq['generated_text']}")
