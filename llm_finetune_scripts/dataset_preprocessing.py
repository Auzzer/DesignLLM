import pandas as pd
import json
import re
from transformers import AutoTokenizer
import torchtune.models.llama3._tokenizer # load for llama3 tokenizer
# Define a function to clean keys by removing forbidden characters
def clean_key(key):
    # Remove forbidden characters: '-', '_', '.', '--', '..'
    key = re.sub(r'[-_.]', '', key)
    # Replace multiple spaces with a single space
    key = re.sub(r'\s+', ' ', key)
    # Strip leading and trailing spaces
    return key.strip()

# Load the tokenizer from the local path
"""
tokenizer = AutoTokenizer.from_pretrained("/projectnb/ece601/24FallA2Group16/checkpoints/Llama3.1-8B-Instruct-huggingface/")

# Add special tokens to the tokenizer (only adding <|eos|>)
special_tokens = {
    'additional_special_tokens': ['<|eos|>']
}
tokenizer.add_special_tokens(special_tokens)

# Define a function to calculate the token length
def calculate_token_length(text):
    return len(tokenizer.encode(text, add_special_tokens=True))
"""

tokenizer_path = "checkpoints/Llama3.1-8B-Instruct/tokenizer.model"
tokenizer = torchtune.models.llama3._tokenizer.Llama3Tokenizer(tokenizer_path)


def calculate_token_length(text):
    return len(tokenizer.encode(text))
# Load the dataset
df = pd.read_csv('InstructionsGenerated/llama_training_dataset_with_templates.csv')
alpaca_data = []

# Iterate through each row in the DataFrame
for _, row in df.iterrows():
    # Clean the keys
    cleaned_instruction = clean_key(row['input'])
    cleaned_output = clean_key(row['output'])
    # Calculate token lengths for instruction and output
    instruction_length = calculate_token_length(cleaned_instruction)
    output_length = calculate_token_length(cleaned_output)

    # Filter out rows where the length exceeds 1024
    if instruction_length >1024 or output_length > 1024:
        continue

    # Add start and end special tokens to instruction and output
    cleaned_instruction = cleaned_instruction  # No modification to instruction
    # Add the stopping sign (<|eos|>) at the end of the output
    cleaned_output = cleaned_output #+ '<|eos|>'
    
    

    # Create a dictionary for each entry
    entry = {
        "instruction": cleaned_instruction,
        "input": " ",  
        "output": cleaned_output
    }
    # Append the entry to the list
    alpaca_data.append(entry)

# Write the cleaned data to a JSON file
with open('./instr_alpaca_v2.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=4)

print("Data has been successfully cleaned and saved to 'instr_alpaca_v2.json'.")
