import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Paths
MODEL_PATH = "checkpoints/Llama3.1-8B-Instruct-huggingface"
DATASET_PATH = "instr_alpaca_v2.json"
OUTPUT_DIR = "results"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",         # Use the optimal precision
    device_map="auto",          # Automatically distribute model across GPUs
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset("json", data_files=DATASET_PATH)

# Split into train and test datasets
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Tokenize the dataset
def preprocess_function(examples):
    inputs = examples["instruction"]
    outputs = examples["output"] ##+ " <|eos|>"  # Append stop token to output
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        outputs, max_length=512, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,       # Adjust to prevent memory overflow
    gradient_accumulation_steps=16,     # Simulate a larger batch size
    learning_rate=5e-5,
    fp16=True,                          # Use mixed precision for efficiency
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    report_to="none",                   # Disable external logging
    ddp_find_unused_parameters=False,   # Required for multi-GPU setups
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
