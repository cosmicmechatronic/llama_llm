#!/usr/bin/env python3
"""
Llama 3.2 Fine-tuning Script
Fine-tunes Llama 3.2 model using QLoRA (Quantized Low-Rank Adaptation)
for memory-efficient training on consumer hardware.
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from typing import List, Dict

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  
OUTPUT_DIR = "./llama-3.2-finetuned"
MAX_LENGTH = 512



def format_prompt(instruction: str, input_text: str, output: str = None) -> str:
    """
    Formats the training data into a conversational prompt format suitable for Llama 3.2.
    """
    if output:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>"""
    else:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def prepare_dataset(data: List[Dict[str, str]], tokenizer) -> Dataset:
    """
    Prepares the dataset for training by tokenizing and formatting.
    """
    formatted_data = []
    
    for item in data:
        formatted_prompt = format_prompt(
            item["instruction"], 
            item["input"], 
            item["output"]
        )
        formatted_data.append({"text": formatted_prompt})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def setup_model_and_tokenizer():
    """
    Sets up the model and tokenizer with quantization for memory efficiency.
    """
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    return model, tokenizer

def setup_lora_config():
    """
    Configures LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning.
    """
    lora_config = LoraConfig(
        r=16,  # Rank of adaptation
        lora_alpha=32,  # LoRA scaling parameter
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def main():
    """
    Main fine-tuning pipeline.
    """
    print(" Starting Llama 3.2 Fine-tuning Process")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    # Create example dataset
    print("Creating example training dataset...")
    training_data = create_example_dataset()
    print(f"Dataset size: {len(training_data)} examples")
    
    # Setup model and tokenizer
    print("🔧 Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    # Prepare dataset
    print(" Preparing dataset...")
    train_dataset = prepare_dataset(training_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduce if OOM
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=500,  # Limit for demo purposes
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=25,
        optim="paged_adamw_32bit",
        evaluation_strategy="no",
        save_strategy="epoch",
        group_by_length=True,
        report_to=None,  # Disable wandb logging
        run_name="llama-3.2-finetune",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Start training
    print(" Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")
    
    # Example inference
    print("\n Testing the fine-tuned model...")
    test_inference(model, tokenizer)

def test_inference(model, tokenizer):
    """
    Tests the fine-tuned model with a sample input.
    """
    model.eval()
    
    test_prompt = format_prompt(
        "You are a helpful customer service assistant. Respond professionally and helpfully.",
        "I want to return an item I bought last week. It doesn't fit properly."
    )
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(" Sample Response:")
    print(response.split("assistant<|end_header_id|>")[-1].strip())

if __name__ == "__main__":

    main()