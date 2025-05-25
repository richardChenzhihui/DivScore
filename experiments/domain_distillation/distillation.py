#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA Fine-tuning Script for Mistral-7B

This script performs LoRA fine-tuning on Mistral-7B model for text generation tasks.
Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters.

License: MIT License
Copyright (c) 2023 Zhihui Chen
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig

# Configure CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ModelCheckpointCallback(TrainerCallback):
    """Custom callback to save model checkpoints after each epoch."""

    def __init__(self, trainer, output_dir: str, tokenizer):
        self.trainer = trainer
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save model checkpoint at the end of each epoch."""
        epoch_dir = os.path.join(
            self.output_dir,
            f"mistral-7B_epoch{state.epoch}"
        )
        os.makedirs(epoch_dir, exist_ok=True)

        self.trainer.model.save_pretrained(epoch_dir)
        self.tokenizer.save_pretrained(epoch_dir)
        print(f"Model checkpoint saved to: {epoch_dir}")


def format_training_sample(sample: dict) -> str:
    """Format training sample into instruction prompt."""
    return f"{sample['response (content)']}"


def load_and_prepare_data(config: dict):
    """Load and preprocess training data."""
    dataset = load_dataset("json", data_files=config["data_path"], split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    processed_data = dataset.map(
        lambda x: {"text": format_training_sample(x)},
        remove_columns=["reasoning (reasoning_content)"]
    )
    return processed_data, tokenizer


def initialize_model(config: dict):
    """Initialize the base model with appropriate settings."""
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = model.to(torch.bfloat16)
    model.lm_head = model.lm_head.to(torch.bfloat16)
    return model


def setup_lora_config(config: dict) -> LoraConfig:
    """Configure LoRA parameters."""
    return LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["norm"]
    )


def configure_training_args(config: dict) -> SFTConfig:
    """Set up training configuration."""
    return SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config['epoch'],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="adamw_torch",
        logging_dir=f"./logs-epoch{config['epoch']}",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        max_seq_length=2048,
        dataset_text_field="text",
        packing=True
    )


def generate_test_outputs(model, tokenizer, dataset, num_samples: int = 3):
    """Generate test outputs from the trained model."""

    def generate_response(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs = inputs.to(torch.bfloat16)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Reduced from original 2048*4 for safety
            temperature=0.7,
            do_sample=True
        )
        return tokenizer.decode(outputs[0])

    for sample in dataset.shuffle().select(range(num_samples)):
        print(f"\n[GROUND TRUTH]\n{sample['response (content)']}")
        print(f"\n[MODEL OUTPUT]\n{generate_response(sample['question'])}")


def main(config: dict):
    """Main training pipeline."""
    set_seed(config["seed"])

    # Data preparation
    processed_data, tokenizer = load_and_prepare_data(config)

    # Model initialization
    model = initialize_model(config)

    # Training setup
    peft_config = setup_lora_config(config)
    training_args = configure_training_args(config)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_data,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    # save model during training
    # trainer.add_callback(ModelCheckpointCallback(trainer, config["output_dir"], tokenizer))

    # Execute training
    print("==== Starting Training ====")
    trainer.train()

    # Final model save
    final_dir = os.path.join(config["output_dir"], f"mistral-7B-epoch{config['epoch']}")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to: {final_dir}")

    # Test generation
    print("\n==== Running Test Generation ====")
    generate_test_outputs(model, tokenizer, processed_data, config["test_samples"])


if __name__ == "__main__":
    # Experiment configuration
    for epoch in [8]:
        print(f"\n{'=' * 30} Epoch {epoch} {'=' * 30}")

        your_path = "..."
        # combined distillation (medical + legal domain knowledge base)
        EXPERIMENT_CONFIG = {
            "data_path": f"{your_path}/datasets/domain_distillation/combined/combined_r1_distill_sft.json",
            "base_model": f"{your_path}/models/Mistral-7B-v0.2",
            "output_dir": f"{your_path}/experiments/domain_distillation/combined_distill/mistral-7B_epoch8",
            "seed": 42,
            "max_seq_len": 2048 * 3,
            "batch_size": 24,
            "lora_rank": 64,
            "lora_alpha": 128,
            "test_samples": 3,
            "epoch": epoch
        }

        main(EXPERIMENT_CONFIG)
